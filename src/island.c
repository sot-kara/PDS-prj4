#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include <omp.h>
#include "ga_model.h"

#define NUM_ISLANDS 10
#define EPOCH_LENGTH 20

// Target function
double true_function(double u1, double u2) {
    return sin(u1 + u2) * sin(u2 * u2);
}

typedef struct {
    int index;
    double fitness;
} SortItem;

int compare_fitness(const void* a, const void* b) {
    double diff = ((SortItem*)a)->fitness - ((SortItem*)b)->fitness;
    return (diff > 0) - (diff < 0);
}

// Global synchronization variables for migration
pthread_mutex_t sync_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t sync_cond = PTHREAD_COND_INITIALIZER;
int arrived_islands = 0;

// Thread arguments and migration buffer
typedef struct {
    int island_id;
    int pop_size;
    int total_gens;
    int M;
    double crossover_rate;
    double mutation_rate;
    double mutation_strength;
    int elite_size;
    Model* population;
    Model incoming_migrant; // Buffer to hold the elite migrant from the previous island
} IslandArgs;

// Global array to access neighboring migration buffers
IslandArgs island_contexts[NUM_ISLANDS];

void* island_worker(void* arg) {
    IslandArgs* ctx = (IslandArgs*)arg;
    unsigned int local_seed = (unsigned int)time(NULL) ^ (ctx->island_id * 12345);
    
    Model* new_population = (Model*)malloc(ctx->pop_size * sizeof(Model));
    double* mse_pop = (double*)malloc(ctx->pop_size * sizeof(double));
    SortItem* sort_arr = (SortItem*)malloc(ctx->pop_size * sizeof(SortItem));
    
    int epochs = ctx->total_gens / EPOCH_LENGTH;
    int train_set_points = 1000;

    for (int e = 0; e < epochs; e++) {
        // --- EVOLUTION PHASE (No locks, fully independent) ---
        for (int gen = 0; gen < EPOCH_LENGTH; gen++) {
            #pragma omp parallel for
            for (int pop_idx = 0; pop_idx < ctx->pop_size; pop_idx++) {
                mse_pop[pop_idx] = fitness_function(&ctx->population[pop_idx], true_function, train_set_points, ctx->M, &local_seed);
                sort_arr[pop_idx].index = pop_idx;
                sort_arr[pop_idx].fitness = mse_pop[pop_idx];
            }

            qsort(sort_arr, ctx->pop_size, sizeof(SortItem), compare_fitness);

            // Elitism
            #pragma omp parallel for
            for (int i = 0; i < ctx->elite_size; i++) {
                new_population[i] = ctx->population[sort_arr[i].index];
            }

            // Breeding
            int offspring_count = ctx->elite_size;
            while (offspring_count < ctx->pop_size) {
                int p1_idx = roulette_wheel_selection(mse_pop, ctx->pop_size, &local_seed);
                int p2_idx = roulette_wheel_selection(mse_pop, ctx->pop_size, &local_seed);

                Model child1, child2;
                crossover(&ctx->population[p1_idx], &ctx->population[p2_idx], &child1, &child2, ctx->crossover_rate, ctx->M);
                
                mutation(&child1, ctx->mutation_rate, ctx->mutation_strength, ctx->M, &local_seed);
                mutation(&child2, ctx->mutation_rate, ctx->mutation_strength, ctx->M, &local_seed);

                if (offspring_count < ctx->pop_size) new_population[offspring_count++] = child1;
                if (offspring_count < ctx->pop_size) new_population[offspring_count++] = child2;
            }
            #pragma omp parallel for
            for (int i = 0; i < ctx->pop_size; i++) {
                ctx->population[i] = new_population[i];
            }
        }

        // --- MIGRATION PHASE (POSIX Synchronization) ---
        // Evaluate one last time to ensure we are exporting the true best of this epoch
        #pragma omp parallel for
        for (int pop_idx = 0; pop_idx < ctx->pop_size; pop_idx++) {
            mse_pop[pop_idx] = fitness_function(&ctx->population[pop_idx], true_function, train_set_points, ctx->M, &local_seed);
            sort_arr[pop_idx].index = pop_idx;
            sort_arr[pop_idx].fitness = mse_pop[pop_idx];
        }
        qsort(sort_arr, ctx->pop_size, sizeof(SortItem), compare_fitness);

        // Define circular ring topology: Island N sends to Island N+1
        int next_island = (ctx->island_id + 1) % NUM_ISLANDS;

        pthread_mutex_lock(&sync_mutex);
        
        // Export the best model to the neighbor's buffer
        island_contexts[next_island].incoming_migrant = ctx->population[sort_arr[0].index];
        arrived_islands++;
        
        // Wait for all islands to finish their epoch and export
        if (arrived_islands < NUM_ISLANDS) {
            pthread_cond_wait(&sync_cond, &sync_mutex);
        } else {
            // Last thread to arrive resets the barrier and wakes the others
            arrived_islands = 0;
            pthread_cond_broadcast(&sync_cond);
            printf("Epoch %d complete. Migration executed across %d islands.\n", (e + 1) * EPOCH_LENGTH, NUM_ISLANDS);
        }
        
        pthread_mutex_unlock(&sync_mutex);

        // Inject the incoming migrant, replacing the weakest individual
        ctx->population[sort_arr[ctx->pop_size - 1].index] = ctx->incoming_migrant;
    }

    free(new_population);
    free(mse_pop);
    free(sort_arr);
    return NULL;
}

int main(void) {
    int total_gens = 200;
    int M = 5; 
    int total_pop_size = 100;
    int island_pop_size = total_pop_size / NUM_ISLANDS;
    int elite_size = 2; // Scaled down since sub-populations are smaller

    pthread_t threads[NUM_ISLANDS];
    unsigned int main_seed = (unsigned int)time(NULL);

    // Initialize the island contexts
    for (int i = 0; i < NUM_ISLANDS; i++) {
        island_contexts[i].island_id = i;
        island_contexts[i].pop_size = island_pop_size;
        island_contexts[i].total_gens = total_gens;
        island_contexts[i].M = M;
        island_contexts[i].crossover_rate = 0.8;
        island_contexts[i].mutation_rate = 0.1;
        island_contexts[i].mutation_strength = 0.4;
        island_contexts[i].elite_size = elite_size;
        island_contexts[i].population = (Model*)malloc(island_pop_size * sizeof(Model));

        // Randomly populate the island
        for (int j = 0; j < island_pop_size; j++) {
            for (int k = 0; k < M; k++) {
                island_contexts[i].population[j].gaussians[k].w = 2.0 * rand_double(0.0, 1.0, &main_seed) - 1.0;
                island_contexts[i].population[j].gaussians[k].c[0] = rand_double(U1_MIN, U1_MAX, &main_seed);
                island_contexts[i].population[j].gaussians[k].c[1] = rand_double(U2_MIN, U2_MAX, &main_seed);
                island_contexts[i].population[j].gaussians[k].sigma[0] = 0.1 + 0.9 * rand_double(0.0, 1.0, &main_seed);
                island_contexts[i].population[j].gaussians[k].sigma[1] = 0.1 + 0.9 * rand_double(0.0, 1.0, &main_seed);
            }
        }
    }

    struct timespec start_time, end_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);

    // Spawn threads
    for (int i = 0; i < NUM_ISLANDS; i++) {
        pthread_create(&threads[i], NULL, island_worker, &island_contexts[i]);
    }

    // Wait for all islands to finish evolution
    for (int i = 0; i < NUM_ISLANDS; i++) {
        pthread_join(threads[i], NULL);
    }

    clock_gettime(CLOCK_MONOTONIC, &end_time);
    double elapsed_time = (end_time.tv_sec - start_time.tv_sec) + 
                          (end_time.tv_nsec - start_time.tv_nsec) / 1e9;

    // Find the global best model across all islands
    double global_best_mse = -1;
    Model global_best_model;

    for (int i = 0; i < NUM_ISLANDS; i++) {
        for (int j = 0; j < island_pop_size; j++) {
            double current_mse = fitness_function(&island_contexts[i].population[j], true_function, 1000, M, &main_seed);
            if (global_best_mse < 0 || current_mse < global_best_mse) {
                global_best_mse = current_mse;
                global_best_model = island_contexts[i].population[j];
            }
        }
        free(island_contexts[i].population);
    }

    printf("\nFinal Best MSE (Island Model): %.6f\n", global_best_mse);
    printf("Island Evolution Time: %.6f seconds\n", elapsed_time);

    FILE *f = fopen("model_output.csv", "w");
    if (f != NULL) {
        fprintf(f, "gaussian_id,weight,c1,c2,sigma1,sigma2\n");
        for (int i = 0; i < M; i++) {
            fprintf(f, "%d,%f,%f,%f,%f,%f\n", 
                i, 
                global_best_model.gaussians[i].w, 
                global_best_model.gaussians[i].c[0], 
                global_best_model.gaussians[i].c[1], 
                global_best_model.gaussians[i].sigma[0], 
                global_best_model.gaussians[i].sigma[1]);
        }
        fclose(f);
        printf("Successfully wrote model parameters to model_output.csv\n");
    }

    return 0;
}