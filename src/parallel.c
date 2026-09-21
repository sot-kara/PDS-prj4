#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include "ga_model.h"

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

int main(void) {
    srand((unsigned int)time(NULL));

    // GA Parameters
    double crossover_rate = 0.8;
    double mutation_rate = 0.1;
    double mutation_strength = 0.4;
    int gens = 200;
    int M = 5; 
    int pop_size = 100;
    int elite_size = 5;
    int train_set_points = 1000;

    Model* population = (Model*)malloc(pop_size * sizeof(Model));
    Model* new_population = (Model*)malloc(pop_size * sizeof(Model));
    double* mse_pop = (double*)malloc(pop_size * sizeof(double));
    SortItem* sort_arr = (SortItem*)malloc(pop_size * sizeof(SortItem));

    // Initialize Population
    for (int j = 0; j < pop_size; j++) {
        for (int i = 0; i < M; i++) {
            population[j].gaussians[i].w = 2.0 * rand_double(0.0, 1.0) - 1.0;
            population[j].gaussians[i].c[0] = rand_double(U1_MIN, U1_MAX);
            population[j].gaussians[i].c[1] = rand_double(U2_MIN, U2_MAX);
            population[j].gaussians[i].sigma[0] = 0.1 + 0.9 * rand_double(0.0, 1.0);
            population[j].gaussians[i].sigma[1] = 0.1 + 0.9 * rand_double(0.0, 1.0);
        }
    }

    // Start OpenMP wall-clock timer
    double start_time = omp_get_wtime();

    // Evolution Loop
    for (int gen = 1; gen <= gens; gen++) {
        
        #pragma omp parallel for
        for (int pop_idx = 0; pop_idx < pop_size; pop_idx++) {
            mse_pop[pop_idx] = fitness_function(&population[pop_idx], true_function, train_set_points, M);
            sort_arr[pop_idx].index = pop_idx;
            sort_arr[pop_idx].fitness = mse_pop[pop_idx];
        }

        qsort(sort_arr, pop_size, sizeof(SortItem), compare_fitness);

        if (gen % 50 == 0 || gen == 1) {
            printf("Generation %d: Best MSE = %.6f\n", gen, sort_arr[0].fitness);
        }

        // Elitism
        for (int i = 0; i < elite_size; i++) {
            new_population[i] = population[sort_arr[i].index];
        }

        // Crossover and Mutation
        int offspring_count = elite_size;
        while (offspring_count < pop_size) {
            int p1_idx = roulette_wheel_selection(mse_pop, pop_size);
            int p2_idx = roulette_wheel_selection(mse_pop, pop_size);

            Model child1, child2;
            crossover(&population[p1_idx], &population[p2_idx], &child1, &child2, crossover_rate, M);
            
            mutation(&child1, mutation_rate, mutation_strength, M);
            mutation(&child2, mutation_rate, mutation_strength, M);

            if (offspring_count < pop_size) new_population[offspring_count++] = child1;
            if (offspring_count < pop_size) new_population[offspring_count++] = child2;
        }

        for (int i = 0; i < pop_size; i++) {
            population[i] = new_population[i];
        }
    }

    #pragma omp parallel for
    for (int pop_idx = 0; pop_idx < pop_size; pop_idx++) {
        mse_pop[pop_idx] = fitness_function(&population[pop_idx], true_function, 1000, M);
    }
    
    int best_idx = 0;
    double best_mse = mse_pop[0];
    for (int i = 1; i < pop_size; i++) {
        if (mse_pop[i] < best_mse) {
            best_mse = mse_pop[i];
            best_idx = i;
        }
    }
    
    Model best_model = population[best_idx];

    // Stop OpenMP timer
    double end_time = omp_get_wtime();
    double elapsed_time = end_time - start_time;

    printf("\nFinal Best MSE: %.6f\n", best_mse);
    printf("Parallel Evolution Time: %.6f seconds\n", elapsed_time);

    // Export to CSV
    FILE *f = fopen("model_output.csv", "w");
    if (f != NULL) {
        fprintf(f, "gaussian_id,weight,c1,c2,sigma1,sigma2\n");
        for (int i = 0; i < M; i++) {
            fprintf(f, "%d,%f,%f,%f,%f,%f\n", 
                i, 
                best_model.gaussians[i].w, 
                best_model.gaussians[i].c[0], 
                best_model.gaussians[i].c[1], 
                best_model.gaussians[i].sigma[0], 
                best_model.gaussians[i].sigma[1]);
        }
        fclose(f);
        printf("Successfully wrote model parameters to model_output.csv\n");
    } else {
        printf("Error: Could not open model_output.csv for writing.\n");
    }

    free(population);
    free(new_population);
    free(mse_pop);
    free(sort_arr);

    return 0;
}