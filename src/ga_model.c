#include "ga_model.h"
#include <stdlib.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Use rand_r for thread-safe, lock-free random generation
double rand_double(double min, double max, unsigned int* seed) {
    return min + (max - min) * ((double)rand_r(seed) / RAND_MAX);
}

double rand_normal(unsigned int* seed) {
    double u1 = ((double)rand_r(seed) / RAND_MAX);
    double u2 = ((double)rand_r(seed) / RAND_MAX);
    if (u1 <= 1e-7) u1 = 1e-7;
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

double gaussian_eval(double u1, double u2, const double c[2], const double sigma[2]) {
    double exp_val = - (pow(u1 - c[0], 2) / (2 * pow(sigma[0], 2)) + 
                        pow(u2 - c[1], 2) / (2 * pow(sigma[1], 2)));
    return exp(exp_val);
}

double fitness_function(const Model* model, TargetFunc func, int N, int M, unsigned int* seed) {
    double mse = 0.0;
    for (int i = 0; i < N; i++) {
        double u1 = rand_double(U1_MIN, U1_MAX, seed);
        double u2 = rand_double(U2_MIN, U2_MAX, seed);
        
        double y_true = func(u1, u2);
        double y_pred = 0.0;
        
        for (int j = 0; j < M; j++) {
            y_pred += model->gaussians[j].w * 
                      gaussian_eval(u1, u2, model->gaussians[j].c, model->gaussians[j].sigma);
        }
        mse += pow(y_true - y_pred, 2);
    }
    return mse / N;
}

void crossover(const Model* parent1, const Model* parent2, Model* child1, Model* child2, double crossover_rate, int M) {
    *child1 = *parent1;
    *child2 = *parent2;
    if (((double)rand() / RAND_MAX) < crossover_rate) {
        int crossover_point = 1 + rand() % (M - 1); 
        for (int i = crossover_point; i < M; i++) {
            child1->gaussians[i] = parent2->gaussians[i];
            child2->gaussians[i] = parent1->gaussians[i];
        }
    }
}

void mutation(Model* individual, double mutation_rate, double mutation_strength, int M, unsigned int* seed) {
    for (int i = 0; i < M; i++) {
        if (rand_double(0.0, 1.0, seed) < mutation_rate) {
            individual->gaussians[i].w += mutation_strength * rand_normal(seed);
        }
        if (rand_double(0.0, 1.0, seed) < mutation_rate) {
            individual->gaussians[i].c[0] += mutation_strength * rand_normal(seed);
            if (individual->gaussians[i].c[0] < U1_MIN) individual->gaussians[i].c[0] = U1_MIN;
            if (individual->gaussians[i].c[0] > U1_MAX) individual->gaussians[i].c[0] = U1_MAX;
        }
        if (rand_double(0.0, 1.0, seed) < mutation_rate) {
            individual->gaussians[i].c[1] += mutation_strength * rand_normal(seed);
            if (individual->gaussians[i].c[1] < U2_MIN) individual->gaussians[i].c[1] = U2_MIN;
            if (individual->gaussians[i].c[1] > U2_MAX) individual->gaussians[i].c[1] = U2_MAX;
        }
        if (rand_double(0.0, 1.0, seed) < mutation_rate) {
            individual->gaussians[i].sigma[0] += mutation_strength * rand_normal(seed);
            if (individual->gaussians[i].sigma[0] < 0.01) individual->gaussians[i].sigma[0] = 0.01;
            if (individual->gaussians[i].sigma[0] > 2.0)  individual->gaussians[i].sigma[0] = 2.0;
        }
        if (rand_double(0.0, 1.0, seed) < mutation_rate) {
            individual->gaussians[i].sigma[1] += mutation_strength * rand_normal(seed);
            if (individual->gaussians[i].sigma[1] < 0.01) individual->gaussians[i].sigma[1] = 0.01;
            if (individual->gaussians[i].sigma[1] > 2.0)  individual->gaussians[i].sigma[1] = 2.0;
        }
    }
}

int roulette_wheel_selection(const double* fitness_values, int pop_size, unsigned int* seed) {
    double max_fitness = fitness_values[0];
    for (int i = 1; i < pop_size; i++) {
        if (fitness_values[i] > max_fitness) {
            max_fitness = fitness_values[i];
        }
    }
    
    double total_fitness = 0.0;
    double* inverted_fitness = (double*)malloc(pop_size * sizeof(double));
    
    for (int i = 0; i < pop_size; i++) {
        inverted_fitness[i] = max_fitness - fitness_values[i] + 1e-9;
        total_fitness += inverted_fitness[i];
    }
    
    double r = rand_double(0.0, total_fitness, seed);
    double cumulative = 0.0;
    int selected_idx = pop_size - 1;
    
    for (int i = 0; i < pop_size; i++) {
        cumulative += inverted_fitness[i];
        if (cumulative >= r) {
            selected_idx = i;
            break;
        }
    }
    
    free(inverted_fitness);
    return selected_idx;
}