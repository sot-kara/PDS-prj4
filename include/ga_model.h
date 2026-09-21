#ifndef GA_MODEL_H
#define GA_MODEL_H

#define MAX_GAUSSIANS 15
#define U1_MIN -1.0
#define U1_MAX 2.0
#define U2_MIN -2.0
#define U2_MAX 1.0

typedef struct {
    double w;
    double c[2];
    double sigma[2];
} Gaussian;

typedef struct {
    Gaussian gaussians[MAX_GAUSSIANS];
} Model;

typedef double (*TargetFunc)(double, double);

// Added thread-local seed parameter to PRNG functions
double rand_double(double min, double max, unsigned int* seed);
double rand_normal(unsigned int* seed);
double gaussian_eval(double u1, double u2, const double c[2], const double sigma[2]);

// Added seed to fitness evaluation
double fitness_function(const Model* model, TargetFunc func, int N, int M, unsigned int* seed);

void crossover(const Model* parent1, const Model* parent2, Model* child1, Model* child2, double crossover_rate, int M);
void mutation(Model* individual, double mutation_rate, double mutation_strength, int M, unsigned int* seed);
int roulette_wheel_selection(const double* fitness_values, int pop_size, unsigned int* seed);

#endif // GA_MODEL_H