# PDS-prj4

## Requirements
- cmake at least 3.10
- matplotlib, pandas and numpy for plotting

## How to build
You can run the build.sh script to build the project (don't forget to chmod +x it).
```bash
chmod +x build.sh
./build.sh
```
This will create a build directory where the executables will be.

Otherwise, you can manually build the project using cmake.


## Running
Run one of the three executables from the build directory:
```bash
./build/sequntial
./build/parallel
./build/island
```
To see output of each on stdout. They will also output a model_output.csv file for plotting the best model's prediction.

## Benchmarking
You can run omp_bench.sh and island_bench.sh to benchmark the corresponding executables.

## Plotting
Run plot_func.py in the same directory with the model_output.csv file to see the approximation compared to the real function.

Run plot_omp_bench.py or plot_islands_bench.py to get the corresponding plots of each benchmark (same directory as the .csv they output).