cmake_minimum_required(VERSION 3.10)
project(GeneticAlgorithmModel)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED True)

# Set optimization flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -Wall -Wextra")

# Set the output directory for all executables to a 'build' folder
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/build)

add_executable(ga_model
    main.c
    ga_model.c
)

# Link math library (required for sin, pow, exp, sqrt)
target_link_libraries(ga_model m)