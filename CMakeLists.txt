cmake_minimum_required(VERSION 3.10)
project(GeneticAlgorithmModel C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED True)

# Set optimization flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -Wall -Wextra")

# Find OpenMP
find_package(OpenMP REQUIRED)

# Find Threads
find_package(Threads REQUIRED)

# Set the executable target directory directly into the current build output directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# ---------------------------------------------------------
# Target 1: Sequential Model
# ---------------------------------------------------------
add_executable(sequential
    src/seq.c
    src/ga_model.c
)
target_include_directories(sequential PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(sequential PRIVATE m)

# ---------------------------------------------------------
# Target 2: Parallel Model
# ---------------------------------------------------------
add_executable(parallel
    src/parallel.c
    src/ga_model.c
)
target_include_directories(parallel PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Link math library AND OpenMP
target_link_libraries(parallel PRIVATE m OpenMP::OpenMP_C)

# ---------------------------------------------------------
# Target 3: Island Model
# ---------------------------------------------------------

add_executable(island
    src/island.c
    src/ga_model.c
)
target_include_directories(island PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(island PRIVATE m Threads::Threads)
target_link_libraries(island PRIVATE m OpenMP::OpenMP_C)