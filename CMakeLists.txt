cmake_minimum_required(VERSION 3.10)
project(GeneticAlgorithmModel C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED True)

# Set optimization flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -Wall -Wextra")

# Set the executable target directory directly into the current build output directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# Add source files
add_executable(ga_model
    src/main.c
    src/ga_model.c
)

# Target include directory for headers
target_include_directories(ga_model PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Link math library
target_link_libraries(ga_model PRIVATE m)