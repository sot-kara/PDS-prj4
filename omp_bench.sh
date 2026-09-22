#!/bin/bash

# Ensure the executable exists
EXEC="./build/parallel"
CSV_FILE="omp_benchmark.csv"

if [ ! -f "$EXEC" ]; then
    echo "Error: Executable $EXEC not found. Please compile the project first."
    exit 1
fi

# Initialize CSV header
echo "threads,time" > "$CSV_FILE"

for i in {1..16}
do
    echo "Running with OMP_NUM_THREADS=$i..."
    export OMP_NUM_THREADS=$i
    
    # Capture standard output
    OUTPUT=$($EXEC)
    
    # Extract the execution time floating point value
    TIME_VAL=$(echo "$OUTPUT" | awk '/Parallel Evolution Time:/ {print $4}')
    
    # Append to CSV
    echo "$i,$TIME_VAL" >> "$CSV_FILE"
done

echo "Benchmarking complete. Results saved to $CSV_FILE."