#!/bin/bash

EXEC="./build/island"
CSV_FILE="island_benchmark.csv"

if [ ! -f "$EXEC" ]; then
    echo "Error: Executable $EXEC not found. Please compile the project first."
    exit 1
fi

# Initialize CSV header
echo "islands,time,mse" > "$CSV_FILE"

# Define the sequence of island counts to test
ISLANDS=(1 2 4 8 10 15 20 50)

for i in "${ISLANDS[@]}"; do
    echo "Running with NUM_ISLANDS=$i..."
    
    # Capture standard output
    OUTPUT=$("$EXEC" "$i")
    
    # Extract the execution time and MSE floating point values
    TIME_VAL=$(echo "$OUTPUT" | awk '/Island Evolution Time:/ {print $4}')
    MSE_VAL=$(echo "$OUTPUT" | awk '/Final Best MSE \(Island Model\):/ {print $6}')
    
    # Append to CSV
    echo "$i,$TIME_VAL,$MSE_VAL" >> "$CSV_FILE"
done

echo "Benchmarking complete. Results saved to $CSV_FILE."