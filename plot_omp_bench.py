import pandas as pd
import matplotlib.pyplot as plt

csv_file = 'omp_benchmark.csv'

try:
    df = pd.read_csv(csv_file)
except FileNotFoundError:
    print(f"Error: Could not find {csv_file}. Run the bash benchmark script first.")
    exit(1)

plt.figure(figsize=(10, 6))
plt.plot(df['threads'], df['time'], marker='o', linestyle='-', color='b', linewidth=2)

plt.title('OpenMP Scaling Performance')
plt.xlabel('Number of OpenMP Threads')
plt.ylabel('Execution Time (seconds)')

# Force x-axis ticks to show every thread count from 1 to 16
plt.xticks(range(1, 17))
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()