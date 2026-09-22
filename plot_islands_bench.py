import pandas as pd
import matplotlib.pyplot as plt

csv_file = 'island_benchmark.csv'

try:
    df = pd.read_csv(csv_file)
except FileNotFoundError:
    print(f"Error: Could not find {csv_file}. Run test_island.sh first.")
    exit(1)

fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot Execution Time on primary y-axis
color_time = 'tab:blue'
ax1.set_xlabel('Number of Islands')
ax1.set_ylabel('Execution Time (seconds)', color=color_time, weight='bold')
line1 = ax1.plot(df['islands'], df['time'], marker='o', linestyle='-', color=color_time, linewidth=2, label='Execution Time')
ax1.tick_params(axis='y', labelcolor=color_time)
ax1.grid(True, linestyle='--', alpha=0.7)

# Set x-axis ticks to match the exact test cases
ax1.set_xticks(df['islands'])

# Create a twin Axes sharing the same x-axis
ax2 = ax1.twinx()

# Plot MSE on secondary y-axis
color_mse = 'tab:red'
ax2.set_ylabel('Final Best MSE', color=color_mse, weight='bold')
line2 = ax2.plot(df['islands'], df['mse'], marker='s', linestyle='--', color=color_mse, linewidth=2, label='MSE')
ax2.tick_params(axis='y', labelcolor=color_mse)

# Combine legends from both axes
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=2)

plt.title('Island Model Performance: Execution Time & MSE vs Number of Islands', pad=25, weight='bold')
fig.tight_layout()

plt.show()