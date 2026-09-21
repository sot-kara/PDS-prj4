import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read the exported model parameters
# Adjust the path if your script is located elsewhere
csv_path = 'model_output.csv'
try:
    df = pd.read_csv(csv_path)
except FileNotFoundError:
    print(f"Error: Could not find {csv_path}. Ensure the C program has run successfully.")
    exit(1)

# Define the bounds based on the problem description
u1_min, u1_max = -1.0, 2.0
u2_min, u2_max = -2.0, 1.0

# Create a meshgrid for 3D plotting
u1_val = np.linspace(u1_min, u1_max, 100)
u2_val = np.linspace(u2_min, u2_max, 100)
u1, u2 = np.meshgrid(u1_val, u2_val)

# Calculate the true function: f(u1, u2) = sin(u1 + u2) * sin(u2^2)
f_true = np.sin(u1 + u2) * np.sin(u2**2)

# Reconstruct the predicted function from the Gaussian model
f_pred = np.zeros_like(f_true)
for index, row in df.iterrows():
    w = row['weight']
    c1, c2 = row['c1'], row['c2']
    sigma1, sigma2 = row['sigma1'], row['sigma2']
    
    # Calculate the exponential term for the current Gaussian
    exp_term = -(((u1 - c1)**2) / (2 * sigma1**2) + ((u2 - c2)**2) / (2 * sigma2**2))
    f_pred += w * np.exp(exp_term)

# Set up the figure for side-by-side plotting
fig = plt.figure(figsize=(14, 6))

# Subplot 1: True Function
ax1 = fig.add_subplot(121, projection='3d')
surf1 = ax1.plot_surface(u1, u2, f_true, cmap='viridis', alpha=0.8, edgecolor='none')
ax1.set_title('Original Function $f(u_1, u_2)$')
ax1.set_xlabel('$u_1$')
ax1.set_ylabel('$u_2$')
ax1.set_zlabel('Function Value')
ax1.view_init(elev=30, azim=45)
fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10)

# Subplot 2: GA Predicted Function
ax2 = fig.add_subplot(122, projection='3d')
surf2 = ax2.plot_surface(u1, u2, f_pred, cmap='plasma', alpha=0.8, edgecolor='none')
ax2.set_title('GA Approximation')
ax2.set_xlabel('$u_1$')
ax2.set_ylabel('$u_2$')
ax2.set_zlabel('Predicted Value')
ax2.view_init(elev=30, azim=45)
fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10)

plt.tight_layout()
plt.show()