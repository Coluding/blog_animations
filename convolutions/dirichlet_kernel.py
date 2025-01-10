import numpy as np
import matplotlib.pyplot as plt


# Define the Dirichlet kernel function
def dirichlet_kernel(u, n):
    numerator = np.sin((2 * n + 1) * u / 2)
    denominator = 2 * np.sin(u / 2)

    # Handle division by zero at u = 0
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(denominator == 0, 2 * n + 1, numerator / denominator)

    return result


# Define u values
u = np.linspace(-4, 4, 1000)  # Range of u values

# Define different values of n
n_values = [100]

# Plot the Dirichlet kernels for different n
plt.figure(figsize=(10, 6))

for n in n_values:
    plt.plot(u, dirichlet_kernel(u, n), label=f"$D_{{{n}}}(u)$")

plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
plt.axvline(0, color='black', linewidth=0.8, linestyle='--')
plt.xlabel("$u$")
plt.ylabel("$D_n(u)$")
plt.title("Dirichlet Kernels for Different $n$")
plt.legend()
plt.grid(True)

fig = plt.gcf()
plt.savefig("dirichlet_kernel_100.png")

# Show the plot
plt.show()
