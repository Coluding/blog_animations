import numpy as np
import matplotlib.pyplot as plt


# Define the Dirichlet kernel function
def dirichlet_kernel(u, n):
    numerator = np.sin((2 * n + 1) * u / 2)
    denominator = 2 * np.sin(u / 2)

    # Handle singularity at u = 0
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(np.abs(denominator) < 1e-10, 2 * n + 1, numerator / denominator)

    return result

def fejer_kernel(u, n):
    sum_Dk = np.zeros_like(u)
    for k in range(n + 1):
        sum_Dk += dirichlet_kernel(u, k)
    return sum_Dk / (n + 1)



# Define u values
u = np.linspace(-4, 4, 1000)  # Range of u values

# Define different values of n
n_values = [1,5,10,20,30]

# Plot the Dirichlet kernels for different n
plt.figure(figsize=(10, 6))

for n in n_values:
    plt.plot(u, fejer_kernel(u, n), label=f"$D_{{{n}}}(u)$")

plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
plt.axvline(0, color='black', linewidth=0.8, linestyle='--')
plt.xlabel("$u$")
plt.ylabel("$D_n(u)$")
plt.title("Fejèr Kernels for Different $n$")
plt.legend()
plt.grid(True)

fig = plt.gcf()
plt.savefig("fejer_kernel.png")

# Show the plot
plt.show()
