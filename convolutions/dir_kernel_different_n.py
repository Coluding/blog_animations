import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Define the Dirichlet kernel function
def dirichlet_kernel(u, n):
    numerator = np.sin((2 * n + 1) * u / 2)
    denominator = 2 * np.sin(u / 2)

    # Handle singularity at u = 0
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(np.abs(denominator) < 1e-10, 2 * n + 1, numerator / denominator)

    return result

# Define the Fejér kernel function
def fejer_kernel(u, n):
    sum_Dk = np.zeros_like(u)
    for k in range(n + 1):
        sum_Dk += dirichlet_kernel(u, k)
    return sum_Dk / (n + 1)

# Define u values in the interval [-π, π]
u_vals = np.linspace(-np.pi, np.pi, 1000)

# Initialize figure
fig, ax = plt.subplots(figsize=(8, 5))
ax.set_xlim(-np.pi, np.pi)
ax.set_ylim(-0.5, 2)  # Adjust limits to show Fejér kernel behavior
ax.set_xlabel("$u$")
ax.set_ylabel("$F_n(u)$")
ax.set_title("Fejér Kernel Animation")
ax.grid(True)

# Initialize plot line
line, = ax.plot([], [], color='red', label=r'$F_n(u)$')
ax.legend()

# Animation update function
def update(n):
    y_vals = fejer_kernel(u_vals, n)  # Compute Fejér kernel for current n
    line.set_data(u_vals, y_vals)
    ax.set_title(f"Fejér Kernel for n={n}")
    return line,

# Create animation (n going from 5 to 200 in steps of 5)
n_values = np.arange(5, 200, 5)
ani = animation.FuncAnimation(fig, update, frames=n_values, interval=300, blit=False)

# Save the animation
ani.save("fejer_kernel_animation.mp4", writer="ffmpeg", fps=10)

# Show animation
plt.show()
