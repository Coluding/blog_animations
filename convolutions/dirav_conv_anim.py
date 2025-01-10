import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import quad


# Define the function to approximate
def f(x):
    return np.where(np.abs(x) < np.pi / 2, 1, -1)  # A simple step function


# Define the Dirichlet kernel function
def dirichlet_kernel(u, n):
    numerator = np.sin((2 * n + 1) * u / 2)
    denominator = 2 * np.sin(u / 2)

    # Handle singularity at u = 0
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(np.abs(denominator) < 1e-10, 2 * n + 1, numerator / denominator)

    return result


# Convolution of f(x) with D_n(u)
def convolution(x, n):
    integral, _ = quad(lambda u: f(x - u) * dirichlet_kernel(u, n), -np.pi, np.pi)
    return integral / np.pi


# Define x and u values
x_vals = np.linspace(-np.pi, np.pi, 200)
u_vals = np.linspace(-np.pi, np.pi, 200)

# Initialize figure
fig, ax = plt.subplots(2, 1, figsize=(8, 8))

# Plot settings
ax[0].set_xlim(-np.pi, np.pi)
ax[0].set_ylim(-1.5, 1.5)
ax[0].set_title("Function $f(x)$ and Dirichlet Kernel $D_n(u)$")
ax[0].set_xlabel("$x$")
ax[0].set_ylabel("Value")
ax[0].grid(True)

ax[1].set_xlim(-np.pi, np.pi)
ax[1].set_ylim(-1.5, 1.5)
ax[1].set_title("Convolution: $s_n(x) = (f * D_n)(x)$")
ax[1].set_xlabel("$x$")
ax[1].set_ylabel("$s_n(x)$")
ax[1].grid(True)

# Plot function and Dirichlet kernel
f_plot, = ax[0].plot(x_vals, f(x_vals), label=r'$f(x)$', color='black', linestyle='dashed')
d_plot, = ax[0].plot(u_vals, dirichlet_kernel(u_vals, 1), label=r'$D_n(u)$', color='blue')

# Plot convolution output
conv_plot, = ax[1].plot([], [], label=r'$s_n(x)$ (Convolution)', color='red')

ax[0].legend()
ax[1].legend()

# Generate convolution values for n = 1
conv_values = np.array([convolution(x, 1) for x in x_vals])


# Animation update function
def update(frame):
    if frame < len(x_vals):  # First phase: Animate convolution for n = 1
        conv_plot.set_data(x_vals[:frame], conv_values[:frame])
        ax[1].set_title(f"Convolution Process for n=1")
    else:  # Second phase: Change n and update convolution
        n = (frame - len(x_vals)) // 5 + 2  # Increase n gradually
        d_plot.set_ydata(dirichlet_kernel(u_vals, n))  # Update kernel
        conv_values_new = np.array([convolution(x, n) for x in x_vals])
        conv_plot.set_data(x_vals, conv_values_new)
        ax[1].set_title(f"Convolution with Dirichlet Kernel for n={n}")
    return d_plot, conv_plot,


# Create animation
frames_total = len(x_vals) + 30  # Convolution first, then different n-values
ani = animation.FuncAnimation(fig, update, frames=frames_total, interval=50, blit=False)

# Save the animation
ani.save("dirichlet_full_convolution.mp4", writer="ffmpeg", fps=30)

# Show animation
plt.show()
