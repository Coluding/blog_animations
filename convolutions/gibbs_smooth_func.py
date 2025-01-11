import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import quad


# Define the smooth function
def f(x):
    return x ** 3   # Smooth function (parabola)


# Define the Dirichlet kernel function
def dirichlet_kernel(u, n):
    numerator = np.sin((2 * n + 1) * u / 2)
    denominator = 2 * np.sin(u / 2)

    # Handle singularity at u = 0
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(np.abs(denominator) < 1e-10, 2 * n + 1, numerator / denominator)

    return result


# Convolution of f(x) with D_n(u) using numerical integration
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
ax[0].set_ylim(-10, 20)  # Adjust limits to show kernel behavior
ax[0].set_title("Function $f(x) = x^3$ and Dirichlet Kernel $D_n(u)$")
ax[0].set_xlabel("$x$")
ax[0].set_ylabel("Value")
ax[0].grid(True)

ax[1].set_xlim(-np.pi, np.pi)
ax[1].set_ylim(-10, 10)
ax[1].set_title("Convolution: $s_n(x) = (f * D_n)(x)$")
ax[1].set_xlabel("$x$")
ax[1].set_ylabel("$s_n(x)$")
ax[1].grid(True)

# Plot function and Dirichlet kernel
f_plot, = ax[0].plot(x_vals, f(x_vals), label=r'$f(x) = x^2$', color='black', linestyle='dashed')
d_plot, = ax[0].plot(u_vals, dirichlet_kernel(u_vals, 1), label=r'$D_n(u)$', color='blue')

# Plot convolution output
conv_plot, = ax[1].plot([], [], label=r'$s_n(x)$ (Convolution)', color='red')

ax[0].legend()
ax[1].legend()


# Animation update function
def update(n):
    # Update Dirichlet kernel
    kernel_values = dirichlet_kernel(u_vals, n)
    d_plot.set_ydata(kernel_values)

    # Compute convolution dynamically for increasing n
    conv_values = np.array([convolution(x, n) for x in x_vals])
    conv_plot.set_data(x_vals, conv_values)

    ax[0].set_title(f"Dirichlet Kernel for n={n}")
    ax[1].set_title(f"Convolution Process for n={n}")

    return d_plot, conv_plot,


# Create animation (n varying from 1 to 100 in steps of 5)
n_values = np.arange(1, 101, 5)
ani = animation.FuncAnimation(fig, update, frames=n_values, interval=300, blit=False)

# Save the animation
ani.save("dirichlet_smooth_function.mp4", writer="ffmpeg", fps=10)

# Show animation
plt.show()
