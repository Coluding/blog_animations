import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import quad


# Define the discontinuous function (step function)
def f(x):
    return np.where(np.abs(x) < np.pi / 2, 1, -1)  # Step function centered at 0

# Define the Fejér kernel function
def fejer_kernel(u, n):
    numerator = np.sin((n + 1) * u / 2) ** 2
    denominator = (2 * (n + 1) * (np.sin(u / 2) ** 2))

    # Handle singularities at u = 2mπ
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(np.abs(denominator) < 1e-10, 1 / (2 * (n + 1)), numerator / denominator)

    return result

# Convolution of f(x) with F_n(u) using numerical integration
def convolution_fejer(x, n):
    integral, _ = quad(lambda u: f(x - u) * fejer_kernel(u, n), -np.pi, np.pi)
    return integral / np.pi


# Convolution of f(x) with F_n(u) using numerical integration
def convolution_fejer(x, n):
    integral, _ = quad(lambda u: f(x - u) * fejer_kernel(u, n), -np.pi, np.pi)
    return integral / np.pi


# Define x and u values
x_vals = np.linspace(-np.pi, np.pi, 200)
u_vals = np.linspace(-np.pi, np.pi, 200)

# Initialize figure
fig, ax = plt.subplots(2, 1, figsize=(8, 8))

# Plot settings
ax[0].set_xlim(-np.pi, np.pi)
ax[0].set_ylim(-0.5, 2)  # Adjust limits to show kernel behavior
ax[0].set_title("Function $f(x)$ and Fejér Kernel $F_n(u)$")
ax[0].set_xlabel("$x$")
ax[0].set_ylabel("Value")
ax[0].grid(True)

ax[1].set_xlim(-np.pi, np.pi)
ax[1].set_ylim(-1.5, 1.5)
ax[1].set_title("Convolution: $s_n(x) = (f * F_n)(x)$")
ax[1].set_xlabel("$x$")
ax[1].set_ylabel("$s_n(x)$")
ax[1].grid(True)

# Plot function and Fejér kernel
f_plot, = ax[0].plot(x_vals, f(x_vals), label=r'$f(x)$', color='black', linestyle='dashed')
f_kernel_plot, = ax[0].plot(u_vals, fejer_kernel(u_vals, 1), label=r'$F_n(u)$', color='red')

conv_plot, = ax[1].plot([], [], label=r'$s_n(x)$ (Convolution)', color='blue')

ax[0].legend()
ax[1].legend()


# Animation update function
def update(n):
    # Update Fejér kernel
    kernel_values = fejer_kernel(u_vals, n)
    f_kernel_plot.set_ydata(kernel_values)

    # Compute convolution dynamically for increasing n
    conv_values = np.array([convolution_fejer(x, n) for x in x_vals])
    conv_plot.set_data(x_vals, conv_values)

    ax[0].set_title(f"Fejér Kernel for n={n}")
    ax[1].set_title(f"Convolution Process for n={n}")

    return f_kernel_plot, conv_plot,


# Create animation (n varying from 1 to 100 in steps of 5)
n_values = np.arange(1, 150, 5)
ani = animation.FuncAnimation(fig, update, frames=n_values, interval=300, blit=False)

# Save the animation
ani.save("fejer_convolution_animation.mp4", writer="ffmpeg", fps=10)

# Show animation
plt.show()
