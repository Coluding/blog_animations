import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import quad


# Define the discontinuous function (step function)
def f(x):
    return x**3  # Step function centered at 0


# Define the Dirichlet kernel function
def dirichlet_kernel(u, n):
    numerator = np.sin((2 * n + 1) * u / 2)
    denominator = 2 * np.sin(u / 2)

    # Handle singularity at u = 0
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(np.abs(denominator) < 1e-10, 2 * n + 1, numerator / denominator)

    return result


# Define the remainder term convolution
def remainder_convolution(x, n):
    integral, _ = quad(lambda u: ((f(x + u) + f(x - u)) / 2 - f(x)) * dirichlet_kernel(u, n), -np.pi, np.pi)
    return integral / np.pi


# Define x and u values
x_vals = np.linspace(-np.pi, np.pi, 200)
u_vals = np.linspace(-np.pi, np.pi, 200)

# Initialize figure
fig, ax = plt.subplots(2, 1, figsize=(8, 8))

# Plot settings
ax[0].set_xlim(-np.pi, np.pi)
ax[0].set_ylim(-1.5, 3.5)  # Adjust limits to show kernel behavior
ax[0].set_title("Function $f(x)$ and Dirichlet Kernel $D_n(u)$")
ax[0].set_xlabel("$x$")
ax[0].set_ylabel("Value")
ax[0].grid(True)

ax[1].set_xlim(-np.pi, np.pi)
ax[1].set_ylim(-0.5, 1)
ax[1].set_title("Error Term: $r_f(x) = s_n(x) - f(x)$")
ax[1].set_xlabel("$x$")
ax[1].set_ylabel("$r_f(x)$")
ax[1].grid(True)

# Plot function and Dirichlet kernel
f_plot, = ax[0].plot(x_vals, f(x_vals), label=r'$f(x)$', color='black', linestyle='dashed')
d_plot, = ax[0].plot(u_vals, dirichlet_kernel(u_vals, 1), label=r'$D_n(u)$', color='blue')

# Plot convolution output (error term)
error_plot, = ax[1].plot([], [], label=r'$r_f(x)$ (Error Term)', color='red')

ax[0].legend()
ax[1].legend()


# Animation update function
def update(n):
    # Update Dirichlet kernel
    kernel_values = dirichlet_kernel(u_vals, n)
    d_plot.set_ydata(kernel_values)

    # Compute error term dynamically for increasing n
    error_values = np.array([remainder_convolution(x, n) for x in x_vals])
    error_plot.set_data(x_vals, error_values)

    ax[0].set_title(f"Dirichlet Kernel for n={n}")
    ax[1].set_title(f"Error Term Convolution for n={n}")

    return d_plot, error_plot,


# Create animation (n varying from 1 to 100 in steps of 5)
n_values = np.arange(1, 101, 5)
ani = animation.FuncAnimation(fig, update, frames=n_values, interval=300, blit=False)

# Save the animation
ani.save("gibbs_error_term.mp4", writer="ffmpeg", fps=10)

# Show animation
plt.show()
