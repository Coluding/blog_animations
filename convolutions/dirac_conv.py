import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Define the function to convolve
def f(x):
    return np.sin(2 * np.pi * x) * np.exp(-x**2) 

# Approximate the Dirac delta function using a narrow Gaussian
def dirac_delta(x, epsilon=0.05):
    return np.exp(-x**2 / (2 * epsilon**2)) / (np.sqrt(2 * np.pi) * epsilon)

# Define the x range
x = np.linspace(-5, 5, 400)
dx = x[1] - x[0]

# Compute function values
y_f = f(x)
y_delta = dirac_delta(x)

# Compute the convolution (approximating the integral with discrete summation)
y_conv = np.convolve(y_f, y_delta, mode='same') * dx

# Create figure and axes
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlim(-5, 5)
ax.set_ylim(1.2 * min(min(y_f), min(y_delta), min(y_conv)), 1.2 * max(max(y_f), max(y_delta), max(y_conv)))

# Plot the original functions
line_f, = ax.plot(x, y_f, label=r'$f(x)$', color='blue')
line_delta, = ax.plot(x, y_delta, label=r'$\delta(x)$', color='red')
line_conv, = ax.plot([], [], label=r'$(f * \delta)(x)$', color='green')

# Initialize the animation
def init():
    line_conv.set_data([], [])
    return line_conv,

# Update function for animation
def update(frame):
    line_conv.set_data(x[:frame], y_conv[:frame])
    return line_conv,

# Create animation
ani = animation.FuncAnimation(fig, update, frames=len(x), init_func=init, blit=True, interval=30)

# Add labels and legend
ax.set_xlabel("x")
ax.set_ylabel("Function value")
ax.legend()
ax.set_title("Convolution of $f(x)$ with Dirac Delta Function")

# Save the animation
ani.save("convolution_dirac_delta.mp4", writer="ffmpeg", fps=30)

# Show the animation
plt.show()
