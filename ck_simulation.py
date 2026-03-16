import numpy as np
import matplotlib.pyplot as plt

# =========================
# Parameters
# =========================
alpha = 1.0
beta = 0.6
steps = 800
dt = 0.05
eps = 1e-4

# =========================
# Reference Integration Core (IRC)
# =========================
irc = np.array([0.0, 0.0])

# Barrier positions
barriers = [
    np.array([2.0, 2.0]),
    np.array([-2.0, 1.5]),
    np.array([1.5, -2.0])
]

# =========================
# Core functions
# =========================
def D(x):
    """Distance to the IRC."""
    return np.linalg.norm(x - irc)

def B(x):
    """Barrier field."""
    val = 0.0
    for b in barriers:
        val += np.exp(-np.linalg.norm(x - b) ** 2)
    return val

def Ck(x):
    """Coherence Complexity."""
    return alpha * D(x) + beta * B(x)

def grad_Ck(x):
    """Numerical gradient of Ck."""
    grad = np.zeros(2)
    for i in range(2):
        dx = np.zeros(2)
        dx[i] = eps
        grad[i] = (Ck(x + dx) - Ck(x - dx)) / (2 * eps)
    return grad

# =========================
# Simulation
# =========================
x = np.array([3.0, -3.0])
trajectory = []

for _ in range(steps):
    trajectory.append(x.copy())
    x = x - grad_Ck(x) * dt

trajectory = np.array(trajectory)

# =========================
# Landscape grid
# =========================
xs = np.linspace(-4, 4, 120)
ys = np.linspace(-4, 4, 120)
Z = np.zeros((len(ys), len(xs)))

for i, xg in enumerate(xs):
    for j, yg in enumerate(ys):
        Z[j, i] = Ck(np.array([xg, yg]))

# =========================
# Plot
# =========================
plt.figure(figsize=(7, 7))

plt.contour(xs, ys, Z, levels=25)
plt.plot(trajectory[:, 0], trajectory[:, 1], linewidth=2, label="Trajectory")
plt.scatter(irc[0], irc[1], s=80, label="IRC")

for b in barriers:
    plt.scatter(b[0], b[1], s=60, marker="x")

plt.title("Ck Landscape and Gradient Dynamics")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.axis("equal")
plt.show()
