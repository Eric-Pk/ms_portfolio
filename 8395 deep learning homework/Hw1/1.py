import numpy as np
import matplotlib.pyplot as plt

data = np.load("hw1_p1.npy")
x = data[:, 0]
y = data[:, 1]
N = len(x)
sigma = 0.5

mu = lambda m: 1.25 * m - 3.75

phi = lambda x: np.array(
    [
        np.exp(-0.5 * (x - mu(m)) ** 2 / sigma**2)
        for m in range(0, 7)
    ]
).T

hessian = phi(x).T.dot(phi(x))
eigenvalues, eigenvectors = np.linalg.eig(hessian)
print(eigenvalues)
max_eig = np.max(eigenvalues)
eps = 2 / max_eig
print("eps_max: ", eps)

w_old = np.random.randn(7)
losses = []
grad_norms = []
convergence_compares = []

for i in range(1000):
    grad = phi(x).T.dot(phi(x).dot(w_old) - y)
    w = w_old - eps * grad
    w_old = w

    loss = 1 / 2 * np.linalg.norm(phi(x).dot(w) - y)
    grad_norm = grad.T.dot(grad)
    convergence_compare = (
        np.log(2 * max_eig * (losses[0] - loss)) - np.log(i + 1)
        if len(losses) > 1
        else 5.0
    )

    losses.append(loss)
    grad_norms.append(grad_norm)
    convergence_compares.append(convergence_compare)

    # print(
    #     f"iter: {i}, loss: {loss:.4f}, grad_norm: {grad_norm:.4f}, convergence_compare: {convergence_compare:.4f}"
    # )

# Plot for Data/Regression
plt.figure(figsize=(10, 5))
plt.title("Data")
plt.scatter(x, y, label="Data")
plt.scatter(x, phi(x).dot(w), color="orange", label="Regression")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.savefig("Data.png")


# Plot for Loss
plt.figure(figsize=(10, 5))
plt.plot(losses, label="Loss")
plt.title("Loss over Iterations")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.legend()
plt.savefig("Loss.png")

# Plot for Gradient Norm
plt.figure(figsize=(10, 5))
plt.plot(grad_norms, label="Gradient Norm", color="orange")
plt.title("Gradient Norm over Iterations")
plt.xlabel("Iteration")
plt.ylabel("Gradient Norm")
plt.legend()
plt.savefig("GradientNorm.png")


# The logarithm of norm-squared of the loss compared with the logarithm of the convergence upper bound
plt.figure(figsize=(10, 5))
plt.plot(convergence_compares, label="Convergence Compare", color="orange")
plt.title("Convergence Compare")
plt.xlabel("Iteration")
plt.ylabel("Convergence Compare")
plt.legend()
plt.savefig("ConvergenceCompare.png")
