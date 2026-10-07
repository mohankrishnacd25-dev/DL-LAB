import numpy as np
from sklearn.datasets import make_moons

# ---- Data ----
X, y = make_moons(n_samples=400, noise=0.20, random_state=1)
y = y.reshape(-1, 1).astype(float)
X = (X - X.mean(0)) / X.std(0) # standardize

# ---- Helpers ----
def relu(z): return np.maximum(0, z)
def drelu(z): return (z > 0).astype(float)
def sig(z): return 1 / (1 + np.exp(-z))

def init(sizes, seed=0):
    rng = np.random.default_rng(seed)
    P = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        # He initialisation
        W = rng.normal(0, np.sqrt(2/a), size=(a, b)) 
        P.append([W, np.zeros((1, b))])
    return P

def grads(P, Xb, yb): 
    # forward + backward on a batch
    zs, acts, a = [], [Xb], Xb
    for i, (W, b) in enumerate(P):
        z = a @ W + b
        zs.append(z)
        a = sig(z) if i == len(P) - 1 else relu(z)
        acts.append(a)
        
    m = Xb.shape[0]
    g = [None] * len(P)
    
    # dL/dz = yhat - y (sigmoid + BCE)
    dz = (acts[-1] - yb) / m 
    
    for i in reversed(range(len(P))):
        g[i] = [acts[i].T @ dz, dz.sum(0, keepdims=True)]
        if i > 0:
            dz = (dz @ P[i][0].T) * drelu(zs[i-1])
    return g

def accuracy(P, X, y):
    a = X
    for i, (W, b) in enumerate(P):
        a = sig(a @ W + b) if i == len(P) - 1 else relu(a @ W + b)
    return np.mean((a > 0.5) == (y > 0.5))

sizes, lr, EPOCHS = [2, 16, 16, 1], 0.5, 200

# ---- (A) Batch Gradient Descent: ALL data, 1 update / epoch ----
P = init(sizes, seed=0)
for epoch in range(EPOCHS):
    g = grads(P, X, y)
    for k in range(len(P)):
        P[k][0] -= lr * g[k][0]
        P[k][1] -= lr * g[k][1]
print("Batch GD accuracy :", accuracy(P, X, y))

# ---- (B) Stochastic Gradient Descent: mini-batches, N/B updates / epoch ----
P = init(sizes, seed=0)
B = 16
idx = np.arange(len(X))
for epoch in range(EPOCHS):
    np.random.shuffle(idx) # shuffle every epoch!
    for s in range(0, len(X), B):
        b = idx[s : s + B]
        g = grads(P, X[b], y[b])
        for k in range(len(P)):
            P[k][0] -= lr * g[k][0]
            P[k][1] -= lr * g[k][1]
print("SGD accuracy :", accuracy(P, X, y))