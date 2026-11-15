import numpy as np

np.random.seed(42)

# Input sequence
X = np.array([
    [1, 0],
    [0, 1],
    [1, 1]
], dtype=float)

input_size = 2
hidden_size = 3

# Initialize weights
Wz = np.random.randn(input_size, hidden_size)
Uz = np.random.randn(hidden_size, hidden_size)
Wr = np.random.randn(input_size, hidden_size)
Ur = np.random.randn(hidden_size, hidden_size)
Wh = np.random.randn(input_size, hidden_size)
Uh = np.random.randn(hidden_size, hidden_size)

# Activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

# Initial hidden state
h = np.zeros(hidden_size)

# GRU forward propagation
for t in range(len(X)):
    z = sigmoid(X[t] @ Wz + h @ Uz)
    r = sigmoid(X[t] @ Wr + h @ Ur)
    h_candidate = np.tanh(X[t] @ Wh + (r * h) @ Uh)
    h = (1 - z) * h + z * h_candidate

    print("Time step:", t+1)
    print("Update gate:", z)
    print("Reset gate:", r)
    print("Hidden state:", h)

print("Final hidden state:", h)
