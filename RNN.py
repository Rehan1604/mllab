import numpy as np

np.random.seed(42)

# Sequence input
X = np.array([
    [1, 0],
    [0, 1],
    [1, 1]
], dtype=float)

input_size = 2
hidden_size = 3
output_size = 1

# Initialize weights
Wx = np.random.randn(input_size, hidden_size)
Wh = np.random.randn(hidden_size, hidden_size)
Wy = np.random.randn(hidden_size, output_size)
bh = np.zeros(hidden_size)
by = np.zeros(output_size)

# Initial hidden state
h = np.zeros(hidden_size)

# RNN forward propagation
for t in range(len(X)):
    h = np.tanh(
        np.dot(X[t], Wx) +
        np.dot(h, Wh) + bh
    )
    y = np.dot(h, Wy) + by

    print("Time step:", t+1)
    print("Hidden state:", h)
    print("Output:", y)

print("Final hidden state:", h)
