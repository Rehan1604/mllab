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
Wf = np.random.randn(input_size, hidden_size)
Uf = np.random.randn(hidden_size, hidden_size)
Wi = np.random.randn(input_size, hidden_size)
Ui = np.random.randn(hidden_size, hidden_size)
Wo = np.random.randn(input_size, hidden_size)
Uo = np.random.randn(hidden_size, hidden_size)
Wc = np.random.randn(input_size, hidden_size)
Uc = np.random.randn(hidden_size, hidden_size)

# Activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

# Initial hidden state and cell state
h = np.zeros(hidden_size)
C = np.zeros(hidden_size)

# LSTM forward propagation
for t in range(len(X)):
    f = sigmoid(X[t] @ Wf + h @ Uf)
    i = sigmoid(X[t] @ Wi + h @ Ui)
    o = sigmoid(X[t] @ Wo + h @ Uo)
    C_candidate = np.tanh(X[t] @ Wc + h @ Uc)
    C = f * C + i * C_candidate
    h = o * np.tanh(C)

    print("Time step:", t+1)
    print("Forget gate:", f)
    print("Input gate:", i)
    print("Output gate:", o)
    print("Cell state:", C)
    print("Hidden state:", h)

print("Final output:", h)
