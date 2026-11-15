import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("mushrooms.csv")

X = pd.get_dummies(df.drop("class", axis=1)).values.astype(float)
y = df["class"].map({"e": 0, "p": 1}).values.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

np.random.seed(1)

W1 = np.random.randn(X_train.shape[1], 20) * 0.1
b1 = np.zeros((1, 20))

W2 = np.random.randn(20, 5) * 0.1
b2 = np.zeros((1, 5))

W3 = np.random.randn(5,1) * 0.1
b3 = np.zeros((1,1))
lr = 0.1
errors = []
for epoch in range(5000):

    Z1 = X_train @ W1 + b1
    A1 = sigmoid(Z1)

    Z2 = A1 @ W2 + b2
    A2 = sigmoid(Z2)

    Z3 = A2 @ W3 + b3
    A3 = sigmoid(Z3)
    loss = -np.mean(
        y_train * np.log(A3 + 1e-8) +
        (1-y_train) * np.log(1-A3 + 1e-8)
    )
    errors.append(loss)
    dZ3 = A3 - y_train
    dW3 = A2.T @ dZ3 / len(X_train)
    db3 = np.mean(dZ3, axis=0, keepdims=True)

    dZ2 = (dZ3 @ W3.T) * A2 * (1-A2)
    dW2 = A1.T @ dZ2 / len(X_train)
    db2 = np.mean(dZ2, axis=0, keepdims=True)

    dZ1 = (dZ2 @ W2.T) * A1*(1-A1)
    dW1 = (X_train).T @ dZ1 / len(X_train)
    db1 = np.mean(dZ1,axis=0,keepdims=True)

    W3 -= lr * dW3
    b3 -= lr * db3

    W2 -= lr * dW2
    b2 -= lr * db2

    W1 -= lr * dW1
    b1 -= lr * db1

A1 = sigmoid(X_test @ W1 + b1)
A2 = sigmoid(A1 @ W2 + b2)
pred = sigmoid(A2 @ W3 + b3)

pred = (pred >= 0.5).astype(int)

nn_accuracy = accuracy_score(y_test, pred)

print("Neural Network Accuracy:", nn_accuracy * 100, "%")

plt.plot(errors)
plt.xlabel("Epoch")
plt.ylabel("Error")
plt.title("Neural Network Error")
plt.show()


tree = DecisionTreeClassifier()
tree.fit(X_train, y_train.ravel())

tree_pred = tree.predict(X_test)

tree_accuracy = accuracy_score(y_test, tree_pred)

print("Decision Tree Accuracy:", tree_accuracy * 100, "%")