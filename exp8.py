import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv("heart.csv")
X = data.drop("condition", axis=1).values
y = data["condition"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

y = np.where(y_train == 0, -1, 1)
n = len(X_train)
C = 1.0
epochs = 10000
lr = 0.01

w = np.zeros(X_train.shape[1])
b = 0

for epoch in range(epochs):
    for i in range(n):
        if y[i] * (np.dot(X_train[i], w) + b) < 1:
            w -= lr * (w - C * y[i] * X_train[i])
            b += lr * C * y[i]
        else:
            w -= lr * w

pred1 = np.where(X_test @ w + b >= 0, 1, 0)
print("Linear SVM accuracy:", accuracy_score(y_test, pred1))

print("Linear SVM support vectors:")
for i in range(n):
    if abs(y[i] * (np.dot(X_train[i], w) + b) - 1) < 0.5:
        print(X_train[i])

gamma = 0.1
alpha = np.zeros(n)
b = 0

K = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        K[i, j] = np.exp(-gamma * np.sum((X_train[i] - X_train[j]) ** 2))

for epoch in range(epochs):
    for i in range(n):
        score = np.sum(alpha * y * K[:, i]) + b
        if y[i] * score < 1:
            alpha[i] = min(alpha[i] + lr, C)
            b += lr * y[i]

pred2 = []
for x in X_test:
    score = 0
    for i in range(n):
        score += alpha[i] * y[i] * np.exp(
            -gamma * np.sum((X_train[i] - x) ** 2)
        )
    pred2.append(1 if score + b >= 0 else 0)

print("RBF SVM support vectors:")
for i in range(n):
    if alpha[i] > 0:
        print(X_train[i])

print("\nRBF SVM accuracy:", accuracy_score(y_test, pred2))

tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)
print("\nDecision Tree accuracy:",
      accuracy_score(y_test, tree.predict(X_test)))

nn = MLPClassifier(hidden_layer_sizes=(10,),
                   max_iter=3000, random_state=42)
nn.fit(X_train, y_train)
print("Neural Network accuracy:",
      accuracy_score(y_test, nn.predict(X_test)))
