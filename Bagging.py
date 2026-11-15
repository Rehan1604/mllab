import numpy as np

X = np.array([[1], [2], [3], [4], [5], [6],
              [7], [8], [9], [10]], dtype=float)
y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

# Train a simple decision stump
def train_stump(X, y):
    best_error = float('inf')
    best_model = None

    for threshold in np.unique(X[:, 0]):
        for polarity in [1, -1]:
            pred = np.ones(len(y))
            pred[polarity * X[:, 0] < polarity * threshold] = 0
            error = np.sum(pred != y)

            if error < best_error:
                best_error = error
                best_model = (threshold, polarity)

    return best_model

# Bagging
def bagging(X, y, n_models=5):
    models = []

    for _ in range(n_models):
        indices = np.random.choice(
            len(X), len(X), replace=True
        )
        model = train_stump(X[indices], y[indices])
        models.append(model)

    return models

# Prediction
def predict(X, models):
    predictions = []

    for threshold, polarity in models:
        pred = np.ones(len(X))
        pred[polarity * X[:, 0] < polarity * threshold] = 0
        predictions.append(pred)

    predictions = np.array(predictions)
    return (np.mean(predictions, axis=0) >= 0.5).astype(int)

models = bagging(X, y)
y_pred = predict(X, models)

print("Actual:", y)
print("Predicted:", y_pred)
print("Accuracy:", np.mean(y == y_pred) * 100)
