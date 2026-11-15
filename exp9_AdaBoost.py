import numpy as np

# Dataset
X = np.array([
    [1], [2], [3], [4], [5], [6]
])

y = np.array([-1, -1, -1, 1, 1, 1])

# Decision stump
def stump(X, weights):
    best_error = float('inf')
    best = None

    for threshold in np.unique(X[:, 0]):
        for polarity in [1, -1]:
            pred = np.ones(len(y))
            pred[polarity * X[:, 0] < polarity * threshold] = -1

            error = np.sum(weights[pred != y])

            if error < best_error:
                best_error = error
                best = (threshold, polarity)

    return best, best_error

# AdaBoost
def adaboost(X, y, n_estimators=5):
    n = len(y)
    weights = np.ones(n) / n
    models = []

    for _ in range(n_estimators):
        stump_model, error = stump(X, weights)

        error = np.clip(error, 1e-10, 1 - 1e-10)
        alpha = 0.5 * np.log((1 - error) / error)

        threshold, polarity = stump_model

        pred = np.ones(n)
        pred[polarity * X[:, 0] < polarity * threshold] = -1

        weights *= np.exp(-alpha * y * pred)
        weights /= np.sum(weights)

        models.append((threshold, polarity, alpha))

    return models

# Prediction
def predict(X, models):
    final = np.zeros(len(X))

    for threshold, polarity, alpha in models:
        pred = np.ones(len(X))
        pred[polarity * X[:, 0] < polarity * threshold] = -1
        final += alpha * pred

    return np.where(final >= 0, 1, -1)

models = adaboost(X, y)
predictions = predict(X, models)

print("Actual:", y)
print("Predicted:", predictions)
print("Accuracy:", np.mean(y == predictions) * 100)
