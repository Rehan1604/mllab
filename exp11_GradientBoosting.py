import numpy as np

# Dataset
X = np.array([1, 2, 3, 4, 5, 6], dtype=float)
y = np.array([2, 4, 5, 8, 10, 12], dtype=float)

# Decision stump for regression
def train_stump(X, residuals):
    best_error = float('inf')
    best_threshold = None
    best_left = None
    best_right = None

    for threshold in np.unique(X):
        left = X <= threshold
        right = X > threshold

        if not np.any(left) or not np.any(right):
            continue

        left_value = np.mean(residuals[left])
        right_value = np.mean(residuals[right])

        pred = np.where(left, left_value, right_value)
        error = np.sum((residuals - pred) ** 2)

        if error < best_error:
            best_error = error
            best_threshold = threshold
            best_left = left_value
            best_right = right_value

    return best_threshold, best_left, best_right

# Gradient Boosting
def gradient_boosting(X, y, n_estimators=5, lr=0.1):
    initial = np.mean(y)
    predictions = np.full(len(y), initial)
    models = []

    for _ in range(n_estimators):
        residuals = y - predictions
        threshold, left, right = train_stump(X, residuals)
        stump_pred = np.where(X <= threshold, left, right)
        predictions += lr * stump_pred
        models.append((threshold, left, right))

    return initial, models

# Prediction
def predict(X, initial, models, lr=0.1):
    result = np.full(len(X), initial)

    for threshold, left, right in models:
        result += lr * np.where(X <= threshold, left, right)

    return result

initial, models = gradient_boosting(X, y)
predictions = predict(X, initial, models)

print("Actual:", y)
print("Predicted:", predictions)
print("MSE:", np.mean((y - predictions) ** 2))
