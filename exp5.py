from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np

df= pd.read_csv("electricity.csv")
#df.info()
df_copy= df.copy()
df_copy.drop_duplicates(inplace=True)
df_copy= pd.get_dummies(df_copy,columns=["structure type"],drop_first=True)
#print(df_copy.columns.tolist())
#plt.scatter(df_copy["site area"], df_copy["electricity cost"])
#plt.show()

x=df_copy.drop("electricity cost",axis=1)
y=df_copy["electricity cost"]

x=x.values
y=y.values

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
def gradient_descent(x, y, weights, bias, learning_rate, epochs):

    n = len(y)
    errors = []

    for i in range(epochs):

        y_pred = np.dot(x, weights) + bias
        error = y - y_pred
        mse = np.mean(error ** 2)
        errors.append(mse)

        dw = -(2 / n) * np.dot(x.T, error)
        db = -(2 / n) * np.sum(error)

        weights=weights-learning_rate*dw
        bias=bias-learning_rate*db

    return weights,bias,errors

weight=np.zeros(x_train.shape[1])
bias = 0
learning_rate=0.01
epochs=1000
errors=[]

weight,bias,errors=gradient_descent(x_train,y_train,weight,bias,learning_rate,epochs)

y_pred_manual = np.dot(x_test, weight) + bias

r2_manual = r2_score(y_test, y_pred_manual)
mse_manual = mean_squared_error(y_test, y_pred_manual)

print("Manual Model Evaluation")
print(f"R-squared (R²): {r2_manual:.4f}")
print(f"Mean Squared Error (MSE): {mse_manual:.4f}")

#inbuilt model
model = LinearRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("Model Evaluation")
print(f"R-squared (R²): {r2:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")

plt.plot(range(1, epochs + 1), errors)

plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.title("Gradient Descent Error vs Epochs")

plt.grid()
plt.show()