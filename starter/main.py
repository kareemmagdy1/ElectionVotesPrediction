import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# read dataset
df = pd.read_csv("../../../downloads/house_data.csv")

x = df["sqft_living"]
y = df["price"]
n = 16500
# split dataset into training and testing data
X_train = x[:n]
Y_train = y[:n]

X_test = x[n:]
Y_test = y[n:]
# Another method to split dataset into training and testing data
# X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.2)
delta = 0.5
theta0 = 0
theta1 = sum(y - x) / 21614
mse_container=[]
iterations=[]
for i in range(1000):
    y_pred = theta0 + theta1 * x
    diff = y_pred[:n] - Y_train
    new_theta0 = theta0 - delta * (sum(diff) / n)
    new_theta1 = theta1 - delta * (sum(diff * x[:n]) / n)
    mse = sum(np.square(y_pred[:n] - Y_train)) / (n)
    mse_container.append(mse)
    iterations.append(i)
    if abs(theta0 - new_theta0) < 0.00001 and abs(theta1 - new_theta1) < 0.00001:
        break
    if abs(theta0 - new_theta0) > 1000 and abs(theta1 - new_theta1) > 1000:
        delta = delta / 10
        i = 0
        continue
    theta0 = new_theta0
    theta1 = new_theta1


plt.xlabel("X")
plt.ylabel("Y")
plt.title("Real vs predicted values")
plt.scatter(X_test, Y_test)
plt.plot(X_test, y_pred[n:], color='red')

plt.show()
mse = sum((y_pred[n:] - Y_test) * (y_pred[n:] - Y_test)) / (2 * (21614 - n))
print(mse)
