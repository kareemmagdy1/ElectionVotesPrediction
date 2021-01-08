import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# read dataset
df = pd.read_csv("house_data.csv")

x1 = df["grade"]
x2 = df["bathrooms"]
x3 = df["lat"]
x4 = df["sqft_living"]
x5 = df["view"]
max_x1 = max(x1)
max_x2 = max(x2)
max_x3 = max(x3)
max_x4 = max(x4)
max_x5 = max(x5)
x1 = x1 / max_x1
x2 = x2 / max_x2
x3 = x3 / max_x3
x4 = x4 / max_x4
x5 = x5 / max_x5
y = df["price"]
max_y = max(y)
#y = y / max_y
n = 16500
# split dataset into training and testing data
X1_train = x1[:n]
X2_train = x2[:n]
X3_train = x3[:n]
X4_train = x4[:n]
X5_train = x5[:n]
Y_train = y[:n]

X1_test = x1[n:]
X2_test = x2[n:]
X3_test = x3[n:]
X4_test = x4[n:]
X5_test = x5[n:]
Y_test = y[n:]
# Another method to split dataset into training and testing data
# X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.2)
delta = 0.01
theta0 = 0
theta1 = sum(y - x1) / 21614
theta2 = sum(y - x2) / 21614
theta3 = sum(y - x3) / 21614
theta4 = sum(y - x4) / 21614
theta5 = sum(y - x5) / 21614
for i in range(1000):
    y_pred = theta0 + theta1 * x1 + theta2 * x2 + theta3 * x3 + theta4 * x4 + theta5 * x5
    diff = y_pred[:n] - Y_train
    new_theta0 = theta0 - delta * (sum(diff) / n)
    new_theta1 = theta1 - delta * (sum(diff * x1[:n]) / n)
    new_theta2 = theta2 - delta * (sum(diff * x2[:n]) / n)
    new_theta3 = theta3 - delta * (sum(diff * x3[:n]) / n)
    new_theta4 = theta4 - delta * (sum(diff * x4[:n]) / n)
    new_theta5 = theta5 - delta * (sum(diff * x5[:n]) / n)
    mse = sum(np.square(y_pred[:n] - Y_train)) / (n)
    if abs(theta0 - new_theta0) < 0.00001 and abs(theta1 - new_theta1) < 0.00001 and abs(
            theta2 - new_theta2) < 0.00001 and abs(theta3 - new_theta3) < 0.00001 and abs(
            theta4 - new_theta4) < 0.00001 and abs(theta5 - new_theta5) < 0.00001:
        break
    #if abs(theta0 - new_theta0) > 0.08 and abs(theta1 - new_theta1) > 0.08 and abs(theta2 - new_theta2) > 0.08 and abs(
      #      theta3 - new_theta3) > 0.08 and abs(theta4 - new_theta4) > 0.08 and abs(theta5 - new_theta5) > 0.08:
       # delta = delta / 10
       # i = 0
       # continue
    theta0 = new_theta0
    theta1 = new_theta1
    theta2 = new_theta2
    theta3 = new_theta3
    theta4 = new_theta4
    theta5 = new_theta5

plt.xlabel("X")
plt.ylabel("Y")
plt.title("Real vs predicted values")
#plt.scatter(X1_test, Y_test)
#plt.show()
mse = sum((y_pred[n:] - Y_test) * (y_pred[n:] - Y_test)) / (2 * (21614 - n))
print("MSE = ", mse * max_y)
print("theta0: ", theta0)
print("theta1: ", theta1)
print("theta2: ", theta2)
print("theta3: ", theta3)
print("theta4: ", theta4)
print("theta5: ", theta5)

print("y = ", theta0, " + ", theta1, "x1 + ", theta2, "x2 + ", theta3, "x3 + ", theta4, "x4 + ", theta5, "x5")
res = theta0 + theta1*(7/max_x1) + theta2/max_x2 + theta3*(47.5112/max_x3) + theta4*(1180/max_x4)
print(res)