import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
# read dataset
df = pd.read_csv("heart.csv")

x1 = df["trestbps"]
x2 = df["chol"]
x3 = df["thalach"]
x4 = df["oldpeak"]

y = df["target"]

delta = 0.01
theta0 = 0
theta1 = 1
theta2 = 1
theta3 = 1
theta4 = 1

for i in range(1000):
    y_pred = 1/(1+math.e**(-1*(theta0+theta1*x1+theta2*x2+theta3*x3+theta4*x4)))
    for i in (y_pred):
        if(i>0.5):
            i=1
        else:
            i=0
    diff = y_pred - y
    theta0 = theta0 - delta * (sum(diff) / len(diff))
    theta1 = theta1 - delta * (sum(diff * x1) / len(diff))
    theta2 = theta2 - delta * (sum(diff * x2) / len(diff))
    theta3 = theta3 - delta * (sum(diff * x3) / len(diff))
    theta4 = theta4 - delta * (sum(diff * x4) / len(diff))
    mse = sum(np.square(y_pred - y)) / (len(y))
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Real vs predicted values")
#plt.scatter(X1_test, Y_test)
#plt.show()
mse = sum((y_pred - y) * (y_pred - y)) / (2 * (len(y)))
print("MSE = ", mse)
print("theta0: ", theta0)
print("theta1: ", theta1)
print("theta2: ", theta2)
print("theta3: ", theta3)
print("theta4: ", theta4)

print("y = ", theta0, " + ", theta1, "x1 + ", theta2, "x2 + ", theta3, "x3 + ", theta4, "x4 + ")

test=1 / (1 + math.e ** (-1 * (theta0 + theta1 * 145 + theta2 * 233 + theta3 * 150 + theta4 * 2.3)))
if(test>0.5):
    print('1')
else:
    print("0")
