import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("china_gdp.csv")
df.head(10)

plt.figure(figsize=(8,5))
x_data, y_data = (df["Year"].values, df["Value"].values)
plt.plot(x_data, y_data, "ro")
plt.ylabel("GDP")
plt.xlabel("Year")
plt.show()

X = np.arange(-5.0, 5.0, 0.1)
Y = 1.0 / (1.0 + np.exp(-X))

plt.plot(X,Y)
plt.ylabel("Dependent Variable")
plt.xlabel("Independent Variable")
plt.show()

def sigmoid (x, Beta_1, Beta_2):
    y = 1 / (1 + np.exp(-Beta_1*(x-Beta_2)))
    return y

beta_1 = 0.10
beta_2 = 1990.0

#logistic function
Y_pred = sigmoid(x_data, beta_1, beta_2)

#plot initial prediction against datapoints
plt.plot(x_data, Y_pred*15000000000000.)
plt.plot(x_data, y_data, "ro")

xdata = x_data/max(x_data)
ydata = y_data/max(y_data)

from scipy.optimize import curve_fit
popt, pcov = curve_fit(sigmoid, xdata, ydata)
#print the final pameters
print(" beta_1 =%f, beta_2= %f" %(popt[0], popt[1]))

x = np.linspace(1960, 2015, 55)
x = x/max(x)
plt.figure(figsize=(8,5))
y = sigmoid(x, *popt)
plt.plot(xdata, ydata, "ro", label="data")
plt.plot(x,y, linewidth=3.0, label="fit")
plt.legend(loc="best")
plt.ylabel("GDP")
plt.xlabel("Year")
plt.show()

from sklearn.metrics import r2_score

# split data to train/test
msk = np.random.rand(len(df)) < 0.8
train_x = x_data[msk]
test_x = x_data[~msk]
train_y = y_data[msk]
test_y = y_data[~msk]

#build model using train set
popt, pcov = curve_fit(sigmoid, train_x, train_y)

# predict using test set
y_hat = sigmoid(test_x,*popt)

print("Mean absolute error: %.2f" % np.mean(np.absolute((y_hat - test_y))))
print("Residual sum of squares (MSE): %.2f" % np.mean(np.absolute((y_hat - test_y))))
print("R2-score: %.2f" % r2_score(y_hat , test_y) )