import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5.0,5.0,0.1)

#simple equation phương trình
#y = 2 *(x) + 3
# non linear regression y = a x^3 + b x^2 + c x + d
#y = 1*(x**3) + 1*(x**2) + 1*x + 3
# quadratic bậc 2 Y = X^2
# y = np.power(x,2)
# y_noise = 2 * np.random.normal(size=x.size);
# ydata = y+ y_noise
# plt.plot(x, ydata, "bo")
# plt.plot(x,y, 'r')
# plt.ylabel("Dependent variable")
# plt.xlabel("Independent variable")
# plt.show()

# exponential số mũ Y = a +b c^X
# X = np.arange(-5.0, 5.0, 0.1)
# Y = np.exp(X)
# logarithmic logarit y = log(x)
# X = np.arange(-5.0, 5.0, 0.1)
# Y = np.log(X)
# sigmoidal/logistic y = a+ (b/(1+c^(x-d)))
X = np.arange(-5.0, 5.0, 0.1)
Y = 1-4/(1+np.power(3, X-2))
print(Y)
plt.plot(X,Y)
plt.ylabel("Dependent variable")
plt.xlabel("Independent variable")
plt.show()
