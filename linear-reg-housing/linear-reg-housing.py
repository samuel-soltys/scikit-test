# Linear regression
# supervised learning model (regression)
# models the line from training data and then test it

from sklearn import datasets
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

boston = datasets.load_boston()

X = boston.data
y = boston.target

print(X.shape)
print(y.shape)

l_reg = linear_model.LinearRegression()

# plt.scatter(X.T[5], y)
# plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

l_reg.fit(X_train, y_train)
predictions = l_reg.predict(X_test)

print(predictions)
# print(accuracy_score(predictions, y_test))

# more data points closer to the linear regression line = highet R^2 value
print("R^2 value:", l_reg.score(X, y))

print("coeff:", l_reg.coef_)
print("intercept:", l_reg.intercept_)