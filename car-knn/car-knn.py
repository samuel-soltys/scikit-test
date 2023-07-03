import numpy as np
import pandas as pd
from sklearn import neighbors, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("car.data")

# print(data.head())

# X values of buying, maint, safety can be vhigh, high, med, low 
X = data[[
    "buying",
    "maint",
    "safety"
]].values

# y values of class can be unacc, acc, good, vgood
y = data[["class"]]

# Covnerting the data of X using LabelEncoder 
Le = LabelEncoder()
for i in range(len(X[0])):
    X[:, i] = Le.fit_transform(X[:, i])


# Converting the data of y using maping
label_maping = {
    "unacc": 0,
    "acc": 1,
    "good": 2,
    "vgood": 3
}
y["class"] = y["class"].map(label_maping)
y = np.array(y)

# Create KNN model
knn = neighbors.KNeighborsClassifier(n_neighbors=25, weights="uniform") #weights="distance"

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# print(X_train.shape)
knn.fit(X_train, y_train)

predictions = knn.predict(X_test)
accuracy = metrics.accuracy_score(y_test, predictions)

# print("predictions:", predictions)
print("accuracy:", accuracy)

a = 720
print("actual value:", y[700:a])
print("predicted value:", knn.predict(X)[700:a])
