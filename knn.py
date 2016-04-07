from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.cross_validation import train_test_split

matrix_iris=np.loadtxt("data/iris/iris.data")
classes=matrix_iris[:,4]
matrix_iris=matrix_iris[:,0:3]
X_train, X_test, Y_train, Y_test = train_test_split(matrix_iris, classes, test_size=0.6)

neigh = KNeighborsClassifier(n_neighbors=10)
neigh.fit(X_train, Y_train) 
Y_returned=neigh.predict(X_test)
results=(Y_returned == Y_test)
print results
print neigh.predict_proba(X_test)


