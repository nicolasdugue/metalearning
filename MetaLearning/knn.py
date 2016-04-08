from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.cross_validation import train_test_split

'''
A simple program to build a knn classifier using IRIS data
'''
matrix_iris=np.loadtxt("../data/iris/iris.data")
classes=matrix_iris[:,4]
matrix_iris=matrix_iris[:,0:3]
X_train, X_test, Y_train, Y_test = train_test_split(matrix_iris, classes, test_size=0.9)

neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(X_train, Y_train) 
Y_returned=neigh.predict(X_test)
results=(Y_returned == Y_test)
print reduce(lambda x, y : int(x) +int(y), results )

print "results : ",reduce(lambda x, y : int(x) +int(y), results ), " / ", len(results)

import pyfmax.fmax as fm

fm_matrix=fm.MatrixClustered(X_train, Y_train)
contrasted_matrix=fm_matrix.contrast_and_select_matrix()
print fm_matrix.get_features_selected_flat()
neigh.fit(contrasted_matrix, Y_train) 
Y_returned=[]
for idx,vector in enumerate(X_test):
    best=0
    max=-1
    for k in range(3):
        vector_contrasted=fm_matrix.contrast_and_select_features(vector, k)
        if (k==0):
            prediction=neigh.predict_proba(vector_contrasted)
        else:
            prediction=prediction + neigh.predict_proba(vector_contrasted)
        if k==2:
            max=np.max(prediction)
            best=np.argmax(prediction)
            print vector, vector_contrasted,prediction, Y_test[idx], best
    Y_returned.append(best)
results=(Y_returned == Y_test)
print "results : ",reduce(lambda x, y : int(x) +int(y), results )