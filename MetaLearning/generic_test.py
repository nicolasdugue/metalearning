'''
Created on Apr 8, 2016

@author: nicolas
'''

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.lda import LDA
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import numpy as np
import pyfmax.fmax as fm

#lr = SVC(probability=True)
lr=LogisticRegression()
norm=StandardScaler()

matrix_iris=np.loadtxt("../data/iris/iris.data")
classes=matrix_iris[:,4]
matrix_iris=matrix_iris[:,0:3]
matrix_iris=norm.fit_transform(matrix_iris)

# metalearner=fm.MetaLearner(matrix_iris, classes, perct_test=0.5)
# metalearner.train(False, lr)
# results=metalearner.predict()
# print "Correct results (no contrast) : ",reduce(lambda x, y : int(x) +int(y), results )
# 
# metalearner.train(True, lr)
# results=metalearner.predict()
# print "Correct results (with contrast) : ", reduce(lambda x, y : int(x) +int(y), results )
#metalearner.pca_train(False)

#Now we test by keeping the two closest classes, the two that are particularly difficult to discriminate from each other
classe1=classes == 1
classe2=classes == 2
classes1and2=classe1+classe2
matrix_iris=matrix_iris[classes1and2]
classes=classes[classes1and2] - 1

metalearner=fm.MetaLearner(matrix_iris, classes, perct_test=0.1, magnitude=3)
metalearner.train(False, lr)
results=metalearner.predict()
print "Correct results (no contrast) : ",reduce(lambda x, y : int(x) +int(y), results )

metalearner.train(True, lr)
results=metalearner.predict()
print "Correct results (with contrast) : ", reduce(lambda x, y : int(x) +int(y), results )
metalearner.pca_dataset_errors_contrast()
