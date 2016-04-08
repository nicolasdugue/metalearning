'''
Created on Apr 8, 2016

@author: nicolas
'''

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pyfmax.fmax as fm

lr = LogisticRegression()

matrix_iris=np.loadtxt("../data/iris/iris.data")
classes=matrix_iris[:,4]
matrix_iris=matrix_iris[:,0:3]

metalearner=fm.MetaLearner(matrix_iris, classes)
metalearner.train(False, lr)
results=metalearner.predict()
print reduce(lambda x, y : int(x) +int(y), results )

metalearner.train(True, lr)
results=metalearner.predict()
print reduce(lambda x, y : int(x) +int(y), results )