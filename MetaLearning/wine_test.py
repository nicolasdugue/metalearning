'''
Created on Apr 12, 2016

@author: dugue
'''

from sklearn.datasets import fetch_mldata
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import pyfmax.fmax as fm

#lr = SVC(probability=True)


import tempfile

test_data = tempfile.mkdtemp()

lr=LogisticRegression()
norm=StandardScaler()

wine = fetch_mldata('uci-20070111 wine', data_home=test_data)
classes= wine["target"] - 1

data1=wine['data']
data2=wine["double3"].T
data3=wine["int2"].T
data4=wine["int4"].T
matrix= np.concatenate((data1, data2), axis=1)
matrix=np.concatenate((matrix, data3), axis=1)
matrix=np.concatenate((matrix, data4), axis=1)
matrix=norm.fit_transform(matrix)

metalearner=fm.MetaLearner(matrix, classes, perct_test=0.5, magnitude=2)
metalearner.train(False, lr)
results=metalearner.predict()

print "Correct results (no contrast) : ",reduce(lambda x, y : int(x) +int(y), results )

#metalearner=fm.MetaLearner(matrix_iris, classes, perct_test=0.2, magnitude=10)
metalearner.train(True, lr)
results=metalearner.predict()
print "Correct results (with contrast) : ", reduce(lambda x, y : int(x) +int(y), results )
print "Features selected : ",metalearner.matrix.get_features_selected_flat()
metalearner.pca_dataset(True, True)



