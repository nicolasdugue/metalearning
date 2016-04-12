'''
Created on Apr 12, 2016

@author: dugue
'''

from sklearn.datasets import fetch_mldata
import numpy as np

import tempfile
test_data = tempfile.mkdtemp()

wine = fetch_mldata('uci-20070111 wine', data_home=test_data)
classes= wine["target"]
data1=wine['data']
data2=wine["double3"].T
data3=wine["int2"].T
data4=wine["int4"].T
matrix= np.concatenate((data1, data2), axis=1)
matrix=np.concatenate((matrix, data3), axis=1)
matrix=np.concatenate((matrix, data4), axis=1)
print matrix.shape
print classes.shape

