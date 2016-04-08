#!/usr/bin/env python
# -*- coding: utf-8 -*-

from scipy.sparse import csr_matrix, csc_matrix
import numpy as np

class MatrixClustered:
	"""
		This class allows to define a matrix which rows (objects) were clustered
		Labels can be used to describe rows (objects) and columns (features)
    	"""

	def __init__(self, matrix, clustering, labels_row=[], labels_col=[]):
		"""
		Matrix and clustering should be at least passed.
		Matrix should be a 2D Array or already a sparse (csr or csc) matrix.
		Clustering should be an array where a value v at index i defines that the object i (row i in matrix) belongs to cluster v
		Labels can be used to describe rows (objects) and columns (features) in the same way as the clustering object
    		"""

		self.matrix_csr = csr_matrix(matrix)

		self.matrix_csc = csc_matrix(matrix)

		self.clustering=clustering
		
		self.clusters=[]
		for idx,elmt in enumerate(self.clustering):
			elmt=int(elmt)
			taille=(len(self.clusters) -1) 
			if elmt >= taille:
				for i in range(elmt - taille):
					self.clusters.append([])
			self.clusters[elmt].append(idx)

		self.labels_row=labels_row

		self.labels_col=labels_col

		self.sum_rows=self.matrix_csr.sum(axis=1)

		self.sum_cols=self.matrix_csc.sum(axis=0)
		
		self.ffmean=np.empty(self.matrix_csr.shape[1])
		self.ffmean.fill(-1.0)
		

		

	def sum_row(self, i):
		"""
		Get the sum of row i
    		"""
		return self.sum_rows[i]

	def sum_col(self, j):
		"""
		Get the sum of column j
		Used in Feature Precision (Predominance)
    		"""
		return self.sum_cols[:,j]
	
	def sum_col_of_cluster(self, j, k):
		"""
		Get the sum of column j
		Used in Feature Precision (Predominance)
    		"""
		column=self.matrix_csc.getcol(j).toarray();
		sum=0
		for idx,elmt in enumerate(column):
			if (self.clustering[idx] == k):
				sum+=elmt
		return sum	
	
	def sum_cluster(self, i):
		"""
		Get the sum of cluster i
		Used in feature recall
    	"""
		cluster=self.clusters[i]
		sum=0
		for row in cluster:
			sum+=self.sum_row(row)
		return sum
	
	def fp(self, j, k):
		"""
		Get the feature precision (or predominance) of feature j in cluster k
    	"""
		numerator=self.sum_col_of_cluster(j, k)
		denominator =self.sum_cluster(k)
		if denominator == 0:
			return 0
		else:
			return numerator / denominator
		
	def fr(self, j, k):
		"""
		Get the feature recall of feature j in cluster k
    	"""
		numerator=self.sum_col_of_cluster(j, k)
		denominator =self.sum_col(j)
		if denominator == 0:
			return 0
		else:
			return numerator / denominator
	
	def ff(self, j, k):
		"""
		Get the feature f measure of feature j in cluster k
    	"""
		fr=self.fr(j,k)
		fp=self.fp(j,k)
		if fr == 0 and fp == 0:
			return 0
		else:
			return (2*fr*fp) /(fr + fp) 
		
	def ff_mean(self, j):
		"""
		Get the mean value of feature f-measure for feature j across all clusters
    	"""
		mean=0
		for k in range(len(self.clusters)):
			mean+=self.ff(j,k)
		self.ffmean[j]=mean / len(self.clusters)
		return self.ffmean[j]
	
	def contrast(self, j,k):
		"""
		Get the contrast of feature j in cluster k
    	"""
		return self.ff(j, k) / self.ff_mean(j)
	
	def ff_mean_all(self):
		"""
		Get the mean value of feature f-measure for all features
    	"""
		mean=0
		for j in range(self.get_cols_number()):
			if (self.ffmean[j] == -1):
				self.ff_mean(j)
			mean+=self.ffmean[j]
		return mean / self.get_cols_number()
			
		
	def get_row_label(self, i):
		"""
		Get the label of row i
    	"""
		if len(self.labels_row) == len(self.clustering):
			return self.labels_row[i]
		else:
			return i+""
		
	def get_col_label(self, j):
		"""
		Get the label of col j
    	"""
		if len(self.labels_col) > 0:
			return self.labels_col[j]
		else:
			return j+""
	
	def get_rows_number(self):
		"""
		Get the number of rows
    	"""
		return self.matrix_csr.shape[0]
	
	def get_cols_number(self):
		"""
		Get the number of cols
    	"""
		return self.matrix_csr.shape[1]

	def __str__(self):
		"""
		toString()
    		"""
		return "Matrix CSR (ordered by rows) :\n" + str(self.matrix_csr)+ "\nMatrix CSC (ordered by columns): \n"+ str(self.matrix_csc) + "\nColumns labels (features) " + str(self.labels_col) + "\nRows labels (objects) " + str(self.labels_row) + "\nClustering :  " + str(self.clustering)+"\nClusters : "+str(self.clusters)
		
