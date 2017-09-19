'''
Hierarchical IP flow clustering

usage:
 -> MFC(input_file)

Requirement:
Python 2.7

Url:
https://github.com/kamalshadi/MFC

Author:
Kamal Shadi

Contact:
kshadi3@gatech.edu

Date:
2017/09/19

Revision:
1.0


Copyright
Networking and Telecommunications Group - Georgia Institute of Technology
'''

from utils import *
import pickle as pk
from ipclass import *
import pickle as pk
import pylab as pl
from myKDtree import *
from dataBackend import *
import networkx as nx
import heapq as hq
import time
from os import listdir
from os.path import isfile, join
import sys


##########################################################

## Set Clustering parameters

K = 10  #number of neighbors
l_run = 4  #drop length
delta = 10  #drop threshold

##########################################################

class MFD:
	def __init__(self, D, df, fname, K=3):
		self.D = D  #raw data
		self.df = df # Dataframe
		self.fName = fname # Save results
		self.K = K # KNN parametrs
		self.N = len(df)
		self.heap = []
		self.KNNG = nx.DiGraph()
		self.KD = None
		self.CD = {}
		self.mfT = tree()

	def initialize(self):
		# Build kdtree
		self.KD = create({(1.0*xx[0],1.0*xx[1]):REC(xx[0],xx[1],xx[2]) for xx in self.D})
		# Build K-NNG and Centroid Dic and update heap
		for w in self.KD.inorder():
			self.mfT.add_leaf(w.value.src1,w.value.dst1,w.value)
			self.KNNG.add_node(w.value, inc = set([]))
			self.CD[w.value.centroid] = 1
		for w in self.KD.inorder():
			tmp = w.value
			A = self.KD.search_knn(tmp.centroid,self.K)
			for a in A[1:]:
				self.KNNG.add_edge(tmp,a[0].value)
				mer = tmp + a[0].value
				if tmp.density()>a[0].value.density():
					mer.update(self.df)
					hq.heappush(self.heap,(-mer.density(),tmp,a[0].value))
				else:
					hq.heappush(self.heap,(1.0,tmp,a[0].value))
				self.KNNG.node[a[0].value]['inc'].add(tmp)


	def MFD_tree(self):
		l = self.KNNG.order()
		while l > 1:
			cur = hq.heappop(self.heap)
			n1 = cur[1]
			n2 = cur[2]
			try:
				self.KNNG[n1][n2]
			except:
				continue

			mer = n1 + n2
			mer.update(self.df)

			######## Update procedure ############

			affected_by_deletion = set([])


			# find node in the merge rectangle > initilize affected node

			dis = (n1.centroid[0]-n2.centroid[0])**2
			dis = dis + (n1.centroid[1]-n2.centroid[1])**2 + 1
			A = self.KD.search_nn_dist(n1.centroid,dis)
			n_to_delete = set([n1,n2])
			for w in A:
				if w.value in mer:
					n_to_delete.add(w.value)

			# delete nodes from KD and KNNG

			for w in n_to_delete:
				w.set_mom(mer)
				mer.add_child(w)
				for nei in self.KNNG.neighbors(w):
					self.KNNG.node[nei]['inc'].remove(w)
				del self.CD[w.centroid]
				self.KD = self.KD.remove(w.centroid)
				affected_by_deletion =\
				affected_by_deletion.union(self.KNNG.node[w]['inc'])
				self.KNNG.remove_node(w)

			affected_by_deletion = affected_by_deletion.difference(n_to_delete)


			# check insertion centroid is unique
			while True:
				try:
					self.CD[mer.centroid]
					a = mer.centroid[0] + num.random.rand()
					b = mer.centroid[1] + num.random.rand()
					mer.centroid = (a,b)
				except KeyError:
					break

			# add node to KD and KNNG and CD
			self.CD[mer.centroid] = 1
			self.KD.add(mer)
			self.KNNG.add_node(mer, inc =set([]))

			A = self.KD.search_knn(mer.centroid,self.K)
			for a in A[1:]:
				self.KNNG.add_edge(mer,a[0].value)
				T = mer + a[0].value
				if mer.density()>a[0].value.density():
					T.update(self.df)
					hq.heappush(self.heap,(-T.density(),mer,a[0].value))
				else:
					hq.heappush(self.heap,(1.0,mer,a[0].value))
				self.KNNG.node[a[0].value]['inc'].add(mer)

			# Augment the affected nodes
			A = self.KD.search_knn(mer.centroid,2*self.K)
			aug = [xx[0].value for xx in A[1:]]

			affected = set(aug).union(affected_by_deletion)

			for w in affected:
				for a in self.KNNG.neighbors(w):
					self.KNNG.remove_edge(w,a)
					try:
						self.KNNG.node[a]['inc'].remove(w)
					except KeyError:
						pass
				A = self.KD.search_knn(w.centroid,self.K)
				for a in A[1:]:
					self.KNNG.add_edge(w,a[0].value)
					T = w + a[0].value
					if mer.density()>a[0].value.density():
						T.update(self.df)
						hq.heappush(self.heap,(-T.density(),w,a[0].value))
					else:
						hq.heappush(self.heap,(1.0,w,a[0].value))
					self.KNNG.node[a[0].value]['inc'].add(w)

			l = self.KNNG.order()
			self.K = min(l,self.K)
		self.mfT.set_root(self.KNNG.nodes()[0])
		with open(self.fName+'.pk','w') as f:
			pk.dump(self.mfT,f)


mypath = '/Users/kshadi/Documents/Cisco_kamal/zDATA3/OUT/'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
sys.setrecursionlimit(100000)
print 'recursion limit'
print sys.getrecursionlimit()


def MFC(fname):
	with open(fname) as f:
		D = pk.load(f)
	l = len(D)
	q = 1
	tot = 0
	for w in D.keys():
		print str(q)+'/'+str(l)
		q = q+1
		print 'clustering for:'
		print w
		st = '_C_' + str(w[0]).replace('.','-').replace('/','s') + '_' + str(w[1]).replace('.','-').replace('/','s')
		if st+'.pk.pk' in onlyfiles:
			continue
		v = D[w]
		print 'Macroflows:'
		print len(v)
		s_t = time.time()
		s = mySparse(v)
		m = MFD(v,s,'OUT/'+st+'.pk')
		m.initialize()
		m.MFD_tree()
		print 'Time:'
		print (time.time() - s_t)
		raw_input('------>')
		tot = tot + (time.time() - s_t);
		print '----------'
	print 'total time:'
	print tot
	
if __name__ == "__main__":
	MFC('2016/11/16/sample.pk')
