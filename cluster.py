from utils import *
import pickle as pk
from ipclass import *
#import networkx as nx
import pickle as pk
import pylab as pl
from sklearn.neighbors import NearestNeighbors

class cluster:
	def __init__(self, D, df, fname, K=10, metric='euclidean', mode='centroid', gradient = False, kind = 'nf'):
		self.D = D  #raw data
		self.df = df # Dataframe
		self.fName = fname # Save results
		self.K = K # KNN parametrs
		self.metric = metric
		self.mode = mode
		self.N = len(df)
		self.gradient = gradient
		self.kind = kind
		

	def cluster(self):
		gradient = self.gradient
		hc = HC(1)
		my_tree = tree1()
		# Building L
		L = [REC(0.0,0.0)]*self.N
		q = 0
		for w in self.D:
			L[q] = REC(w[0],w[1],w[2])
			my_tree.add_leaf(w[0],w[1],L[q])
			q = q + 1
		l = len(L)
		er = 0
		while l>1:
			print 'length :'+str(l)
			# KNN
			if self.K > l:
				self.K = l
			if self.mode == 'centroid':
				ki = [xx.centroid for xx in L]
			elif self.mode == 'center':
				ki = [xx.center for xx in L]
			elif self.mode == 'vcentroid':
				ki = [xx.vcentroid for xx in L]
			else:
				# here you need to develope minimum linkage algorithm
				pass
			nbrs = NearestNeighbors(n_neighbors=self.K, algorithm='ball_tree'\
			,metric = self.metric).fit(ki)
			indices = nbrs.kneighbors(ki, return_distance = False)
			level = 0
			his = -1.0
			for q in range(l):
				cur = L[q]
				can = n_hop_neighbor(indices,q,1)
				for w in can:
					if L[w].density(self.kind)>cur.density(self.kind):
						continue
					try:
						curr = cur.__add__(L[w],self.df)
					except ZeroDivisionError:
						print cur
						print L[w]
						#raw_input('----')
						er = er + 1
						print '---------'
					if self.kind == 'nf':
						if not gradient:
							den = float(curr.vol)/curr.nf()
						else:
							denom = curr.nf() - cur.nf()
							nom = curr.vol - cur.vol
							try:
								den = nom/denom
							except ZeroDivisionError:
								# Small rectangle within larger one
								print 'density:'
								print cur
								print L[w]
								print curr
								den = float('inf')
								#raw_input('----')
								print '~~~~~~~'
					elif self.kind == 'nh':
						if not gradient:
							den = float(curr.vol)/curr.nh()
						else:
							denom = curr.nh() - cur.nh()
							nom = curr.vol - cur.vol
							den = nom/denom
					elif self.kind == 'diag':
						if not gradient:
							den = float(curr.vol)/curr.diag(self.metric)
						else:
							denom = curr.diag(self.metric) - cur.diag(self.metric)
							nom = curr.vol - cur.vol
							den = nom/denom
					else:
						pass
					if den>his:
						RC = curr
						RC.opt = den
						RC.level = level
						his = den
						fin = (q,w)
				
			if his < 0:
				break
			level = level + 1
			mg = n_hop_neighbor(indices,fin[0],1) |\
			n_hop_neighbor(indices,fin[1],1)
			mg = set(mg)
			mg.discard(fin[1])
			mg.discard(fin[0])
			delme =[fin[0],fin[1]]
			for ck in mg:
				if L[ck] in RC:
					delme.append(ck)

			for th in delme:
				L[th].set_mom(RC)
				RC.add_child(L[th])
			L = [i for j, i in enumerate(L) if j not in delme]
			L.append(RC)
			l = len(L)

		if l > 1:
			rt = L[0]
			for w in L[1:]:
				rt = rt.__add__(w,self.df)
			rt.level = level
			if self.kind == 'nf':
				den = float(rt.vol)/rt.nf()
			elif self.kind == 'nh':
				den = float(rt.vol)/rt.nh()
			elif self.kind == 'diag':
				den = float(rt.vol)/rt.diag(self.metric)
			else:
				pass
			rt.opt = den
			for w in L:
				rt.add_child(w)
				w.set_mom(rt)
		else:
			rt = L[0]
		my_tree.add_root(rt)
		hc.trees[0] = my_tree
		print 'Saving....'
		with open('res/'+self.fName + '.pk','w') as f:
			pk.dump(hc, f)
		print 'Saved.'
		print 'Error:'+str(er)
			
if __name__=='__main__':
	for w in ['A','B','C','D','E','F']:
		fn = 'case1_'+w
		out = read_data(fn)
		#~ #D = D[1:100]
		df = mySparse(out)
		fname = fn+'_tree'
		C = cluster(out, df, fname)
		print C.size()
		raw_input()
		D = cluster(out, df, fname+'_F', gradient = False)
		C.cluster()
		D.cluster()
