import pandas as pd
import numpy as num
import copy
# This is IP rectangle class (2D)
# Only "euclidean" and "chebyshev" metric is supported
class REC:
	def __init__(self, *args, **dic):
		l = len(args)
		if l == 2:
			self.src1 = args[0]
			self.src2 = args[0]
			self.dst1 = args[1]
			self.dst2 = args[1]
			self.vol = 0.0
			self.np = 0
		elif l == 3:
			self.src1 = args[0]
			self.src2 = args[0]
			self.dst1 = args[1]
			self.dst2 = args[1]
			self.vol = args[2]
			self.np = 1
		elif l >= 4:
			self.src1 = args[0]
			self.src2 = args[1]
			self.dst1 = args[2]
			self.dst2 = args[3]
			if l == 4:
				self.vol = 0.0
				self.np = 0
			elif l == 5:
				self.vol = args[4]
				self.np = 1
			else:
				self.vol = args[4]
				self.np = args[5]
		try:
			self.center = dic['center']
		except KeyError:
			self.center = ((self.src1+self.src2)/2.0,(self.dst1+self.dst2)/2.0)
		try:
			self.centroid = dic['vcentroid']
		except KeyError:
			self.centroid = self.center
		try:
			self.vcentroid = dic['pcentroid']
		except KeyError:
			self.vcentroid = self.center
		try:
			self.vcentroid = dic['pcentroid']
		except KeyError:
			self.vcentroid = self.center
		self.ID = -1
		self.level = 0
		self.opt = -1.0  # optimization metric for HCX algorithm
		self.mom = None
		self.children = []
		self.den = -1.0
		self.rho = -1.0

	def expand(self,*args):
		l = len(args)
		if l == 1:
			dz = args[0]
			self.src1 = self.src1 - dz
			self.src2 = self.src2 + dz
			self.dst1 = self.dst1 - dz
			self.dst2 = self.dst2 + dz
			
			
			
	def __contains__(self, other):
		if isinstance(other,REC):
			if self.src1 <= other.src1 and self.src2 >= other.src2:
				if self.dst1 <= other.dst1 and self.dst2 >= other.dst2:
					return True
		else:
			if other[0]>=self.src1 and other[0]<=self.src2:
				if other[1]>=self.dst1 and other[1]<=self.dst2:
					return True
		return False
		
	def __str__(self):
		a1 = str(self.src1)
		a2 = str(self.src2)
		b1 = str(self.dst1)
		b2 = str(self.dst2)
		q = 'REC('+a1+'>'+b1+','+a2+'>'+b2+'|'+str(self.vol)+'/'+str(self.np)+')'
		return q
		
	def __hash__(self):
		return hash(('rec', self.src1, self.src2, self.dst1, self.dst2))
		
	def __eq__(self,other):
		if self.src1 == other.src1 and self.dst1==other.dst1:
			if self.src2 == other.src2 and self.dst2==other.dst2:
				return True
		return False
	
	def __neq__(self,other):
		return not (self == other)

	def add_child(self, x):
		self.children.append(x)
		
	def set_mom(self, x):
		self.mom = x
		
	def intersect(self, other, df = None):
		x1 = max(self.src1, other.src1)
		x2 = min(self.src2,other.src2)
		y1 = max(self.dst1,other.dst1)
		y2 = min(self.dst2,other.dst2)
		if (x1 <= x2 and y1 <= y2):
			R = REC(x1,x2,y1,y2)
			if df is None:
				return R
			R.update(df)
			return R

	def vertices(self):
		i = 0
		while i<4:
			if i == 0:
				yield (self.src1,self.dst1)
			if i == 1:
				yield (self.src2,self.dst1)
			if i == 2:
				yield (self.src1,self.dst2)
			if i == 3:
				yield (self.src2,self.dst2)
			i = i + 1
	
	def edges(self):
		i = 0
		while i<4:
			if i == 0:
				yield ('v',self.src1,self.dst1,self.dst2)
			if i == 1:
				yield ('v',self.src2,self.dst1,self.dst2)
			if i == 2:
				yield ('h',self.dst1,self.src1,self.src2)
			if i == 3:
				yield ('h',self.dst2,self.src1,self.src2)
			i = i + 1
	
	def distance(self, other, metric = "euclidean", linkage = 'min'):
		if linkage == 'min':
			if self.intersect(other) is not None:
				return 0.0
			dis = float('inf')
			for v in self.vertices():
				for e in other.edges():
					tmp = line2point(v,e,metric)
					if tmp < dis:
						dis = tmp
			for v in other.vertices():
				for e in self.edges():
					tmp = line2point(v,e,metric)
					if tmp < dis:
						dis = tmp
			return float(dis)
		elif linkage == 'centroid':
			p1 = self.centroid
			p2 = other.centroid
		elif linkage =='vcentroid':
			p1 = self.vcentroid
			p2 = other.vcentroid
		else:
			p1 = self.center
			p2 = other.center
		if metric == 'euclidean':
			return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**.5
		else:
			return float(max(abs(p1[0]-p2[0]), abs(p1[1]-p2[1])))
			
		
	def update(self, df):
		df.query(self)
		
	def nf(self):
		return float((self.src2-self.src1+1)*(self.dst2-self.dst1+1))
		
	def diag(self,metric ='euclidean'):
		if metric == 'euclidean':
			a1 = float(self.src2 - self.src1)**2
			a2 = float(self.dst2 - self.dst1)**2
			return (a1+a2)**0.5
		else:
			a1 = (self.src2 - self.src1)
			a2 = (self.dst2 - self.dst1)
			return float(max(a1,a2))
			
	def nh(self):
		return float((self.src2-self.src1+1)+(self.dst2-self.dst1+1))
		
		
	def __add__(self, other, df = None):
		R = REC(0,0)
		R.src1 = min(self.src1,other.src1)
		R.src2 = max(self.src2,other.src2)
		R.dst1 = min(self.dst1,other.dst1)
		R.dst2 = max(self.dst2,other.dst2)
		if df is not None:
			R.update(df)
		return R
		
	def density(self, kind = 'nf', nom = 'vol'):
		if nom == 'vol':
			if kind == 'nf':
				return self.vol/self.nf()
			elif kind == 'nh':
				return self.vol/self.nh()
			elif kind == 'diag':
				return self.vol/self.diag()
			else:
				raise InputError
		else:
			if kind == 'nf':
				return self.np/self.nf()
			elif kind == 'nh':
				return self.np/self.nh()
			elif kind == 'diag':
				return self.np/self.diag()
			else:
				raise InputError
			
	
	# Remove these after debugging
			
class tree:
	def __init__(self):
		self.root = None
		self.leaves = {}
		
	def set_root(self, nn):
		self.root = nn	
	
	def myLeaf(self,w):
		out = self.root
		if w not in out:
			return None
		c = out
		while True:
			fg = False
			for cur in out.children:
				if w in cur:
					fg = True
					break
			if fg:
				out = copy.copy(cur)
			if not fg:
				break
		return out
		
	def add_leaf(self,src,dst,nn):
		self.leaves[(src,dst)] = nn


#~ class tree2:
	#~ def __init__(self):
		#~ self.root = None
		#~ self.leaves = {}
		
	#~ def add_root(self, nn):
		#~ self.root = nn
		
	#~ def add_leaf(self,src,dst,nn):
		#~ self.leaves[(src,dst)] = nn

#~ class HC:
	#~ def __init__(self, nc):
		#~ self.nc = nc
		#~ self.met = {}
		#~ for i in range(nc):
			#~ self.met[i] = []
		#~ self.met[i+1] = []
		#~ self.trees = [tree1()]*nc
		#~ self.comp_tree = None
	
def line2point(v,e, metric):
	if e[0] == 'h':
		if v[0] >= e[3]:
			if metric == "chebyshev":
				return max(abs(v[0]-e[3]),abs(v[1] - e[1]))
			else:
				return ((v[0]-e[3])**2 + (v[1] - e[1])**2)**.5
		elif v[0] <= e[2]:
			if metric == "chebyshev":
				return max(abs(v[0]-e[2]),abs(v[1] - e[1]))
			else:
				return ((v[0]-e[2])**2 + (v[1] - e[1])**2)**.5
		else:
			return abs(v[1] - e[1])
	if e[0] == 'v':
		if v[1] >= e[3]:
			if metric == "chebyshev":
				return max(abs(v[1]-e[3]),abs(v[0] - e[1]))
			else:
				return ((v[1]-e[3])**2 + (v[0] - e[1])**2)**.5
		elif v[0] <= e[2]:
			if metric == "chebyshev":
				return max(abs(v[1]-e[2]),abs(v[0] - e[1]))
			else:
				return ((v[1]-e[2])**2 + (v[0] - e[1])**2)**.5
		else:
			return abs(v[0] - e[1])
			
def tokey(D):
	return '-'.join\
	([str(xx[1]) for xx in sorted(D.items()) if xx[0]!= '_nc'])
	
# n'th hop neighbor of k'th point
def n_hop_neighbor(ind,k,n=1):
    i = 0
    tmp1 = set({k})
    if n == 0:
        return tmp1
    if n == 1:
        tmp = set(ind[k])
        tmp.discard(k)
        return tmp
    dif = set({k})
    while i<n:
        if len(dif) == 0:
            return set({})
        tmp2 = set({})
        for w in dif:
            for ww in ind[w]:
                tmp2.add(ww)
        dif = tmp2 - tmp1
        tmp1 = tmp1 | dif
        i = i + 1
        if i == n:
            dif.discard(k)
    return dif

# return next hop to reach "p" from "k" in "n" hops
def from_to_hop(ind,k,p,n):
    if n == 0:
        return set({k})
    if n == 1:
        return set({p})
    fin = set({})
    for w in n_hop_neighbor(ind,k,1):
        tmp = n_hop_neighbor(ind,w,n-1)
        if p in tmp:
            if p in n_hop_neighbor(ind,k,n):
                fin.add(w)
    return fin
	
	
#~ if __name__ == '__main__':
	#~ df = [(2,4,5.2),(3,3,120.1),(20,0,0.4),(0,450,0.7)]
	#~ d2 = D2(df)
