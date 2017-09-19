# my databackend for huge list of 2D IP addresses

from bisect import bisect_left
from utils import *
from ipclass import *
from cluster import *
import sqlite3 as sq

def sliceList(u,s):
	# This function finds a slice
	# of sorted list u between s[0] and s[1]
    i1 = bisect_left(u,s[0])
    i2 = bisect_left(u,s[1])
    return u[i1:i2]

class mySparse:
	# D is a n*3 array, list os lists or tuples
    def __init__(self, D):
		# sort based on destinations
        D = sorted(D, key = lambda x:x[1])
        self.l = len(D)
        quell = set([xx[0] for xx in D])
        self.srcs = sorted(list(quell)) # sorted list of sources
        self.n_srcs = len(self.srcs)
        # self.d_srcs = dict(zip(self.srcs, range(self.n_srcs))) # dictionary of srcs
        ziel = set([xx[1] for xx in D])
        dsts =sorted(list(ziel)) # sorted list of sdestinations
        self.n_dsts = len(dsts)
        #self.d_dsts = dict(zip(dsts, range(self.n_dsts))) # dictionary of srcs
        m = self.srcs[-1] - self.srcs[0] + 1
        n = dsts[-1] - dsts[0] + 1
        self.shape = (m,n)
        self.DD = {}
        self.imp = {}
        
        # making a backend dictionary
        for d in D:
            try:
                self.imp[d[0]] = self.imp[d[0]] + [d[1]]
            except KeyError:
                self.imp[d[0]] = [d[1]]
            self.DD[(d[0],d[1])] = d[2]

    # Slice query             
    def __getitem__(self,key):
        v = []
        if isinstance(key[0],slice):
            loop1 = (key[0].start,key[0].stop+1)
        else:
            loop1 = (key[0],key[0])
        if isinstance(key[1],slice):
            loop2 = (key[1].start,key[1].stop+1)
        else:
            loop2 = (key[1],key[1])
        out = []
        dim1 = sliceList(self.srcs,loop1)
        for src in dim1:
			cans = self.imp[src]
			dim2 = sliceList(cans,loop2)
			for dst in dim2:
				out.append((src,dst))
        return out
        
    def query(self,rec):
		# centroid
		# vcentroid
		# volume
		# density
		# number of points
        vol = 0
        loop1 = (rec.src1, rec.src2+1)
        loop2 = (rec.dst1, rec.dst2+1)
        cen_src = (loop1[0] + loop1[1] - 1.0)/2.0
        cen_dst = (loop2[0] + loop2[1] - 1.0)/2.0
        dim1 = sliceList(self.srcs,loop1)
        np = 0
        out = {}
        
        for src in dim1:
			cans = self.imp[src]
			dim2 = sliceList(cans,loop2)
			for dst in dim2:
				val = self.DD[(src,dst)]
				np = np + 1
				vol = vol + val
				if np == 1:
					pcen_src = src
					pcen_dst = dst
					vcen_src = src
					vcen_dst = dst
				else:
				    pcen_src = (float(np-1)/np)*pcen_src + float(src)/np
				    pcen_dst = (float(np-1)/np)*pcen_dst + float(dst)/np
				    vcen_src = ((vol-val)/vol)*vcen_src + (val/vol)*src
				    vcen_dst = ((vol-val)/vol)*vcen_dst + (val/vol)*dst
        if np == 0:
			return None
        rec.vol = vol
        rec.center = (cen_src,cen_dst)
        rec.centroid = (vcen_src,vcen_dst)
        #rec.pcentroid = (pcen_src,pcen_dst)
        rec.np = np
        return rec
      
    def __len__(self):
		return self.l
		
def returnDataBackend(df, per = .2, nf = 0):
	def Mbyt(w):
		if w[-1] == 'M':
			return float(w[-2])
		elif w[-1] == 'G':
			return float(w[-2])*1000
		else:
			try:
				return float(w[-1])
			except:
				print w
			
	with open(df) as f:
		if nf>0:
			z = [(0.0, 1, 1)]*nf
		ls = f.readlines()
		q = 0
		for i,line in enumerate(ls):
			if i == 0:
				continue
			if ':' in line:
				break
			if nf>0 and q == nf:
				break
			w = line.split()
			tmp = float(w[-1])
			z[q] = (int(w[0]), int(w[1]), tmp)
			q = q + 1
		t = sorted(z[0:q], reverse = True, key = lambda x:x[2])
		if nf == 0:
			l = int(i*per)
			t = t[0:l]
		else:
			t = t[0:q]
	I = mySparse(t)
	#~ dat = '/Users/kshadi/Documents/Cisco_kamal/zDATA3/2016/11/16/t10.csv'
	#~ print 'SQ...'
	#~ save_sqlite(t,dat)
	#~ raw_input('database created>')
	print 'Clustering ...'
	C = cluster(t, I, '_res_1',gradient = False)
	C.cluster()
	
def save_sqlite(t,df):
	conn = sq.connect(df.replace('.csv','.db'))
	c = conn.cursor()
	c.execute('''CREATE TABLE meta
             (src integer, dst integer, cls text, vol real)''')
	l = len(t)
	z = [(1,2,'a',1.0)]*l
	for i,w in enumerate(t):
		src = IP(int2ip(w[1]))
		dst = IP(int2ip(w[2]))
		srcS = src.tosubnet(16)
		dstS = dst.tosubnet(16)
		cls = str(srcS) + '>' + str(dstS)
		vol = w[0]
		z[i] = (w[1],w[2],cls,vol)
	c.executemany('INSERT INTO meta VALUES (?,?,?,?)', z)
	conn.commit()
	conn.close()
		
def subset_df(df,num,cls = "64.101.0.0/16>10.154.0.0/16" ):
	conn = sq.connect(df)
	c = conn.cursor()
	sqq = '''select src,dst,vol from meta where cls="XXX"'''
	sqq = sqq.replace('XXX',cls)
	A = c.execute(sqq)
	with open(df.replace('.db','_'+str(num)+'.csv'),'w') as f:
		f.write('src dst vol\n')
		for a in A:
			tmp = ' '.join([str(xx) for xx in a])+'\n'
			f.write(tmp)
	conn.close()

if __name__ == '__main__':
	dat = '/Users/kshadi/Documents/Cisco_kamal/zDATA3/2016/11/16/t10_1.csv'
	#subset_df(dat,4)
	returnDataBackend(dat, per = .2, nf = 900)
	#~ D = [(10,1,5.0),(7,2,7.0),(10,8,1.0),(11,7,2.0),(13,7,3.0)]
	
	#~ I = mySparse(D)
	#~ R = REC(1,13,0,12)
	#~ R.update(I)
	#~ print R
	#~ print I.query(R)
	#~ print R.pcentroid
	#~ s = [3,2,5,6,4,2,8,3,9]
	#~ s = sorted(s)
	#~ print sliceList(s,(3,8))
	#REC(3219784754>174491921,3219784754>174491921|43.574256/1)
	#REC(3219784754>174024061,3219784754>174024061|40.292848/1)
	#~ r1 = REC(3219784754,174491921,43.1)
	#~ r2 = REC(3219784754,174024061,6.5)
	#~ D = [(3219784754,174024061,6.5),(3219784754,174491921,43.1)]
	#~ D1 = mySparse(D)
	#~ r = r1 + r2
	#~ print r
	#~ print D1.query(r)
