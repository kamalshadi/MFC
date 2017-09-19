# This code is to generate results of figure TVRegDiff in the report
# The figure is used to motivate regularized derivation

from hcSynthesis import logRand
import numpy as num
import pylab as pl
from TVRegDiff import *
import pickle as pk

n1 = 5 # cluster at node 5
n2 = 14 # cluster at node 14
n = range(20) # walk path length
	
ask = raw_input('new>')
if ask=='y':
	n1 = 5 # cluster at node 5
	n2 = 14 # cluster at node 14
	n = range(20) # walk path length
	V1 = 16
	V2 = 10
	V3 = 3
	Z = [0.0]*len(n)
	for i in n:
		print i
		if i < n1:
			Z[i] = num.mean(logRand(1000)) * V1;
		elif i < n2:
			Z[i] = num.mean(logRand(1000)) * V2;
		else:
			Z[i] = num.mean(logRand(1000)) * V3;
			
	with open('tmp.pk','w') as f:
		pk.dump(Z,f)
else:
	with open('tmp.pk','r') as f:
		Z = pk.load(f)

deriv_sm = TVRegDiff(Z, 100, 5e-2, dx=0.05, ep=1e-1, scale='small', plotflag=0)
deriv_sm = deriv_sm[0:len(n)]
Z[10]=14.3
Z[11] = 8.1
Z[7]=11.3
Z[15]= 6.5
Z[16]= 2.5
ndf = num.diff(Z)
ndf = [ndf[0]]+ list(ndf)
fig, ax = pl.subplots(1)
axx = pl.twinx()
ax.plot(Z,'r-*',lw = 2, label = r'$\Delta density$')
axx.plot(ndf,'g--d',lw = 2,label = 'Ordinary differentiation')
axx.plot(deriv_sm,'b-o',lw = 2,label = 'Regularized differentiation')
ax.set_xlabel('Hop number', fontsize = 15)
ax.set_ylabel(r'$\Delta$density'+' in logscale', fontsize = 15)
axx.set_ylabel('Derivative of '+r'$\Delta density$', fontsize = 15)
axx.legend(bbox_to_anchor=(1, 1.22), fancybox=True, framealpha=0.5)
ax.legend(bbox_to_anchor=(0.35, 1.16), fancybox=True, framealpha=0.5)
ax.axvspan(0, n1-1, facecolor='r', alpha=0.3)
ax.axvspan(n1-1, n2-1, facecolor='g', alpha=0.3)
#ax.axvspan(n2-1, len(n), facecolor='b', alpha=0.1)
ax.text(1,7,'Cluster 1', fontsize = 15)
ax.text(7,7,'Cluster 2', fontsize = 15)
print len(deriv_sm)
print len(ndf)
print len(Z)
print len(n)
ax.set_xlim([0,19])
axx.set_xlim([0,19])
ax.set_xticks([n1-1,n2-1])
pl.show()
	
