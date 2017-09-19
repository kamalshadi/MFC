# What is MFC? #

MFC is a hierarchical clustering algorithm for network flow data that can summarize terabytes of IP traffic
into a parsimonious tree model - each node of the tree is a dense traffic region.
More details in [MFC](http://dl.acm.org/citation.cfm?id=3098598).

> Shadi, Kamal, Preethi Natarajan, and Constantine Dovrolis. "Hierarchical IP flow clustering." Proceedings of the Workshop on Big Data Analytics and Machine Learning for Data Communication Networks. ACM, 2017.

# Package Requirements #

MFC runs in python 2.7 environment. You should have following python libraries in the your interpreter
1. Networkx
2. ipclass
3. sklearn
4. pylab

# Input Data Format

As recommended in the paper, It's a good practice to feed MFC IP traffics in /16x/16 blocks. All the blocks, should be hashed in a python dictionary. For each block X use following formatting:

```
$ D[X] = [(traffic conversation 1),(traffic conversation 2),...,(traffic conversation N)]
```
Each traffic conversation is a three-dimensional vector:
```
(source IP, Destination IP, Volume)
```
# Output Data Format
Each input block creates a single output file in the OUT directory. The file for block X (named after the block key string) contains the IPRECs of the clusters in the block. You can load these files with pickle:
```
with open(<filename>) as f:
    IPREC = pickle.load(f)
```
The IPREC class has following attributes:
+ vol       -> volume of the IPREC
+ density() -> density of the IPREC
+ np        -> Number of conversations in the IPREC
+ src1      -> first source IP
+ src2      -> last source IP
+ dst1      -> first destionation IP
+ dst2      -> last destination IP

# Usage

All discussed parameters in the paper are set in the header of MFC.py file.
The default parameters worked best for our dataset. You can fine tune the parameters based on your need and the run the MFC in python shell using:

```
MFC(D)
```
Where D is the pickle file of the input dictionary.

# Question
Please direct all your questions and kindly report bugs to me at kshadi3@gatech.edu
