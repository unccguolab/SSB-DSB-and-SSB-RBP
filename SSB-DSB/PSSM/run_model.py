import re,os,sys
import pickle
import numpy as np
from sklearn import svm
from sklearn import neural_network
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

path=sys.argv[1]
models=sys.argv[2]

##Sigmoid function
def sigmoid(x):
    return(1.0/(1+np.exp(-x)))

##Reading PSSM
def read_PSSM(path):
    total_x={}
    for filename in os.listdir(path):
        if os.path.splitext(filename)[1]=='.pssm':
            lines=[]
            fp=open(path+filename,'r')
            for i in fp:
                i=re.sub('\n','',i)
                lines.append(i)
            fp.close()
            del lines[0:3]
            each=[]
            for x in lines:
                y=x.split()
                if len(y)==44:
                    tem=[]
                    for a in y[2:22]:
                        tem.append(sigmoid(int(a)))
                    each.append(tem)
            total_x[filename]=each
    return(total_x)

##Combine
def Combine(pssm):
    max_len=1500*20
    total={}
    for i in pssm.keys():
        each=[]
        pssm_list=pssm[i]
        for a in range(len(pssm_list)):
            each+=pssm_list[a]
        while len(each)<=max_len:
            pad=[0.0]*20
            each+=pad
        total[i]=each
    return(total)

##Building datasets
def datasets(path):
    data=read_PSSM(path)
    combine_datasets=Combine(data)
    return(combine_datasets)

##Predict
data=datasets(path)
with open(models,'rb') as f:
    clf=pickle.load(f)
for i in data.keys():
    prediction=clf.predict([data[i]])
    if prediction==0:
        out='SSB'
    else:
        out='DSB'
    print(i+'\t'+out)
