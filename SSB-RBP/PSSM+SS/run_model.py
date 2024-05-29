import re,os,sys
import pickle
import numpy as np
from sklearn import svm
from sklearn import neural_network
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

pssm_path=sys.argv[1]
ss_path=sys.argv[2]
models=sys.argv[3]

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
            total_x[filename[:-8]]=each
    return(total_x)

##Reading SS
def read_SS(path):
    total_x={}
    for filename in os.listdir(path):
        if os.path.splitext(filename)[1]=='.ss3':
            lines=[]
            fp=open(path+filename,'r')
            for i in fp:
                i=re.sub('\n','',i)
                lines.append(i)
            fp.close()
            del lines[0:2]
            each=[]
            for x in lines:
                y=x.split()
                each.append([float(y[3]),float(y[4]),float(y[5])])
            total_x[filename[:-4]]=each
    return(total_x)

##Combine
def Combine(pssm,ss):
    max_len=1500*23
    total={}
    for i in pssm.keys():
        each=[]
        pssm_list=pssm[i]
        ss_list=ss[i]
        for a in range(len(pssm_list)):
            each+=pssm_list[a]
            each+=ss_list[a]
        while len(each)<=max_len:
            pad=[0.0]*23
            each+=pad
        total[i]=each
    return(total)

##Building datasets
def datasets(pssm_path,ss_path):
    pssm=read_PSSM(pssm_path)
    ss=read_SS(ss_path)
    combine_datasets=Combine(pssm,ss)
    return(combine_datasets)

##Predict
data=datasets(pssm_path,ss_path)
with open(models,'rb') as f:
    clf=pickle.load(f)
for i in data.keys():
    prediction=clf.predict([data[i]])
    if prediction==0:
        out='SSB'
    else:
        out='RBP'
    print(i+'\t'+out)
