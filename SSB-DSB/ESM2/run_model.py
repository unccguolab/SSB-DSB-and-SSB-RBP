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

##Dataset
def datasets(path):
    total={}
    for filename in os.listdir(path):
        if os.path.splitext(filename)[1]=='.esm':
            ll=[]
            fp=open(path+filename,'r')
            for i in fp:
                i=re.sub('\n','',i)
                ll.append(float(i))
            fp.close()
            total[filename[:-4]]=ll
    return(total)

##Predict
data=datasets(path)
with open(models,'rb') as f:
    clf=pickle.load(f)
for i in data.keys():
    prediction=clf.predict([data[i]])
    if prediction==1:
        out='SSB'
    else:
        out='DSB'
    print(i+'\t'+out)
