import sys
import os.path
import pickle
import pandas as pd
import numpy as np
sys.path.insert(0,os.path.join('D:'+os.path.sep,'Research Stuff','libsvm-3.23','python',))
from sklearn.svm import LinearSVC,NuSVC,SVC

from ML.PD_Classification import CL_PL,CL_PI, CL_CC, CL_PS,CL_KM
from ML.Base import LandscapesParameterBucket,CL_ParameterBucket
from ML.feature_functions import F_Image,F_PSignature
from MakeData import PointCloud as gpc

#generate persistence diagrams

#%% load the data

folderToLoad = os.path.join('D:'+os.path.sep,
                                    'Data Archive',
                                    'Persistence Landscapes',
                                    )
sys.path.insert(0,folderToLoad)

folderToLoad2 = os.path.join('D:'+os.path.sep,
                                    'Research Stuff',
                                    'Embedding Dataset',
                                    )
sys.path.insert(0,folderToLoad2)


with open(os.path.join(folderToLoad, "Persistence_Landscapes_2inch_chatter(593Case).txt"), "rb") as fp:
    PL = pickle.load(fp)

Diagrams=pd.read_csv(os.path.join(folderToLoad2, "Persistence_Diagram_Turning_Chatter_2inch(593Case).csv"))

#Compressing 
compressed_pd_train = Diagrams.groupby(['RPM','DOC','Label','Case']).apply(lambda x: np.transpose(np.array([x.birth_time, x.death_time])))
compressed_pd_train = compressed_pd_train.reset_index()
compressed_pd_train = compressed_pd_train.rename(columns={0: "PD"})
PD1 = compressed_pd_train.iloc[:, 4].values
N=len(PD1)
labels1=(compressed_pd_train.iloc[:, 2].values).reshape(N,1)   


Diagrams=pd.read_csv(os.path.join(folderToLoad2, "Persistence_Diagram_Turning_Chatter_4.5inch.csv"))

#Compressing 
compressed_pd_test = Diagrams.groupby(['RPM','DOC','Label','Case']).apply(lambda x: np.transpose(np.array([x.birth_time, x.death_time])))
compressed_pd_test = compressed_pd_test.reset_index()
compressed_pd_test = compressed_pd_test.rename(columns={0: "PD"})
PD2 = compressed_pd_test.iloc[:, 4].values
N=len(PD2)
labels2=(compressed_pd_test.iloc[:, 2].values).reshape(N,1)   

#convert diagrams into data frame
df_train=compressed_pd_train
df_test=compressed_pd_test
dgmColLabel = ['PD']


#%% landscapes
from ML.Base import LandscapesParameterBucket
from ML.PD_Classification import CL_PL
import ML.feature_functions as fF


params = LandscapesParameterBucket()
params.clf_model = SVC
params.test_size =0.33
params.Labels = labels1
params.PL_Number = [1]
print(params)

result = CL_PL(PL,params)

#%% images

from ML.Base import CL_ParameterBucket
from ML.feature_functions import F_Image
from ML.PD_Classification import CL_PI

params = CL_ParameterBucket()
params.clf_model = SVC
params.test_size =0.33
params.Labels = labels1
print(params)
TF_Learning = False
plot = False
D_Img=[]
plot=[]
F_Images = F_Image(PD1,0.1,0.10,plot,TF_Learning,D_Img)
results = CL_PI(F_Images['F_Matrix'],params)

# transfer learning 
params = CL_ParameterBucket()
params.TF_Learning = True
params.clf_model = SVC
params.test_size =0.33
params.training_labels = labels1
params.test_labels = labels2
print(params)
TF_Learning = True
F_Images = F_Image(PD1,0.1,0.10,plot,TF_Learning,D_Img,PD2)
F_train = F_Images['F_train']
F_test = F_Images['F_test']
results = CL_PI(F_Images['F_train'],params,F_Images['F_test'])



#%% carlsson coordinates


import ML.Base
from ML.Base import CL_ParameterBucket
from ML.PD_Classification import CL_CC
## Traditional Classification

params = CL_ParameterBucket()
params.clf_model = SVC
params.test_size =0.33
params.Labels = labels1
params.TF_Learning =False
params.FN = 5
print(params)

results = CL_CC(PD1,params)

## Transfer Learning
params = CL_ParameterBucket()
params.clf_model = SVC
params.test_size =0.33
params.training_labels = labels1
params.test_labels = labels2
params.TF_Learning =True
params.FN = 5
print(params)

results = CL_CC(PD1,params,PD2)


#%% paths signatures
from MakeData import PointCloud as gpc
from ML.Base import CL_ParameterBucket
from ML.PD_Classification import CL_PS
import ML.feature_functions as fF

# generate persistence diagrams
df1 = gpc.testSetManifolds(numDgms = 5, numPts = 30) 
Diagrams1 = df1['Dgm1'].values
Labels1 = df1['trainingLabel']

df2 = gpc.testSetManifolds(numDgms = 5, numPts = 30) 
Diagrams2 = df1['Dgm1'].values
Labels2 = df1['trainingLabel']


# compute landscape for landscapes
PerLand1=np.ndarray(shape=(len(Diagrams1)),dtype=object)
PerLand2=np.ndarray(shape=(len(Diagrams2)),dtype=object)
for i in range(0, len(Diagrams1)):
    Land1=fF.PLandscape(Diagrams1[i])
    PerLand1[i]=Land1.AllPL
    Land2=fF.PLandscape(Diagrams2[i])
    PerLand2[i]=Land2.AllPL

# compute features using first landscapes
features1 = fF.F_PSignature(PerLand1,L_Number=[1])
features2 = fF.F_PSignature(PerLand2,L_Number=[1])

# traditional classification

# adjust parameters
params = CL_ParameterBucket()
params.clf_model = SVC
params.test_size =0.33
params.Labels = Labels1
params.TF_Learning = False
print(params)

#classification
results = CL_PS(features1,params)

#transfer learning 
params = CL_ParameterBucket()
params.clf_model = SVC
params.test_size =0.33
params.training_labels = Labels1
params.test_labels = Labels2
params.TF_Learning = True
print(params)

results = CL_PS(features1,params,features2)

#%% Kernel Method
a=[]
# simple persistence diagram
a.append(np.array([(1,5),(1,4),(2,14),(3,4),(4,8.1),(6,7),(7,8.5),(9,12)]))
a.append(np.array([(1.2,3.6),(2,2.6),(2,7),(3.7,4),(5,7.3),(5.5,7),(9,11),(9,12),(12,17)]))
a.append(np.array([(2,8),(3,4),(5.6,7)]))

labels = [0,1,0]

params = CL_ParameterBucket()
params.test_size =0.33
params.Labels = labels
params.sigma = 0.25
results = CL_KM(a,params)

#%% template functions

# traditional classification
from ML.Base import ParameterBucket
import ML.feature_functions as fF
from ML.PD_Classification  import getPercentScore

params = ParameterBucket()
params.setBoundingBox(df_train[dgmColLabel], pad = .05)
params.jacobi_poly = 'cheb1'  # choose the interpolating polynomial
params.d = 20
params.feature_function = fF.interp_polynomial
params.useAdaptivePart = False
params.clf_model = SVC
params.TF_Learning =False
print(params)

#----------------------------------
# Run the experiment
#----------------------------------
num_runs = 10
accuracy_test_set = np.zeros((num_runs))
accuracy_train_set = np.zeros((num_runs))  
results = np.zeros((1,5))  
for i in np.arange(num_runs):
    xx = getPercentScore(df_train,labels_col = 'Label',dgm_col = dgmColLabel,params = params,normalize = False,verbose = False)
    accuracy_test_set[i] = xx['score']
    accuracy_train_set[i] = xx['score_training']

results[0,1]=np.std(accuracy_test_set)
results[0,3]=np.std(accuracy_train_set)
results[0,0]=np.mean(accuracy_test_set)
results[0,2]=np.mean(accuracy_train_set)

print(results)




# transfer learning
from ML.Base import ParameterBucket
import ML.feature_functions as fF
from ML.PD_Classification  import getPercentScore

params = ParameterBucket()
params.setBoundingBox(df_train[dgmColLabel], pad = .05)
params.jacobi_poly = 'cheb1'  # choose the interpolating polynomial
params.d = 20
params.feature_function = fF.interp_polynomial
params.useAdaptivePart = False
params.TF_Learning =True
params.clf_model = SVC
print(params)

#----------------------------------
# Run the experiment
#----------------------------------
num_runs = 10
accuracy_test_set = np.zeros((num_runs))
accuracy_train_set = np.zeros((num_runs))  
results = np.zeros((1,5))  
for i in np.arange(num_runs):
    xx = getPercentScore(df_train,labels_col = 'Label',dgm_col = dgmColLabel,params = params,normalize = False,verbose = False,DgmsDF_test=df_test)
    accuracy_test_set[i] = xx['score']
    accuracy_train_set[i] = xx['score_training']

results[0,1]=np.std(accuracy_test_set)
results[0,3]=np.std(accuracy_train_set)
results[0,0]=np.mean(accuracy_test_set)
results[0,2]=np.mean(accuracy_train_set)

print(results)



