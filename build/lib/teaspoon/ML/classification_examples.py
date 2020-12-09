"""

This file provides examples for classification using featurization methods 
available in machine learning moduel of teaspoon.

"""
#%%  -------------- Persistence Landscapes -------------------------------
import numpy as np
import teaspoon.ML.feature_functions as Ff
import teaspoon.ML.PD_Classification as PD_CL

from teaspoon.ML.Base import LandscapesParameterBucket
from teaspoon.MakeData.PointCloud import testSetManifolds
from sklearn.svm import SVC

# generate persistence diagrams
df = testSetManifolds(numDgms = 10, numPts = 100) 
Diagrams_H1= df['Dgm1'].sort_index().values
Labels = df['trainingLabel'].sort_index().values

#parameters for classification
params = LandscapesParameterBucket()
params.clf_model = SVC
params.test_size = 0.33
params.Labels = Labels
params.PL_Number = [1,2,3,4,5,6,7,8]
print(params)

# Compute the persistence landscapes
PerLand=np.ndarray(shape=(60),dtype=object)
for i in range(0, 60):
    Land=Ff.PLandscape(Diagrams_H1[i])
    PerLand[i]=Land.AllPL

# Perform classification
result = PD_CL.CL_PL(PerLand,params)
#%%  -------------- Persistence Images -------------------------------
import numpy as np
import teaspoon.ML.feature_functions as Ff
import teaspoon.ML.PD_Classification as CL_PD

from teaspoon.ML.Base import CL_ParameterBucket
from teaspoon.MakeData.PointCloud import testSetManifolds
from sklearn.svm import SVC

# generate two sets of persistence diagrams
df_1 = testSetManifolds(numDgms = 10, numPts = 100)
df_2 = testSetManifolds(numDgms = 10, numPts = 100)
Diagrams_H1_1= df_1['Dgm1'].sort_index().values
Diagrams_H1_2= df_2['Dgm1'].sort_index().values

Labels_1 = df_1['trainingLabel'].sort_index().values
Labels_2 = df_2['trainingLabel'].sort_index().values

TF_Learning = False
plot = False
D_Img=[]

params = CL_ParameterBucket()
params.clf_model = SVC
params.test_size =0.33
params.Labels = Labels_1
params.TF_Learning  = False
print(params)


F_Images =Ff.F_Image(Diagrams_H1_1,0.1,0.10,plot,TF_Learning,D_Img)
results = CL_PD.CL_PI(F_Images['F_Matrix'],params)

# classification using transfer learning
# compute the feature matrix for second set of persistence diagrams
TF_Learning =True
F_Images_2 = Ff.F_Image(Diagrams_H1_1,0.1,0.10,plot,TF_Learning,D_Img,Diagrams_H1_2)

params = CL_ParameterBucket()
params.clf_model = SVC
params.test_size =0.33
params.training_labels = Labels_1
params.test_labels = Labels_2
params.TF_Learning  = True
print(params)

results = CL_PD.CL_PI(F_Images_2['F_train'],params,F_Images_2['F_test'])
#%%  -------------- Carlsson Coordinates -------------------------------

import numpy as np
import teaspoon.ML.feature_functions as Ff
import teaspoon.ML.PD_Classification as CL_PD

from teaspoon.ML.Base import CL_ParameterBucket
from teaspoon.MakeData.PointCloud import testSetManifolds
from sklearn.svm import SVC

# generate two sets of persistence diagrams
df_1 = testSetManifolds(numDgms = 10, numPts = 100)
df_2 = testSetManifolds(numDgms = 10, numPts = 100)
Diagrams_H1_1= df_1['Dgm1'].sort_index().values
Diagrams_H1_2= df_2['Dgm1'].sort_index().values
# labels
Labels_1 = df_1['trainingLabel'].sort_index().values
Labels_2 = df_2['trainingLabel'].sort_index().values

# parameters used in classification without transfer learning
params = CL_ParameterBucket()
params.clf_model = SVC
params.test_size =0.33
params.Labels = Labels_1
params.TF_Learning =False
params.FN = 5
print(params)
# classification on one persistence diagram set
results = CL_PD.CL_CC(Diagrams_H1_1,params)

# parameters used in classification with transfer learning
params = CL_ParameterBucket()
params.clf_model = SVC
params.test_size =0.33
params.training_labels = Labels_1
params.test_labels = Labels_2
params.TF_Learning =True
params.FN = 5
print(params)

results = CL_PD.CL_CC(Diagrams_H1_1,params,Diagrams_H1_2)

#%%  -------------- Path Signatures -------------------------------

import numpy as np
import teaspoon.ML.feature_functions as Ff
import teaspoon.ML.PD_Classification as CL_PD

from teaspoon.ML.Base import CL_ParameterBucket
from teaspoon.MakeData.PointCloud import testSetManifolds
from sklearn.svm import SVC

# generate two sets of persistence diagrams
df_1 = testSetManifolds(numDgms = 2, numPts = 100)
df_2 = testSetManifolds(numDgms = 2, numPts = 100)
Diagrams_H1_1= df_1['Dgm1'].sort_index().values
Diagrams_H1_2= df_2['Dgm1'].sort_index().values
# labels
Labels_1 = df_1['trainingLabel'].sort_index().values
Labels_2 = df_2['trainingLabel'].sort_index().values

#compute persistence landscapes for both sets of persistence diagram
PerLand1=np.ndarray(shape=(12),dtype=object)
PerLand2=np.ndarray(shape=(12),dtype=object)

for i in range(0, 12):
    Land=Ff.PLandscape(Diagrams_H1_1[i])
    PerLand1[i]=Land.AllPL
    Land=Ff.PLandscape(Diagrams_H1_2[i])
    PerLand2[i]=Land.AllPL

# compute features using first landscapes
features1 = Ff.F_PSignature(PerLand1,L_Number=[1])
features2 = Ff.F_PSignature(PerLand2,L_Number=[1])
# traditional classification
# adjust parameters
params = CL_ParameterBucket()
params.clf_model = SVC
params.test_size =0.33
params.Labels = Labels_1
params.TF_Learning = False
print(params)

results = CL_PD.CL_PS(features1,params)

#transfer learning
params = CL_ParameterBucket()
params.clf_model = SVC
params.test_size =0.33
params.training_labels = Labels_1
params.test_labels = Labels_2
params.TF_Learning = True
print(params)

results = CL_PD.CL_PS(features1,params,features2)
#%%  -------------- Kernel Method  -------------------------------

import numpy as np
import teaspoon.ML.feature_functions as Ff
from teaspoon.ML.PD_Classification import CL_KM
from teaspoon.ML.Base import CL_ParameterBucket
from teaspoon.MakeData.PointCloud import testSetManifolds

# generate two sets of persistence diagrams
df_1 = testSetManifolds(numDgms = 5, numPts = 100)
Diagrams_H1_1= df_1['Dgm1'].sort_index().values
Labels_1 = df_1['trainingLabel'].sort_index().values

#convert string labels into integers ones 
Labels_ = np.zeros((len(Diagrams_H1_1)))
for i in range(len(Diagrams_H1_1)):
    if Labels_1[i]=='Torus':
        Labels_[i]=0
    elif Labels_1[i]=='Annulus':
        Labels_[i]=1
    elif Labels_1[i]=='Cube':
        Labels_[i]=2   
    elif Labels_1[i]=='3Cluster':
        Labels_[i]=3     
    elif Labels_1[i]=='3Clusters of 3Clusters':  
        Labels_[i]=4           
    elif Labels_1[i]=='Sphere':  
        Labels_[i]=5 

params = CL_ParameterBucket()
params.test_size =0.33
params.Labels = Labels_
params.sigma = 0.25
results = CL_KM(Diagrams_H1_1,params)


#%%  -------------- Template Functions -------------------------------

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



