# -*- coding: utf-8 -*-
"""
Created on Thu May 14 13:50:21 2020

@author: yesillim
"""
import numpy as np
from termcolor import colored
import matplotlib.pyplot as plt
import sys, os
from ML.feature_functions import PLandscape,F_Landscape,PLandscapes,KernelMethod,F_Image, F_CCoordinates,F_PSignature
from ML.Base import LandscapesParameterBucket
a=[]
# simple persistence diagram
a.append(np.array([(1,5),(1,4),(2,14),(3,4),(4,8.1),(6,7),(7,8.5),(9,12)]))
a.append(np.array([(1.2,3.6),(2,2.6),(2,7),(3.7,4),(5,7.3),(5.5,7),(9,11),(9,12),(12,17)]))

#%% Persistence landscapes

# Landscape computation function examples. 
# out1=PLandscapes(a[0])
# out2=PLandscapes(a[0],[2,3])

# 1st Example ----------------------------------------
PLC  = PLandscape(a[0])
number = PLC .PL_number
points  = PLC .AllPL
des = PLC .DesPL
print(number)
print(points)
print(des)
fig = PLC.PLandscape_plot(PLC.AllPL['Points'])

# fig.savefig('All_Landscapes.png',bbox_inches = 'tight', dpi=300)

#2nd Example ----------------------------------------
PLC  = PLandscape(a[0],[2,3])
print(PLC .PL_number)
print(PLC .AllPL)
print(PLC .DesPL)
fig = PLC.PLandscape_plot(PLC.AllPL['Points'],[2,3])
fig.show()
# fig.savefig('Des_Landscapes.png',bbox_inches = 'tight', dpi=300)

# 3rd example ---------------------------------------
from sklearn.svm import LinearSVC,NuSVC,SVC
params = LandscapesParameterBucket()
params.clf_model = LinearSVC
params.test_size =0.5
params.Labels = None
params.PL_Number = [2]
print(params)


PerLand=np.ndarray(shape=(2),dtype=object)
for i in range(0, 2):
    Land=PLandscape(a[i])
    PerLand[i]=Land.AllPL

labels = [0,1]

feature, Sorted_mesh = F_Landscape(PerLand,params)


#%% Persistence images
TF_Learning = False
D_Img=[]
plot=False
feature_PI = F_Image(a,0.1,0.15,plot,TF_Learning, D_Img)

feature = F_Image(a,0.01,0.15,plot,TF_Learning, D_Img)

# fig[1].savefig('PI_Example_2.png',bbox_inches = 'tight', dpi=300)


#%% Carlsson Coordinates
FN=3
PD=a
FeatureMatrix,TotalNumComb,CombList=F_CCoordinates(PD,FN)

#%% path signatures

PerLand=np.ndarray(shape=(2),dtype=object)
for i in range(0, 2):
    Land=PLandscape(a[i])
    PerLand[i]=Land.AllPL

L_number = [2]

feature_PS = F_PSignature(PerLand,L_number)

#%% kernel method 
perDgm1 = a[0]
perDgm2 = a[1]
sigma=0.25
kernel = KernelMethod(perDgm1, perDgm2, sigma)
print(kernel)
