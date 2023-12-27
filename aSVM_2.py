#%%
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 12:11:43 2023

@author: berej
"""

#from cuml.model_selection import train_test_split
#from cuml.model_selection import GridSearchCV
#import cudf
#import cuml
#cuml.set_global_output_type('numpy')
#from cuml import svm
import pandas as pd
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import os
import glob2
import matplotlib.pyplot as plt
#from skimage.transform import resize
#from skimage.io import imread
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix,roc_curve, RocCurveDisplay
import pickle

from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV

Categories=['negative', 'positive']
size = '10'

flat_data_arr = []
neur_arr = []
target_arr = []
test_data_arr = []
test_target_arr = []
#%%

datadir= os.getcwd()
for i in Categories:
    
    print(f'loading... category : {i}')
    path=os.path.join(datadir,i)
    file = glob2.glob(os.path.join(path,'*'+size+'*npy'))[0]
    data = np.load(file)
    for k in range(len(data)):
        flat_data_arr.append(data[k].flatten())
        target_arr.append(Categories.index(i))
    print(f'loaded train category:{i} successfully')
# =============================================================================
#     for k in range(len(data)):
#         for l in range(len(data[k])):
#             test_data_arr.append(data[k][l])
#             test_target_arr.append(Categories.index(i))
#     print(f'loaded test category:{i} successfully')
# =============================================================================

# =============================================================================
# datadir= os.getcwd()
# for i in Categories:
#     
#     print(f'loading... category : {i}')
#     path=os.path.join(datadir,i)
#     file = glob2.glob(os.path.join(path,'*'+size+'*npy'))[0]
#     data = np.load(file)
#     for k in range(len(data)):
#         for l in range(len(data[k])):
#           flat_data_arr.append(data[k][l].flatten())
#           target_arr.append(Categories.index(i))
#     print(f'loaded train category:{i} successfully')
# # =============================================================================
# #     for k in range(len(data)):
# #         for l in range(len(data[k])):
# #             test_data_arr.append(data[k][l])
# #             test_target_arr.append(Categories.index(i))
# #     print(f'loaded test category:{i} successfully')
# # =============================================================================
# =============================================================================


del data
test_data = np.array(test_data_arr)
del test_data_arr
flat_data=np.array(flat_data_arr)
del flat_data_arr
target=np.array(target_arr)
del target_arr
df=pd.DataFrame(flat_data)
df['Target']=target
#del data



# =============================================================================
# test_data = np.array(test_data_arr)
# del test_data_arr
# flat_data=np.array(flat_data_arr)
# neur_data=np.array(neur_arr)
# del flat_data_arr
# target=np.array(target_arr)
# del target_arr
# df=pd.DataFrame(neur_data)
# df['Target']=target
# =============================================================================

#%%

x=df.iloc[:,:-1] #everything but the last column
y=df.iloc[:,-1] #the last column
del flat_data
del df
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,stratify=y)
del x
del y
print('Splitted Successfully')

#%%
param_grid={'C':[0.1,1,10,100],'gamma':[0.0001,0.001,0.1,1],'kernel':['rbf','poly']}
svc=svm.SVC(probability=True)
print("The training of the model has started, please wait for while as it may take few minutes to complete")
model=GridSearchCV(svc,param_grid, verbose=1)
model.fit(x_train,y_train)
print('The Model is trained')
#%%

y_pred=model.predict(x_test)
print("The predicted Data is :", y_pred)
print("The actual data is:", np.array(y_test))
print(f"The model is {accuracy_score(y_pred,y_test)*100}% accurate")

ROC = RocCurveDisplay.from_estimator(model, x_test, y_test)

fp = np.where(((y_pred == 1) & (np.array(y_test) == 0)))
fn = np.where(((y_pred == 0) & (np.array(y_test) == 1)))
testframe = np.array(x_test)
falsepos = testframe[fp]
falseneg = testframe[fn]

#%%

xd = ROC.line_.get_xdata()
yd = ROC.line_.get_ydata()
label = ROC.line_.get_label()
acc_score = accuracy_score(y_pred, y_test)

np.save('2560s 30x30 ROC 0.87 AUC', np.array([xd, yd, acc_score]))

#%%

pickle.dump(model,open('img_model_30_2_2560s.p','wb'))
print("Pickle is dumped successfully")

#%%

model=pickle.load(open('img_model_30_2_2560s.p','rb'))

#%%

datalen = len(test_data)//2

i = np.random.randint(641, datalen)
q = np.random.randint(0,2)

index = i+q*datalen

#img= test_data[i+q*datalen]
img = np.reshape(falsepos[3], (-1, 30))
plt.imshow(img)
plt.show()
print(f'Sample is {Categories[test_target_arr[index]]}')

#%%
l=[img.flatten()]
probability=model.predict_proba(l)
for ind,val in enumerate(Categories):
    proba = probability[0][ind]*100
    print(f'{val} = {proba:3.2f}%'.format(val = val, proba = proba))
print("The predicted image is : "+Categories[model.predict(l)[0]])
print(f'Is the image a {Categories[model.predict(l)[0]]} ?(y/n)')
while(True):
  b=input()
  if(b=="y" or b=="n"):
    break
  print("please enter either y or n")

if(b=='n'):
  print("What is the image?")
  for i in range(len(Categories)):
    print(f"Enter {i} for {Categories[i]}")
  k=int(input())
  while(k<0 or k>=len(Categories)):
    print(f"Please enter a valid number between 0-{len(Categories)-1}")
    k=int(input())
  print("Please wait for a while for the model to learn from this image :)")
  flat_arr=flat_data_arr.copy()
  tar_arr=target_arr.copy()
  tar_arr.append(k)
  flat_arr.extend(l)
  tar_arr=np.array(tar_arr)
  flat_df=np.array(flat_arr)
  df1=pd.DataFrame(flat_df)
  df1['Target']=tar_arr
  model1=GridSearchCV(svc,param_grid)
  x1=df1.iloc[:,:-1]
  y1=df1.iloc[:,-1]
  x_train1,x_test1,y_train1,y_test1=train_test_split(x1,y1,test_size=0.20,random_state=77,stratify=y1)
  d={}
  for i in model.best_params_:
    d[i]=[model.best_params_[i]]
  model1=GridSearchCV(svc,d)
  model1.fit(x_train1,y_train1)
  y_pred1=model.predict(x_test1)
  print(f"The model is now {accuracy_score(y_pred1,y_test1)*100}% accurate")
  pickle.dump(model1,open('img_model.p','wb'))
print("Thank you for your feedback")
