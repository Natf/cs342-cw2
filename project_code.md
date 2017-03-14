

```python
# imports
import pandas as pd
import numpy as np

import sys

import os
from os import listdir
from os.path import isfile, join

from sklearn.decomposition import PCA

from sklearn.neural_network import MLPClassifier

import keras
import joblib

from sklearn.svm import SVC
import cv2
from matplotlib import pyplot as plt
print "loaded imports"
```

    Using TensorFlow backend.


    loaded imports



```python
def loadAndCropImage(filename):
    img = cv2.imread(filename, 0)
    img = cv2.resize(img, (640, 360)) 
    return img
```


```python
def loadRawTrainingData():
    training_data = np.array([])
    training_labels = np.array([])
    folders = [d for d in listdir("./train/") if not isfile(join("./train/", d)) and d not in ["cropped"]]
    for category in folders:
        percent = 0
        currentfile = 0.0
        images = [file for file in listdir("./train/"+category)]
        np.random.shuffle(images)
        images = images[0:20] #limit number of images used
        print ("Loading: " + category)
        for image in images:
            percent = currentfile/float(len(images))*100.0;
            print("\r(" + str(percent) + "%)                  "),
            currentfile += 1
            img = loadAndCropImage("./train/" + category + "/" + image)
            vector_data = img.reshape(1,230400) 

            if len(training_data) == 0:
                training_data = np.append(training_data, vector_data)
                training_data = training_data.reshape(1,230400)
            else:
                training_data   = np.concatenate((training_data, vector_data), axis=0)
            training_labels = np.append(training_labels,category)
            
            if (currentfile == len(images)):
                print("\r(100%)                  ")
                
    print("loaded training data")
                
    return {'data':training_data, 'labels':training_labels}
            
RAW_training = loadRawTrainingData()
RAW_training_data = RAW_training['data']
RAW_training_labels = RAW_training['labels']
```

    Loading: ALB
    (100%)                      
    Loading: BET
    (100%)                         
    Loading: DOL
    (100%)                         
    Loading: LAG
    (100%)                             
    Loading: NoF
    (100%)                             
    Loading: OTHER
    (100%)                             
    Loading: SHARK
    (100%)                             
    Loading: YFT
    (100%)                                  
    loaded training data



```python
X = RAW_training_data
y = RAW_training_labels
y = y.reshape(y.shape[0],)
RAW_MLP = MLPClassifier()
RAW_MLP.fit(X,y)
```




    MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
           beta_2=0.999, early_stopping=False, epsilon=1e-08,
           hidden_layer_sizes=(100,), learning_rate='constant',
           learning_rate_init=0.001, max_iter=200, momentum=0.9,
           nesterovs_momentum=True, power_t=0.5, random_state=None,
           shuffle=True, solver='adam', tol=0.0001, validation_fraction=0.1,
           verbose=False, warm_start=False)




```python
def getPrediction(filename):
    img = loadAndCropImage('./train/LAG/img_00091.jpg')
    vector_data = img.reshape(1,230400) 
    
    return RAW_MLP.predict_proba(vector_data)

def savePredictions():
    headers = ["image", "ALB", "BET", "DOL", "LAG", "NoF", "OTHER", "SHARK", "YFT"]
    predictions = np.array([])
    maxi = 0;
    print "start"
    for file in listdir("./test_stg1/"):
        prediction = getPrediction("./test_stg1/"+file)
        prediction_s = np.array(prediction, dtype='string')
        prediction_s = np.insert(prediction_s, 0, file)
        prediction_s = prediction_s.reshape(1,9)
        if len(predictions) == 0:
            predictions = np.append(predictions, prediction_s)
            predictions = predictions.reshape(1,9)
        else:
            predictions   = np.concatenate((predictions, prediction_s), axis=0)
    
    df = pd.DataFrame(predictions, columns=headers)
    df.to_csv('results.csv', index=False, header=True, sep=',')
    print "saved to results.csv"
    
savePredictions()
        
```

    start
    [['img_00005.jpg' '0.0' '0.0' ..., '0.0' '0.0' '0.0']
     ['img_00007.jpg' '0.0' '0.0' ..., '0.0' '0.0' '0.0']
     ['img_00009.jpg' '0.0' '0.0' ..., '0.0' '0.0' '0.0']
     ..., 
     ['img_07908.jpg' '0.0' '0.0' ..., '0.0' '0.0' '0.0']
     ['img_07910.jpg' '0.0' '0.0' ..., '0.0' '0.0' '0.0']
     ['img_07921.jpg' '0.0' '0.0' ..., '0.0' '0.0' '0.0']]
    saved to results.csv



```python
score = RAW_MLP.score(RAW_training_data, RAW_training_labels);
print score
```

    0.18125



```python
def loadHOGTrainingData():
    training_data = np.array([])
    training_labels = np.array([])
    folders = [d for d in listdir("./train/") if not isfile(join("./train/", d)) and d not in ["cropped"]]
    for category in folders:
        percent = 0
        currentfile = 0.0
        images = [file for file in listdir("./train/"+category)]
        np.random.shuffle(images)
        images = images[0:20] #limit number of images used
        print ("Loading: " + category)
        for image in images:
            percent = currentfile/float(len(images))*100.0;
            print("\r(" + str(percent) + "%)                  "),
            currentfile += 1
            img = loadAndCropImage("./train/" + category + "/" + image)
            hog = cv2.HOGDescriptor()
            h = hog.compute(img)
            h = h.astype(np.float64)
            np.random.shuffle(h)
            h = h[0:30000,:] # trim vector so all are same size
            vector_data = h.reshape(1,30000) 

            if len(training_data) == 0:
                training_data = np.append(training_data, vector_data)
                training_data = training_data.reshape(1,30000)
            else:
                training_data   = np.concatenate((training_data, vector_data), axis=0)
            training_labels = np.append(training_labels,category)
            
            if (currentfile == len(images)):
                print("\r(100%)                  ")
                
    print("loaded training data")
                
    return {'data':training_data, 'labels':training_labels}
            
HOG_training = loadHOGTrainingData()
HOG_training_data = HOG_training["data"]
HOG_training_labels = HOG_training["labels"]
```

    Loading: ALB
    (100%)                                      
    Loading: BET
    (100%)                                      
    Loading: DOL
    (90.0%)                                   


```python
pca = PCA(n_components=0.7, whiten=True)
HOG_PCA_training_data = pca.fit_transform(HOG_training_data)
print(len(pca.explained_variance_ratio_)) 
df.to_csv('HOG_PCA_training_data.csv', index=False, header=False, sep=',')
```

    start
    101
    finished



```python
X = HOG_PCA_training_data
y = HOG_training_labels
y = y.reshape(y.shape[0],)
HOG_PCA_MLP = MLPClassifier()
HOG_PCA_MLP.fit(X,y)
joblib.dump(HOG_PCA_MLP, 'HOG_PCA_MLP.pkl')
```




    ['fishy_svm.pkl']




```python
HOG_PCA_MLP = joblib.load('HOG_PCA_MLP.pkl')
hog = cv2.HOGDescriptor()
def getPrediction(filename):
    img = loadAndCropImage(filename)
    hog = cv2.HOGDescriptor()
    h = hog.compute(img)
    h = h.astype(np.float64)
    np.random.shuffle(h)
    h = h[0:30000,:] # trim vector so all are same size
    vector_data = h.reshape(1,30000) 
    vector_data = pca.transform(vector_data)
    
    return HOG_PCA_MLP.predict_proba(vector_data)

def savePredictions():
    headers = ["image", "ALB", "BET", "DOL", "LAG", "NoF", "OTHER", "SHARK", "YFT"]
    predictions = np.array([])
    maxi = 0;
    print "start"
    for file in listdir("./test_stg1/"):
        prediction = getPrediction("./test_stg1/"+file)
        prediction_s = np.array(prediction, dtype='string')
        prediction_s = np.insert(prediction_s, 0, file)
        prediction_s = prediction_s.reshape(1,9)
        if len(predictions) == 0:
            predictions = np.append(predictions, prediction_s)
            predictions = predictions.reshape(1,9)
        else:
            predictions   = np.concatenate((predictions, prediction_s), axis=0)
    
    df = pd.DataFrame(predictions, columns=headers)
    df.to_csv('results.csv', index=False, header=True, sep=',')
    print "saved to results.csv"
    
savePredictions()
        
```

    start
    [['img_00005.jpg' '0.101625829947' '0.090168200702' ..., '0.0866519483585'
      '0.224467081952' '0.113276597506']
     ['img_00007.jpg' '0.150154870059' '0.0915591696494' ..., '0.103151863622'
      '0.174157797734' '0.106738991248']
     ['img_00009.jpg' '0.112565931175' '0.101217377506' ..., '0.104064295386'
      '0.223307787316' '0.117894210084']
     ..., 
     ['img_07908.jpg' '0.118601702101' '0.141178431376' ..., '0.0838477643902'
      '0.155612804668' '0.111646313732']
     ['img_07910.jpg' '0.0855785380803' '0.0591888437491' ...,
      '0.0395609406577' '0.264990589688' '0.126941001534']
     ['img_07921.jpg' '0.141956116339' '0.0955614675587' ..., '0.0670661841732'
      '0.172235934501' '0.147568995479']]
    saved to results.csv



```python
score = HOG_PCA_MLP.score(HOG_PCA_training_data, HOG_training_labels);
print score
```

    1.0



```python

```
