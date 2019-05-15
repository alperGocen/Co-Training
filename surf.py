import cv2
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import accuracy_score
from os import walk
import numpy as np
import FaceCropper as fc
from os import listdir
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import keras
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

    
def extract_surf_features(img_to_extract):
    feature_vector = np.array([])
    surf = cv2.xfeatures2d.SURF_create()
    surf.setHessianThreshold(400)   
    kp,des = surf.detectAndCompute(img_to_extract,None)
    i=0
    for feature in des:
        feature_vector = np.append(feature_vector,feature)
        i+=1
        if i==50:
            break
    feature_vector = feature_vector.flatten()
    feature_vector = feature_vector.reshape(-1,1) 
    return feature_vector 


def extract_all_features(directory):
    features = np.array([])
    for name in listdir(directory):
        filename = directory + '/' + name
        image = cv2.imread(filename)
        if(image is not None):
            features = np.append(features,extract_surf_features(image))
        else:
            continue
    return features

def createModel1(x_train,y_train,epochs,batch_size):
    model = Sequential()
    model.add(Dense(512, activation='relu',input_shape=(20,)))
    model.add(Dense(256,activation='relu'))
    model.add(Dense(128,activation='relu'))
    model.add(Dense(64,activation='relu'))
    model.add(Dense(32,activation='relu'))
    model.add(Dense(10,activation='softmax'))
    model.summary()
    
    model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
    
    model.fit(x_train, y_train,
              epochs=epochs,
              batch_size=batch_size)
    return model

def createModel2(x_train,y_train,epochs,batch_size):
    model2 = Sequential()
    model2.add(Dense(512, activation='relu',input_shape=(20,)))
    model2.add(Dense(512, activation='relu'))
    model2.add(Dense(256,activation='relu'))
    model2.add(Dense(256,activation='relu'))
    model2.add(Dense(128,activation='relu'))
    model2.add(Dense(64,activation='relu'))
    model2.add(Dense(32,activation='relu'))
    model2.add(Dense(10,activation='softmax'))
    model2.summary()
    
    model2.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
    
    model2.fit(x_train, y_train,
              epochs=epochs,
              batch_size=batch_size)
    return model2

def getNextFit(out1,out2):    
    print("Out1 : ",out1)
    print("Out2 : ",out2)
    out1_max = np.max(out1)
    out2_max = np.max(out2)
    out1_class = np.array([],dtype=np.float32)
    out2_class = np.array([],dtype=np.float32)
    for i in range(len(out1[0])):
        if out1_max == out1[0][i]:
           out1_class=np.append( out1_class,np.float32(1))
           
        else:
           out1_class=np.append( out1_class,np.float32(0))
        i+=1
     
    for i in range(len(out2[0])):
        if out2_max == out2[0][i]:
           out2_class=np.append( out2_class,np.float32(1))
           
        else:
           out2_class=np.append( out2_class,np.float32(0))
        i+=1
    
    print("Out1 prob : ",out1_max," class : ",out1_class)
    print("Out2 prob : ",out2_max," class : ",out2_class)

    if out1_max >= out2_max:
        out_class = np.array([out1_class],dtype=np.float32)
        model_id = 2
    else:
        out_class = np.array([out2_class],dtype=np.float32)
        model_id = 1  
    return out_class,model_id



x_train = np.random.random((1000, 20))
y_train = keras.utils.to_categorical(np.random.randint(10, size=(1000, 1)), num_classes=10)
x_train2 = np.random.random((1000, 20))
y_train2 = keras.utils.to_categorical(np.random.randint(10, size=(1000, 1)), num_classes=10)

epochs = 20
batch_size = 128

model1 = createModel1(x_train,y_train,epochs,batch_size)
model2 = createModel2(x_train2,y_train2,epochs,batch_size)

model1.evaluate(x_train2,y_train2)
model2.evaluate(x_train,y_train)

for i in range(50):
    x = np.random.random((1, 20))
    out1 = model1.predict(x, verbose=1)
    out2 = model2.predict(x,verbose=1)
    
    out_class,model_id =  getNextFit(out1,out2)
    
    if model_id == 1:
         model1.fit(x,out_class,epochs=1,batch_size=128)
    else:
         model2.fit(x,out_class,epochs=1,batch_size=128)
        
model1.evaluate(x_train2,y_train2)
model2.evaluate(x_train,y_train)    

img = cv2.imread('alper.jpeg')
f = extract_surf_features(img)
vector = extract_all_features('AaronJudge')



