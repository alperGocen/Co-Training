import cv2
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import accuracy_score
from os import walk
import numpy as np
from LocalBinaryPatterns import LocalBinaryPatterns
#import FaceCropper as fc
from os import listdir
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import keras
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from skimage import feature
import numpy as np

    
def extract_surf_features(img_to_extract):
    feature_vector = np.array([])
    surf = cv2.xfeatures2d.SURF_create()
    surf.setHessianThreshold(800)   
    kp,des = surf.detectAndCompute(img_to_extract,None)
    i=0
    if len(kp) < 50:
        return None        
    for feature in des:
        feature_vector = np.append(feature_vector,feature)
        i+=1
        if i==50:
            break
    feature_vector = feature_vector.flatten()
    return feature_vector 

def extract_lbp_features(img_to_extract):
    desc = LocalBinaryPatterns(98,8)
    desc_feature = desc.describe(img_to_extract)
    return desc_feature

def extract_all_lbp_features(directory):
    features = []
    for name in listdir(directory):
        filename = directory + '/' + name
        image = cv2.imread(filename,0)
        if(image is not None):
            feature_des = extract_lbp_features(image)
            if feature_des is not None:
                features.append(feature_des)
        else:
            continue
    features = np.array(features)
    return features
    
al = cv2.imread('alper.jpeg',0)
ec = extract_lbp_features(al)
feat_Aaron = extract_all_lbp_features('AaronJudge')
feat_adam = extract_all_lbp_features('AdamSandler')

def extract_all_features(directory):
    features = []
    for name in listdir(directory):
        filename = directory + '/' + name
        image = cv2.imread(filename)
        if(image is not None):
            feature_des = extract_surf_features(image)
            if feature_des is not None:
                features.append(feature_des)
        else:
            continue
    features = np.array(features)
    return features

def createModel1(x_train,y_train,epochs,batch_size):
    model = Sequential()
    model.add(Dense(1024, activation='relu',input_shape=(100,)))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256,activation='relu'))
    model.add(Dense(256,activation='relu'))
    model.add(Dense(256,activation='relu'))
    model.add(Dense(128,activation='relu'))
    model.add(Dense(128,activation='relu'))
    model.add(Dense(128,activation='relu'))
    model.add(Dense(64,activation='relu'))
    model.add(Dense(64,activation='relu'))
    model.add(Dense(64,activation='relu'))
    model.add(Dense(32,activation='relu'))
    model.add(Dense(32,activation='relu'))
    model.add(Dense(16,activation='relu'))
    model.add(Dense(16,activation='relu'))
    model.add(Dense(16,activation='relu'))
    model.add(Dense(2,activation='softmax'))
    model.summary()
    
    model.compile(optimizer=keras.optimizers.Adam(lr=0.0001), loss='categorical_crossentropy',metrics=['accuracy'])
    
    model.fit(x_train[300:], y_train[300:],
              epochs=epochs,
              
              batch_size=batch_size,validation_data=(x_train[:300],y_train[:300]))
    return model

def createModel2(x_train,y_train,epochs,batch_size):
    model2 = Sequential()
    model2.add(Dense(512, activation='relu',input_shape=(3200,)))
    model2.add(Dense(512, activation='relu'))
    model2.add(Dense(512, activation='relu'))
    model2.add(Dense(256,activation='relu'))
    model2.add(Dense(256,activation='relu'))
    model2.add(Dense(256,activation='relu'))
    model2.add(Dense(128,activation='relu'))
    model2.add(Dense(128,activation='relu'))
    model2.add(Dense(128,activation='relu'))
    model2.add(Dense(64,activation='relu'))
    model2.add(Dense(64,activation='relu'))
    model2.add(Dense(64,activation='relu'))
    model2.add(Dense(32,activation='relu'))
    model2.add(Dense(32,activation='relu'))
    model2.add(Dense(32,activation='relu'))
    model2.add(Dense(2,activation='softmax'))
    model2.summary()
    
    model2.compile(optimizer='Ada', loss='binary_crossentropy',metrics=['accuracy'])
    
    model2.fit(x_train[300:], y_train[300:],
              epochs=epochs,
              
              batch_size=batch_size,validation_data=(x_train[:300],y_train[:300]))
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
           out2_class=np.append(out2_class,np.float32(1))
           
        else:
           out2_class=np.append(out2_class,np.float32(0))
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





epochs = 200
batch_size = 32



model1.evaluate(x_train2,y_train2)
model2.evaluate(x_train,y_train)


for i in range(50):
    x = np.random.random((1, 20))
    out1 = model1.predict(x, verbose=1)
    out2 = model2.predict(x,verbose=1)
    
    out_class,model_id =  getNextFit(out1,out2)
    
    if model_id == 1:
         model2.fit(x,out_class,epochs=1,batch_size=128)
    else:
         model1.fit(x,out_class,epochs=1,batch_size=128)
        
model1.evaluate(x_train2,y_train2)
model2.evaluate(x_train,y_train)    




vector_aaron = extract_all_features('AaronJudge')
vector_adam = extract_all_features('AdamSandler')
vector_set_y = np.array([],dtype=np.float32)



total_set_x = []
total_set_y = []
for vector in feat_Aaron:
    total_set_x.append(vector)

for vector in feat_adam:
    total_set_x.append(vector)

total_set_x = np.array(total_set_x)

for i in range (0,len(feat_Aaron)):
    total_set_y.append([1.,0.])
    
for i in range(0,len(feat_adam)):
    total_set_y.append([0.,1.])

total_set_y= np.array(total_set_y)


idx = np.random.permutation(len(total_set_x))
total_set_x,total_set_y = total_set_x[idx],total_set_y[idx]




model1 = createModel1(total_set_x,total_set_y,epochs,batch_size)
model1.evaluate(total_set_x[:300],total_set_y[:300],batch_size=32)


filename = 'AdamSandler' + '/' + "6.jpg"
test=extract_lbp_features(cv2.imread(filename,0))
test = test.T
test = test.reshape(-1,1)


model1.predict(test,verbose=1)




