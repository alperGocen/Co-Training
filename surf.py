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
from keras.layers import Dropout
import numpy as np
from skimage import feature
import numpy as np
import pickle 
from keras.regularizers import l2
import random
from keras.models import model_from_json
from keras import regularizers
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation
from keras.layers import LeakyReLU

# --------------surf region -----------------------------   
def extract_surf_features(img_to_extract):
    feature_vector = np.array([])
    surf = cv2.xfeatures2d.SURF_create()
    surf.setHessianThreshold(500)   
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
#-----------------end of surf region ----------------------------
    

#-------------------lbp region ------------------------
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

#--------------------end of lbp region ----------------------
    



#--------------------------------lbp model -----------------------------
def createModel1(x_train,y_train,epochs,batch_size,lr):
    model = Sequential()
    model.add(Dense(1024, activation='relu',input_shape=(100,)))
    model.add(Dropout(0.2))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256,activation='relu'))
    model.add(Dense(256,activation='relu'))
    model.add(Dense(256,activation='relu'))
    model.add(Dense(128,activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128,activation='relu'))
    model.add(Dense(128,activation='relu'))
    model.add(Dense(64,activation='relu'))
    model.add(Dense(64,activation='relu'))
    model.add(Dense(64,activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32,activation='relu'))
    model.add(Dense(32,activation='relu'))
    model.add(Dense(16,activation='relu'))
    model.add(Dense(16,activation='relu'))
    model.add(Dense(16,activation='relu'))
    model.add(Dense(2,activation='softmax'))
    model.summary()
    
    model.compile(optimizer=keras.optimizers.Adam(lr=lr), loss='binary_crossentropy',metrics=['accuracy'])
    
    model_his=model.fit(x_train[300:], y_train[300:],
              epochs=epochs,
              
              batch_size=batch_size,validation_data=(x_train[:300],y_train[:300]))
    return model,model_his
#--------------------------end of lbp model ---------------------
    

#----------------------------- surf model-----------------------------

def createModel2(x_train,y_train,epochs,batch_size,lr):
    model2 = Sequential()
    
    model2.add(Dense(512, input_shape=(3200,),kernel_initializer='random_uniform'))
    model2.add(BatchNormalization())
    model2.add(LeakyReLU(alpha=0.1))
    model2.add(Dropout(0.2))
    
    """
    model2.add(Dense(512,kernel_initializer='random_uniform'))
    model2.add(BatchNormalization())
    model2.add(Activation("sigmoid"))
    model2.add(Dropout(0.5))
    
    model2.add(Dense(512,kernel_initializer='random_uniform'))
    model2.add(BatchNormalization())
    model2.add(Activation("sigmoid"))
    model2.add(Dropout(0.5))"""
    
    model2.add(Dense(256,kernel_initializer='random_uniform'))
    model2.add(BatchNormalization())
    model2.add(LeakyReLU(alpha=0.1))
    #model2.add(Dropout(0.5))
    """
    model2.add(Dense(256,kernel_initializer='random_uniform'))
    model2.add(BatchNormalization())
    model2.add(Activation("sigmoid"))
    model2.add(Dropout(0.5))
    
    model2.add(Dense(256, kernel_initializer='random_uniform'))
    model2.add(BatchNormalization())
    model2.add(Activation("sigmoid"))
    model2.add(Dropout(0.5))"""
    
    model2.add(Dense(128,kernel_initializer='random_uniform'))
    model2.add(BatchNormalization())
    model2.add(LeakyReLU(alpha=0.1))
    #model2.add(Dropout(0.5))
    """
    model2.add(Dense(128,kernel_initializer='random_uniform'))
    model2.add(BatchNormalization())
    model2.add(Activation("sigmoid"))
    model2.add(Dropout(0.5))
    
    model2.add(Dense(128, kernel_initializer='random_uniform'))
    model2.add(BatchNormalization())
    model2.add(Activation("sigmoid"))
    model2.add(Dropout(0.5))"""
    
    model2.add(Dense(64,kernel_initializer='random_uniform'))
    model2.add(BatchNormalization())
    model2.add(LeakyReLU(alpha=0.1))
    #model2.add(Dropout(0.5))
    """
    model2.add(Dense(64,kernel_initializer='random_uniform'))
    model2.add(BatchNormalization())
    model2.add(Activation("sigmoid"))
    model2.add(Dropout(0.5))
    
    model2.add(Dense(64, kernel_initializer='random_uniform'))
    model2.add(BatchNormalization())
    model2.add(Activation("sigmoid"))
    model2.add(Dropout(0.5))"""
    
    model2.add(Dense(32,kernel_initializer='random_uniform'))
    model2.add(BatchNormalization())
    model2.add(LeakyReLU(alpha=0.1))
    #model2.add(Dropout(0.5))
    """
    model2.add(Dense(32,kernel_initializer='random_uniform'))
    model2.add(BatchNormalization())
    model2.add(Activation("sigmoid"))
    model2.add(Dropout(0.5))
    
    model2.add(Dense(32, kernel_initializer='random_uniform'))
    model2.add(BatchNormalization())
    model2.add(Activation("sigmoid"))
    model2.add(Dropout(0.5))"""
    
    model2.add(Dense(2))
    model2.add(BatchNormalization())
    model2.add(Activation('softmax'))
    model2.summary()
    
    model2.compile(optimizer=keras.optimizers.SGD(lr=lr), loss='categorical_crossentropy',metrics=['accuracy'])
    
    model2_his=model2.fit(x_train[300:], y_train[300:],
              epochs=epochs,
              
              batch_size=batch_size,validation_data=(x_train[:300],y_train[:300]))
    return model2,model2_his
#-------------------------------------end of surf model -------------------


#-------------------------------------utility functions -------------------------
def getNextFit(out1,out2):    
    print("Surf : ",out1)
    print("Lbp : ",out2)
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
        model_id = 2 #Lbp train edilecek
    else:
        out_class = np.array([out2_class],dtype=np.float32)
        model_id = 1  # Surf Train Edilecek
    return out_class,model_id 



def inputGenerator(feat_Aaron,feat_adam):
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
    
    return total_set_x,total_set_y

def saveFeatures(filename,feat):
    np.savetxt(filename,feat,fmt='%s')

def loadFeatures(filename):
    return np.loadtxt(filename)
    
#----------------------------end of utility functions-------------------------



#--------surf feature create and Save-----------
    
def saveSurfFeatures(filename):
    features = extract_all_features(filename)
    saveFeatures(filename+'_surf_features.txt',features)
#----------end of surf feature create and Save---------
    
    
#-----------lbp feature create and Save---------------
def saveLbpFeatures(filename):
    features = extract_all_lbp_features(filename)
    saveFeatures(filename+'_lbp_features.txt',features)
#----------end of lbp feature create and Save---------







# IMPORTANT Aaron = 0   Adam = 1

#----------------------------Training models -------------------------

def trainSurfModel(epochs,batch_size,feat_aaron_surf,feat_adam_surf,lr):
    total_set_x_surf,total_set_y_surf = inputGenerator(feat_aaron_surf,feat_adam_surf)
    model_surf,model_surf_his = createModel2(total_set_x_surf,total_set_y_surf,epochs,batch_size,lr)
    plt.plot(model_surf_his.history['acc'])
    plt.plot(model_surf_his.history['val_acc'])
    plt.plot(model_surf_his.history['loss'])
    plt.legend(['acc', 'val_acc','loss'], loc='upper left')
    plt.savefig("surf-1")
    return model_surf

def trainLbpModel(epochs,batch_size,feat_Aaron,feat_adam,lr):
    total_set_x_lbp,total_set_y_lbp = inputGenerator(feat_Aaron,feat_adam)
    model_lbp,model_lbp_his = createModel1(total_set_x_lbp,total_set_y_lbp,epochs,batch_size,lr)
    plt.plot(model_lbp_his.history['acc'])
    plt.plot(model_lbp_his.history['val_acc'])
    plt.plot(model_lbp_his.history['loss'])
    plt.legend(['acc', 'val_acc','loss'], loc='upper left')
    plt.savefig("lbp-3")
    return model_lbp


#--------------------------end of Training models -------------------------

def createUnLabeledTrainData(filename1, filename2):
    list1 = listdir(filename1)
    for i in range(len(list1)):
        list1[i] = filename1+"/"+list1[i]
        
    list2 =  listdir(filename2)
    for i in range(len(list2)):
        list2[i] = filename2+"/"+list2[i]
        
    directory = list1 + list2
    
    random.shuffle(directory)
    return directory

def coTraining(directory,model_surf,model_lbp):
    
    for name in directory:
            
            image = cv2.imread(name)
            
            if(image is not None):
                surfImg= extract_surf_features(image).reshape(-1,1).T
                lbpImg= extract_lbp_features(cv2.imread(name,0)).reshape(-1,1).T
                surf_predict = model_surf.predict(surfImg, verbose=1)
                lbp_predict = model_lbp.predict(lbpImg,verbose = 1)
                out_class,model_id =  getNextFit(surf_predict,lbp_predict)
                
                if model_id == 1:
                     print("lbp daha iyi")
                     model_surf.fit(surfImg,out_class,epochs=1,batch_size=32)
                else:
                     print("surf daha iyi")
                     model_lbp.fit(lbpImg,out_class,epochs=1,batch_size=32)
                


#--------------------------------Main----------------------------------------

""" Just do them one to obtain all features
saveSurfFeatures('AaronJudge')
saveLbpFeatures('AaronJudge')

saveSurfFeatures('AdamSandler')
saveLbpFeatures('AdamSandler')
"""

feat_aaron_surf = loadFeatures("AaronJudge_surf_features.txt")
feat_adam_surf = loadFeatures("AdamSandler_surf_features.txt")

feat_aaron_lbp = loadFeatures("AaronJudge_lbp_features.txt")
feat_adam_lbp = loadFeatures("AdamSandler_lbp_features.txt")                     
       

model_surf = trainSurfModel(200,64,feat_aaron_surf,feat_adam_surf,0.0001) 
model_lbp = trainLbpModel(200,32,feat_aaron_lbp,feat_adam_lbp,0.001)       

unLabeled_data_paths = createUnLabeledTrainData("adamtest","aarontest")     

coTraining(unLabeled_data_paths,model_surf,model_lbp)
  
                     
                     
                     
#-----------------------------End Of Main -------------------------------------










