import cv2
import numpy as np
import tensorflow
import keras
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.models import load_model # run thu m
from numpy import reshape
import time

vid = cv2.VideoCapture(0)
model=load_model('FinalProjectAI.h5')
Ptime  = 0
while(True):
    
    Ctime= time.time()
    ret, img = vid.read()
    img = cv2.resize(img, (240,240))
    img_arr=img_to_array(img)
    img_arr=np.reshape(img_arr,(1,240,240,3 ))
    img_arr=img_arr.astype('float')
    img_arr/=255
    
    y=model.predict(img_arr)
    for i in range(0,20,2): 
       
        cv2.circle(img, (int(y[0][i]),round(y[0][i+1])),2,(255,0,0),2) 
    
    fps = 1/(Ctime -  Ptime)
    img = cv2.putText(img,str(round(fps,1)),(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2 )
    Ptime = Ctime
    cv2.imshow('frame', img)
      
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print(img_arr.shape)
        break
  
vid.release()
cv2.destroyAllWindows()
