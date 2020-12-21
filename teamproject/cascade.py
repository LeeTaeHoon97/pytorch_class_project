import cv2
import numpy as np
import os


def do_cascade(image):
    path="C:/dev/Pytorch-workspace_class/teamproject"   
    img = cv2.imread(image)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    cascade=cv2.CascadeClassifier('C:/Users/dlxog/.conda/envs/pytorch_env/Library/etc/haarcascades/haarcascade_frontalface_alt.xml')


    face=cascade.detectMultiScale(gray)
    
  

    for (x,y,w,h) in face:
        roi=img[y:y+h,x:x+w]
    #img 영역자르기의 경우  dst = src[100:600, 200:700]의 경우  src[높이(행), 너비(열)]이다. 즉 ,list형식
    
    _roi=cv2.resize(roi,dsize=(380,410))

    cv2.imwrite(path+"/images/cropped/caded.jpg",_roi)
    
