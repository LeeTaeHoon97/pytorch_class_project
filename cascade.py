import cv2
import numpy as np
import os


def do_cascade(image):
    path="C:\dev\Pytorch-workspace_class"   
    img = cv2.imread(image)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    
    # cascade=cv2.CascadeClassifier('C:/Users/dlxog/.conda/envs/pytorch_env/Library/etc/haarcascades/haarcascade_frontalface_default.xml')
    cascade=cv2.CascadeClassifier('C:/Users/dlxog/.conda/envs/pytorch_env/Library/etc/haarcascades/haarcascade_frontalface_alt.xml')


    face=cascade.detectMultiScale(gray)
    
  

    for (x,y,w,h) in face:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),1)
        roi=img[y:y+h,x:x+w]
    #img 영역자르기의 경우  dst = src[100:600, 200:700]의 경우  src[높이(행), 너비(열)]이다. 즉 ,list형식
    
    cv2.imwrite(path+"/images/caded.jpg",roi)
    
    os.remove(image)

#do_cascade("C:/dev/Pytorch-workspace_class/images/test5.jpg")