import cv2
import numpy as np



def do_cascade(image):
    img = cv2.imread(image)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    cascade=cv2.CascadeClassifier('C:/Users/dlxog/.conda/envs/pytorch_env/Library/etc/haarcascades/haarcascade_frontalface_default.xml')

    face=cascade.detectMultiScale(gray)
    
    print(face)

    for (x,y,w,h) in face:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),1)
        cv2.imshow('test',img[x:x+w][y:y+h])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

print("start")
do_cascade("C:/dev/Pytorch-workspace_class/images/ccc.jpg")