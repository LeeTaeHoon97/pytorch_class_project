
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import uic
import sys
import cascade
import Model
import time


form_class = uic.loadUiType("ui/inter.ui")[0]

loaded_model=Model.Model()


class WindowClass(QMainWindow, form_class) :
  def __init__(self) :
    super().__init__()
    self.setupUi(self)
    self.analyze_btn.clicked.connect(self.analyze_btnFunc)
    self.upload_btn.clicked.connect(self.upload_btnFunc)
  def analyze_btnFunc(self) :
    result=loaded_model.get_value()
    self.angry_bar.setValue(result[0][0])
    self.disgust_bar.setValue(result[0][1])
    self.fear_bar.setValue(result[0][2])
    self.happy_bar.setValue(result[0][3])
    self.neutral_bar.setValue(result[0][4])
    self.sad_bar.setValue(result[0][5])
    self.surprise_bar.setValue(result[0][6])
  def upload_btnFunc(self) :
    self.upload_btn.setText("")
    fname = QFileDialog.getOpenFileName(self, 'Open file', "",
                                            "JPEG(*.jpg;*.jpeg;*.png)")
    cascade.do_cascade(fname[0])
    img_path="images/cropped/caded.jpg"
    self.upload_btn.setStyleSheet("background-image : url('{}');width:100%;height:100%;".format(img_path))
  

if __name__ == "__main__" :
  app = QApplication(sys.argv)
  myWindow = WindowClass()
  myWindow.show()
  app.exec_()

