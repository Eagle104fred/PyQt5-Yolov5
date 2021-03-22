'''
将detect.py作为模块导入 实时检测
'''

import cv2
import sys   # 添加上一级路径
sys.path.append('../')
# from detect_model import *
from single_detect import *


# cap=cv2.VideoCapture(0)
# cap = cv2.VideoCapture('../data/video/ship.avi')  # 路径返回上一级然后开始加载文件
cap = cv2.VideoCapture('rtsp://admin:SMUwm_007@192.168.1.110/id=1')
wth = detectapi(weights='../weights/best.pt')   # 找到对应文件路径
# wth = detectapi(weights='../weights/yolov5s.pt')
while True:

    rec,img = cap.read()

    result,names =wth.detect([img])
    img=result[0][0] #第一张图片的处理结果图片
    '''
    for cls,(x1,y1,x2,y2),conf in result[0][1]: #第一张图片的处理结果标签。
        print(cls,x1,y1,x2,y2,conf)
        cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0))
        cv2.putText(img,names[cls],(x1,y1-20),cv2.FONT_HERSHEY_DUPLEX,1.5,(255,0,0))
    '''
    cv2.imshow("vedio",img)

    if cv2.waitKey(1)==ord('q'):

        break