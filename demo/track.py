import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
import numpy as np 
from data import BaseTransform, VOC_CLASSES as labelmap
from ssd import build_ssd
import torch
from torch.autograd import Variable
import cv2
import time
from imutils.video import FPS, WebcamVideoStream
import argparse
from sort import Sort
from PIL import Image

COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
FONT = cv2.FONT_HERSHEY_SIMPLEX

# video_path = '../2.mp4'
video_path = '/home/zhb/Desktop/SSD_HaiShen_pytorch/4.mp4'
video = cv2.VideoCapture(video_path)
# video = WebcamVideoStream(src=0).start()
tracker = Sort()

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--weights', default='weight2/ssd300_COCO_95000.pth',
                    type=str, help='Trained state_dict file path')
parser.add_argument('--cuda', default=True, type=bool,
                    help='Use cuda in live demo')
parser.add_argument('--num_class',default=8,type=int)
parser.add_argument('--threhold',default=0.5,type=float)
args = parser.parse_args()

####################################################################
if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else: torch.set_default_tensor_type('torch.FloatTensor')
###################################################################
    net = build_ssd('test', 300,args.num_class)    # initialize SSD
    net.load_state_dict(torch.load(args.weights))
    transform = BaseTransform(net.size, (104/256.0, 117/256.0, 123/256.0))
kalman_tracker = Sort()
def predict(frame):
        height, width = frame.shape[:2]
        x = torch.from_numpy(transform(frame)[0]).permute(2, 0, 1)
        x = Variable(x.unsqueeze(0))
    # if args.cuda:
        x = x.cuda()
        y = net(x)  # forward pass
        detections = y.data
        # scale each detection back up to the image
        scale = torch.Tensor([width, height, width, height])
        # for i in range(detections.size(1)):
        #     j = 0
        #     while detections[0, i, j, 0] >= args.threhold:
        #         pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
        #         cv2.rectangle(frame,
        #                       (int(pt[0]), int(pt[1])),
        #                       (int(pt[2]), int(pt[3])),
        #                       COLORS[i % 3], 2)
        #         trust_level = str(detections[0, i, j, 0]).split('tensor')[1]
        #         class_name = labelmap[i-1]
        #         # if class_name == 'holothurian':
        #         #     class_name ="海参"
        #         cv2.putText(frame, class_name+trust_level, (int(pt[0]), int(pt[1])),
        #                     FONT, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        #         j += 1
        holothurian_trust_level = detections[0, 1, :, 0]# return trust level
        holothurian_detections = detections[0, 1, :, 1:]# return locations  (x1,y1) (x2,y2)
        # print(holothurian_trust_level)
        # print(holothurian_detections)
        result = []
        j = 0
        while(holothurian_trust_level[j] >= args.threhold):
            pt = list((holothurian_detections[j] * scale).cpu().numpy())
            pt = [int(x) for x in pt]
            pt.append(float(holothurian_trust_level[j]))# obj_confidence 
            pt.append(1)# class_confidence
            pt.append(0)# class_prediction 0 is holothurian
            # pt is : (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
            result.append(pt)
            j += 1
        # return frame
        
        return np.array(result,np.float) # return with [[x1,y1,x2,y2,score,classname]]

while(True):
    ###--- read ---###
    ret , frame = video.read()
    key = cv2.waitKey(1) & 0xFF
    img_PIL = Image.fromarray(frame)
    
    ###---process---###
    detections = predict(frame)
    if detections is not None:
        # print(detections)
        tracked_objects = kalman_tracker.update(detections)

    ###---show---###
    
    for x1, y1, x2, y2, obj_id, cls_pred in tracked_objects: # (x1,y1) 左上  (x2,y2)右下 (cls_pred 的序号)(cls_pred 类别代号)
            cv2.rectangle(frame,(int(x1),int(y1)),(int(x2),int(y2)),COLORS[1],2)
            text = str('holothurian'+':No.'+str(obj_id))
            cv2.putText(frame,text , (int(x1),int(y1)),
                            FONT, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    if key == ord('p'):  # pause
            while True:
                key2 = cv2.waitKey(1) or 0xff
                cv2.imshow('frame', frame)
                if key2 == ord('p'):  # resume
                    break
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.imshow('frame',frame)

cv2.destroyAllWindows()
