from __future__ import print_function
import torch
from torch.autograd import Variable
import cv2
import time
from imutils.video import FPS, WebcamVideoStream
import argparse

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--weights', default='weight2/ssd300_COCO_105000.pth',
                    type=str, help='Trained state_dict file path')
parser.add_argument('--cuda', default=True, type=bool,
                    help='Use cuda in live demo')
parser.add_argument('--num_class',default=8,type=int)
parser.add_argument('--threhold',default=0.2,type=float)
args = parser.parse_args()

COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
FONT = cv2.FONT_HERSHEY_SIMPLEX

video_path = '/home/zhb/Desktop/SSD_HaiShen_pytorch/2.mp4'
video = cv2.VideoCapture(video_path)
# save_video = cv2.VideoWriter(video)
def cv2_demo(net, transform):
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
        i = 1 
        j = 0
        while detections[0, i, j, 0] >= args.threhold:# i=1 holothurian
            class_name = labelmap[i-1]
            # if class_name == 'holothurian' :
            pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
            cv2.rectangle(frame,
                        (int(pt[0]), int(pt[1])),
                        (int(pt[2]), int(pt[3])),
                        COLORS[i % 3], 2)
            trust_level = str(detections[0, i, j, 0]).split('tensor')[1]
                
                # if class_name == 'holothurian':
                #     class_name ="海参"
            
            cv2.putText(frame, class_name+trust_level, (int(pt[0]), int(pt[1])),
                            FONT, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            j += 1
        return frame

    # start video stream thread, allow buffer to fill
    print("[INFO] starting threaded video stream...")
    # stream = WebcamVideoStream(src=0).start()  # default camera
    time.sleep(1.0)
    # start fps timer
    # loop over frames from the video file stream
    while True:
        # grab next frame
        # frame = stream.read()
        ret,frame = video.read()
        key = cv2.waitKey(1) & 0xFF

        # update FPS counter
        fps.update()
        frame = predict(frame)

        # keybindings for display
        if key == ord('p'):  # pause
            while True:
                key2 = cv2.waitKey(1) or 0xff
                cv2.imshow('frame', frame)
                if key2 == ord('p'):  # resume
                    break
            ###---窗口大小可调整---###
        cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
        ###---窗口大小可调整---###
        cv2.imshow('frame', frame)
        if key == 27:  # exit
            break


if __name__ == '__main__':
    import sys
    from os import path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

    from data import BaseTransform, VOC_CLASSES as labelmap
    from ssd import build_ssd
####################################################################
    if torch.cuda.is_available():
        if args.cuda:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else: torch.set_default_tensor_type('torch.FloatTensor')
###################################################################
    net = build_ssd('test', 300,args.num_class)    # initialize SSD
    net.load_state_dict(torch.load(args.weights))
    transform = BaseTransform(net.size, (104/256.0, 117/256.0, 123/256.0))

    fps = FPS().start()
    cv2_demo(net.eval(), transform)

    # stop the timer and display FPS information
    fps.stop()

    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    # cleanup
    cv2.destroyAllWindows()
    stream.stop()
