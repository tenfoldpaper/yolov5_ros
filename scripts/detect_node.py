#!/usr/bin/python3
"""Run inference with a YOLOv5 model on images, videos, directories, streams

Usage:
    $ python path/to/detect.py --source path/to/img.jpg --weights yolov5s.pt --img 640
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import time

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, colorstr, non_max_suppression, \
    apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
from utils.augmentations import Albumentations, augment_hsv, copy_paste, letterbox, mixup, random_perspective

import rospy
from sensor_msgs.msg import Image
from yolov5_detector.msg import DetectionMsg, DetectionArray
from cv_bridge import CvBridge, CvBridgeError
import message_filters

class yolov5_node:

    def __init__(self,
        weights='yolov5s.pt',  # model.pt path(s)
        source_topic='/camera/color/image_raw',  # RGB image topic
        depth_topic='/camera/aligned_depth_to_color/image_raw', # Depth image topic
        camera_topic='/camera/color/camerainfo', # camera info topic
        depth_camera_topic='camera/depth/camerainfo', # Depth camera info topic
        imgsz=640,  # inference size (pixels)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=True,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project='runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        publish=False,
        detection_topic='/yolov5/detection'
        ):
        
        
        # Assign everything to self first 
        self.source_topic = rospy.get_param("~source_topic")
        self.weights = rospy.get_param("~weights")
        self.depth_topic = rospy.get_param("~depth_topic")
        self.camera_topic = rospy.get_param("~camera_topic")
        self.depth_camera_topic = rospy.get_param("~depth_camera_topic")
        
        self.imgsz = rospy.get_param("~imgsz")
        self.conf_thres = rospy.get_param("~conf_thres")
        self.iou_thres = rospy.get_param("~iou_thres")
        self.max_det = rospy.get_param("~max_det")
        self.device = rospy.get_param("~device")
        
        self.view_img = rospy.get_param("~view_img")
        self.save_txt = rospy.get_param("~save_txt")
        self.save_conf = rospy.get_param("~save_conf")
        self.save_crop = rospy.get_param("~save_crop")
        self.nosave = rospy.get_param("~nosave")
        
        classes = rospy.get_param("~classes")
        if(classes == ""):
            classes = None
        self.classes = classes
        self.agnostic_nms = rospy.get_param("~agnostic_nms")
        self.augment = rospy.get_param("~augment")
        self.visualize = rospy.get_param("~visualize")
        self.update = rospy.get_param("~update")
        
        self.project = rospy.get_param("~project")
        self.name = rospy.get_param("~name")
        self.exist_ok = rospy.get_param("~exist_ok")
        self.line_thickness = int(rospy.get_param("~line_thickness"))
        self.hide_labels = rospy.get_param("~hide_labels")
        
        self.hide_conf = rospy.get_param("~hide_conf")
        self.half = rospy.get_param("~half")
        self.detection_topic = rospy.get_param("~detection_topic")
        self.publish = rospy.get_param("~publish")
        self.bridge = CvBridge()
        self.detection_count = 0
        ## Initialisation is more or less identical as the original yolov5 detect module
        self.save_img = not nosave #and not source.endswith('.txt')  # save inference images
        # webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        #     ('rtsp://', 'rtmp://', 'http://', 'https://')) definitely not using webcam
        
        rospy.loginfo("Initialising YOLOv5 detector node with the following parameters:\n" + \
            f"Source RGB topic        : {self.source_topic}\n" + \
            f"Depth topic             : {self.depth_topic}\n" + \
            f"Camera info topic       : {self.camera_topic}\n" + \
            f"Depth camera info topic : {self.depth_camera_topic}\n" + \
            f"Weights                 : {self.weights}\n" + \

            f"Image size              : {self.imgsz}\n" + \
            f"Confidence threshold    : {self.conf_thres}\n" + \
            f"IOU threshold           : {self.iou_thres}\n" + \
            f"Max. detections         : {self.max_det}\n" + \
            f"Device                  : {self.device}\n" + \
            
            f"View image              : {self.view_img}\n" + \
            f"Save txt                : {self.save_txt}\n" + \
            f"Save conf               : {self.save_conf}\n" + \
            f"Save crop               : {self.save_crop}\n" + \
            f"Nosave                  : {self.nosave}\n" + \
            
            f"Classes                 : {self.classes}\n" + \
            f"Agnostic NMS            : {self.agnostic_nms}\n" + \
            f"Augment                 : {self.augment}\n" + \
            f"Visualize               : {self.visualize}\n" + \
            f"Update                  : {self.update}\n" + \
            
            f"Project dir             : {self.project}\n" + \
            f"Name                    : {self.name}\n" + \
            f"exist_ok                : {self.exist_ok}\n" + \
            f"Line thickness          : {self.line_thickness}\n" + \
            f"Hide labels             : {self.hide_labels}\n" + \
            
            f"Hide confidence value   : {self.hide_conf}\n" + \
            f"Half                    : {self.half}\n" + \
            f"Detection topic         : {self.detection_topic}\n" + \
            f"Publish                 : {self.publish}\n"
        )

        # Directories
        self.save_dir = increment_path(Path(self.project) / self.name, exist_ok=self.exist_ok)  # increment run
        (self.save_dir / 'labels' if self.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Initialize
        set_logging()
        # since we are running this on nano, we shouldn't need to set this, but for cross-platform compat let's just leave it 
        self.device = select_device(device)
        self.half &= self.device.type != 'cpu'  # half precision only supported on CUDA
        # Load model
        self.model = attempt_load(self.weights, map_location=self.device)  # load FP32 model
        self.stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(self.imgsz, s=self.stride)  # check image size
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names  # get class names
        
        if self.half:
            self.model.half()  # to FP16

        # Second-stage classifier
        self.classify = False
        self.modelc = None
        if self.classify:
            self.modelc = load_classifier(name='resnet50', n=2)  # initialize
            self.modelc.load_state_dict(torch.load('resnet50.pt', map_location=self.device)['model']).to(self.device).eval()
        self.detection_publisher = None
        if(self.publish):
            rospy.loginfo("Publish mode. initializing publisher.")
            self.detection_publisher = rospy.Publisher(self.detection_topic, DetectionMsg, queue_size=5)

        # Dataloader -- the idea here is to fetch the data from a rostopic instead of using the dataloader classes.
        # Webcam seems to "collect" a few images until it hits a certain batch size, then continue on. Not sure if that's what you want to do.
        '''
        if webcam:
            view_img = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=imgsz, stride=stride)
            bs = len(dataset)  # batch_size
        else:
            dataset = LoadImages(source, img_size=imgsz, stride=stride)
            bs = 1  # batch_size
        '''

        bs = 1
        self.vid_path, self.vid_writer = [None] * bs, [None] * bs

        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once
        # Might need to do some tuning on the queue_size to get 
        self.img_subscriber = message_filters.Subscriber(self.source_topic, Image, queue_size=5, buff_size=2**24)
        self.depth_subscriber = message_filters.Subscriber(self.depth_topic, Image, queue_size=5, buff_size=2**24)
        #self.ts = message_filters.TimeSynchronizer([self.img_subscriber, self.depth_subscriber], 1)
        self.ts = message_filters.ApproximateTimeSynchronizer([self.img_subscriber, self.depth_subscriber], queue_size=1, slop=0.1)
        self.ts.registerCallback(self.detector_callback)
        # image seems to be at about 30MB/s, so set the buffer to be able to handle twice that.
        rospy.loginfo("Initialised the image subscriber successfully!")
    
    @torch.no_grad()
    def detector_callback(self, rgb_data, depth_data):
        
        if(self.detection_count > 100):
            rospy.loginfo("Stopping detection to avoid overloading")
            exit()
            return

        # first, convert the obtained Image data to a cv2-type format, which is just a numpy array.
        depth_data.encoding = "mono16"
        img = self.bridge.imgmsg_to_cv2(rgb_data, "bgr8")
        depth_img = self.bridge.imgmsg_to_cv2(depth_data, "mono16")
        
        # Then we do some steps that are included in the datasets.py's LoadImage class, before proceeding with rest of the func.
        img0 = img.copy()	
        img = letterbox(img, self.imgsz, stride=self.stride)[0]
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # we don't really need to deal with paths right now, I suppose
        path = ''

        # Inference
        t1 = time_synchronized()
        pred = self.model(img,
                     augment=self.augment,
                     visualize= False)[0]

        # Apply NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)
        t2 = time_synchronized()

        # Apply Classifier
        if self.classify:
            pred = apply_classifier(pred, self.modelc, img, img0)
	
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            #if webcam:  # batch_size >= 1
            #    p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
            #else:
            #    p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)
            p, s, im0, frame = f'detection_image_{self.detection_count}_{i}', '', img0.copy(), 0
            p = Path(p)  # to Path
            save_path = str(self.save_dir / p.name) + '.png'  # img.jpg
            depth_save_path = str(self.save_dir / p.name) + '_depth.png'
            txt_path = str(self.save_dir / 'labels' / p.stem) + (f'_{frame}')  # img.txt

            if(i == 0):
                self.detection_count += 1
            
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if self.save_crop else im0  # for save_crop
            msg = DetectionMsg()
            msg.rgb_image = rgb_data
            msg.depth_image = depth_data
            msg.detection_count = len(det)
            msg.detection_array = []
            # we want to publish the topic even if there is no detection.
            if len(det):
                
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if self.publish:
                        tempArr = DetectionArray()
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                        tempArr.detection_info = np.asarray((cls, *xywh)).astype(np.float)
                        msg.detection_array.append(tempArr)
                        
                    if self.save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if self.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if self.save_img or self.save_crop or self.view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if self.hide_labels else (self.names[c] if self.hide_conf else f'{self.names[c]} {conf:.2f}')
                        plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=self.line_thickness)
                        if self.save_crop:
                            save_one_box(xyxy, imc, file=self.save_dir / 'crops' / self.names[c] / f'{p.stem}.jpg', BGR=True)
            if self.publish:
                print("Publishing message")
                self.detection_publisher.publish(msg)
                
            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')
            time.sleep(0.5)

            # Stream results
            if self.view_img:
                cv2.imshow('rgbimg', im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if self.save_img:
                cv2.imwrite(save_path, im0)          
                cv2.imwrite(depth_save_path, depth_img)   
'''                
                if False: # don't want anything to do with datasets here
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if self.vid_path[i] != save_path:  # new video
                        self.vid_path[i] = save_path
                        if isinstance(self.vid_writer[i], cv2.VideoWriter):
                            self.vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        self.vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    self.vid_writer[i].write(im0)
            
'''
            
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source_topic', type=str, default='/camera/color/image_raw', help='name of the RGB camera topic. See rostopic list for available topics.')
    parser.add_argument('--camera_topic', type=str, default='/camera/color/camera_info', help='name of the camera info topic. See rostopic list for available topics.')
    parser.add_argument('--depth_topic', type=str, default='/camera/aligned_depth_to_color/image_raw', help='name of the depth camera topic.')
    parser.add_argument('--depth_camera_topic', type=str, default='/camera/aligned_depth_to_color/camera_info', help='name of the depth camera info topic. See rostopic list for available topics.')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--detection_topic', type=str, default='/yolov5/detection', help='name of the DetectionMsg topic to publish to.')
    parser.add_argument('--publish', action='store_true', help='Enable publishing detection to the topic defined in detection_topic')
    
    opt = parser.parse_args()
    return opt


#def main(opt):
def main():
    #print(colorstr('detect: ') + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))
    check_requirements(exclude=('tensorboard', 'thop'))
    rospy.init_node("YOLOv5_ros_node", anonymous=True)
    detect_node = yolov5_node()#(**vars(opt))
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")


if __name__ == "__main__":
    #opt = parse_opt()
    #print(opt)
    #main(opt)
    main()

