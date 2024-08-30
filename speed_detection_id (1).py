import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel


from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
from collections import deque
import numpy as np
import timeit
palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
data_deque = {}
#-------50km--------
# cy1=382
# cy2=664
#----70km------------
cy1=397
cy2=677
#------- --------------
# cy1=234
# cy2=374
#------60__4----------
# cy1=345
# cy2=637
#-----80----------------
# cy1=456
# cy2=575
#------40----------------
# cy1=403
# cy2=702
#-------------30km------------
# cy1=536
# cy2=841
#--------------[79-80]km-------------
# cy1=461
# cy2=575

#Speed Varibals

l={}
l1={}
speed={}


#vehicles total counting variables
array_ids = []
counting = 0
modulo_counting = 0

#Tracking vehicles
vehicles_entering = {}
vehicles_elapsed_time = {}
distance = 30 #meters (assumption)



##########################################################################################
def xyxy_to_xywh(*xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h





def draw_boxes(img, bbox, identities=None, categories=None, names=None, offset=(0, 0),j=None):
    # for key in list(data_deque):
    #   if key not in identities:
    #     data_deque.pop(key)
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        cat = int(categories[i]) if categories is not None else 0
        id = int(identities[i]) if identities is not None else 0
        # color = compute_color_for_labels(id)
        data = (int((box[0]+box[2])/2),(int((box[1]+box[3])/2)))
        label = str(id) + ":"+ names[cat]
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        #cv2.rectangle(img, (x1, y1), (x2, y2), (255,144,30), 1)
        #cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), (255,144,30), -1)
        #cv2.putText(img, label, (x1, y1 - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.6, [255, 255, 255], 1)
        # cv2.circle(img, data, 6, color,-1)
        
        #c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
        midpoint_x = x1+((x2-x1)/2)
        midpoint_y = y1+((y2-y1)/2)
        center_point = (int(midpoint_x),int(midpoint_y))
        midpoint_color = (0,255,0)
        off=2
        

        #check if object on in the start_area box
        # in_on_start_area = cv2.pointPolygonTest(np.array(start_area,np.int32),(int(midpoint_x),int(midpoint_y)),False)
        cv2.circle(img,center_point,radius=1,color=(0,0,255),thickness=2)

        if midpoint_y<(cy1)   :
            l[id]=j
            
                   
        if id in l:
            if midpoint_y>cy1 and midpoint_y<cy2  :
                
                l1[id]=j
                E=(l1[id]-l[id])/30
                if E>0 :
                    speed[id]=(30/E)*3.6
                    sp=speed[id]
                    print(speed[id])
                    print(id)
                    # cv2.rectangle(img, (x1, y1), (x2, y2), (0,0,255), 1)
                    # cv2.rectangle(img, (x1, y1 - 20), (x1 + w + 100, y1), (255,144,30), -1)
                    # cv2.putText(img, "Speed "+str(int(speed[id]))+" km/h", (x1, y1 - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.6, [255, 255, 255], 1)
            if midpoint_y >cy2 :   
                cv2.rectangle(img, (x1, y1), (x2, y2), (0,0,255), 1)
                #cv2.circle(img,center_point,radius=1,color=(0,0,255),thickness=2)

                cv2.rectangle(img, (x1, y1 - 20), (x1 + w + 100, y1), (255,144,30), -1)
                cv2.putText(img, "Speed "+str(int(speed[id]))+" km/h", (x1, y1 - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.6, [255, 255, 255], 1)     



        # if cy1<(midpoint_y+off) and cy1>(midpoint_y-off) :
            
        #     cv2.circle(img,center_point,radius=1,color=(255,0,),thickness=2)
        #     # vehicles_entering[id] = time.time()
        #     # vehicles_entering[id] = datetime.datetime.now()
        #     vehicles_entering[id] = timeit.default_timer()
        #     Te=time.time()
        #     re=time.localtime(Te)
            
        #     #cv2.putText(img, " first Time "+str(re)+" s", (x1, y1 - 30),cv2.FONT_HERSHEY_SIMPLEX, 0.6, [0, 0, 55], 1)
      


        # if id in vehicles_entering:
            

        #     # if cy2<(midpoint_y+off) and cy2>(midpoint_y-off):
        #     if midpoint_y >cy2 :    
        #         # elapsed_time = time.time() - vehicles_entering[id]
        #         # elapsed_time = datetime.datetime.now() - vehicles_entering[id]
        #         elapsed_time = timeit.default_timer() - vehicles_entering[id]
        #         T=timeit.default_timer()
               
        #         #cv2.putText(img, "last Time "+str(T)+" s", (x1, y1 - 50),cv2.FONT_HERSHEY_SIMPLEX, 0.6, [0, 0, 55], 1)
                
        #         # if id not in vehicles_elapsed_time:
        #         #     vehicles_elapsed_time[id] = elapsed_time

        #         # if id in vehicles_elapsed_time:
        #         #     elapsed_time = vehicles_elapsed_time[id]
              
                


        #         # a_speed_ms = distance /elapsed_time
        #         # a_speed_kh = a_speed_ms * 3.6

        #         #@@@cv2.rectangle(img, (x1, y1), (x2, y2), (0,0,255), 1)
        #         #cv2.circle(img,center_point,radius=1,color=(0,0,255),thickness=2)

        #         #@@@cv2.rectangle(img, (x1, y1 - 20), (x1 + w + 100, y1), (255,144,30), -1)
        #         #@@@cv2.putText(img, "Speed "+str(int(speed[id]))+" km/h", (x1, y1 - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.6, [255, 255, 255], 1)
        #         #cv2.putText(img, "Elapsed Time "+str(elapsed_time) +" s", (x1, y1 - 30),cv2.FONT_HERSHEY_SIMPLEX, 0.6, [0, 0, 55], 1)
                
                

                
                # cv2.circle(img, data, 6, color,-1)
            '''
            else:
                cv2.rectangle(img, (x1, y1), (x2, y2), (255,144,30), 1)
                #cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), (255,144,30), -1)
                cv2.putText(img, label, (x1, y1 - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.6, [255, 255, 255], 1)
                # cv2.circle(img, data, 6, color,-1)
            '''

        # if (midpoint_x > area1_pointA[0] and midpoint_x < area1_pointD[0]) and (midpoint_y > area1_pointA[1] and midpoint_y < area1_pointD[1]):
            
        #     midpoint_color = (0,0,255)
        #     print('Kategori : '+str(cat))
            
        #     #add vehicles counting
        #     if len(array_ids) > 0:
        #         if id not in array_ids:
        #             array_ids.append(id)
        #     else:
        #         array_ids.append(id)
            
        #cv2.circle(img,center_point,radius=1,color=(0,0,255),thickness=2)
        #cv2.rectangle(img, (x1, y1), (x2, y2), (0,0,255), 1)  
        

        
    return img






def load_classes(path):
    # Loads *.names file at 'path'
    with open(path, 'r') as f:
        names = f.read().split('\n')
    return list(filter(None, names)) 
def detect(save_img=False):
    names, source, weights, view_img, save_txt, imgsz, trace = opt.names, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace 
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    # initialize deepsort
    cfg_deep = get_config()
    cfg_deep.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")
    deepsort = DeepSort(cfg_deep.DEEPSORT.REID_CKPT,
                        max_dist=cfg_deep.DEEPSORT.MAX_DIST, min_confidence=cfg_deep.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg_deep.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg_deep.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg_deep.DEEPSORT.MAX_AGE, n_init=cfg_deep.DEEPSORT.N_INIT, nn_budget=cfg_deep.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = load_classes(names)
    #colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    j=0
    
 

    for path, img, im0s, vid_cap in dataset:
        j=j+1
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string
                xywh_bboxs = []
                confs = []
                oids = []
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    x_c, y_c, bbox_w, bbox_h = xyxy_to_xywh(*xyxy)
                    xywh_obj = [x_c, y_c, bbox_w, bbox_h]
                    xywh_bboxs.append(xywh_obj)
                    confs.append([conf.item()])
                    oids.append(int(cls))
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    #if save_img or view_img:  # Add bbox to image
                        #label = f'{names[int(cls)]} {conf:.2f}'
                        #plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)
                xywhs = torch.Tensor(xywh_bboxs)
                confss = torch.Tensor(confs)
                
                outputs = deepsort.update(xywhs, confss, oids, im0)
                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -2]
                    categories = outputs[:, -1]
                    print(bbox_xyxy)
                   
                    
                        

                    draw_boxes(im0, bbox_xyxy,identities,categories,names,j=j)
            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')
            # #-----50km_Trim---------------------------------------------------------
            # cv2.line(im0,(895,382),(1150,382),(0,255,0),2)
            # cv2.line(im0,(740,664),(1150,664),(0,255,0),2)
            # cv2.putText(im0, 'Line1 ', (815,382), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,2), 2, cv2.LINE_AA)
            # cv2.putText(im0, 'Line2', (660,664), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,2), 2, cv2.LINE_AA)
            #-----70km_Trim-----------------------------------------------------------
            cv2.line(im0,(745,397),(1024,397),(0,255,0),2)
            cv2.line(im0,(600,677),(1030,677),(0,255,0),2)
            cv2.putText(im0, 'Line1 ', (640,397), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,2), 2, cv2.LINE_AA)
            cv2.putText(im0, 'Line2', (510,677), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,2), 2, cv2.LINE_AA)


            #-----  -----------------------------------------------------------
            # cv2.line(im0,(639,234),(913,234),(0,255,0),2)
            # cv2.line(im0,(428,374),(888,374),(0,255,0),2)
            # cv2.putText(im0, 'Line1 ', (580,234), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,2), 2, cv2.LINE_AA)
            # cv2.putText(im0, 'Line2', (370,374), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,2), 2, cv2.LINE_AA)

            #-----60__4-----------------------------------------------------------
            
            # cv2.line(im0,(870,350),(1116,350),(0,255,0),2)
            # cv2.line(im0,(695,630),(1116,630),(0,255,0),2)
            # cv2.putText(im0, 'Line1 ', (780,350), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,2), 2, cv2.LINE_AA)
            # cv2.putText(im0, 'Line2', (600,630), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,2), 2, cv2.LINE_AA)
            #--------80---------------------------------------------------------------------------
            
            # cv2.line(im0,(1042,456),(1237,456),(0,255,0),2)
            # cv2.line(im0,(988,575),(1247,575),(0,255,0),2)
            # cv2.putText(im0, 'Line1 ', (950,456), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,2), 2, cv2.LINE_AA)
            # cv2.putText(im0, 'Line2', (900,575), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,2), 2, cv2.LINE_AA)
            #-----------40 ------------------------------------------------------
            # cv2.line(im0,(876,403),(1140,403),(0,255,0),2)
            # cv2.line(im0,(732,702),(1140,702),(0,255,0),2)
            # cv2.putText(im0, 'Line1 ', (780,403), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,2), 2, cv2.LINE_AA)
            # cv2.putText(im0, 'Line2', (630,702), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,2), 2, cv2.LINE_AA)
            #----------------30km------------------------------------------------
            # cv2.line(im0,(660,536),(914,536),(0,255,0),2)
            # cv2.line(im0,(416,841),(875,841),(0,255,0),2)
            # cv2.putText(im0, 'Line1 ', (570,536), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,2), 2, cv2.LINE_AA)
            # cv2.putText(im0, 'Line2', (320,841), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,2), 2, cv2.LINE_AA)
            #-------------[79_80]km---------------------------------------------------
            # cv2.line(im0,(927,461),(1100,461),(0,255,0),2)
            # cv2.line(im0,(879,575),(1117,575),(0,255,0),2)
            # cv2.putText(im0, 'Line1 ', (830,461), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,2), 2, cv2.LINE_AA)
            # cv2.putText(im0, 'Line2', (800,575), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,2), 2, cv2.LINE_AA)







            #cv2.putText(im0, 'speed = '+str( speed), (200,500), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,2), 2, cv2.LINE_AA)
            

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')
    print(l)
    print(l1)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--names', type=str, default='data/coco.names', help='*.cfg path')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    parser.add_argument('--trailslen', type=int, default=64, help='trails size (new parameter)')
    opt = parser.parse_args()
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
