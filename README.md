# Vehicle-Detection-Tracking-and-Speed-Estimation-Based-on-Deep-Learning-Yolov7
model consists of three main phases: Detection phase, tracking phase, and speed calculation phase:
We employ the YOLOV7 for detection,and for tracking we use the DeepSORT algorithm to assign IDs 
for vehicles and track them.

To calculate the  speed of moving objects, we need the distance between two points and the duration. So to estimate speed using our proposed system, we need to know the parameters:
Firstly, you should measure the distance on the ground.
 ![](images/Picture1.png)

Secondly, as for time, it depends on the number of frames on which the vehicle moves between the first and second line, so the time between line1 and line2 is calculated by the formula T= 𝑁/𝑓𝑏𝑠
Where N is the number of frames, and FPS (frame per second) is the frame rate of the video.

Now can calculate the speed by the formula:
speed=distance/T




## Steps to run the code 
### the work environment
Google clabe : you need to creat a new notebook.

- Clone  the repostory 
```
git clone https://github.com/WalidDa10/Vehicle-Detection-Tracking-and-Speed-Estimation-Based-on-Deep-Learning-Yolov7.git
```
- Install the dependecies
To install required dependencies run:
```
pip install -r requirements.txt
```
- For run 

Notes: You can download yolov7-tiny or yolov7 weights from YOLOV7 assets repository

https://github.com/WongKinYiu/yolov7.git
```
!python speed_detection_id.py --weights yolov7.pt  --img 640  --source Video_Trim3_Trim.mp4
```
###Results 
#### Vehicles Detection, Tracking
![](images/Picture2.png)

#### Vehicles Detection, Tracking, Speed-Estimation 
![](images/Picture3.png)
