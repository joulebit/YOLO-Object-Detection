# YOLO-Object-Detection
## Motivation
This project was started early in desember to get an understanding of the You Only Look Once object detection technique and implementation. As well as to get a foundational code to build other applicable computer vision programs on top of in the future.

## Installation
Because of GitHub's limit on file sizes, the yolov3 configurations and weights could not be added to the repo, so they need to be downloaded from elsewhere on the internet, [here](https://pjreddie.com/darknet/yolo/) for example and put in the same folder as the project.  
Clone the repository
```
git clone https://github.com/joulebit/YOLO-Object-Detection.git
```
Navigate into the project folder
```
cd YOLO-Object-Detection
```
Install the dependencies
```
pip3 install -r requirements.txt
```
Add the missing yolov3 files as mentioned above, and add your videos and images in their respective folders, and then you should be able to run the commands
```
py .\yolo.py --image images/dashcam.jpg --output output/dashcam.jpg

py .\yolo_vid.py --input videos/car_chase.mp4 --output output/car_chase.avi
```

## Results
Hardware used was: 
Intel(R) Core(TM) i5-4210U CPU @ 1.70ghz, with 8.00 GB of RAM  
Average processing time was 2 seconds per frame.  
To achieve REAL Time object detection, one would need to have an Nvidia card and use CUDA as [this](https://github.com/pjreddie/darknet/wiki/YOLO:-Real-Time-Object-Detection) for example.  

Click the videos below to see examples of the output  
[![Watch the video](https://img.youtube.com/vi/rnOFH0mKXPQ/hqdefault.jpg)](https://www.youtube.com/watch?v=rnOFH0mKXPQ)
[![Watch the video](https://img.youtube.com/vi/qroXchzbG-g/hqdefault.jpg)](https://www.youtube.com/watch?v=qroXchzbG-g)
[![Watch the video](https://img.youtube.com/vi/kpcbC94QhrY/hqdefault.jpg)](https://www.youtube.com/watch?v=kpcbC94QhrY)
[![Watch the video](https://img.youtube.com/vi/2fqsZcPlq4c/hqdefault.jpg)](https://www.youtube.com/watch?v=2fqsZcPlq4c)
[![Watch the video](https://img.youtube.com/vi/0FioCeH1L00/hqdefault.jpg)](https://www.youtube.com/watch?v=0FioCeH1L00)


## Additions for future projects
#### Adding your own objects
Adding custom objects as described [here](https://medium.com/@manivannan_data/how-to-train-yolov3-to-detect-custom-objects-ccbcafeb13d2). But it requires a good dataset and preferably a GPU.

## Acknowledgments
* You Only Look Once:Unified, Real-Time Object Detection. [Research paper by Joseph Redmon and others.](https://pjreddie.com/media/files/papers/yolo.pdf)
* Implementation example made by Adrian Rosebrock over at [pyimagesearch](https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/)
