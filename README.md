## libtorch_yolov3 for windows
### Background
Using the pytorch's c++ interface libtorch to complete yolov3 deployment, the pytorch version of yolov3 can get from [yolov3](https://github.com/ultralytics/yolov3.git), The model used in this code is yolov3-spp.

### Install
1.Install vs2015 or higher version
2.Install CMake(3.12 or higher)
3.Download the libtorch1.2.0, you can get from
[libtorch1.2.0](https://pan.baidu.com/s/1Ap-OMf8qSNtGwrUy2dGVkg )
提取码：cf7v 
Unzip the compressed package and write /path/to/libtorch1.2.0/libtorch-win-shared-with-deps-1.2.0/libtorch to the Windows PATH.

4.Download the opencv, you can get from
[opencv](https://pan.baidu.com/s/1u5jdKE-RvV910_ATBAQiUQ)
提取码：pu2n
Unzip the compressed package and write /path/to/opencv/build/x64/vc14/bin, /path/to/opencv/build to the Windows PATH. 
Then create system variables OpenCV_DIR, the value is /path/to/opencv/build.

5.Download the code
```
git clone https://github.com/1348722633/libtorch_yolov3.git
mkdir model
```
Then download the mode from 
[yolov3_model](https://pan.baidu.com/s/1BnpsyUSqiN1mYgAM0UWlpA)
提取码：wl70 
unzip the model file to the directory named model.
```
mkdir build
cd build 
cmake .. -G"Visual Studio 14 2015 Win64"
```
6.Then open the .sln file and Generate release version.
