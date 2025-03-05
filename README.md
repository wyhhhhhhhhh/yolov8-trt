# YOLOV8-OBB使用说明

## 数据预处理

### step1

首先通过标注我们有原始的图片文件和标注的json文件，我们将jpg文件整理到一个images文件夹下，json标注文件整理到annotations文件夹下，然后对其进行ROI裁剪，目标尺寸为640*480(也可以在crop.py里面进行修改)，然后运行datasets/scripts路径下的**crop.py**文件，将对原始图片裁剪到目标尺寸，同时也对标注文件进行了同等裁剪，输出的处理后的数据分别存在了output_images下和output_annotations下；之后需要将json文件转成yolo训练所需的txt标注文件，执行datasets/scripts路径下的**json2txt.py**便得到了DOTAV1格式的txt文件,这个时候就可以得到txt文件，保存在了**croped_txt文件夹**下。

### step2

将处理好的数据放入datasets下，图片放入images下，标注txt放入labels下，要注意这个txt还需要进行处理，所以应该放入**labels/train_original**下，之后回到主目录，运行**pre_data.py**文件，将DOTAV1格式的标注txt文件转成yolo训练所需要的txt格式，新的txt文件也将生成在**labels/train**文件夹下。到此，数据预处理完成。

在datasets/scripts路径下还有两个可视化脚本，分别是pre_view.py和view.py，用于**可视化**DOTAV1格式的txt标注和yolo格式的txt标注，也可以在转换前后分别执行一次，可视化检查标注是否有误。其中python **pre_view.py**用的是**labels/train_original**下的txt文件，即**DOTAV1格式**的txt；**python view.py**用的是**labels/train**下的txt文件，即**yolo格式**的txt。

## 训练

打开yolo.py,在datasets路径下新建一个**real_data.yaml**来加载自己的数据，修改为自己的数据文件路径即可。

然后执行**python yolo.py**即可开始训练。

## 预测

输入待预测的图片路径即可，执行**python yolo_predict.py**完成预测。
## TensorRT加速推理

位于obb_detection_trt.py文件的主要内容，里面的代码有些杂乱，是做项目时额外的部分，如果自己使用，需要自己进行提取主干代码，关于TensorRT的环境配置部分，请移步我的博客
https://blog.csdn.net/qq_46454669/article/details/145777359