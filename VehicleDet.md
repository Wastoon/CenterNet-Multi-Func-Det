# VehicleDet by CenterNet

## 1. Introduction

[CenterNet](https://github.com/xingyizhou/CenterNet)作为无锚目标检测模型，将对象表示为可缩放的高斯模板，并使用多任务框架回归到对象的其他已定义属性（例如对象的长度和宽度） ，对象边缘和中心之间的偏移量等）。 与基于区域提议的两级检测器相比，自下而上的框架具有明显更高的推理速度，它也可以被视为对象建模单级检测器的极限形式。 但是，与两阶段检测器相比，centerNet在准确性方面仍处于劣势。 有关更多详细信息，您可以阅读原始论文：[Object as point](http://arxiv.org/abs/1904.07850)。

## 2. Task Background

停车场车辆检测任务共包含了6个大类的检测对象，分别为car, suv, van, other, truck, bus。当考虑车体颜色后，一共扩展出共计31个类别的检测目标，分别为：

+ 1：car_white
+ 2：suv_red
+ 3：suv_white
+ 4：car_black
+ 5：car_red
+ 6：suv_black
+ 7：van_white
+ 8：suv_yellow
+ 9：car_yellow
+ 10：car_blue
+ 11：car_unknown
+ 12：other
+ 13：truck_white
+ 14：van_black
+ 15：bus_green
+ 16：truck_blue
+ 17：bus_black
+ 18：bus_white
+ 19：truck_red
+ 20：car_green
+ 21：truck_green
+ 22：truck_black
+ 23：truck_unknown
+ 24：suv_unknown
+ 25：van_yellow
+ 26：truck_yellow
+ 27：van_unknown
+ 28：van_green
+ 29：van_blue
+ 30：car_silver_gray
+ 31：bus_yellow

## 3. How to use?

### 3.1 Environment Setup

代码在Ubuntu16.04，GPUs=1080ti上通过测试

#### Step1: 创建conda虚拟环境

```
conda create -n vehicle_det python=3.7
```

#### Step2: 激活新建的conda环境

```
source activate vehicle_det
```

#### Step3: 安装所需pytorch版本（使用了pytorch1.4.0）

高于1.0.0版本的pytorch都可以，下文中使用到的可变形卷积（DCN）适配的pytorch版本高于1.0.0。如果安装了0.4.0的pytorch，需要切换DCN的git分支来适配低版本的pyotrch。

```
conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch
```

#### Step 4: 安装COCOAPI

因为训练数据集的标注格式为COCO格式，所以需要安装cocoapi套件。选择一个路径 PATH=/---你的路径--/cocoapi

```
###COCOAPI = /---path you choose--/cocoapi
cd /---path you choose--/
mkdir cocoapi
cd cocoapi
git clone https://github.com/cocodataset/cocoapi.git
cd PythonAPI
make
python setup.py install --user
```

#### Step 5: 克隆这个仓库并切换至vehicle_det分支:

```
####CenterNet_path = /---path you choose--/centernet
cd /---path you choose--/
mkdir centernet
cd centernet
git clone https://github.com/Wastoon/CenterNet-Multi-Func-Det.git
cd CenterNet-Multi-Func-Det
git checkout vehicle_det
```

#### Step 6: 安装几个依赖包

```
pip install -r requirements.txt
```

#### Step 7: 编译DCN[可选部分]

如果你要用到使用了DCN的backbone，就需要进行第七步，否则跳过这一部分。在进行这一步时，默认你已经安装了高于1.0.0版本的pytorch。

```
cd /src/lib/models/networks
git clone git@github.com:CharlesShang/DCNv2.git
cd DCNv2  ###make sure you are on the master branch
./make.sh         # build
python testcpu.py    # run examples and gradient check on cpu
python testcuda.py   # run examples and gradient check on gpu
```

#### Step 8: 编译NMS[可选部分]

如果你要用到了多尺度的测试，使用NMS来过滤结果。

```
cd $centernet_root_path/src/lib/external
make
```

### 3.2 How to train model?

#### 准备数据集

本仓库使用的是COCO格式的训练标注，但原始标注工具提供的是`.xml`格式的标注，需要借助[R-CenterNet](https://github.com/ZeroE04/R-CenterNet)中的格式转换脚本[`voc2coco.py`](https://github.com/ZeroE04/R-CenterNet/blob/master/labelGenerator/voc2coco.py)将标注转换为COCO格式，详情见[R-CenterNet](https://github.com/ZeroE04/R-CenterNet)。

#### 训练命令示例：

```
cd $centernet_root_path
python src/main.py --task vehicle_det \
									 --dataset ALLVehicle \
									 --gpus 0 \
									 --arch dla_34 \
									 --exp_id dla34_all_vehicle
									 --load_model ''
									 --aug_rot 0 \
									 --rotate 0 \
									 --flip 0.5 \
									 --batch_size 4 \
                   --num_epochs 20 \
                   --debug 0
```

+ `--task`为必填项，停车场车辆检测使用`vehicle_det`。
+ `--dataset`为必填项，选择车辆检测任务的训练数据集`ALLVehicle`。
+ `--gpus`为可选项，指定训练使用的GPU设备号。
+ `--arch`选择训练使用的模型backbone，可选项有`res_18`，`res_101`，`resdcn_18`，`resdcn_101`，`dlav0_34`，`dla_34`，`hourglass`。
+ `--exp_id `为训练数据保存子文件夹，完整的训练数据保存地址为`$centernet_root_path\exp\vehicle_det\dla34_all_vehicle`。
+ `--load_model`为必填项，训练开始时指定为''。训练中断后，使用`--resume`断点加载模型，`load_model`为断点模型地址。
+ `--aug_rot`为训练过程中使用旋转增强的概率阈值，当旋转概率低于阈值时进行随机旋转增强，旋转范围为$\pm$rotate。训练过程中建议的旋转角度不要超过正负5度，否则会影响模型对于边界框WH的预测。
+ `--flip`为训练过程中使用水平翻转增强的概率宇宙，当翻转概率低于阈值时进行翻转。
+ `--batch_size`为训练样本批次大小。
+ `--num_epochs`为训练迭代周期次数。
+ `--debug`控制训练过程的样本可视化，当`debug`为2时，训练batch自动调整为1，会随训练进行可视化出模型输入，ground_truth，以及模型的全部输出，当`debug`为0时，不随训练显示模型输入输出。

训练过程损失可视化：

```
tensorboard --logdir $centernet_root_path\exp\vehicle_det\dla34_all_vehicle
```

### 3.3 Inference demo

demo命令示例：

```
video_path = /--视频文件夹--/2020-12-08_143641_245.avi
python src/demo.py --gpus 0\
									 --demo $video_path \
									 --load_model $your_model_path \
									 --task vehicle_det \
									 --dataset ALLVehicle \
									 --debug 5 \
									 --arch dla_34 \
									 --vis_thresh 0.3 \
									 --center_thresh 0.3 \ 
									 --show_label \
									 --test_scales "0.5, 0.6, 0.7, 0.8,1.0, 1.1, 1.2,1.3,1.4,1.5" \
									 --nms \
									 --output_video_demo $your_savepath/2020-12-08_143641_245.avi
```

+ `--vis_thresh`为经过多尺度检测后，整合所有的检测结果完成后，要显示出来的最后筛选阈值。
+ `--center_thresh`为单个尺度下，检测到目标的筛选阈值。
+ `--test_scales`为测试时的图像放大倍数。经过scales的放大倍数后，会裁剪出$1920\times1440$的图像作为输入。
+ `--nms`对多尺度检测后的结果进行soft-nms。
+ `--show_label`为在视频流输出结果中显示检测出车辆的类别标签和置信度得分。
+ `--load_model`为测试模型的地址。
+ `--output_video_demo`为输出视频的保存地址。当不指定`output_video_demo`参数，即不指定视频保存位置时，测试期间显示器会显示实时的车辆检测结果。

==模型输出格式==：

`detector.run(image)`的输出为一个字典，如下：

```
ret = detector.run(image)
ret:{
			'results':{
								1:np.array(shape=N1x6)
								2:np.array(shape=N2x6)
								3:np.array(shape=N3x6)
								4:np.array(shape=N4x6)
								5:np.array(shape=N5x6)
								6:np.array(shape=N6x6)
								7:np.array(shape=N7x6)
								...
								30:np.array(shape=N29x6)
								31:np.array(shape=N31x6)
			}
			'tot':处理一张图全部用时，
			'load':加载图片用时，
			'pre':预处理图片用时，
			'net':网络前向用时，
			'merge':多个尺度下结果融合用时，
			'image_name':测试图片名称，
			'vis_img'：最终可视化图像numpy输出
}
```

在`ret`里面，`N1,N2,N3,N4,...,N31`分别代表第`i`个类别下车辆的数目；每辆车的共包含长度为6的信息量，含义分别为`[x1, y1, x2, y2, angle, prob]`，其中用到的是前4个表示边界框的坐标量和最后一个表示类别的置信度。

## 4. Model Zoo

在测试过程中输入的分辨率会影响检测速度，一定程度上会影响精度，因此提供了两种分辨率下训练的模型，更改测试分辨率的命令为：

```
video_path = /--视频文件夹--/2020-12-08_143641_245.avi
python src/demo.py --gpus 0\
									 --demo $video_path \
									 --load_model $your_model_path \
									 --task vehicle_det \
									 --dataset ALLVehicle \
									 --debug 5 \
									 --arch dla_34 \
									 --vis_thresh 0.1 \
									 --center_thresh 0.1 \ 
									 --show_label \
									 --test_scales "0.5, 0.6, 0.7, 0.8,1.0, 1.1, 1.2,1.3,1.4,1.5" \
									 --nms \
									 --output_video_demo $your_savepath/2020-12-08_143641_245.avi \
									 --test_resolution "1920,1440"
```

#### Resdcn18（建议测试时将图片的分辨率调整为960x960）

| 模型代数 | 模型                                                         |
| -------- | ------------------------------------------------------------ |
| 5        | [Resdcn_18_model_5](https://drive.google.com/file/d/1mzq8GMZPhfMn3LBdP7YJeezbqE4DcClO/view?usp=sharing) |
| 20       | [Resdcn_18_model_20](https://drive.google.com/file/d/15gzzwCwS-F1ay13wRAWgr5cz1P182BIZ/view?usp=sharing) |
| 50       | [Resdcn_18_model_50](https://drive.google.com/file/d/1yDjNcT54nUGg-Zp20McDoRSpNJV5hyqq/view?usp=sharing) |
| 100      | [Resdcn_18_model_100](https://drive.google.com/file/d/1N2jBS2TNdu2i6TTQ6pZqfi26-U8x1vad/view?usp=sharing) |

#### DLA34（建议测试时将图片的分辨率调整为960x960）

| 模型代数 | 模型                                                         |
| -------- | ------------------------------------------------------------ |
| 2        | **[DLA34_model_2](https://drive.google.com/file/d/1GFLE86BW6hdJTVW6NZz79Gqy3g08V8TP/view?usp=sharing)** |
| 5        | **[DLA34_model_5](https://drive.google.com/file/d/1V7wzqRlNw8iXXh3EIOVrzfsbJV5YMo2A/view?usp=sharing)** |
| 10       | [DLA34_model_10](https://drive.google.com/file/d/1ffrgNVSy32GYLbIACcM_N0leMz912Vmy/view?usp=sharing) |
| 20       | [DLA34_model_20](https://drive.google.com/file/d/1VTOi8onVOlvIGEejIzlas0_ukDjfw3dh/view?usp=sharing) |
| 50       | [DLA34_model_50](https://drive.google.com/file/d/1mzq8GMZPhfMn3LBdP7YJeezbqE4DcClO/view?usp=sharing) |

#### DLA34（建议测试时将图片的分辨率调整为1920x1440）

| 模型代数 | 模型                                                         |
| -------- | ------------------------------------------------------------ |
| 1        | [DLA34_model_1](https://drive.google.com/file/d/1hi-EKU7oCARiyCpz3V2DKC86xKW72bXB/view?usp=sharing) |
| 5        | [DLA34_model_5](https://drive.google.com/file/d/1rcDhIS3-7DpnksSg8Kol-eXxx-_qKI1H/view?usp=sharing) |
| 10       | [DLA34_model_10](https://drive.google.com/file/d/1HrO8kJ1Ck06bapb8PZEswXzgc35tddrJ/view?usp=sharing) |
| 20       | **[DLA34_model_20](https://drive.google.com/file/d/1heSTVCeC8rxfJvQL2Z5a6Ec7cggB800l/view?usp=sharing)** |
| 50       | [DLA34_model_50](https://drive.google.com/file/d/1heSTVCeC8rxfJvQL2Z5a6Ec7cggB800l/view?usp=sharing) |

表格中加粗的模型是在测试视频中检出效果较好的模型。

