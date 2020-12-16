



# Enhanced-CenterNet for Bottom-Up Computer Vision Task (Work Summary)

## 1. Introduction

[CenterNet](https://github.com/xingyizhou/CenterNet), as an anchor-free target detection model, represents the object as a scale-adaptive Gaussian template, and uses a multi-task framework to regress to other defined properties of the object (such as the length and width of the object, the offset between the edge of the object and center, etc. ). Compared with the two-stage detector based on region proposal, the bottom-up framework has a significantly higher inference speed, and it can also be regarded as the limit form of the single-stage detector for object modeling. However, centerNet is still at a disadvantage in terms of map accuracy compared with the two-stage detector. For more details, you can read the origin paper: [Object as point.](http://arxiv.org/abs/1904.07850)

## 2. Abstract

Based on the original centerNet using a single Gaussian template to represent objects, this repo uses an adjustable number of Gaussian templates to fully cover a single object, which effectively improves the map of centerNet on COCO dataset. In addition, this repo is adapted to the deepfashion2 dataset for clothing keypoint detection, and this repo is also adapted to the 300W dataset for facial landmark detection.

### 3. Expand Task

* GridNeighbordet for General Object Detection

  经典的centerNet会回归出通道数为80（COCO数据集）或21（pascalVOC数据集）的中心点的heatmap。然而有时候某个物体的质心位置上并不包含或很少包含该物体的信息，因为这个物体的质量分布不均匀或者由于遮挡，使得物体中心点处包含了背景的信息甚至是其他物体的信息。因此我们学习了物体表面多个点的heatmap, 通过对检测结果的投票来提高检测的ap和ar。

  The classic centerNet will predict the heatmap of the center point with the number of channels 80 (COCO dataset) or 21 (pascalVOC dataset). However, sometimes the center of mass of an object does not contain or rarely contains the information of the object, because the mass distribution of the object is uneven or due to occlusion, which makes center of the object contain background information or even other object information . Therefore, I adjust the number of learned heatmap of singe object, and improved the detection ap and ar by voting on the detection results. It can be seen in the figure below that this problem exists in objects with uneven mass distribution and objects that are blocked.

  Therefore, we use multiple points on the target to represent the object, not just the center point. The specific representation method of the object is shown in the figure below:

  ![fig6](/Users/mry/Desktop/typora/500ppi/fig6.png)

  You can use the input parameter ==point_flags== in [How to use](#4. How to use?) to control the combination and number of heatmaps to be used for the object. There are a total of 9 locations of heatmaps that can be used. By matching the heatmaps of different locations, you can ensure that the heatmaps are on the object.

  ![fig4_1](/Users/mry/Desktop/typora/500ppi/fig4_1.png)

  ### Only one point heatmap

  ![fig2](/Users/mry/Desktop/typora/500ppi/fig2.png)

  **Object Detection on COCO validation**

  | Backbone      | AP/FPS     | Flip AP/FPS | Multi-scale AP/FPS |
  | ------------- | ---------- | ----------- | ------------------ |
  | Hourglass-104 | 40.3 / 14  | 42.2 / 7.8  | 45.1 / 1.4         |
  | DLA-34        | 37.4 / 52  | 39.2 / 28   | 41.7 / 4           |
  | ResNet-101    | 34.6 / 45  | 36.2 / 25   | 39.3 / 4           |
  | ResNet-18     | 28.1 / 142 | 30.0 / 71   | 33.2 / 12          |

  ### Three points heatmap

  ![fig1](/Users/mry/Desktop/typora/500ppi/fig1.png)

  ![fig4](/Users/mry/Desktop/typora/500ppi/fig4.png)

  **Object Detection on COCO validation**

  | Backbone      | AP/FPS    | Flip AP/FPS | Multi-scale AP/FPS |
  | ------------- | --------- | ----------- | ------------------ |
  | Hourglass-104 | 42.8 / 12 | 44.6 / 6.8  | 47.7 / 1.2         |

  ### More points heatmap

  ![fig3](/Users/mry/Desktop/typora/500ppi/fig3.png)

  **Object Detection on COCO validation**

  | Backbone      | AP/FPS   | Flip AP/FPS | Multi-scale AP/FPS |
  | ------------- | -------- | ----------- | ------------------ |
  | Hourglass-104 | 44.1 / 8 | 45.8 / 4.4  | 48.7 / 0.8         |

* Facial Landmark Detection

  Centernet非常适合面部关键点检测任务。人脸数据集300W中包含68个关键点，人脸数据集AFLW中包含19个关键点。我将任务分为两步走，第一步首先回归面部的中心点，然后将回归其余关键点与面部中心点的偏移量，通过后处理中坐标的移动得到面部关键点的坐标。除了使用这一种方法检测关键点，我还提供了对于面部68个关键点的heatmap的直接回归，通过回归map的方式一次性进行面部关键点的检测，然后在后处理过程中使用池化方式提取热图峰值，最终得到关键点的结果。当然，这两种方法使用了不同形式的监督信号，除了最后面的投影头不一样之外，模型的encoder和decoder结构都是相同的。

  CenterNet is very suitable for face key point detection tasks. The face datasets 300W and 300VW contain 68 facial landmarks, and the face dataset AFLW contains 19 facial landmarks. I divided the task into two steps. The first step is to predict the center of the face (define the nose tip as the center of the face, and the center  defined by different datasets are different), and then predict the offset of other landmarks to the center.Final facial landmark results are obtained by moving the coordinates in post-processing to obtain the coordinates of the facial landmark. In addition to using this method to detect facial landmark, I also provide a direct regression of the heatmap for the 68 facial landmark. The heatmap  of whole facial landmarks are predicted at one time through the regression map, and then pooling is used in the post-processing process to extract the peak value of the heatmap. Of course, these two methods use different forms of supervision signals. Except for the different projection heads, the encoder and decoder structures of the model are the same.

  ![fig7](/Users/mry/Desktop/typora/500ppi/fig7.png)

  **Facial Landmark Detection on 300W test Dataset**

  |             Method              | Common-NME(ION) | Common-Failture Rate | Common-Fr | Challenge-NME(ION) | Challenge-Failture Rate | Challenge-Fr | Full-NME(ION) | Full-Failture Rate | Full-Fr  |
  | :-----------------------------: | :-------------: | :------------------: | :-------: | :----------------: | :---------------------: | :----------: | :-----------: | :----------------: | :------: |
  |           heatmap+reg           |     3.6708      |       0.7233%        |  0.1805%  |       5.7647       |         4.2553%         |   30.3704%   |    3.9750     |      1.2365%       | 6.0958%  |
  |       heatmap+reg+rotate        |     3.5375      |          0           | 0.05415%  |       6.9251       |        14.7059%         |   24.444%    |    4.0666     |      2.2971%       | 5.2250%  |
  | heatmap+reg+reg+keep_resolution |     4.6417      |       1.4953%        |  3.4296%  |       6.9097       |         7.8152%         |   52.5926%   |    4.8841     |      2.1703%       | 13.0624% |

* DeepFashion2 Keypoint Detection

  ![fig8](/Users/mry/Desktop/typora/500ppi/fig8.png)

## 4. How to use?

### Environment Setup

The code was tested on Unbuntu16.04 server with 4 1080Ti GPUs.

#### Step 1: Create a new virtual environment with conda 

```
conda create -n centernet_bottomup python=3.7
```

#### Step 2: Activate the new environment

```
source activate centernet_bottomup
```

#### Step 3: Install pytorch needed

I use pytorch==1.4.0, maybe higher version can also be ok.

```
conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch
```

#### Step 4: Install COCOAPI

Choose a place yourself by convenient, and we define the PATH=/---path you choose--/cocoapi

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

#### Step 5: Clone this repo:

```
####CenterNet_path = /---path you choose--/centernet
cd /---path you choose--/
mkdir centernet
cd centernet
git clone https://github.com/Wastoon/CenterNet-Multi-Func-Det.git
cd CenterNet-Multi-Func-Det
```

#### Step 6: Install the requirements

```
pip install -r requirements.txt
```

#### Step 7: Compile deformable convolution

If you need deformable convolution, you can choose compile it in this way.

```
cd /src/lib/models/networks
git clone git@github.com:CharlesShang/DCNv2.git
cd DCNv2  ###make sure you are on the master branch
./make.sh         # build
python testcpu.py    # run examples and gradient check on cpu
python testcuda.py   # run examples and gradient check on gpu
```

### Use GridNeighbor-based Enhanced CenterNet

* **Model:** GridNeighbor-based Enhanced CenterNet are used to perform general object detection, which support COCO dataset and PascalVOC dataset under Hourglass Network. If you want to play with other network much lighter, you can modify the structure of Network just like what I have added in Houglass. 

* **DataSet Preparation**:

  Download the images (2017 Train, 2017 Val, 2017 Test) from [coco website](http://cocodataset.org/#download).

  Download annotation files (2017 train/val and test image info) from [coco website](http://cocodataset.org/#download). 

  ```
  ###coco_dataset_train_img_path = /---path you choose--/train2017
  ###coco_dataset_val_img_path = /---path you choose--/val2017
  ###coco_dataset_test_img_path = /---path you choose--/test2017
  ###coco_dataset_val_img_path = /---path you choose--/val2017
  ###coco_dataset_anno_path = /---path you choose--/annotations
  ###CenterNet_path = /---path you choose--/centernet
  
  cd $CenterNet_path
  mkdir data
  cd data
  mkdir coco
  cd coco
  ln -s $coco_dataset_train_img_path ./
  ln -s $coco_dataset_val_img_path ./
  ln -s coco_dataset_val_img_path ./
  ln -s coco_dataset_anno_path ./
  ```

  After this, your data folder just like this:

  ```
  ${CenterNet_path}
  |--- data
  `--- |--- coco
      `--- |--- annotations
              |--- instances_train2017.json
              |--- instances_val2017.json
              |--- person_keypoints_train2017.json
              |--- person_keypoints_val2017.json
              |--- image_info_test-dev2017.json
          |---|--- train2017
          |---|--- val2017
          |---|--- test2017
  ```

+ **Download some pre_trained model**: You can download many available models [here](https://github.com/xingyizhou/CenterNet/blob/master/readme/MODEL_ZOO.md) , thanks to [Centernet](https://github.com/xingyizhou/CenterNet).

+ **Train model example**:

  ```
  python src/main.py --task gridneighbordet --dataset coco --exp_id HG_ctdet_coco --debug 2 --load_model models/ctdet_coco_hg.pth --point_flags '0,4,8'
  ```

### Facial Landmark Detection

- 300-W consits of several different datasets
- Create directory to save images and annotations: mkdir ~/datasets/landmark-datasets/300W
- To download i-bug: https://ibug.doc.ic.ac.uk/download/annotations/ibug.zip
- To download afw: https://ibug.doc.ic.ac.uk/download/annotations/afw.zip
- To download helen: https://ibug.doc.ic.ac.uk/download/annotations/helen.zip
- To download lfpw: https://ibug.doc.ic.ac.uk/download/annotations/lfpw.zip
- To download the bounding box annotations: https://ibug.doc.ic.ac.uk/media/uploads/competitions/bounding_boxes.zip
- In the folder of `~/datasets/landmark-datasets/300W`, there are four zip files ibug.zip, afw.zip, helen.zip, and lfpw.zip
```
unzip ibug.zip -d ibug
mv ibug/image_092\ _01.jpg ibug/image_092_01.jpg
mv ibug/image_092\ _01.pts ibug/image_092_01.pts

unzip afw.zip -d afw
unzip helen.zip -d helen
unzip lfpw.zip -d lfpw
unzip bounding_boxes.zip ; mv Bounding\ Boxes Bounding_Boxes
```
The 300W directory is in `$HOME/datasets/landmark-datasets/300W` and the sturecture is:
```
-- afw
-- Bounding_boxes
-- helen
-- ibug
-- lfpw
```

Then you use the script convert the facial landmark`.pts` annotation to COCO format in `src/tools/300W/preprocess_300w_img.py`. 

```
python preprocess_300w_img.py
```

Don't forget change the img path and annotation path to your own. And after this script, you can get the train annotation`train.json`, val annotation `val.json`, 3 test annotations `common.json`,`challenge.json`, `full.json` and train img folder `train`, val img folder `val`, and 3 test img folders `commonsubset`,`challengesubset`, `fullset`. Then place the image and generated annotation in this way:

```
###Generated_train_img_path = /---path you choose--/train
###Generated_val_img_path = /---path you choose--/val
###Generated_test_common_img_path = /---path you choose--/commonsubset
###Generated_test_full_img_path = /---path you choose--/challengesubset
###Generated_test_challenge_img_path = /---path you choose--/fullset
###Generated_train_annotation = /---path you choose--/train.json
###Generated_val_annotation = /---path you choose--/val.json
###Generated_test_common_annotation = /---path you choose--/common.json
###Generated_test_full_annotation = /---path you choose--/full.json
###Generated_test_challenge_annotation = /---path you choose--/challenge.json

###CenterNet_path = /---path you choose--/centernet

cd $CenterNet_path
cd data
mkdir 300w
cd 300w
ln -s $Generated_train_img_path ./
ln -s $Generated_val_img_path ./
ln -s $Generated_test_common_img_path ./
ln -s $Generated_test_challenge_img_path ./
ln -s $Generated_test_full_img_path ./
mkdir annotations
cd annotations
cp $Generated_train_annotation ./
cp $Generated_val_annotation ./
cp $Generated_test_common_annotation ./
cp $Generated_test_full_annotation ./
cp $Generated_test_challenge_annotation ./
```

After this, your data folder just like this:

```
$CenterNet_path
|---data
`---|---coco
    |---300w
    `---|annotations
     		`---|---train.json
     		`---|---val.json
     		`---|---common.json
     		`---|---full.json
     		`---|---challenge.json
     		|---train
     		|---val
     		|---commonsubset
     		|---challengesubset
     		|---fullset
```

+ **Train model example**:

  ```
  python src/main.py --task landmark --load_model '' --dataset 300W --exp_id HG_landmark_300W --debug 2
  ```

### DeepFashion2 Detection

* **DataSet Preparation**:

  After download the [ Deepfashion2-dataset](https://drive.google.com/drive/folders/125F48fsMBz2EF0Cpqk6aaHet5VH399Ok),  you need fetch the unzip password through this [link](https://docs.google.com/forms/d/e/1FAIpQLSeIoGaFfCQILrtIZPykkr8q_h9qQ5BoTYbjvf95aXbid0v2Bw/viewform?usp=sf_link). And you can fetch the img and annos these two directory. Once you have got img and annotation of deepfashion2 dataset, you need to convert it to COCO format by running the script in `src/tools/deepfashion2/deepfashion2coco.py`.

  ```
  python deepfashion2coco.py
  ```

  Don't forget change the path of deepfashion annotations and images in stript. Then place the image and generated annotation in this way:

  ```
  ###deepfashion2_dataset_train_img_path = /---path you choose--/train
  ###deepfashion2_dataset_val_img_path = /---path you choose--/val
  ###deepfashion2_dataset_anno_path = /---path you choose--/anno
  ###Generated_deepfashion2_train_annotation = /---path you choose--/deepfashion_train.json
  ###Generated_deepfashion2_val_annotation = /---path you choose--/deepfashion_val.json
  ###CenterNet_path = /---path you choose--/centernet
  
  cd $CenterNet_path
  cd data
  mkdir deepfashion2
  cd deepfashion2
  ln -s $deepfashion2_dataset_img_path ./
  ln -s $deepfashion2_dataset_val_img_path ./
  mkdir annotations
  cd annotations
  cp $Generated_deepfashion2_train_annotation ./
  cp $Generated_deepfashion2_val_annotation ./
  ```

  After this, your data folder just like this:

  ```
  $CenterNet_path
  |---data
  `---|---coco
      |---300w
      |---deepfashion2
      `---|annotations
      		`---|---deepfashion_train.json
      		    |---deepfashion_val.json
          |---train
          |---val
  ```

+ **Train model example**:

  ```
  python src/main.py --task cloth --dataset deepfashion2 --exp_id HG_deepfashion2 --debug 2 --load_model ''
  ```

## 5. Licence

TSAL itself is released under  MIT License (refer to the LICENSE file for details).

## 6. Acknowledgements

* CenterNet: Object as point. [CenterNet](https://github.com/xingyizhou/CenterNet) from Xingyi Zhou, Dequan Wang, Philipp Krähenbühl.
* DCNv2: Deformable Convolutional Networks V2.[DCNv2](https://github.com/CharlesShang/DCNv2).

