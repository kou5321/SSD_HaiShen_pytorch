# 水下目标在线识别系统——海参的识别与计数

这是一个中国海洋大学SRDP创新创业引导专项，本项目采用SSD算法通过ROV的摄像头对水下目标（海参）进行识别并计数，最后要做到清点出ROV走过的地方的所有海参个数。  
算法是基于[ssd.pytorch](https://github.com/amdegroot/ssd.pytorch) 这位大神的，并结合我们自己的项目进行了相应的修改。 项目所用的框架为[pytorch](https://pytorch.org/)。
|成员||
|-|-|
|张心亮|（组长）|
|刘翔||
|扶玉斌||
|万路||
|汪洋||

## 数据集说明

数据集一方面采用 [URPC](http://www.cnurpc.org/a/js/)的数据集作为主要数据集，另一方面我们自己收集了384张陆地上和水下的海参图片，由自己手动标注完成。 两个数据集混合处理后当做这个项目的数据集。数据集格式与VOC2007的数据集格式完全相同。格式为：

- VOCdevkit
  - VOC2007
    - Annotation
        - 000001.xml
        - 000002.xml
        - ……
    - JPEGImages
        - 000001.jpg
        - 000002.jpg
        - ……
    - ImageSets
        - train.txt
        - train_val.txt
        - ……

**note** :里面的图片和相应的标注文件都是本项目所用的。
## 使用说明

1. 搭建环境
首先需要搭建环境，建议用anaconda。 执行如下命令即可安装：`conda env create -f haishen.yaml`  

2. 训练  
进行训练：`python train.py --dataset_root path/to/VOCdevkit --batch_size 8`  如果显卡现存允许可以把`batch_size`设置大一些，这个视情况而定，但经过实际发现设置为 16 训练较稳定。  
权重每5000次迭代保存一次，保存目录在 `weights/` 文件夹下。 

3. 实时监测  
`python demo/live.py --weights 训练好的权重名称.pth`
