# CenterNet (object as points) 简历展现使用

对于工作经历的项目纪录, 工作期间由tensorflow实现, 这里使用PyTorch整理实现

实际工作的检测示例在notebook文件中PyCV/train_centernet/inference.ipynb(比较大,网速不好打不开, 建议点击下载本地notebook打开, 链接https://github.com/Xierry/PyCV/tree/master/train_centernet/inference.ipynb) centernet的置信度并不是正常情况下不是大面积99% + 所以不要奇怪, 个人测试的阈值在50% ~ 60% 可以很好区分正常检测目标和低置信度目标

模型源码在PyCV/CenterNet文件夹下, 链接https://github.com/Xierry/PyCV/tree/master/CenterNet

可以在PyCV/train_centernet/voc.py中自定义自己的数据集, 自己修改self.idx_to_className类别属性, self._read_rgb_image图片读取函数和self._get_annotation标签读取即可https://github.com/Xierry/PyCV/tree/master/train_centernet/voc.py

PyCV/train_centernet/train.py为个人写的参考简单训练脚本, 五脏俱全仅供参考https://github.com/Xierry/PyCV/tree/master/train_centernet/train.py
