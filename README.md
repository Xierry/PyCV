# CenterNet (object as points) 简历展现使用

对于工作经历的项目纪录, 工作期间由tensorflow实现, 这里使用PyTorch整理实现(规避敏感问题)

实际工作的检测示例在notebook文件中PyCV/train_centernet/MINI_inference.ipynb(链接https://github.com/Xierry/PyCV/tree/master/train_centernet/MINI_inference.ipynb) centernet的置信度并不是大面积99%+ 所以不要奇怪, 个人测试的阈值在50% ~ 60% 可以很好区分正常检测目标和低置信度目标

模型源码在PyCV/CenterNet文件夹下, 链接https://github.com/Xierry/PyCV/tree/master/CenterNet

可以在PyCV/train_centernet/voc.py中自定义自己的数据集, 自己修改类别属性(self.idx_to_className), 图片读取函数(self._read_rgb_image)和标签读取函数(self._get_annotation)即可https://github.com/Xierry/PyCV/tree/master/train_centernet/voc.py

PyCV/train_centernet/train.py为个人写的参考简单训练脚本, 五脏俱全仅供参考https://github.com/Xierry/PyCV/tree/master/train_centernet/train.py
