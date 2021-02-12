

class Config:
    ################ 模型部分
    num_classes = 1 # 类别数量, (不包括背景)
    backbone = 'r18'# backbone 使用的骨架网络

    lr = 6e-4
    batch_size = 4
    gradient_accumulation=4
    device='cuda:0'
    clip_value = 100

    loss_alpha = 1.
    loss_beta = 0.2
    loss_gamma = 1.

    pretrained=False
    deconv_with_bias=False
    topk = 40
    threshold=0.1

    AMSGRAD = True
    
    root = '../../Data/voc_all/VOC2007/'
    # split = 'trainval'
    resize_size = 512
    num_workers = 0
    
    mean=(0.485, 0.456, 0.406)
    std=(0.229, 0.224, 0.225)

    # train
    optimizer = 'AdamW'

    lr_schedule = 'WarmupMultiStepLR'
    weight_decay = 1e-4
    gamma = 0.3 # 0.1
    milestones = (1500, 2000, 2500, 3000)
    max_iter = 3000
    
    warmup_iters = 1000

    # freeze_bn = True
    bn_momentum = 0.1

    # head
    head_channel = 64


    # steps_per_epoch = 200
    def __str__(self):
        s = '\n'
        for k, v in Config.__dict__.items():
            if k[:2] == '__' and k[-2:] == '__':
                continue
            s += k + ':  ' + str(v) + '\n'
        return s
