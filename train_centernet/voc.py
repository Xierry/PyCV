import torch
import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import albumentations as A
import xml.etree.ElementTree as ET


def preprocess_img(image, input_size=512, pad=32, th_pad=22,
                   mean=(0.485, 0.456, 0.406), 
                   std=(0.229, 0.224, 0.225)):

    image_h, image_w, _ = image.shape
    if image_h < image_w:
        resize_w = input_size
        resize_h = int(image_h * input_size / image_w + 0.5)
        pad_pos, pad_neg = (input_size - resize_h) % pad, resize_h % pad
        _pad = pad_pos if pad_pos < th_pad else -pad_neg
        resize_h = resize_h + _pad
    else:
        resize_h = input_size
        resize_w = int(image_w * input_size / image_h + 0.5)
        pad_pos, pad_neg = (input_size - resize_w) % pad, resize_w % pad
        _pad = pad_pos if pad_pos < th_pad else -pad_neg
        resize_w = resize_w + _pad

    image=torch.as_tensor(
        (cv2.resize(image, (resize_w, resize_h)) / 255.0 - mean) / std, dtype=torch.float32
    ).permute(2,0,1).unsqueeze(0)
    return image, (image_h, image_w), (resize_h, resize_w)


class VOCDataset(torch.utils.data.Dataset):
    idx_to_className = ('__background__', 'circle')
    def __init__(self, root='../../Data/voc_all/VOC2007/', 
                       resize_size=512,
                       mean=(0.485, 0.456, 0.406), 
                       std=(0.229, 0.224, 0.225)):

        self.className_to_idx = dict((name, i) for i, name in enumerate(self.idx_to_className))

        self.root = root
        self.ids = self._get_ids()
        self.img_fmt = os.path.join(root, 'JPEGImages/{}.jpg')
        self.ann_fmt = os.path.join(root, 'Annotations/{}.xml')

        self.num_classes = len(self.idx_to_className) - 1
        self.down_stride = 4

        self.resize_size = resize_size
        self.mean = mean
        self.std = std
        self.collate_fn = collate_fn
        self.train()

    def __getitem__(self, idx):

        image = self._read_rgb_image(idx)
        boxes, classes = self._get_annotation(idx)
        if self.transform is not None:
            image, boxes, classes = self.transform(image, boxes, classes)

        info = {}
        image, _, (resize_height, resize_width) = preprocess_img(image, input_size=self.resize_size, mean=self.mean, std=self.std)
        gt_hm_witdh, gt_hm_height = resize_width // self.down_stride, resize_height // self.down_stride

        box_scale = np.asarray([[gt_hm_witdh, gt_hm_height]])
        wh = (boxes[..., 2:] - boxes[..., :2]) * box_scale
        ct = (boxes[..., :2] + boxes[..., 2:]) * (box_scale / 2)

        obj_mask = torch.ones(len(classes))
        hm = np.zeros((self.num_classes, gt_hm_height, gt_hm_witdh), dtype=np.float32)
        for i, (box_wh, class_idx, ct_int) in enumerate(zip(np.floor(wh + 0.5), classes, np.floor(ct + 0.5).astype('int32'))):
            radius = gaussian_radius(box_wh)

            if (hm[:, ct_int[1], ct_int[0]] == 1).any():
                obj_mask[i] = 0
                continue

            draw_umich_gaussian(hm[class_idx - 1], ct_int, max(0, int(radius)))
            if hm[class_idx - 1, ct_int[1], ct_int[0]] != 1:
                obj_mask[i] = 0
                print("hm[{}, {}, {}] != 1 ".format(class_idx - 1, ct_int[1], ct_int[0]))

        hm = torch.as_tensor(hm, dtype=torch.float32)
        obj_mask = obj_mask.eq(1)
        # boxes = torch.as_tensor(boxes, dtype=torch.float32)[obj_mask]
        # classes = torch.as_tensor(classes, dtype=torch.int64)[obj_mask]
        info['ct'] = torch.as_tensor(ct, dtype=torch.float32)[obj_mask].T
        info['wh'] = torch.as_tensor(wh, dtype=torch.float32)[obj_mask].T

        assert hm.eq(1).sum().item() == info['ct'].size(-1), \
            "image_name: {}, hm peer: {}, object num: {}".format(self.ids[idx], hm.eq(1).sum().item(), info['ct'].size(-1))

        return image.squeeze(), hm, info # image.squeeze(), boxes, classes, hm, info

    def __len__(self):
        return len(self.ids)

    def _read_rgb_image(self, i):
        image_dir = self.img_fmt.format(self.ids[i])
        image = cv2.imread(image_dir)[..., (2,1,0)]
        return image

    def _get_annotation(self, i):
        anno_dir = self.ann_fmt.format(self.ids[i])

        objects = ET.parse(anno_dir).findall('object')

        boxes = []
        labels = []
        for obj in objects:
            name = obj.find('name').text.lower().strip()
            label = self.className_to_idx.get(name, None)
            if label is None:
                continue
            labels.append(label)

            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text) - 1
            ymin = int(bndbox.find('ymin').text) - 1
            xmax = int(bndbox.find('xmax').text) - 1
            ymax = int(bndbox.find('ymax').text) - 1
            boxes.append([xmin, ymin, xmax, ymax])

        return np.array(boxes, dtype=np.float32), np.array(labels, dtype=np.float32)

    def train(self, training=True, transform=None):
        self.training=training
        self.transform = transform or self.build_transforms()
        return self
    
    def eval(self, transform=None):
        return self.train(False, transform)

    def _get_ids(self):

        ids = []
        # df = pd.read_csv('{}use_classes/rectangle.csv'.format(self.root), dtype={'id': 'str', 'label': 'int32'})
        # ids += 3 * df.loc[df['label'].astype('int32') > 0, 'id'].tolist()

        df = pd.read_csv('{}use_classes/circle.csv'.format(self.root), dtype={'id': 'str', 'label': 'int32'})
        ids += df.loc[df['label'].astype('int32') > 0, 'id'].tolist()
        
        np.random.shuffle(ids)
        return tuple(ids) # tuple(set(ids))

    def build_transforms(self):
    
        transform = [
            ToPercentCoords()
        ]

        if self.training:
            transform = [
                AlbuCompose([
                    A.RandomSizedBBoxSafeCrop(self.resize_size, self.resize_size, p=0.85),
                    A.RandomGamma(p=0.5),
                    A.CLAHE(p=0.5), # 对比度受限自适应
                    A.RandomBrightnessContrast(p=0.2),
                    # A.RandomContrast(), 
                    # A.HueSaturationValue(p=1),
                    A.ShiftScaleRotate(p=0.3, scale_limit=0.2, rotate_limit=20),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.2),
                    A.ChannelShuffle(p=0.3),
                    # A.OneOf([
                    #     A.Blur(blur_limit=7, p=0.5),
                    #     A.MotionBlur(blur_limit=7, p=1),
                    #     A.MedianBlur(blur_limit=7, p=1),
                    #     A.GaussianBlur(blur_limit=7, p=1),
                    # ]),
                ], bbox_params = {
                        'format': 'pascal_voc',
                        'label_fields': ['labels']
                    }
                ),

                ########## 放一些自定义变换
                # Expand(self.mean),
                # RandomSampleCrop(),
            ] + transform

        return Compose(transform)

    def make_loader(self, batch_size=2, num_workers=0, collate_fn=None):
        return torch.utils.data.DataLoader(self, 
            batch_size=batch_size, 
            collate_fn=collate_fn if collate_fn else self.collate_fn, 
            num_workers=num_workers
        )
    
    def show_images(self, image_idxes, results=None, step='default', threshold=.3, scale=6):
        # random.sample(range(len(vset)), 8)
        if isinstance(image_idxes, (torch.Tensor, np.ndarray)):
            image_idxes = image_idxes.tolist()
        elif isinstance(image_idxes, tuple):
            image_idxes = list(image_idxes)
        if results is None:
            results=[]
            for i in image_idxes:
                boxes, labels=self._get_annotation(i)
                scores = 100 * np.ones(labels.shape, dtype=np.float32)
                results.append((boxes, scores, labels.astype('int64')))

        batch_size = len(image_idxes)
        num_cols = 2**(int(np.log2(batch_size)) // 2 + 1)
        num_rows = batch_size // num_cols + bool(batch_size % num_cols)

        _, sub_axes= plt.subplots(num_rows, num_cols, figsize=(num_cols * scale, num_rows * scale))
        axes = sub_axes.flatten()

        colors = plt.cm.hsv(np.linspace(0, 1, len(self.idx_to_className))).tolist()
        for i, (image_idx, (boxes, scores, labels)) in enumerate(zip(image_idxes, results)):
            rgb_image = self._read_rgb_image(image_idx).astype('uint8') if isinstance(image_idx, (int, float)) else image_idx
            h, w, _ = rgb_image.shape
            box_scale = np.array([w, h, w, h])
            currentAxis = axes[i]
            currentAxis.imshow(rgb_image)
            currentAxis.axes.get_xaxis().set_visible(False)
            currentAxis.axes.get_yaxis().set_visible(False)
            for box, score, label in zip(boxes, scores, labels): # [topk, 4], [topk], [topk]
                if isinstance(box, torch.Tensor):
                    box=box.detach().cpu().numpy()
                if isinstance(score, torch.Tensor):
                    score=score.detach().cpu().numpy()
                if isinstance(label, torch.Tensor):
                    label=label.detach().cpu().numpy()
                if score < threshold:
                    continue
                label_name = self.idx_to_className[label]
                display_txt = '{}: {:.2f}'.format(label_name, score)
                color = colors[label]

                pt = box * box_scale if box.max() < 2 else box
                coords = (pt[0], pt[1]), pt[2]-pt[0]+1, pt[3]-pt[1]+1
                currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
                currentAxis.text(pt[0], pt[1], display_txt, bbox={'facecolor':color, 'alpha':0.5})
        # os.makedirs('./vis', exist_ok=True)
        # plt.savefig('./vis/step_{}.jpg'.format(step))
        plt.show()

    def show_tensor_images(self, image_idxes, step='default', threshold=.3, scale=6):
        
        if isinstance(image_idxes, (torch.Tensor, np.ndarray)):
            image_idxes = image_idxes.tolist()
        elif isinstance(image_idxes, tuple):
            image_idxes = list(image_idxes)

        batch_size = len(image_idxes)
        num_cols = 2**(int(np.log2(batch_size)) // 2 + 1)
        num_rows = batch_size // num_cols + bool(batch_size % num_cols)

        _, sub_axes= plt.subplots(num_rows, num_cols, figsize=(num_cols * scale, num_rows * scale))
        axes = sub_axes.flatten()

        colors = plt.cm.hsv(np.linspace(0, 1, len(self.idx_to_className))).tolist()
        for i, image_idx in enumerate(image_idxes):

            image = self._read_rgb_image(image_idx)
            boxes, labels = self._get_annotation(image_idx)
            image, boxes, labels = self.transform(image, boxes, labels.astype(np.int32))

            scores = 100 * np.ones(labels.shape, dtype=np.float32)
            if isinstance(image, torch.Tensor):
                image=image.detach().cpu().numpy()
            if image.max() < 2:
                image *= 255
            rgb_image = image.astype(np.uint8)
            h, w, _ = rgb_image.shape
            box_scale = np.array([w, h, w, h])
            currentAxis = axes[i]
            currentAxis.imshow(rgb_image)
            currentAxis.axes.get_xaxis().set_visible(False)
            currentAxis.axes.get_yaxis().set_visible(False)
            for box, score, label in zip(boxes, scores, labels): # [topk, 4], [topk], [topk]
                if isinstance(box, torch.Tensor):
                    box=box.detach().cpu().numpy()
                if isinstance(score, torch.Tensor):
                    score=score.detach().cpu().numpy()
                if isinstance(label, torch.Tensor):
                    label=label.detach().cpu().numpy()
                if score < threshold:
                    continue
                label_name = self.idx_to_className[label]
                display_txt = '{}: {:.2f}'.format(label_name, score)
                color = colors[label]

                pt = box * box_scale if box.max() < 2 else box
                coords = (pt[0], pt[1]), pt[2]-pt[0]+1, pt[3]-pt[1]+1
                currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
                currentAxis.text(pt[0], pt[1], display_txt, bbox={'facecolor':color, 'alpha':0.5})
        # os.makedirs('./vis', exist_ok=True)
        # plt.savefig('./vis/step_{}.jpg'.format(step))
        plt.show()

class AlbuCompose(A.Compose):
    def __init__(self, *args, **kwargs):
        super(AlbuCompose, self).__init__(*args, **kwargs)

    def __call__(self, image, boxes, labels):
        sample = super(AlbuCompose, self).__call__(image=image, bboxes=boxes, labels=labels)
        image, boxes, labels= sample['image'], sample['bboxes'], sample['labels']
        return image, boxes, labels


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, boxes=None, labels=None):

        for t in self.transforms:
            image, boxes, labels = t(image, boxes, labels)
            # if boxes is not None: boxes, labels = remove_empty_boxes(boxes, labels)
        return image, boxes, labels


class ToPercentCoords(object):

    def __call__(self, image, boxes=None, labels=None):

        height, width, _ = image.shape
        boxes, labels = self._to_dtypes(boxes, labels)
        boxes /= (width, height, width, height)
        return image, boxes, labels

    def _to_dtypes(self, boxes=None, labels=None):
        if boxes is not None and not isinstance(boxes, np.ndarray):
            boxes = np.asarray(boxes)
        if labels is not None and not isinstance(labels, np.ndarray):
            labels = np.asarray(labels, dtype='int64')
        return boxes, labels


def collate_fn(batch):
    # batch_imgs, batch_boxes, batch_classes, batch_hms, infos = zip(*batch)
    batch_imgs, batch_hms, infos = zip(*batch)
    chs, heights, widths = zip(*[img.shape for img in batch_imgs])
    max_h, max_w = max(heights), max(widths) # max_num_boxes = max(box.shape[0] for box in boxes_list)

    pad_imgs = []
    pad_hms = []
    for i, (img, hm) in enumerate(zip(batch_imgs, batch_hms)):

        _, h, w = img.shape
        pad_imgs.append(torch.nn.functional.pad(img, (0, max_w - w, 0, max_h - h), value=0.))

        _, h, w = hm.shape
        pad_hms.append(torch.nn.functional.pad(hm, (0, max_w // 4 - w, 0, max_h // 4 - h), value=0.))
    
    batch_imgs = torch.stack(pad_imgs)
    batch_hms = torch.stack(pad_hms)

    return batch_imgs, batch_hms, infos


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    else: 
        raise NotImplementedError
    return heatmap


def gaussian_radius(det_size, min_overlap=0.7):
    # https://zhuanlan.zhihu.com/p/96856635 动机
    # https://cloud.tencent.com/developer/article/1669896
    width, height = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2
    # r1 = (b1 + sq1) / (2 * a1)

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2
    # r2 = (b2 + sq2) / (2 * a2)

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    # r3 = (b3 + sq3) / (2 * a3)
    return min(r1, r2, r3)


def draw_msra_gaussian(heatmap, center, sigma):
    tmp_size = sigma * 3
    mu_x = int(center[0] + 0.5)
    mu_y = int(center[1] + 0.5)
    w, h = heatmap.shape[0], heatmap.shape[1]
    ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
    br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
    if ul[0] >= h or ul[1] >= w or br[0] < 0 or br[1] < 0:
        return heatmap
    size = 2 * tmp_size + 1
    x = np.arange(0, size, 1, np.float32)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    g_x = max(0, -ul[0]), min(br[0], h) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], w) - ul[1]
    img_x = max(0, ul[0]), min(br[0], h)
    img_y = max(0, ul[1]), min(br[1], w)
    heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(
        heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]],
        g[g_y[0]:g_y[1], g_x[0]:g_x[1]])
    return heatmap
