import torch
from tqdm import tqdm
from voc import VOCDataset
from config import Config
import pdb #, random
import sys
sys.path.append('..')
from CenterNet import CenterNet
from CenterNet.optim.lr_scheduler import WarmupMultiStepLR


cfg = Config()
vSet = VOCDataset(root=cfg.root, resize_size=cfg.resize_size)
net = CenterNet(num_classes=vSet.num_classes, 
                backbone=cfg.backbone, cfg=cfg, 
                device=cfg.device).train()
device = net.device

params = [p for p in net.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(params, lr=cfg.lr, 
                              weight_decay=cfg.weight_decay, 
                              amsgrad=cfg.AMSGRAD)
optimizer.zero_grad()
scheduler = WarmupMultiStepLR(optimizer, milestones=cfg.milestones, gamma=cfg.gamma, warmup_iters=cfg.warmup_iters)
            # torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1/(epoch+1))
            # torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
            # torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1, last_epoch=-1)
            # torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma, last_epoch=-1)
            # torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=0, last_epoch=-1)
            # torch.optim.lr_scheduler.CosineAnnealingWarmRestarts() 
            # https://zhuanlan.zhihu.com/p/261134624 https://blog.csdn.net/qyhaill/article/details/103043637

if __name__ == "__main__":

    step_iters=tqdm(range(1, cfg.max_iter+1), initial=1)
    for step in step_iters:
        try:
            images, hms, infos = next(iterator)
        except:
            iterator = iter(vSet.train().make_loader(batch_size=cfg.batch_size, num_workers=cfg.num_workers))
            images, hms, infos = next(iterator)

        cls_loss, reg_loss = net(inputs=images.to(device), targets=(hms, infos))

        if cfg.gradient_accumulation > 1:
            cls_loss /= cfg.gradient_accumulation
            reg_loss /= cfg.gradient_accumulation

        loss = cls_loss + reg_loss
        loss.backward()
        torch.nn.utils.clip_grad_value_(params, cfg.clip_value)

        if step % cfg.gradient_accumulation == 0:
            optimizer.step()
            optimizer.zero_grad()
        ################# 打印损失 if step % 1 == 0:
            step_iters.set_postfix(ClsLoss=cls_loss.item(), RegLoss=reg_loss.item(), Loss=loss.item(), Lr=optimizer.param_groups[0]['lr'])

        # if step % cfg.steps_per_epoch == 0:
        scheduler.step() 

        # if step % 3000 == 0: vSet.show_images(image_idxes, step='{}_gt'.format(step))
        # if step % 5000 == 0:
        #     net.eval()
        #     with torch.no_grad():
        #         detections = net(images)
        #     vSet.show_images(image_idxes, results=detections, threshold=0.26, step='{}_pred'.format(step))
        #     net.train() # if step > 2: break

    weights='./weights/centernet.pth'
    torch.save({k: tensor.detach().cpu() for k, tensor in net.state_dict().items()}, weights)
    net.load_state_dict(torch.load(weights))

    # r = net.to('cpu').detect_one_image(
    # cv2.resize(cv2.imread('./weights/pikachu.jpg'), (512, 512))[..., (2,1,0)], th=0.05)
