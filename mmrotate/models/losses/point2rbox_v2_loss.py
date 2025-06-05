# Copyright (c) OpenMMLab. All rights reserved.
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from mmdet.models.losses.utils import weighted_loss

from mmrotate.registry import MODELS
from mmrotate.models.losses.gaussian_dist_loss import postprocess


@weighted_loss
def gwd_sigma_loss(pred, target, fun='log1p', tau=1.0, alpha=1.0, normalize=True):
    """Gaussian Wasserstein distance loss.
    Modified from gwd_loss. 
    gwd_sigma_loss only involves sigma in Gaussian, with mu ignored.

    Args:
        pred (torch.Tensor): Predicted bboxes.
        target (torch.Tensor): Corresponding gt bboxes.
        fun (str): The function applied to distance. Defaults to 'log1p'.
        tau (float): Defaults to 1.0.
        alpha (float): Defaults to 1.0.
        normalize (bool): Whether to normalize the distance. Defaults to True.

    Returns:
        loss (torch.Tensor)

    """
    Sigma_p = pred
    Sigma_t = target

    whr_distance = Sigma_p.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    whr_distance = whr_distance + Sigma_t.diagonal(
        dim1=-2, dim2=-1).sum(dim=-1)

    _t_tr = (Sigma_p.bmm(Sigma_t)).diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    _t_det_sqrt = (Sigma_p.det() * Sigma_t.det()).clamp(1e-7).sqrt()
    whr_distance = whr_distance + (-2) * (
        (_t_tr + 2 * _t_det_sqrt).clamp(1e-7).sqrt())

    distance = (alpha * alpha * whr_distance).clamp(1e-7).sqrt()

    if normalize:
        scale = 2 * (
            _t_det_sqrt.clamp(1e-7).sqrt().clamp(1e-7).sqrt()).clamp(1e-7)
        distance = distance / scale

    return postprocess(distance, fun=fun, tau=tau)


def bhattacharyya_coefficient(pred, target):
    """Calculate bhattacharyya coefficient between 2-D Gaussian distributions.

    Args:
        pred (Tuple): tuple of (xy, sigma).
            xy (torch.Tensor): center point of 2-D Gaussian distribution
                with shape (N, 2).
            sigma (torch.Tensor): covariance matrix of 2-D Gaussian distribution
                with shape (N, 2, 2).
        target (Tuple): tuple of (xy, sigma).

    Returns:
        coef (Tensor): bhattacharyya coefficient with shape (N,).
    """
    xy_p, Sigma_p = pred
    xy_t, Sigma_t = target

    _shape = xy_p.shape

    xy_p = xy_p.reshape(-1, 2)
    xy_t = xy_t.reshape(-1, 2)
    Sigma_p = Sigma_p.reshape(-1, 2, 2)
    Sigma_t = Sigma_t.reshape(-1, 2, 2)

    Sigma_M = (Sigma_p + Sigma_t) / 2
    dxy = (xy_p - xy_t).unsqueeze(-1)
    t0 = torch.exp(-0.125 * dxy.permute(0, 2, 1).bmm(torch.linalg.solve(Sigma_M, dxy)))
    t1 = (Sigma_p.det() * Sigma_t.det()).clamp(1e-7).sqrt()
    t2 = Sigma_M.det()

    coef = t0 * (t1 / t2).clamp(1e-7).sqrt()[..., None, None]
    coef = coef.reshape(_shape[:-1])
    return coef

# 修改gaussian_overlap_loss函数以支持close_mask
@weighted_loss
def gaussian_overlap_loss(pred, target=None, close_mask=None, alpha=0.01, beta=0.6065, debug=False):
    """Calculate Gaussian overlap loss based on bhattacharyya coefficient.

    Args:
        pred (Tuple): tuple of (xy, sigma).
            xy (torch.Tensor): center point of 2-D Gaussian distribution
                with shape (N, 2).
            sigma (torch.Tensor): covariance matrix of 2-D Gaussian distribution
                with shape (N, 2, 2).
        target: unused parameter, kept for compatibility with @weighted_loss decorator
        close_mask (torch.Tensor, optional): mask indicating which instance pairs
            should be ignored in overlap loss computation. Shape (N, N).
        debug (bool): Whether to print debug information.

    Returns:
        loss (Tensor): overlap loss with shape (N, N).
    """
    mu, sigma = pred
    B = mu.shape[0]
    
    if debug:
        print(f"[DEBUG] gaussian_overlap_loss: B={B}, mu.shape={mu.shape}, sigma.shape={sigma.shape}")
        if close_mask is not None:
            print(f"[DEBUG] close_mask provided: shape={close_mask.shape}, sum={close_mask.sum().item()}")
            close_pairs = torch.where(close_mask)
            if len(close_pairs[0]) > 0:
                print(f"[DEBUG] Close pairs: {list(zip(close_pairs[0].cpu().numpy(), close_pairs[1].cpu().numpy()))}")
        else:
            print("[DEBUG] No close_mask provided")
    
    mu0 = mu[None].expand(B, B, 2)
    sigma0 = sigma[None].expand(B, B, 2, 2)
    mu1 = mu[:, None].expand(B, B, 2)
    sigma1 = sigma[:, None].expand(B, B, 2, 2)
    loss = bhattacharyya_coefficient((mu0, sigma0), (mu1, sigma1))
    loss[torch.eye(B, dtype=bool)] = 0
    
    if debug:
        print(f"[DEBUG] Loss before close_mask: mean={loss.mean().item():.6f}, max={loss.max().item():.6f}")
    
    # 如果提供了close_mask，则将接近的实例对的损失设为0
    if close_mask is not None:
        original_loss_sum = loss.sum().item()
        loss[close_mask] = 0
        if debug:
            new_loss_sum = loss.sum().item()
            print(f"[DEBUG] Loss after close_mask: sum changed from {original_loss_sum:.6f} to {new_loss_sum:.6f}")
    
    loss = F.leaky_relu(loss - beta, negative_slope=alpha) + beta * alpha
    loss = loss.sum(-1)
    
    if debug:
        print(f"[DEBUG] Final loss: mean={loss.mean().item():.6f}, shape={loss.shape}")
    
    return loss
@MODELS.register_module()
class GaussianOverlapLoss(nn.Module):
    """Gaussian Overlap Loss.

    Args:
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'. Options are "none", "mean" and
            "sum".
        loss_weight (float, optional): Weight of loss. Defaults to 1.0.
        debug (bool, optional): Whether to print debug information. Defaults to False.

    Returns:
        loss (torch.Tensor)
    """

    def __init__(self,
                 reduction='mean',
                 loss_weight=1.0,
                 lamb=1e-4,
                 debug=False):
        super(GaussianOverlapLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.lamb = lamb
        self.debug = debug

    def forward(self,
                pred,
                close_mask=None,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function."""
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        assert len(pred[0]) == len(pred[1])

        if self.debug:
            print(f"[DEBUG] GaussianOverlapLoss forward: pred shapes={[p.shape for p in pred]}")

        sigma = pred[1]
        L = torch.linalg.eigh(sigma)[0].clamp(1e-7).sqrt()
        loss_lamb = F.l1_loss(L, torch.zeros_like(L), reduction='none')
        loss_lamb = self.lamb * loss_lamb.log1p().mean()
        
        # 修正调用方式 - 添加debug参数
        overlap_loss = gaussian_overlap_loss(
            pred,
            target=None,
            close_mask=close_mask,
            debug=self.debug,  # 传递debug参数
            weight=weight,
            reduction=reduction,
            avg_factor=avg_factor)
        
        total_loss = self.loss_weight * (loss_lamb + overlap_loss)
        
        if self.debug:
            print(f"[DEBUG] GaussianOverlapLoss: lamb_loss={loss_lamb.item():.6f}, "
                  f"overlap_loss={overlap_loss.item():.6f}, total_loss={total_loss.item():.6f}")
        
        return total_loss
    
    
def gaussian_2d(xy, mu, sigma, normalize=False):
    dxy = (xy - mu).unsqueeze(-1)
    t0 = torch.exp(-0.5 * dxy.permute(0, 2, 1).bmm(torch.linalg.solve(sigma, dxy)))
    if normalize:
        t0 = t0 / (2 * np.pi * sigma.det().clamp(1e-7).sqrt())
    return t0



import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

def plot_gaussian_voronoi_watershed(*images, labels=None, class_names=None):
    """Plot figures for debug with voronoi boundaries and colored watershed masks.
    
    Args:
        *images: Variable number of images (image, cls_bg, markers, ...)
        labels: Tensor containing class labels for each instance
        class_names: List of class names for legend
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    import matplotlib.patches as mpatches
    import numpy as np
    
    # 调整图片大小以容纳图例
    fig_width = len(images) * 4 + 2  # 额外空间给图例
    plt.figure(dpi=300, figsize=(fig_width, 4))
    plt.tight_layout()
    
    for i in range(len(images)):
        img = images[i]
        img = (img - img.min()) / (img.max() - img.min())
        if img.dim() == 3:
            img = img.permute(1, 2, 0)
        img = img.detach().cpu().numpy()
        plt.subplot(1, len(images), i + 1)
        
        # 对于第一张图（原图），叠加维诺图边界和分水岭mask
        if i == 0 and len(images) >= 3:
            # 显示原图
            plt.imshow(img, cmap='gray' if len(img.shape) == 2 else None)
            
            # 获取cls_bg和markers
            cls_bg = images[1]  # 类别背景图
            markers = images[2]  # 分水岭结果
            
            cls_bg_np = cls_bg.detach().cpu().numpy()
            markers_np = markers.detach().cpu().numpy()
            
            # 1. 绘制维诺图边界（鲜艳红色线条）
            # 找到边界：cls_bg中值为15的位置是背景，边界就是这些区域的边缘
            voronoi_boundary = np.zeros_like(cls_bg_np, dtype=bool)
            
            # 使用形态学操作找到不同区域之间的边界
            from scipy import ndimage
            import time
            # 对于每个非背景区域，找到其边界
            unique_regions = np.unique(cls_bg_np)
            for region_id in unique_regions:
                if region_id != 15 and region_id != -1:  # 排除背景和无效区域
                    region_mask = (cls_bg_np == region_id)
                    # 膨胀然后减去原区域得到边界
                    dilated = ndimage.binary_dilation(region_mask)
                    boundary = dilated & ~region_mask
                    voronoi_boundary |= boundary
            
            # 绘制维诺图边界
            if voronoi_boundary.any():
                # 创建边界的轮廓
                y_coords, x_coords = np.where(voronoi_boundary)
                plt.scatter(x_coords, y_coords, c='red', s=0.1, alpha=0.8)
            
            # 2. 绘制分水岭结果的彩色透明mask
            if labels is not None:
                labels_np = labels.detach().cpu().numpy()
                unique_labels = np.unique(labels_np)
                
                # 生成鲜明的颜色
                colors = plt.cm.tab10(np.linspace(0, 1, 10))  # 使用tab10颜色映射
                if len(unique_labels) > 10:
                    # 如果类别超过10个，使用更多颜色
                    colors = plt.cm.tab20(np.linspace(0, 1, 20))
                
                # 为每个类别创建mask并叠加显示
                for idx, label in enumerate(unique_labels):
                    # 找到属于该类别的实例索引
                    class_instances = np.where(labels_np == label)[0]
                    
                    # 创建该类别的总mask
                    class_mask = np.zeros_like(markers_np, dtype=bool)
                    for instance_idx in class_instances:
                        # markers中实例的标签是从1开始的
                        instance_mask = (markers_np == instance_idx + 1)
                        class_mask |= instance_mask
                    
                    if class_mask.any():
                        # 创建彩色透明mask
                        color_rgba = colors[idx % len(colors)]
                        
                        # 创建RGBA图像用于透明叠加
                        overlay = np.zeros((*class_mask.shape, 4))
                        overlay[class_mask] = [*color_rgba[:3], 0.5]  # 50% 透明度
                        
                        # 叠加显示
                        plt.imshow(overlay, alpha=0.7)
                
                # 创建图例并放在图片外面
                legend_patches = []
                for idx, label in enumerate(unique_labels):
                    if class_names is not None and label < len(class_names):
                        class_name = class_names[label]
                    else:
                        class_name = f'类别 {label}'
                    
                    color_rgba = colors[idx % len(colors)]
                    legend_patches.append(
                        mpatches.Patch(color=color_rgba[:3], 
                                     label=class_name, alpha=0.7)
                    )
                
                # 将图例放在图片右侧外面
                if legend_patches:
                    plt.legend(handles=legend_patches, 
                             bbox_to_anchor=(1.05, 1), 
                             loc='upper left',
                             fontsize=8, 
                             fancybox=True, 
                             shadow=True)
            
            # 添加标题说明
            plt.title('原图+维诺图边界(红)+分水岭mask', fontsize=10)
            
        elif i == 3:
            plt.imshow(img)
            x = np.linspace(0, 1024, 1024)
            y = np.linspace(0, 1024, 1024)
            X, Y = np.meshgrid(x, y)
            plt.contourf(X, Y, img, levels=8, cmap=plt.get_cmap('magma'))
            plt.title('高斯分布', fontsize=10)
        else:
            plt.imshow(img, cmap='viridis' if i > 0 else None)
            if i == 1:
                plt.title('类别背景图', fontsize=10)
            elif i == 2:
                plt.title('分水岭结果', fontsize=10)
        
        plt.xticks([])
        plt.yticks([])
    
    # 调整布局以容纳图例
    plt.tight_layout()
    plt.subplots_adjust(right=0.85)  # 为图例留出空间
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    plt.savefig(f'debug/{timestamp}-Gaussian-Voronoi.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def gaussian_voronoi_watershed_loss(mu, sigma,
                                    label, image, 
                                    pos_thres, neg_thres, 
                                    close_mask=None,
                                    down_sample=2, topk=0.95, 
                                    default_sigma=4096,
                                    voronoi='gaussian-orientation',
                                    alpha=0.1,
                                    debug=False):
    J = len(sigma)
    if J == 0:
        return sigma.sum()
    
    if debug:
        print(f"[DEBUG] gaussian_voronoi_watershed_loss: J={J}, mu.shape={mu.shape}")
        if close_mask is not None:
            print(f"[DEBUG] close_mask shape={close_mask.shape}, sum={close_mask.sum().item()}")
        else:
            print("[DEBUG] No close_mask provided")
    
    D = down_sample
    H, W = image.shape[-2:]
    h, w = H // D, W // D
    x = torch.linspace(0, h, h, device=mu.device)
    y = torch.linspace(0, w, w, device=mu.device)
    xy = torch.stack(torch.meshgrid(x, y, indexing='xy'), -1)
    
    # 处理实例合并 - 关键修改点
    effective_mu = mu.clone()
    effective_sigma = sigma.clone()
    effective_labels = label.clone()
    active_instances = torch.arange(J, device=mu.device)  # 跟踪哪些实例是活跃的
    
    if close_mask is not None and J >= 2:
        if debug:
            print(f"[DEBUG] Processing close_mask for label merging...")
        
        merge_groups = []
        processed = torch.zeros(J, dtype=torch.bool, device=mu.device)
        
        for i in range(J):
            if processed[i]:
                continue
            group = [i]
            for j in range(i + 1, J):
                if close_mask[i, j] and not processed[j]:
                    group.append(j)
                    processed[j] = True
            processed[i] = True
            merge_groups.append(group)
        
        if debug:
            print(f"[DEBUG] Found {len(merge_groups)} merge groups:")
            for idx, group in enumerate(merge_groups):
                if len(group) > 1:
                    print(f"[DEBUG]   Group {idx}: instances {group} -> unified label {label[group[0]].item()}")
        
        # 真正合并实例参数
        new_active_instances = []
        new_mu = []
        new_sigma = []
        new_labels = []
        
        for group_idx, group in enumerate(merge_groups):
            if len(group) > 1:  # 需要合并的组
                # 合并策略：可以选择不同的合并方式
                # 方案1：使用第一个实例的参数
                representative_idx = group[0]
                new_mu.append(effective_mu[representative_idx])
                new_sigma.append(effective_sigma[representative_idx])
                new_labels.append(effective_labels[representative_idx])
                new_active_instances.append(representative_idx)
                
                # 方案2：使用加权平均（可选）
                # group_mu = torch.stack([effective_mu[i] for i in group]).mean(0)
                # group_sigma = torch.stack([effective_sigma[i] for i in group]).mean(0)
                # new_mu.append(group_mu)
                # new_sigma.append(group_sigma)
                # new_labels.append(effective_labels[group[0]])
                # new_active_instances.append(group[0])
                
                if debug:
                    print(f"[DEBUG] Merged group {group} -> using instance {representative_idx}")
            else:  # 单独的实例
                idx = group[0]
                new_mu.append(effective_mu[idx])
                new_sigma.append(effective_sigma[idx])
                new_labels.append(effective_labels[idx])
                new_active_instances.append(idx)
        
        # 更新为合并后的参数
        effective_mu = torch.stack(new_mu)
        effective_sigma = torch.stack(new_sigma)
        effective_labels = torch.stack(new_labels)
        active_instances = torch.tensor(new_active_instances, device=mu.device)
        J_effective = len(effective_mu)  # 更新有效实例数量
        
        if debug:
            print(f"[DEBUG] After merging: J_effective={J_effective}, original J={J}")
    else:
        J_effective = J

    # 使用合并后的参数计算Voronoi图
    vor = mu.new_zeros(J_effective, h, w)
    mm = (effective_mu.detach() / D).round()
    
    if voronoi == 'standard':
        sg = sigma.new_tensor((default_sigma, 0, 0, default_sigma)).reshape(2, 2)
        sg = sg / D ** 2
        for j, m in enumerate(mm):
            vor[j] = gaussian_2d(xy.view(-1, 2), m[None], sg[None]).view(h, w)
    elif voronoi == 'gaussian-orientation':
        L, V = torch.linalg.eigh(effective_sigma)
        L = L.detach().clone()
        L = L / (L[:, 0:1] * L[:, 1:2]).sqrt() * default_sigma
        sg = V.matmul(torch.diag_embed(L)).matmul(V.permute(0, 2, 1)).detach()
        sg = sg / D ** 2
        for j, (m, s) in enumerate(zip(mm, sg)):
            vor[j] = gaussian_2d(xy.view(-1, 2), m[None], s[None]).view(h, w)
    elif voronoi == 'gaussian-full':
        sg = effective_sigma.detach() / D ** 2
        for j, (m, s) in enumerate(zip(mm, sg)):
            vor[j] = gaussian_2d(xy.view(-1, 2), m[None], s[None]).view(h, w)
    
    # val: max prob, vor: belong to which instance, cls: belong to which class
    val, vor = torch.max(vor, 0)
    if D > 1:
        vor = vor[:, None, :, None].expand(-1, D, -1, D).reshape(H, W)
        val = F.interpolate(
            val[None, None], (H, W), mode='bilinear', align_corners=True)[0, 0]
    
    # 使用合并后的标签
    cls = effective_labels[vor]
    kernel = val.new_ones((1, 1, 3, 3))
    kernel[0, 0, 1, 1] = -8
    ridges = torch.conv2d(vor[None].float(), kernel, padding=1)[0] != 0
    vor += 1
    pos_thres = val.new_tensor(pos_thres)
    neg_thres = val.new_tensor(neg_thres)
    vor[val < pos_thres[cls]] = 0
    vor[val < neg_thres[cls]] = J_effective + 1
    vor[ridges] = J_effective + 1

    cls_bg = torch.where(vor == J_effective + 1, 15, cls)
    cls_bg = torch.where(vor == 0, -1, cls_bg)

    # PyTorch does not support watershed, use cv2
    img_uint8 = (image - image.min()) / (image.max() - image.min()) * 255
    img_uint8 = img_uint8.permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)
    img_uint8 = cv2.medianBlur(img_uint8, 3)
    markers = vor.detach().cpu().numpy().astype(np.int32)
    markers = vor.new_tensor(cv2.watershed(img_uint8, markers))
    
    if debug:
        # 传递合并后的标签给可视化函数
        plot_gaussian_voronoi_watershed(image, cls_bg, markers, labels=effective_labels)

    # 计算损失时需要考虑原始实例和有效实例的映射关系
    L, V = torch.linalg.eigh(sigma)  # 使用原始sigma计算损失
    L_target = []
    
    # 为每个原始实例计算target
    for j in range(J):
        # 找到这个原始实例对应的有效实例
        if close_mask is not None:
            # 找到j在哪个合并组中
            effective_idx = None
            for group_idx, group in enumerate(merge_groups):
                if j in group:
                    effective_idx = group_idx
                    break
            if effective_idx is not None:
                # 在markers中查找对应的区域 (标记从1开始)
                xy = (markers == effective_idx + 1).nonzero()[:, (1, 0)].float()
            else:
                xy = torch.empty(0, 2, device=mu.device)
        else:
            xy = (markers == j + 1).nonzero()[:, (1, 0)].float()
            
        if len(xy) == 0:
            L_target.append(L[j].detach())
            continue
        xy = xy - mu[j]  # 使用原始mu
        xy = V[j].T.matmul(xy[:, :, None])[:, :, 0]
        max_x = torch.max(torch.abs(xy[:, 0]))
        max_y = torch.max(torch.abs(xy[:, 1]))
        L_target.append(torch.stack((max_x, max_y)) ** 2)
    
    L_target = torch.stack(L_target)
    L = torch.diag_embed(L)
    L_target = torch.diag_embed(L_target)
    loss = gwd_sigma_loss(L, L_target.detach(), reduction='none')
    loss = torch.topk(loss, int(np.ceil(len(loss) * topk)), largest=False)[0].mean() 

    if debug:
        print(f"[DEBUG] Final loss: {loss.item():.6f}")
    
    return loss, (vor, markers)


# 同样需要修正 VoronoiWatershedLoss 的 forward 方法
@MODELS.register_module()
class VoronoiWatershedLoss(nn.Module):
    """Voronoi Watershed Loss.

    Args:
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'. Options are "none", "mean" and
            "sum".
        loss_weight (float, optional): Weight of loss. Defaults to 1.0.

    Returns:
        loss (torch.Tensor)
    """

    def __init__(self,
                 down_sample=2,
                 reduction='mean',
                 loss_weight=1.0,
                 topk=0.95,
                 alpha=0.1,
                 debug=False):
        super(VoronoiWatershedLoss, self).__init__()
        self.down_sample = down_sample
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.topk = topk
        self.alpha = alpha
        self.debug = debug

    def forward(self, pred, label, image, pos_thres, neg_thres, 
                voronoi='orientation', close_mask=None):
        """Forward function.

        Args:
            pred (Tuple): Tuple of (xy, sigma).
                xy (torch.Tensor): Center point of 2-D Gaussian distribution
                    with shape (N, 2).
                sigma (torch.Tensor): covariance matrix of 2-D Gaussian distribution
                    with shape (N, 2, 2).
            label (torch.Tensor): Labels for each instance.
            image (torch.Tensor): The image for watershed with shape (3, H, W).
            pos_thres (list): Positive thresholds for each class.
            neg_thres (list): Negative thresholds for each class.
            voronoi (str): Type of voronoi diagram.
            close_mask (torch.Tensor, optional): mask indicating which instance pairs
                should be treated as the same instance. Shape (N, N).

        Returns:
            torch.Tensor: The calculated loss
        """
        loss, self.vis = gaussian_voronoi_watershed_loss(
            *pred, 
            label,
            image, 
            pos_thres, 
            neg_thres, 
            close_mask=close_mask,  # 使用关键字参数传递
            down_sample=self.down_sample, 
            topk=self.topk,
            voronoi=voronoi,
            alpha=self.alpha,
            debug=self.debug)
        return self.loss_weight * loss

def rbbox2roi(bbox_list):
    """Convert a list of bboxes to roi format.

    Args:
        bbox_list (list[Tensor]): a list of bboxes corresponding to a batch
            of images.

    Returns:
        Tensor: shape (n, 6), [batch_ind, cx, cy, w, h, a]
    """
    rois_list = []
    for img_id, bboxes in enumerate(bbox_list):
        if bboxes.size(0) > 0:
            img_inds = bboxes.new_full((bboxes.size(0), 1), img_id)
            rois = torch.cat([img_inds, bboxes[:, :5]], dim=-1)
        else:
            rois = bboxes.new_zeros((0, 6))
        rois_list.append(rois)
    rois = torch.cat(rois_list, 0)
    return rois


def plot_edge_map(feat, edgex, edgey):
    """Plot figures for debug."""
    import matplotlib.pyplot as plt
    plt.figure(dpi=300, figsize=(4, 4))
    plt.tight_layout()
    fileid = np.random.randint(0, 20)
    for i in range(len(feat)):
        img0 = feat[i, :3]
        img0 = (img0 - img0.min()) / (img0.max() - img0.min())
        img1 = edgex[i, :3]
        img1 = (img1 - img1.min()) / (img1.max() - img1.min())
        img2 = edgey[i, :3]
        img2 = (img2 - img2.min()) / (img2.max() - img2.min())
        img3 = img1 + img2
        img3 = (img3 - img3.min()) / (img3.max() - img3.min())
        img = torch.cat((torch.cat((img0, img2), -1), 
                         torch.cat((img1, img3), -1)), -2
                         ).permute(1, 2, 0).detach().cpu().numpy()
        N = int(np.ceil(np.sqrt(len(feat))))
        plt.subplot(N, N, i + 1)
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
    plt.savefig(f'debug/Edge-Map-{fileid}.png')
    plt.close()


@MODELS.register_module()
class EdgeLoss(nn.Module):
    """Edge Loss.

    Args:
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'. Options are "none", "mean" and
            "sum".
        loss_weight (float, optional): Weight of loss. Defaults to 1.0.

    Returns:
        loss (torch.Tensor)
    """

    def __init__(self,
                 resolution=24,
                 max_scale=1.6,
                 sigma=6,
                 reduction='mean',
                 loss_weight=1.0,
                 debug=False):
        super(EdgeLoss, self).__init__()
        self.resolution = resolution
        self.max_scale = max_scale
        self.sigma = sigma
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.center_idx = self.resolution / self.max_scale
        self.debug = debug

        self.roi_extractor = MODELS.build(dict(
            type='RotatedSingleRoIExtractor',
                roi_layer=dict(
                    type='RoIAlignRotated',
                    out_size=(2 * self.resolution + 1),
                    sample_num=2,
                    clockwise=True),
            out_channels=1,
            featmap_strides=[1],
            finest_scale=1024))

        edge_idx = torch.arange(0, self.resolution + 1)
        edge_distribution = torch.exp(-((edge_idx - self.center_idx) ** 2) / (2 * self.sigma ** 2))
        edge_distribution[0] = edge_distribution[-1] = 0
        self.register_buffer('edge_idx', edge_idx)
        self.register_buffer('edge_distribution', edge_distribution)

    def forward(self, pred, edge):
        """Forward function.

        Args:
            pred (Tuple): Batched predicted rboxes
            edge (torch.Tensor): The edge map with shape (B, 1, H, W).

        Returns:
            torch.Tensor: The calculated loss
        """
        G = self.resolution
        C = self.center_idx
        roi = rbbox2roi(pred)
        roi[:, 3:5] *= self.max_scale
        feat = self.roi_extractor([edge], roi)
        if len(feat) == 0:
            return pred[0].new_tensor(0)
        featx = feat.sum(1).abs().sum(1)
        featy = feat.sum(1).abs().sum(2)
        featx2 = torch.flip(featx[:, :G + 1], (-1,)) + featx[:, G:]
        featy2 = torch.flip(featy[:, :G + 1], (-1,)) + featy[:, G:]  # (N, 25)
        ex = ((featx2 * self.edge_distribution).softmax(1) * self.edge_idx).sum(1) / C
        ey = ((featy2 * self.edge_distribution).softmax(1) * self.edge_idx).sum(1) / C
        exy = torch.stack((ex, ey), -1)
        rbbox_concat = torch.cat(pred, 0)
        
        if self.debug:
            edgex = featx[:, None, None, :].expand(-1, 1, 2 * self.resolution + 1, -1)
            edgey = featy[:, None, :, None].expand(-1, 1, -1, 2 * self.resolution + 1)
            plot_edge_map(feat, edgex, edgey)

        return self.loss_weight * F.smooth_l1_loss(rbbox_concat[:, 2:4], 
                                      (rbbox_concat[:, 2:4] * exy).detach(),
                                      beta=8)


@MODELS.register_module()
class Point2RBoxV2ConsistencyLoss(nn.Module):
    """Consistency Loss.

    Args:
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'. Options are "none", "mean" and
            "sum".
        loss_weight (float, optional): Weight of loss. Defaults to 1.0.

    Returns:
        loss (torch.Tensor)
    """

    def __init__(self,
                 reduction='mean',
                 loss_weight=1.0):
        super(Point2RBoxV2ConsistencyLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, ori_pred, trs_pred, square_mask, aug_type, aug_val):
        """Forward function.

        Args:
            ori_pred (Tuple): (Sigma, theta)
            trs_pred (Tuple): (Sigma, theta)
            square_mask: When True, the angle is ignored
            aug_type: 'rot', 'flp', 'sca'
            aug_val: Rotation or scale value

        Returns:
            torch.Tensor: The calculated loss
        """
        ori_gaus, ori_angle = ori_pred
        trs_gaus, trs_angle = trs_pred

        if aug_type == 'rot':
            rot = ori_gaus.new_tensor(aug_val)
            cos_r = torch.cos(rot)
            sin_r = torch.sin(rot)
            R = torch.stack((cos_r, -sin_r, sin_r, cos_r), dim=-1).reshape(-1, 2, 2)
            ori_gaus = R.matmul(ori_gaus).matmul(R.permute(0, 2, 1))
            d_ang = trs_angle - ori_angle - aug_val
        elif aug_type == 'flp':
            ori_gaus = ori_gaus * ori_gaus.new_tensor((1, -1, -1, 1)).reshape(2, 2)
            d_ang = trs_angle + ori_angle
        else:
            sca = ori_gaus.new_tensor(aug_val)
            ori_gaus = ori_gaus * sca
            d_ang = trs_angle - ori_angle
        
        loss_ssg = gwd_sigma_loss(ori_gaus.bmm(ori_gaus), trs_gaus.bmm(trs_gaus))
        d_ang = (d_ang + math.pi / 2) % math.pi - math.pi / 2
        loss_ssa = F.smooth_l1_loss(d_ang, torch.zeros_like(d_ang), reduction='none', beta=0.1)
        loss_ssa = loss_ssa[~square_mask].sum() / max(1, (~square_mask).sum())

        return self.loss_weight * (loss_ssg + loss_ssa)
