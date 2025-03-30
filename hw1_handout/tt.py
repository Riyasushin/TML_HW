def compute_cam_per_layer(
        self,
        input_tensor: torch.Tensor) -> np.ndarray:
    # 获取所有层的激活和梯度（保持设备一致性）
    activations_list = [a.detach().cpu().numpy() 
                       for a in self.activations_and_grads.activations]
    grads_list = [g.detach().cpu().numpy() 
                 for g in self.activations_and_grads.gradients]

    target_size = (input_tensor.size(-1), input_tensor.size(-2))
    cam_per_target_layer = []

    # 并行化处理各层
    for layer_idx, target_layer in enumerate(self.target_layers):
        # 安全获取对应层的激活和梯度
        try:
            layer_act = activations_list[layer_idx]
            layer_grad = grads_list[layer_idx]
        except IndexError:
            print(f"Skipping {target_layer}: Activations/grads not available")
            continue

        # 梯度全局平均池化 (GAP)
        weights = np.mean(layer_grad, axis=(2, 3), keepdims=True)
        
        # 计算类激活图 (带通道对齐)
        cam = np.sum(layer_act * weights, axis=1, keepdims=True)
        
        # 多尺度融合增强细节
        cam = self._multi_scale_refine(cam, target_size)

        # 非线性增强对比度
        cam = self._contrast_enhance(cam)

        # 标准化到 [0,1] 范围
        cam = self._normalize_cam(cam)
        
        cam_per_target_layer.append(cam)

    return np.stack(cam_per_target_layer)

def _multi_scale_refine(self, cam, target_size):
    """多尺度细化增强空间一致性"""
    refined = np.zeros_like(cam)
    for scale in [0.5, 1.0, 2.0]:
        scaled = F.interpolate(
            cam, 
            scale_factor=scale, 
            mode='bilinear',
            align_corners=False
        )
        back_scaled = F.interpolate(
            scaled,
            size=target_size,
            mode='bilinear',
            align_corners=False
        )
        refined += back_scaled
    return refined / 3.0

def _contrast_enhance(self, cam):
    """对比度增强（带自适应阈值）"""
    # 自适应直方图均衡化
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    return clahe.apply((cam * 255).astype(np.uint8)) / 255.0

def _normalize_cam(self, cam):
    """稳健的归一化处理"""
    eps = 1e-8
    cam_min = np.percentile(cam, 2)  # 使用2%分位数作为最小值
    cam_max = np.percentile(cam, 98) # 使用98%分位数作为最大值
    return np.clip((cam - cam_min) / (cam_max - cam_min + eps), 0, 1)