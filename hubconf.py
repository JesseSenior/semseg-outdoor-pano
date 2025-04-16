import torch
import torch.nn as nn
import torch.nn.functional as F
from models import UNetEquiconv

_MODEL_URL = "https://github.com/JesseSenior/semseg-outdoor-pano/releases/download/v1.0.0/equiconv.pth.tar"
_LAYER_URL = "https://github.com/JesseSenior/semseg-outdoor-pano/releases/download/v1.0.0/layer_256x512.pt"
_OFFSET_URL = "https://github.com/JesseSenior/semseg-outdoor-pano/releases/download/v1.0.0/offset_256x512.pt"


class EquiSegModel(nn.Module):
    """封装后的全景分割模型，包含预处理和后处理"""

    def __init__(self, base_model, img_size=(256, 512)):
        super().__init__()
        self.base_model = base_model
        self.img_size = img_size
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def preprocess(self, x):
        """标准化并调整尺寸"""
        if x.shape[-2:] != self.img_size:
            x = F.interpolate(x, size=self.img_size, mode="bilinear", align_corners=True)
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def forward(self, x: torch.Tensor):
        if not torch.is_floating_point(x):
            x = x.float() / 255.0  # 自动归一化到0-1
        x = x.permute(2, 0, 1)[None, ...]

        original_shape = x.shape[-2:]
        x = self.preprocess(x)

        result = self.base_model(x[[0, 0]])[:1]
        result = F.interpolate(result, size=original_shape, mode="bilinear", align_corners=True)
        return result.squeeze().argmax(dim=0)


def unet_equiconv(pretrained=True, layer_dict=None, offset_dict=None, **kwargs):
    """带等距卷积的UNet模型 - 360度图像处理优化版

    Args:
        pretrained (bool): 是否加载预训练参数
        layer_dict: 预训练层配置（自动下载）
        offset_dict: 卷积偏移配置（自动下载）
        **kwargs: 其他模型参数

    返回:
        EquiSegModel封装后的模型实例
    """
    # 自动加载默认配置
    if layer_dict is None:
        layer_dict = torch.hub.load_state_dict_from_url(_LAYER_URL)
    if offset_dict is None:
        offset_dict = torch.hub.load_state_dict_from_url(_OFFSET_URL)

    n_classes = 8
    img_size = (256, 512)
    base_model = UNetEquiconv(n_class=n_classes, layer_dict=layer_dict, offset_dict=offset_dict, **kwargs)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(_MODEL_URL, map_location="cpu")
        base_model.load_state_dict(state_dict)

    return EquiSegModel(base_model, img_size=img_size)
