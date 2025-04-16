import torch
from models import UNet, UNetEquiconv

# 预训练模型文件URL
_MODEL_URL = 'https://github.com/JesseSenior/semseg-outdoor-pano/releases/download/v1.0.0/equiconv.pth.tar'
_LAYER_URL = 'https://github.com/JesseSenior/semseg-outdoor-pano/releases/download/v1.0.0/layer_256x512.pt'
_OFFSET_URL = 'https://github.com/JesseSenior/semseg-outdoor-pano/releases/download/v1.0.0/offset_256x512.pt'

def unet_equiconv(n_classes=3, pretrained=False, 
                 layer_dict=None, offset_dict=None, **kwargs):
    """带等距卷积的UNet模型 - 360度图像处理优化版
    
    Args:
        n_classes (int): 输出类别数，默认为3
        pretrained (bool): 是否加载预训练参数（需要配置layer/offset文件）
        layer_dict: 预训练层配置
        offset_dict: 卷积偏移配置
        **kwargs: 其他模型参数
        
    返回:
        UNetEquiconv实例
    """
    # 自动加载默认配置
    if layer_dict is None or offset_dict is None:
        layer_dict = torch.hub.load_state_dict_from_url(_LAYER_URL)
        offset_dict = torch.hub.load_state_dict_from_url(_OFFSET_URL)
    
    model = UNetEquiconv(n_class=n_classes, 
                        layer_dict=layer_dict, 
                        offset_dict=offset_dict, **kwargs)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(_MODEL_URL, map_location='cpu')
        model.load_state_dict(state_dict)
    return model
