import torch
import argparse
import numpy as np
from PIL import Image


def main():
    # 参数设置
    parser = argparse.ArgumentParser(description="Panoramic Segmentation using torch.hub")
    parser.add_argument("--img-path", type=str, required=True, help="Input image path")
    parser.add_argument("--output", type=str, default="output.png", help="Output path")
    parser.add_argument("--img-height", type=int, default=256)
    parser.add_argument("--img-width", type=int, default=512)
    args = parser.parse_args()

    # 加载模型
    model = torch.hub.load(".", "unet_equiconv", pretrained=True, source="local").cuda()
    model.eval()

    # 加载并预处理图像
    img = Image.open(args.img_path).convert("RGB")
    img = torch.from_numpy(np.array(img)).cuda()

    # 推理
    with torch.no_grad():
        output = model(img)

    # 后处理
    pred = output.cpu().numpy()

    # 创建彩色分割图
    palette = np.array(
        [
            [0, 0, 0],
            [128, 0, 0],  # Flat
            [0, 128, 0],  # Construction
            [128, 128, 0],  # Object
            [0, 0, 128],  # Nature
            [128, 0, 128],  # Sky
            [0, 128, 128],  # Person
            [128, 128, 128],  # Vehicle
            [64, 0, 0],
        ],
        dtype=np.uint8,
    )

    colored_pred = Image.fromarray(palette[pred].astype(np.uint8))

    # 保存结果
    # os.makedirs(os.path.dirname(args.output), exist_ok=True)
    colored_pred.save(args.output)
    print(f"Saved segmentation result to {args.output}")


if __name__ == "__main__":
    main()
