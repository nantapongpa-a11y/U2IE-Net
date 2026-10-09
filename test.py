import os
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm

from REM import RetinexEnhancementModule
from MainModel import MainModel

# 1. 极速推理专用 Dataset
class InferenceDataset(Dataset):
    def __init__(self, img_dir, new_size=(256, 256)):
        self.img_dir = img_dir
        # 自动过滤常见的图像格式后缀
        valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
        self.img_paths = [
            os.path.join(img_dir, f) for f in os.listdir(img_dir)
            if f.lower().endswith(valid_exts)
        ]
        
        self.transform = T.Compose([
            T.Resize(new_size),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        filename = os.path.basename(img_path)
        img = Image.open(img_path).convert('RGB')
        return self.transform(img), filename


# 2. 推理主函数
def run_inference(input_dir, output_dir, checkpoint_path, new_size=(256, 256), batch_size=8, device_str='cuda:0', num_workers=4):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(device_str if torch.cuda.is_available() else 'cpu')

    # 初始化模型结构
    print(f">>> 正在初始化模型并加载权重: {checkpoint_path}")
    rem = RetinexEnhancementModule()
    model = MainModel(rem_module=rem).to(device)
    
    # 加载权重与状态设置
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()  # 切换到评估模式

    # 构建数据装载器
    dataset = InferenceDataset(input_dir, new_size=new_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    print(f">>> 待处理图像总数: {len(dataset)}，开始推理...")

    # 执行推理计算 (不计算梯度)
    with torch.no_grad():
        for imgs, filenames in tqdm(loader, desc="Inference Progress", unit="batch"):
            imgs = imgs.to(device)

            outputs = model(U=imgs)
            # 提取去雾/水下增强结果 Uwb2G (索引 2) 或 Retinex 输出 Uwb (索引 -1)
            output_img = outputs[2]
            output_clamped = torch.clamp(output_img, 0.0, 1.0)

            # 逐张保存结果
            for i in range(len(filenames)):
                save_path = os.path.join(output_dir, filenames[i])
                save_image(output_clamped[i], save_path)

    print(f"\n>>> 推理完成！增强后的结果已保存至: {output_dir}")


if __name__ == '__main__':
    # ================= 配置推理参数 (可通过命令行覆盖) =================
    parser = argparse.ArgumentParser(description="U2IE-Net inference")
    parser.add_argument("--input_dir", default=r"./Inputs", help="输入图像目录")
    parser.add_argument("--output_dir", default=r"./Outputs", help="输出保存目录")
    parser.add_argument("--checkpoint", default=r"./checkpoints/u2ie_epoch_50.pth", help="模型权重路径")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_workers", type=int, default=min(4, os.cpu_count() or 1))
    args = parser.parse_args()
    # ================================================

    run_inference(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        checkpoint_path=args.checkpoint,
        new_size=(args.image_size, args.image_size),
        batch_size=args.batch_size,
        device_str=args.device,
        num_workers=args.num_workers
    )
