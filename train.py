import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm  # 用于可视化进度条

import utils
from REM import RetinexEnhancementModule
from MainModel import MainModel
import LossCaculate as loss_fn


# 1. 超参数与配置 (默认值与论文一致，可通过命令行覆盖)
def parse_args():
    p = argparse.ArgumentParser(description="Train U2IE-Net")
    p.add_argument("--lr", type=float, default=0.0005)
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--save_interval", type=int, default=10, help="每隔多少个 Epoch 保存一次指定轮次权重")
    p.add_argument("--num_imgs", type=int, default=5000)
    p.add_argument("--u_path", default="./underwater", help="unpaired underwater images")
    p.add_argument("--h_path", default="./haze", help="hazy images (paired with --g_path by sorted filename)")
    p.add_argument("--g_path", default="./clear", help="clear images")
    p.add_argument("--output_dir", default="./OpenData")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--num_workers", type=int, default=min(4, os.cpu_count() or 1))
    p.add_argument("--resume", action="store_true",
                   help="continue from <output_dir>/checkpoints/u2ie_resume.pth if it exists")
    return p.parse_args()


# 2. 训练主逻辑
def train_model(model, train_loader, optimizer, num_epochs, save_dir, save_every, device,
                start_epoch=0, history=None):
    model.train()
    os.makedirs(save_dir, exist_ok=True)

    history = history or {"l1": [], "l2": [], "l3": [], "total": []}
    resume_path = os.path.join(save_dir, "u2ie_resume.pth")

    # 外层 Epoch 进度条
    epoch_bar = tqdm(range(start_epoch, num_epochs), desc="Training Progress", unit="epoch",
                     initial=start_epoch, total=num_epochs)

    for epoch in epoch_bar:
        running_loss1, running_loss2, running_loss3, running_total = 0.0, 0.0, 0.0, 0.0

        # 内层 Batch 进度条
        batch_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{num_epochs}",
            leave=False,
            unit="batch"
        )

        for U, H, G in batch_bar:
            U, H, G = U.to(device), H.to(device), G.to(device)

            # 打乱 U 的顺序
            U = U[torch.randperm(U.size(0))]

            # 前向计算
            t1, H2G, G2H, ref1s_h, new_ref1s_h, ill1s_h, new_ill1s_h = model(H=H, G=G)
            t2, new_ill1s_u, Uwb = model(U=U)

            # 损失函数计算
            l1 = (loss_fn.calculate_ssim_loss(ref1s_h, new_ref1s_h) +
                  loss_fn.calculate_ssim_loss(H2G, G) +
                  loss_fn.calculate_ssim_loss(G2H, H) +
                  0.15 * loss_fn.calculate_ssim_loss(ill1s_h, new_ill1s_h))

            l2 = 0.005 * (loss_fn.caculate_variation_loss(Uwb) +
                          loss_fn.caculate_variation_loss(t1) +
                          loss_fn.caculate_variation_loss(t2))

            l3 = loss_fn.caculate_mid_difference_loss(new_ill1s_u) + loss_fn.caculate_max_difference_loss(new_ill1s_u)

            totalloss = l1 + l2 + l3

            # 反向传播
            optimizer.zero_grad()
            totalloss.backward()
            optimizer.step()

            running_loss1 += l1.item()
            running_loss2 += l2.item()
            running_loss3 += l3.item()
            running_total += totalloss.item()

            # 实时更新 Batch 进度条的后缀 Loss 显示
            batch_bar.set_postfix({"Loss": f"{totalloss.item():.4f}"})

        num_batches = len(train_loader)
        history["l1"].append(running_loss1 / num_batches)
        history["l2"].append(running_loss2 / num_batches)
        history["l3"].append(running_loss3 / num_batches)
        history["total"].append(running_total / num_batches)

        # 1) 保存最新权重 (latest.pth)
        torch.save(model.state_dict(), os.path.join(save_dir, "u2ie_latest.pth"))

        # 2) 按指定轮次间隔保存指定 checkpoint
        current_epoch = epoch + 1
        if current_epoch % save_every == 0 or current_epoch == num_epochs:
            torch.save(model.state_dict(), os.path.join(save_dir, f"u2ie_epoch_{current_epoch}.pth"))

        # 3) 断点续训状态 (Colab 断线后可用 --resume 继续)
        tmp_path = resume_path + ".tmp"
        torch.save({
            "epoch": current_epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "history": history,
        }, tmp_path)
        os.replace(tmp_path, resume_path)

        # 更新 Epoch 进度条后缀
        epoch_bar.set_postfix({
            "L1": f"{history['l1'][-1]:.3f}",
            "L2": f"{history['l2'][-1]:.3f}",
            "L3": f"{history['l3'][-1]:.3f}",
            "Total": f"{history['total'][-1]:.3f}"
        })

    return history["l1"], history["l2"], history["l3"], history["total"]

# 绘制训练曲线函数
def plot_loss_curves(l1, l2, l3, total, save_path):
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.figure(figsize=(16, 5))
    epochs_range = range(1, len(total) + 1)

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, l1, label='SSIM Loss (L1)', linewidth=2)
    plt.plot(epochs_range, l2, label='Variation Loss (L2)', linewidth=2)
    plt.plot(epochs_range, l3, label='Illumination Loss (L3)', linewidth=2)
    plt.title('Sub-Loss Curves', fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, total, label='Total Loss', color='crimson', linewidth=2)
    plt.title('Total Loss Curve', fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

if __name__ == '__main__':
    args = parse_args()
    new_size = (args.image_size, args.image_size)
    loss_dir = os.path.join(args.output_dir, "lossdata")
    ckpt_dir = os.path.join(args.output_dir, "checkpoints")

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f">>> 使用设备: {device}")

    # 数据路径与加载
    print(">>> 正在读取并构建数据集...")
    paths = [args.u_path, args.h_path, args.g_path]
    train_dataset = utils.build_data(paths, new_size=new_size, num_imgs=args.num_imgs)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,     # 开启锁页内存，加快 CPU 到 GPU 的张量传输速度
        persistent_workers=args.num_workers > 0 # 保持 Worker 进程不销毁，避免每个 Epoch 重新创建进程的开销
    )
    print(f">>> 数据集加载完成，共计 {len(train_dataset)} 样本，{len(train_loader)} 个 Batch。")

    # 初始化模型与优化器
    rem = RetinexEnhancementModule()
    model = MainModel(rem_module=rem).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    start_epoch, history = 0, None
    resume_path = os.path.join(ckpt_dir, "u2ie_resume.pth")
    if args.resume and os.path.exists(resume_path):
        state = torch.load(resume_path, map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch, history = state["epoch"], state["history"]
        print(f">>> 从 {resume_path} 恢复训练，已完成 {start_epoch} 个 Epoch")

    # 执行训练
    l1, l2, l3, total = train_model(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        num_epochs=args.epochs,
        save_dir=ckpt_dir,
        save_every=args.save_interval,
        device=device,
        start_epoch=start_epoch,
        history=history
    )

    # 5. 保存 Loss 日志 CSV
    os.makedirs(loss_dir, exist_ok=True)
    loss_save_path = os.path.join(loss_dir, "training_loss.csv")
    df = pd.DataFrame({
        'Epoch': list(range(1, len(total) + 1)),
        'Loss1': l1,
        'Loss2': l2,
        'Loss3': l3,
        'Total_Loss': total
    })
    df.to_csv(loss_save_path, index=False)
    print(f"\n>>> 训练完成！Loss 数据已保存至: {loss_save_path}")

    # 6. 保存最终轮次模型 (保留您原先命名的权重文件路径)
    final_pth_path = os.path.join(ckpt_dir, "u2ie.pth")
    torch.save(model.state_dict(), final_pth_path)
    print(f">>> 最终权重已保存至: {final_pth_path}")

    # 7. 可视化绘制并保存 Losses 曲线图
    chart_save_path = os.path.join(loss_dir, "training_loss_curve.png")
    plot_loss_curves(l1, l2, l3, total, chart_save_path)
    print(f">>> 损失曲线图已生成并保存至: {chart_save_path}")
