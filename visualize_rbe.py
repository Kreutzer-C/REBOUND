"""
visualize_rbe.py
================
可视化 RBE-Theory 所需的两类核心特征图：

  1. 空间差分图（Structural Boundary Map）
     B(v_i) = Σ_{j∈N4} |v_i − v_j|

  2. 预测香农熵图（Predictive Entropy Map）
     H(P_i) = −Σ_c p_ic · log(p_ic)

脚本会对指定 case 的所有切片逐一生成一张并排三联 PNG：
  {save_dir}/{domain}_{case_id}_slice{z:04d}.png
  布局：| Image + Label Contours | Boundary Map | Entropy Map |

用法
----
python visualize_rbe.py \\
    --data_dir  /opt/data/private/REBOUND/datasets/ABDOMINAL/processed \\
    --domain    CHAOST2 \\
    --case_id   0001 \\
    --split     test \\
    --metadata  /opt/data/private/REBOUND/datasets/ABDOMINAL/processed/metadata.json \\
    --model_cfg /opt/data/private/REBOUND/networks/R50_ViTB16_config.json \\
    --ckpt      /path/to/checkpoint.pth \\
    --img_size  256 \\
    --save_dir  /opt/data/private/REBOUND/results/rbe_vis \\
    [--model_type CSANet]            # 可选: CSANet / CSANet_V2 / CSANet_V3
    [--device cuda:0]
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# ── 项目根目录已在 sys.path 中（脚本与 networks/ 同级）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataloaders.dataset_CSANet import CSANet_SliceDataset, RandomGenerator_new
from networks import CSANet, CSANet_V2, CSANet_V3
from utils import load_config_as_namespace


# ──────────────────────────────────────────────────────────────────────────────
# 差分图
# ──────────────────────────────────────────────────────────────────────────────

def compute_boundary_map(image: np.ndarray) -> np.ndarray:
    """
    计算四邻域空间差分绝对值之和。

    Parameters
    ----------
    image : np.ndarray, shape (H, W), float32
        已归一化到 [0, 1] 的单通道切片。

    Returns
    -------
    np.ndarray, shape (H, W), float32
        B(v_i) = |v_i − v_up| + |v_i − v_down| + |v_i − v_left| + |v_i − v_right|
        边界像素使用复制填充（等价于差分为 0 的方向不贡献）。
    """
    img = image.astype(np.float32)

    # 上下左右平移（边缘复制填充）
    up    = np.pad(img[:-1, :], ((1, 0), (0, 0)), mode='edge')   # v_up
    down  = np.pad(img[1:,  :], ((0, 1), (0, 0)), mode='edge')   # v_down
    left  = np.pad(img[:, :-1], ((0, 0), (1, 0)), mode='edge')   # v_left
    right = np.pad(img[:, 1:],  ((0, 0), (0, 1)), mode='edge')   # v_right

    boundary = (
        np.abs(img - up)    +
        np.abs(img - down)  +
        np.abs(img - left)  +
        np.abs(img - right)
    )
    return boundary


# ──────────────────────────────────────────────────────────────────────────────
# 熵图
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_pred_and_entropy(
    prev_img: np.ndarray,
    curr_img: np.ndarray,
    next_img: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    img_size: int,
) -> tuple:
    """
    用模型预测给定切片三元组，同时返回预测标签图和香农熵图。

    Parameters
    ----------
    prev_img, curr_img, next_img : np.ndarray, shape (H, W)
    model : 已 eval() 的分割模型
    device : torch 设备
    img_size : 模型期望的输入分辨率

    Returns
    -------
    pred : np.ndarray, shape (H, W), int64
        argmax 预测类别图
    entropy : np.ndarray, shape (H, W), float32
        H(P_i) = −Σ_c p_ic · log(p_ic)，单位 nats（log 以 e 为底）
    """
    H, W = curr_img.shape[:2]
    need_resize = (H != img_size or W != img_size)

    def to_tensor(arr: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(arr.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
        if need_resize:
            t = F.interpolate(t, size=(img_size, img_size), mode='bilinear', align_corners=False)
        return t

    logits = model(to_tensor(prev_img), to_tensor(curr_img), to_tensor(next_img))
    # logits: (1, C, H', W')
    probs = torch.softmax(logits, dim=1)  # (1, C, H', W')

    # 预测标签：argmax
    pred_t = torch.argmax(probs, dim=1)   # (1, H', W')

    # 香农熵 H = -Σ p·log(p)，clamp 避免 log(0)
    eps = 1e-8
    entropy_t = -(probs * torch.log(probs.clamp(min=eps))).sum(dim=1)  # (1, H', W')

    # 恢复到原始分辨率
    if need_resize:
        pred_t = F.interpolate(
            pred_t.unsqueeze(0).float(), size=(H, W), mode='nearest'
        ).squeeze(0)
        entropy_t = F.interpolate(
            entropy_t.unsqueeze(0), size=(H, W), mode='bilinear', align_corners=False
        ).squeeze(0)

    pred    = pred_t.squeeze(0).cpu().numpy().astype(np.int64)
    entropy = entropy_t.squeeze(0).cpu().numpy().astype(np.float32)
    return pred, entropy


# ──────────────────────────────────────────────────────────────────────────────
# 保存并排三联图（含 label 轮廓叠加）
# ──────────────────────────────────────────────────────────────────────────────

# 多类别轮廓颜色（跳过背景 class-0）
_CONTOUR_COLORS = [
    "#44FF44",  # class 1 — 绿
    "#FFFF00",  # class 2 — 黄
    "#FF4444",  # class 3 — 红
    "#4488FF",  # class 4 — 蓝
    "#FF44FF",  # class 5 — 洋红
    "#44FFFF",  # class 6 — 青
    "#FF8800",  # class 7 — 橙
    "#AA44FF",  # class 8 — 紫
]


def _draw_label_contours(ax, label: np.ndarray) -> None:
    """在 ax 上为每个前景类别绘制外轮廓线。"""
    classes = sorted(c for c in np.unique(label) if c != 0)
    for c in classes:
        color = _CONTOUR_COLORS[(c - 1) % len(_CONTOUR_COLORS)]
        binary = (label == c).astype(np.float32)
        # contour level=0.5 恰好提取二值掩膜边缘
        try:
            ax.contour(binary, levels=[0.5], colors=[color], linewidths=1.0)
        except Exception:
            pass  # 该类别面积为零，跳过


def _draw_pred_overlay(ax, image: np.ndarray, pred: np.ndarray) -> None:
    """在 ax 上将预测掩膜以半透明彩色面积叠加到灰度原图上。

    背景（class 0）保持透明，前景各类别使用 _CONTOUR_COLORS 着色，alpha=0.45。
    """
    ax.imshow(image, cmap="gray", interpolation="nearest", vmin=0.0, vmax=1.0)

    H, W = pred.shape
    # 构造 RGBA 覆盖层：背景全透明
    overlay = np.zeros((H, W, 4), dtype=np.float32)
    classes = sorted(c for c in np.unique(pred) if c != 0)
    for c in classes:
        hex_color = _CONTOUR_COLORS[(c - 1) % len(_CONTOUR_COLORS)]
        # 将 hex 颜色解析为 RGB float
        r = int(hex_color[1:3], 16) / 255.0
        g = int(hex_color[3:5], 16) / 255.0
        b = int(hex_color[5:7], 16) / 255.0
        mask = pred == c
        overlay[mask, 0] = r
        overlay[mask, 1] = g
        overlay[mask, 2] = b
        overlay[mask, 3] = 0.45  # alpha

    ax.imshow(overlay, interpolation="nearest")


def save_combined_png(
    image: np.ndarray,
    label: np.ndarray,
    pred: np.ndarray,
    boundary: np.ndarray,
    entropy: np.ndarray,
    save_dir: str,
    filename_prefix: str,
) -> None:
    """将原图（含 label 轮廓）、预测叠加图、差分图、熵图并排保存为单张 PNG。

    布局：| Image + Label Contours | Image + Pred Overlay | Boundary Map | Entropy Map |
    """
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4), dpi=150)

    # ── 1：原图 + label 轮廓 ──────────────────────────────────────────────────
    axes[0].imshow(image, cmap="gray", interpolation="nearest", vmin=0.0, vmax=1.0)
    _draw_label_contours(axes[0], label)
    axes[0].set_title("Image + Label Contours", fontsize=9)
    axes[0].set_axis_off()

    # ── 2：原图 + 预测面积叠加 ────────────────────────────────────────────────
    _draw_pred_overlay(axes[1], image, pred)
    axes[1].set_title("Image + Pred Overlay", fontsize=9)
    axes[1].set_axis_off()

    # ── 3：差分图 ─────────────────────────────────────────────────────────────
    im1 = axes[2].imshow(boundary, cmap="hot", interpolation="nearest")
    fig.colorbar(im1, ax=axes[2], fraction=0.046, pad=0.04)
    axes[2].set_title("Boundary Map  B(v)", fontsize=9)
    axes[2].set_axis_off()

    # ── 4：熵图 ───────────────────────────────────────────────────────────────
    im2 = axes[3].imshow(entropy, cmap="plasma", interpolation="nearest")
    fig.colorbar(im2, ax=axes[3], fraction=0.046, pad=0.04)
    axes[3].set_title("Entropy Map  H(P)", fontsize=9)
    axes[3].set_axis_off()

    fig.suptitle(filename_prefix, fontsize=8, y=1.01)
    fig.tight_layout(pad=0.4)

    out_path = os.path.join(save_dir, f"{filename_prefix}.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────────────────────────────────────

def build_model(model_type: str, model_cfg: dict, ckpt_path: str, device: torch.device):
    """加载指定类型的分割模型并返回 eval 状态的实例。"""
    # config = load_config_as_namespace(model_cfg)

    model_map = {"CSANet": CSANet, "CSANet_V2": CSANet_V2, "CSANet_V3": CSANet_V3}
    if model_type not in model_map:
        raise ValueError(f"未知 model_type='{model_type}'，可选: {list(model_map.keys())}")

    model = model_map[model_type](model_cfg).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    # 支持 DataParallel 保存的权重（module. 前缀）
    if all(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    print(f"[✓] 模型 {model_type} 已从 {ckpt_path} 加载")
    return model


def main():
    parser = argparse.ArgumentParser(
        description="可视化 RBE-Theory 差分图与预测熵图"
    )
    parser.add_argument("--data_dir",   type=str, default="./datasets/ABDOMINAL/processed_DDFP/",  help="数据集根目录（含各 domain 子目录）")
    parser.add_argument("--domain",     type=str, default="CHAOST2",  help="目标域名，例如 CHAOST2 / BTCV")
    parser.add_argument("--case_id",    type=str, default="0001",  help="要可视化的 case ID，例如 0001")
    parser.add_argument("--split",      default="test", help="数据集划分: train / test（默认 test）")
    parser.add_argument("--exp_path",   type=str, required=True, help="实验目录路径")
    parser.add_argument("--img_size",   type=int, default=256, help="模型输入分辨率（默认 256）")
    parser.add_argument("--save_dir",   default=None,  help="PNG 保存目录")
    parser.add_argument("--model_type", default="CSANet",
                        choices=["CSANet", "CSANet_V2", "CSANet_V3"],
                        help="模型类型（默认 CSANet）")
    parser.add_argument("--device",     default="cuda:0" if torch.cuda.is_available() else "cpu",
                        help="运行设备（默认 cuda:0）")
    args = parser.parse_args()

    metadata = os.path.join(args.exp_path, "configs_backup", "metadata.json")
    model_cfg = os.path.join(args.exp_path, "configs_backup", "model_config.json")
    ckpt = os.path.join(args.exp_path, "checkpoints", "best_checkpoint.pth")
    if args.save_dir is not None:
        args.save_dir = os.path.join(args.save_dir, args.domain)
        os.makedirs(args.save_dir, exist_ok=True)
    else:
        args.save_dir = os.path.join(args.exp_path, "rbe_vis")
        os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device(args.device)

    # ── 1. 加载 metadata ──────────────────────────────────────────────────────
    with open(metadata, "r") as f:
        metadata = json.load(f)

    # ── 2. 构建 Dataset（val 模式：仅 Resize，不做随机增强）────────────────────
    transform = RandomGenerator_new(
        output_size=(args.img_size, args.img_size),
        phase="val",
    )
    db = CSANet_SliceDataset(
        base_dir=args.data_dir,
        domain_name=args.domain,
        split=args.split,
        metadata=metadata,
        transform=transform,
    )
    print(f"[✓] Dataset: domain={args.domain}, split={args.split}, 共 {len(db)} 张切片")

    # ── 3. 筛选目标 case 的所有切片，按 z 序排列 ─────────────────────────────
    case_slices = []  # list of (z_idx, img_np, mask_np)
    for idx in range(len(db)):
        sample = db[idx]
        if sample["case_name"] != args.case_id:
            continue
        z   = int(sample["slice_name"])
        img  = sample["image"]
        mask = sample["mask"]
        # 转 numpy (H, W)
        img_np  = img.squeeze().numpy()  if isinstance(img,  torch.Tensor) else np.squeeze(img).astype(np.float32)
        mask_np = mask.squeeze().numpy() if isinstance(mask, torch.Tensor) else np.squeeze(mask).astype(np.int64)
        case_slices.append((z, img_np, mask_np))

    if not case_slices:
        print(
            f"[✗] 在 split='{args.split}' 中未找到 case_id='{args.case_id}'。\n"
            f"    请检查 --case_id 和 --split 是否正确。"
        )
        sys.exit(1)

    case_slices.sort(key=lambda t: t[0])
    print(f"[✓] Case {args.case_id}: 找到 {len(case_slices)} 张切片")

    # ── 4. 加载模型 ───────────────────────────────────────────────────────────
    model = build_model(args.model_type, model_cfg, ckpt, device)

    # ── 5. 逐切片计算差分图 + 熵图，保存并排 PNG ─────────────────────────────
    images_list = [s[1] for s in case_slices]
    N = len(images_list)

    for i, (z_idx, curr_img, mask) in enumerate(tqdm(case_slices, desc="Visualizing slices")):
        prev_img = images_list[max(0, i - 1)]
        next_img = images_list[min(N - 1, i + 1)]

        # 差分图
        boundary = compute_boundary_map(curr_img)

        # 预测图 + 熵图（同一次前向传播）
        pred, entropy = compute_pred_and_entropy(
            prev_img, curr_img, next_img,
            model, device, args.img_size,
        )

        # 并排四联图（label 轮廓 | 预测叠加 | 差分图 | 熵图）
        prefix = f"{args.domain}_{args.case_id}_slice{z_idx:04d}"
        case_save_dir = os.path.join(args.save_dir, args.case_id)
        os.makedirs(case_save_dir, exist_ok=True)
        save_combined_png(curr_img, mask, pred, boundary, entropy, case_save_dir, prefix)

    print(f"\n[✓] 所有可视化结果已保存至: {args.save_dir}")


if __name__ == "__main__":
    main()
