"""
visualize_boundary.py
=====================
对指定 case 的每张切片，并排可视化多种边界提取算法结果，保存为单张 PNG。

算法列表
--------
  1. 4-Diff  (RBE)   : B(v_i) = Σ_{j∈N4} |v_i − v_j|
  2. 8-Diff          : B(v_i) = Σ_{j∈N8} |v_i − v_j|（含对角方向）
  3. Sobel-X         : ∂I/∂x（Sobel 水平算子响应）
  4. Sobel-Y         : ∂I/∂y（Sobel 垂直算子响应）
  5. Sobel-Mag       : √(Sx²+Sy²)
  6. Prewitt-Mag     : √(Px²+Py²)
  7. Scharr-Mag      : √(Cx²+Cy²)（对角方向更精确）
  8. Roberts-Cross   : √((I[i,j]−I[i+1,j+1])²+(I[i+1,j]−I[i,j+1])²)
  9. Laplacian       : ΔI（二阶微分，绝对值显示）
 10. LoG             : Laplacian of Gaussian（高斯平滑后取拉普拉斯）
 11. Canny           : 自适应双阈值边缘（二值输出）

布局
----
| Image | 4-Diff | 8-Diff | Sobel-X | Sobel-Y | Sobel-Mag |
| Prewitt-Mag | Scharr-Mag | Roberts | Laplacian | LoG | Canny |

用法
----
python visualize_boundary.py \\
    --data_dir ./datasets/ABDOMINAL/processed_DDFP/ \\
    --domain   BTCV \\
    --case_id  0001 \\
    --split    test \\
    --metadata ./datasets/ABDOMINAL/processed_DDFP/metadata.json \\
    [--img_size 256] \\
    [--save_dir /path/to/output] \\
    [--canny_sigma 1.0]
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter, convolve
from skimage.feature import canny
from tqdm import tqdm
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataloaders.dataset_CSANet import CSANet_SliceDataset, RandomGenerator_new


# ──────────────────────────────────────────────────────────────────────────────
# 各边界提取算法（输入均为 float32 (H,W)，已归一化到 [0,1]）
# 输出均为 float32 (H,W)，未必归一化，由绘制函数按像素范围显示
# ──────────────────────────────────────────────────────────────────────────────

def _pad_edge(img: np.ndarray, n: int = 1) -> np.ndarray:
    return np.pad(img, n, mode="edge")


def boundary_4diff(img: np.ndarray) -> np.ndarray:
    """4邻域差分绝对值之和（RBE 原始定义）。"""
    p = _pad_edge(img)
    return (
        np.abs(img - p[:-2, 1:-1])  +   # up
        np.abs(img - p[2:,  1:-1])  +   # down
        np.abs(img - p[1:-1, :-2])  +   # left
        np.abs(img - p[1:-1, 2:])       # right
    )


def boundary_8diff(img: np.ndarray) -> np.ndarray:
    """8邻域差分绝对值之和（含对角方向）。"""
    p = _pad_edge(img)
    result = np.zeros_like(img)
    for di in range(3):
        for dj in range(3):
            if di == 1 and dj == 1:
                continue  # 跳过中心点自身
            result += np.abs(img - p[di: di + img.shape[0], dj: dj + img.shape[1]])
    return result


def _conv2d(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """2D 卷积（scipy），保持边界。"""
    return convolve(img.astype(np.float64), kernel, mode="nearest").astype(np.float32)


# Sobel kernels
_SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64) / 8.0
_SOBEL_Y = _SOBEL_X.T

def boundary_sobel_x(img: np.ndarray) -> np.ndarray:
    return np.abs(_conv2d(img, _SOBEL_X))

def boundary_sobel_y(img: np.ndarray) -> np.ndarray:
    return np.abs(_conv2d(img, _SOBEL_Y))

def boundary_sobel_mag(img: np.ndarray) -> np.ndarray:
    sx = _conv2d(img, _SOBEL_X)
    sy = _conv2d(img, _SOBEL_Y)
    return np.sqrt(sx ** 2 + sy ** 2)


# Prewitt kernels
_PREWITT_X = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float64) / 6.0
_PREWITT_Y = _PREWITT_X.T

def boundary_prewitt_mag(img: np.ndarray) -> np.ndarray:
    px = _conv2d(img, _PREWITT_X)
    py = _conv2d(img, _PREWITT_Y)
    return np.sqrt(px ** 2 + py ** 2)


# Scharr kernels（对角方向精度更高）
_SCHARR_X = np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], dtype=np.float64) / 32.0
_SCHARR_Y = _SCHARR_X.T

def boundary_scharr_mag(img: np.ndarray) -> np.ndarray:
    cx = _conv2d(img, _SCHARR_X)
    cy = _conv2d(img, _SCHARR_Y)
    return np.sqrt(cx ** 2 + cy ** 2)


# Roberts Cross
def boundary_roberts(img: np.ndarray) -> np.ndarray:
    """Roberts 交叉算子梯度幅值。"""
    p = _pad_edge(img)
    g_diag1 = img - p[2:,  2:][: img.shape[0], : img.shape[1]]   # I[i,j] - I[i+1,j+1]
    g_diag2 = p[2:, :-2][: img.shape[0], : img.shape[1]] - p[:-2, 2:][: img.shape[0], : img.shape[1]]  # I[i+1,j] - I[i,j+1]
    return np.sqrt(g_diag1 ** 2 + g_diag2 ** 2).astype(np.float32)


# Laplacian（4邻域离散拉普拉斯）
_LAPLACIAN_KERNEL = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)

def boundary_laplacian(img: np.ndarray) -> np.ndarray:
    return np.abs(_conv2d(img, _LAPLACIAN_KERNEL))


# Laplacian of Gaussian
def boundary_log(img: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """高斯平滑后取拉普拉斯（绝对值）。"""
    smoothed = gaussian_filter(img.astype(np.float64), sigma=sigma)
    lap = _conv2d(smoothed.astype(np.float32), _LAPLACIAN_KERNEL)
    return np.abs(lap)


# Canny（输出二值 float32）
def boundary_canny(img: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """scikit-image Canny，输出 float32 {0, 1}。"""
    edges = canny(img.astype(np.float64), sigma=sigma)
    return edges.astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# 方法注册表：(显示名称, 调用函数)
# ──────────────────────────────────────────────────────────────────────────────

def build_methods(canny_sigma: float = 1.0, log_sigma: float = 1.0):
    """返回 [(name, fn), ...] 列表，fn 接受 (H,W) float32 图像。"""
    return [
        ("4-Diff (RBE)",   boundary_4diff),
        ("8-Diff",         boundary_8diff),
        ("Sobel-X",        boundary_sobel_x),
        ("Sobel-Y",        boundary_sobel_y),
        ("Sobel-Mag",      boundary_sobel_mag),
        ("Prewitt-Mag",    boundary_prewitt_mag),
        ("Scharr-Mag",     boundary_scharr_mag),
        ("Roberts",        boundary_roberts),
        ("Laplacian",      boundary_laplacian),
        (f"LoG σ={log_sigma}",   lambda img, s=log_sigma: boundary_log(img, s)),
        (f"Canny σ={canny_sigma}", lambda img, s=canny_sigma: boundary_canny(img, s)),
    ]


# ──────────────────────────────────────────────────────────────────────────────
# 可视化与保存
# ──────────────────────────────────────────────────────────────────────────────

def save_boundary_comparison(
    image: np.ndarray,
    label: np.ndarray,
    results: list,          # list of (name: str, map: np.ndarray)
    save_dir: str,
    filename_prefix: str,
    ncols: int = 6,
) -> None:
    """将原图与各边界图以网格形式并排保存为单张 PNG。

    Parameters
    ----------
    image   : (H, W) float32 原始切片
    label   : (H, W) int64   分割标签（用于轮廓叠加）
    results : [(方法名, 边界图), ...]
    ncols   : 每行子图数（含原图列），默认 6
    """
    os.makedirs(save_dir, exist_ok=True)

    n_methods = len(results)
    n_panels  = 1 + n_methods            # 原图 + 各方法
    nrows     = (n_panels + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * 2.8, nrows * 2.8),
        dpi=150,
    )
    axes_flat = np.array(axes).reshape(-1)

    # ── 第 0 格：原图 + label 轮廓 ────────────────────────────────────────────
    ax0 = axes_flat[0]
    ax0.imshow(image, cmap="gray", interpolation="nearest", vmin=0.0, vmax=1.0)
    _draw_label_contours(ax0, label)
    ax0.set_title("Image\n+ Label Contours", fontsize=7)
    ax0.set_axis_off()

    # ── 后续格：各边界图 ──────────────────────────────────────────────────────
    for k, (name, bmap) in enumerate(results):
        ax = axes_flat[k + 1]
        # 对每张图独立做 min-max 归一化以便视觉对比
        vmin, vmax = bmap.min(), bmap.max()
        if vmax > vmin:
            display = (bmap - vmin) / (vmax - vmin)
        else:
            display = bmap
        cmap = "gray" if name.startswith("Canny") else "hot"
        ax.imshow(display, cmap=cmap, interpolation="nearest", vmin=0.0, vmax=1.0)
        ax.set_title(name, fontsize=7)
        ax.set_axis_off()

    # ── 多余格隐藏 ────────────────────────────────────────────────────────────
    for ax in axes_flat[n_panels:]:
        ax.set_visible(False)

    fig.suptitle(filename_prefix, fontsize=8, y=1.01)
    fig.tight_layout(pad=0.3, h_pad=2.5)

    out_path = os.path.join(save_dir, f"{filename_prefix}.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# 复用 visualize_rbe.py 的轮廓颜色定义
_CONTOUR_COLORS = [
    "#44FF44", "#FFFF00", "#FF4444", "#4488FF",
    "#FF44FF", "#44FFFF", "#FF8800", "#AA44FF",
]

def _draw_label_contours(ax, label: np.ndarray) -> None:
    classes = sorted(c for c in np.unique(label) if c != 0)
    for c in classes:
        color  = _CONTOUR_COLORS[(c - 1) % len(_CONTOUR_COLORS)]
        binary = (label == c).astype(np.float32)
        try:
            ax.contour(binary, levels=[0.5], colors=[color], linewidths=0.8)
        except Exception:
            pass


# ──────────────────────────────────────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="多种边界提取算法可视化对比"
    )
    parser.add_argument("--data_dir",    type=str,
                        default="./datasets/ABDOMINAL/processed_DDFP/",
                        help="数据集根目录（含各 domain 子目录）")
    parser.add_argument("--domain",      type=str, default="BTCV",
                        help="目标域名，例如 CHAOST2 / BTCV")
    parser.add_argument("--case_id",     type=str, default="0001",
                        help="要可视化的 case ID，例如 0001")
    parser.add_argument("--split",       type=str, default="test",
                        help="数据集划分: train / test（默认 test）")
    parser.add_argument("--metadata",    type=str, default=None,
                        help="metadata.json 路径（不指定则自动在 data_dir 下查找）")
    parser.add_argument("--img_size",    type=int, default=256,
                        help="Resize 分辨率（默认 256）")
    parser.add_argument("--save_dir",    type=str, default=None,
                        help="PNG 保存目录（默认 ./boundary_vis/{domain}/{case_id}/）")
    parser.add_argument("--ncols",       type=int, default=6,
                        help="每行子图数，含原图列（默认 6）")
    parser.add_argument("--canny_sigma", type=float, default=1.0,
                        help="Canny 高斯平滑 σ（默认 1.0）")
    parser.add_argument("--log_sigma",   type=float, default=1.0,
                        help="LoG 高斯平滑 σ（默认 1.0）")
    args = parser.parse_args()

    # ── 路径推断 ──────────────────────────────────────────────────────────────
    if args.metadata is None:
        args.metadata = os.path.join(args.data_dir, "metadata.json")

    if args.save_dir is None:
        args.save_dir = os.path.join(
            "boundary_vis", args.domain, args.case_id
        )

    os.makedirs(args.save_dir, exist_ok=True)

    # ── 1. 加载 metadata ──────────────────────────────────────────────────────
    with open(args.metadata, "r") as f:
        metadata = json.load(f)

    # ── 2. 构建 Dataset（val 模式：仅 Resize）────────────────────────────────
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

    # ── 3. 筛选目标 case 的切片，按 z 序排列 ─────────────────────────────────
    case_slices = []
    for idx in range(len(db)):
        sample = db[idx]
        if sample["case_name"] != args.case_id:
            continue
        z    = int(sample["slice_name"])
        img  = sample["image"]
        mask = sample["mask"]
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

    # ── 4. 构建方法列表 ───────────────────────────────────────────────────────
    methods = build_methods(
        canny_sigma=args.canny_sigma,
        log_sigma=args.log_sigma,
    )
    print(f"[✓] 共 {len(methods)} 种边界提取算法: {[m[0] for m in methods]}")

    # ── 5. 逐切片计算并保存 ───────────────────────────────────────────────────
    for z_idx, img_np, mask_np in tqdm(case_slices, desc="Processing slices"):
        results = [(name, fn(img_np)) for name, fn in methods]

        prefix = f"{args.domain}_{args.case_id}_slice{z_idx:04d}"
        save_boundary_comparison(
            img_np, mask_np, results,
            save_dir=args.save_dir,
            filename_prefix=prefix,
            ncols=args.ncols,
        )

    print(f"\n[✓] 所有可视化结果已保存至: {args.save_dir}")


if __name__ == "__main__":
    main()
