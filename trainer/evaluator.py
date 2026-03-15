"""
Evaluator for 2.5D medical image segmentation (CSANet / REBOUND) — slice-based.

Design
------
The evaluator consumes a :class:`~dataloaders.dataset_CSANet.CSANet_SliceDataset`
instance instead of a volume dataset.  All slices in the dataset are first
grouped by ``case_name`` and sorted by ``slice_name`` (z-index).  For each
case the ordered 2-D slices are fed through the model as 2.5D
(prev, curr, next) triplets; the per-slice predictions are stacked into a
3-D prediction volume from which Dice and ASSD are computed.

Advantages over the former volume-based approach
-------------------------------------------------
* Works correctly with any preprocessing pipeline regardless of whether the
  stored slice and volume spatial dimensions agree.
* When the slice spatial size differs from ``args.img_size``, each slice is
  bilinearly resampled to ``(img_size, img_size)`` for inference and the
  prediction is nearest-neighbour mapped back to the original size — all
  within :meth:`_infer_case`.

Saving (no reference-image mechanism)
--------------------------------------
When *save_predictions* is *True*, three aligned NIfTI files are written for
every case: **image**, **pred**, and **label**.  All three share identical
voxel spacing (from ``processing_log.csv``), origin ``(0, 0, 0)``, and
identity direction matrix.  They can therefore be overlaid directly in 3D
Slicer without any alignment step and without the original source volume as a
reference.

Intended usage
--------------
1. Per-epoch validation inside a Trainer::

       db_val    = CSANet_SliceDataset(
           data_dir, target, 'test', metadata,
           transform=…RandomGenerator_new(img_size, phase='val'),
       )
       # db_val is pre-loaded once at construction time
       evaluator = Evaluator(args, metadata, model, device, db_eval=db_val, logger=self.logger)
       # Each epoch: only inference + metric computation
       metrics   = evaluator.evaluate()

2. Standalone test evaluation (``test.py`` entry point)::

       db_test   = CSANet_SliceDataset(
           data_dir, target, 'test', metadata,
           transform=…RandomGenerator_new(img_size, phase='test'),
       )
       evaluator = Evaluator(args, metadata, model, device, db_eval=db_test)
       metrics   = evaluator.evaluate(
           save_predictions=True,
           save_dir='results/preds',
       )
"""

import csv
import os
from collections import defaultdict

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.metrics import compute_assd, compute_dice_per_class


class Evaluator:
    """Slice-based volume-level evaluator for a 2.5D segmentation model.

    All expensive preprocessing — iterating the dataset, grouping slices by
    case, sorting, converting to numpy, stacking into label volumes, and
    looking up per-case physical spacing — is performed **once** during
    ``__init__``.  Subsequent calls to :meth:`evaluate` only run model
    inference and metric computation.

    Parameters
    ----------
    args : argparse.Namespace
        Must contain at least ``data_dir`` and ``img_size``.
    metadata : dict
        Dataset metadata.  Must contain ``num_classes``.
    model : torch.nn.Module
        Segmentation model already moved to *device*.
    device : torch.device
    db_eval : CSANet_SliceDataset
        Slice-level dataset used for evaluation.  Pre-loaded at init time.
    logger : logging.Logger, optional
        Routes log messages; falls back to ``print`` when *None*.
    """

    def __init__(self, args, metadata, model, device, db_eval, logger=None):
        self.args        = args
        self.metadata    = metadata
        self.model       = model
        self.device      = device
        self.num_classes = metadata["num_classes"]
        self.logger      = logger

        self._spacing_map = self._load_spacing_map()

        # Domain / split info (fixed for the lifetime of this evaluator)
        self._domain_name = db_eval.domain_name
        self._split       = db_eval.split

        # Pre-load all data once
        self._prepare_cases(db_eval)

    # ------------------------------------------------------------------
    # One-time data preparation (called from __init__)
    # ------------------------------------------------------------------

    def _prepare_cases(self, db_eval) -> None:
        """Load every slice from *db_eval*, group by case, and cache results.

        After this method returns, the following instance attributes are
        populated and remain constant for the lifetime of the evaluator:

        ``_case_ids`` : list[str]
            Sorted list of case identifiers.
        ``_images_per_case`` : dict[str, list[np.ndarray]]
            Ordered 2-D float32 slices ``(H, W)`` per case.
        ``_label_vols`` : dict[str, np.ndarray]
            Stacked label volume ``(D, H, W)`` int64 per case.
        ``_image_vols`` : dict[str, np.ndarray]
            Stacked image volume ``(D, H, W)`` float32 per case
            (used when saving NIfTI predictions).
        ``_real_spacings`` : dict[str, tuple]
            Physical ``(z_sp, y_sp, x_sp)`` spacing in mm per case.
        """
        total_slices = len(db_eval)
        self._log(
            f"Pre-loading {total_slices} slices from "
            f"'{self._domain_name}' [{self._split}] ..."
        )

        case_dict = self._group_slices_by_case(db_eval)

        self._case_ids        : list              = sorted(case_dict.keys())
        self._images_per_case : dict[str, list]   = {}
        self._label_vols      : dict[str, np.ndarray] = {}
        self._image_vols      : dict[str, np.ndarray] = {}
        self._real_spacings   : dict[str, tuple]  = {}

        for case_id, slices in case_dict.items():
            images_2d = [s[1] for s in slices]
            masks_2d  = [s[2] for s in slices]
            self._images_per_case[case_id] = images_2d
            self._label_vols[case_id]      = np.stack(masks_2d,  axis=0).astype(np.int64)
            self._image_vols[case_id]      = np.stack(images_2d, axis=0).astype(np.float32)
            self._real_spacings[case_id]   = self._get_spacing(self._domain_name, case_id)

        self._log(
            f"Ready: {len(self._case_ids)} case(s), {total_slices} slices pre-loaded."
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(
        self,
        isotropic_spacing: bool = False,
        show_details: bool = False,
        save_predictions: bool = False,
        save_dir: str = None,
    ) -> dict:
        """Run inference and compute volume-level metrics on the cached data.

        All slice data was pre-loaded during ``__init__``.  Each call to
        this method only performs model forward passes and metric computation.

        Parameters
        ----------
        isotropic_spacing : bool
            When *True*, use isotropic 1 mm spacing for ASSD computation
            and NIfTI saving, ignoring the physical spacing from
            ``processing_log.csv``.  When *False* (default), use the
            per-case real physical spacing.
        show_details : bool
            Print per-class metrics to console after evaluation.
        save_predictions : bool
            Save per-case **image**, **pred**, and **label** as ``.nii.gz``.
        save_dir : str, optional
            Required when *save_predictions* is *True*.

        Returns
        -------
        dict
            Flat metrics dict: ``dice_mean``, ``assd_mean``,
            ``dice_class{c}``, ``assd_class{c}``, …
        """
        if save_predictions:
            assert save_dir is not None, (
                "save_dir must be provided when save_predictions=True"
            )

        self.model.eval()

        domain_name  = self._domain_name
        split        = self._split
        spacing_mode = "isotropic 1 mm" if isotropic_spacing else "real physical spacing"
        self._log(
            f"[{split}] Evaluating {len(self._case_ids)} case(s) "
            f"from domain '{domain_name}' [{spacing_mode}] ..."
        )

        all_dice: dict[int, list] = defaultdict(list)
        all_assd: dict[int, list] = defaultdict(list)
        per_case_rows: list[dict] = []

        for case_id in tqdm(self._case_ids, desc=f"Eval [{split}]", leave=False):
            images_2d = self._images_per_case[case_id]
            label_vol = self._label_vols[case_id]
            spacing   = (1.0, 1.0, 1.0) if isotropic_spacing else self._real_spacings[case_id]

            # ── 1. 2.5-D inference ────────────────────────────────────
            pred_vol = self._infer_case(images_2d)                       # (D, H, W)

            # ── 2. Dice per foreground class ──────────────────────────
            dice_scores = compute_dice_per_class(
                pred_vol, label_vol,
                num_classes=self.num_classes,
                include_background=False,
            )
            for c, d in dice_scores.items():
                all_dice[c].append(d)

            # ── 3. ASSD per foreground class ──────────────────────────
            case_assd: dict[int, float] = {}
            for c in range(1, self.num_classes):
                pred_c  = (pred_vol  == c).astype(np.uint8)
                label_c = (label_vol == c).astype(np.uint8)
                assd_val = compute_assd(pred_c, label_c, spacing=spacing)
                all_assd[c].append(assd_val)
                case_assd[c] = assd_val

            # ── 4. Accumulate per-case CSV row ────────────────────────
            row: dict = {"case_id": case_id}
            for c in range(1, self.num_classes):
                row[f"dice_class{c}"] = dice_scores.get(c, 0.0)
                row[f"assd_class{c}"] = case_assd.get(c, float("inf"))

            dice_vals = [dice_scores.get(c, 0.0) for c in range(1, self.num_classes)]
            assd_vals = [case_assd.get(c, float("inf")) for c in range(1, self.num_classes)]
            row["dice_mean"] = float(np.mean(dice_vals)) if dice_vals else 0.0
            finite_assd = [v for v in assd_vals if np.isfinite(v)]
            row["assd_mean"] = float(np.mean(finite_assd)) if finite_assd else float("inf")
            per_case_rows.append(row)

            # ── 5. Optionally save image / pred / label NIfTI ─────────
            if save_predictions:
                self._save_nii(self._image_vols[case_id], case_id, save_dir, "image",  domain_name, spacing)
                self._save_nii(pred_vol.astype(np.int8),  case_id, save_dir, "pred",   domain_name, spacing)
                self._save_nii(label_vol.astype(np.int8), case_id, save_dir, "label",  domain_name, spacing)

        # ── 6. Aggregate + report ──────────────────────────────────────
        metrics = self._aggregate_metrics(all_dice, all_assd)

        if show_details:
            self._show_metrics(metrics, split)

        if save_dir is not None:
            self._save_metrics_csv(
                per_case_rows, metrics, save_dir, split, domain_name, isotropic_spacing
            )

        return metrics

    # ------------------------------------------------------------------
    # Slice grouping (used once during _prepare_cases)
    # ------------------------------------------------------------------

    def _group_slices_by_case(self, db_eval) -> dict:
        """Iterate *db_eval* and group samples by ``case_name``.

        Each element of the returned list is a tuple
        ``(z_index: int, image: np.ndarray (H,W), mask: np.ndarray (H,W))``.
        Both arrays are 2-D; tensors produced by transforms are converted
        to numpy automatically.

        Returns
        -------
        dict
            ``{case_id: [(z_idx, image_2d, mask_2d), ...]}``, sorted by
            ascending z-index within each case.
        """
        case_dict: dict = defaultdict(list)

        for idx in range(len(db_eval)):
            sample    = db_eval[idx]
            case_id   = sample["case_name"]
            slice_idx = int(sample["slice_name"])

            image = sample["image"]
            mask  = sample["mask"]

            # Normalise to 2-D float32 numpy (handles tensor from transforms)
            if isinstance(image, torch.Tensor):
                image = image.squeeze().cpu().numpy().astype(np.float32)
            else:
                image = np.squeeze(np.asarray(image, dtype=np.float32))

            if isinstance(mask, torch.Tensor):
                mask = mask.squeeze().cpu().numpy().astype(np.int64)
            else:
                mask = np.squeeze(np.asarray(mask)).astype(np.int64)

            case_dict[case_id].append((slice_idx, image, mask))

        # Sort by ascending z-index within each case
        return {
            cid: sorted(slices, key=lambda t: t[0])
            for cid, slices in case_dict.items()
        }

    # ------------------------------------------------------------------
    # 2.5-D inference on an ordered slice list
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _infer_case(self, images: list) -> np.ndarray:
        """Run per-slice inference on an ordered list of 2-D image slices.

        The inference mode is determined automatically by ``args.model``:

        * **2.5-D** (``'CSANet' in args.model``): each slice is fed as the
          triplet ``(images[i-1], images[i], images[i+1])`` with
          edge-replication at boundaries.
        * **2-D** (all other models): only the current slice is passed,
          i.e. ``model(curr_t)``.

        When the spatial size ``(H, W)`` of the slices differs from
        ``args.img_size``, each slice is bilinearly resampled to
        ``(img_size, img_size)`` before the forward pass, and the prediction
        is nearest-neighbour mapped back to ``(H, W)`` to match the original
        label volume.

        Parameters
        ----------
        images : list[np.ndarray]
            Ordered float32 arrays, each shape ``(H, W)``.

        Returns
        -------
        np.ndarray, shape ``(D, H, W)``, dtype int64
        """
        N           = len(images)
        H, W        = images[0].shape[:2]
        img_size    = self.args.img_size
        need_resize = (H != img_size or W != img_size)

        pred_slices: list[np.ndarray] = []
        for i in range(N):
            target_size = img_size if need_resize else None
            curr_t = self._slice_to_tensor(images[i], target_size)

            if self.args.is_25d:
                prev_t = self._slice_to_tensor(images[max(0, i - 1)],     target_size)
                next_t = self._slice_to_tensor(images[min(N - 1, i + 1)], target_size)
                logits = self.model(prev_t, curr_t, next_t)  # (1, C, *, *)
            else:
                logits = self.model(curr_t)                  # (1, C, *, *)

            pred = torch.argmax(logits, dim=1)               # (1, *, *)

            # Map prediction back to original (H, W) if resized
            if need_resize:
                pred = F.interpolate(
                    pred.unsqueeze(0).float(),
                    size=(H, W),
                    mode='nearest',
                ).squeeze(0)                                 # (1, H, W)

            pred_slices.append(pred.squeeze(0).cpu().numpy())  # (H, W)

        return np.stack(pred_slices, axis=0).astype(np.int64)  # (D, H, W)

    def _slice_to_tensor(
        self, arr: np.ndarray, target_size: int = None
    ) -> torch.Tensor:
        """Convert a 2-D float32 array to a ``(1, 1, H, W)`` device tensor.

        Parameters
        ----------
        arr : np.ndarray, shape ``(H, W)``
        target_size : int, optional
            When provided, bilinearly resizes to ``(target_size, target_size)``.
        """
        t = (
            torch.from_numpy(np.asarray(arr, dtype=np.float32))
            .unsqueeze(0)   # → (1, H, W)
            .unsqueeze(0)   # → (1, 1, H, W)
            .to(self.device)
        )
        if target_size is not None:
            t = F.interpolate(
                t,
                size=(target_size, target_size),
                mode='bilinear',
                align_corners=False,
            )
        return t

    # ------------------------------------------------------------------
    # NIfTI saving — no reference-image mechanism
    # ------------------------------------------------------------------

    def _save_nii(
        self,
        volume: np.ndarray,
        case_id: str,
        output_dir: str,
        suffix: str,
        domain_name: str,
        spacing: tuple = None,
    ) -> None:
        """Save a 3-D numpy volume as a ``.nii.gz`` NIfTI file.

        The three files written per case (``image``, ``pred``, ``label``) all
        receive the **same** voxel spacing, origin ``(0, 0, 0)``, and identity
        direction matrix, so they overlay correctly in 3D Slicer without any
        reference image or post-hoc alignment.

        Parameters
        ----------
        volume : np.ndarray
            3-D array in ``(D, H, W)`` / ``(z, y, x)`` order.
        spacing : tuple, optional
            ``(z_sp, y_sp, x_sp)`` in mm.  Falls back to isotropic 1 mm
            when *None*.
        """
        os.makedirs(output_dir, exist_ok=True)

        img = sitk.GetImageFromArray(volume)

        # SimpleITK SetSpacing expects (x, y, z)
        z_sp, y_sp, x_sp = spacing if spacing is not None else (1.0, 1.0, 1.0)
        img.SetSpacing((float(x_sp), float(y_sp), float(z_sp)))
        # Origin and direction stay at SimpleITK defaults → (0,0,0) + identity

        out_path = os.path.join(output_dir, f"{domain_name}_{case_id}_{suffix}.nii.gz")
        sitk.WriteImage(img, out_path)
        print(f"Saved {out_path}")

    # ------------------------------------------------------------------
    # Metrics aggregation / display / CSV
    # ------------------------------------------------------------------

    def _aggregate_metrics(
        self,
        all_dice: dict,
        all_assd: dict,
    ) -> dict:
        """Compute mean Dice and ASSD across volumes per class and overall.

        Infinite ASSD values are excluded from the mean.

        Returns
        -------
        dict
            Keys: ``dice_class{c}``, ``assd_class{c}``,
            ``dice_mean``, ``assd_mean``.
        """
        metrics: dict = {}
        dice_means: list[float] = []
        assd_means: list[float] = []

        for c in sorted(all_dice.keys()):
            mean_dice = float(np.mean(all_dice[c]))
            metrics[f"dice_class{c}"] = mean_dice
            dice_means.append(mean_dice)

        for c in sorted(all_assd.keys()):
            finite = [v for v in all_assd[c] if np.isfinite(v)]
            mean_assd = float(np.mean(finite)) if finite else float("inf")
            metrics[f"assd_class{c}"] = mean_assd
            assd_means.append(mean_assd)

        metrics["dice_mean"] = float(np.mean(dice_means)) if dice_means else 0.0
        finite_assd = [v for v in assd_means if np.isfinite(v)]
        metrics["assd_mean"] = (
            float(np.mean(finite_assd)) if finite_assd else float("inf")
        )
        return metrics

    def _show_metrics(self, metrics: dict, split: str) -> None:
        """Print a human-readable per-class summary."""
        print(
            f"[{split}] dice_mean={metrics.get('dice_mean', 0.0):.4f} | "
            f"assd_mean={metrics.get('assd_mean', float('inf')):.4f}"
        )
        lines = []
        for c in range(1, self.num_classes):
            dice = metrics.get(f"dice_class{c}", 0.0)
            assd = metrics.get(f"assd_class{c}", float("inf"))
            lines.append(f"    Class {c:>2d}: Dice={dice:.4f}  ASSD={assd:.4f}")
        if lines:
            print(f"[{split}] Per-class breakdown:\n" + "\n".join(lines))

    def _save_metrics_csv(
        self,
        per_case_rows: list,
        overall_metrics: dict,
        output_dir: str,
        split: str,
        domain_name: str,
        isotropic_spacing: bool,
    ) -> None:
        """Write per-case and overall-mean metrics to a CSV file.

        Saved at ``{output_dir}/metrics_{split}_{domain_name}_{suffix}.csv``
        where *suffix* is ``iso`` when isotropic spacing was used, or empty
        when real physical spacing was used.
        """
        if not per_case_rows:
            return

        os.makedirs(output_dir, exist_ok=True)
        spacing_tag = "iso" if isotropic_spacing else ""
        csv_path    = os.path.join(
            output_dir, f"metrics_{split}_{domain_name}{'_' + spacing_tag if spacing_tag else ''}.csv"
        )
        fieldnames = list(per_case_rows[0].keys())

        def _fmt(v):
            if isinstance(v, float):
                return f"{v:.4f}" if np.isfinite(v) else "inf"
            return v

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for row in per_case_rows:
                writer.writerow({k: _fmt(v) for k, v in row.items()})

            mean_row: dict = {"case_id": "MEAN"}
            for fn in fieldnames[1:]:
                mean_row[fn] = _fmt(overall_metrics.get(fn, 0.0))
            writer.writerow(mean_row)

        print(f"Metrics CSV saved → {csv_path}")

    # ------------------------------------------------------------------
    # Per-case spacing helpers
    # ------------------------------------------------------------------

    def _load_spacing_map(self) -> dict:
        """Load per-case final voxel spacing from ``processing_log.csv``.

        Returns
        -------
        dict
            ``{(dataset_name, case_id): (z_sp, y_sp, x_sp)}``
        """
        log_path    = os.path.join(self.args.data_dir, "processing_log.csv")
        spacing_map: dict = {}

        if not os.path.exists(log_path):
            self._log(
                f"processing_log.csv not found at '{log_path}'. "
                "Falling back to isotropic 1 mm spacing for all cases.",
                level="warning",
            )
            return spacing_map

        with open(log_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                domain  = row["dataset"].strip()
                case_id = row["case_id"].strip()
                parts   = row["final_spacing_xyz"].strip().split("x")
                x_sp, y_sp, z_sp = float(parts[0]), float(parts[1]), float(parts[2])
                spacing_map[(domain, case_id)] = (z_sp, y_sp, x_sp)

        self._log(
            f"Loaded spacing for {len(spacing_map)} cases from processing_log.csv."
        )
        return spacing_map

    def _get_spacing(self, domain_name: str, case_id: str) -> tuple:
        """Return ``(z, y, x)`` spacing in mm for one case."""
        key = (domain_name, case_id)
        if key not in self._spacing_map:
            self._log(
                f"Spacing not found for ({domain_name}, {case_id}). Using isotropic 1 mm.",
                level="debug",
            )
        return self._spacing_map.get(key, (1.0, 1.0, 1.0))

    # ------------------------------------------------------------------
    # Internal logging helper
    # ------------------------------------------------------------------

    def _log(self, msg: str, level: str = "info") -> None:
        """Route a message to the logger (if set) or stdout."""
        if self.logger is not None:
            getattr(self.logger, level)(msg)
        else:
            print(msg)
