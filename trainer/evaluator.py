"""
Evaluator for 2.5D medical image segmentation (CSANet / REBOUND).

For each volume in a given split the *complete* 3-D volume (including
background-only slices that were discarded during preprocessing) is loaded
from the ``volumes/`` directory via :class:`CSANet_VolumeDataset`.  Each
slice is then processed as a (prev, current, next) 2.5D triplet and the
per-slice predictions are stacked into a 3-D prediction volume.  Dice and
ASSD are computed per foreground class using the **per-case physical voxel
spacing** read from ``processing_log.csv``.

Intended usage
--------------
1. Per-epoch validation inside a Trainer (``db_val`` built once in ``train()``)::

       evaluator = Evaluator(args, metadata, model, device, logger=self.logger)
       metrics = evaluator.evaluate(db_eval=db_val)

2. Standalone test evaluation (``test.py`` entry point)::

       db_test = CSANet_VolumeDataset(base_dir, domain_name, split='test', metadata)
       evaluator = Evaluator(args, metadata, model, device)
       metrics = evaluator.evaluate(
           db_eval=db_test,
           save_predictions=True,
           save_dir='results/preds',
       )
"""

import csv
import logging
import os
from collections import defaultdict

import numpy as np
import SimpleITK as sitk
import nibabel as nib
import torch
from tqdm import tqdm

from utils.metrics import compute_assd, compute_dice_per_class


class Evaluator:
    """Volume-level evaluator for a 2.5D segmentation model.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.  Must contain at least ``data_dir``
        and ``dataset``.
    metadata : dict
        Dataset metadata loaded from ``metadata.json``.  Must contain
        ``num_classes`` and optionally ``splits``.
    model : torch.nn.Module
        The segmentation model (e.g. CSANet).  Already moved to *device*.
    device : torch.device
        Torch device to run inference on.
    logger : logging.Logger, optional
        Logger instance; a default logger is created when omitted.
    """

    def __init__(self, args, metadata, model, device, logger=None):
        self.args = args
        self.metadata = metadata
        self.model = model
        self.device = device
        self.num_classes = metadata["num_classes"]
        self.logger = logger or logging.getLogger(__name__)

        # Pre-load per-case spacing from processing_log.csv
        self._spacing_map = self._load_spacing_map()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(
        self,
        db_eval,
        log_details: bool = False,
        save_predictions: bool = False,
        save_dir: str = None,
    ) -> dict:
        """Run volume-level evaluation on a pre-built dataset object.

        The dataset is expected to be a :class:`CSANet_VolumeDataset` instance
        (or any object that exposes ``__len__``, ``__getitem__`` returning
        ``{'image', 'mask', 'case_name'}`` dicts, plus ``domain_name`` and
        ``split`` attributes).

        Physical voxel spacing used for ASSD is read per-case from
        ``processing_log.csv``; isotropic 1 mm is assumed as a fallback.

        When *save_dir* is given:

        * Prediction and label volumes are saved as ``pred_{case_id}.nii.gz``
          / ``label_{case_id}.nii.gz`` (only when *save_predictions* is
          *True*).
        * Per-case and overall average metrics are **always** written to
          ``{save_dir}/metrics_{split}.csv``.

        Parameters
        ----------
        db_eval : CSANet_VolumeDataset
            Dataset instance built by the caller (trainer or test script).
        save_predictions : bool
            Whether to save prediction/label volumes as ``.nii.gz``.
        save_dir : str, optional
            Directory for saving predictions and/or metrics CSV.

        Returns
        -------
        dict
            Flat metrics dictionary with keys ``'dice_mean'``,
            ``'assd_mean'``, ``'dice_class{c}'``, ``'assd_class{c}'``, …
            Ready to pass directly to a trainer's ``_log_metrics``.
        """
        if save_predictions:
            assert save_dir is not None, ("save_dir must be provided when save_predictions=True")
            
        self.model.eval()

        domain_name = db_eval.domain_name
        split = db_eval.split

        self.logger.info(
            f"[{split}] Evaluating {len(db_eval)} volumes "
            f"from domain '{domain_name}' ..."
        )

        all_dice: dict[int, list] = defaultdict(list)  # class → [dice per vol]
        all_assd: dict[int, list] = defaultdict(list)  # class → [assd per vol]
        per_case_rows: list[dict] = []

        for idx in tqdm(range(len(db_eval)), desc=f"Eval [{split}]", leave=False):
            sample = db_eval[idx]
            # CSANet_VolumeDataset yields raw numpy arrays (no transform applied)
            image_vol = sample["image"].astype(np.float32)   # (D, H, W)
            label_vol = sample["mask"].astype(np.int64)       # (D, H, W)
            case_id   = sample["case_name"]                   # e.g. "0001"

            # --- 2.5D inference → 3-D prediction volume ---
            pred_vol = self._infer_volume(image_vol)          # (D, H, W)

            # --- Per-case spacing for ASSD (z, y, x) in mm ---
            spacing = self._get_spacing(domain_name, case_id)

            # --- Dice per foreground class ---
            dice_scores = compute_dice_per_class(
                pred_vol, label_vol,
                num_classes=self.num_classes,
                include_background=False,
            )
            for c, d in dice_scores.items():
                all_dice[c].append(d)

            # --- ASSD per foreground class ---
            case_assd: dict[int, float] = {}
            for c in range(1, self.num_classes):
                pred_c  = (pred_vol  == c).astype(np.uint8)
                label_c = (label_vol == c).astype(np.uint8)
                assd_val = compute_assd(pred_c, label_c, spacing=spacing)
                all_assd[c].append(assd_val)
                case_assd[c] = assd_val

            # --- Accumulate per-case row for CSV ---
            row: dict = {"case_id": case_id}
            for c in range(1, self.num_classes):
                row[f"dice_class{c}"] = dice_scores.get(c, 0.0)
                row[f"assd_class{c}"] = case_assd.get(c, float("inf"))
            per_case_rows.append(row)

            # --- Optionally save prediction / label volumes ---
            if save_predictions:
                # Use the original image as geometry reference so that
                # origin, direction and spacing all match the source volume,
                # enabling correct overlay in 3D Slicer.
                ref_img_path = os.path.join(
                    db_eval.data_dir, f"img_{case_id}.nii.gz"
                )
                self._save_nii(
                    pred_vol.astype(np.int8), case_id, save_dir,
                    suffix="pred", ref_img_path=ref_img_path,
                )
                self._save_nii(
                    label_vol.astype(np.int8), case_id, save_dir,
                    suffix="label", ref_img_path=ref_img_path,
                )

        # --- Aggregate metrics ---
        metrics = self._aggregate_metrics(all_dice, all_assd)

        if log_details:
            self._log_metrics(metrics, split)

        # --- Always write CSV when save_dir is provided ---
        if save_dir is not None:
            self._save_metrics_csv(per_case_rows, metrics, save_dir, split)

        return metrics

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_spacing_map(self) -> dict:
        """Load per-case final voxel spacing from ``processing_log.csv``.

        The CSV column ``final_spacing_xyz`` stores spacing as
        ``"{x}x{y}x{z}"`` (e.g. ``"0.9023x0.9023x3.0000"``).
        SimpleITK returns arrays in ``(z, y, x)`` order, so this method
        returns spacing as ``(z, y, x)`` tuples.

        Returns
        -------
        dict
            ``{(dataset_name, case_id): (z_sp, y_sp, x_sp)}``
        """
        log_path = os.path.join(
            self.args.data_dir, self.args.dataset,
            "processed", "processing_log.csv",
        )
        spacing_map: dict = {}

        if not os.path.exists(log_path):
            self.logger.warning(
                f"processing_log.csv not found at '{log_path}'. "
                "Falling back to isotropic 1 mm spacing for all cases."
            )
            return spacing_map

        with open(log_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                domain  = row["dataset"].strip()
                case_id = row["case_id"].strip()
                # "0.9023x0.9023x3.0000"  →  x, y, z
                parts = row["final_spacing_xyz"].strip().split("x")
                x_sp, y_sp, z_sp = float(parts[0]), float(parts[1]), float(parts[2])
                # Store in (z, y, x) to match numpy array axis order
                spacing_map[(domain, case_id)] = (z_sp, y_sp, x_sp)

        self.logger.info(
            f"Loaded spacing for {len(spacing_map)} cases from processing_log.csv."
        )
        return spacing_map

    def _get_spacing(self, domain_name: str, case_id: str) -> tuple:
        """Return ``(z, y, x)`` spacing in mm for one case."""
        key = (domain_name, case_id)
        if key not in self._spacing_map:
            self.logger.debug(
                f"Spacing not found for ({domain_name}, {case_id}). "
                "Using isotropic 1 mm."
            )
        return self._spacing_map.get(key, (1.0, 1.0, 1.0))

    @torch.no_grad()
    def _infer_volume(self, image_vol: np.ndarray) -> np.ndarray:
        """Run 2.5D inference on a complete volume.

        For each slice *i*, the triplet ``(image[i-1], image[i], image[i+1])``
        is fed into the model.  Boundary slices use edge-replication: the
        first/last slice stands in for the missing neighbour.

        Parameters
        ----------
        image_vol : np.ndarray, shape ``(D, H, W)``
            Full 3-D image volume, float32.

        Returns
        -------
        np.ndarray, shape ``(D, H, W)``, dtype int64
            Argmax class predictions assembled slice by slice.
        """
        D = image_vol.shape[0]
        pred_slices: list[np.ndarray] = []

        for i in range(D):
            prev_i = max(0, i - 1)
            next_i = min(D - 1, i + 1)

            def _to_tensor(arr: np.ndarray) -> torch.Tensor:
                """(H, W) → (1, 1, H, W) on device."""
                return (
                    torch.from_numpy(arr.astype(np.float32))
                    .unsqueeze(0)   # → (1, H, W)
                    .unsqueeze(0)   # → (1, 1, H, W)
                    .to(self.device)
                )

            prev_t = _to_tensor(image_vol[prev_i])
            curr_t = _to_tensor(image_vol[i])
            next_t = _to_tensor(image_vol[next_i])

            # model(prev, curr, next) → logits (1, C, H, W)
            logits = self.model(prev_t, curr_t, next_t)
            pred   = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()  # (H, W)
            pred_slices.append(pred)

        return np.stack(pred_slices, axis=0).astype(np.int64)  # (D, H, W)

    def _save_nii(
        self,
        volume: np.ndarray,
        case_id: str,
        output_dir: str,
        suffix: str,
        ref_img_path: str = None,
    ) -> None:
        """Save a 3-D numpy volume as a ``.nii.gz`` file via SimpleITK.

        When *ref_img_path* is supplied the saved file inherits the
        **complete geometry** (spacing, origin, direction cosines) of the
        reference image via ``CopyInformation``.  This is required for
        segmentation overlays in 3D Slicer to align correctly with the
        original scalar volume.

        Parameters
        ----------
        volume : np.ndarray
            3-D array in ``(D, H, W)`` / ``(z, y, x)`` order.
        case_id : str
            Case identifier used in the output filename.
        output_dir : str
            Output directory (created if absent).
        suffix : str
            Filename prefix, e.g. ``'pred'`` or ``'label'``.
        ref_img_path : str, optional
            Path to the original ``.nii.gz`` image whose geometry should be
            copied.  When omitted only spacing metadata is unavailable and
            the default SimpleITK geometry (origin=0, identity direction)
            is used.
        """
        os.makedirs(output_dir, exist_ok=True)

        img = sitk.GetImageFromArray(volume)

        if ref_img_path and os.path.exists(ref_img_path):
            # Copy spacing + origin + direction from the source image so
            # that the segmentation file is geometrically aligned with it.
            ref = sitk.ReadImage(ref_img_path)
            img.CopyInformation(ref)
        else:
            self.logger.warning(
                f"Reference image not found: '{ref_img_path}'. "
                "Saved NIfTI will lack correct origin/direction and may "
                "not overlay properly in 3D Slicer."
            )

        out_path = os.path.join(output_dir, f"{suffix}_{case_id}.nii.gz")
        sitk.WriteImage(img, out_path)
        self.logger.debug(f"Saved {out_path}")

    def _aggregate_metrics(
        self,
        all_dice: dict,
        all_assd: dict,
    ) -> dict:
        """Compute mean Dice and ASSD across volumes per class and overall.

        Infinite ASSD values (empty prediction or ground-truth mask) are
        excluded from the mean; if every volume yields ``inf`` for a class,
        that class reports ``inf``.

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

    def _log_metrics(self, metrics: dict, split: str) -> None:
        """Print a human-readable summary via ``self.logger``."""
        self.logger.info(
            f"[{split}] dice_mean={metrics.get('dice_mean', 0.0):.4f} | "
            f"assd_mean={metrics.get('assd_mean', float('inf')):.4f}"
        )
        lines = []
        for c in range(1, self.num_classes):
            dice = metrics.get(f"dice_class{c}", 0.0)
            assd = metrics.get(f"assd_class{c}", float("inf"))
            lines.append(f"    Class {c:>2d}: Dice={dice:.4f}  ASSD={assd:.4f}")
        if lines:
            self.logger.info(
                f"[{split}] Per-class breakdown:\n" + "\n".join(lines)
            )

    def _save_metrics_csv(
        self,
        per_case_rows: list,
        overall_metrics: dict,
        output_dir: str,
        split: str,
    ) -> None:
        """Write per-case and overall-mean metrics to a CSV file.

        The CSV is saved at ``{output_dir}/metrics_{split}.csv``.  The last
        row is a ``MEAN`` summary row aggregated over all cases.

        Parameters
        ----------
        per_case_rows : list[dict]
            One dict per evaluated volume.  All dicts share the same keys.
        overall_metrics : dict
            Aggregated metrics returned by :meth:`_aggregate_metrics`.
        output_dir : str
            Directory to write the CSV into (created if absent).
        split : str
            Split label used in the filename.
        """
        if not per_case_rows:
            return

        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, f"metrics_{split}.csv")
        fieldnames = list(per_case_rows[0].keys())  # ['case_id', 'dice_class1', ...]

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            # --- Per-case rows ---
            for row in per_case_rows:
                formatted = {}
                for k, v in row.items():
                    if isinstance(v, float):
                        formatted[k] = f"{v:.4f}" if np.isfinite(v) else "inf"
                    else:
                        formatted[k] = v
                writer.writerow(formatted)

            # --- Overall MEAN row ---
            mean_row: dict = {"case_id": "MEAN"}
            for fn in fieldnames[1:]:
                val = overall_metrics.get(fn, 0.0)
                mean_row[fn] = f"{val:.4f}" if np.isfinite(val) else "inf"
            writer.writerow(mean_row)

        self.logger.info(f"Metrics CSV saved → {csv_path}")
