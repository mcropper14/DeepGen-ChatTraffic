"""
Evaluation script for ChatTraffic.

Two modes:
  generate  -- load the diffusion model, generate predictions from validation captions,
                compare against ground-truth and report metrics.
  compare   -- compare a directory of pre-generated .npy files against ground-truth.

Channel conventions (2-channel data, third channel is a zero pad):
  ch0  congestion level  0-5    (normalised ÷ 5  during training)
  ch1  speed (km/h)      0-99   (normalised ÷ 150 during training)

Usage examples
--------------
# generate mode (runs the model on the val split)
python scripts/evaluate.py generate \
    --config  configs/latent-diffusion/chattraffic.yaml \
    --ckpt    logs/<run>/checkpoints/last.ckpt \
    --split   datasets/traffic/validation.txt \
    --data_root datasets/traffic \
    --n_samples 200          # omit to evaluate the full split
    --ddim_steps 200

# compare mode (evaluate pre-generated .npy files)
python3 scripts/evaluate.py compare \
    --pred_dir  outputs/chattraffic-samples/samples \
    --split     datasets/traffic/validation.txt \
    --data_root datasets/traffic

python3 scripts/evaluate.py generate \
    --config  configs/latent-diffusion/chattraffic.yaml \
    --ckpt    /mnt/ebs/projects/ChatTraffic/logs/2026-04-15T06-11-20_traffic/checkpoints/epoch=000015.ckpt \
    --split   datasets/traffic/validation.txt \
    --data_root datasets/traffic \
    --n_samples 200          
    --ddim_steps 200


"""



import argparse
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

# ── denormalisation constants ────────────────────────────────────────────────
# ch0 = congestion level (0-5),    normalised ÷ 5    during training
# ch1 = speed (km/h, 0-150),       normalised ÷ 150  during training
# ch2 = passing time (sec, 0-3600),normalised ÷ 3600 during training
#        (zero-padded when dataset only has 2 channels)
CHANNEL_SCALE = np.array([5.0,   150.0,  3600.0])
CHANNEL_CLAMP = np.array([5.0,   150.0,  3600.0])
CHANNEL_NAMES = ["congestion (0-5)", "speed (km/h)", "passing_time (sec)"]

N_ROAD_SEGMENTS = 1260   # number of valid road nodes in the 36×36 grid


# ── road-segment extraction (matches plot_map.py restore_matrix_36) ──────────

def to_road_segments(grid_2d):
    """
    Extract the 1260 valid road-segment values from a (36, 36) channel array.
    Column-major flattening, matching restore_matrix_36 in plot_map.py.
    """
    return grid_2d.T.flatten()[:N_ROAD_SEGMENTS]   # (1260,)


# ── metric helpers ────────────────────────────────────────────────────────────

def mae(pred, gt):
    return float(np.mean(np.abs(pred - gt)))

def rmse(pred, gt):
    return float(np.sqrt(np.mean((pred - gt) ** 2)))

def mape(pred, gt, eps=1e-6):
    """Masked MAPE: skips positions where gt < eps to avoid division by ~zero."""
    mask = gt >= eps
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((pred[mask] - gt[mask]) / gt[mask])) * 100)

def bias(pred, gt):
    """Mean signed error (positive = over-prediction)."""
    return float(np.mean(pred - gt))

def pearson(pred, gt):
    p, g = pred.flatten(), gt.flatten()
    if p.std() < 1e-8 or g.std() < 1e-8:
        return float("nan")
    return float(np.corrcoef(p, g)[0, 1])

def ssim_channel(pred, gt, data_range):
    """Structural Similarity Index for a single 2-D channel (H×W)."""
    try:
        from skimage.metrics import structural_similarity
        return float(structural_similarity(pred, gt, data_range=data_range))
    except ImportError:
        return float("nan")

def accuracy_rounded(pred, gt):
    """Fraction of road nodes where round(pred) == round(gt); for discrete congestion."""
    return float(np.mean(np.round(pred) == np.round(gt)))


# ── paper metrics (Table II format) ──────────────────────────────────────────

def compute_paper_metrics(preds_norm, gts_raw):
    """
    Table II format: MAE + RMSE for congestion, speed, and passing time,
    all evaluated on the 1260 valid road segments.

    preds_norm : (N, 36, 36, 3)  normalised model output [0, 1]
    gts_raw    : (N, 36, 36, C)  raw ground-truth in original units (C = 2 or 3)
    """
    results = {}
    n_gt_ch = gts_raw.shape[-1]

    ch_specs = [
        (0, "congestion",    CHANNEL_SCALE[0], CHANNEL_CLAMP[0]),
        (1, "speed_km/h",    CHANNEL_SCALE[1], CHANNEL_CLAMP[1]),
        (2, "passing_t_sec", CHANNEL_SCALE[2], CHANNEL_CLAMP[2]),
    ]
    for c, label, scale, clamp in ch_specs:
        pred_roads, gt_roads = [], []
        for i in range(len(preds_norm)):
            pred_ch = np.clip(preds_norm[i, :, :, c] * scale, 0.0, clamp)
            if c < n_gt_ch:
                gt_ch = gts_raw[i, :, :, c].astype(np.float32)
            else:
                gt_ch = np.zeros_like(pred_ch)   # ch2 absent → zero GT
            pred_roads.append(to_road_segments(pred_ch))
            gt_roads.append(to_road_segments(gt_ch))

        pred_flat = np.concatenate(pred_roads)
        gt_flat   = np.concatenate(gt_roads)
        results[f"paper/{label}_MAE"]  = mae(pred_flat, gt_flat)
        results[f"paper/{label}_RMSE"] = rmse(pred_flat, gt_flat)
        results[f"paper/{label}_MAPE%"] = mape(pred_flat, gt_flat)

    return results


# ── extended metrics ──────────────────────────────────────────────────────────

def compute_metrics(preds_norm, gts_raw):
    """
    Full metric suite across all 3 channels + combined.

    preds_norm : (N, H, W, 3) normalised model output  [0, 1]
    gts_raw    : (N, H, W, C) raw ground-truth values  (C = 2 or 3)
    """
    results = {}
    n_gt_ch = gts_raw.shape[-1]

    for c, (scale, clamp, name) in enumerate(zip(CHANNEL_SCALE, CHANNEL_CLAMP, CHANNEL_NAMES)):
        pred_raw = np.clip(preds_norm[..., c] * scale, 0.0, clamp)
        if c < n_gt_ch:
            gt_raw = gts_raw[..., c].astype(np.float32)
        else:
            gt_raw = np.zeros_like(pred_raw)
        label = f"ch{c}_{name.split()[0]}"

        results[f"{label}/MAE"]     = mae(pred_raw, gt_raw)
        results[f"{label}/RMSE"]    = rmse(pred_raw, gt_raw)
        results[f"{label}/MAPE(%)"] = mape(pred_raw, gt_raw)
        results[f"{label}/bias"]    = bias(pred_raw, gt_raw)
        results[f"{label}/pearson"] = pearson(pred_raw, gt_raw)
        results[f"{label}/SSIM"]    = float(np.mean([
            ssim_channel(pred_raw[i], gt_raw[i], data_range=clamp)
            for i in range(len(pred_raw))
        ]))
        if c == 0:
            results[f"{label}/rounded_accuracy"] = accuracy_rounded(pred_raw, gt_raw)

        pred_roads = np.array([to_road_segments(pred_raw[i]) for i in range(len(pred_raw))])
        gt_roads   = np.array([to_road_segments(gt_raw[i])   for i in range(len(gt_raw))])
        results[f"{label}/roads_MAE"]     = mae(pred_roads, gt_roads)
        results[f"{label}/roads_RMSE"]    = rmse(pred_roads, gt_roads)
        results[f"{label}/roads_MAPE(%)"] = mape(pred_roads, gt_roads)
        results[f"{label}/roads_pearson"] = pearson(pred_roads, gt_roads)

    # combined normalised [0,1] across all 3 channels
    n_ch = len(CHANNEL_SCALE)
    pred_flat = np.concatenate([
        np.clip(preds_norm[..., c] * CHANNEL_SCALE[c], 0, CHANNEL_CLAMP[c]) / CHANNEL_CLAMP[c]
        for c in range(n_ch)
    ], axis=0).flatten()
    gt_flat = np.concatenate([
        (gts_raw[..., c].astype(np.float32) if c < n_gt_ch
         else np.zeros(gts_raw.shape[:3], dtype=np.float32)).flatten() / CHANNEL_CLAMP[c]
        for c in range(n_ch)
    ])
    results["combined/MAE"]     = mae(pred_flat, gt_flat)
    results["combined/RMSE"]    = rmse(pred_flat, gt_flat)
    results["combined/pearson"] = pearson(pred_flat, gt_flat)

    return results


def print_metrics(paper, extended):
    # ── Table II style: congestion | speed | passing time ─────────────────────
    width = 78
    print("\n" + "=" * width)
    print("  TABLE II METRICS  (1260 road segments)")
    print("=" * width)
    print(f"  {'':30} {'Congestion':^14} {'Speed (km/h)':^14} {'Passing (sec)':^14}")
    print(f"  {'':30} {'MAE':>6} {'RMSE':>6}   {'MAE':>6} {'RMSE':>6}   {'MAE':>6} {'RMSE':>6}")
    print("-" * width)

    def _get(d, key, default=float("nan")):
        return d.get(key, default)

    cong_mae  = _get(paper, "paper/congestion_MAE")
    cong_rmse = _get(paper, "paper/congestion_RMSE")
    spd_mae   = _get(paper, "paper/speed_km/h_MAE")
    spd_rmse  = _get(paper, "paper/speed_km/h_RMSE")
    pt_mae    = _get(paper, "paper/passing_t_sec_MAE")
    pt_rmse   = _get(paper, "paper/passing_t_sec_RMSE")
    print(f"  {'Road segments':<30} {cong_mae:>6.3f} {cong_rmse:>6.3f}   "
          f"{spd_mae:>6.2f} {spd_rmse:>6.2f}   {pt_mae:>6.1f} {pt_rmse:>6.1f}")

    # also show MAPE row
    cong_mape = _get(paper, "paper/congestion_MAPE%")
    spd_mape  = _get(paper, "paper/speed_km/h_MAPE%")
    pt_mape   = _get(paper, "paper/passing_t_sec_MAPE%")
    print(f"  {'  MAPE (%)':<30} {cong_mape:>6.2f} {'':>6}   "
          f"{spd_mape:>6.2f} {'':>6}   {pt_mape:>6.2f}")

    print("\n" + "=" * width)
    print("  EXTENDED METRICS")
    print("=" * width)
    print(f"  {'Metric':<44} {'Value':>10}")
    print("-" * width)
    for k, v in sorted(extended.items()):
        print(f"  {k:<44} {v:>10.4f}")
    print("=" * width + "\n")


# ── data helpers ──────────────────────────────────────────────────────────────

def load_split(split_file, data_root, n_samples=None):
    """Return list of (data_path, text_path) for the given split file."""
    split = os.path.splitext(os.path.basename(split_file))[0]
    with open(split_file) as f:
        stems = [l.strip() for l in f if l.strip()]
    if n_samples:
        stems = stems[:n_samples]
    pairs = []
    for stem in stems:
        data_path = os.path.join(data_root, split, "data", f"{stem}.npy")
        text_path = os.path.join(data_root, split, "text", f"{stem}.txt")
        if os.path.exists(data_path) and os.path.exists(text_path):
            pairs.append((data_path, text_path))
    return pairs


def load_gt(data_path):
    arr = np.load(data_path, allow_pickle=True).astype(np.float32)
    # Returns (H, W, 2) or (H, W, 3) depending on dataset version.
    # compute_paper_metrics / compute_metrics handle both via n_gt_ch checks.
    return arr


def load_caption(text_path):
    with open(text_path) as f:
        return f.read().splitlines()[0]


# ── generate mode ─────────────────────────────────────────────────────────────

def run_generate(args):
    from omegaconf import OmegaConf
    from ldm.util import instantiate_from_config
    from ldm.models.diffusion.ddim import DDIMSampler

    config = OmegaConf.load(args.config)
    print(f"Loading model from {args.ckpt}")
    pl_sd = torch.load(args.ckpt, map_location="cpu")
    model = instantiate_from_config(config.model)
    model.load_state_dict(pl_sd["state_dict"], strict=False)
    model.cuda().eval()

    struc = np.load(os.path.join(args.data_root, "matrix.npy"))

    pairs = load_split(args.split, args.data_root, args.n_samples)
    print(f"Evaluating on {len(pairs)} samples …")

    all_preds, all_gts = [], []

    from ldm.models.diffusion.flow_matching import LatentFlowMatching
    use_flow = isinstance(model, LatentFlowMatching)
    if use_flow:
        print("Detected LatentFlowMatching model — using Euler ODE sampler.")
    else:
        sampler = DDIMSampler(model)

    with torch.no_grad():
        with model.ema_scope():
            for data_path, text_path in tqdm(pairs):
                caption = load_caption(text_path)
                gt      = load_gt(data_path)

                c = model.get_learned_conditioning([caption])
                uc = model.get_learned_conditioning([""]) if args.guidance_scale != 1.0 else None

                if use_flow:
                    samples = model.sample_flow(
                        cond=c,
                        batch_size=1,
                        shape=[3, 36, 36],
                        n_steps=args.ddim_steps,
                        unconditional_guidance_scale=args.guidance_scale,
                        unconditional_conditioning=uc,
                        verbose=False,
                    )
                else:
                    samples, _ = sampler.sample(
                        S=args.ddim_steps,
                        conditioning=c,
                        struc=struc,
                        batch_size=1,
                        shape=[3, 36, 36],
                        verbose=False,
                        unconditional_guidance_scale=args.guidance_scale,
                        unconditional_conditioning=uc,
                        eta=args.ddim_eta,
                    )
                decoded = model.decode_first_stage(samples)
                pred = torch.clamp(decoded, 0.0, 1.0)[0]
                pred = pred.permute(1, 2, 0).cpu().numpy()  # (H, W, 3)

                all_preds.append(pred)
                all_gts.append(gt)

    preds_norm = np.stack(all_preds)   # (N, H, W, 3)
    gts_raw    = np.stack(all_gts)     # (N, H, W, 2)
    paper    = compute_paper_metrics(preds_norm, gts_raw)
    extended = compute_metrics(preds_norm, gts_raw)
    print_metrics(paper, extended)

    if args.save_metrics:
        np.save(args.save_metrics, {**paper, **extended})
        print(f"Metrics saved to {args.save_metrics}")


# ── compare mode ──────────────────────────────────────────────────────────────

def run_compare(args):
    """
    Match .npy files in pred_dir to GT by filename stem (alphabetical order as
    fallback if stems don't align).
    """
    pairs = load_split(args.split, args.data_root, args.n_samples)

    pred_files = sorted([
        f for f in os.listdir(args.pred_dir) if f.endswith(".npy")
    ])

    n = min(len(pairs), len(pred_files))
    if n == 0:
        sys.exit("No matching samples found.")
    print(f"Comparing {n} samples …")

    all_preds, all_gts = [], []
    for (data_path, _), pred_fname in zip(pairs[:n], pred_files[:n]):
        pred = np.load(os.path.join(args.pred_dir, pred_fname)).astype(np.float32)
        if pred.ndim == 3 and pred.shape[0] in (2, 3):  # C H W → H W C
            pred = pred.transpose(1, 2, 0)
        all_preds.append(pred)
        all_gts.append(load_gt(data_path))

    preds_norm = np.stack(all_preds)
    gts_raw    = np.stack(all_gts)
    paper    = compute_paper_metrics(preds_norm, gts_raw)
    extended = compute_metrics(preds_norm, gts_raw)
    print_metrics(paper, extended)

    if args.save_metrics:
        np.save(args.save_metrics, {**paper, **extended})
        print(f"Metrics saved to {args.save_metrics}")


# ── autoencoder mode ──────────────────────────────────────────────────────────

def run_autoencoder(args):
    """
    Evaluate the autoencoder in isolation: encode each GT sample then decode it,
    and compare the reconstruction against the original.  This measures the
    reconstruction ceiling — the best any downstream diffusion model can achieve.
    """
    from omegaconf import OmegaConf
    from ldm.util import instantiate_from_config

    config = OmegaConf.load(args.config)
    print(f"Loading autoencoder from {args.ckpt}")
    pl_sd = torch.load(args.ckpt, map_location="cpu")

    # The autoencoder checkpoint may be wrapped in a LightningModule state_dict
    sd = pl_sd.get("state_dict", pl_sd)
    ae = instantiate_from_config(config.model)
    ae.load_state_dict(sd, strict=False)
    ae.cuda().eval()

    pairs = load_split(args.split, args.data_root, args.n_samples)
    print(f"Evaluating autoencoder reconstruction on {len(pairs)} samples …")

    all_preds, all_gts = [], []

    with torch.no_grad():
        for data_path, _ in tqdm(pairs):
            gt = load_gt(data_path)              # (H, W, 2)  raw values

            # zero-pad to 3 channels, normalise, convert to tensor
            h, w = gt.shape[:2]
            x = gt.astype(np.float32)
            x[:, :, 0] /= CHANNEL_SCALE[0]      # congestion ÷ 5
            x[:, :, 1] /= CHANNEL_SCALE[1]      # speed ÷ 150
            x = np.concatenate([x, np.zeros((h, w, 1), dtype=np.float32)], axis=2)
            t = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).cuda()  # (1,3,H,W)

            # encode → sample posterior → decode
            posterior = ae.encode(t)
            z = posterior.sample()
            recon = ae.decode(z)
            recon = torch.clamp(recon, 0.0, 1.0)[0].permute(1, 2, 0).cpu().numpy()

            all_preds.append(recon)
            all_gts.append(gt)

    preds_norm = np.stack(all_preds)
    gts_raw    = np.stack(all_gts)
    paper    = compute_paper_metrics(preds_norm, gts_raw)
    extended = compute_metrics(preds_norm, gts_raw)

    print("\n  [Autoencoder reconstruction — upper bound for diffusion quality]")
    print_metrics(paper, extended)

    if args.save_metrics:
        np.save(args.save_metrics, {**paper, **extended})
        print(f"Metrics saved to {args.save_metrics}")


# ── sequence mode ─────────────────────────────────────────────────────────────

def run_sequence(args):
    """
    Temporal consistency evaluation for a generated sequence directory.

    Uses the same MAE/RMSE/MAPE/Pearson/SSIM metric suite as the generate/compare
    modes, applied to consecutive frame pairs (frame[t] as GT, frame[t+1] as pred).
    This measures how much the sequence changes frame-to-frame in physical units.

    Also reports per-frame statistics and cong–speed correlation.

    Expects .npy files shaped (H, W, C) or (C, H, W) in normalised [0,1].
    Files are sorted alphabetically, matching generate_traffic_sequence.py output.
    """
    npy_files = sorted([
        os.path.join(args.seq_dir, f)
        for f in os.listdir(args.seq_dir) if f.endswith(".npy")
    ])
    if len(npy_files) < 2:
        sys.exit(f"Need at least 2 .npy files in {args.seq_dir}, found {len(npy_files)}.")

    frames = []
    for p in npy_files:
        arr = np.load(p).astype(np.float32)
        if arr.ndim == 3 and arr.shape[0] in (2, 3):   # C H W → H W C
            arr = arr.transpose(1, 2, 0)
        frames.append(arr)

    frames = np.stack(frames)   # (T, H, W, C)
    T, H, W, C = frames.shape

    print(f"\nSequence: {T} frames  |  shape per frame: ({H}, {W}, {C})")
    print(f"Source: {args.seq_dir}\n")

    # ── plausibility ─────────────────────────────────────────────────────────
    in_range = float(np.mean((frames >= 0.0) & (frames <= 1.0)))

    # ── road-segment series for per-frame stats and cong–speed corr ──────────
    road_frames = []
    for t in range(T):
        cong  = frames[t, :, :, 0].T.flatten()[:N_ROAD_SEGMENTS]
        speed = frames[t, :, :, 1].T.flatten()[:N_ROAD_SEGMENTS]
        road_frames.append(np.stack([cong, speed], axis=1))
    road_series = np.stack(road_frames)   # (T, 1260, 2)

    temporal_std_cong  = float(road_series[:, :, 0].std(axis=0).mean())
    temporal_std_speed = float(road_series[:, :, 1].std(axis=0).mean())

    mean_cong  = road_series[:, :, 0].mean(axis=1)
    mean_speed = road_series[:, :, 1].mean(axis=1)
    if mean_cong.std() > 1e-6 and mean_speed.std() > 1e-6:
        cong_speed_corr = float(np.corrcoef(mean_cong, mean_speed)[0, 1])
    else:
        cong_speed_corr = float("nan")

    # ── per-frame summary ─────────────────────────────────────────────────────
    width = 72
    print("=" * width)
    print("  PER-FRAME SUMMARY  (road segments, normalised [0,1])")
    print("=" * width)
    print(f"  {'Frame':<8} {'cong mean':>10} {'cong std':>10} {'speed mean':>11} {'speed std':>10}")
    print("-" * width)
    for t in range(T):
        c0 = road_series[t, :, 0]
        c1 = road_series[t, :, 1]
        print(f"  {t:<8} {c0.mean():>10.4f} {c0.std():>10.4f} {c1.mean():>11.4f} {c1.std():>10.4f}")

    # ── frame-to-frame metrics using the standard metric suite ───────────────
    # Treat frame[t] as "ground truth" and frame[t+1] as "prediction".
    # compute_paper_metrics / compute_metrics expect:
    #   preds_norm : (N, H, W, 3) normalised [0,1]
    #   gts_raw    : (N, H, W, 2) raw values in original units
    preds_norm = frames[1:]                          # frames 1..T-1
    gts_norm   = frames[:-1]                         # frames 0..T-2

    # Denormalise GT to raw units for the metric functions (all 3 channels)
    n_ch = min(C, len(CHANNEL_SCALE))
    gts_raw = np.zeros((*gts_norm.shape[:3], n_ch), dtype=np.float32)
    for c in range(n_ch):
        gts_raw[..., c] = gts_norm[..., c] * CHANNEL_SCALE[c]

    print(f"\n  Frame-to-frame metrics  ({T-1} consecutive pairs)\n")
    paper    = compute_paper_metrics(preds_norm, gts_raw)
    extended = compute_metrics(preds_norm, gts_raw)
    print_metrics(paper, extended)

    # ── sequence-level summary ────────────────────────────────────────────────
    print("=" * width)
    print("  SEQUENCE-LEVEL SUMMARY")
    print("=" * width)
    rows = [
        ("Frames",                        T),
        ("Values in [0,1] (%)",           in_range * 100),
        ("Temporal std — cong  (roads)",  temporal_std_cong),
        ("Temporal std — speed (roads)",  temporal_std_speed),
        ("Cong–speed corr across frames", cong_speed_corr),
    ]
    for label, val in rows:
        print(f"  {label:<44} {val:>10.4f}")
    print("=" * width)
    print("  Note: frame-to-frame MAE/RMSE measure smoothness (lower = smoother).")
    print("  Cong–speed corr ≈ -1 is physically realistic.\n")

    if args.save_metrics:
        all_metrics = {
            **paper, **extended,
            "n_frames": T,
            "values_in_range_pct": in_range * 100,
            "temporal_std_cong": temporal_std_cong,
            "temporal_std_speed": temporal_std_speed,
            "cong_speed_corr": cong_speed_corr,
        }
        np.save(args.save_metrics, all_metrics)
        print(f"Metrics saved to {args.save_metrics}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ChatTraffic evaluation")
    sub = parser.add_subparsers(dest="mode", required=True)

    # shared args
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--split",      default="datasets/traffic/validation.txt")
    shared.add_argument("--data_root",  default="datasets/traffic")
    shared.add_argument("--n_samples",  type=int, default=None,
                        help="limit number of samples (default: full split)")
    shared.add_argument("--save_metrics", default=None,
                        help="optional path to save metrics dict as .npy")

    # generate sub-command
    gen = sub.add_parser("generate", parents=[shared])
    gen.add_argument("--config",         required=True)
    gen.add_argument("--ckpt",           required=True)
    gen.add_argument("--ddim_steps",     type=int,   default=200)
    gen.add_argument("--ddim_eta",       type=float, default=0.0)
    gen.add_argument("--guidance_scale", type=float, default=1.0)

    # compare sub-command
    cmp = sub.add_parser("compare", parents=[shared])
    cmp.add_argument("--pred_dir", required=True,
                     help="directory of pre-generated .npy files")

    # autoencoder sub-command
    ae = sub.add_parser("autoencoder", parents=[shared])
    ae.add_argument("--config", required=True,
                    help="autoencoder config yaml")
    ae.add_argument("--ckpt",   required=True,
                    help="autoencoder checkpoint")

    # sequence sub-command
    seq = sub.add_parser("sequence",
                         help="temporal consistency eval for a generated sequence")
    seq.add_argument("--seq_dir",     required=True,
                     help="directory of .npy frames from generate_traffic_sequence.py")
    seq.add_argument("--save_metrics", default=None,
                     help="optional path to save metrics dict as .npy")

    args = parser.parse_args()

    if args.mode == "generate":
        run_generate(args)
    elif args.mode == "autoencoder":
        run_autoencoder(args)
    elif args.mode == "sequence":
        run_sequence(args)
    else:
        run_compare(args)


if __name__ == "__main__":
    main()
