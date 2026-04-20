"""
Temporal traffic sequence generator.

Pipeline:
  1. Ollama (local LLM) generates N time-stamped traffic captions.
  2. ChatTraffic runs the diffusion/flow sampler on each caption → keyframe latent.
  3. Keyframe latents are linearly interpolated for smooth in-between frames.
  4. All frames are decoded, saved as .npy + Folium HTML maps + an animated GIF.

Usage
-----
python scripts/generate_traffic_sequence.py \
    --config  configs/latent-diffusion/chattraffic.yaml \
    --ckpt    logs/<run>/checkpoints/last.ckpt \
    --start_time "Monday 7:00am" \
    --duration_hours 12 \
    --n_keyframes 6 \
    --interp_frames 4 \
    --outdir  outputs/traffic_sequence

# Flow-matching checkpoint works identically:
python scripts/generate_traffic_sequence.py \
    --config  configs/latent-diffusion/chattraffic_flow.yaml \
    --ckpt    logs/<flow-run>/checkpoints/last.ckpt \
    --ddim_steps 50 ...

# Skip Ollama — supply your own captions file (one caption per line):
python scripts/generate_traffic_sequence.py ... --captions_file my_captions.txt
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import requests
import torch
from PIL import Image
from tqdm import tqdm


# ── colour helpers (copied from plot_map.py) ──────────────────────────────────

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import matplotlib.animation as animation

CHANNEL_SCALE = np.array([5.0, 150.0])
CHANNEL_CLAMP = np.array([5.0,  99.0])
CHANNEL_NAMES = ["congestion", "speed_kmh"]


# ── Ollama caption generation ─────────────────────────────────────────────────

CAPTION_SYSTEM = (
    "You generate realistic, concise traffic condition descriptions for "
    "Beijing's road network. Each description is exactly one sentence. "
    "Reference specific Beijing landmarks when relevant: 2nd/3rd/4th/5th ring roads, "
    "Chang'an Avenue, Guomao, Sanlitun, Zhongguancun, Xidan, Chaoyang, Wangfujing, "
    "Dongzhimen, Xizhimen, Liuliqiao, Siyuan Bridge, Airport Expressway, G4/G6 motorways. "
    "Reflect real Beijing traffic patterns: weekday rush hours are 7-9am and 5-8pm. "
    "Use precise traffic vocabulary: road closure, serious/general traffic accident, "
    "construction zone, lane reduction, waterlogging, vehicle breakdown, police checkpoint, "
    "tidal lane reversal, congestion spillback, incident clearance."
)

CAPTION_SYSTEM_EDGE = (
    "You generate UNUSUAL and EXTREME traffic condition descriptions for Beijing's road network. "
    "Each description is exactly one sentence and must depict a rare, high-impact, or compounding event. "
    "Reference specific Beijing landmarks: 2nd/3rd/4th/5th ring roads, Chang'an Avenue, Guomao, "
    "Sanlitun, Zhongguancun, Xidan, Chaoyang, Wangfujing, Airport Expressway, G4/G6 motorways. "
    "Edge-case categories to draw from:\n"
    "  WEATHER: heavy snowfall/ice, flash flooding/waterlogging, dense fog (visibility <50m), "
    "sandstorm reducing visibility, black ice on elevated sections.\n"
    "  INCIDENTS: multi-vehicle pile-up blocking all lanes, hazardous material spill, "
    "bridge/tunnel emergency closure, vehicle fire on ring road, broken-down truck on ramp.\n"
    "  EVENTS: National Day parade route closure, marathon race diversion, "
    "large concert dispersal at Workers Stadium/National Stadium, state visit motorcade escort.\n"
    "  INFRASTRUCTURE: traffic signal failure at major intersection, motorway contraflow, "
    "emergency road works, burst water main causing sinkhole.\n"
    "  CASCADING: accident on 3rd ring causing spillback onto 4th ring, "
    "closure on one route pushing all traffic onto parallel arterial.\n"
    "Use precise vocabulary: road closure, serious traffic accident, lane reduction, "
    "waterlogging, vehicle breakdown, congestion spillback, incident clearance, diversion route."
)

CAPTION_USER_TEMPLATE = """\
Generate exactly {n} traffic condition descriptions for Beijing, evenly spaced \
over {hours} hours starting at {start}.
{context_line}

Rules:
- One sentence per line, no numbering, no bullet points, no extra text.
- Each sentence must naturally follow from the previous (coherent temporal progression).
- Show how the situation evolves: onset → peak → partial/full clearance.
- Vary congestion level realistically across the time window.
- Output exactly {n} lines.\
"""

CAPTION_USER_TEMPLATE_EDGE = """\
Generate exactly {n} edge-case traffic condition descriptions for Beijing, \
evenly spaced over {hours} hours starting at {start}.
{context_line}

The sequence must depict an UNUSUAL, HIGH-IMPACT traffic scenario. Choose one or more:
- Extreme weather event (snow, flood, fog, sandstorm)
- Major multi-vehicle incident or infrastructure failure
- Large public event causing network-wide diversion
- Cascading failures where one closure triggers gridlock elsewhere

Rules:
- One sentence per line, no numbering, no bullet points, no extra text.
- Show the scenario evolving: initial trigger → escalation → peak disruption → slow recovery.
- Reference specific roads/landmarks in Beijing for each step.
- Make conditions more severe than normal rush-hour traffic.
- Output exactly {n} lines.\
"""


def generate_captions_ollama(n, start_time, duration_hours, context,
                              ollama_model, ollama_url, timeout=600,
                              edge_cases=False):
    system  = CAPTION_SYSTEM_EDGE if edge_cases else CAPTION_SYSTEM
    tmpl    = CAPTION_USER_TEMPLATE_EDGE if edge_cases else CAPTION_USER_TEMPLATE
    temp    = 0.9 if edge_cases else 0.7

    context_line = f"Additional context: {context}" if context else ""
    user_prompt = tmpl.format(
        n=n, hours=duration_hours, start=start_time, context_line=context_line
    )

    payload = {
        "model": ollama_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user_prompt},
        ],
        "stream": True,
        "options": {"temperature": temp, "num_predict": 768},
    }

    mode = "edge-case" if edge_cases else "standard"
    print(f"Calling Ollama ({ollama_model}) at {ollama_url} [{mode} mode] …")
    print("Generating captions: ", end="", flush=True)
    try:
        resp = requests.post(
            f"{ollama_url}/api/chat", json=payload,
            timeout=timeout, stream=True,
        )
        resp.raise_for_status()
    except requests.exceptions.ConnectionError:
        sys.exit(
            f"\nCannot reach Ollama at {ollama_url}.\n"
            "Make sure Ollama is running (`ollama serve`) or pass --captions_file."
        )
    except requests.exceptions.ReadTimeout:
        sys.exit(
            f"\nOllama timed out after {timeout}s. "
            "Try a smaller model or pass --captions_file with pre-written captions."
        )

    import json as _json
    raw_chunks = []
    for line in resp.iter_lines():
        if not line:
            continue
        try:
            chunk = _json.loads(line)
        except ValueError:
            continue
        token = chunk.get("message", {}).get("content", "")
        raw_chunks.append(token)
        # Print a dot for each newline token so the user sees progress
        if "\n" in token:
            print(".", end="", flush=True)
        if chunk.get("done"):
            break
    print()  # newline after dots

    raw = "".join(raw_chunks)
    captions = [l.strip() for l in raw.splitlines() if l.strip()]

    # Trim or pad to exactly n
    captions = captions[:n]
    while len(captions) < n:
        captions.append(captions[-1] if captions else "Normal traffic conditions in Beijing.")

    print(f"Generated {len(captions)} captions.")
    return captions


def load_captions_file(path):
    with open(path) as f:
        return [l.strip() for l in f if l.strip()]


# ── model loading ─────────────────────────────────────────────────────────────

def load_model(config_path, ckpt_path):
    from omegaconf import OmegaConf
    from ldm.util import instantiate_from_config

    config = OmegaConf.load(config_path)
    print(f"Loading model from {ckpt_path} …")
    pl_sd = torch.load(ckpt_path, map_location="cpu")
    model = instantiate_from_config(config.model)
    model.load_state_dict(pl_sd["state_dict"], strict=False)
    model.cuda().eval()
    return model


# ── latent generation ─────────────────────────────────────────────────────────

def caption_to_latent(caption, model, sampler, args, is_flow):
    """Run the diffusion/flow sampler for one caption; return raw latent (1,3,36,36)."""
    c  = model.get_learned_conditioning([caption])
    uc = model.get_learned_conditioning([""]) if args.guidance_scale != 1.0 else None

    if is_flow:
        latent = model.sample_flow(
            cond=c,
            batch_size=1,
            shape=[3, 36, 36],
            n_steps=args.ddim_steps,
            unconditional_guidance_scale=args.guidance_scale,
            unconditional_conditioning=uc,
            verbose=False,
        )
    else:
        latent, _ = sampler.sample(
            S=args.ddim_steps,
            conditioning=c,
            batch_size=1,
            shape=[3, 36, 36],
            verbose=False,
            unconditional_guidance_scale=args.guidance_scale,
            unconditional_conditioning=uc,
            eta=args.ddim_eta,
        )
    return latent   # (1, 3, 36, 36)


# ── interpolation ─────────────────────────────────────────────────────────────

def interpolate_latents(latents, interp_frames):
    """
    Given K keyframe latents, return K + (K-1)*interp_frames total latents
    by linearly interpolating in latent space between each consecutive pair.
    """
    frames = []
    for i in range(len(latents) - 1):
        frames.append(latents[i])
        for j in range(1, interp_frames + 1):
            alpha = j / (interp_frames + 1)
            frames.append((1 - alpha) * latents[i] + alpha * latents[i + 1])
    frames.append(latents[-1])
    return frames   # list of (1, 3, 36, 36) tensors


# ── decoding ──────────────────────────────────────────────────────────────────

def decode_latent(latent, model):
    """Decode one latent → numpy array (36, 36, 3) in [0, 1]."""
    decoded = model.decode_first_stage(latent)
    pred = torch.clamp(decoded, 0.0, 1.0)[0]
    return pred.permute(1, 2, 0).cpu().numpy()  # (H, W, 3)


# ── Folium map rendering ──────────────────────────────────────────────────────

def _restore_matrix_36(square):
    ret = np.zeros((36 * 36, 3))
    for i in range(36):
        for d in range(3):
            ret[36 * i: 36 * (i + 1), d] = square[:, i, d]
    return ret[:1260, :]


_GCJ_AXIS   = 6378245.0
_GCJ_OFFSET = 0.00669342162296594323
_PI         = math.pi


def _gcj2wgs(gcj_lat, gcj_lon):
    def _trans_lat(x, y):
        r = -100 + 2*x + 3*y + 0.2*y*y + 0.1*x*y + 0.2*math.sqrt(abs(x))
        r += (20*math.sin(6*x*_PI) + 20*math.sin(2*x*_PI)) * 2/3
        r += (20*math.sin(y*_PI) + 40*math.sin(y/3*_PI)) * 2/3
        r += (160*math.sin(y/12*_PI) + 320*math.sin(y/30*_PI)) * 2/3
        return r

    def _trans_lon(x, y):
        r = 300 + x + 2*y + 0.1*x*x + 0.1*x*y + 0.1*math.sqrt(abs(x))
        r += (20*math.sin(6*x*_PI) + 20*math.sin(2*x*_PI)) * 2/3
        r += (20*math.sin(x*_PI) + 40*math.sin(x/3*_PI)) * 2/3
        r += (150*math.sin(x/12*_PI) + 300*math.sin(x/30*_PI)) * 2/3
        return r

    x, y = gcj_lon - 105.0, gcj_lat - 35.0
    dLat = _trans_lat(x, y)
    dLon = _trans_lon(x, y)
    rad_lat = gcj_lat / 180 * _PI
    magic = math.sin(rad_lat)
    magic = 1 - _GCJ_OFFSET * magic * magic
    sq = math.sqrt(magic)
    dLat = (dLat * 180) / ((_GCJ_AXIS * (1 - _GCJ_OFFSET)) / (magic * sq) * _PI)
    dLon = (dLon * 180) / (_GCJ_AXIS / sq * math.cos(rad_lat) * _PI)
    return gcj_lat - dLat, gcj_lon - dLon


def _value_to_color(v, dimension):
    if dimension == 1:
        colors = ['#9c1111', '#e69138', '#f2dd29', '#6ad739', '#306850']
    else:
        colors = ['#306850', '#6ad739', '#f2dd29', '#e69138', '#9c1111']
    cmap = LinearSegmentedColormap.from_list('_', colors)
    rgb = cmap(float(np.clip(v, 0, 1)), bytes=True)[:3]
    return mcolors.rgb2hex((rgb[0]/255, rgb[1]/255, rgb[2]/255))


def save_html_map(pred, roads, out_path, dimension, caption, timestamp,
                  val_min=None, val_max=None):
    import folium
    r = _restore_matrix_36(pred)
    vals = r[:, dimension]

    # Normalise to [0, 1] for colour mapping.
    # If global bounds are supplied (cross-frame comparison), use them;
    # otherwise normalise per-frame so the full colour scale is always used.
    lo = val_min if val_min is not None else vals.min()
    hi = val_max if val_max is not None else vals.max()
    span = hi - lo
    vals_norm = (vals - lo) / span if span > 1e-6 else np.full_like(vals, 0.5)

    san_map = folium.Map(
        location=(roads[0][0][0][0], roads[0][0][0][1]),
        zoom_start=13,
        tiles='CartoDB positron',
        attr='Beijing traffic',
        control_scale=True,
    )
    folium.map.Marker(
        [roads[0][0][0][0] + 0.08, roads[0][0][0][1]],
        icon=folium.DivIcon(html=f'<div style="font-size:14px;font-weight:bold;'
                                 f'background:white;padding:4px;border-radius:4px;">'
                                 f'{timestamp}</div>')
    ).add_to(san_map)
    # Caption banner at bottom
    folium.map.Marker(
        [roads[0][0][0][0] - 0.08, roads[0][0][0][1]],
        icon=folium.DivIcon(html=f'<div style="font-size:11px;background:rgba(255,255,255,0.85);'
                                 f'padding:4px 6px;border-radius:4px;max-width:340px;">'
                                 f'{caption}</div>')
    ).add_to(san_map)

    for i, road_list in enumerate(roads):
        color = _value_to_color(vals_norm[i], dimension)
        for section in road_list:
            raw_val = vals[i] * (150.0 if dimension == 1 else 5.0)
            unit = "km/h" if dimension == 1 else ""
            folium.PolyLine(section, color=color, weight=4, opacity=0.85,
                            tooltip=f"{raw_val:.1f}{unit}").add_to(san_map)

    san_map.save(out_path)


# ── animated GIF ──────────────────────────────────────────────────────────────

def save_gif(frames, timestamps, out_path, fps=4):
    """
    frames     : list of (36, 36, 3) numpy arrays in [0, 1]
    timestamps : list of strings, one per frame
    """
    images = []
    for pred, ts in zip(frames, timestamps):
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        fig.suptitle(ts, fontsize=10, wrap=True)

        for ax, ch, name, scale, clamp in [
            (axes[0], 0, "Congestion (0-5)",  5.0,  5.0),
            (axes[1], 1, "Speed (km/h)",      150.0, 99.0),
        ]:
            raw = np.clip(pred[:, :, ch] * scale, 0, clamp)
            colors = ['#306850','#6ad739','#f2dd29','#e69138','#9c1111'] if ch == 0 \
                else ['#9c1111','#e69138','#f2dd29','#6ad739','#306850']
            cmap_ch = LinearSegmentedColormap.from_list('_', colors)
            im = ax.imshow(raw, cmap=cmap_ch, vmin=0, vmax=clamp, origin='upper')
            ax.set_title(name, fontsize=9)
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        fig.tight_layout()

        # render figure to PIL image
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
        images.append(Image.fromarray(buf))
        plt.close(fig)

    duration_ms = int(1000 / fps)
    images[0].save(
        out_path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
    )
    print(f"GIF saved to {out_path}  ({len(images)} frames @ {fps} fps)")


# ── frame timestamp helper ────────────────────────────────────────────────────

def build_timestamps(start_time, duration_hours, n_keyframes, interp_frames):
    """Return a label string for every output frame (keyframes + interpolated)."""
    n_total = n_keyframes + (n_keyframes - 1) * interp_frames
    total_mins = duration_hours * 60
    step_mins  = total_mins / (n_total - 1) if n_total > 1 else 0

    import datetime
    # Try to parse start_time into a datetime so we can format nicely
    labels = []
    try:
        base = datetime.datetime.strptime(start_time.strip(), "%A %I:%M%p")
    except ValueError:
        try:
            base = datetime.datetime.strptime(start_time.strip(), "%A %I:%M %p")
        except ValueError:
            base = None

    for i in range(n_total):
        if base is not None:
            t = base + datetime.timedelta(minutes=i * step_mins)
            labels.append(t.strftime("%A %I:%M %p"))
        else:
            labels.append(f"{start_time} + {int(i * step_mins)}min")
    return labels


# ── road geometry loader ──────────────────────────────────────────────────────

def load_roads(roads_json):
    road_data = json.load(open(roads_json))
    roads = []
    for road in road_data:
        roads.append([])
        for section in road:
            roads[-1].append([])
            coords = section['coordList']
            for i in range(0, len(coords), 2):
                lat, lon = _gcj2wgs(coords[i+1], coords[i])
                roads[-1][-1].append((lat, lon))
    return roads


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ChatTraffic temporal sequence generator")
    # model
    parser.add_argument("--config",         required=True)
    parser.add_argument("--ckpt",           required=True)
    parser.add_argument("--ddim_steps",     type=int,   default=50)
    parser.add_argument("--ddim_eta",       type=float, default=0.0)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    # sequence
    parser.add_argument("--start_time",      default="Monday 7:00am",
                        help="Natural language start time (e.g. 'Monday 7:00am')")
    parser.add_argument("--duration_hours",  type=float, default=12.0,
                        help="Total time window in hours")
    parser.add_argument("--n_keyframes",     type=int,   default=6,
                        help="Number of captioned keyframes to generate")
    parser.add_argument("--interp_frames",   type=int,   default=3,
                        help="Interpolated frames inserted between each keyframe pair")
    # captions
    parser.add_argument("--captions_file",  default=None,
                        help="Optional: txt file with one caption per line (skips Ollama)")
    parser.add_argument("--context",        default="",
                        help="Extra context for Ollama, e.g. 'rainy day' or 'national holiday'")
    parser.add_argument("--edge_cases",    action="store_true",
                        help="Use edge-case prompt: extreme weather, major incidents, cascading failures")
    parser.add_argument("--ollama_model",   default="llama3",
                        help="Ollama model name (default: llama3)")
    parser.add_argument("--ollama_url",     default="http://localhost:11434")
    parser.add_argument("--ollama_timeout", type=int, default=600,
                        help="Seconds to wait for Ollama response (default: 600)")
    # output
    parser.add_argument("--outdir",         default="outputs/traffic_sequence")
    parser.add_argument("--roads",          default="datasets/traffic/Roads1260.json")
    parser.add_argument("--dimension",      type=int, default=1, choices=[0, 1],
                        help="Channel for HTML maps: 0=congestion, 1=speed (default)")
    parser.add_argument("--gif_fps",        type=int, default=4)
    parser.add_argument("--no_maps",        action="store_true",
                        help="Skip per-frame Folium HTML maps (faster)")
    parser.add_argument("--no_gif",         action="store_true",
                        help="Skip animated GIF output")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    maps_dir = os.path.join(args.outdir, "maps")
    npys_dir = os.path.join(args.outdir, "frames")
    os.makedirs(maps_dir, exist_ok=True)
    os.makedirs(npys_dir, exist_ok=True)

    # ── 1. captions ───────────────────────────────────────────────────────────
    if args.captions_file:
        captions = load_captions_file(args.captions_file)
        if len(captions) < args.n_keyframes:
            print(f"Warning: captions file has {len(captions)} lines but "
                  f"--n_keyframes={args.n_keyframes}. Using all available.")
            args.n_keyframes = len(captions)
        captions = captions[:args.n_keyframes]
    else:
        captions = generate_captions_ollama(
            n=args.n_keyframes,
            start_time=args.start_time,
            duration_hours=args.duration_hours,
            context=args.context,
            ollama_model=args.ollama_model,
            ollama_url=args.ollama_url,
            timeout=args.ollama_timeout,
            edge_cases=args.edge_cases,
        )

    captions_path = os.path.join(args.outdir, "captions.txt")
    with open(captions_path, "w") as f:
        for i, c in enumerate(captions):
            f.write(f"[Keyframe {i+1}] {c}\n")
    print(f"Captions written to {captions_path}")

    # ── 2. load model ─────────────────────────────────────────────────────────
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model = load_model(args.config, args.ckpt)

    from ldm.models.diffusion.flow_matching import LatentFlowMatching
    is_flow = isinstance(model, LatentFlowMatching)
    sampler = None
    if not is_flow:
        from ldm.models.diffusion.ddim import DDIMSampler
        sampler = DDIMSampler(model)

    # ── 3. generate keyframe latents ──────────────────────────────────────────
    print(f"\nGenerating {args.n_keyframes} keyframe latents …")
    keyframe_latents = []
    with torch.no_grad():
        with model.ema_scope():
            for i, caption in enumerate(tqdm(captions, desc="Keyframes")):
                latent = caption_to_latent(caption, model, sampler, args, is_flow)
                keyframe_latents.append(latent)

    # ── 4. interpolate ────────────────────────────────────────────────────────
    print(f"Interpolating ({args.interp_frames} frames between each keyframe pair) …")
    all_latents = interpolate_latents(keyframe_latents, args.interp_frames)
    n_total = len(all_latents)
    print(f"Total frames: {n_total}  "
          f"({args.n_keyframes} keyframes + "
          f"{(args.n_keyframes-1)*args.interp_frames} interpolated)")

    # ── 5. decode all frames ──────────────────────────────────────────────────
    print("Decoding all frames …")
    decoded_frames = []
    with torch.no_grad():
        for latent in tqdm(all_latents, desc="Decoding"):
            decoded_frames.append(decode_latent(latent, model))

    # build per-frame timestamp labels
    timestamps = build_timestamps(
        args.start_time, args.duration_hours, args.n_keyframes, args.interp_frames
    )

    # build per-frame caption labels (keyframes get real caption; interp frames get blend note)
    frame_captions = []
    step = args.interp_frames + 1
    for i in range(n_total):
        kf_idx = i // step
        sub_idx = i % step
        if sub_idx == 0:
            frame_captions.append(captions[kf_idx])
        else:
            c1 = captions[min(kf_idx,     len(captions)-1)]
            c2 = captions[min(kf_idx + 1, len(captions)-1)]
            frame_captions.append(f"[interp {sub_idx}/{step}] {c1[:40]}… → {c2[:40]}…")

    # ── 6. save .npy files ────────────────────────────────────────────────────
    print("Saving .npy frames …")
    for i, (pred, ts) in enumerate(zip(decoded_frames, timestamps)):
        np.save(os.path.join(npys_dir, f"frame_{i:04d}.npy"), pred)

    # ── 7. save HTML maps ─────────────────────────────────────────────────────
    if not args.no_maps and os.path.exists(args.roads):
        print("Building Folium HTML maps …")
        roads = load_roads(args.roads)

        # Compute global value bounds so colours are comparable across frames.
        all_vals = np.stack([
            _restore_matrix_36(f)[:, args.dimension] for f in decoded_frames
        ])
        g_min, g_max = float(all_vals.min()), float(all_vals.max())
        print(f"  Global {['congestion','speed'][args.dimension]} range: "
              f"{g_min:.3f} – {g_max:.3f}")

        for i, (pred, ts, cap) in enumerate(
                tqdm(zip(decoded_frames, timestamps, frame_captions),
                     total=n_total, desc="HTML maps")):
            out_path = os.path.join(maps_dir, f"frame_{i:04d}.html")
            save_html_map(pred, roads, out_path,
                          dimension=args.dimension, caption=cap, timestamp=ts,
                          val_min=g_min, val_max=g_max)
    elif not args.no_maps:
        print(f"Skipping HTML maps — {args.roads} not found.")

    # ── 8. animated GIF ───────────────────────────────────────────────────────
    if not args.no_gif:
        gif_path = os.path.join(args.outdir, "traffic_sequence.gif")
        save_gif(decoded_frames, timestamps, gif_path, fps=args.gif_fps)

    # ── 9. summary ────────────────────────────────────────────────────────────
    print("\n" + "="*56)
    print("  OUTPUT SUMMARY")
    print("="*56)
    print(f"  Captions   : {captions_path}")
    print(f"  Frames     : {npys_dir}/  ({n_total} .npy files)")
    if not args.no_maps and os.path.exists(args.roads):
        print(f"  HTML maps  : {maps_dir}/  ({n_total} interactive maps)")
    if not args.no_gif:
        print(f"  Animation  : {os.path.join(args.outdir, 'traffic_sequence.gif')}")
    print("="*56)


if __name__ == "__main__":
    main()
