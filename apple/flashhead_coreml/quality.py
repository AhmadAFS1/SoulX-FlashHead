"""A/B quality metrics and labelled comparison videos for CPU renders.

Each run directory comes from ``cpu_runtime`` (``frames.npy``, ``run.json``,
``video.mp4``). Pixel metrics compare decoded RGB frames directly; landmark
metrics reuse the historical PRO review definitions
(``benchmarks/pro_quantization_v2_20260918/review.py``): MediaPipe FaceMesh
inner-lip opening normalised by inter-ocular distance, paired opening
correlation, and the oral-region Laplacian-variance ratio on frames where both
mouths are open (teeth/edge sharpness). Lip-audio correlation is the best
Pearson correlation, within +/-2 frames, between a run's mouth opening and the
clip's per-frame speech RMS.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np

FPS, SAMPLE_RATE = 25, 16000


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float('inf') if mse == 0 else float(10 * np.log10(255.0 ** 2 / mse))


def ssim(a, b):
    from skimage.metrics import structural_similarity
    return float(structural_similarity(a, b, channel_axis=2, data_range=255))


def landmarks(frames):
    import cv2
    import mediapipe as mp
    rows = []
    with mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True,
                                         min_detection_confidence=.5, min_tracking_confidence=.5) as mesh:
        for index, rgb in enumerate(frames):
            prediction = mesh.process(rgb)
            row = {'frame': index, 'detected': bool(prediction.multi_face_landmarks)}
            if prediction.multi_face_landmarks:
                h, w = rgb.shape[:2]
                points = np.array([(p.x * w, p.y * h) for p in prediction.multi_face_landmarks[0].landmark])
                opening = np.linalg.norm(points[13] - points[14])
                centre = points[[13, 14]].mean(0)
                width = np.linalg.norm(points[61] - points[291])
                cx, cy = np.round(centre).astype(int)
                half_h, half_w = max(3, round(opening * .75)), max(8, round(width * .32))
                oral = cv2.cvtColor(np.ascontiguousarray(rgb[max(0, cy - half_h):cy + half_h + 1,
                                                             max(0, cx - half_w):cx + half_w + 1]), cv2.COLOR_RGB2GRAY)
                row.update(opening_px=float(opening),
                           opening_norm=float(opening / max(np.linalg.norm(points[33] - points[263]), 1e-6)),
                           oral_edge_energy=float(cv2.Laplacian(oral, cv2.CV_64F).var()),
                           mouth_center=centre.tolist(), mouth_width=float(width),
                           mouth_box=[int(max(0, cx - 2 * half_w)), int(max(0, cy - 2 * half_h - 6)),
                                      int(min(w, cx + 2 * half_w)), int(min(h, cy + 2 * half_h + 6))])
            rows.append(row)
    return rows


def speech_rms(audio, frames):
    per = SAMPLE_RATE // FPS
    padded = np.zeros(frames * per, dtype=np.float32)
    padded[:min(len(audio), len(padded))] = audio[:len(padded)]
    return np.sqrt((padded.reshape(frames, per) ** 2).mean(1))


def lip_audio_correlation(rows, rms):
    opening = np.array([r.get('opening_norm', np.nan) for r in rows])
    best = None
    for lag in range(-2, 3):
        a = opening[max(0, lag):len(opening) + min(0, lag)]
        b = rms[max(0, -lag):len(rms) + min(0, -lag)]
        valid = np.isfinite(a)
        if valid.sum() > 10:
            c = float(np.corrcoef(a[valid], b[valid])[0, 1])
            if best is None or c > best[0]:
                best = (c, lag)
    return {'correlation': best[0], 'lag_frames': best[1]} if best else None


def mouth_psnr(a_frames, b_frames, rows):
    values = []
    for a, b, row in zip(a_frames, b_frames, rows):
        if row.get('detected'):
            x0, y0, x1, y1 = row['mouth_box']
            values.append(psnr(a[y0:y1, x0:x1], b[y0:y1, x0:x1]))
    finite = [v for v in values if np.isfinite(v)]
    return {'frames': len(values), 'identical_crops': len(values) - len(finite),
            'median_db': float(np.median(finite)) if finite else float('inf'),
            'min_db': float(np.min(finite)) if finite else float('inf')}


def paired(baseline_rows, candidate_rows):
    pairs = [(l, r) for l, r in zip(baseline_rows, candidate_rows) if l['detected'] and r['detected']]
    opened = [(l, r) for l, r in pairs if l['opening_px'] >= 4 and r['opening_px'] >= 4]
    if len(pairs) < 2:
        return {'detected_pairs': len(pairs)}
    corr = np.corrcoef([l['opening_norm'] for l, _ in pairs], [r['opening_norm'] for _, r in pairs])[0, 1]
    return {'detected_pairs': len(pairs), 'both_open_pairs': len(opened),
            'opening_correlation': float(corr) if np.isfinite(corr) else None,
            'median_abs_opening_difference': float(np.median([abs(l['opening_norm'] - r['opening_norm']) for l, r in pairs])),
            'median_mouth_center_distance_px': float(np.median([np.linalg.norm(np.subtract(l['mouth_center'], r['mouth_center']))
                                                                for l, r in pairs])),
            'median_edge_ratio_on_both_open': float(np.median([r['oral_edge_energy'] / max(l['oral_edge_energy'], 1e-6)
                                                              for l, r in opened])) if opened else None}


def latent_errors(baseline_dir: Path, candidate_dir: Path):
    """Per-window NRMSE of the denoised latents, generated slots only."""
    a, b = np.load(baseline_dir / 'latents.npy'), np.load(candidate_dir / 'latents.npy')
    rows = []
    for w in range(min(len(a), len(b))):
        m = 1 if w == 0 else 2
        x, y = a[w, :, m:].astype(np.float64), b[w, :, m:].astype(np.float64)
        rows.append(float(np.sqrt(np.mean((x - y) ** 2)) / max(np.sqrt(np.mean(x * x)), 1e-12)))
    return rows


def window_psnr(a, b, per=28):
    out = []
    for start in range(0, len(a), per):
        values = [psnr(x, y) for x, y in zip(a[start:start + per], b[start:start + per])]
        finite = [v for v in values if np.isfinite(v)]
        out.append(float(np.median(finite)) if finite else float('inf'))
    return out


def compare(baseline_dir: Path, candidate_dir: Path, rows_cache: dict):
    a, b = np.load(baseline_dir / 'frames.npy'), np.load(candidate_dir / 'frames.npy')
    if a.shape != b.shape:
        raise ValueError(f'Frame shapes differ: {a.shape} vs {b.shape}')
    diff = np.abs(a.astype(np.int16) - b.astype(np.int16))
    per_frame_psnr = [psnr(x, y) for x, y in zip(a, b)]
    finite = [v for v in per_frame_psnr if np.isfinite(v)]
    da, db = np.diff(a.astype(np.float64), axis=0), np.diff(b.astype(np.float64), axis=0)
    rows_a = rows_cache.setdefault(str(baseline_dir), landmarks(a))
    rows_b = rows_cache.setdefault(str(candidate_dir), landmarks(b))
    colour = np.abs(a.reshape(len(a), -1, 3).mean(1) - b.reshape(len(b), -1, 3).mean(1)).mean()
    return {
        'frames': int(len(a)), 'identical_frames': int(sum(np.array_equal(x, y) for x, y in zip(a, b))),
        'max_abs_u8': int(diff.max()), 'mean_abs_u8': float(diff.mean()),
        'psnr_db': {'median': float(np.median(finite)) if finite else float('inf'),
                    'min': float(np.min(finite)) if finite else float('inf')},
        'ssim': {'median': float(np.median([ssim(x, y) for x, y in zip(a[::4], b[::4])]))},
        'mouth_psnr_db': mouth_psnr(a, b, rows_a),
        'temporal_delta_rmse_u8': float(np.sqrt(np.mean((da - db) ** 2))) if len(a) > 1 else 0.0,
        'mean_colour_shift_u8': float(colour),
        'landmarks': paired(rows_a, rows_b),
        'window_median_psnr_db': window_psnr(a, b),
        'window_latent_nrmse': latent_errors(baseline_dir, candidate_dir),
    }


def run_summary(run_dir: Path, rows_cache: dict):
    from .cpu_runtime import read_audio
    meta = json.loads((run_dir / 'run.json').read_text())
    frames = np.load(run_dir / 'frames.npy')
    rows = rows_cache.setdefault(str(run_dir), landmarks(frames))
    audio = read_audio(Path(meta['audio']))
    detected = [r for r in rows if r['detected']]
    opened = [r for r in detected if r['opening_px'] >= 4]
    dit = [w['seconds']['dit_step0'] + w['seconds']['dit_step1'] for w in meta['windows']]
    return {'run': str(run_dir), 'mode': meta['mode'], 'variant': meta['variant'], 'frames': meta['frames'],
            'frames_sha256': meta['frames_sha256'], 'detected_frames': len(detected),
            'median_opening_norm': float(np.median([r['opening_norm'] for r in detected])) if detected else None,
            'median_open_oral_edge_energy': float(np.median([r['oral_edge_energy'] for r in opened])) if opened else None,
            'lip_audio': lip_audio_correlation(rows, speech_rms(audio, len(frames))),
            'cpu_dit_seconds_per_window_median': float(np.median(dit[1:] if len(dit) > 1 else dit)),
            'cpu_window_seconds_median': float(np.median([w['seconds']['window'] for w in meta['windows']][1:] or [0]))}


# --------------------------------------------------------------------------
# Comparison video: labelled panels plus an amplified difference panel.
# --------------------------------------------------------------------------
def _font(size):
    from PIL import ImageFont
    for path in ('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
                 '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf'):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def comparison_video(output: Path, runs: list[tuple[str, Path]], diff_pair=(0, 1), gain=8, scale=2, title=''):
    from PIL import Image, ImageDraw
    from .cpu_runtime import read_audio, write_video
    frames = [np.load(path / 'frames.npy') for _, path in runs]
    count = min(len(f) for f in frames)
    a, b = frames[diff_pair[0]][:count], frames[diff_pair[1]][:count]
    heat = np.clip(np.abs(a.astype(np.int16) - b.astype(np.int16)).max(axis=3) * gain, 0, 255).astype(np.uint8)
    labels = [label for label, _ in runs] + [f'|{runs[diff_pair[1]][0].split(chr(10))[0]} - '
                                             f'{runs[diff_pair[0]][0].split(chr(10))[0]}| x{gain}']
    h, w = frames[0].shape[1:3]
    header = 44 * scale // 2 + (18 if title else 0)
    font, small = _font(11 * scale // 2 + 2), _font(10 * scale // 2 + 1)
    out = []
    for i in range(count):
        panels = [f[i] for f in frames] + [np.stack([heat[i]] * 3, axis=2)]
        canvas = Image.new('RGB', (len(panels) * w * scale, h * scale + header), (18, 18, 18))
        draw = ImageDraw.Draw(canvas)
        if title:
            draw.text((6, 3), title, fill=(230, 230, 230), font=small)
        for k, (panel, label) in enumerate(zip(panels, labels)):
            canvas.paste(Image.fromarray(panel).resize((w * scale, h * scale), Image.Resampling.NEAREST),
                         (k * w * scale, header))
            for line_no, line in enumerate(label.split('\n')[:2]):
                draw.text((k * w * scale + 6, (18 if title else 0) + 4 + line_no * (12 * scale // 2 + 3)), line,
                          fill=(255, 214, 102) if line_no == 0 else (200, 200, 200), font=font if line_no == 0 else small)
        draw.text((len(panels) * w * scale - 70, header - 14), f'{i:04d}', fill=(150, 150, 150), font=small)
        out.append(np.asarray(canvas))
    meta = json.loads((runs[0][1] / 'run.json').read_text())
    audio = read_audio(Path(meta['audio']))[:count * (SAMPLE_RATE // FPS)]
    write_video(output, np.stack(out), audio, crf=14)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--reference', type=Path, required=True, help='Baseline run directory (e.g. INT8 baseline)')
    parser.add_argument('--candidate', type=Path, action='append', required=True)
    parser.add_argument('--floor', type=Path, help='FP32 run used to report the quantization floor')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cache = {}
    report = {'reference': run_summary(args.reference, cache), 'candidates': {}}
    if args.floor:
        report['floor'] = run_summary(args.floor, cache)
        report['floor_vs_reference'] = compare(args.floor, args.reference, cache)
    for candidate in args.candidate:
        report['candidates'][str(candidate)] = {'summary': run_summary(candidate, cache),
                                                'vs_reference': compare(args.reference, candidate, cache)}
        if args.floor:
            report['candidates'][str(candidate)]['vs_floor'] = compare(args.floor, candidate, cache)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=1)[:4000])


if __name__ == '__main__':
    main()
