"""Distil a block-pruned Wan decoder from the full decoder on the pipeline's own latents.

Why
---
Bypassing decoder residual blocks is the only structural lever whose failure mode is *soft*
rather than *wrong* (docs/research/PRO_40FPS_ITERATION_LOG_2026-09-21.md, 2026-09-22
section): skipping `upsamples[9,10,13,14]` at 576x320 runs at 27.4 FPS with lip motion
tracking the reference (opening correlation 0.90) and stable colour, but the mouth loses
~75% of its edge energy because the removed full-resolution blocks are the ones that put the
high-frequency detail in. The surviving blocks were never asked to do that job alone. This
script asks them to: it fine-tunes the surviving modules of the pruned decoder to reproduce
the full decoder's frames on latents the DiT actually produced (dumped by the harness with
`SOULX_DUMP_DIR`, no overlap-skip, so every pair is a full 9-latent window and its 33 frames
from the shipped FP16 stage-engine decoder).

What
----
* Student: the stock VAE with `install_decoder_block_skip(vae, --skip)`; only the modules in
  `--train` (default: the Resample feeding each pruned group, the surviving block of each
  pruned group, and the head) are trainable, kept in fp32 with bf16 autocast; everything
  upstream stays frozen bf16, so autograd only spans the trainable tail.
* Forward mirrors `WanVAE_.decode` exactly (scale, `conv2`, per-latent-frame decoder calls
  through the shared causal cache) with truncated backprop: each latent frame's loss is
  backpropagated on its own and the cache entries are detached afterwards, so memory is one
  4-frame chunk at 576x320.
* Loss: mouth-weighted L1 on the frames plus L1 on spatial gradients (the sharpness the
  pruning removed), against the teacher frames clamped to [-1, 1] like `WanVAE.decode`.
* Output: the full VAE state_dict in the STOCK module layout (the harness and
  `decoder_stages.py --vae-weights` load it before any rewrite), bf16, plus a JSON log with
  the hold-out PSNR / sharpness before and after.

Evaluation numbers are decoder-only fidelity to the teacher on held-out windows; the real
gate is still the harness run + review.py + labelled video.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
MOUTH = (slice(193, 289), slice(86, 246))  # rows, cols of the review's mouth crop at 576x320


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=ROOT / "benchmarks/pro_30fps_20260922/distill-data")
    ap.add_argument("--skip", type=int, nargs="+", default=[9, 10, 13, 14])
    ap.add_argument("--train", nargs="+", default=["upsamples.7", "upsamples.8", "upsamples.11", "upsamples.12", "head"])
    ap.add_argument("--holdout", default="tts-conversational", help="fixture id held out for evaluation")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--min-lr", type=float, default=5e-6)
    ap.add_argument("--grad-weight", type=float, default=1.0)
    ap.add_argument("--mouth-weight", type=float, default=3.0)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--max-windows", type=int, default=None, help="debug: cap training windows per epoch")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--out", type=Path, default=ROOT / "benchmarks/pro_30fps_20260922/distill/vae_skip9-10-13-14_ft.pth")
    ap.add_argument("--log", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init-weights", type=Path, default=None,
                    help="stock-layout state_dict to load before pruning (continue a previous distillation)")
    return ap.parse_args()


def load_pair(path):
    d = torch.load(path, map_location="cpu", weights_only=False)
    zs = d["args"][0]  # (16, T, h, w) bf16, the tensor WanVAE.decode received
    frames = d["out"]  # (1, 3, 1 + 4*(T-1), H, W) bf16, already clamped to [-1, 1]
    if zs.ndim != 4 or frames.ndim != 5 or frames.shape[2] != 1 + 4 * (zs.shape[1] - 1):
        raise ValueError(f"unexpected dump shapes {tuple(zs.shape)} / {tuple(frames.shape)} in {path}")
    return zs, frames


def frame_slice(i):
    return slice(0, 1) if i == 0 else slice(4 * i - 3, 4 * i + 1)


def spatial_grads(x):
    return x[..., 1:, :] - x[..., :-1, :], x[..., :, 1:] - x[..., :, :-1]


def make_weight(shape, mouth_weight, device):
    w = torch.ones(shape[-2:], device=device)
    w[MOUTH] = mouth_weight
    return w / w.mean()


def loss_fn(pred, tgt, weight, grad_weight):
    l1 = ((pred - tgt).abs() * weight).mean()
    gx_p, gy_p = spatial_grads(pred)
    gx_t, gy_t = spatial_grads(tgt)
    grad = ((gx_p - gx_t).abs() * weight[1:, :]).mean() + ((gy_p - gy_t).abs() * weight[:, 1:]).mean()
    return l1 + grad_weight * grad


def psnr(a, b):
    mse = ((a - b) ** 2).mean().item()
    return 10 * math.log10(4.0 / max(mse, 1e-12))  # range 2 ([-1, 1])


def sharpness(a):
    gx, gy = spatial_grads(a)
    return gx.abs().mean().item() + gy.abs().mean().item()


class Student:
    def __init__(self, vae, skip, train_names):
        from soulx_rtc.pro_decoder_ops import install_decoder_block_skip

        self.vae = vae
        self.model = vae.model
        # stock-layout copy for saving (taken BEFORE the skip wrappers rename nothing but wrap modules)
        self.stock_state = {k: v.detach().clone().cpu() for k, v in self.model.state_dict().items()}
        self.skip_manifest = install_decoder_block_skip(vae, list(skip))
        for p in self.model.parameters():
            p.requires_grad_(False)
        decoder = self.model.decoder
        self.trainable = {}
        for name in train_names:
            module = decoder.head if name == "head" else decoder.upsamples[int(name.split(".")[1])]
            if module.__class__.__name__ == "SkippedResidualBlock":
                raise ValueError(f"{name} is a skipped block; cannot train it")
            module.float()
            for p in module.parameters():
                p.requires_grad_(True)
            self.trainable[name] = module
        self.params = [p for m in self.trainable.values() for p in m.parameters()]
        self.n_params = sum(p.numel() for p in self.params)
        scale = vae.scale
        self.mean = scale[0].view(1, -1, 1, 1, 1).to(torch.float32)
        self.inv_std = scale[1].view(1, -1, 1, 1, 1).to(torch.float32)

    def window(self, zs, teacher, *, train, weight=None, grad_weight=1.0):
        """One 9-latent window through the pruned decoder with truncated backprop.

        Mirrors WanVAE_.decode: clear cache, de-normalise, conv2, then one decoder call per
        latent frame sharing _feat_map / _conv_idx; WanVAE.decode's final clamp is applied.
        """
        model = self.model
        model.clear_cache()
        z = zs.to("cuda", torch.float32).unsqueeze(0)
        z = z / self.inv_std + self.mean
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            x = model.conv2(z.to(torch.bfloat16))
        losses, outs = [], []
        for i in range(z.shape[2]):
            model._conv_idx = [0]
            ctx = torch.enable_grad() if train else torch.no_grad()
            with ctx, torch.autocast("cuda", dtype=torch.bfloat16):
                out_i = model.decoder(x[:, :, i : i + 1], feat_cache=model._feat_map, feat_idx=model._conv_idx)
            pred = out_i.float().clamp(-1, 1)
            tgt = teacher[:, :, frame_slice(i)].to("cuda", torch.float32)
            if train:
                loss = loss_fn(pred, tgt, weight, grad_weight)
                loss.backward()
                losses.append(loss.item())
            else:
                outs.append(pred.detach())
            # truncated BPTT: the causal cache carries values, not gradients, across chunks
            for k, c in enumerate(model._feat_map):
                if torch.is_tensor(c):
                    model._feat_map[k] = c.detach()
        model.clear_cache()
        return losses, (torch.cat(outs, dim=2) if outs else None)

    def export_state(self):
        state = dict(self.stock_state)
        for name, module in self.trainable.items():
            prefix = f"decoder.{name}."
            for k, v in module.state_dict().items():
                key = prefix + k
                if key not in state:
                    raise KeyError(f"trained key {key} not in the stock layout")
                state[key] = v.detach().to(torch.bfloat16).cpu()
        return state


@torch.no_grad()
def evaluate(student, pairs):
    rows = []
    for zs, teacher in pairs:
        _, pred = student.window(zs, teacher, train=False)
        tgt = teacher.to("cuda", torch.float32)
        pm, tm = pred[..., MOUTH[0], MOUTH[1]], tgt[..., MOUTH[0], MOUTH[1]]
        rows.append({
            "psnr": psnr(pred, tgt), "mouth_psnr": psnr(pm, tm),
            "sharpness_ratio": sharpness(pred) / sharpness(tgt),
            "mouth_sharpness_ratio": sharpness(pm) / sharpness(tm),
        })
    return {k: sum(r[k] for r in rows) / len(rows) for k in rows[0]} | {"windows": len(rows)}


def main():
    args = parse()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    from soulx_rtc.gpu_lease import acquire_gpu_lease
    from flash_head.wan.modules import WanVAE

    lease = acquire_gpu_lease(None)
    try:
        started = time.time()
        vae = WanVAE(vae_path=str(ROOT / "models/SoulX-FlashHead-1_3B/VAE_Wan/Wan2.1_VAE.pth"), dtype=torch.bfloat16, device="cuda")
        if args.init_weights is not None:
            state = torch.load(args.init_weights, map_location="cpu", weights_only=True)
            missing, unexpected = vae.model.load_state_dict(state, strict=False)
            if unexpected:
                raise SystemExit(f"--init-weights has unexpected keys: {list(unexpected)[:5]}")
            print(f"initialised from {args.init_weights} ({len(state)} keys, {len(missing)} missing)", flush=True)
        student = Student(vae, args.skip, args.train)
        files = sorted(glob.glob(str(args.data / "dump-*" / "decode-*.pt")))
        if not files:
            raise SystemExit(f"no dumps under {args.data}")
        hold = [f for f in files if f"dump-{args.holdout}-" in f]
        train_files = [f for f in files if f not in hold]
        print(f"pairs: {len(train_files)} train / {len(hold)} hold-out ({args.holdout}); trainable params {student.n_params/1e6:.2f} M in {list(student.trainable)}", flush=True)
        hold_pairs = [load_pair(f) for f in hold]
        log = {"args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
               "skip_manifest": student.skip_manifest, "trainable_params": student.n_params,
               "train_pairs": len(train_files), "holdout_pairs": len(hold), "epochs": []}
        before = evaluate(student, hold_pairs)
        log["before"] = before
        print("hold-out BEFORE:", json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in before.items()}), flush=True)
        if args.eval_only:
            return
        weight = None
        opt = torch.optim.AdamW(student.params, lr=args.lr, betas=(0.9, 0.99), weight_decay=0.0)
        per_epoch = len(train_files) if args.max_windows is None else min(args.max_windows, len(train_files))
        total_steps = args.epochs * per_epoch
        step = 0
        for epoch in range(args.epochs):
            order = list(train_files)
            random.shuffle(order)
            order = order[:per_epoch]
            epoch_losses = []
            t0 = time.time()
            for f in order:
                zs, teacher = load_pair(f)
                if weight is None:
                    weight = make_weight(teacher.shape, args.mouth_weight, "cuda")
                lr = args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * step / max(total_steps, 1)))
                for g in opt.param_groups:
                    g["lr"] = lr
                opt.zero_grad(set_to_none=True)
                losses, _ = student.window(zs, teacher, train=True, weight=weight, grad_weight=args.grad_weight)
                torch.nn.utils.clip_grad_norm_(student.params, args.clip)
                opt.step()
                step += 1
                epoch_losses.append(sum(losses) / len(losses))
                if step % 10 == 0:
                    print(f"epoch {epoch} step {step}/{total_steps} loss {epoch_losses[-1]:.4f} (mean {sum(epoch_losses)/len(epoch_losses):.4f}) lr {lr:.2e} {time.time()-t0:.0f}s", flush=True)
            after = evaluate(student, hold_pairs)
            row = {"epoch": epoch, "train_loss": sum(epoch_losses) / len(epoch_losses), "seconds": time.time() - t0, "holdout": after}
            log["epochs"].append(row)
            print(f"epoch {epoch} done: train loss {row['train_loss']:.4f}; hold-out", json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in after.items()}), flush=True)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            torch.save(student.export_state(), args.out)
        log["after"] = log["epochs"][-1]["holdout"]
        log["wall_seconds"] = time.time() - started
        log_path = args.log or args.out.with_suffix(".json")
        log_path.write_text(json.dumps(log, indent=2) + "\n")
        print("saved", args.out, "and", log_path, flush=True)
    finally:
        lease.close()


if __name__ == "__main__":
    main()
