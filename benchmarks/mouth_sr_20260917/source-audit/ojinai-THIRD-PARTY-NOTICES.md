# Third-Party Notices — Ojin Avatar Service (`oj-avatar-service:kit-clean`)

This document lists the third-party software and model weights bundled in the delivered avatar
container image, the license under which each is redistributed, and the copyright/attribution
notices those licenses require. It is intended as a contract attachment for the KIT handover.

- **Delivery model:** the image is delivered **run-only**. This
  document covers only *third-party* components redistributed inside the image; Ojin's own code and
  the mouth-refiner weights are proprietary and are not third-party.
- **Last reviewed:** 2026-07-23.

---

## 1. Compliance summary (read this first)

| Concern | Status |
|---|---|
| **GPL / AGPL executables or libraries** | **None in the delivered image.** All GPL ffmpeg binaries/libraries have been removed — see §6. |
| **Copyleft (LGPL / MPL) components** | Present, all **dynamically linked and replaceable** → compliant. See §5. |
| **Permissive libraries (MIT / BSD / Apache-2.0 / ISC)** | Redistributable; attributions reproduced in §7 / Appendix A. |
| **NVIDIA CUDA / cuDNN / TensorRT** | Redistributable as runtime libraries under NVIDIA's SLAs. See §4. |

---

## 2. Bundled model weights

All weights are baked into the image under `/app/models/flashhead/flashhead/models/`. Only the
**Lite** (oris-portrait) pipeline ships; the Pro pipeline and its Wan2.1 VAE are purged at build
time (`bake_weights.sh`).

| Component | Source | License (SPDX / name) |
|---|---|---|
| **SoulX-FlashHead-1_3B** (DiT / Lite weights) | HF `Soul-AILab/SoulX-FlashHead-1_3B` | **Apache-2.0** (declared on model card) |
| **LTX-Video VAE** (Lite VAE, `VAE_LTX/`) | Lightricks `Lightricks/LTX-Video`, bundled inside the SoulX repo |**RAIL-M** — `license: other` |
| **wav2vec2-base-960h** (audio encoder) | HF `facebook/wav2vec2-base-960h` | **Apache-2.0** |
| **MediaPipe FaceLandmarker** (`face_landmarker.task`, mouth ROI) | Google MediaPipe public CDN | **Apache-2.0** |
| **2× upscaler** (SRVGGNetCompact, TensorRT engine) | Real-ESRGAN architecture; Ojin-built engine | **BSD-3-Clause** (Real-ESRGAN architecture/weights) — attribution required |
| **mouth-refiner** (`refiner_weights.pt`) | Ojin S3 (proprietary) | **Proprietary — Ojin/Journee** (not third-party) |

**Upstream lineage (not shipped as weights, listed for completeness):** SoulX-FlashHead is built upon
**Wan2.1** (`github.com/Wan-Video/Wan2.1`, Apache-2.0) and uses distillation techniques from **DMD2**
and **Self-Forcing++**. The Wan VAE (Pro pipeline) is **purged** from this image.

### License links
- SoulX-FlashHead-1_3B — https://huggingface.co/Soul-AILab/SoulX-FlashHead-1_3B
- LTX-Video VAE — https://huggingface.co/Lightricks/LTX-Video/blob/main/ltx-video-2b-v0.9.license.txt
- wav2vec2-base-960h — https://huggingface.co/facebook/wav2vec2-base-960h
- MediaPipe FaceLandmarker — https://ai.google.dev/edge/mediapipe
- Real-ESRGAN (SRVGGNetCompact) — https://github.com/xinntao/Real-ESRGAN

---

## 3. Notable redistributed libraries

The image bundles the Python inference stack and its native dependencies. The most significant
components are listed here; the full package manifest is in Appendix A.

| Component | License (SPDX) | Notes |
|---|---|---|
| PyTorch (`torch`, `torchvision`) | BSD-3-Clause | Core tensor/DL runtime |
| NVIDIA CUDA / cuDNN / NCCL / cuBLAS runtime (`nvidia-*-cu12`) | NVIDIA proprietary (redistributable) | See §4 |
| TensorRT (`tensorrt*`) | NVIDIA proprietary (redistributable) | Upscaler + engine runtime; see §4 |
| `transformers`, `diffusers`, `accelerate`, `tokenizers`, `huggingface-hub` | Apache-2.0 | HF model stack |
| `flash-attn`, `xformers` | BSD-3-Clause | Attention kernels |
| `opencv-python`, `opencv-python-headless` | Apache-2.0 (code) + **LGPL-2.1** bundled ffmpeg libs + (non-headless) **LGPLv3** Qt5 | See §5 |
| `numpy`, `scipy`, `scikit-image`, `scikit-learn`, `pandas` | BSD-3-Clause | |
| `mediapipe` | Apache-2.0 | FaceLandmarker runtime |
| `librosa` | ISC | Audio features |
| `Pillow` | MIT-CMU (HPND) | Imaging |
| `sentencepiece`, `protobuf`, `absl-py`, `grpcio` | Apache-2.0 / BSD-3-Clause | |
| `pydantic`, `fastapi`, `starlette`, `uvicorn`, `websockets` (proxy) | MIT / BSD-3-Clause | Client-facing proxy |
| `imageio` / `imageio-ffmpeg` (wrapper) | BSD-2-Clause | Wrapper kept; **bundled GPL ffmpeg binary removed** — see §6 |
| `easydict` | **LGPLv3** | See §5 |
| `soxr` | **LGPL-2.1-or-later** | Resampler (librosa); see §5 |

Full, per-package license IDs are in **Appendix A**.

---

## 4. NVIDIA components (CUDA, cuDNN, NCCL, TensorRT)

The image bundles the NVIDIA CUDA 12.8 runtime libraries, cuDNN, NCCL, and TensorRT (as Python
wheels under `nvidia-*-cu12` and `tensorrt*`, plus the CUDA base image runtime). These are
**proprietary NVIDIA software**, redistributed under NVIDIA's license agreements:

- CUDA Toolkit / runtime libraries — **NVIDIA CUDA Toolkit EULA / Software License Agreement**, which
  grants the right to redistribute the specified runtime libraries as part of an application.
- cuDNN — **NVIDIA cuDNN Software License Agreement** (redistribution of the runtime libraries with
  an application is permitted).
- TensorRT — **NVIDIA TensorRT Software License Agreement** (runtime redistribution permitted).

Required notice: *"This software contains source code provided by NVIDIA Corporation."* No NVIDIA
source is modified; only the redistributable runtime libraries are included. References:
- CUDA EULA — https://docs.nvidia.com/cuda/eula/index.html
- cuDNN SLA — https://docs.nvidia.com/deeplearning/cudnn/sla/index.html
- TensorRT SLA — https://docs.nvidia.com/deeplearning/tensorrt/sla/index.html

---

## 5. Copyleft components present (LGPL / MPL) — compliance statement

No GPL/AGPL is present (§6). The following weak-copyleft components remain and are **compliant** for
this delivery because each is **dynamically linked** and shipped in a form the recipient can replace,
with its source/license available in the image:

| Component | License | Why compliant here |
|---|---|---|
| `easydict` | LGPLv3 | Pure-Python; shipped as replaceable `.py` source in site-packages. |
| `soxr` | LGPL-2.1-or-later | C extension (`.so`), dynamically loaded and replaceable; a transitive dep of `librosa`. |
| `opencv` bundled ffmpeg libs (`libavcodec/format/util/swscale`) | LGPL-2.1 | Dynamically linked `.so`; **no** x264/x265/postproc (no `--enable-gpl`), so LGPL not GPL. |
| `opencv-python` (non-headless) bundled **Qt5** | LGPLv3 | Dynamically linked `.so`, replaceable. *(Follow-up: consolidate to `opencv-python-headless` only to drop the Qt5 LGPLv3 surface — needs a rebuild + import re-verification.)* |
| `certifi`, `pathspec`, `orjson`, `tqdm` | MPL-2.0 | File-level copyleft; unmodified, source present in site-packages. |

**LGPL/MPL obligations met:** (1) the license texts are included (in each package's site-packages
`*.dist-info`/`LICENSE`, and referenced here); (2) the components are unmodified; (3) they are
dynamically linked and the recipient can replace them (site-packages ships the `.py`/`.so`, not a
statically-linked blob). No relinking-facilitation beyond that is required for dynamic LGPL linking.

---

## 6. GPL components removed from the image

Three GPL ffmpeg sources exist in the upstream Python/OS dependency graph. **None is used by the
delivered service** — the only ffmpeg-invoking module (`server/input_output/mp4_video_output.py`) is
imported solely by non-shipped `examples/` scripts, and audio reaches the container as raw PCM (no
media container is ever decoded). All three are therefore **removed at build time** so the delivered
image contains **no redistributed GPL code**:

1. **apt `ffmpeg`** (`/usr/bin/ffmpeg`, GPL) — **not installed** in the runtime stage (`Dockerfile`).
2. **`imageio-ffmpeg`'s bundled static ffmpeg** (`ffmpeg-linux64-*`, ~73 MB, **GPLv3** John Van Sickle
   build) — **deleted** from site-packages; the BSD-2-Clause `imageio_ffmpeg` Python wrapper is kept.
3. **`decord`'s bundled ffmpeg** — its libraries include **`libx264` (GPL-2.0-or-later)** and
   **`libpostproc` (GPL)** — the whole `decord` package is **removed**. It is unused: our code never
   imports it, and `transformers` references it only behind `is_decord_available()`, which reports
   `False` once the package is gone (its video path is not exercised by this service).

---

## 7. Required license texts

Many components share the four permissive licenses below. Reproduced here to satisfy the "reproduce
the copyright notice and this permission notice" requirement common to MIT/BSD; the full text of the
Apache License 2.0 is referenced (it is long and identical for all Apache-2.0 components).

### Representative copyright holders
- PyTorch — Copyright (c) Meta Platforms, Inc. and affiliates; and the PyTorch contributors.
- NumPy / SciPy — Copyright (c) NumPy Developers; SciPy Developers.
- OpenCV — Copyright (c) OpenCV team.
- Real-ESRGAN (SRVGGNetCompact upscaler) — Copyright (c) 2021 Xintao Wang.
- Hugging Face `transformers`/`diffusers`/`tokenizers` — Copyright (c) Hugging Face Inc.
- Pillow — Copyright (c) 1997-2011 Secret Labs AB, 1995-2011 Fredrik Lundh, 2010-present contributors.
- (Per-package copyright lines are in each package's `LICENSE` file inside the image.)

### MIT License

```
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial
portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```

### BSD 2-Clause License

```
Redistribution and use in source and binary forms, with or without modification, are permitted
provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions
   and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions
   and the following disclaimer in the documentation and/or other materials provided with the
   distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
IMPLIED WARRANTIES ... ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY ... DAMAGES ... ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE.
```

### BSD 3-Clause License

```
Redistribution and use in source and binary forms, with or without modification, are permitted
provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions
   and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions
   and the following disclaimer in the documentation and/or other materials provided with the
   distribution.
3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse
   or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
IMPLIED WARRANTIES ... ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY ... DAMAGES ... ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE.
```

### ISC License

```
Permission to use, copy, modify, and/or distribute this software for any purpose with or without fee
is hereby granted, provided that the above copyright notice and this permission notice appear in all
copies.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS
SOFTWARE ... IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY ... DAMAGES ... ARISING OUT OF OR IN
CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
```

### Apache License 2.0

Full text: https://www.apache.org/licenses/LICENSE-2.0 . Apache-2.0 requires that recipients receive
a copy of the license and retain any `NOTICE` file distributed with the component; each Apache-2.0
package in the image carries its `LICENSE` (and `NOTICE`, where present) inside its site-packages
`*.dist-info` directory.

### Native / system libraries (apt)
- **OpenSSL** (`libssl3`/`libcrypto`) — Apache-2.0 (OpenSSL 3.x). (oj-server)
- **libjpeg-turbo** (`libturbojpeg`) — BSD-3-Clause / IJG / zlib. (oj-server)
- **libsndfile** — LGPL-2.1 (dynamically linked `.so`; replaceable — compliant as in §5).
- **libglvnd** (`libgl1` — provides `libGL.so.1`) — BSD-3-Clause-like (MIT/BSD). Required by opencv.
- **GLib** (`libglib2.0-0` — `libgthread`) — LGPL-2.1 (dynamically linked; replaceable — as in §5). Required by opencv.
- **Mesa GL/EGL** (`libgles2-mesa`/`libegl1`) — MIT. Required by MediaPipe FaceLandmarker.

---

## Appendix A — Python package manifest

Representative list generated from the pinned build environment (`name  version  license`).
`decord` is excluded (removed from the image, §6). Some entries below are transitive/dev packages of
the build environment; **for the authoritative list, regenerate from the delivered image**:

```
docker run --rm --entrypoint python3 oj-avatar-service:kit-clean - <<'PY'
import importlib.metadata as m
for d in sorted(m.distributions(), key=lambda d: d.metadata["Name"].lower()):
    md = d.metadata
    lic = md.get("License-Expression") or next(
        (c.split("::")[-1].strip() for c in md.get_all("Classifier", []) if c.startswith("License")),
        (md.get("License") or "UNKNOWN").splitlines()[0][:60])
    print(f'{md["Name"]:<34} {md["Version"]:<16} {lic}')
PY
```

```
absl-py                            2.4.0            Apache-2.0
accelerate                         1.13.0           Apache
aiohappyeyeballs                   2.6.1            Python Software Foundation
aiohttp                            3.13.5           Apache-2.0 AND MIT
aiosignal                          1.4.0            Apache
annotated-doc                      0.0.4            MIT
annotated-types                    0.7.0            MIT
anyio                              4.9.0            MIT
async-lru                          2.0.5            MIT
asynciolimiter                     1.2.0            MIT
attrs                              26.1.0           MIT
audioread                          3.1.0            MIT
backports.asyncio.runner           1.2.0            Python Software Foundation
beautifulsoup4                     4.15.0           MIT
boto3                              1.42.83          Apache-2.0
botocore                           1.42.83          Apache-2.0
cbor2                              5.9.0            MIT
certifi                            2025.1.31        Mozilla Public 2.0
cffi                               2.0.0            MIT
charset-normalizer                 3.4.7            MIT
click                              8.1.8            BSD
colorama                           0.4.6            BSD
colored                            2.3.0            MIT
contourpy                          1.3.2            BSD
coverage                           7.8.0            Apache-2.0
cycler                             0.12.1           BSD
decorator                          5.1.1            BSD
Deprecated                         1.3.1            MIT
detect-installer                   0.1.0            0BSD
diffusers                          0.38.0           Apache
DistVAE                            0.0.0b5          UNKNOWN
dnspython                          2.7.0            ISC
easydict                           1.13             GNU Lesser General Public v3
einops                             0.8.2            MIT
email-validator                    2.3.0            The Unlicense
email_validator                    2.2.0            The Unlicense
exceptiongroup                     1.3.1            MIT
fastapi                            0.121.3          MIT
fastapi-cli                        0.0.20           MIT
fastapi-cloud-cli                  0.11.0           MIT
fastapi-utils                      0.8.0            MIT
fastar                             0.8.0            MIT
filelock                           3.29.1           MIT
filetype                           1.2.0            MIT
flash_attn                         2.8.0.post2      BSD
flatbuffers                        25.12.19         Apache
fonttools                          4.63.0           MIT
frozenlist                         1.8.0            Apache-2.0
fsspec                             2026.4.0         BSD-3-Clause
ftfy                               6.3.1            Apache-2.0
googleapis-common-protos           1.75.0           Apache
grpclib                            0.4.9            BSD
h11                                0.16.0           MIT
h2                                 4.3.0            MIT
hf-xet                             1.5.0            Apache-2.0
hf_transfer                        0.1.9            UNKNOWN
hpack                              4.1.0            MIT
httpcore                           1.0.9            BSD-3-Clause
httptools                          0.6.4            MIT
httpx                              0.28.1           BSD
huggingface_hub                    0.36.2           Apache
hyperframe                         6.1.0            MIT
idna                               3.10             BSD
imageio                            2.36.1           BSD
imageio-ffmpeg                     0.5.1            BSD
importlib_metadata                 8.5.0            Apache
iniconfig                          2.1.0            MIT
itsdangerous                       2.2.0            BSD
Jinja2                             3.1.6            BSD
jmespath                           1.1.0            MIT
joblib                             1.4.2            BSD
kiwisolver                         1.5.0            BSD
lazy_loader                        0.4              BSD
librosa                            0.10.2.post1     ISC
librt                              0.7.5            MIT
llvmlite                           0.47.0           BSD-2-Clause AND Apache-2.0 WITH LLVM-exception
loguru                             0.7.3            MIT
markdown-it-py                     3.0.0            MIT
MarkupSafe                         3.0.2            BSD
matplotlib                         3.10.9           Python Software Foundation
mdurl                              0.1.2            MIT
mediapipe                          0.10.35          Apache
ml_dtypes                          0.5.4            Apache-2.0
modal                              1.4.2            Apache-2.0
mpmath                             1.3.0            BSD
msgpack                            1.1.2            Apache-2.0
multidict                          6.7.1            Apache 2.0
mypy                               1.19.1           MIT
mypy-extensions                    1.0.0            MIT
networkx                           3.4.2            BSD
numba                              0.65.1           BSD
numpy                              2.3.3            BSD
nvidia-cublas-cu12                 12.8.3.14        Other/Proprietary
nvidia-cuda-cupti-cu12             12.8.57          Other/Proprietary
nvidia-cuda-nvrtc-cu12             12.8.61          Other/Proprietary
nvidia-cuda-runtime-cu12           12.8.57          Other/Proprietary
nvidia-cudnn-cu12                  9.7.1.26         Other/Proprietary
nvidia-cufft-cu12                  11.3.3.41        Other/Proprietary
nvidia-cufile-cu12                 1.13.0.11        Other/Proprietary
nvidia-curand-cu12                 10.3.9.55        Other/Proprietary
nvidia-cusolver-cu12               11.7.2.55        Other/Proprietary
nvidia-cusparse-cu12               12.5.7.53        Other/Proprietary
nvidia-cusparselt-cu12             0.6.3            NVIDIA Proprietary Software
nvidia-nccl-cu12                   2.26.2           Other/Proprietary
nvidia-nvjitlink-cu12              12.8.61          Other/Proprietary
nvidia-nvtx-cu12                   12.8.55          Other/Proprietary
onnx                               1.21.0           Apache-2.0
opencv-contrib-python              4.13.0.92        Apache
opencv-python                      4.13.0.92        Apache
opencv-python-headless             4.13.0.92        Apache
opentelemetry-api                  1.28.0           Apache
opentelemetry-distro               0.49b0           Apache-2.0
opentelemetry-exporter-otlp-proto-common 1.28.0           Apache
opentelemetry-exporter-otlp-proto-http 1.28.0           Apache
opentelemetry-instrumentation      0.49b0           Apache-2.0
opentelemetry-proto                1.28.0           Apache
opentelemetry-sdk                  1.28.0           Apache
opentelemetry-semantic-conventions 0.49b0           Apache
orjson                             3.11.9           MPL-2.0 AND
packaging                          24.2             Apache; BSD
pathspec                           0.12.1           Mozilla Public 2.0
pillow                             12.2.0           MIT-CMU
pip                                26.1.2           MIT
platformdirs                       4.10.0           MIT
pluggy                             1.5.0            MIT
polygraphy                         0.49.18          Apache 2.0
pooch                              1.9.0            BSD-3-Clause
propcache                          0.4.1            Apache
protobuf                           6.33.6           3-Clause BSD
psutil                             5.9.8            BSD
PyAudio                            0.2.14           MIT
pycparser                          3.0              BSD-3-Clause
pydantic                           2.11.3           MIT
pydantic-extra-types               2.11.1           MIT
pydantic-settings                  2.8.1            MIT
pydantic_core                      2.33.1           MIT
Pygments                           2.19.1           BSD
PyJWT                              2.12.0           MIT
pyloudnorm                         0.2.0            MIT
pynvml                             11.5.3           BSD
pyparsing                          3.3.2            MIT
PySDL2                             0.9.17           Public Domain; zlib/libpng
pytest                             8.3.5            MIT
pytest-asyncio                     0.26.0           Apache-2.0
pytest-cov                         4.1.0            MIT
pytest-httpx                       0.35.0           MIT
pytest-mock                        3.15.1           MIT
python-dateutil                    2.9.0.post0      BSD; Apache
python-dotenv                      1.1.0            BSD
python-json-logger                 3.3.0            BSD
python-multipart                   0.0.29           Apache-2.0
PyYAML                             6.0.2            MIT
regex                              2026.5.9         Apache-2.0 AND CNRI-Python
requests                           2.34.2           Apache
rich                               14.0.0           MIT
rich-toolkit                       0.18.1           MIT
rignore                            0.7.6            MIT
ruff                               0.11.4           MIT
s3transfer                         0.16.0           Apache
safetensors                        0.8.0rc1         Apache
scikit-image                       0.25.0           BSD
scikit-learn                       1.6.0            BSD
scipy                              1.15.0           BSD
sentencepiece                      0.2.1            UNKNOWN
sentry-sdk                         2.56.0           BSD
setuptools                         79.0.1           UNKNOWN
shellingham                        1.5.4            ISC
six                                1.17.0           MIT
sniffio                            1.3.1            MIT; Apache
sounddevice                        0.5.5            MIT
soundfile                          0.14.0           BSD
soupsieve                          2.8.4            MIT
soxr                               1.1.0            LGPL-2.1-or-later
starlette                          0.49.1           BSD-3-Clause
sympy                              1.14.0           BSD
synchronicity                      0.12.1           Apache
tenacity                           8.2.2            Apache
tensorrt                           10.16.1.11       Other/Proprietary
tensorrt_cu13                      10.16.1.11       Other/Proprietary
tensorrt_cu13_bindings             10.16.1.11       Other/Proprietary
tensorrt_cu13_libs                 10.16.1.11       Other/Proprietary
threadpoolctl                      3.5.0            BSD
tifffile                           2024.12.12       BSD
tokenizers                         0.22.2           Apache
toml                               0.10.2           MIT
tomli                              2.4.1            MIT
torch                              2.7.1+cu128      BSD
torchvision                        0.22.1+cu128     BSD
tqdm                               4.67.1           MIT; Mozilla Public 2.0
transformers                       4.57.3           Apache
triton                             3.3.1            MIT
typer                              0.15.2           MIT
types-certifi                      2021.10.8.3      Apache
types-colorama                     0.4.15           Apache
types-toml                         0.10.8.20240310  Apache
typing-inspect                     0.9.0            MIT
typing-inspection                  0.4.0            MIT
typing_extensions                  4.13.1           PSF-2.0
ujson                              5.12.1           BSD-3-Clause AND TCL
urllib3                            2.7.0            MIT
uvicorn                            0.40.0           BSD-3-Clause
uvloop                             0.21.0           Apache; MIT
watchfiles                         1.0.5            MIT
wcwidth                            0.8.1            MIT
websockets                         15.0.1           BSD
Werkzeug                           3.1.6            BSD-3-Clause
wrapt                              1.17.3           BSD
wsproto                            1.3.2            MIT
xformers                           0.0.31           BSD
xfuser                             0.4.5            UNKNOWN
yarl                               1.23.0           Apache-2.0
yunchang                           0.6.4            Apache
zipp                               4.1.0            MIT
```
