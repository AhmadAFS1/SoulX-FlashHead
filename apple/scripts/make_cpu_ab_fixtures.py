"""Rebuild the CPU A/B audio fixtures (16 kHz mono PCM16) with fixed hashes.

Sources: the SoulX FlashHead Hugging Face Space example podcast and the
hf-internal-testing LibriSpeech dummy validation parquet (speaker 1272).
Calibration clips are only used to fit the FFN reduction; evaluation clips
are held out from calibration. Run from the repository root:
    .venv-coreml/bin/python apple/scripts/make_cpu_ab_fixtures.py
"""
import hashlib
import io
from pathlib import Path
import urllib.request

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

OUT = Path('artifacts/cpu-validation/inputs')
SOURCES = {
    'podcast_sichuan_16k.wav': ('https://huggingface.co/spaces/Soul-AILab/SoulX-FlashHead/resolve/main/examples/podcast_sichuan_16k.wav',
                                '2eab639d2104b84292945a0ca902b12794163480b7f1bbefc56bc5400f6172f9'),
    'librispeech_dummy.parquet': ('https://huggingface.co/datasets/hf-internal-testing/librispeech_asr_dummy/resolve/main/clean/validation-00000-of-00001.parquet',
                                  '4e69a06fa5edc90921e5e7e39a7084881f8b3ed9c805c574f4f39c6fde27c603'),
}
EXPECTED = {
    'cal_sichuan_0_13s.wav': 'ded4834f2682ccb03b4f365b2150a6da30ef6552835add53deb3e7128b6322ef',
    'cal_english_utt4_13s.wav': '44f097030613655a05b1f77c97bf8f4a4ad411bb241261faaf6a02ab3742be4d',
    'eval_english_plosive.wav': '7294437d2a12d003d6fb01b7c2eabdddb98ea9a3dd87bf1f5cb1ea76066738ef',
    'eval_sichuan_40s.wav': 'd0dc590066ac748dfcf84d350b1481f1d92e0e48ce810eddf36d92022ebe3b6b',
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, (url, digest) in SOURCES.items():
        path = OUT / name
        if not path.exists():
            urllib.request.urlretrieve(url, path)
        if sha(path) != digest:
            raise ValueError(f'{path} hash mismatch')
    pod, rate = sf.read(OUT / 'podcast_sichuan_16k.wav', dtype='float32')
    assert rate == 16000
    table = pq.read_table(OUT / 'librispeech_dummy.parquet').to_pydict()

    def utterance(i):
        audio, r = sf.read(io.BytesIO(table['audio'][i]['bytes']), dtype='float32')
        assert r == 16000
        return audio

    window = 28 * 640
    clips = {
        'cal_sichuan_0_13s.wav': pod[:12 * window],
        'cal_english_utt4_13s.wav': utterance(4)[:12 * window],
        # "KALIKO WENT TO THE BIG GONG AND POUNDED ON IT ..." + 0.5 s silence + "HE DOESN'T WORK AT ALL"
        'eval_english_plosive.wav': np.concatenate([utterance(38), np.zeros(8000, np.float32), utterance(24)])[:int(9.48 * 16000)],
        'eval_sichuan_40s.wav': pod[40 * 16000:int(49.48 * 16000)],
    }
    for name, audio in clips.items():
        sf.write(OUT / name, audio.astype(np.float32), 16000, subtype='PCM_16')
        status = 'ok' if sha(OUT / name) == EXPECTED[name] else 'HASH DIFFERS (check soundfile/libsndfile version)'
        print(f'{name}: {len(audio) / 16000:.2f} s {status}')


if __name__ == '__main__':
    main()
