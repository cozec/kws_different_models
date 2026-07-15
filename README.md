# Keyword Spotting (KWS) — Three Reproduction Experiments

Reproductions of three open-source keyword-spotting projects on a modern
Apple-Silicon Mac (arm64, Python 3.12, PyTorch 2.13 / TF 2.21 / Keras 3.15).
All three repos were written in the 2020–2021 CUDA/Linux era and none ran
as-published; each needed targeted surgery, documented below.

## Scoreboard

| # | Repo | Approach | Paper | Dataset | Result |
|---|------|----------|-------|---------|--------|
| 1 | [Wav2Keyword](Wav2Keyword/) | Wav2Vec 2.0 base fine-tune + conv head | Wav2KWS (Seo et al., IEEE Access 2021) | Google Speech Commands v0.01, 22 classes | **97.23% in 3 epochs** (paper: 97.9% @ 100 epochs) + live streaming demo |
| 2 | [Spoken-Keyword-Spotting](Spoken-Keyword-Spotting/) | CNN embeddings → PCA → One-Class SVM ("marvin" hotword) | — | Google Speech Commands v0.01 | 99.6% acc, F1 0.88 (true test set) |
| 3 | [KWS_pytorch](KWS_pytorch/) | Small CNN on STFT + **filler class** | Deep KWS (Chen, Parada & Heigold, ICASSP 2014) | MobvoiHotwords (OpenSLR-87, ~6% subset) | 96.3% acc; FA 0.073% @ FR 19.3% (τ=0.9) |

Candidate survey that led to these picks: [KWS_GITHUB_PROJECTS.md](KWS_GITHUB_PROJECTS.md).

---

## 1. Wav2Keyword (qute012 / dobby-seo) — Wav2KWS

**Architecture** — pretrained Wav2Vec 2.0 base (94.4M params: 7-layer conv
front-end + 12-layer transformer, LibriSpeech 960 h) + 2.2M-param conv decoder
whose first layer (kernel 25, dilation 2) has receptive field exactly 49 = the
transformer's output length, collapsing 1 s of audio to a single vector.
Asymmetric fine-tuning: encoder lr 1e-5, head lr 5e-4.

**📐 Architecture diagram:** [`Wav2Keyword/architecture.html`](Wav2Keyword/architecture.html)
— color encodes learning rate (blue = pretrained encoder @ 1e-5, amber = head
from scratch @ 5e-4), tensor shapes on every connector, light/dark themes.
GitHub doesn't render HTML in the file view; open it locally or
[view it rendered](https://htmlpreview.github.io/?https://github.com/cozec/kws_different_models/blob/main/Wav2Keyword/architecture.html).

**Results** (Speech Commands v0.01, 22 classes: 20 words + unknown + silence;
43,173 train / 5,880 test; MPS, ~70 min):

| Epoch | train acc | test acc |
|---:|---:|---:|
| 1 | 69.95% | 96.72% |
| 2 | 81.93% | 96.82% |
| 3 | 83.13% | **97.23%** |

Paper claims 97.9% (V1) after 100 epochs; 3 epochs lands within 0.7 pt —
the pretrained representation does the heavy lifting. (Train < test because
training clips get heavy augmentation; test clips are clean.)

**Fixes required**
- Repo is a frozen py3.6-era fairseq fork, unbuildable on Python 3.12/arm64;
  script also imports a `speech_commands` module missing from the repo →
  ported to `torchaudio.pipelines.WAV2VEC2_BASE` (the same `wav2vec_small.pt`
  checkpoint), zero fairseq
- **Bug:** `Softmax` applied before `CrossEntropyLoss` in training (double
  softmax) → model returns raw logits
- CUDA → MPS; dataset restructured from v0.01 into `training|testing/<class>`
  symlinks with materialized `unknown` (10 non-command words, subsampled to
  10%) and `silence` (1 s background-noise cuts) — `build_dataset.py`

**Run:**
```bash
cd Wav2Keyword
python build_dataset.py          # symlink layout from Speech Commands v0.01
python wav2keyword_local.py      # 3-epoch fine-tune on MPS
```

### Streaming demo (`stream_demo.py`)

Sliding 1 s window over a ring buffer, re-scored every 100 ms (8.7 ms/inference
on MPS — real-time factor 0.009), posterior smoothing over 5 windows, threshold
τ=0.85, 1 s refractory. `unknown`/`silence` are rejection classes and never
fire.

- `python stream_demo.py sim` — 18 s stream built from unseen test clips:
  **8/8 keywords detected, 0 false alarms**
- `python stream_demo.py mic [-v]` — live microphone; `-v` prints the model's
  top hypothesis (including unknown/silence) whenever it changes

---

## 2. Spoken-Keyword-Spotting (vineeths96)

**Pipeline** — 31-class Keras CNN on log-mel filterbanks (994k params) → 256-d
embedding layer → PCA(32, whitened) → One-Class SVM tuned by Bayesian
optimization → binary "marvin" wake-word detector.

**Full retrain results** (Speech Commands v0.01; CNN: 25 epochs, 94.5% val acc):

| Split | Accuracy | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|
| Validation | 0.9969 | 0.9728 | 0.8938 | 0.9316 | 0.9309 |
| Test (9,916 files) | 0.9962 | 0.9026 | 0.8580 | 0.8797 | 0.8781 |

The README's 97.7% recall headline is a *validation* number; a stale-cache bug
(below) made the original "test" silently re-evaluate validation data. The
table above is measured on the actual test set.

**Fixes required**
- TF 2.2 / Python 3.8 Intel pins → modern TF 2.21 / Keras 3.15 stack (Python 3.12 venv)
- Keras 3: restore static shapes after `tf.py_function`; rebuild feature
  extractor via fresh `Input` (no `.input` on loaded Sequential); drop
  `use_multiprocessing`; sklearn 1.x import path
- sklearn 1.9 rejects non-finite libSVM coefficients → whitened PCA +
  penalize-failed-fits in the tuning objective
- **Bug:** saved the untuned scratch SVM instead of the best-params one
- **Bug:** test reused the validation tf.data cache file (`kws_val_cache`)

**Run:** `cd Spoken-Keyword-Spotting/src && ../.venv/bin/python run_all.py`
(full retrain; auto-downloads data) or `run_eval.py` (SVM + test only).

---

## 3. KWS_pytorch (hongfeixue) + Deep KWS filler class

**Task** — MobvoiHotwords wake-words ("Hi Xiaowen" / "Nihao Wenwen"). Trained
on a streamed 3 GB / 30,225-wav subset of the 17 GB corpus (manifests filtered
to extracted files: train 4,594 / dev 756 / test 2,269 positives).

**Stage 1 — repo as published (2-class):** 96.78% test accuracy, matching the
README's CNN row (95.17%). But with no negative class, *any* input — real
non-keyword speech, silence, even white noise — is forced into one of the two
hotwords, often at >90% confidence.

**Stage 2 — filler class per the Deep KWS paper** (labels: 0=filler, 1=HIW,
2=NHWW; negatives merged 3:1 into training; keyword fires only if its
posterior beats a threshold τ, filler excluded from the decision as in the
paper's Eq. 3):

| true \ pred | filler | HIW | NHWW |
|---|---:|---:|---:|
| filler (5,499) | 5,466 | 2 | 31 |
| HIW (1,144) | 69 | 907 | 168 |
| NHWW (1,125) | 10 | 5 | 1,110 |

| τ | FA rate | FR rate |
|---:|---:|---:|
| 0.30 | 0.891% | 10.36% |
| 0.50 | 0.600% | 11.19% |
| 0.90 | 0.073% | 19.30% |

After the fix, the same negative clips / silence / white noise all route to
filler at ~1.000 confidence.

**Fixes required** (the repo is two mismatched experiment halves)
- Committed loader emitted (98, 1640) Deep-KWS stacked features that no
  committed model accepts → rewrote to the STFT front-end (161×101) the models
  were written for
- Unconditional `.cuda()` calls, hardcoded `/home/disk1/...` path, removed
  `Variable(volatile=True)` API — which also **disabled gradients in the
  training loop** (the original never learned)
- `nll_loss` on raw logits → added `log_softmax`; `out_dim=6` → actual class
  count; torchcodec-less audio loading via `soundfile`

**Run:** `cd KWS_pytorch && .venv/bin/python run_local.py` (2-class) or
`run_filler.py` (3-class + FA/FR sweep). Data: stream a subset of
[OpenSLR-87](https://www.openslr.org/87) via `stream_extract.sh`, manifests
auto-filtered.

---

## Environment

- macOS / Apple Silicon (arm64), Python 3.12 (brew) — system Python 3.14 has
  no TF/torch wheels
- Wav2Keyword & KWS_pytorch: shared `.venv` in `KWS_pytorch/` — PyTorch 2.13
  (MPS), torchaudio 2.11, soundfile, sounddevice
- Spoken-Keyword-Spotting: own `.venv` — TensorFlow 2.21, Keras 3.15,
  scikit-learn 1.9, scikit-optimize, python_speech_features
- Not committed (see `.gitignore`s): datasets (`input/`, `data/`), venvs,
  checkpoints, logs, retrained binaries
