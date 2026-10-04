# Neural Audio Restoration

A small audio-restoration proof of concept for reducing shellac-record crackle and surface noise while retaining some of the recording's vintage character. It combines synthetic noise generation with a compact waveform denoiser trained on clean music.

The current best listening candidate is neural model v5. The models are experimental: the training corpus contains only six clean jazz tracks, and synthetic noise cannot fully represent every real record transfer. Compare results by listening, especially for musical detail that may be mistaken for noise.

## Project Contents

- `clean/`: clean jazz tracks used to create training examples.
- `cusb_victor_19647_01_b32119_01d.mp3`: the original historical recording restoration target; it is not included in the clean training set.
- `tools/noise_engine.py`: creates synthetic hiss, rumble, clicks, and shellac-style crackle, and writes paired clean/noisy examples or preview audio.
- `tools/denoiser_model.py`: the configurable 1D convolutional waveform network.
- `tools/train_denoiser.py`: trains the network with noise generated on the fly and a held-out track for validation.
- `tools/apply_denoiser.py`: applies a saved checkpoint to an audio file.
- `outputs/`: generated previews, checkpoints, training logs, and restored audio.

The model operates on mono audio at 22.05 kHz. V5 uses 32 channels and dilated convolution blocks with dilation rates `1, 2, 4, 8, 16, 8, 4, 2, 1`. Inference adds 4.5 dB gain by default and limits peaks to -1 dBFS; both settings are configurable.

## Public-Domain Recording Sources

The [UCSB Library list of 1925 recordings entering the public domain](https://www.library.ucsb.edu/1925-recordings-digitized-ucsb-entering-public-domain-january-1-2026) points to recordings that entered the U.S. public domain on January 1, 2026, with catalog entries in the [Discography of American Historical Recordings](https://adp.library.ucsb.edu/). The list spans many performers, styles, and languages. These are historical transfers, often with their own noise and limited bandwidth, so they are restoration inputs rather than clean targets for paired denoiser training.

They could still broaden the project's real-world evaluation set and help characterize authentic crackle and transfer artifacts. Use them as training targets only when a genuinely clean counterpart of the same performance is available; otherwise the model could learn to treat historically characteristic sound as removable noise.

Before downloading or redistributing a particular digital file, check that item's rights and the source site's terms. Public-domain status can depend on jurisdiction, and the status of a historical recording is distinct from terms applying to a library's digitized file or website.

## Setup

Requirements: Python 3.10 or newer, FFmpeg, and (on macOS) Homebrew if you need to install FFmpeg.

```sh
brew install ffmpeg
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

The scripts prefer Apple's MPS backend when available, then CUDA, then CPU.

## Listen To A Noise Preview

Create a 30-second shellac-style noisy excerpt and its clean reference:

```sh
.venv/bin/python tools/noise_engine.py \
  --preview-only \
  --noise-profile shellac-transfer \
  --preview-snr 18 \
  --output-dir outputs/previews/my-shellac-preview
```

The output directory contains `preview_noisy.wav` and `preview_clean.wav`. The SNR value controls the continuous hiss/rumble bed; the shellac profile also adds separate crackle impulses.

## Train A Model

The trainer discovers MP3 files under `clean/` by default. It holds one track out for validation and synthesizes a fresh noisy version of each sampled segment during training, so training pairs are not written to disk.

Train a v5-sized model into a new output directory:

```sh
.venv/bin/python tools/train_denoiser.py \
  --epochs 10 \
  --steps-per-epoch 50 \
  --batch-size 8 \
  --channels 32 \
  --dilations 1 2 4 8 16 8 4 2 1 \
  --output-dir outputs/neural_model_custom
```

The directory will contain `denoiser.pt` (the best validation checkpoint) and `training.csv` (per-epoch losses). Choose a new output directory or intentionally overwrite an existing run.

## Restore Audio

Apply the existing v5 checkpoint to the original recording:

```sh
.venv/bin/python tools/apply_denoiser.py \
  --input sound.mp3 \
  --model outputs/neural_model_v5/denoiser.pt \
  --output outputs/neural_model_v5/sound_denoised_v5.wav
```

Restore the Victor recording with v5:

```sh
.venv/bin/python tools/apply_denoiser.py \
  --input cusb_victor_19647_01_b32119_01d.mp3 \
  --model outputs/neural_model_v5/denoiser.pt \
  --output outputs/neural_model_v5/cusb_victor_19647_01_b32119_01d_denoised.wav
```

The output is a 22.05 kHz mono WAV. To change loudness gain or peak protection, pass `--gain-db` or `--peak-ceiling-dbfs`, for example `--gain-db 2 --peak-ceiling-dbfs -1`.

## Reproducibility And Limitations

Training uses seed `1925` by default, but noise is sampled during training. Results can vary across hardware and library versions. `outputs/noise_dataset/` contains an earlier, explicitly saved dataset made with the softer `soft-surface` profile; it is not the shellac training set used by the later neural models.

This project is a listening-oriented experiment, not a validated archival-restoration system. Keep the original recording and compare restored audio against it before choosing a final master.
