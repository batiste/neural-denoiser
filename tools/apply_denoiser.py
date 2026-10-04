import argparse
from pathlib import Path

import torch

from denoiser_model import WaveformDenoiser, choose_device
from noise_engine import SAMPLE_RATE, decode_mono, write_wav


CHUNK_SAMPLES = 8192
CONTEXT_SAMPLES = 256


def denoise_audio(model, audio, device):
    output = torch.empty(len(audio), dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        for core_start in range(0, len(audio), CHUNK_SAMPLES):
            core_end = min(len(audio), core_start + CHUNK_SAMPLES)
            input_start = max(0, core_start - CONTEXT_SAMPLES)
            input_end = min(len(audio), core_end + CONTEXT_SAMPLES)
            chunk = torch.from_numpy(audio[input_start:input_end]).reshape(1, 1, -1).to(device)
            cleaned = model(chunk).squeeze().cpu()
            local_start = core_start - input_start
            local_end = local_start + core_end - core_start
            output[core_start:core_end] = cleaned[local_start:local_end]
    return output.numpy()


def main():
    parser = argparse.ArgumentParser(description="Apply a trained waveform denoiser to audio.")
    parser.add_argument("--input", type=Path, default=Path("sound.mp3"))
    parser.add_argument("--model", type=Path, default=Path("outputs/neural_model/denoiser.pt"))
    parser.add_argument("--output", type=Path, default=Path("outputs/neural_model/sound_denoised.wav"))
    parser.add_argument("--gain-db", type=float, default=4.5)
    parser.add_argument("--peak-ceiling-dbfs", type=float, default=-1.0)
    args = parser.parse_args()

    device = choose_device()
    checkpoint = torch.load(args.model, map_location=device, weights_only=True)
    model = WaveformDenoiser(
        checkpoint.get("channels", 16),
        tuple(checkpoint.get("dilations", (1, 2, 4, 8, 4, 2))),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    audio = decode_mono(args.input.resolve(), SAMPLE_RATE)
    cleaned = denoise_audio(model, audio, device)
    cleaned *= 10 ** (args.gain_db / 20)
    peak_ceiling = 10 ** (args.peak_ceiling_dbfs / 20)
    peak = float(abs(cleaned).max())
    if peak > peak_ceiling:
        cleaned *= peak_ceiling / peak
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_wav(args.output, cleaned, SAMPLE_RATE)
    print(f"Saved denoised audio to {args.output.resolve()}")
    print(f"Applied {args.gain_db:g} dB gain with a {args.peak_ceiling_dbfs:g} dBFS peak ceiling")
    print(f"Model epoch {checkpoint['epoch']}; validation loss {checkpoint['validation_loss']:.5f}")


if __name__ == "__main__":
    main()