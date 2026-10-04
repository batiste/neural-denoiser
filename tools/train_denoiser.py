import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional

from denoiser_model import CHANNELS, DILATIONS, WaveformDenoiser, choose_device
from noise_engine import (
    SAMPLE_RATE,
    apply_record_bandwidth,
    decode_mono,
    discover_clean_sources,
    make_noise,
    rms,
)


CHUNK_SAMPLES = 8192
TRAIN_PROFILES = (
    "shellac-transfer",
    "shellac-transfer",
    "shellac-transfer",
    "shellac-transfer",
    "soft-surface",
    "fine-hiss",
)


def make_pair(audio, randomizer, segment_samples):
    start = randomizer.randrange(len(audio) - segment_samples + 1)
    source = audio[start : start + segment_samples]
    profile = randomizer.choice(TRAIN_PROFILES)
    target_cutoff = randomizer.uniform(4500, 6500)
    clean = apply_record_bandwidth(source, SAMPLE_RATE, target_cutoff)
    if profile == "shellac-transfer":
        input_cutoff = randomizer.uniform(2400, 4300)
        degraded = apply_record_bandwidth(source, SAMPLE_RATE, input_cutoff)
    else:
        degraded = source

    snr_db = randomizer.uniform(14, 26)
    noise_generator = np.random.default_rng(randomizer.randrange(0, 2**32))
    crackle_scale = randomizer.uniform(1.5, 3.0) if profile == "shellac-transfer" else 1.0
    noise = make_noise(
        noise_generator,
        len(clean),
        SAMPLE_RATE,
        snr_db,
        profile,
        crackle_scale=crackle_scale,
    )
    noise *= max(rms(degraded), 1e-8)
    noisy = degraded + noise
    peak = max(float(np.max(np.abs(clean))), float(np.max(np.abs(noisy))), 1e-12)
    gain = min(1.0, 0.98 / peak)
    return clean * gain, noisy * gain


def batch_from_audio(audio_tracks, randomizer, batch_size, segment_samples):
    clean_batch = []
    noisy_batch = []
    for _ in range(batch_size):
        track = randomizer.choice(audio_tracks)
        clean, noisy = make_pair(track, randomizer, segment_samples)
        clean_batch.append(clean)
        noisy_batch.append(noisy)
    return np.stack(noisy_batch), np.stack(clean_batch)


def loss_function(predicted, clean):
    waveform_loss = functional.l1_loss(predicted, clean)
    predicted_spec = torch.stft(
        predicted.squeeze(1), n_fft=512, hop_length=128, return_complex=True
    ).abs()
    clean_spec = torch.stft(
        clean.squeeze(1), n_fft=512, hop_length=128, return_complex=True
    ).abs()
    frequencies = torch.linspace(
        0, SAMPLE_RATE / 2, predicted_spec.shape[1], device=predicted.device
    )
    upper_mid_weight = 1 + 3 * ((frequencies - 1000) / 4000).clamp(0, 1)
    spectral_error = torch.abs(
        torch.log1p(predicted_spec * 100) - torch.log1p(clean_spec * 100)
    )
    spectral_loss = (spectral_error * upper_mid_weight[None, :, None]).mean()
    return 0.7 * waveform_loss + 0.3 * spectral_loss


def to_tensor_pair(noisy, clean, device):
    noisy_tensor = torch.from_numpy(noisy).unsqueeze(1).to(device)
    clean_tensor = torch.from_numpy(clean).unsqueeze(1).to(device)
    return noisy_tensor, clean_tensor


def evaluate(model, validation_audio, device, seed, segment_samples):
    randomizer = random.Random(seed)
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(12):
            noisy, clean = batch_from_audio(validation_audio, randomizer, 1, segment_samples)
            noisy_tensor, clean_tensor = to_tensor_pair(noisy, clean, device)
            losses.append(float(loss_function(model(noisy_tensor), clean_tensor).item()))
    return sum(losses) / len(losses)


def main():
    parser = argparse.ArgumentParser(
        description="Train a small denoiser with variable shellac bandwidth and synthetic noise."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/neural_model"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--steps-per-epoch", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--channels", type=int, default=CHANNELS)
    parser.add_argument("--dilations", type=int, nargs="+", default=list(DILATIONS))
    parser.add_argument("--seed", type=int, default=1925)
    parser.add_argument("--validation-track", type=Path)
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        help="clean music files; defaults to MP3s under clean/",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    sources = args.inputs or discover_clean_sources(root)
    sources = [path.resolve() for path in sources]
    if len(sources) < 2:
        parser.error("At least two clean tracks are required for track-level validation.")
    if args.epochs < 1 or args.steps_per_epoch < 1 or args.batch_size < 1:
        parser.error("Epochs, steps per epoch, and batch size must be positive.")

    validation_track = args.validation_track.resolve() if args.validation_track else sources[-1]
    if validation_track not in sources:
        parser.error("--validation-track must name one of the input tracks.")
    training_sources = [path for path in sources if path != validation_track]

    print("Decoding clean tracks at 22.05 kHz mono...")
    training_audio = [decode_mono(path, SAMPLE_RATE) for path in training_sources]
    validation_audio = [decode_mono(validation_track, SAMPLE_RATE)]
    if min(map(len, training_audio + validation_audio)) < CHUNK_SAMPLES:
        parser.error("Every track must contain at least one training chunk.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device()
    print(f"Training on {device}; {len(training_sources)} train tracks, 1 held-out validation track")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "denoiser.pt"
    history_path = output_dir / "training.csv"
    if args.channels < 4 or any(dilation < 1 for dilation in args.dilations):
        parser.error("Channels must be at least 4 and dilations must be positive.")
    model = WaveformDenoiser(args.channels, tuple(args.dilations)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005)
    randomizer = random.Random(args.seed)
    best_validation_loss = float("inf")

    with history_path.open("w", newline="", encoding="utf-8") as history_file:
        writer = csv.writer(history_file)
        writer.writerow(["epoch", "training_loss", "validation_loss"])
        for epoch in range(1, args.epochs + 1):
            model.train()
            training_losses = []
            for _ in range(args.steps_per_epoch):
                noisy, clean = batch_from_audio(
                    training_audio, randomizer, args.batch_size, CHUNK_SAMPLES
                )
                noisy_tensor, clean_tensor = to_tensor_pair(noisy, clean, device)
                optimizer.zero_grad(set_to_none=True)
                loss = loss_function(model(noisy_tensor), clean_tensor)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                training_losses.append(float(loss.item()))

            training_loss = sum(training_losses) / len(training_losses)
            validation_loss = evaluate(
                model, validation_audio, device, args.seed + epoch, CHUNK_SAMPLES
            )
            writer.writerow([epoch, f"{training_loss:.6f}", f"{validation_loss:.6f}"])
            history_file.flush()
            print(
                f"Epoch {epoch:02d}/{args.epochs}: train={training_loss:.5f}, "
                f"validation={validation_loss:.5f}"
            )

            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "sample_rate": SAMPLE_RATE,
                        "channels": args.channels,
                        "dilations": tuple(args.dilations),
                        "validation_track": validation_track.name,
                        "noise_profiles": TRAIN_PROFILES,
                        "epoch": epoch,
                        "validation_loss": validation_loss,
                    },
                    checkpoint_path,
                )

    print(f"Best checkpoint: {checkpoint_path} (validation loss {best_validation_loss:.5f})")
    print(f"Training history: {history_path}")


if __name__ == "__main__":
    main()