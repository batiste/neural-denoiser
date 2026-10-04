import argparse
import csv
import random
import subprocess
import wave
from pathlib import Path

import numpy as np


SAMPLE_RATE = 22050
TARGET_RECORDINGS = {"sound.mp3", "cusb_victor_19647_01_b32119_01d.mp3"}
NOISE_PROFILES = {
    "light-crackle": (500, 8000, 25, 180, 0.25, 5, 1.4),
    "soft-surface": (450, 6500, 18, 160, 0.35, 8, 1.8),
    "fine-hiss": (1200, 10000, 50, 220, 0.12, 2, 0.6),
    "shellac-transfer": (450, 6500, 25, 180, 0.12, 70, 1.0),
}


def discover_clean_sources(root):
    clean_directory = root / "clean"
    clean_sources = sorted(clean_directory.glob("*.mp3"))
    if clean_sources:
        return clean_sources
    return sorted(
        path for path in root.glob("*.mp3") if path.name not in TARGET_RECORDINGS
    )


def rms(signal):
    return float(np.sqrt(np.mean(np.square(signal, dtype=np.float64))))


def band_limited_noise(generator, sample_count, sample_rate, low_hz, high_hz):
    spectrum = np.fft.rfft(generator.standard_normal(sample_count))
    frequencies = np.fft.rfftfreq(sample_count, 1 / sample_rate)
    high_pass = frequencies**2 / (frequencies**2 + low_hz**2)
    low_pass = 1 / np.sqrt(1 + (frequencies / high_hz) ** 8)
    signal = np.fft.irfft(spectrum * high_pass * low_pass, n=sample_count)
    signal /= max(rms(signal), 1e-12)
    return signal.astype(np.float32)


def apply_record_bandwidth(signal, sample_rate, cutoff_hz):
    pad_size = min(len(signal) - 1, round(sample_rate * 0.1))
    padded = np.pad(signal, (pad_size, pad_size), mode="reflect")
    frequencies = np.fft.rfftfreq(len(padded), 1 / sample_rate)
    response = 1 / np.sqrt(1 + (frequencies / cutoff_hz) ** 8)
    filtered = np.fft.irfft(np.fft.rfft(padded) * response, n=len(padded))
    return filtered[pad_size : pad_size + len(signal)].astype(np.float32)


def crackle_noise(generator, sample_count, sample_rate, clicks_per_second, shellac=False):
    signal = np.zeros(sample_count, dtype=np.float32)
    click_count = generator.poisson(sample_count / sample_rate * clicks_per_second)
    positions = generator.integers(0, sample_count, size=click_count)

    for position in positions:
        length = int(generator.integers(1, max(2, round(sample_rate * 0.004))))
        end = min(sample_count, position + length)
        decay = np.exp(-np.arange(end - position) / max(1, length * 0.18))
        if shellac:
            level = generator.random()
            if level < 0.02:
                amplitude = generator.uniform(0.7, 1.0)
            elif level < 0.16:
                amplitude = generator.uniform(0.35, 0.55)
            else:
                amplitude = generator.uniform(0.12, 0.3)
        else:
            amplitude = generator.uniform(0.2, 1.0)
        amplitude *= generator.choice((-1, 1))
        signal[position:end] += amplitude * decay

    return signal


def make_noise(generator, sample_count, sample_rate, snr_db, profile, crackle_scale=1.0):
    hiss_low, hiss_high, rumble_low, rumble_high, rumble_gain, click_rate, click_gain = (
        NOISE_PROFILES[profile]
    )
    hiss = band_limited_noise(generator, sample_count, sample_rate, hiss_low, hiss_high)
    rumble = band_limited_noise(generator, sample_count, sample_rate, rumble_low, rumble_high)
    crackle = crackle_noise(
        generator,
        sample_count,
        sample_rate,
        clicks_per_second=click_rate,
        shellac=profile == "shellac-transfer",
    )

    desired_noise_rms = 10 ** (-snr_db / 20)
    continuous = hiss + rumble * rumble_gain
    continuous *= desired_noise_rms / max(rms(continuous), 1e-12)
    if profile == "shellac-transfer":
        noise = continuous + crackle * 22 * crackle_scale
    else:
        noise = continuous + crackle * click_gain
        noise *= desired_noise_rms / max(rms(noise), 1e-12)
    return noise.astype(np.float32)


def decode_mono(path, sample_rate):
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "pipe:1",
    ]
    result = subprocess.run(command, check=True, capture_output=True)
    return np.frombuffer(result.stdout, dtype="<f4").copy()


def write_wav(path, signal, sample_rate):
    pcm = np.round(np.clip(signal, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(pcm.tobytes())


def mix_at_snr(clean, generator, snr_db, sample_rate, profile="soft-surface"):
    if profile == "shellac-transfer":
        clean = apply_record_bandwidth(clean, sample_rate, cutoff_hz=3300)
    noise = make_noise(generator, len(clean), sample_rate, snr_db, profile)
    noise *= max(rms(clean), 1e-8)
    noisy = clean + noise
    peak = max(float(np.max(np.abs(clean))), float(np.max(np.abs(noisy))), 1e-12)
    gain = min(1.0, 0.98 / peak)
    return clean * gain, noisy * gain


def main():
    parser = argparse.ArgumentParser(
        description="Generate reproducible gramophone-noise audio pairs for denoising experiments."
    )
    parser.add_argument("inputs", nargs="*", type=Path, help="clean source audio files")
    parser.add_argument("--output-dir", type=Path, default=Path("noise_dataset"))
    parser.add_argument("--segments-per-track", type=int, default=16)
    parser.add_argument("--segment-seconds", type=float, default=8)
    parser.add_argument("--snr-min", type=float, default=0)
    parser.add_argument("--snr-max", type=float, default=18)
    parser.add_argument("--preview-snr", type=float, default=6)
    parser.add_argument(
        "--noise-profile", choices=NOISE_PROFILES, default="soft-surface"
    )
    parser.add_argument("--preview-only", action="store_true")
    parser.add_argument("--seed", type=int, default=1925)
    parser.add_argument("--validation-track", type=Path)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    sources = args.inputs or discover_clean_sources(root)
    sources = [path.resolve() for path in sources]

    if len(sources) < 2:
        parser.error("Provide at least two clean tracks for train/validation separation.")
    if args.segments_per_track < 1 or args.segment_seconds <= 0:
        parser.error("Segment count and duration must be positive.")
    if args.snr_min > args.snr_max:
        parser.error("--snr-min must not exceed --snr-max.")

    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        parser.error(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    validation_track = args.validation_track.resolve() if args.validation_track else sources[-1]
    if validation_track not in sources:
        parser.error("--validation-track must name one of the input files.")
    training_sources = [path for path in sources if path != validation_track]
    selected_sources = [validation_track] if args.preview_only else sources
    source_audio = {path: decode_mono(path, SAMPLE_RATE) for path in selected_sources}
    segment_samples = round(args.segment_seconds * SAMPLE_RATE)
    rng = random.Random(args.seed)
    metadata = []

    splits = () if args.preview_only else (
        ("train", training_sources),
        ("validation", [validation_track]),
    )
    for split, split_sources in splits:
        for subfolder in ("clean", "noisy"):
            (output_dir / split / subfolder).mkdir(parents=True, exist_ok=True)

        for source in split_sources:
            audio = source_audio[source]
            if len(audio) < segment_samples:
                parser.error(f"Source is shorter than one segment: {source}")

            for segment_index in range(args.segments_per_track):
                start = rng.randrange(len(audio) - segment_samples + 1)
                clean = audio[start : start + segment_samples]
                snr_db = rng.uniform(args.snr_min, args.snr_max)
                segment_seed = rng.randrange(0, 2**32)
                noise_rng = np.random.default_rng(segment_seed)
                clean_out, noisy_out = mix_at_snr(
                    clean, noise_rng, snr_db, SAMPLE_RATE, "soft-surface"
                )
                name = f"{source.stem}_{segment_index:03d}.wav"
                clean_path = Path(split) / "clean" / name
                noisy_path = Path(split) / "noisy" / name
                write_wav(output_dir / clean_path, clean_out, SAMPLE_RATE)
                write_wav(output_dir / noisy_path, noisy_out, SAMPLE_RATE)
                metadata.append(
                    [
                        split,
                        source.name,
                        str(clean_path),
                        str(noisy_path),
                        "soft-surface",
                        f"{snr_db:.2f}",
                        segment_seed,
                    ]
                )

    preview_audio = source_audio[validation_track]
    preview_samples = min(len(preview_audio), 30 * SAMPLE_RATE)
    preview_clean, preview_noisy = mix_at_snr(
        preview_audio[:preview_samples],
        np.random.default_rng(args.seed + 1),
        args.preview_snr,
        SAMPLE_RATE,
        args.noise_profile,
    )
    write_wav(output_dir / "preview_clean.wav", preview_clean, SAMPLE_RATE)
    write_wav(output_dir / "preview_noisy.wav", preview_noisy, SAMPLE_RATE)

    with (output_dir / "metadata.csv").open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            ["split", "source", "clean_path", "noisy_path", "noise_profile", "snr_db", "seed"]
        )
        writer.writerows(metadata)

    if args.preview_only:
        print(f"Created {args.noise_profile} preview at {args.preview_snr:g} dB SNR in {output_dir}")
    else:
        print(f"Created {len(metadata)} aligned clean/noisy pairs in {output_dir}")
    print(f"Held out for validation: {validation_track.name}")
    print(f"Listening comparison: {output_dir / 'preview_clean.wav'} and {output_dir / 'preview_noisy.wav'}")


if __name__ == "__main__":
    main()