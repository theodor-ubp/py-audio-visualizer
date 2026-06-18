# Unbound Planet Audio Visualizer - Cross-platform GUI (Windows / macOS / Linux)

Turn any audio file into clean, high-resolution **visuals** + a **detailed TXT report** in one click.  
Built for creators who want fast, offline analysis with a minimal, no-console UI.

[![Platform](https://img.shields.io/badge/platform-win%20|%20mac%20|%20linux-success)](#downloads)
[![GUI](https://img.shields.io/badge/UI-Tkinter-blue)](#features)
[![Python](https://img.shields.io/badge/python-3.10%2B-blueviolet)](#build-from-source)
[![FFmpeg](https://img.shields.io/badge/FFmpeg-bundled-success)](#ffmpeg--codecs)
[![Nova · Unbound Planet](https://img.shields.io/badge/Nova-Unbound%20Planet-0f6)](https://unboundplanet.com/nova)

---

## Overview

**UP Audio Visualizer** converts audio into:
- **Waveform** (PNG)
- **Spectrogram (Linear Hz)** (PNG)
- **Spectrogram (Log scale)** (PNG)
- **Dynamics over time (RMS)** (PNG)
- **TXT report** with: tempo (≈ BPM), key *(approx.)*, average RMS (dB), spectral centroid, rolloff (85%), zero-crossing rate, crest factor, peak amplitude, plus container/codec metadata (codec, bitrate, channels, sample rate, duration, tags, size), **lyrics** if embedded, and integrated loudness (LUFS) via `ebur128`.

All processing runs **locally**. No internet needed.  
Heavy libraries (`numpy`, `librosa`, `matplotlib`) are **lazy-loaded** only when you click **Create** for fast app startup.

---

## Features

- **Browse…** picker + **Open Folder** shortcut  
- **Lazy imports** for snappy launch  
- **Animated progress bar** (no terminal window)  
- **300 DPI** PNG outputs, saved **next to your source** (existing files auto-overwrite)  
- Sensible defaults; no configuration required  

**Supported formats:** `wav, flac, mp3, ogg, m4a/aac, aiff/aif/aifc`  
(Compressed formats rely on **FFmpeg**.)

---

## Downloads

Grab the latest release from **[GitHub Releases](https://github.com/theodor-ubp/py-audio-visualizer/releases)**:

- **Windows (x64)** — Portable `.exe`  
- **macOS (ARM64 only)** — `.app` inside `.dmg`  
- **Linux (x64)** — `.deb` installer (adds app drawer shortcut, FFmpeg bundled)

---

## Quick start

1. **Download** your platform build and launch the app.  
2. Click **Browse…** and choose an audio file.  
   **Supported formats:** WAV, FLAC, MP3, OGG, M4A, AAC, AIFF (AIF/AIFC)  
   **Not supported:** DRM-protected files (M4P, Audible AA/AAX), streaming containers, proprietary codecs  
3. Click **Create**.  
4. Find the generated **PNGs** and **TXT** next to your source file.  

> On macOS Gatekeeper: Right-click → **Open** (first run) if the system blocks unidentified apps.

---

## Output files

For an input file `MyTrack.flac`, you’ll get:

MyTrack_waveform.png
MyTrack_spectrogram_linear.png
MyTrack_spectrogram_log.png
MyTrack_dynamics.png
MyTrack.txt


The **TXT** summarizes container/codec metadata (via `ffprobe`) and analysis metrics (via `librosa`), and appends **lyrics** if the tag exists.

---

## FFmpeg & Codecs

This app needs **FFmpeg** (`ffmpeg` and `ffprobe`) to decode/inspect compressed formats.

- **Bundled** (all releases ship with FFmpeg binaries), or  
- **System install**: If you lose or delete the bundled execs, ensure `ffmpeg` and `ffprobe` resolve on your system `PATH`.

**Get FFmpeg:** <https://ffmpeg.org/> (LGPL v2.1). See **Licenses** below.

---

## System requirements

- **OS:**  
  - Windows 10/11  
  - macOS 12+ (Apple Silicon)  
  - Linux x64 (Ubuntu 22.04+ tested)

- **CPU/RAM:** Any modern CPU; ≥ 2 GB RAM recommended for large files/spectrograms  
- **Storage:** PNGs at 300 DPI can be large; ensure free disk space  

> Performance tips:  
> - First run may be slower while Matplotlib caches fonts; we pre-set `MPLCONFIGDIR` to a writable cache for faster subsequent runs.  
> - On Windows, HiDPI support is enabled.  

---

## Troubleshooting

- **“FFmpeg Missing” dialog**  
  Ensure `ffmpeg` + `ffprobe` are either bundled in the release folder or installed system-wide.  

- **MP3 won’t load / tempo seems off**  
  Ensure FFmpeg is available. Tempo and key are **estimates**; highly percussive or atypical material can confuse beat tracking or tonal analysis.  

- **First run is slow**  
  Matplotlib builds a font cache the first time. Subsequent runs are faster.  

- **macOS blocked app**  
  Right-click → **Open** to run once. Alternatively, adjust System Settings → Privacy & Security.  
