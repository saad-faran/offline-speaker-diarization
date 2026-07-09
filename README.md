# Offline Speaker Diarization

Turn any recorded audio/video into a **"who spoke when"** timeline — reliably, with no
speaker fragmentation. Built on the open-source [`pyannote.audio`](https://github.com/pyannote/pyannote-audio)
neural diarizer (`speaker-diarization-community-1`).

- **Input:** an audio/video file (or a media URL).
- **Output:** number of speakers, a speaker-change timeline (printed + JSON), and a
  self-contained colored-timeline **HTML** you can open in a browser.
- **Runs on:** Windows / Linux / macOS — GPU (**NVIDIA CUDA** or Apple MPS) or CPU.
- 100% open-source, no paid APIs, no cloud.

> A one-page explainer of how the pipeline works is in
> [`docs/pipeline_overview.pdf`](docs/pipeline_overview.pdf).

---

## 1. Prerequisites

| Requirement | Notes |
|---|---|
| **Python 3.10+** | 3.10–3.12 recommended |
| **ffmpeg** | must be on your `PATH` (used to decode/convert audio) |
| **NVIDIA GPU + CUDA** | strongly recommended; ~10–30× faster than CPU |
| **Hugging Face account + token** | free; needed to download the model weights once |
| **yt-dlp** | optional — only if you pass a URL instead of a file |

**Install ffmpeg**
- Windows: `winget install Gyan.FFmpeg` (or `choco install ffmpeg`), then restart the terminal.
- Linux: `sudo apt install ffmpeg`
- macOS: `brew install ffmpeg`

---

## 2. Setup (any OS)

```bash
# 1) Clone
git clone https://github.com/saad-faran/offline-speaker-diarization.git
cd offline-speaker-diarization

# 2) Create a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate

# 3) Install PyTorch + torchaudio FIRST, TOGETHER, from ONE CUDA index.
#    Pick an index that has BOTH packages at the SAME version (see note below).
#    A widely-available matched pair that runs on any recent NVIDIA GPU:
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
#    CPU-only (any machine, slower):
#    pip install torch torchaudio

# 4) Install the rest
pip install -r requirements.txt
```

> **Important — torch and torchaudio must be the *same version*.** They're compiled
> together; mixing e.g. `torch 2.13` with `torchaudio 2.11` crashes with an
> `undefined symbol` error. Newer `torch` releases sometimes ship *before* a matching
> `torchaudio`, and very-new CUDA indexes (e.g. `cu132`) may have no `torchaudio` at all.
> If your first choice fails, use an index where both exist at the same version —
> `cu128` currently provides a matched `torch 2.11.0 + torchaudio 2.11.0`, and a CUDA 12.8
> build runs fine on newer GPUs (drivers are backward-compatible).
>
> Then verify the GPU is visible — this should print `True`:
> `python -c "import torch; print(torch.cuda.is_available())"`

---

## 3. Hugging Face token (one-time)

The model weights are gated (free — you just accept the terms):

1. Create a **read** token: <https://huggingface.co/settings/tokens>
2. Open this model page while logged in and click **“Agree / Accept”**:
   <https://huggingface.co/pyannote/speaker-diarization-community-1>
3. Give the token to the app — either:
   - copy `.env.example` to `.env` and paste the token, **or**
   - set an environment variable:
     - Windows (PowerShell): `setx HF_TOKEN "hf_xxx"` (reopen the terminal)
     - Linux / macOS: `export HF_TOKEN="hf_xxx"`

The weights (~1–2 GB) download **once** and are cached locally; runs after that are offline.

---

## 4. Usage

### One-command tool (`verify.py`)

```bash
# Known speaker count (most reliable when you know it):
python verify.py path/to/interview.mp3 --speakers 2

# Unknown count (auto-detect):
python verify.py path/to/podcast.wav

# Straight from a URL (needs yt-dlp):
python verify.py "https://www.youtube.com/watch?v=<ID>" --speakers 2
```

It prints a summary, writes `verify_result.json`, and opens `timeline.html`:

```
======================================================
  RESULT: 2 speaker(s)  |  7 speaker changes
======================================================
  #  1  00:00-00:31  (31.4s)  SPEAKER_01
  #  2  00:31-00:53  (21.7s)  SPEAKER_00
  ...
```

### As a library (`diarizer.py`)

```python
from diarizer import load_pipeline, diarize_file

pipe = load_pipeline()                       # auto-selects CUDA / MPS / CPU
timeline, n = diarize_file("meeting.wav", num_speakers=None, min_speaker_dur=20.0)
# timeline = [(start_sec, end_sec, "SPEAKER_XX"), ...]
print(n, "speakers")
```

---

## 5. Getting clean results (tuning)

The engine rarely fragments, but real audio varies. Two knobs cover almost everything:

| Situation | Fix |
|---|---|
| **You know the number of speakers** | `--speakers N` — best accuracy, removes guesswork |
| One expressive speaker split into extra IDs | raise `--threshold` (e.g. `0.85` → `0.9`) |
| Two similar voices merged into one | lower `--threshold` (e.g. `0.85` → `0.7`) |
| Tiny phantom speakers from short interjections | raise `--min-speaker-dur` (e.g. `20` → `30`) |

There is no single value that is perfect for every video (a known trade-off of unknown-count
diarization) — when the format is fixed, prefer `--speakers N`.

---

## 6. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `torchcodec is not available / Could not load libtorchcodec` (esp. **Windows**) | Harmless — this project reads audio in-memory and does **not** use torchcodec. Make sure you're on the latest code (`git pull`); the warning at import can be ignored. |
| `Could not download ... 401/403` | HF token missing, or you didn't accept the model terms (step 3) |
| `torch.cuda.is_available()` is `False` | you installed the CPU wheel — reinstall the CUDA build (step 2) |
| `undefined symbol` / import crash, or `torchaudio 2.x+cpu` alongside `torch 2.y+cuXXX` | **version/build mismatch** — `torch` and `torchaudio` must share the *exact same version* from the *same* CUDA index. Reinstall: `pip uninstall -y torchaudio` then `pip install torchaudio==<your-torch-version> --index-url https://download.pytorch.org/whl/cuXXX` |
| `'ffmpeg' not found on PATH` | install ffmpeg and reopen the terminal |
| `'yt-dlp' not found` | only needed for URL input: `pip install yt-dlp` |
| Very slow | you're on CPU — use an NVIDIA GPU / CUDA build |

---

## 7. How it works (short version)

`Voice Activity Detection → Segmentation → Speaker embeddings → Clustering → Timeline.`
A neural model detects real speaker boundaries and clusters voice embeddings with a global
view of the whole file, so a single expressive speaker isn't split into phantom IDs. Two
safeguards clean up real-world audio: an adjustable clustering threshold, and merging of tiny
low-support clusters into the nearest real speaker. Full picture:
[`docs/pipeline_overview.pdf`](docs/pipeline_overview.pdf).

## License

MIT — see [LICENSE](LICENSE). Model weights are governed by their own licenses on Hugging Face.
