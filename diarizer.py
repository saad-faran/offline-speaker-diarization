#!/usr/bin/env python3
"""
diarizer.py — Offline speaker diarization engine.

Engine: pyannote `speaker-diarization-community-1`
        (neural VAD + segmentation + speaker embeddings + clustering).

Public API
----------
    load_pipeline(device=None, token=None, clustering_threshold=None)
    diarize_file(path, pipe=None, num_speakers=None,
                 min_change_dur=0.8, min_speaker_dur=0.0)  ->  (timeline, n_speakers)

`timeline` is a list of (start_sec, end_sec, "SPEAKER_XX") merged speaker-turn blocks.

Why this is robust (no fragmentation)
-------------------------------------
A neural diarizer detects real speaker boundaries and clusters embeddings with a
global view of the whole file — so a single expressive speaker is not split into
phantom IDs. Two extra safeguards handle real-world audio:
  * `clustering_threshold` (VBx) — merge more/less aggressively.
  * `min_speaker_dur`      — absorb tiny short-utterance clusters into the nearest
                             real speaker (kills phantom speakers from <1-2s blips).

Runs on GPU (CUDA / Apple MPS) or CPU — device is auto-selected.
"""
import os
import numpy as np
import torch
import soundfile as sf
from pyannote.audio import Pipeline

MODEL = "pyannote/speaker-diarization-community-1"


def _load_waveform(path):
    """Decode an audio file into an in-memory waveform dict for pyannote.

    Reading the file ourselves (via soundfile/libsndfile) and passing a
    {'waveform': (channel, time) tensor, 'sample_rate': int} dict means pyannote
    never invokes `torchcodec` — which is fragile on Windows (needs FFmpeg shared
    DLLs + a matching torch build). This makes the pipeline portable everywhere.

    Expects a soundfile-readable format (wav/flac/ogg). `verify.py` already converts
    any input to a 16 kHz mono wav via ffmpeg, so this always receives a wav.
    """
    audio, sr = sf.read(path, dtype="float32", always_2d=True)   # (time, channels)
    audio = audio.mean(axis=1)                                   # downmix to mono
    waveform = torch.from_numpy(audio).unsqueeze(0)             # (1, time)
    return {"waveform": waveform, "sample_rate": sr}
MIN_CHANGE_DUR = 0.8   # ignore turns shorter than this when building the timeline


def _cos(a, b):
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(a @ b)


def load_pipeline(device=None, token=None, clustering_threshold=None):
    """Load the pyannote pipeline once.

    device:  'cuda' | 'mps' | 'cpu'  (auto-selected when None: CUDA > MPS > CPU).
    token:   Hugging Face access token; falls back to the HF_TOKEN env var.
    clustering_threshold:
        VBx agglomerative threshold. None keeps pyannote's calibrated default (0.6).
        RAISE it (e.g. 0.85) to merge more aggressively when one expressive speaker is
        being over-split into several IDs; LOWER it if two distinct speakers are being
        merged into one. This is the main accuracy knob for the unknown-speaker-count case.
    """
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    token = token or os.environ.get("HF_TOKEN")
    pipe = Pipeline.from_pretrained(MODEL, token=token)
    pipe.to(torch.device(device))
    if clustering_threshold is not None:
        pipe.instantiate({"clustering": {"Fa": 0.07, "Fb": 0.8,
                                         "threshold": clustering_threshold},
                          "segmentation": {"min_duration_off": 0.0}})
    return pipe


def _merge(turns, gap=0.6, min_dur=0.0):
    """Merge consecutive same-speaker turns; drop sub-`min_dur` blips."""
    out = []
    for s, e, sid in sorted(turns):
        if out and out[-1][2] == sid and s - out[-1][1] <= gap:
            out[-1] = (out[-1][0], max(out[-1][1], e), sid)
        else:
            out.append([s, e, sid])
    return [(s, e, sid) for s, e, sid in out if e - s >= min_dur]


def _consolidate_small(turns, lab2emb, min_speaker_dur):
    """Merge low-support clusters (< `min_speaker_dur` s of total speech) into their
    nearest surviving cluster by centroid cosine similarity. Removes phantom speakers
    formed from noisy short-utterance embeddings, without loosening the global merge
    threshold (which would risk merging genuinely distinct speakers)."""
    if min_speaker_dur <= 0 or not lab2emb:
        return turns
    dur = {}
    for s, e, l in turns:
        dur[l] = dur.get(l, 0.0) + (e - s)
    survivors = [l for l in dur if dur[l] >= min_speaker_dur
                 and lab2emb.get(l) is not None and not np.isnan(lab2emb[l]).any()]
    if not survivors:
        return turns
    remap = {}
    for l in dur:
        if l in survivors:
            continue
        emb = lab2emb.get(l)
        if emb is None or np.isnan(emb).any():
            remap[l] = survivors[0]
            continue
        remap[l] = max(survivors, key=lambda sv: _cos(emb, lab2emb[sv]))
    return [(s, e, remap.get(l, l)) for s, e, l in turns]


def diarize_file(path, pipe=None, num_speakers=None, min_change_dur=MIN_CHANGE_DUR,
                 min_speaker_dur=0.0):
    """
    Diarize a whole audio file.

    Returns
    -------
    (timeline, n_speakers)
        timeline    : list of (start_sec, end_sec, "SPEAKER_XX") merged turn blocks
        n_speakers  : number of distinct speakers in the timeline

    Parameters
    ----------
    num_speakers    : pass e.g. 2 when you KNOW the count (most reliable for fixed formats).
    min_change_dur  : drop speaker turns shorter than this (seconds) from the timeline.
    min_speaker_dur : if > 0, merge clusters with less than this much total speech into the
                      nearest real speaker (recommended ~20 for conversational audio when the
                      speaker count is unknown).
    """
    pipe = pipe or load_pipeline()
    kw = {"num_speakers": num_speakers} if num_speakers else {}
    out = pipe(_load_waveform(path), **kw)   # in-memory audio -> no torchcodec dependency
    ann = out.exclusive_speaker_diarization        # one speaker per instant -> clean changes
    embs = out.speaker_embeddings
    labels = ann.labels()
    lab2emb = {l: (embs[i] if embs is not None and i < len(embs) else None)
               for i, l in enumerate(labels)}
    turns = [(t.start, t.end, l) for t, _, l in ann.itertracks(yield_label=True)]
    turns = _consolidate_small(turns, lab2emb, min_speaker_dur)
    timeline = _merge(turns, min_dur=min_change_dur)
    n = len(set(t[2] for t in timeline))
    return timeline, n


if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
    audio = sys.argv[1] if len(sys.argv) > 1 else None
    if not audio:
        print("usage: python diarizer.py <audio_file> [num_speakers]")
        raise SystemExit(1)
    nspk = int(sys.argv[2]) if len(sys.argv) > 2 else None
    tl, n = diarize_file(audio, num_speakers=nspk,
                         min_speaker_dur=0.0 if nspk else 20.0)
    print(f"\n{n} speaker(s):")
    for s, e, sp in tl:
        print(f"  {int(s//60):02d}:{int(s%60):02d}-{int(e//60):02d}:{int(e%60):02d}  {sp}")
