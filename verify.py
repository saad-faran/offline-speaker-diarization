#!/usr/bin/env python3
"""
verify.py — one-command diarization + visual timeline.

    python verify.py path/to/audio.(wav|mp3|m4a|...)
    python verify.py "https://www.youtube.com/watch?v=<ID>"      # needs yt-dlp
    python verify.py <input> --speakers 2                        # force known count
    python verify.py <input> --threshold 0.85 --min-speaker-dur 20   # tune auto-count

Diarizes with pyannote community-1, prints a speaker/change summary, and writes a
self-contained colored-timeline HTML (opens in your browser) plus a JSON result.

Requirements on PATH: ffmpeg (always), yt-dlp (only for URL input).
Works on Windows / Linux / macOS, on GPU (CUDA/MPS) or CPU.
"""
import os, sys, glob, shutil, subprocess, json, argparse, webbrowser, pathlib
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
from diarizer import load_pipeline, diarize_file

PALETTE = ["#3498DB", "#2ECC71", "#E74C3C", "#F39C12", "#9B59B6",
           "#1ABC9C", "#E91E63", "#FF9800", "#34495E", "#16A085"]
OUT_DIR = os.getcwd()


def _tool(name):
    exe = shutil.which(name)
    if not exe:
        sys.exit(f"ERROR: '{name}' not found on PATH. Please install it (see README).")
    return exe


def fetch_audio(src):
    """Return a 16 kHz mono wav path. Downloads (URL) and/or converts as needed."""
    ffmpeg = _tool("ffmpeg")
    if os.path.exists(src):
        if src.lower().endswith("_16k.wav"):
            return src
        out = os.path.join(OUT_DIR, pathlib.Path(src).stem + "_16k.wav")
        subprocess.run([ffmpeg, "-y", "-i", src, "-ac", "1", "-ar", "16000", out],
                       check=True, capture_output=True)
        return out
    # treat as URL -> download bestaudio, then convert
    ytdlp = _tool("yt-dlp")
    # clear any previous download so a new URL is always fetched fresh
    # (yt-dlp skips downloading if the target filename already exists)
    for f in glob.glob(os.path.join(OUT_DIR, "_dl*")):
        try:
            os.remove(f)
        except OSError:
            pass
    raw = os.path.join(OUT_DIR, "_dl.wav")
    subprocess.run([ytdlp, "-f", "bestaudio", "-x", "--audio-format", "wav",
                    "--force-overwrites", "--no-continue",
                    "-o", os.path.join(OUT_DIR, "_dl.%(ext)s"), src], check=True)
    out = os.path.join(OUT_DIR, "_dl_16k.wav")
    subprocess.run([_tool("ffmpeg"), "-y", "-i", raw, "-ac", "1", "-ar", "16000", out],
                   check=True, capture_output=True)
    return out


def fmt(t):
    return f"{int(t//60):02d}:{int(t%60):02d}"


def build_html(timeline, n, total, out_path):
    color = {sp: PALETTE[i % len(PALETTE)]
             for i, sp in enumerate(sorted(set(t[2] for t in timeline)))}
    bars = ""
    for s, e, sp in timeline:
        w = max((e - s) / total * 100, 0.05)
        bars += (f'<div title="{sp}: {fmt(s)}-{fmt(e)}" style="display:inline-block;'
                 f'width:{w:.3f}%;height:60px;background:{color[sp]};vertical-align:top;'
                 f'border-right:1px solid rgba(0,0,0,.15)"></div>')
    chips = "".join(
        f'<span style="background:{c};color:#fff;padding:4px 12px;border-radius:16px;'
        f'font:600 13px sans-serif;margin:3px">{sp}</span>' for sp, c in color.items())
    changes = sum(1 for i in range(1, len(timeline)) if timeline[i][2] != timeline[i-1][2])
    html = f"""<!doctype html><meta charset=utf-8>
<body style="background:#0d0d1f;color:#eee;font-family:sans-serif;padding:26px">
<h2>Speaker Timeline — {n} speaker(s), {changes} changes over {fmt(total)}</h2>
<div style="width:100%;border-radius:10px;overflow:hidden;border:1px solid #333">{bars}</div>
<div style="display:flex;justify-content:space-between;color:#888;font:11px monospace;margin-top:4px">
{''.join(f'<span>{fmt(total*i/8)}</span>' for i in range(9))}</div>
<div style="margin-top:16px">{chips}</div>
<table style="margin-top:20px;border-collapse:collapse;font:13px monospace">
<tr style="color:#888"><td>#</td><td>start</td><td>end</td><td>dur</td><td>speaker</td></tr>
{''.join(f'<tr><td>{i+1}</td><td>{fmt(s)}</td><td>{fmt(e)}</td><td>{e-s:.1f}s</td>'
         f'<td style=color:{color[sp]}>{sp}</td></tr>' for i,(s,e,sp) in enumerate(timeline))}
</table></body>"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)


def main():
    ap = argparse.ArgumentParser(description="Offline speaker diarization + visual timeline.")
    ap.add_argument("input", help="audio file path or media URL")
    ap.add_argument("--speakers", type=int, default=None,
                    help="force a known speaker count (most reliable when you know it)")
    ap.add_argument("--threshold", type=float, default=0.85,
                    help="auto-count merge threshold (0.6=pyannote default; 0.85 merges "
                         "an over-split expressive speaker). Ignored with --speakers.")
    ap.add_argument("--min-speaker-dur", type=float, default=20.0,
                    help="merge clusters with less total speech than this (s) into the "
                         "nearest real speaker. Ignored with --speakers.")
    ap.add_argument("--no-open", action="store_true", help="don't auto-open the HTML")
    args = ap.parse_args()

    print(f"-> preparing audio: {args.input}")
    wav = fetch_audio(args.input)
    mode = f"known N={args.speakers}" if args.speakers else \
           f"auto-N (threshold={args.threshold}, min_speaker_dur={args.min_speaker_dur}s)"
    print(f"-> diarizing [{mode}] ... (first run downloads the model; GPU strongly recommended)")
    pipe = load_pipeline(clustering_threshold=None if args.speakers else args.threshold)
    tl, n = diarize_file(wav, pipe=pipe, num_speakers=args.speakers,
                         min_speaker_dur=0.0 if args.speakers else args.min_speaker_dur)

    import soundfile as sf
    info = sf.info(wav)
    total = info.frames / info.samplerate
    changes = sum(1 for i in range(1, len(tl)) if tl[i][2] != tl[i-1][2])

    print(f"\n{'='*54}\n  RESULT: {n} speaker(s)  |  {changes} speaker changes\n{'='*54}")
    for i, (s, e, sp) in enumerate(tl):
        print(f"  #{i+1:>3}  {fmt(s)}-{fmt(e)}  ({e-s:4.1f}s)  {sp}")

    out_html = os.path.join(OUT_DIR, "timeline.html")
    out_json = os.path.join(OUT_DIR, "verify_result.json")
    build_html(tl, n, total, out_html)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"n_speakers": n, "n_changes": changes,
                   "timeline": [[round(s, 2), round(e, 2), sp] for s, e, sp in tl]},
                  f, indent=2)
    print(f"\nVisual timeline: {out_html}\nJSON: {out_json}")
    if not args.no_open:
        webbrowser.open(pathlib.Path(out_html).resolve().as_uri())


if __name__ == "__main__":
    main()
