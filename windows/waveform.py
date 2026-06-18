# UP Audio Visualizer

import matplotlib
matplotlib.use("Agg")

import sys, threading, subprocess, os, webbrowser, gc, ctypes, time, traceback, shutil, json
from pathlib import Path
from PIL import Image, ImageTk, ImageDraw
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import tkinter.font as tkfont

IS_WIN = (os.name == "nt")
IS_MAC = (sys.platform == "darwin")
IS_LIN = (sys.platform.startswith("linux"))

if IS_MAC:
    _mpldir = os.path.expanduser("~/Library/Caches/matplotlib")
elif IS_WIN:
    _mpldir = os.path.join(
        os.getenv("LOCALAPPDATA") or os.getenv("APPDATA") or os.path.expanduser("~"),
        "matplotlib"
    )
else:  # Linux and others
    _mpldir = os.path.expanduser("~/.cache/matplotlib")
os.makedirs(_mpldir, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", _mpldir)

APP_TITLE = "UP Audio Visualizer"
ABOUT_IMG = "unbound.png"   # ship with app (PyInstaller: --add-data "unbound.png;.")

AUDIO_FILTERS = [
    ("Audio files", "*.wav *.flac *.mp3 *.ogg *.m4a *.aac *.aiff *.aif *.aifc"),
    ("All files", "*.*"),
]

def resource_path(rel: str) -> Path:
    base = getattr(sys, "_MEIPASS", os.path.abspath("."))
    return Path(base) / rel

def _ensure_ffmpeg():
    """Ensure FFmpeg & FFprobe are available (Windows .exe or macOS/Linux binary)."""
    exe_dir = Path(sys._MEIPASS) if hasattr(sys, "_MEIPASS") else Path(".")

    ffmpeg = exe_dir / ("ffmpeg.exe" if IS_WIN else "ffmpeg")
    ffprobe = exe_dir / ("ffprobe.exe" if IS_WIN else "ffprobe")

    if IS_MAC:
        bundle_res = Path(sys.argv[0]).resolve().parent.parent / "Resources"
        ffmpeg = bundle_res / "ffmpeg"
        ffprobe = bundle_res / "ffprobe"

    if ffmpeg.exists() and ffprobe.exists():
        bin_dir = ffmpeg.parent
        os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")
        return True

    if shutil.which("ffmpeg") and shutil.which("ffprobe"):
        return True

    return False

def _enable_windows_hi_dpi():
    if not IS_WIN:
        return
    try:
        ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
    except Exception:
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            try:
                ctypes.windll.user32.SetProcessDPIAware()
            except Exception:
                pass

_enable_windows_hi_dpi()

def analyze(audio_path: Path, log_cb=lambda m: None):
    from matplotlib import pyplot as plt
    import numpy as np
    import librosa, librosa.display

    matplotlib.rcParams["font.family"] = "DejaVu Sans"
    OUT_DPI = 150 if IS_LIN else 300

    audio_path = audio_path.expanduser().resolve(strict=True)
    out_dir = audio_path.parent
    out_prefix = audio_path.stem

    def out(name: str) -> Path:
        return out_dir / f"{out_prefix}_{name}.png"

    log_cb(f"Audio: {audio_path.name}")

    try:
        y, sr = librosa.load(str(audio_path), sr=None, mono=True)
    except Exception as ex:
        msg = f"Could not read audio: {audio_path.name}\n\n{ex}"
        if audio_path.suffix.lower() == ".mp3":
            msg += (
                "\n\nTip: MP3 reading may require FFmpeg in PATH.\n"
                "Install FFmpeg and restart the app."
            )
        raise RuntimeError(msg) from ex

    # Waveform
    plt.figure(figsize=(12, 4))
    librosa.display.waveshow(y, sr=sr, alpha=0.85)
    plt.title("Waveform"); plt.xlabel("Time (s)"); plt.ylabel("Amplitude"); plt.tight_layout()
    plt.savefig(out("waveform"), dpi=OUT_DPI); plt.close()

    # Spectrograms
    S = librosa.stft(y, n_fft=2048, hop_length=512, win_length=2048)
    import numpy as np
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)

    plt.figure(figsize=(12, 6))
    librosa.display.specshow(S_db, sr=sr, hop_length=512, x_axis="time", y_axis="hz", cmap="magma")
    plt.colorbar(format="%+2.0f dB", label="Intensity")
    plt.title("Spectrogram (Linear Hz)"); plt.tight_layout()
    plt.savefig(out("spectrogram_linear"), dpi=OUT_DPI); plt.close()

    plt.figure(figsize=(12, 6))
    librosa.display.specshow(S_db, sr=sr, hop_length=512, x_axis="time", y_axis="log", cmap="magma")
    plt.colorbar(format="%+2.0f dB", label="Intensity")
    plt.title("Spectrogram (Log Scale)"); plt.tight_layout()
    plt.savefig(out("spectrogram_log"), dpi=OUT_DPI); plt.close()

    # Tempo
    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo_scalar = float(tempo)
    except Exception:
        t_arr = librosa.beat.tempo(y=y, sr=sr, aggregate=None)
        tempo_scalar = float(np.asarray(t_arr).ravel()[0])
    log_cb(f"Tempo ≈ {tempo_scalar:.1f} BPM")

    # Dynamics
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    times = librosa.times_like(rms, sr=sr, hop_length=512)
    plt.figure(figsize=(12, 4))
    plt.plot(times, rms, linewidth=1.2)
    plt.title("Dynamics Over Time (RMS Energy)")
    plt.xlabel("Time (s)"); plt.ylabel("Energy"); plt.tight_layout()
    plt.savefig(out("dynamics"), dpi=OUT_DPI); plt.close()

    avg_rms_db = 20 * np.log10(np.mean(rms) + 1e-12)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    avg_centroid = float(np.mean(centroid))

    try:
        key_guess = librosa.feature.chroma_cqt(y=y, sr=sr).mean(axis=1).argmax()
        keys = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
        key_name = keys[int(key_guess)]
    except Exception:
        key_name = None

    peak_val = float(np.max(np.abs(y)))
    crest = peak_val / (np.mean(np.abs(y)) + 1e-12)
    rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))

    plt.close("all"); gc.collect()

    return dict(
        tempo=tempo_scalar,
        avg_rms_db=avg_rms_db,
        avg_centroid=avg_centroid,
        key=key_name,
        sr=sr,
        peak=peak_val,
        crest=crest,
        rolloff=rolloff,
        zcr=zcr
    )

def write_metadata_txt(audio_path: Path, metrics: dict):
    txt_path = audio_path.with_suffix(".txt")

    # ffprobe → JSON
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_format", "-show_streams", "-show_entries",
             "stream=codec_name,codec_long_name,profile,codec_type,channels,channel_layout,sample_rate,bits_per_sample,bit_rate",
             str(audio_path)],
            capture_output=True, text=True, check=True
        )
        meta = json.loads(result.stdout)
    except Exception:
        meta = {}

    stream = next((s for s in meta.get("streams", []) if s.get("codec_type") == "audio"), {})
    fmt    = meta.get("format", {})
    tags   = fmt.get("tags", {}) or {}

    lines = []
    lines.append(f"File: {audio_path.name}")

    # Codec / Format
    if stream.get("codec_long_name"):
        codec_line = stream["codec_long_name"]
        if stream.get("profile"):
            codec_line += f" ({stream['profile']})"
        lines.append(f"Codec: {codec_line}")
    if fmt.get("format_long_name"):
        lines.append(f"Format: {fmt['format_long_name']}")
    elif fmt.get("format_name"):
        lines.append(f"Format: {fmt['format_name']}")

    # Bitrate
    if fmt.get("bit_rate"):
        lines.append(f"Bitrate: {int(fmt['bit_rate'])//1000} kbps")

    # Sample rate
    if stream.get("sample_rate"):
        sr = int(stream["sample_rate"])
        lines.append(f"Sample Rate: {sr/1000:.1f} kHz")

    # Channels
    if stream.get("channels"):
        ch = int(stream["channels"])
        layout = stream.get("channel_layout", "")
        lines.append(f"Channels: {layout or ch}")

    # Bit depth
    if stream.get("bits_per_sample") and int(stream["bits_per_sample"]) > 0:
        lines.append(f"Bit Depth: {stream['bits_per_sample']} bit")

    # File size
    if fmt.get("size"):
        size_mb = int(fmt["size"]) / (1024*1024)
        lines.append(f"Size: {size_mb:.1f} MB")

    # Duration
    if fmt.get("duration"):
        dur = float(fmt["duration"])
        mins, secs = divmod(dur, 60)
        lines.append(f"Duration: {int(mins):02d}:{secs:05.2f}")

    # Other tags (except lyrics)
    lyrics_text = None
    for k, v in tags.items():
        if k.lower() in ("lyrics", "unsyncedlyrics"):
            lyrics_text = v
        else:
            lines.append(f"{k.replace('_',' ').title()}: {v}")

    # Loudness (optional)
    try:
        loud_cmd = [
            "ffmpeg", "-i", str(audio_path),
            "-filter_complex", "ebur128=framelog=verbose",
            "-f", "null", "-"
        ]
        result = subprocess.run(loud_cmd, stderr=subprocess.PIPE, text=True)
        lufs = None
        for line in result.stderr.splitlines():
            if "Integrated loudness" in line:
                parts = line.strip().split()
                lufs = parts[-2]
                break
        if lufs:
            lines.append(f"Loudness (Integrated): {lufs} LUFS")
    except Exception:
        lines.append("Loudness: (could not measure)")

    # Analysis metrics
    lines.append("")
    lines.append("=== Analysis Metrics ===")
    lines.append(f"Tempo: ~{metrics['tempo']:.1f} BPM")
    if metrics.get("key"):
        lines.append(f"Key: {metrics['key']} (approx.)")
    lines.append(f"Avg RMS Energy: {metrics['avg_rms_db']:.1f} dB")
    lines.append(f"Spectral Centroid (brightness): {metrics['avg_centroid']:.0f} Hz")
    if "rolloff" in metrics:
        lines.append(f"Spectral Rolloff (85% energy): {metrics['rolloff']:.0f} Hz")
    if "zcr" in metrics:
        lines.append(f"Zero-Crossing Rate: {metrics['zcr']:.4f}")
    if "crest" in metrics:
        lines.append(f"Crest Factor (Peak/RMS): {metrics['crest']:.2f}")
    if "peak" in metrics:
        lines.append(f"Peak Amplitude: {metrics['peak']:.3f}")

    # Lyrics last
    if lyrics_text:
        lines.append("\n=== Lyrics ===")
        lines.append(lyrics_text.strip())

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def fatal_ffmpeg_missing(parent):
    win = tk.Toplevel(parent)
    win.title("FFmpeg Missing")
    win.configure(bg="#0f0f0f")
    win.resizable(False, False)
    win.transient(parent)
    win.grab_set()
    win.lift()
    win.focus_force()

    ffm = "ffmpeg.exe" if IS_WIN else "ffmpeg"
    ffp = "ffprobe.exe" if IS_WIN else "ffprobe"

    tk.Label(
        win,
        text=f"This app needs FFmpeg ({ffm} and {ffp})\n"
             "to decode and analyze audio.\n\n"
             "Place them next to this program or install system-wide.",
        fg="#ffffff", bg="#0f0f0f",
        font=("Segoe UI", 11) if IS_WIN else ("Helvetica", 12),
        wraplength=360, justify="center"
    ).pack(pady=20, padx=20)

    def _exit():
        parent.destroy()

    try:
        exit_btn = RoundedButton(win, text="Exit", command=_exit)
        exit_btn.pack(pady=12)
        win._exit_btn = exit_btn
    except Exception:
        tk.Button(win, text="Exit", command=_exit).pack(pady=12)

    win.update_idletasks()
    w, h = 400, 210
    sw, sh = win.winfo_screenwidth(), win.winfo_screenheight()
    x, y = (sw - w) // 2, (sh - h) // 2
    win.geometry(f"{w}x{h}+{x}+{y}")
    win.protocol("WM_DELETE_WINDOW", _exit)

class RoundedEntry(tk.Frame):
    def __init__(self, master, textvariable=None, height=34, radius=14,
                 fill="#181818", border="#2a2a2a", fg="#ffffff",
                 parent_bg="#0f0f0f", padx=12,
                 font=("Segoe UI", 10) if IS_WIN else ("Helvetica", 14), **kw):
        super().__init__(master, bg=parent_bg, **kw)
        self.h, self.r = int(height), int(radius)
        self.fill, self.border, self.parent_bg = fill, border, parent_bg
        self.padx = int(padx)

        self.canvas = tk.Canvas(self, height=self.h, bg=self.parent_bg,
                                highlightthickness=0, bd=0)
        self.canvas.pack(fill="x", expand=True)

        self.var = textvariable if textvariable is not None else tk.StringVar()
        self.entry = tk.Entry(self, textvariable=self.var, relief="flat", bd=0,
                              bg=self.fill, fg=fg, insertbackground=fg,
                              font=font)
        self._entry_win = self.canvas.create_window(
            self.padx, self.h//2, window=self.entry, anchor="w", height=self.h-10, width=10
        )
        self.canvas.bind("<Configure>", self._redraw)

    def _redraw(self, _=None):
        w = max(40, self.canvas.winfo_width())
        h = self.h
        r = min(self.r, h//2)
        self.canvas.delete("pill")
        img = Image.new("RGBA", (w, h), (0,0,0,0))
        draw = ImageDraw.Draw(img)
        draw.rounded_rectangle((0,0,w-1,h-1), r, fill=self.fill, outline=self.border, width=1)
        tkimg = ImageTk.PhotoImage(img)
        self._bg_tk = tkimg
        self.canvas.create_image(0, 0, image=tkimg, anchor="nw", tags="pill")
        self.canvas.coords(self._entry_win, self.padx, h//2)
        self.canvas.itemconfigure(self._entry_win, width=w - 2*self.padx)

    def get(self): return self.var.get()
    def set(self, v): self.var.set(v)

class RoundedButton(tk.Canvas):
    def __init__(self, master, text="", command=None,
                 radius=14, padding=(20, 10),
                 bg="#1e1e1e", fg="#ffffff",
                 hover="#2a2a2a", active="#333333",
                 font=("Segoe UI", 10, "bold") if IS_WIN else ("Helvetica", 14, "bold"),
                 border="#444444", **kw):
        super().__init__(master, highlightthickness=0, bd=0, bg=master["bg"], **kw)
        self.command = command
        self.text = text
        self.radius = radius
        self.colors = dict(bg=bg, fg=fg, hover=hover, active=active, border=border)
        self.font = font
        self.padding = padding
        self._img_ref = None
        self._draw(bg)
        self.bind("<Enter>", lambda e: self._draw(self.colors["hover"]))
        self.bind("<Leave>", lambda e: self._draw(self.colors["bg"]))
        self.bind("<ButtonPress-1>", lambda e: self._draw(self.colors["active"]))
        self.bind("<ButtonRelease-1>", self._on_release)

    def _draw(self, fill):
        f = tkfont.Font(font=self.font)
        w = f.measure(self.text) + self.padding[0]*2
        h = f.metrics("linespace") + self.padding[1]*2
        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.rounded_rectangle((0, 0, w-1, h-1), self.radius,
                               fill=fill, outline=self.colors["border"], width=1)
        self._img_ref = ImageTk.PhotoImage(img)
        self.delete("all")
        self.create_image(0, 0, image=self._img_ref, anchor="nw")
        self.create_text(w//2, h//2, text=self.text,
                         fill=self.colors["fg"], font=self.font)
        self.config(width=w, height=h)

    def _on_release(self, _):
        if self.command:
            self.command()
        self._draw(self.colors["hover"])

class UPMeter(tk.Canvas):
    def __init__(self, master, width=680, height=24, radius=12,
                 bg_color="#181818", gloss_top=(255,255,255,32), gloss_bottom=(0,0,0,96),
                 grad_a="#60a5fa", grad_b="#a78bfa", grad_c="#7c3aed", **kw):
        super().__init__(master, width=width, height=height, highlightthickness=0, bd=0, bg=master["bg"], **kw)
        self.w, self.h, self.r = width, height, radius
        self._running = False
        self._t0 = 0.0
        self._frac = 0.0
        self._bg_params = (bg_color, gloss_top, gloss_bottom)
        self._grad_params = (grad_a, grad_b, grad_c)
        self._render_all()

    def _rounded_rect(self, img, xy, r, fill, outline=None, ow=1):
        draw = ImageDraw.Draw(img)
        draw.rounded_rectangle(xy, r, fill=fill, outline=outline, width=ow)

    def _render_bg(self, base_hex, gloss_top_rgba, gloss_bot_rgba):
        img = Image.new("RGBA", (self.w, self.h), (0,0,0,0))
        self._rounded_rect(img, (0,0,self.w,self.h), self.r, fill=base_hex)
        self._rounded_rect(img, (0,0,self.w,self.h), self.r,
                           fill=None, outline=(255,255,255,28), ow=1)
        gloss = Image.new("RGBA", (self.w, self.h), (0,0,0,0))
        gt = Image.new("RGBA", (self.w, self.h//2), gloss_top_rgba)
        gb = Image.new("RGBA", (self.w, self.h - self.h//2), gloss_bot_rgba)
        gloss.paste(gt, (0,0), gt); gloss.paste(gb, (0,self.h//2), gb)
        img.alpha_composite(gloss)
        return img

    def _render_grad(self, a_hex, b_hex, c_hex):
        def hex_to_rgb(h):
            h = h.lstrip("#"); return tuple(int(h[i:i+2],16) for i in (0,2,4))
        a = hex_to_rgb(a_hex); b = hex_to_rgb(b_hex); c = hex_to_rgb(c_hex)
        W, H = self.w, self.h
        img = Image.new("RGBA", (W, H), (0,0,0,0)); px = img.load()
        for x in range(W):
            t = x / (W-1) if W > 1 else 1.0
            if t <= 0.7: u = t / 0.7; col = tuple(int(a[i] + (b[i]-a[i])*u) for i in range(3))
            else: u = (t - 0.7) / 0.3; col = tuple(int(b[i] + (c[i]-b[i])*u) for i in range(3))
            for y in range(H): px[x,y] = (*col, 255)
        overlay = Image.new("RGBA", (W,H), (255,255,255,40))
        img.alpha_composite(overlay)
        return img

    def _render_all(self):
        bg_color, gloss_top, gloss_bottom = self._bg_params
        self._bg_img = self._render_bg(bg_color, gloss_top, gloss_bottom)
        self._bg_tk  = ImageTk.PhotoImage(self._bg_img)
        self.delete("all")
        self._bg_id  = self.create_image(0, 0, image=self._bg_tk, anchor="nw")
        self._grad_img_full = self._render_grad(*self._grad_params)
        self._fill_tk = None
        self._fill_id = None
        self.set(self._frac)

    def resize(self, width):
        width = int(max(10, width))
        if width == self.w: return
        self.w = width
        self.config(width=self.w)
        self._render_all()

    def set(self, frac: float):
        self._frac = max(0.0, min(1.0, float(frac)))
        fill_w = int(self.w * self._frac)
        if self._fill_id:
            self.delete(self._fill_id); self._fill_id = None
        if fill_w <= 0: return
        cropped = self._grad_img_full.crop((0, 0, fill_w, self.h))
        mask = Image.new("L", (fill_w, self.h), 0)
        draw = ImageDraw.Draw(mask)
        draw.rounded_rectangle((0,0,fill_w,self.h), self.r, fill=255)
        cropped.putalpha(mask)
        self._fill_tk = ImageTk.PhotoImage(cropped)
        self._fill_id = self.create_image(0, 0, image=self._fill_tk, anchor="nw")

    def start(self, duration_ms=2200):
        if self._running: return
        self._running = True
        self._t0 = time.perf_counter()
        self._duration = max(300, int(duration_ms))
        self._tick()

    def stop(self):
        self._running = False
        self.set(0.0)

    def _ease(self, t):
        return t*t*(3 - 2*t)

    def _tick(self):
        if not self._running: return
        elapsed = (time.perf_counter() - self._t0) * 1000.0
        t = (elapsed % self._duration) / self._duration
        self.set(self._ease(t))
        self.after(16, self._tick)

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("900x320")
        self.minsize(900, 320)
        self.configure(bg="#0f0f0f")

        # Prefer native theme
        style = ttk.Style(self)
        themes = []
        if IS_MAC: themes += ["aqua"]
        if IS_WIN: themes += ["vista", "xpnative"]
        themes += ["clam"]
        for theme in themes:
            try:
                style.theme_use(theme)
                break
            except tk.TclError:
                continue
        style.configure('.', font=('Segoe UI', 10) if IS_WIN else ('Helvetica', 14))
        style.configure("UP.TButton", padding=(14, 8))

        self._build_menu()
        self._build_ui()
        self._prefill_from_argv()

    def _build_menu(self):
        m = tk.Menu(self)

        helpm = tk.Menu(m, tearoff=0)
        helpm.add_command(
            label="Check for Updates",
            command=lambda: webbrowser.open("https://github.com/theodor-ubp/py-audio-visualizer")
        )
        helpm.add_command(
            label="Support",
            command=lambda: webbrowser.open("https://ko-fi.com/tehodor9449790")
        )
        helpm.add_separator()
        helpm.add_command(label="About", command=self._show_about)

        m.add_cascade(label="Help", menu=helpm)
        self.config(menu=m)

    def _build_ui(self):
        pad = 14

        title = tk.Label(
            self, text="Visualize Audio → PNG", fg="#ffffff", bg="#0f0f0f",
            font=("Segoe UI", 18, "bold") if IS_WIN else ("Helvetica", 18, "bold")
        )
        title.pack(pady=(18, 8))

        # Clickable main logo
        logo_path = resource_path(ABOUT_IMG)
        if logo_path.exists():
            img = Image.open(logo_path); w, h = img.size
            img = img.resize((w // 6, h // 6), Image.LANCZOS)
            tkimg = ImageTk.PhotoImage(img)
            self._main_logo_ref = tkimg
            lbl_logo = tk.Label(self, image=tkimg, bg="#0f0f0f", cursor="hand2")
            lbl_logo.pack(pady=(0, 10))
            lbl_logo.bind("<Button-1>", lambda e: webbrowser.open("https://unboundplanet.com/"))

        row = tk.Frame(self, bg="#0f0f0f"); row.pack(fill="x", padx=pad, pady=(8, 4))
        self.path_var = tk.StringVar()
        self.path_input = RoundedEntry(
            row, textvariable=self.path_var, height=36, radius=14,
            fill="#181818", border="#2a2a2a", fg="#ffffff", parent_bg="#0f0f0f",
            padx=12, font=("Segoe UI", 10) if IS_WIN else ("Helvetica", 14)
        )
        self.path_input.pack(side="left", fill="x", expand=True)

        self.btn_browse = RoundedButton(row, text="Browse…", command=self._browse)
        self.btn_browse.pack(side="left", padx=(8, 0))

        tk.Label(
            self,
            text="Results are saved next to the source audio (same folder). Overwrites the old pics/txt automatically.",
            fg="#ffffff", bg="#0f0f0f", font=("Segoe UI", 10) if IS_WIN else ("Helvetica", 14),
            justify="center"
        ).pack(padx=pad, pady=(2, 10), anchor="center")

        btnrow = tk.Frame(self, bg="#0f0f0f"); btnrow.pack(pady=(0, 6))
        self.btn_create = RoundedButton(btnrow, text="Create", command=self._on_create)
        self.btn_create.pack(side="left", padx=6)
        self.btn_open = RoundedButton(btnrow, text="Open Folder", command=self._open_folder)
        self.btn_open.pack(side="left", padx=6)
        self.btn_about = RoundedButton(btnrow, text="About", command=self._show_about)
        self.btn_about.pack(side="left", padx=6)

        self.status_var = tk.StringVar(value="Ready.")
        tk.Label(
            self, textvariable=self.status_var, fg="#ffffff", bg="#0f0f0f",
            font=("Segoe UI", 10) if IS_WIN else ("Helvetica", 14)
        ).pack(anchor="w", padx=pad, pady=(2, 6))

        if not _ensure_ffmpeg():
            fatal_ffmpeg_missing(self)
            return

        self.meter = UPMeter(self, width=900 - 2*pad, height=26, radius=13)
        self.meter.pack(fill="x", padx=pad, pady=(0, 4))

        def _sync_meter():
            inner = max(100, self.winfo_width() - 2*pad)
            self.meter.resize(inner)
        self.after(50, _sync_meter)
        self.bind("<Configure>", lambda e: _sync_meter())

    def _show_about(self):
        win = tk.Toplevel(self)
        win.title("About")
        win.configure(bg="#0f0f0f")
        win.resizable(False, False)

        img_path = resource_path(ABOUT_IMG)
        if img_path.exists():
            img = Image.open(img_path).resize((300, 137), Image.LANCZOS)
            tkimg = ImageTk.PhotoImage(img)
            lbl_img = tk.Label(win, image=tkimg, bg="#0f0f0f", cursor="hand2")
            lbl_img.image = tkimg
            lbl_img.pack(padx=20, pady=(20, 10))
            lbl_img.bind("<Button-1>", lambda e: webbrowser.open("https://unboundplanet.com/"))

        row = tk.Frame(win, bg="#0f0f0f"); row.pack(padx=20, pady=(0, 8))
        tk.Label(
            row, text="UP Audio Visualizer 1.0 -",
            fg="#ffffff", bg="#0f0f0f",
            font=("Segoe UI", 11, "bold") if IS_WIN else ("Helvetica", 14, "bold")
        ).pack(side="left")
        upd = tk.Label(
            row, text="Check for Updates",
            fg="#4da6ff", bg="#0f0f0f",
            font=("Segoe UI", 11, "underline") if IS_WIN else ("Helvetica", 14, "underline"),
            cursor="hand2"
        )
        upd.pack(side="left")
        upd.bind("<Button-1>", lambda e: webbrowser.open("https://github.com/theodor-ubp/py-audio-visualizer"))

        support_link = tk.Label(
            row,
            text="Support",
            fg="#4da6ff", bg="#0f0f0f",
            font=("Segoe UI", 11, "underline") if IS_WIN else ("Helvetica", 14, "underline"),
            cursor="hand2"
        )
        support_link.pack(side="left", padx=(12, 0))
        support_link.bind(
            "<Button-1>",
            lambda e: webbrowser.open("https://ko-fi.com/tehodor9449790")
        )

        desc = (
            "An Unbound Planet • Nova project\n\n"
            "This tool converts audio files into high-quality visualizations "
            "and extracts detailed metadata:\n\n"
            "• Waveform and spectrograms (linear & log scale)\n"
            "• Tempo, key, loudness, RMS energy, spectral features\n"
            "• Codec, bitrate, sample rate, channels, file size, tags\n"
            "• Lyrics (if embedded in the file)\n\n"
            "Results: PNG images + a detailed TXT report saved "
            "next to the source audio."
        )
        tk.Label(
            win, text=desc, fg="#ffffff", bg="#0f0f0f",
            font=("Segoe UI", 11) if IS_WIN else ("Helvetica", 14),
            justify="left", wraplength=460
        ).pack(padx=20, pady=(0, 12))

        link = tk.Label(
            win, text="https://unboundplanet.com/", fg="#4da6ff",
            bg="#0f0f0f", cursor="hand2",
            font=("Segoe UI", 12, "underline") if IS_WIN else ("Helvetica", 15, "underline")
        )
        link.pack(pady=(0, 20))
        link.bind("<Button-1>", lambda e: webbrowser.open("https://unboundplanet.com/"))
        credit = tk.Label(
            win,
            text="Uses FFmpeg - https://ffmpeg.org - Licensed under LGPL v2.1.",
            fg="#4da6ff", bg="#0f0f0f",
            font=("Segoe UI", 9, "underline") if IS_WIN else ("Helvetica", 12, "underline"),
            cursor="hand2"
        )
        credit.pack(pady=(4, 16))
        credit.bind("<Button-1>", lambda e: webbrowser.open("https://ffmpeg.org"))

    def _prefill_from_argv(self):
        if len(sys.argv) >= 2 and Path(sys.argv[1]).exists():
            self.path_var.set(sys.argv[1])

    def _browse(self):
        p = filedialog.askopenfilename(title="Choose audio file",
                                       filetypes=AUDIO_FILTERS,
                                       initialdir=str(Path.home()))
        if p: self.path_var.set(p)

    def _open_folder(self):
        path = self.path_var.get().strip()
        if not path: return
        folder = str(Path(path).expanduser().resolve().parent)
        try:
            if IS_WIN:
                os.startfile(folder)
            elif IS_MAC:
                subprocess.run(["open", folder], check=False)
            else:
                subprocess.run(["xdg-open", folder], check=False)
        except Exception:
            pass

    def _start_status_toggle(self, base_msg: str):
        self._toggle_running = True
        self._toggle_state = True
        def _tick():
            if not getattr(self, "_toggle_running", False): return
            self.status_var.set(base_msg if self._toggle_state else "Working…")
            self._toggle_state = not self._toggle_state
            self.after(3000, _tick)
        _tick()

    def _stop_status_toggle(self, final_msg: str):
        self._toggle_running = False
        self.status_var.set(final_msg)

    def _set_running(self, running: bool, status: str = ""):
        state = "disabled" if running else "normal"
        for b in (self.btn_browse, self.btn_create, self.btn_open, self.btn_about):
            b.config(state=state)
        if running:
            self.meter.start(2200)
            if status: self.status_var.set(status)
        else:
            self.meter.stop()
            if status: self.status_var.set(status)

    def _status(self, msg: str):
        self.status_var.set(msg)
        self.update_idletasks()

    def _on_create(self):
        path = self.path_var.get().strip()
        if not path:
            messagebox.showwarning("Missing file", "Choose an audio file first.")
            return
        p = Path(path)

        def run():
            try:
                self._set_running(True, f"Audio: {p.name}")
                self.after(0, lambda: self._start_status_toggle(f"Audio: {p.name}"))
                metrics = analyze(p, log_cb=self._status)
                write_metadata_txt(p, metrics)
                self.after(0, lambda: self._stop_status_toggle("Done."))
                self._set_running(False)
            except Exception as e:
                self.after(0, lambda: self._stop_status_toggle("Error."))
                self._set_running(False)
                tb = "".join(traceback.format_exception(e))
                messagebox.showerror("Error", tb[-1200:])

        threading.Thread(target=run, daemon=True).start()

if __name__ == "__main__":
    App().mainloop()
