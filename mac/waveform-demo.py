# Unbound Planet — Audio Visualizer (.app GUI)
# GUI: choose an audio file, click Create → saves PNGs next to the source file.

import threading, traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless backend; no interactive windows
import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display

import tkinter as tk
from tkinter import filedialog, messagebox

APP_TITLE = "UP Audio Visualizer"
ABOUT_TEXT = (
    "UP Audio Visualizer\n"
    "Exports waveform, spectrograms (linear & log), tempo, and RMS.\n\n"
    "An Unbound Planet • Nova project\n"
    "https://unboundplanet.com/"
)

AUDIO_FILTERS = [
    ("Audio files", "*.wav *.flac *.mp3 *.ogg *.m4a *.aac *.aiff *.aif *.aifc"),
    ("All files", "*.*"),
]

def analyze(audio_path: Path, log_cb=print):
    audio_path = audio_path.expanduser().resolve(strict=True)
    out_dir = audio_path.parent
    out_prefix = audio_path.stem

    def out(name: str) -> Path:
        return out_dir / f"{out_prefix}_{name}.png"

    log_cb(f"[i] Audio: {audio_path}")
    log_cb(f"[i] Output dir: {out_dir}")

    # 1) Load mono (analysis) at native sr
    y, sr = librosa.load(str(audio_path), sr=None, mono=True)

    # 2) Waveform
    plt.figure(figsize=(12, 4))
    librosa.display.waveshow(y, sr=sr, alpha=0.85)
    plt.title("Waveform")
    plt.xlabel("Time (s)"); plt.ylabel("Amplitude"); plt.tight_layout()
    wf_path = out("waveform"); plt.savefig(wf_path, dpi=300); plt.close()
    log_cb(f"[✓] Saved {wf_path}")

    # 3) Spectrograms
    S = librosa.stft(y, n_fft=2048, hop_length=512, win_length=2048)
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)

    # Linear Hz
    plt.figure(figsize=(12, 6))
    librosa.display.specshow(S_db, sr=sr, hop_length=512, x_axis="time", y_axis="hz", cmap="magma")
    plt.colorbar(format="%+2.0f dB", label="Intensity")
    plt.title("Spectrogram (Linear Hz)"); plt.tight_layout()
    sp_lin_path = out("spectrogram_linear"); plt.savefig(sp_lin_path, dpi=300); plt.close()
    log_cb(f"[✓] Saved {sp_lin_path}")

    # Log
    plt.figure(figsize=(12, 6))
    librosa.display.specshow(S_db, sr=sr, hop_length=512, x_axis="time", y_axis="log", cmap="magma")
    plt.colorbar(format="%+2.0f dB", label="Intensity")
    plt.title("Spectrogram (Log Scale)"); plt.tight_layout()
    sp_log_path = out("spectrogram_log"); plt.savefig(sp_log_path, dpi=300); plt.close()
    log_cb(f"[✓] Saved {sp_log_path}")

    # 4) Tempo (BPM)
    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo_scalar = float(tempo)
    except Exception:
        t_arr = librosa.beat.tempo(y=y, sr=sr, aggregate=None)
        tempo_scalar = float(np.asarray(t_arr).ravel()[0])
    log_cb(f"[i] Estimated Tempo: {tempo_scalar:.2f} BPM")

    # 5) Dynamics (RMS)
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    times = librosa.times_like(rms, sr=sr, hop_length=512)
    plt.figure(figsize=(12, 4))
    plt.plot(times, rms, linewidth=1.2)
    plt.title("Dynamics Over Time (RMS Energy)")
    plt.xlabel("Time (s)"); plt.ylabel("Energy"); plt.tight_layout()
    dyn_path = out("dynamics"); plt.savefig(dyn_path, dpi=300); plt.close()
    log_cb(f"[✓] Saved {dyn_path}")

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("760x360")
        self.configure(bg="#0f0f0f")
        self._build_menu()
        self._build_ui()

    def _build_menu(self):
        m = tk.Menu(self)
        helpm = tk.Menu(m, tearoff=0)
        helpm.add_command(label="About", command=lambda: messagebox.showinfo("About", ABOUT_TEXT))
        m.add_cascade(label="Help", menu=helpm)
        self.config(menu=m)

    def _build_ui(self):
        pad = 12

        # Title
        title = tk.Label(self, text="Visualize Audio → PNG", fg="#e6e6e6", bg="#0f0f0f",
                         font=("SF Pro Display", 18, "bold"))
        title.pack(pady=(18, 8))

        # Path row
        row = tk.Frame(self, bg="#0f0f0f")
        row.pack(fill="x", padx=pad, pady=(8, 4))

        self.path_var = tk.StringVar()
        entry = tk.Entry(row, textvariable=self.path_var, bg="#181818", fg="#ddd",
                         insertbackground="#ddd", relief="flat", highlightthickness=1)
        entry.pack(side="left", fill="x", expand=True, ipady=6)

        def browse():
            p = filedialog.askopenfilename(
                title="Choose audio file",
                filetypes=AUDIO_FILTERS,
                initialdir=str(Path.home())
            )
            if p:
                self.path_var.set(p)

        btn = tk.Button(row, text="Browse…", command=browse)
        btn.pack(side="left", padx=(8, 0))

        # Note
        note = tk.Label(
            self,
            text="Note: results are saved next to the source audio (same folder).",
            fg="#aaa", bg="#0f0f0f"
        )
        note.pack(anchor="w", padx=pad, pady=(2, 10))

        # Create button
        create = tk.Button(self, text="Create", command=self._on_create, width=16)
        create.pack(pady=(0, 8))

        # Log area
        self.log = tk.Text(self, height=10, bg="#111", fg="#cfcfcf",
                           insertbackground="#cfcfcf", relief="flat")
        self.log.pack(fill="both", expand=True, padx=pad, pady=(6, 12))

    def log(self, msg):  
        pass

    def _log(self, msg: str):
        self.log.insert("end", msg + "\n")
        self.log.see("end")
        self.update_idletasks()

    def _on_create(self):
        path = self.path_var.get().strip()
        if not path:
            messagebox.showwarning("Missing file", "Choose an audio file first.")
            return
        p = Path(path)
        # Run analysis in a thread so the UI stays responsive
        def run():
            try:
                self._log("— Running…")
                analyze(p, log_cb=self._log)
                self._log("— Done.")
            except Exception as e:
                self._log("! Error:\n" + "".join(traceback.format_exception(e)))
                messagebox.showerror("Error", str(e))
        threading.Thread(target=run, daemon=True).start()

if __name__ == "__main__":
    App().mainloop()
