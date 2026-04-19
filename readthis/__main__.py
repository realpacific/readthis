#!/usr/bin/env python3

import argparse
import json
import logging
import os
import re
import sys
import threading
import time
import warnings
import pathlib

# Allow PyTorch to fall back from MPS (Apple Silicon GPU) to CPU for unsupported ops.
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
warnings.filterwarnings("ignore")
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

# If the model is already in the Hugging Face cache, force offline mode so the
# pipeline never attempts a network check on startup
try:
    from huggingface_hub import scan_cache_dir
    cached_repos = {repo.repo_id for repo in scan_cache_dir().repos}
    if "hexgrad/Kokoro-82M" in cached_repos:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    else:
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "0"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "0"
except Exception:
    pass


def _load_config():
    config_path = pathlib.Path.home() / ".config" / "readthis" / "config.json"
    if not config_path.exists():
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text("{}")
    # Fall back to config.json in the current directory for local dev.
    with open(config_path) as f:
        return json.load(f)
    return {}


_config = _load_config()

# Audio output sample rate expected by the Kokoro model.
SAMPLE_RATE = 24000

# How many seconds a single left/right arrow keypress seeks.
SEEK_SECONDS = 5

SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

URL_PATTERN = re.compile(r"^https?://")

VOICES = [
    "af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica", "af_kore",
    "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
    "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam", "am_michael",
    "am_onyx", "am_puck", "am_santa",
    "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
    "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
    "ef_dora", "em_alex", "em_santa",
    "ff_siwis",
    "hf_alpha", "hf_beta", "hm_omega", "hm_psi",
    "if_sara", "im_nicola",
    "jf_alpha", "jf_gongitsune", "jf_nezumi", "jf_tebukuro", "jm_kumo",
    "pf_dora", "pm_alex", "pm_santa",
    "zf_xiaobei", "zf_xiaoni", "zf_xiaoxiao", "zf_xiaoyi",
    "zm_yunjian", "zm_yunxi", "zm_yunxia", "zm_yunyang",
]

generation_done = threading.Event()
quit_requested = threading.Event()


def get_text(input_arg):
    """Resolve the text to speak from stdin, clipboard, a URL, or a literal string."""
    # Piped input takes priority: `echo "hello" | python speak.py`
    if not sys.stdin.isatty():
        text = sys.stdin.read().strip()
        if not text:
            raise SystemExit("No input from stdin.")
        return text

    # No argument → read whatever is on the clipboard.
    if input_arg is None:
        import pyperclip
        text = pyperclip.paste()
        if not text:
            raise SystemExit("Clipboard is empty.")
        return text

    # URL → fetch and extract the article body with trafilatura.
    if URL_PATTERN.match(input_arg):
        import trafilatura
        downloaded = trafilatura.fetch_url(input_arg)
        if downloaded is None:
            raise SystemExit("Failed to fetch URL.")
        text = trafilatura.extract(downloaded)
        if not text:
            raise SystemExit("Could not extract text from URL.")
        return text

    # Otherwise treat the argument as the literal text to speak.
    return input_arg


def _format_time(sample_count):
    """Convert a sample offset to a MM:SS display string."""
    total_seconds = int(sample_count / SAMPLE_RATE)
    minutes, seconds = divmod(total_seconds, 60)
    return f"{minutes:02d}:{seconds:02d}"


def _play_streaming(audio_buffer):
    """Play audio from a growing buffer while generation continues in the background.

    How the shared buffer works
    ---------------------------
    `audio_buffer` is a one-element list that holds a numpy float32 array.
    The generation thread grows the audio by replacing `audio_buffer[0]` with a
    new, longer array (via np.concatenate). In CPython, assigning to a list index
    is protected by the GIL and is therefore atomic — the audio callback can read
    `audio_buffer[0]` at any time and will always see a valid, complete array,
    never a half-written one. This means no lock is needed in the hot callback path.

    The sounddevice OutputStream callback runs in a dedicated real-time audio thread.
    It reads `playback_pos` samples ahead on every call to fill the hardware buffer.
    `playback_lock` protects `playback_pos` and `is_paused` between the callback
    thread and the main input-polling thread.
    """
    import queue
    import readchar
    import sounddevice as sd

    # On Unix, when stdin is a pipe readchar reads from it instead of the terminal.
    # Redirect stdin to /dev/tty so keypresses still work. On Windows, readchar uses
    # msvcrt which reads from the console directly — no redirect needed.
    # Falls back to simple blocking playback when no terminal exists (headless).
    import platform
    original_stdin = sys.stdin
    tty_stream = None
    if platform.system() != "Windows" and not sys.stdin.isatty():
        try:
            import io
            tty_stream = io.open("/dev/tty", "r")
            sys.stdin = tty_stream
        except OSError:
            generation_done.wait()
            sd.play(audio_buffer[0], samplerate=SAMPLE_RATE)
            sd.wait()
            return

    playback_pos = [0]
    is_paused = [False]
    playback_lock = threading.Lock()
    seek_samples = SEEK_SECONDS * SAMPLE_RATE
    spinner_frame = [0]
    key_queue = queue.Queue()

    def audio_callback(outdata, frame_count, time_info, status):
        """Fill the sounddevice output buffer each cycle.

        Called from a real-time audio thread — must not block.
        Reads `frame_count` samples from `audio_buffer[0]` starting at
        `playback_pos[0]`. Outputs silence when paused, when buffering
        (playback has caught up to generation), or past the end of audio.
        """
        # Atomic read: always a valid numpy array, no lock needed (see docstring).
        current_buf = audio_buffer[0]
        available_samples = len(current_buf)

        with playback_lock:
            if is_paused[0]:
                outdata[:] = 0
                return

            current_sample = playback_pos[0]

            # Playback has reached or passed the available audio.
            if current_sample >= available_samples:
                outdata[:] = 0
                # If generation is also finished, auto-pause at the end.
                if generation_done.is_set():
                    is_paused[0] = True
                # Otherwise output silence and wait for more audio to arrive.
                return

            end_sample = current_sample + frame_count

            if end_sample > available_samples:
                # Partial buffer: copy what exists, zero-pad the rest.
                samples_to_copy = available_samples - current_sample
                outdata[:samples_to_copy,
                        0] = current_buf[current_sample:available_samples]
                outdata[samples_to_copy:] = 0
                playback_pos[0] = available_samples
            else:
                # Full buffer: copy exactly frame_count samples.
                outdata[:, 0] = current_buf[current_sample:end_sample]
                playback_pos[0] = end_sample

    def render_status_line():
        """Overwrite the current terminal line with playback position and controls."""
        with playback_lock:
            current_sample = playback_pos[0]
            currently_paused = is_paused[0]

        total_available_samples = len(audio_buffer[0])
        state_icon = "⏸" if currently_paused else "▶"

        if generation_done.is_set():
            generating_suffix = "  ✓"
        else:
            frame = SPINNER_FRAMES[spinner_frame[0] % len(SPINNER_FRAMES)]
            spinner_frame[0] += 1
            generating_suffix = f"  {frame}"

        sys.stderr.write(
            f"\r\033[K{state_icon} "
            f"{_format_time(current_sample)} / {_format_time(total_available_samples)}"
            f"  [space] pause  [←/→] ±{SEEK_SECONDS}s  [q] quit"
            f"{generating_suffix}"
        )
        sys.stderr.flush()

    def key_reader():
        try:
            while not quit_requested.is_set():
                key_queue.put(readchar.readkey())
        except Exception:
            pass

    # Keyboard input is handled by readchar in a dedicated daemon thread, which puts
    # each keypress into a queue. The main thread drains that queue with a 0.25s timeout
    threading.Thread(target=key_reader, daemon=True).start()

    stream = sd.OutputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        callback=audio_callback,
        blocksize=1024,
    )
    stream.start()
    render_status_line()

    try:
        while True:
            try:
                key = key_queue.get(timeout=0.25)
            except queue.Empty:
                render_status_line()
                continue

            if key == readchar.key.SPACE:
                with playback_lock:
                    is_paused[0] = not is_paused[0]
            elif key == readchar.key.LEFT:
                with playback_lock:
                    playback_pos[0] = max(0, playback_pos[0] - seek_samples)
            elif key == readchar.key.RIGHT:
                with playback_lock:
                    samples_available = len(audio_buffer[0])
                    playback_pos[0] = min(
                        max(samples_available - 1, 0), playback_pos[0] + seek_samples)
            elif key in ("q", "Q", readchar.key.CTRL_C):
                quit_requested.set()
                break

            render_status_line()
    finally:
        stream.stop()
        stream.close()
        if tty_stream is not None:
            sys.stdin = original_stdin
            tty_stream.close()
        sys.stderr.write("\r\033[K")
        sys.stderr.flush()


def speak(text, voice="af_heart", speed=1.0, lang="a"):
    """Generate speech for `text` and stream it to the audio device.

    Architecture overview
    ---------------------
    A single KPipeline runs in a background thread and yields audio chunks sentence-by-sentence.
    Each chunk is appended to a shared buffer, then _play_streaming starts consuming that buffer as soon
    as the first chunk is available — so the user hears audio almost immediately
    rather than waiting for the full text to be synthesised first.
    """
    import numpy as np
    from kokoro import KPipeline

    pipeline = KPipeline(lang_code=lang, repo_id="hexgrad/Kokoro-82M")

    # audio_buffer[0] starts empty and grows as the generation thread appends chunks.
    # Using a list lets the generation thread replace the reference (list[0] = new_array)
    # which is an atomic operation under CPython's GIL — the playback callback can
    # always read audio_buffer[0] and get a valid complete array.
    audio_buffer = [np.zeros(0, dtype=np.float32)]

    def generate_audio():
        """Run the TTS pipeline and append each audio chunk to the shared buffer."""
        for _graphemes, _phonemes, audio_chunk in pipeline(text, voice=voice, speed=speed):
            if quit_requested.is_set():
                break
            # Extend the buffer by creating a new concatenated array and swapping
            # the reference. The old array is garbage-collected by Python once no
            # other thread holds a reference to it.
            audio_buffer[0] = np.concatenate([audio_buffer[0], audio_chunk])
        generation_done.set()

    generation_thread = threading.Thread(target=generate_audio, daemon=True)
    generation_thread.start()

    # Animate a spinner until the first audio chunk is ready.
    spinner_idx = 0
    while len(audio_buffer[0]) == 0:
        frame = SPINNER_FRAMES[spinner_idx % len(SPINNER_FRAMES)]
        sys.stderr.write(f"\r{frame} Generating...")
        sys.stderr.flush()
        spinner_idx += 1
        time.sleep(0.08)

    sys.stderr.write("\r\033[K")
    sys.stderr.flush()

    _play_streaming(audio_buffer)

    # Ensure the generation thread has finished before returning, even if the
    # user quit playback early (the daemon flag means it won't block process exit).
    generation_thread.join()


def _config_path():
    return pathlib.Path.home() / ".config" / "readthis" / "config.json"



def _save_config(updates):
    path = _config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    current = {}
    if path.exists():
        with open(path) as f:
            current = json.load(f)
    current.update(updates)
    with open(path, "w") as f:
        json.dump(current, f, indent=2)
    print(current)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "config":
        parser = argparse.ArgumentParser(
            prog="readthis config", description="Read or write config.json settings")
        parser.add_argument("--voice", choices=VOICES, help="Set default voice")
        parser.add_argument("--speed", type=float, help="Set default speech speed multiplier")
        args = parser.parse_args(sys.argv[2:])
        updates = {}
        if args.voice is not None:
            updates["voice"] = args.voice
        if args.speed is not None:
            updates["speed"] = args.speed
        _save_config(updates)
        return

    parser = argparse.ArgumentParser(description="Text-to-speech using Kokoro")
    parser.add_argument("input", nargs="?", default=None,
                        help="Text to speak, URL to an article, or omit to read from clipboard")
    parser.add_argument("--voice", default=_config.get("voice", "af_heart"), choices=VOICES,
                        help="Voice name (default: af_heart, or set in config.json)")
    parser.add_argument("--speed", type=float, default=_config.get("speed", 1.0),
                        help="Speech speed multiplier (default: 1.0, or set in config.json)")
    parser.add_argument("--lang", default=_config.get("lang", "a"),
                        help="Language code (default: a)")
    args = parser.parse_args()

    text = get_text(args.input)
    speak(text, voice=args.voice, speed=args.speed, lang=args.lang)


if __name__ == "__main__":
    main()
