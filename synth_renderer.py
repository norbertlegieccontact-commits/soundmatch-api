"""
SoundMatch Synth Renderer
━━━━━━━━━━━━━━━━━━━━━━━━━━
Approximates Serum's signal chain in pure Python/numpy.
Used internally to render presets and compare against the target audio.

Not meant to be sample-perfect — just close enough that spectral comparison
gives Claude useful feedback for refinement.
"""

import numpy as np
from scipy import signal


SR = 44100  # sample rate
DURATION = 2.0  # render duration in seconds


# ==============================================================
# WAVEFORM GENERATORS (Serum-like)
# ==============================================================

def gen_saw(freq, duration, sr=SR):
    """Anti-aliased sawtooth."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Use a simple bandlimited approach: sum of sine harmonics up to Nyquist
    out = np.zeros_like(t)
    n_harmonics = int((sr / 2) / freq)
    for h in range(1, min(n_harmonics, 30) + 1):
        out += (1.0 / h) * np.sin(2 * np.pi * freq * h * t)
    return (2.0 / np.pi) * out

def gen_square(freq, duration, sr=SR):
    """Anti-aliased square."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    out = np.zeros_like(t)
    n_harmonics = int((sr / 2) / freq)
    for h in range(1, min(n_harmonics, 30) + 1, 2):
        out += (1.0 / h) * np.sin(2 * np.pi * freq * h * t)
    return (4.0 / np.pi) * out

def gen_triangle(freq, duration, sr=SR):
    """Bandlimited triangle."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    out = np.zeros_like(t)
    n_harmonics = int((sr / 2) / freq)
    for h in range(1, min(n_harmonics, 30) + 1, 2):
        out += ((-1) ** ((h - 1) // 2)) / (h ** 2) * np.sin(2 * np.pi * freq * h * t)
    return (8.0 / (np.pi ** 2)) * out

def gen_sine(freq, duration, sr=SR):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return np.sin(2 * np.pi * freq * t)


WAVEFORM_FNS = {
    "saw": gen_saw,
    "square": gen_square,
    "triangle": gen_triangle,
    "sine": gen_sine,
    "wavetable": gen_saw,  # fallback to saw for wavetable
}


# ==============================================================
# OSCILLATOR WITH UNISON
# ==============================================================

def render_oscillator(osc_params, base_freq, duration=DURATION, sr=SR):
    """Render an oscillator with unison voices and detune."""
    
    if not osc_params.get("enabled", True):
        return np.zeros(int(sr * duration))
    
    waveform = osc_params.get("waveform", "saw")
    octave = osc_params.get("octave", 0)
    detune = osc_params.get("detune", 0.0)
    voices = osc_params.get("unison_voices", 1)
    voice_detune = osc_params.get("unison_detune", 0.0)
    gain = osc_params.get("gain", 0.8)
    
    # Apply octave + fine detune
    freq = base_freq * (2 ** octave) * (1 + detune * 0.05)
    
    gen_fn = WAVEFORM_FNS.get(waveform, gen_saw)
    
    if voices == 1:
        out = gen_fn(freq, duration, sr)
    else:
        # Unison: render multiple slightly detuned voices
        out = np.zeros(int(sr * duration))
        max_cents = voice_detune * 50  # max 50 cents detune at full
        for v in range(voices):
            # Spread voices around center
            cents = (v - (voices - 1) / 2) * max_cents / (voices - 1) if voices > 1 else 0
            voice_freq = freq * (2 ** (cents / 1200))
            out += gen_fn(voice_freq, duration, sr)
        out /= voices  # normalize
    
    return out * gain


# ==============================================================
# FILTER
# ==============================================================

def apply_filter(audio, filter_params, sr=SR):
    """Apply low-pass / high-pass / bandpass filter."""
    
    if not filter_params.get("enabled", False):
        return audio
    
    cutoff_norm = filter_params.get("cutoff", 0.7)  # 0..1
    resonance = filter_params.get("resonance", 0.0)
    f_type = filter_params.get("type", "LP12")
    
    # Map 0..1 cutoff to log-scale 20Hz..18kHz
    cutoff_hz = 20 * (900 ** cutoff_norm)
    cutoff_hz = max(20, min(cutoff_hz, sr / 2 - 1))
    
    # Q factor from resonance (0.5 = no resonance, up to ~10 = self-oscillation)
    q = 0.5 + resonance * 9.5
    
    # Determine filter order
    order = 4 if "24" in f_type else 2
    
    # Filter type
    if "LP" in f_type:
        btype = "lowpass"
    elif "HP" in f_type:
        btype = "highpass"
    elif "BP" in f_type:
        btype = "bandpass"
    elif "Notch" in f_type:
        btype = "bandstop"
    else:
        btype = "lowpass"
    
    # Design with butter for stability; resonance via Q approximation
    if btype in ("bandpass", "bandstop"):
        # For band filters, width depends on Q
        bw = cutoff_hz / q
        low = max(20, cutoff_hz - bw / 2)
        high = min(sr / 2 - 1, cutoff_hz + bw / 2)
        if low < high:
            sos = signal.butter(order, [low, high], btype=btype, fs=sr, output="sos")
        else:
            return audio
    else:
        sos = signal.butter(order, cutoff_hz, btype=btype, fs=sr, output="sos")
    
    filtered = signal.sosfilt(sos, audio)
    
    # Add resonance via peaking filter near cutoff
    if resonance > 0.1:
        peak_sos = signal.iirpeak(cutoff_hz, q * 2, fs=sr)
        peak_sos = signal.tf2sos(peak_sos[0], peak_sos[1])
        peak_signal = signal.sosfilt(peak_sos, audio) * resonance * 0.5
        filtered = filtered + peak_signal
    
    return filtered


# ==============================================================
# ADSR ENVELOPE
# ==============================================================

def apply_adsr(audio, env_params, duration=DURATION, sr=SR):
    """Apply ADSR amplitude envelope."""
    
    attack = env_params.get("attack", 0.0)
    decay = env_params.get("decay", 0.3)
    sustain = env_params.get("sustain", 0.7)
    release = env_params.get("release", 0.3)
    
    # Map 0..1 to actual times (log scale)
    # 0 -> 0.001s, 0.5 -> ~0.3s, 1.0 -> ~10s
    a_time = 0.001 + (attack ** 2) * 9.999
    d_time = 0.001 + (decay ** 2) * 4.999
    r_time = 0.001 + (release ** 2) * 4.999
    
    total_samples = int(sr * duration)
    env = np.zeros(total_samples)
    
    a_samples = min(int(a_time * sr), total_samples)
    d_samples = min(int(d_time * sr), total_samples - a_samples)
    
    # Note-on portion: 70% of duration
    note_on_samples = int(total_samples * 0.7)
    
    # Attack (linear ramp 0 -> 1)
    if a_samples > 0:
        env[:a_samples] = np.linspace(0, 1, a_samples)
    
    # Decay (exp 1 -> sustain)
    if d_samples > 0 and a_samples + d_samples <= note_on_samples:
        env[a_samples:a_samples + d_samples] = np.linspace(1, sustain, d_samples)
    
    # Sustain
    sus_start = a_samples + d_samples
    if sus_start < note_on_samples:
        env[sus_start:note_on_samples] = sustain
    
    # Release (exp sustain -> 0)
    r_samples = min(int(r_time * sr), total_samples - note_on_samples)
    if r_samples > 0:
        env[note_on_samples:note_on_samples + r_samples] = np.linspace(sustain, 0, r_samples)
    
    return audio * env


# ==============================================================
# EFFECTS
# ==============================================================

def apply_reverb(audio, params, sr=SR):
    """Simple reverb via convolution with exponential decay."""
    if not params.get("enabled", False):
        return audio
    
    wet = params.get("wet", 0.3)
    size = params.get("size", 0.5)
    damping = params.get("damping", 0.5)
    
    # IR length depends on size
    ir_seconds = 0.3 + size * 3.7  # 0.3s to 4s
    ir_samples = int(ir_seconds * sr)
    
    # Generate exponentially decaying noise as IR
    np.random.seed(42)  # deterministic
    ir = np.random.randn(ir_samples) * np.exp(-np.linspace(0, 5 * (1 + damping), ir_samples))
    ir = ir / np.max(np.abs(ir) + 1e-9) * 0.5
    
    # Low-pass IR based on damping
    damp_cutoff = 20000 * (1 - damping * 0.8)
    if damp_cutoff < sr / 2:
        sos = signal.butter(2, damp_cutoff, btype="lowpass", fs=sr, output="sos")
        ir = signal.sosfilt(sos, ir)
    
    # Convolve (use FFT for speed)
    reverbed = signal.fftconvolve(audio, ir, mode="full")[:len(audio)]
    
    return audio * (1 - wet) + reverbed * wet


def apply_distortion(audio, params):
    """Soft-clipping distortion."""
    if not params.get("enabled", False):
        return audio
    
    drive = params.get("drive", 0.3)
    wet = params.get("wet", 0.5)
    
    drive_amount = 1 + drive * 20
    distorted = np.tanh(audio * drive_amount) / drive_amount * 1.5
    
    return audio * (1 - wet) + distorted * wet


def apply_chorus(audio, params, sr=SR):
    """Simple chorus via modulated delay."""
    if not params.get("enabled", False):
        return audio
    
    depth = params.get("depth", 0.5)
    wet = params.get("wet", 0.5)
    rate = 1.5  # Hz, fixed
    
    n = len(audio)
    t = np.arange(n) / sr
    delay_ms = 15 + 10 * depth * np.sin(2 * np.pi * rate * t)
    delay_samples = (delay_ms * sr / 1000).astype(int)
    
    chorused = np.zeros_like(audio)
    for i in range(n):
        src = i - delay_samples[i]
        if 0 <= src < n:
            chorused[i] = audio[src]
    
    return audio * (1 - wet) + chorused * wet


def apply_delay(audio, params, sr=SR):
    """Simple delay with feedback."""
    if not params.get("enabled", False):
        return audio
    
    wet = params.get("wet", 0.3)
    feedback = min(params.get("feedback", 0.4), 0.85)
    delay_ms = 250  # default 1/4 note at 120 BPM
    
    delay_samples = int(delay_ms * sr / 1000)
    out = audio.copy()
    
    for i in range(delay_samples, len(out)):
        out[i] += out[i - delay_samples] * feedback
    
    return audio * (1 - wet) + out * wet


# ==============================================================
# MAIN RENDER FUNCTION
# ==============================================================

def render_preset(params: dict, freq: float = 440.0, duration: float = DURATION, sr: int = SR):
    """Render a preset to audio. Returns numpy array."""
    
    # Render oscillators
    audio_a = render_oscillator(params.get("oscillator_a", {}), freq, duration, sr)
    audio_b = render_oscillator(params.get("oscillator_b", {"enabled": False}), freq, duration, sr)
    
    audio = audio_a + audio_b
    audio = audio / (np.max(np.abs(audio)) + 1e-9) * 0.8
    
    # Apply filter
    audio = apply_filter(audio, params.get("filter", {}), sr)
    
    # Apply amp envelope
    audio = apply_adsr(audio, params["envelope_amp"], duration, sr)
    
    # Apply FX in series
    fx = params.get("fx", {})
    audio = apply_distortion(audio, fx.get("distortion", {}))
    audio = apply_chorus(audio, fx.get("chorus", {}), sr)
    audio = apply_delay(audio, fx.get("delay", {}), sr)
    audio = apply_reverb(audio, fx.get("reverb", {}), sr)
    
    # Final normalize
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.85
    
    return audio
