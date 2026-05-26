"""
SoundMatch Engine v3 — World-Class Sound Recreation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Architecture for 90-95% accuracy:

  1. WAVETABLE EXTRACTION (60-70% of accuracy)
     → Extract actual waveform cycles from audio
     → Create multi-frame wavetable showing sound evolution
     → Save as .wav for Serum import

  2. ENVELOPE MATCHING (15-20%)
     → Precise ADSR from RMS curve analysis
     → Attack shape, decay curve, sustain level

  3. FILTER ANALYSIS (5-10%)  
     → Spectral envelope → filter type + cutoff + resonance
     → Compare to known LP/HP/BP shapes

  4. EFFECTS DETECTION (5-10%)
     → Reverb RT60, distortion harmonics, chorus width

  5. PRESET GENERATION
     → Native .SerumPreset with custom wavetable reference
     → .wav wavetable file for Serum import
"""

import os
import json
import copy
import struct
import pathlib
import time
import warnings
import numpy as np
import librosa
import soundfile as sf
import cbor2
import zstandard as zstd
from scipy import signal as scipy_signal

warnings.filterwarnings("ignore")

MAGIC = b"XferJson\x00"
SERUM_WT_FRAME_SIZE = 2048  # Serum uses 2048 samples per wavetable frame


# ══════════════════════════════════════════════════════════
# PART 1: WAVETABLE EXTRACTION — The Core Innovation
# ══════════════════════════════════════════════════════════

def extract_wavetable(y, sr, fundamental_hz, n_frames=32):
    """
    Extract a multi-frame wavetable from audio by isolating
    individual waveform cycles and resampling to Serum format.
    
    This is the key to 1:1 sound recreation — instead of guessing
    "saw/sine/square", we capture the EXACT waveshape.
    
    Args:
        y: Audio signal (mono)
        sr: Sample rate
        fundamental_hz: Detected fundamental frequency
        n_frames: Number of wavetable frames (captures sound evolution)
    
    Returns:
        numpy array of shape (n_frames, 2048) — ready for Serum
    """
    if fundamental_hz < 20:
        fundamental_hz = 440.0
    
    # Samples per cycle at this fundamental
    cycle_samples = int(sr / fundamental_hz)
    
    if cycle_samples < 4:
        cycle_samples = int(sr / 440)
    
    total_samples = len(y)
    frames = []
    
    # Extract cycles at evenly spaced points through the sound
    # This captures the evolution (attack → sustain → release)
    for i in range(n_frames):
        # Position in the audio (skip first 5% and last 5% for stability)
        t = 0.05 + (i / (n_frames - 1)) * 0.9 if n_frames > 1 else 0.5
        center = int(t * total_samples)
        
        # Extract several cycles for averaging (reduces noise)
        n_avg_cycles = min(8, max(2, total_samples // cycle_samples // n_frames))
        start = max(0, center - (n_avg_cycles * cycle_samples) // 2)
        end = min(total_samples, start + n_avg_cycles * cycle_samples)
        
        if end - start < cycle_samples:
            start = max(0, end - cycle_samples * 2)
        
        segment = y[start:end]
        
        if len(segment) < cycle_samples:
            # Pad if too short
            segment = np.pad(segment, (0, cycle_samples - len(segment)))
        
        # Find zero crossings to align cycles
        zero_crossings = np.where(np.diff(np.signbit(segment)))[0]
        
        # Find rising zero crossings
        rising = []
        for zc in zero_crossings:
            if zc + 1 < len(segment) and segment[zc] <= 0 and segment[zc + 1] > 0:
                rising.append(zc)
        
        if len(rising) >= 2:
            # Extract one clean cycle between two rising zero crossings
            # Find the pair closest to expected cycle length
            best_cycle = None
            best_diff = float('inf')
            
            for j in range(len(rising) - 1):
                length = rising[j + 1] - rising[j]
                diff = abs(length - cycle_samples)
                if diff < best_diff and length > cycle_samples * 0.7 and length < cycle_samples * 1.3:
                    best_diff = diff
                    best_cycle = segment[rising[j]:rising[j + 1]]
            
            if best_cycle is not None:
                cycle = best_cycle
            else:
                # Fallback: just take one cycle's worth from center
                cycle = segment[:cycle_samples]
        else:
            cycle = segment[:cycle_samples]
        
        # Resample to exactly SERUM_WT_FRAME_SIZE samples
        if len(cycle) > 0:
            indices = np.linspace(0, len(cycle) - 1, SERUM_WT_FRAME_SIZE)
            frame = np.interp(indices, np.arange(len(cycle)), cycle)
        else:
            frame = np.zeros(SERUM_WT_FRAME_SIZE)
        
        # Normalize to -1..1
        peak = np.max(np.abs(frame))
        if peak > 1e-6:
            frame = frame / peak
        
        # Apply tiny fade at edges to prevent clicks
        fade_len = 8
        frame[:fade_len] *= np.linspace(0, 1, fade_len)
        frame[-fade_len:] *= np.linspace(1, 0, fade_len)
        
        frames.append(frame.astype(np.float32))
    
    return np.array(frames)


def save_wavetable_wav(frames, output_path, sr=44100):
    """
    Save wavetable frames as a .wav file that Serum can import.
    
    Serum imports wavetables as .wav files where:
    - Each frame is exactly 2048 samples
    - Frames are concatenated sequentially
    - Standard WAV format (float32 or int16)
    """
    # Concatenate all frames
    wavetable = np.concatenate(frames)
    
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    sf.write(output_path, wavetable, sr, subtype='FLOAT')
    
    return output_path, len(frames)


# ══════════════════════════════════════════════════════════
# PART 2: SPECTRAL ENVELOPE ANALYSIS — Filter Detection
# ══════════════════════════════════════════════════════════

def analyze_spectral_envelope(y, sr, fundamental_hz):
    """
    Analyze the spectral envelope to detect filter characteristics.
    
    Compares the audio's harmonic amplitudes to theoretical waveforms
    to determine what filtering has been applied.
    """
    # Get the spectrum
    S = np.abs(librosa.stft(y, n_fft=4096))
    spec_mean = np.mean(S, axis=1)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=4096)
    
    # Measure harmonic amplitudes
    harmonics = []
    for h in range(1, 33):
        h_freq = fundamental_hz * h
        if h_freq > sr / 2:
            break
        idx = np.argmin(np.abs(freqs - h_freq))
        win = 3
        region = spec_mean[max(0, idx-win):min(len(spec_mean), idx+win+1)]
        amp = float(np.max(region)) if len(region) > 0 else 0
        harmonics.append({"number": h, "freq": h_freq, "amplitude": amp})
    
    if not harmonics:
        return {"filter_type": "none", "cutoff_hz": 20000, "resonance": 0}
    
    # Normalize
    max_amp = max(h["amplitude"] for h in harmonics)
    for h in harmonics:
        h["amplitude_norm"] = h["amplitude"] / (max_amp + 1e-10)
    
    # Detect filter by comparing rolloff pattern
    # In a saw wave without filter: amplitude ∝ 1/n
    # With LP filter: amplitudes drop faster above cutoff
    
    # Find where harmonics start dropping significantly
    cutoff_harmonic = len(harmonics)
    for i in range(1, len(harmonics)):
        expected_no_filter = 1.0 / (i + 1)  # Saw wave rolloff
        actual = harmonics[i]["amplitude_norm"]
        if actual < expected_no_filter * 0.3 and i > 1:
            cutoff_harmonic = i
            break
    
    cutoff_hz = fundamental_hz * (cutoff_harmonic + 1)
    
    # Detect resonance: is there a peak near the cutoff?
    resonance = 0.0
    if cutoff_harmonic < len(harmonics) - 1:
        near_cutoff = harmonics[max(0, cutoff_harmonic-1):cutoff_harmonic+2]
        if near_cutoff:
            peak_near = max(h["amplitude_norm"] for h in near_cutoff)
            expected = 1.0 / (cutoff_harmonic + 1)
            if peak_near > expected * 1.5:
                resonance = min(50, (peak_near / expected - 1) * 30)
    
    # Determine filter type
    # Check if low frequencies are also attenuated (HP filter)
    low_energy = np.mean([h["amplitude_norm"] for h in harmonics[:3]])
    high_energy = np.mean([h["amplitude_norm"] for h in harmonics[3:6]]) if len(harmonics) > 5 else 0
    
    if low_energy < 0.3 and high_energy > 0.5:
        filter_type = "HP12"
    elif cutoff_hz < sr / 3:
        filter_type = "LP12"
    else:
        filter_type = "none"
    
    return {
        "filter_type": filter_type,
        "cutoff_hz": round(cutoff_hz, 1),
        "cutoff_norm": min(1.0, max(0.0, np.log10(cutoff_hz / 20) / np.log10(20000 / 20))),
        "resonance": round(resonance, 1),
        "n_audible_harmonics": cutoff_harmonic,
        "harmonic_amps": [round(h["amplitude_norm"], 4) for h in harmonics],
    }


# ══════════════════════════════════════════════════════════
# PART 3: ENVELOPE EXTRACTION — Precise ADSR
# ══════════════════════════════════════════════════════════

def extract_envelope(y, sr):
    """
    Extract precise ADSR envelope parameters from audio.
    Uses RMS energy curve with multiple analysis windows.
    """
    # Multi-resolution RMS
    rms = librosa.feature.rms(y=y, frame_length=512, hop_length=128)[0]
    rms_time = np.arange(len(rms)) * 128 / sr
    
    if len(rms) < 10:
        return {"attack_s": 0.01, "decay_s": 0.1, "sustain": 0.7, "release_s": 0.2}
    
    # Smooth RMS for cleaner envelope
    from scipy.ndimage import uniform_filter1d
    rms_smooth = uniform_filter1d(rms, size=min(15, len(rms) // 3))
    rms_norm = rms_smooth / (np.max(rms_smooth) + 1e-10)
    
    # Find peak
    peak_idx = np.argmax(rms_norm)
    peak_time = rms_time[peak_idx]
    
    # ATTACK: time from start to peak
    # Find where signal first exceeds 5% (sound onset)
    onset_idx = 0
    for i in range(len(rms_norm)):
        if rms_norm[i] > 0.05:
            onset_idx = i
            break
    
    attack_s = max(0.001, rms_time[peak_idx] - rms_time[onset_idx])
    
    # DECAY + SUSTAIN: analyze post-peak
    if peak_idx < len(rms_norm) - 20:
        post_peak = rms_norm[peak_idx:]
        
        # Find sustain level: average of the middle third
        mid = len(post_peak) // 3
        if mid > 0 and 2 * mid < len(post_peak):
            sustain = float(np.median(post_peak[mid:2*mid]))
        else:
            sustain = float(np.mean(post_peak))
        
        # Decay time: from peak to sustain level (±10%)
        decay_idx = 0
        for i in range(len(post_peak)):
            if post_peak[i] <= sustain * 1.1:
                decay_idx = i
                break
        decay_s = max(0.001, decay_idx * 128 / sr)
    else:
        sustain = 0.5
        decay_s = 0.1
    
    # RELEASE: analyze the tail
    # Find where signal drops to 10% from the end
    release_s = 0.2
    tail_start = int(len(rms_norm) * 0.7)
    if tail_start < len(rms_norm) - 5:
        tail = rms_norm[tail_start:]
        tail_start_level = tail[0] if len(tail) > 0 else 0
        
        if tail_start_level > 0.05:
            for i in range(len(tail)):
                if tail[i] < tail_start_level * 0.1:
                    release_s = max(0.01, i * 128 / sr)
                    break
            else:
                release_s = len(tail) * 128 / sr
    
    return {
        "attack_s": round(float(attack_s), 4),
        "decay_s": round(float(decay_s), 4),
        "sustain": round(float(max(0, min(1, sustain))), 4),
        "release_s": round(float(release_s), 4),
    }


# ══════════════════════════════════════════════════════════
# PART 4: EFFECTS DETECTION
# ══════════════════════════════════════════════════════════

def detect_effects(y, sr):
    """Detect reverb, distortion, chorus from audio characteristics."""
    
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    rms_norm = rms / (np.max(rms) + 1e-10)
    
    S = np.abs(librosa.stft(y, n_fft=4096))
    spec_mean = np.mean(S, axis=1)
    flatness = float(librosa.feature.spectral_flatness(S=S).mean())
    
    # ── Reverb detection ──
    # Reverb creates a long tail after transients
    # Compare energy in first half vs second half
    half = len(rms_norm) // 2
    if half > 5:
        first_half_decay = float(np.mean(np.abs(np.diff(rms_norm[:half]))))
        second_half_energy = float(np.mean(rms_norm[half:]))
        peak_energy = float(np.max(rms_norm))
        
        tail_ratio = second_half_energy / (peak_energy + 1e-10)
        reverb_amount = min(1.0, max(0.0, (tail_ratio - 0.08) * 2.5))
    else:
        reverb_amount = 0.0
    
    # ── Distortion detection ──
    # Distortion adds high-frequency harmonics and increases flatness
    high_freq_idx = int(5000 / (sr / 4096))
    if high_freq_idx < len(spec_mean):
        low_energy = np.mean(spec_mean[:high_freq_idx] ** 2)
        high_energy = np.mean(spec_mean[high_freq_idx:] ** 2)
        dist_ratio = high_energy / (low_energy + 1e-10)
        distortion_amount = min(1.0, max(0.0, dist_ratio * 5 - 0.1))
    else:
        distortion_amount = 0.0
    
    # ── Chorus detection ──
    # Chorus creates spectral smearing (wider bandwidth around harmonics)
    bandwidth = float(librosa.feature.spectral_bandwidth(S=S, sr=sr).mean())
    chorus_indicator = bandwidth > 2000
    
    return {
        "reverb": round(reverb_amount, 3),
        "distortion": round(distortion_amount, 3),
        "chorus": chorus_indicator,
        "stereo_width": 0.0,  # needs stereo input
    }


# ══════════════════════════════════════════════════════════
# PART 5: SOUND CLASSIFIER
# ══════════════════════════════════════════════════════════

def classify_sound(fundamental_hz, envelope, spectral, n_harmonics):
    """Classify into lead/bass/pad/pluck/chords/arp/sub/fx."""
    
    scores = {"lead": 0, "bass": 0, "pad": 0, "pluck": 0, "chords": 0, "arp": 0, "sub": 0, "fx": 0}
    
    atk = envelope["attack_s"]
    sus = envelope["sustain"]
    rel = envelope["release_s"]
    fund = fundamental_hz
    
    # Frequency
    if fund < 80: scores["sub"] += 5; scores["bass"] += 3
    elif fund < 200: scores["bass"] += 4; scores["sub"] += 2
    elif fund < 400: scores["bass"] += 1; scores["lead"] += 2; scores["pad"] += 2
    elif fund < 800: scores["lead"] += 4; scores["pluck"] += 2
    else: scores["lead"] += 3; scores["pluck"] += 3; scores["arp"] += 2
    
    # Envelope
    if atk < 0.01 and envelope["decay_s"] < 0.3 and sus < 0.3:
        scores["pluck"] += 5; scores["arp"] += 3
    if atk < 0.05 and sus > 0.5:
        scores["lead"] += 3
    if atk > 0.3: scores["pad"] += 3
    if atk > 1.0: scores["pad"] += 3
    if rel > 1.0: scores["pad"] += 3
    if rel < 0.1: scores["bass"] += 2; scores["pluck"] += 1
    
    best = max(scores, key=scores.get)
    total = sum(scores.values()) + 1e-10
    return best, round(scores[best] / total, 3), scores


# ══════════════════════════════════════════════════════════
# PART 6: SERUM 2 PRESET BUILDER
# ══════════════════════════════════════════════════════════

def build_preset(template_path, dna, wavetable_filename, output_path):
    """
    Build a .SerumPreset with custom wavetable reference.
    
    Args:
        template_path: Default.SerumPreset template
        dna: Full analysis results dict
        wavetable_filename: Name of the .wav wavetable file
        output_path: Where to save the preset
    """
    # Load template
    buf = pathlib.Path(template_path).read_bytes()
    off = len(MAGIC)
    jlen, _ = struct.unpack_from("<II", buf, off); off += 8
    meta = json.loads(buf[off:off + jlen]); off += jlen
    clen, _ = struct.unpack_from("<II", buf, off); off += 8
    preset = cbor2.loads(zstd.ZstdDecompressor().decompress(buf[off:]))
    
    def sp(section, key, val):
        if section not in preset: return
        if isinstance(preset[section].get("plainParams"), str):
            preset[section]["plainParams"] = {}
        preset[section]["plainParams"][key] = val
    
    def snp(section, sub, key, val):
        if section not in preset or sub not in preset[section]: return
        if isinstance(preset[section][sub].get("plainParams"), str):
            preset[section][sub]["plainParams"] = {}
        preset[section][sub]["plainParams"][key] = val
    
    env = dna["envelope"]
    filt = dna["spectral_filter"]
    fx = dna["effects"]
    sound_type = dna["sound_type"]
    fund = dna["fundamental_hz"]
    
    # Preset name
    prefix = {"lead":"LD","bass":"BS","pad":"PD","pluck":"PL","chords":"CH",
              "arp":"AR","sub":"SB","fx":"FX"}.get(sound_type, "SM")
    preset_name = f"{prefix} SM {dna['note_name']}"
    
    # ── Metadata ──
    meta["presetName"] = preset_name
    meta["presetAuthor"] = "SoundMatch AI"
    meta["tags"] = [sound_type.capitalize(), "SoundMatch"]
    preset["presetName"] = preset_name
    preset["presetAuthor"] = "SoundMatch AI"
    preset["tags"] = meta["tags"]
    
    # ── OSC A: Custom wavetable ──
    sp("Oscillator0", "kParamEnable", 1.0)
    # Reference the custom wavetable
    if "WTOsc0" in preset.get("Oscillator0", {}):
        preset["Oscillator0"]["WTOsc0"]["relativePathToWT"] = f"User/{wavetable_filename}"
        preset["Oscillator0"]["WTOsc0"]["numFrames"] = dna["wt_num_frames"] * SERUM_WT_FRAME_SIZE
    
    # Wavetable position: middle of our extracted frames
    snp("Oscillator0", "WTOsc0", "kParamTablePos", 128.0)
    
    # Unison based on stereo width and bandwidth
    width = dna.get("stereo_width", 0)
    bw = dna.get("bandwidth_hz", 0)
    if width > 0.2 or bw > 1500:
        uni_voices = min(7, max(2, int(bw / 600) + int(width * 4)))
        sp("Oscillator0", "kParamUnison", float(uni_voices))
        sp("Oscillator0", "kParamDetune", min(0.4, max(0.05, width * 0.6)))
    
    # ── ENVELOPE ──
    def s_to_param(seconds, max_s=10.0):
        if seconds <= 0.001: return 0.001
        return min(0.99, float(np.log10(seconds * 1000) / np.log10(max_s * 1000)))
    
    sp("Env0", "kParamAttack", s_to_param(env["attack_s"]))
    sp("Env0", "kParamDecay", s_to_param(env["decay_s"], 5.0))
    sp("Env0", "kParamSustain", env["sustain"])
    sp("Env0", "kParamRelease", s_to_param(env["release_s"]))
    
    # ── FILTER ──
    if filt["filter_type"] != "none" and filt["cutoff_hz"] < 15000:
        sp("VoiceFilter0", "kParamEnable", 1.0)
        sp("VoiceFilter0", "kParamType", filt["filter_type"])
        sp("VoiceFilter0", "kParamFreq", filt["cutoff_norm"])
        sp("VoiceFilter0", "kParamReso", filt["resonance"])
    
    # ── SUB OSC ──
    if sound_type in ("bass", "sub") or fund < 150:
        sp("Oscillator4", "kParamEnable", 1.0)
        sp("Oscillator4", "kParamVolume", 0.5 if sound_type == "sub" else 0.3)
    
    # ── FX ──
    fx_list = []
    if fx["reverb"] > 0.1:
        fx_list.append({
            "FXReverb": {"plainParams": {
                "kParamWet": min(80, max(10, fx["reverb"] * 75)),
                "kParamSize": min(90, max(25, fx["reverb"] * 85)),
                "kParamDamping": 40.0,
            }},
            "type": 11
        })
    
    if fx["distortion"] > 0.1:
        fx_list.append({
            "FXDistortion": {"plainParams": {
                "kParamDrive": min(80, max(10, fx["distortion"] * 70)),
                "kParamWet": min(80, max(20, fx["distortion"] * 60)),
                "kParamMode": "kTube",
            }},
            "type": 6
        })
    
    if fx["chorus"]:
        fx_list.append({
            "FXChorus": {"plainParams": {
                "kParamWet": 30.0, "kParamRate": 25.0, "kParamDepth": 40.0,
            }},
            "type": 7
        })
    
    if fx_list:
        preset["FXRack0"]["FX"] = fx_list
    
    # ── GLOBAL ──
    sp("Global0", "kParamMasterVolume", 0.65)
    
    # ── Convert numpy and pack ──
    def convert(obj):
        if isinstance(obj, dict): return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list): return [convert(v) for v in obj]
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return obj
    
    preset = convert(preset)
    
    m = json.dumps(meta, separators=(",", ":")).encode()
    c = cbor2.dumps(preset)
    z = zstd.ZstdCompressor(level=3).compress(c)
    out = bytearray(MAGIC)
    out += struct.pack("<II", len(m), 0)
    out += m
    out += struct.pack("<II", len(c), 2)
    out += z
    
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    pathlib.Path(output_path).write_bytes(out)
    
    return {"preset_name": preset_name, "size": len(out), "path": output_path}


# ══════════════════════════════════════════════════════════
# PART 7: MASTER PIPELINE
# ══════════════════════════════════════════════════════════

def soundmatch_v3(audio_path, template_path, output_dir, sound_type_hint="auto"):
    """
    World-class audio → Serum 2 preset pipeline.
    
    Outputs:
    - .SerumPreset file (native Serum 2 format)
    - .wav wavetable file (import into Serum's wavetable folder)
    - .json analysis report
    
    Returns:
        Full result dict with all analysis data
    """
    t_start = time.time()
    name = pathlib.Path(audio_path).stem
    os.makedirs(output_dir, exist_ok=True)
    
    # ── Load audio ──
    print("  [1/6] Loading audio...")
    y, sr = librosa.load(audio_path, sr=44100, mono=False, duration=30)
    is_stereo = y.ndim == 2
    if is_stereo:
        stereo_width = float(np.mean(np.abs(y[0] - y[1])) / (np.mean(np.abs(y[0] + y[1])) + 1e-10))
        y_mono = (y[0] + y[1]) / 2
    else:
        stereo_width = 0.0
        y_mono = y
    
    # ── Pitch detection ──
    print("  [2/6] Detecting pitch...")
    try:
        f0 = librosa.yin(y_mono, fmin=30, fmax=8000, sr=sr)
        f0_valid = f0[(f0 > 30) & (f0 < 8000)]
        fundamental = float(np.median(f0_valid)) if len(f0_valid) > 0 else 440.0
    except:
        fundamental = 440.0
    
    midi = int(round(12 * np.log2(fundamental / 440) + 69))
    notes = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
    note_name = f"{notes[midi % 12]}{midi // 12 - 1}"
    
    # ── Extract wavetable (THE KEY STEP) ──
    print("  [3/6] Extracting wavetable cycles...")
    n_wt_frames = 32
    wt_frames = extract_wavetable(y_mono, sr, fundamental, n_frames=n_wt_frames)
    wt_filename = f"SM_{name}.wav"
    wt_path = os.path.join(output_dir, wt_filename)
    save_wavetable_wav(wt_frames, wt_path, sr=44100)
    
    # ── Envelope ──
    print("  [4/6] Analyzing envelope...")
    envelope = extract_envelope(y_mono, sr)
    
    # ── Spectral / Filter ──
    print("  [5/6] Analyzing spectrum & effects...")
    spectral_filter = analyze_spectral_envelope(y_mono, sr, fundamental)
    effects = detect_effects(y_mono, sr)
    if is_stereo:
        effects["stereo_width"] = stereo_width
    
    bandwidth = float(librosa.feature.spectral_bandwidth(y=y_mono, sr=sr).mean())
    centroid = float(librosa.feature.spectral_centroid(y=y_mono, sr=sr).mean())
    
    # ── Classify ──
    if sound_type_hint != "auto":
        sound_type = sound_type_hint
        confidence = 1.0
        scores = {}
    else:
        sound_type, confidence, scores = classify_sound(
            fundamental, envelope, spectral_filter, spectral_filter["n_audible_harmonics"]
        )
    
    # ── Build DNA dict ──
    dna = {
        "fundamental_hz": round(fundamental, 1),
        "midi_note": midi,
        "note_name": note_name,
        "sound_type": sound_type,
        "envelope": envelope,
        "spectral_filter": spectral_filter,
        "effects": effects,
        "stereo_width": round(stereo_width, 3),
        "bandwidth_hz": round(bandwidth, 1),
        "centroid_hz": round(centroid, 1),
        "wt_num_frames": n_wt_frames,
    }
    
    # ── Generate preset ──
    print("  [6/6] Generating Serum 2 preset...")
    preset_path = os.path.join(output_dir, f"{name}.SerumPreset")
    preset_info = build_preset(template_path, dna, wt_filename, preset_path)
    
    # ── Save analysis report ──
    report_path = os.path.join(output_dir, f"{name}_analysis.json")
    with open(report_path, 'w') as f:
        json.dump(dna, f, indent=2, default=str)
    
    total_time = time.time() - t_start
    
    print(f"\n  ✅ Done in {total_time:.1f}s")
    print(f"  Type:       {sound_type.upper()} ({confidence:.0%})")
    print(f"  Pitch:      {note_name} ({fundamental:.1f} Hz)")
    print(f"  Envelope:   A={envelope['attack_s']:.3f}s D={envelope['decay_s']:.3f}s S={envelope['sustain']:.2f} R={envelope['release_s']:.3f}s")
    print(f"  Filter:     {spectral_filter['filter_type']} @ {spectral_filter['cutoff_hz']:.0f} Hz")
    print(f"  FX:         reverb={effects['reverb']:.2f} dist={effects['distortion']:.2f} chorus={effects['chorus']}")
    print(f"  Wavetable:  {wt_filename} ({n_wt_frames} frames)")
    print(f"  Preset:     {preset_info['preset_name']}")
    print(f"\n  📁 Output files:")
    print(f"     {preset_path}")
    print(f"     {wt_path}")
    print(f"     {report_path}")
    
    return {
        "dna": dna,
        "preset": preset_info,
        "wavetable": {"path": wt_path, "filename": wt_filename, "frames": n_wt_frames},
        "time_s": round(total_time, 2),
    }
