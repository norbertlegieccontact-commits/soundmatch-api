"""
SoundMatch Spectral Comparison & Refinement Loop
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Compares rendered output against target audio and tells Claude what to fix.
"""

import numpy as np
import librosa


def extract_features(audio, sr=44100):
    """Extract spectral features for comparison."""
    
    if len(audio) < sr * 0.1:  # too short
        return None
    
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr)))
    rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr)))
    flatness = float(np.mean(librosa.feature.spectral_flatness(y=audio)))
    rms = float(np.mean(librosa.feature.rms(y=audio)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(audio)))
    
    # MFCC for timbre comparison
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    
    # Envelope shape
    env = np.abs(audio)
    env_smooth = np.convolve(env, np.ones(int(sr * 0.01)) / int(sr * 0.01), mode="same")
    env_norm = env_smooth / (np.max(env_smooth) + 1e-9)
    
    # Find peak position (attack)
    peak_idx = int(np.argmax(env_norm))
    peak_time = peak_idx / sr
    
    # Find decay endpoint
    if peak_idx < len(env_norm) - 1:
        after_peak = env_norm[peak_idx:]
        # Find where envelope drops to 1/e
        decay_targets = np.where(after_peak < 0.37)[0]
        decay_time = (decay_targets[0] if len(decay_targets) else len(after_peak)) / sr
    else:
        decay_time = 0
    
    return {
        "centroid": centroid,
        "rolloff": rolloff,
        "flatness": flatness,
        "rms": rms,
        "zcr": zcr,
        "mfcc_mean": mfcc_mean.tolist(),
        "peak_time": peak_time,
        "decay_time": decay_time,
    }


def compare_audio(target_audio, rendered_audio, sr=44100):
    """Compare two audio signals and return diff description for Claude."""
    
    target_feat = extract_features(target_audio, sr)
    rendered_feat = extract_features(rendered_audio, sr)
    
    if not target_feat or not rendered_feat:
        return None, 0.0
    
    diffs = []
    similarities = []
    
    # Brightness comparison (centroid)
    c_target = target_feat["centroid"]
    c_rendered = rendered_feat["centroid"]
    c_ratio = c_rendered / (c_target + 1)
    if c_ratio > 1.25:
        diffs.append(f"Rendered is TOO BRIGHT ({c_rendered:.0f}Hz vs target {c_target:.0f}Hz). LOWER filter cutoff by ~15-20%.")
    elif c_ratio < 0.8:
        diffs.append(f"Rendered is TOO DARK ({c_rendered:.0f}Hz vs target {c_target:.0f}Hz). RAISE filter cutoff by ~15-20%.")
    similarities.append(1 - min(abs(c_ratio - 1), 1))
    
    # Texture/noisiness
    f_target = target_feat["flatness"]
    f_rendered = rendered_feat["flatness"]
    if f_target > 0.1 and f_rendered < f_target * 0.5:
        diffs.append(f"Target has MORE NOISE/TEXTURE (flatness {f_target:.3f} vs {f_rendered:.3f}). Add distortion or increase unison detune.")
    similarities.append(1 - min(abs(f_target - f_rendered) * 5, 1))
    
    # Attack
    pt_target = target_feat["peak_time"]
    pt_rendered = rendered_feat["peak_time"]
    attack_diff = pt_rendered - pt_target
    if abs(attack_diff) > 0.05:
        if attack_diff > 0:
            diffs.append(f"Rendered attack is TOO SLOW ({pt_rendered:.3f}s vs {pt_target:.3f}s). DECREASE envelope_amp.attack.")
        else:
            diffs.append(f"Rendered attack is TOO FAST ({pt_rendered:.3f}s vs {pt_target:.3f}s). INCREASE envelope_amp.attack.")
    
    # Decay
    dt_target = target_feat["decay_time"]
    dt_rendered = rendered_feat["decay_time"]
    if abs(dt_target - dt_rendered) > 0.1:
        if dt_rendered < dt_target:
            diffs.append(f"Rendered decays TOO QUICKLY ({dt_rendered:.3f}s vs target {dt_target:.3f}s). INCREASE envelope_amp.sustain or decay.")
        else:
            diffs.append(f"Rendered sustains TOO LONG ({dt_rendered:.3f}s vs target {dt_target:.3f}s). DECREASE envelope_amp.sustain or decay.")
    
    # MFCC distance (timbre)
    mfcc_t = np.array(target_feat["mfcc_mean"])
    mfcc_r = np.array(rendered_feat["mfcc_mean"])
    mfcc_dist = np.linalg.norm(mfcc_t - mfcc_r) / np.linalg.norm(mfcc_t + 1)
    mfcc_sim = max(0, 1 - mfcc_dist)
    similarities.append(mfcc_sim)
    
    # RMS energy
    if rendered_feat["rms"] < target_feat["rms"] * 0.5:
        diffs.append("Rendered is QUIETER than target. Increase gain or reduce filter cutoff.")
    
    # Overall similarity score
    overall = float(np.mean(similarities))
    
    diff_text = "\n".join(f"- {d}" for d in diffs) if diffs else "Very close match, minor refinements only."
    
    return diff_text, overall


def refine_preset_iteratively(target_audio, claude_fn, render_fn, max_iters=3, target_sim=0.88):
    """
    Iteratively refine preset using Claude → render → compare → Claude feedback loop.
    
    Args:
        target_audio: numpy array of target audio
        claude_fn: function(analysis, sound_type, prev_params, diff_text) -> params
        render_fn: function(params, freq) -> audio
        max_iters: max refinement iterations
        target_sim: stop when similarity >= this value
    
    Returns:
        (best_params, best_similarity, iterations_used)
    """
    # This is a coordinator function - actual rendering and Claude calls happen elsewhere
    raise NotImplementedError("Use the loop directly in api.py for now")
