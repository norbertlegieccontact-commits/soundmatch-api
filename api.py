"""
SoundMatch API v3 — Level 2 + Feedback Loop
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Claude AI generates preset
- Renders via internal Python synth
- Compares spectrum vs target
- Sends diff back to Claude for refinement
- Repeats until similarity >= threshold or max iterations
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import tempfile, shutil, os, json, uuid, time, pathlib, warnings
import numpy as np
import librosa
import soundfile as sf

warnings.filterwarnings("ignore")

app = FastAPI(title="SoundMatch API", version="3.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

TEMPLATE_PATH = os.environ.get("SERUM_TEMPLATE", "Default.SerumPreset")
GENERATED_DIR = pathlib.Path("generated")
GENERATED_DIR.mkdir(exist_ok=True)

# Feature flags via env vars
USE_FEEDBACK_LOOP = os.environ.get("USE_FEEDBACK_LOOP", "true").lower() == "true"
MAX_ITERATIONS = int(os.environ.get("MAX_ITERATIONS", "2"))
SIMILARITY_TARGET = float(os.environ.get("SIMILARITY_TARGET", "0.85"))


def analyze_audio(audio_path: str, sound_type: str = "lead") -> tuple:
    """Deep spectral analysis. Returns (features_dict, raw_audio)."""
    y, sr = librosa.load(audio_path, sr=44100, mono=True, duration=10.0)
    
    # Pitch detection
    f0, voiced, _ = librosa.pyin(y, fmin=librosa.note_to_hz("C1"), fmax=librosa.note_to_hz("C8"))
    voiced_f0 = f0[voiced] if voiced is not None else np.array([])
    voiced_f0 = voiced_f0[~np.isnan(voiced_f0)] if len(voiced_f0) > 0 else np.array([])
    
    if len(voiced_f0) > 0:
        median_f0 = float(np.median(voiced_f0[voiced_f0 > 0])) if any(voiced_f0 > 0) else 440.0
        try:
            pitch = librosa.hz_to_note(median_f0)
        except Exception:
            pitch = "A4"
    else:
        median_f0 = 440.0
        pitch = "A4"
    
    # Spectral features
    spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    spectral_rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
    spectral_flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)))
    rms = float(np.mean(librosa.feature.rms(y=y)))
    
    # Harmonics
    harmonic, _ = librosa.effects.hpss(y)
    stft = np.abs(librosa.stft(harmonic))
    freqs = librosa.fft_frequencies(sr=sr)
    
    harmonics = []
    if median_f0 > 0:
        fundamental_bin = np.argmin(np.abs(freqs - median_f0))
        fundamental_amp = float(np.mean(stft[fundamental_bin, :]))
        for i in range(1, 9):
            harmonic_freq = median_f0 * i
            if harmonic_freq < sr / 2:
                bin_idx = np.argmin(np.abs(freqs - harmonic_freq))
                harmonic_amp = float(np.mean(stft[bin_idx, :]))
                ratio = harmonic_amp / (fundamental_amp + 1e-10)
                harmonics.append(round(ratio, 3))
    
    if not harmonics:
        harmonics = [1.0, 0.5, 0.25, 0.12, 0.06]
    
    # Envelope
    envelope = np.abs(y)
    smooth_win = int(sr * 0.01)
    env_smooth = np.convolve(envelope, np.ones(smooth_win) / smooth_win, mode='same')
    env_norm = env_smooth / (np.max(env_smooth) + 1e-9)
    
    peak_idx = int(np.argmax(env_norm))
    attack_time = float(peak_idx / sr)
    
    if peak_idx < len(env_norm) - 1:
        after_peak = env_norm[peak_idx:]
        decay_target = 0.37  # 1/e
        decay_indices = np.where(after_peak < decay_target)[0]
        decay_time = float(decay_indices[0] / sr) if len(decay_indices) > 0 else float(len(after_peak) / sr)
        sustain_level = float(np.mean(after_peak[len(after_peak) // 2:])) if len(after_peak) > 100 else 0.5
    else:
        decay_time = 0.1
        sustain_level = 0.7
    
    release_time = float(len(y[peak_idx:]) / sr) * 0.3
    
    brightness = min(1.0, spectral_centroid / 8000.0)
    
    odd_sum = sum(harmonics[i] for i in range(0, len(harmonics), 2))
    even_sum = sum(harmonics[i] for i in range(1, len(harmonics), 2)) + 0.001
    roughness = min(1.0, odd_sum / (even_sum * 2))
    
    noisiness = min(1.0, float(spectral_flatness) * 10)
    
    if len(voiced_f0) > 10:
        f0_std = float(np.std(voiced_f0[voiced_f0 > 0]))
        has_vibrato = f0_std > median_f0 * 0.01
    else:
        has_vibrato = False
    
    features = {
        "pitch": pitch,
        "fundamental_hz": round(median_f0, 1),
        "spectral_centroid": round(spectral_centroid, 1),
        "spectral_rolloff": round(spectral_rolloff, 1),
        "harmonics": harmonics[:8],
        "attack_time": round(attack_time, 4),
        "decay_time": round(decay_time, 4),
        "sustain_level": round(sustain_level, 3),
        "release_time": round(release_time, 4),
        "rms": round(rms, 4),
        "brightness": round(brightness, 3),
        "roughness": round(roughness, 3),
        "noisiness": round(noisiness, 3),
        "has_vibrato": has_vibrato,
        "sound_type": sound_type,
    }
    
    return features, y, median_f0


@app.get("/api/health")
def health():
    has_claude = bool(os.environ.get("ANTHROPIC_API_KEY"))
    return {
        "status": "healthy",
        "engine": "Level 2 + Feedback Loop" if (has_claude and USE_FEEDBACK_LOOP) else ("Level 2 - Claude AI" if has_claude else "Level 1 - Rules"),
        "claude_enabled": has_claude,
        "feedback_loop": USE_FEEDBACK_LOOP and has_claude,
        "max_iterations": MAX_ITERATIONS,
        "template_exists": os.path.exists(TEMPLATE_PATH),
    }


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...), sound_type: str = Form("lead")):
    job_id = str(uuid.uuid4())[:8]
    tmp_dir = pathlib.Path(tempfile.mkdtemp())
    
    log_steps = []
    
    try:
        # 1. Save upload
        audio_path = tmp_dir / f"input{pathlib.Path(file.filename).suffix}"
        with open(audio_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        # 2. Deep audio analysis
        log_steps.append({"step": "analysis", "status": "running"})
        analysis, target_audio, target_freq = analyze_audio(str(audio_path), sound_type)
        log_steps[-1]["status"] = "done"
        
        # 3. Get Claude params (or fall back to rules)
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        claude_params = None
        best_params = None
        best_similarity = 0.0
        iterations_used = 0
        engine_used = "rules"
        
        if api_key:
            try:
                from ai_engine import analyze_with_claude
                log_steps.append({"step": "claude_v1", "status": "running"})
                claude_params = analyze_with_claude(analysis, sound_type)
                log_steps[-1]["status"] = "done"
                best_params = claude_params
                engine_used = "claude-ai"
                
                # 4. Feedback loop (if enabled)
                if USE_FEEDBACK_LOOP and target_audio is not None:
                    try:
                        from synth_renderer import render_preset
                        from spectral_diff import compare_audio
                        
                        for iteration in range(MAX_ITERATIONS):
                            log_steps.append({"step": f"render_v{iteration+1}", "status": "running"})
                            rendered = render_preset(claude_params, freq=target_freq, duration=2.0)
                            log_steps[-1]["status"] = "done"
                            
                            log_steps.append({"step": f"compare_v{iteration+1}", "status": "running"})
                            # Trim target to match rendered length
                            tgt_trimmed = target_audio[:len(rendered)] if len(target_audio) > len(rendered) else np.pad(target_audio, (0, len(rendered) - len(target_audio)))
                            diff_text, similarity = compare_audio(tgt_trimmed, rendered)
                            log_steps[-1]["status"] = "done"
                            log_steps[-1]["similarity"] = round(similarity, 3)
                            
                            if similarity > best_similarity:
                                best_similarity = similarity
                                best_params = claude_params
                            
                            iterations_used = iteration + 1
                            
                            if similarity >= SIMILARITY_TARGET:
                                break
                            
                            if iteration < MAX_ITERATIONS - 1 and diff_text:
                                # Refine
                                log_steps.append({"step": f"claude_v{iteration+2}", "status": "running"})
                                refined = analyze_with_claude(analysis, sound_type, claude_params, diff_text)
                                claude_params = refined
                                log_steps[-1]["status"] = "done"
                        
                        engine_used = f"claude-ai+feedback ({iterations_used} iter)"
                    except Exception as e:
                        print(f"Feedback loop error: {e}")
                        log_steps.append({"step": "feedback_loop", "status": "failed", "error": str(e)[:100]})
            except Exception as e:
                print(f"Claude failed: {e}")
                log_steps.append({"step": "claude_v1", "status": "failed", "error": str(e)[:100]})
        
        # 5. Build .SerumPreset file
        log_steps.append({"step": "build_preset", "status": "running"})
        
        if not best_params:
            # Fallback: simple rules-based mapping (no Claude)
            a = analysis
            brightness = a.get("brightness", 0.5)
            best_params = {
                "preset_name": "SM " + sound_type.upper() + " " + a.get("pitch", "A4"),
                "oscillator_a": {
                    "waveform": "saw" if brightness > 0.4 else "triangle",
                    "octave": 0, "detune": 0.2 if sound_type in ("lead", "pad") else 0.0,
                    "unison_voices": 5 if sound_type in ("lead", "pad") else 1,
                    "unison_detune": 0.3 if sound_type in ("lead", "pad") else 0.0,
                    "gain": 0.8, "table_pos": 0, "warp_type": "kPD_OSC"
                },
                "oscillator_b": {"enabled": False},
                "filter": {
                    "enabled": True, "type": "LP24" if sound_type != "bass" else "LP12",
                    "cutoff": min(0.95, 0.3 + brightness * 0.65),
                    "resonance": 0.15, "drive": 0.0
                },
                "envelope_amp": {
                    "attack": min(1.0, a.get("attack_time", 0.01) / 10),
                    "decay": min(1.0, a.get("decay_time", 0.3) / 5),
                    "sustain": a.get("sustain_level", 0.7),
                    "release": min(1.0, a.get("release_time", 0.3) / 5),
                },
                "envelope_filter": {"enabled": False},
                "lfo1": {"enabled": False},
                "fx": {
                    "reverb": {"enabled": sound_type in ("pad", "pluck", "lead"), "wet": 0.3, "size": 0.5, "damping": 0.5},
                    "delay": {"enabled": False},
                    "distortion": {"enabled": a.get("roughness", 0) > 0.5, "drive": 0.3, "wet": 0.5},
                    "chorus": {"enabled": sound_type == "pad", "depth": 0.4, "wet": 0.3}
                },
                "reasoning": "Rules-based fallback (Claude unavailable)"
            }
            engine_used = "rules-fallback"
        
        if os.path.exists(TEMPLATE_PATH):
            from preset_builder import build_preset_from_params
            preset_bytes, preset_name = build_preset_from_params(best_params, TEMPLATE_PATH)
        else:
            raise HTTPException(status_code=500, detail="No template found")
        
        log_steps[-1]["status"] = "done"
        
        # 6. Save preset
        preset_path = GENERATED_DIR / f"{job_id}.SerumPreset"
        with open(preset_path, "wb") as f:
            f.write(preset_bytes)
        
        return {
            "job_id": job_id,
            "preset_name": preset_name,
            "engine": engine_used,
            "iterations": iterations_used,
            "similarity": round(best_similarity, 3) if best_similarity > 0 else None,
            "analysis": analysis,
            "claude_params": best_params,
            "has_wavetable": False,
            "confidence": round(best_similarity, 3) if best_similarity > 0 else (0.85 if engine_used == "claude-ai" else 0.70),
            "log": log_steps,
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)[:300], "log": log_steps},
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.get("/api/download/{job_id}")
def download_preset(job_id: str):
    preset_path = GENERATED_DIR / f"{job_id}.SerumPreset"
    if not preset_path.exists():
        raise HTTPException(status_code=404, detail="Preset not found or expired")
    return FileResponse(
        str(preset_path),
        media_type="application/octet-stream",
        filename=f"SoundMatch_{job_id}.SerumPreset"
    )
