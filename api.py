"""
SoundMatch API Backend
━━━━━━━━━━━━━━━━━━━━━━
FastAPI server that receives audio files,
analyzes them, and returns Serum 2 presets.

Run: uvicorn api:app --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import tempfile
import shutil
import os
import json
import uuid
import time
import pathlib
import struct
import copy
import warnings
import numpy as np
import librosa
import cbor2
import zstandard as zstd

warnings.filterwarnings("ignore")

app = FastAPI(title="SoundMatch API", version="3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MAGIC = b"XferJson\x00"
TEMPLATE_PATH = os.environ.get("SERUM_TEMPLATE", "Default.SerumPreset")
OUTPUT_DIR = "generated"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Import engine components ──
from engine_v3 import (
    extract_wavetable, save_wavetable_wav,
    extract_envelope, analyze_spectral_envelope,
    detect_effects, classify_sound,
    SERUM_WT_FRAME_SIZE,
)


def analyze_and_generate(audio_path, job_id):
    """Full pipeline: audio → analysis → preset."""
    t0 = time.time()

    # Load
    y, sr = librosa.load(audio_path, sr=44100, mono=False, duration=30)
    is_stereo = y.ndim == 2
    y_mono = (y[0] + y[1]) / 2 if is_stereo else y
    stereo_width = float(np.mean(np.abs(y[0]-y[1])) / (np.mean(np.abs(y[0]+y[1]))+1e-10)) if is_stereo else 0

    # Pitch
    f0 = librosa.yin(y_mono, fmin=30, fmax=8000, sr=sr)
    f0v = f0[(f0 > 30) & (f0 < 8000)]
    fund = float(np.median(f0v)) if len(f0v) > 0 else 440.0
    midi = int(round(12 * np.log2(fund / 440) + 69))
    notes = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
    note_name = f"{notes[midi%12]}{midi//12-1}"

    # Analysis
    env = extract_envelope(y_mono, sr)
    spec = analyze_spectral_envelope(y_mono, sr, fund)
    fx = detect_effects(y_mono, sr)
    bw = float(librosa.feature.spectral_bandwidth(y=y_mono, sr=sr).mean())
    centroid = float(librosa.feature.spectral_centroid(y=y_mono, sr=sr).mean())
    n_harm = spec["n_audible_harmonics"]

    # Classify
    stype, conf, scores = classify_sound(fund, env, spec, n_harm)

    # Wavetable extraction
    wt_frames = extract_wavetable(y_mono, sr, fund, n_frames=32)
    wt_path = os.path.join(OUTPUT_DIR, f"{job_id}_wavetable.wav")
    save_wavetable_wav(wt_frames, wt_path)

    # ── Build preset (standalone, no custom WT reference) ──
    buf = pathlib.Path(TEMPLATE_PATH).read_bytes()
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

    # Name
    prefix = {"lead":"LD","bass":"BS","pad":"PD","pluck":"PL","sub":"SB","chords":"CH","arp":"AR"}.get(stype,"SM")
    preset_name = f"{prefix} SM {note_name}"
    meta["presetName"] = preset_name
    meta["presetAuthor"] = "SoundMatch AI"
    meta["tags"] = [stype.capitalize(), "SoundMatch"]
    preset["presetName"] = preset_name
    preset["presetAuthor"] = "SoundMatch AI"
    preset["tags"] = meta["tags"]

    # WT Position
    if n_harm <= 2: wt_pos = 5.0
    elif n_harm <= 5: wt_pos = 50.0
    elif n_harm <= 10: wt_pos = 100.0
    elif n_harm <= 20: wt_pos = 140.0
    else: wt_pos = 170.0

    sp("Oscillator0", "kParamEnable", 1.0)
    snp("Oscillator0", "WTOsc0", "kParamTablePos", wt_pos)

    if stereo_width > 0.15 or bw > 1500:
        uni = min(7.0, max(2.0, float(int(bw/500) + int(stereo_width*4))))
        sp("Oscillator0", "kParamUnison", uni)
        sp("Oscillator0", "kParamDetune", min(0.35, max(0.05, stereo_width * 0.5)))

    if stype == "pad" and n_harm > 5:
        sp("Oscillator1", "kParamEnable", 1.0)
        sp("Oscillator1", "kParamOctave", 1.0)
        sp("Oscillator1", "kParamDetune", 0.15)
        snp("Oscillator1", "WTOsc1", "kParamTablePos", max(0, wt_pos - 30))

    if stype in ("bass", "sub") or fund < 150:
        sp("Oscillator4", "kParamEnable", 1.0)
        sp("Oscillator4", "kParamVolume", 0.6)

    def s2p(s, max_s=10.0):
        if s <= 0.001: return 0.001
        return min(0.99, float(np.log10(s*1000) / np.log10(max_s*1000)))

    sp("Env0", "kParamAttack", s2p(env["attack_s"]))
    sp("Env0", "kParamDecay", s2p(env["decay_s"], 5.0))
    sp("Env0", "kParamSustain", float(env["sustain"]))
    sp("Env0", "kParamRelease", s2p(env["release_s"]))

    if spec["filter_type"] != "none" and spec["cutoff_hz"] < 15000:
        sp("VoiceFilter0", "kParamEnable", 1.0)
        sp("VoiceFilter0", "kParamType", spec["filter_type"])
        sp("VoiceFilter0", "kParamFreq", float(spec["cutoff_norm"]))
        sp("VoiceFilter0", "kParamReso", float(spec["resonance"]))

    fx_list = []
    if fx["reverb"] > 0.1:
        fx_list.append({"FXReverb": {"plainParams": {
            "kParamWet": min(80.0, fx["reverb"]*75),
            "kParamSize": min(90.0, fx["reverb"]*85),
            "kParamDamping": 40.0,
        }}, "type": 11})
    if fx_list:
        preset["FXRack0"]["FX"] = fx_list

    sp("Global0", "kParamMasterVolume", 0.65)

    # Convert numpy
    def conv(o):
        if isinstance(o, dict): return {k: conv(v) for k, v in o.items()}
        if isinstance(o, list): return [conv(v) for v in o]
        if hasattr(o, 'item'): return o.item()
        if hasattr(o, 'tolist'): return o.tolist()
        return o
    preset = conv(preset)

    # Pack
    m_bytes = json.dumps(meta, separators=(",",":")).encode()
    c = cbor2.dumps(preset)
    z = zstd.ZstdCompressor(level=3).compress(c)
    out = bytearray(MAGIC) + struct.pack("<II", len(m_bytes), 0) + m_bytes + struct.pack("<II", len(c), 2) + z

    preset_path = os.path.join(OUTPUT_DIR, f"{job_id}.SerumPreset")
    pathlib.Path(preset_path).write_bytes(out)

    elapsed = time.time() - t0

    return {
        "job_id": job_id,
        "preset_name": preset_name,
        "preset_path": preset_path,
        "wavetable_path": wt_path,
        "sound_type": stype,
        "confidence": round(conf * 100),
        "note_name": note_name,
        "fundamental_hz": round(fund, 1),
        "envelope": {k: round(float(v), 4) for k, v in env.items()},
        "filter": {"type": spec["filter_type"], "cutoff_hz": round(spec["cutoff_hz"]), "resonance": round(spec["resonance"], 1)},
        "effects": {k: round(float(v), 3) if isinstance(v, (float, np.floating)) else v for k, v in fx.items()},
        "n_harmonics": n_harm,
        "stereo_width": round(stereo_width, 3),
        "preset_size": len(out),
        "time_s": round(elapsed, 2),
    }


@app.post("/api/analyze")
async def analyze_audio(file: UploadFile = File(...)):
    """Upload audio file → get analysis + preset."""
    if not file.filename.lower().endswith(('.wav', '.mp3', '.flac', '.ogg', '.aiff')):
        raise HTTPException(400, "Supported formats: WAV, MP3, FLAC, OGG, AIFF")

    job_id = str(uuid.uuid4())[:8]

    with tempfile.NamedTemporaryFile(suffix=os.path.splitext(file.filename)[1], delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        result = analyze_and_generate(tmp_path, job_id)
        return JSONResponse(result)
    except Exception as e:
        raise HTTPException(500, f"Analysis failed: {str(e)}")
    finally:
        os.unlink(tmp_path)


@app.get("/api/download/{job_id}")
async def download_preset(job_id: str):
    """Download generated .SerumPreset file."""
    path = os.path.join(OUTPUT_DIR, f"{job_id}.SerumPreset")
    if not os.path.exists(path):
        raise HTTPException(404, "Preset not found")
    return FileResponse(path, filename=f"SoundMatch_{job_id}.SerumPreset",
                       media_type="application/octet-stream")


@app.get("/api/download/{job_id}/wavetable")
async def download_wavetable(job_id: str):
    """Download extracted wavetable .wav file."""
    path = os.path.join(OUTPUT_DIR, f"{job_id}_wavetable.wav")
    if not os.path.exists(path):
        raise HTTPException(404, "Wavetable not found")
    return FileResponse(path, filename=f"SM_{job_id}.wav",
                       media_type="audio/wav")


@app.get("/api/health")
async def health():
    return {"status": "ok", "engine": "v3", "template": os.path.exists(TEMPLATE_PATH)}
