"""
SoundMatch AI Engine v3 - Expert Sound Designer
With few-shot examples, validation, and two-stage refinement.
"""

import os
import json
import re

# Lazy import - only loaded when analyze_with_claude is called
# This lets us validate / build presets without anthropic installed


# ==============================================================
# EXPERT SYSTEM PROMPT
# ==============================================================

SYSTEM_PROMPT = """You are a world-class synthesizer sound designer with 25 years of experience programming Serum 2 presets for major artists. You have an exceptional ear for timbre and know exactly how spectral analysis translates to Serum parameters.

You will receive detailed audio analysis and must return EXACT Serum 2 parameter values that recreate the sound.

# KEY INSIGHTS FROM EXPERIENCE

## Waveform Selection (from harmonic content)
- All harmonics present, decreasing: SAW wave
- Only odd harmonics (1, 3, 5, 7): SQUARE wave  
- Strong fundamental, weak harmonics: SINE
- Odd harmonics with rapid rolloff: TRIANGLE
- Complex non-harmonic spectrum: WAVETABLE (Analog, PWM, or Bell)

## Filter Cutoff Mapping (spectral centroid → freq)
- Dark/muffled (< 1000 Hz centroid) → cutoff 0.3-0.45
- Warm (1000-2500 Hz) → cutoff 0.45-0.65
- Bright (2500-5000 Hz) → cutoff 0.65-0.80
- Very bright (5000+ Hz) → cutoff 0.80-0.95

## Filter Resonance (from spectral peakiness)
- Smooth sweep sounds: 0.0-0.15
- Plucky/resonant: 0.25-0.50
- Acid/aggressive: 0.50-0.80

## Envelope Mapping (from attack/decay times)
- Attack time < 0.005s → A=0.0
- Attack time 0.005-0.05s → A=0.05-0.15
- Attack time 0.05-0.3s → A=0.15-0.40
- Attack time > 0.3s → A=0.40-0.80 (pad territory)

## Sound Type Templates (your starting points)

### LEAD
- OSC: Saw or Square, 5-7 unison voices, detune 0.2-0.4
- Filter: LP24, cutoff 0.65-0.85, reso 0.1-0.3
- ENV: A=0.0-0.05, D=0.2-0.4, S=0.6-0.8, R=0.2-0.4
- FX: Reverb 20-30% wet, optional chorus

### BASS
- OSC: Saw or Triangle, 1-2 voices, NO unison detune
- Filter: LP12, cutoff 0.30-0.55, reso 0.1-0.4
- ENV: A=0.0, D=0.3-0.5, S=0.4-0.7, R=0.1-0.2
- FX: Light distortion, no reverb (kills bass)

### PAD
- OSC: Saw + Saw detuned octave up, 7 voices, detune 0.3-0.5
- Filter: LP24, cutoff 0.50-0.75, reso 0.0-0.1
- ENV: A=0.40-0.70 (slow), D=0.3, S=0.8, R=0.5-0.7
- FX: Heavy reverb (50-70%), chorus, optional delay

### PLUCK
- OSC: Triangle or Sine, 1 voice, no detune
- Filter: LP24 with envelope mod, cutoff 0.55, reso 0.2-0.4
- ENV: A=0.0, D=0.10-0.20 (fast decay!), S=0.0-0.1, R=0.15
- Filter ENV: amount 0.3-0.5, D=0.1
- FX: Reverb 30-40%, optional delay for trance pluck

### CHORDS
- OSC: Saw, 3 voices, detune 0.1
- Filter: LP12, cutoff 0.55-0.70, reso 0.05
- ENV: A=0.02, D=0.3, S=0.7, R=0.4
- FX: Reverb 30-40%, chorus 20%

### ARP
- OSC: Square or Saw, 1-2 voices
- Filter: LP24, cutoff 0.60, reso 0.3-0.5
- ENV: A=0.0, D=0.15, S=0.3, R=0.1
- FX: Delay 40% (1/8 dotted), reverb 25%

# OUTPUT FORMAT
Return ONLY valid JSON. No markdown, no explanation outside JSON. Always include "reasoning" field briefly explaining choices.
"""

# ==============================================================
# FEW-SHOT EXAMPLES
# ==============================================================

EXAMPLES = [
    # Example 1: Bright supersaw lead
    {
        "input": {
            "sound_type": "lead",
            "pitch": "A4",
            "fundamental_hz": 440.0,
            "spectral_centroid": 4200,
            "harmonics": [1.0, 0.78, 0.62, 0.51, 0.42, 0.35, 0.29, 0.24],
            "attack_time": 0.003,
            "decay_time": 0.18,
            "sustain_level": 0.78,
            "release_time": 0.35,
            "brightness": 0.78,
            "roughness": 0.35,
            "noisiness": 0.08,
        },
        "output": {
            "preset_name": "LD Bright Supersaw A4",
            "oscillator_a": {
                "waveform": "saw",
                "octave": 0,
                "detune": 0.32,
                "unison_voices": 7,
                "unison_detune": 0.35,
                "gain": 0.85,
                "table_pos": 0,
                "warp_type": "kPD_OSC"
            },
            "oscillator_b": {"enabled": False},
            "filter": {
                "enabled": True,
                "type": "LP24",
                "cutoff": 0.78,
                "resonance": 0.12,
                "drive": 0.15
            },
            "envelope_amp": {
                "attack": 0.02,
                "decay": 0.25,
                "sustain": 0.78,
                "release": 0.35
            },
            "envelope_filter": {"enabled": False},
            "lfo1": {"enabled": False},
            "fx": {
                "reverb": {"enabled": True, "wet": 0.28, "size": 0.55, "damping": 0.4},
                "delay": {"enabled": False},
                "distortion": {"enabled": False},
                "chorus": {"enabled": True, "depth": 0.3, "wet": 0.2}
            },
            "reasoning": "Bright lead with full harmonic content suggests saw wave with heavy unison. Fast attack and high sustain create classic trance/EDM supersaw."
        }
    },
    # Example 2: Deep dark reese bass
    {
        "input": {
            "sound_type": "bass",
            "pitch": "A1",
            "fundamental_hz": 55.0,
            "spectral_centroid": 650,
            "harmonics": [1.0, 0.45, 0.85, 0.30, 0.65, 0.20, 0.40, 0.15],
            "attack_time": 0.002,
            "decay_time": 0.35,
            "sustain_level": 0.65,
            "release_time": 0.18,
            "brightness": 0.15,
            "roughness": 0.75,
            "noisiness": 0.15,
        },
        "output": {
            "preset_name": "BS Dark Reese A1",
            "oscillator_a": {
                "waveform": "saw",
                "octave": 0,
                "detune": 0.15,
                "unison_voices": 2,
                "unison_detune": 0.18,
                "gain": 0.9,
                "table_pos": 0,
                "warp_type": "kPD_OSC"
            },
            "oscillator_b": {
                "enabled": True,
                "waveform": "saw",
                "octave": 0,
                "detune": -0.08,
                "unison_voices": 2,
                "gain": 0.7
            },
            "filter": {
                "enabled": True,
                "type": "LP12",
                "cutoff": 0.35,
                "resonance": 0.18,
                "drive": 0.4
            },
            "envelope_amp": {
                "attack": 0.0,
                "decay": 0.4,
                "sustain": 0.65,
                "release": 0.2
            },
            "envelope_filter": {"enabled": False},
            "lfo1": {
                "enabled": True,
                "rate": 0.25,
                "shape": "sine",
                "target": "filter"
            },
            "fx": {
                "reverb": {"enabled": False},
                "delay": {"enabled": False},
                "distortion": {"enabled": True, "drive": 0.35, "wet": 0.5},
                "chorus": {"enabled": False}
            },
            "reasoning": "Dark bass with strong odd harmonics and high roughness = classic Reese: two detuned saws, heavy LP filter, slow LFO modulation, distortion."
        }
    },
    # Example 3: Lush pad
    {
        "input": {
            "sound_type": "pad",
            "pitch": "C3",
            "fundamental_hz": 130.81,
            "spectral_centroid": 1800,
            "harmonics": [1.0, 0.65, 0.42, 0.28, 0.18, 0.12, 0.08, 0.05],
            "attack_time": 1.2,
            "decay_time": 0.5,
            "sustain_level": 0.88,
            "release_time": 1.8,
            "brightness": 0.45,
            "roughness": 0.15,
            "noisiness": 0.05,
        },
        "output": {
            "preset_name": "PD Lush Strings C3",
            "oscillator_a": {
                "waveform": "saw",
                "octave": 0,
                "detune": 0.4,
                "unison_voices": 7,
                "unison_detune": 0.45,
                "gain": 0.8,
                "table_pos": 0,
                "warp_type": "kPD_OSC"
            },
            "oscillator_b": {
                "enabled": True,
                "waveform": "saw",
                "octave": 1,
                "detune": 0.25,
                "unison_voices": 5,
                "gain": 0.4
            },
            "filter": {
                "enabled": True,
                "type": "LP24",
                "cutoff": 0.62,
                "resonance": 0.05,
                "drive": 0.0
            },
            "envelope_amp": {
                "attack": 0.55,
                "decay": 0.4,
                "sustain": 0.88,
                "release": 0.72
            },
            "envelope_filter": {"enabled": False},
            "lfo1": {
                "enabled": True,
                "rate": 0.15,
                "shape": "sine",
                "target": "pitch"
            },
            "fx": {
                "reverb": {"enabled": True, "wet": 0.6, "size": 0.85, "damping": 0.5},
                "delay": {"enabled": True, "wet": 0.2, "feedback": 0.35},
                "distortion": {"enabled": False},
                "chorus": {"enabled": True, "depth": 0.5, "wet": 0.4}
            },
            "reasoning": "Long attack and release with high sustain = pad. Two layered saw oscillators with octave separation creates wide stereo image. Heavy reverb and chorus for lushness."
        }
    }
]


# ==============================================================
# VALIDATION
# ==============================================================

def clamp(value, lo, hi):
    """Clamp value to [lo, hi] range."""
    try:
        v = float(value)
        return max(lo, min(hi, v))
    except (TypeError, ValueError):
        return lo

def validate_params(params: dict) -> dict:
    """Ensure all parameters are in valid Serum 2 ranges."""
    
    # Oscillator A
    osc_a = params.get("oscillator_a", {})
    osc_a["octave"] = int(clamp(osc_a.get("octave", 0), -4, 4))
    osc_a["detune"] = clamp(osc_a.get("detune", 0.0), 0.0, 1.0)
    osc_a["unison_voices"] = int(clamp(osc_a.get("unison_voices", 1), 1, 16))
    osc_a["unison_detune"] = clamp(osc_a.get("unison_detune", 0.0), 0.0, 1.0)
    osc_a["gain"] = clamp(osc_a.get("gain", 0.8), 0.0, 1.0)
    osc_a["table_pos"] = int(clamp(osc_a.get("table_pos", 0), 0, 256))
    
    valid_warps = ["kPD_OSC", "kSync", "kBend+", "kBend-", "kPWM", "kAsymmetry"]
    if osc_a.get("warp_type") not in valid_warps:
        osc_a["warp_type"] = "kPD_OSC"
    
    valid_waves = ["saw", "square", "sine", "triangle", "wavetable"]
    if osc_a.get("waveform") not in valid_waves:
        osc_a["waveform"] = "saw"
    
    params["oscillator_a"] = osc_a
    
    # Oscillator B (optional)
    osc_b = params.get("oscillator_b", {"enabled": False})
    if osc_b.get("enabled"):
        osc_b["octave"] = int(clamp(osc_b.get("octave", 0), -4, 4))
        osc_b["detune"] = clamp(osc_b.get("detune", 0.0), -1.0, 1.0)
        osc_b["unison_voices"] = int(clamp(osc_b.get("unison_voices", 1), 1, 8))
        osc_b["gain"] = clamp(osc_b.get("gain", 0.5), 0.0, 1.0)
    params["oscillator_b"] = osc_b
    
    # Filter
    filt = params.get("filter", {})
    if filt.get("enabled", True):
        filt["cutoff"] = clamp(filt.get("cutoff", 0.7), 0.0, 1.0)
        filt["resonance"] = clamp(filt.get("resonance", 0.2), 0.0, 1.0)
        filt["drive"] = clamp(filt.get("drive", 0.0), 0.0, 1.0)
        valid_filters = ["LP12", "LP24", "HP12", "HP24", "BP", "Notch"]
        if filt.get("type") not in valid_filters:
            filt["type"] = "LP12"
        filt["enabled"] = True
    params["filter"] = filt
    
    # Envelope amp
    env = params.get("envelope_amp", {})
    env["attack"] = clamp(env.get("attack", 0.0), 0.0, 1.0)
    env["decay"] = clamp(env.get("decay", 0.3), 0.0, 1.0)
    env["sustain"] = clamp(env.get("sustain", 0.7), 0.0, 1.0)
    env["release"] = clamp(env.get("release", 0.3), 0.0, 1.0)
    params["envelope_amp"] = env
    
    # Filter envelope
    env_f = params.get("envelope_filter", {"enabled": False})
    if env_f.get("enabled"):
        env_f["attack"] = clamp(env_f.get("attack", 0.0), 0.0, 1.0)
        env_f["decay"] = clamp(env_f.get("decay", 0.2), 0.0, 1.0)
        env_f["sustain"] = clamp(env_f.get("sustain", 0.0), 0.0, 1.0)
        env_f["release"] = clamp(env_f.get("release", 0.2), 0.0, 1.0)
        env_f["amount"] = clamp(env_f.get("amount", 0.3), -1.0, 1.0)
    params["envelope_filter"] = env_f
    
    # LFO
    lfo = params.get("lfo1", {"enabled": False})
    if lfo.get("enabled"):
        lfo["rate"] = clamp(lfo.get("rate", 0.3), 0.0, 1.0)
        valid_shapes = ["sine", "triangle", "square", "saw"]
        if lfo.get("shape") not in valid_shapes:
            lfo["shape"] = "sine"
        valid_targets = ["pitch", "filter", "volume", "pan"]
        if lfo.get("target") not in valid_targets:
            lfo["target"] = "filter"
    params["lfo1"] = lfo
    
    # FX
    fx = params.get("fx", {})
    for fx_type in ["reverb", "delay", "distortion", "chorus"]:
        f = fx.get(fx_type, {"enabled": False})
        if f.get("enabled"):
            f["wet"] = clamp(f.get("wet", 0.3), 0.0, 1.0)
            if fx_type == "reverb":
                f["size"] = clamp(f.get("size", 0.5), 0.0, 1.0)
                f["damping"] = clamp(f.get("damping", 0.5), 0.0, 1.0)
            elif fx_type == "delay":
                f["feedback"] = clamp(f.get("feedback", 0.4), 0.0, 0.95)
            elif fx_type == "distortion":
                f["drive"] = clamp(f.get("drive", 0.3), 0.0, 1.0)
            elif fx_type == "chorus":
                f["depth"] = clamp(f.get("depth", 0.5), 0.0, 1.0)
        fx[fx_type] = f
    params["fx"] = fx
    
    return params


# ==============================================================
# CLAUDE CALL
# ==============================================================

def analyze_with_claude(audio_analysis: dict, sound_type: str = "lead", 
                        previous_attempt: dict = None, diff_info: str = None) -> dict:
    """Send audio analysis to Claude. Optionally refine based on previous diff."""
    
    import anthropic  # Lazy import
    
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set in environment")
    
    client = anthropic.Anthropic(api_key=api_key)
    
    # Build the user message
    a = audio_analysis
    
    user_msg = f"""Analyze this audio and create a Serum 2 preset.

# AUDIO ANALYSIS
- Sound type: {sound_type}
- Fundamental pitch: {a.get('pitch', '—')} ({a.get('fundamental_hz', 0):.1f} Hz)
- Spectral centroid: {a.get('spectral_centroid', 0):.0f} Hz
- Spectral rolloff: {a.get('spectral_rolloff', 0):.0f} Hz
- Harmonic ratios: {a.get('harmonics', [])[:8]}
- Attack time: {a.get('attack_time', 0):.4f}s
- Decay time: {a.get('decay_time', 0):.4f}s
- Sustain level: {a.get('sustain_level', 0):.3f}
- Release time: {a.get('release_time', 0):.4f}s
- RMS energy: {a.get('rms', 0):.4f}
- Brightness: {a.get('brightness', 0):.3f} (0=dark, 1=bright)
- Roughness: {a.get('roughness', 0):.3f} (high = odd harmonics, distortion)
- Noisiness: {a.get('noisiness', 0):.3f}
- Has vibrato: {a.get('has_vibrato', False)}
"""

    if previous_attempt and diff_info:
        user_msg += f"""
# REFINEMENT MODE
Your previous attempt produced these parameters:
{json.dumps(previous_attempt, indent=2)}

The rendered output differed from the target as follows:
{diff_info}

Adjust the parameters to better match the target. Return the FULL updated JSON.
"""

    user_msg += """
Return the Serum 2 preset as JSON matching this exact schema:
{
  "preset_name": "string (format: 'XX Name PitchNote', e.g. 'LD Bright Saw A4')",
  "oscillator_a": {"waveform": "saw|square|sine|triangle|wavetable", "octave": int, "detune": 0-1, "unison_voices": 1-16, "unison_detune": 0-1, "gain": 0-1, "table_pos": 0-256, "warp_type": "kPD_OSC|kSync|kBend+|kPWM"},
  "oscillator_b": {"enabled": bool, "waveform": "...", "octave": int, "detune": -1 to 1, "unison_voices": 1-8, "gain": 0-1},
  "filter": {"enabled": bool, "type": "LP12|LP24|HP12|BP", "cutoff": 0-1, "resonance": 0-1, "drive": 0-1},
  "envelope_amp": {"attack": 0-1, "decay": 0-1, "sustain": 0-1, "release": 0-1},
  "envelope_filter": {"enabled": bool, "attack": 0-1, "decay": 0-1, "sustain": 0-1, "release": 0-1, "amount": -1 to 1},
  "lfo1": {"enabled": bool, "rate": 0-1, "shape": "sine|triangle|square|saw", "target": "pitch|filter|volume"},
  "fx": {
    "reverb": {"enabled": bool, "wet": 0-1, "size": 0-1, "damping": 0-1},
    "delay": {"enabled": bool, "wet": 0-1, "feedback": 0-1},
    "distortion": {"enabled": bool, "drive": 0-1, "wet": 0-1},
    "chorus": {"enabled": bool, "depth": 0-1, "wet": 0-1}
  },
  "reasoning": "brief explanation"
}

Return ONLY JSON. No markdown wrappers, no preamble."""

    # Build messages with few-shot examples
    messages = []
    for ex in EXAMPLES:
        messages.append({"role": "user", "content": f"# AUDIO ANALYSIS\n{json.dumps(ex['input'], indent=2)}\n\nReturn JSON preset."})
        messages.append({"role": "assistant", "content": json.dumps(ex["output"], indent=2)})
    
    messages.append({"role": "user", "content": user_msg})
    
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        system=SYSTEM_PROMPT,
        messages=messages,
    )
    
    response_text = response.content[0].text.strip()
    
    # Strip any markdown wrappers
    if response_text.startswith("```"):
        match = re.search(r"```(?:json)?\s*(.+?)\s*```", response_text, re.DOTALL)
        if match:
            response_text = match.group(1)
    
    # Find JSON object
    start = response_text.find("{")
    end = response_text.rfind("}")
    if start >= 0 and end > start:
        response_text = response_text[start:end+1]
    
    params = json.loads(response_text)
    params = validate_params(params)
    
    return params


# ==============================================================
# PRESET BUILDER (Claude params → .SerumPreset binary)
# ==============================================================

def build_preset_from_claude_params(params: dict, template_path: str):
    """Convert Claude's parameter JSON into a real .SerumPreset binary."""
    
    import struct
    import zstandard as zstd
    import cbor2
    
    with open(template_path, "rb") as f:
        template_data = f.read()
    
    magic = template_data[:9]  # "XferJson\0"
    json_len = struct.unpack_from("<I", template_data, 9)[0]
    json_start = 17
    json_bytes = template_data[json_start:json_start + json_len]
    meta = json.loads(json_bytes)
    
    cbor_offset = json_start + json_len
    cbor_decompressed_len = struct.unpack_from("<I", template_data, cbor_offset)[0]
    cbor_version = struct.unpack_from("<I", template_data, cbor_offset + 4)[0]
    cbor_data_compressed = template_data[cbor_offset + 8:]
    
    dctx = zstd.ZstdDecompressor()
    cbor_data = dctx.decompress(cbor_data_compressed)
    preset_dict = cbor2.loads(cbor_data)
    
    # Extract from validated params
    osc_a = params["oscillator_a"]
    osc_b = params.get("oscillator_b", {})
    filt = params.get("filter", {})
    env = params["envelope_amp"]
    env_f = params.get("envelope_filter", {})
    lfo = params.get("lfo1", {})
    fx = params.get("fx", {})
    
    # Helper: assign params to a block
    def set_block_params(block_name, plain_params):
        if block_name not in preset_dict:
            return
        block = preset_dict[block_name]
        if not isinstance(block, dict):
            return
        existing = block.get("plainParams", {})
        if existing == "default":
            existing = {}
        existing.update(plain_params)
        block["plainParams"] = existing
    
    # === Oscillator A ===
    osc_a_params = {
        "kParamEnable": True,
        "kParamOctave": float(osc_a["octave"]),
        "kParamDetune": float(osc_a["detune"]),
        "kParamUnisonVoices": int(osc_a["unison_voices"]),
        "kParamUnisonDetune": float(osc_a["unison_detune"]),
        "kParamGain": float(osc_a["gain"]),
        "kParamPan": 0.0,
        "kParamTablePos": float(osc_a["table_pos"]) / 256.0,
    }
    if osc_a.get("warp_type") and osc_a["warp_type"] != "kPD_OSC":
        osc_a_params["kParamWarpMenu"] = osc_a["warp_type"]
    set_block_params("Oscillator0", osc_a_params)
    
    # === Oscillator B ===
    if osc_b.get("enabled"):
        osc_b_params = {
            "kParamEnable": True,
            "kParamOctave": float(osc_b.get("octave", 0)),
            "kParamDetune": float(osc_b.get("detune", 0.0)),
            "kParamUnisonVoices": int(osc_b.get("unison_voices", 1)),
            "kParamGain": float(osc_b.get("gain", 0.5)),
        }
        set_block_params("Oscillator1", osc_b_params)
    
    # === Filter ===
    if filt.get("enabled"):
        filter_params = {
            "kParamEnable": True,
            "kParamFreq": float(filt["cutoff"]),
            "kParamReso": float(filt["resonance"]),
            "kParamType": filt.get("type", "LP12"),
        }
        if filt.get("drive", 0) > 0:
            filter_params["kParamDrive"] = float(filt["drive"])
        set_block_params("VoiceFilter0", filter_params)
    
    # === Amp Envelope ===
    set_block_params("Env0", {
        "kParamAttack": float(env["attack"]),
        "kParamDecay": float(env["decay"]),
        "kParamSustain": float(env["sustain"]),
        "kParamRelease": float(env["release"]),
    })
    
    # === Filter Envelope ===
    if env_f.get("enabled"):
        set_block_params("Env1", {
            "kParamAttack": float(env_f.get("attack", 0.0)),
            "kParamDecay": float(env_f.get("decay", 0.2)),
            "kParamSustain": float(env_f.get("sustain", 0.0)),
            "kParamRelease": float(env_f.get("release", 0.2)),
        })
    
    # === LFO ===
    if lfo.get("enabled"):
        shape_map = {"sine": "Sine", "triangle": "Triangle", "square": "Square", "saw": "Saw"}
        set_block_params("LFO0", {
            "kParamEnable": True,
            "kParamRate": float(lfo.get("rate", 0.3)),
            "kParamMode": "Free",
            "kParamType": shape_map.get(lfo.get("shape"), "Sine"),
        })
    
    # === FX ===
    if fx.get("reverb", {}).get("enabled"):
        r = fx["reverb"]
        set_block_params("FXRack0", {
            "kParamEnable": True,
            "kParamWet": float(r.get("wet", 0.3)),
            "kParamSize": float(r.get("size", 0.5)),
            "kParamDamping": float(r.get("damping", 0.5)),
        })
    
    if fx.get("distortion", {}).get("enabled"):
        d = fx["distortion"]
        set_block_params("FXRack1", {
            "kParamEnable": True,
            "kParamDrive": float(d.get("drive", 0.3)),
            "kParamWet": float(d.get("wet", 0.5)),
        })
    
    if fx.get("delay", {}).get("enabled"):
        dl = fx["delay"]
        set_block_params("FXRack2", {
            "kParamEnable": True,
            "kParamWet": float(dl.get("wet", 0.3)),
            "kParamFeedback": float(dl.get("feedback", 0.4)),
        })
    
    # === Update metadata ===
    preset_name = params.get("preset_name", "SoundMatch AI")
    meta["presetName"] = preset_name
    meta["artist"] = "SoundMatch AI"
    meta["comments"] = params.get("reasoning", "")[:200]
    
    # Repack everything
    new_json = json.dumps(meta, separators=(",", ":")).encode("utf-8")
    new_cbor = cbor2.dumps(preset_dict)
    
    cctx = zstd.ZstdCompressor(level=3)
    new_cbor_compressed = cctx.compress(new_cbor)
    
    result = (
        magic +
        struct.pack("<II", len(new_json), 0) +
        new_json +
        struct.pack("<II", len(new_cbor), cbor_version) +
        new_cbor_compressed
    )
    
    return result, preset_name
