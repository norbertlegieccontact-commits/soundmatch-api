"""
SoundMatch Dataset Builder
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Tool to prepare training data from preset packs.

Workflow:
1. Put .SerumPreset files in input/ folder
2. Run this script
3. For each preset, it:
   - Extracts the parameters from the .SerumPreset binary
   - Renders an audio preview using our internal synth
   - Saves: dataset/[name].wav + dataset/[name].json

Output is ready for training a neural network OR for adding as
few-shot examples in the AI engine.

USAGE:
    python build_dataset.py --input ./my_presets/ --output ./dataset/
"""

import os
import json
import struct
import argparse
import pathlib
from typing import Optional

import numpy as np
import soundfile as sf
import cbor2
import zstandard as zstd


def extract_preset(preset_path: str) -> Optional[dict]:
    """Extract metadata + parameters from .SerumPreset binary."""
    
    with open(preset_path, "rb") as f:
        data = f.read()
    
    # Check magic
    if data[:9] != b"XferJson\0":
        print(f"  ! Skipping {preset_path} - not a valid SerumPreset")
        return None
    
    try:
        json_len = struct.unpack_from("<I", data, 9)[0]
        json_start = 17
        json_bytes = data[json_start:json_start + json_len]
        meta = json.loads(json_bytes)
        
        cbor_offset = json_start + json_len
        cbor_compressed = data[cbor_offset + 8:]
        
        dctx = zstd.ZstdDecompressor()
        cbor_data = dctx.decompress(cbor_compressed)
        preset_dict = cbor2.loads(cbor_data)
        
        return {
            "meta": meta,
            "preset_dict": preset_dict,
        }
    except Exception as e:
        print(f"  ! Failed to parse {preset_path}: {e}")
        return None


def preset_to_claude_format(preset_data: dict, sound_type: str = "lead") -> dict:
    """Convert parsed Serum preset into Claude-friendly parameter format.
    
    This extracts key parameters from the CBOR dict and creates the same
    JSON structure that our AI engine uses.
    """
    pd = preset_data["preset_dict"]
    meta = preset_data["meta"]
    
    def get_block_params(block_name):
        block = pd.get(block_name, {})
        if not isinstance(block, dict):
            return {}
        plain = block.get("plainParams", {})
        if plain == "default":
            return {}
        return plain
    
    osc0 = get_block_params("Oscillator0")
    osc1 = get_block_params("Oscillator1")
    filt0 = get_block_params("VoiceFilter0")
    env0 = get_block_params("Env0")
    env1 = get_block_params("Env1")
    fx0 = get_block_params("FXRack0")
    fx1 = get_block_params("FXRack1")
    fx2 = get_block_params("FXRack2")
    
    # Map waveform from table position (rough guess)
    table_pos = osc0.get("kParamTablePos", 0)
    if isinstance(table_pos, float) and table_pos < 0.25:
        waveform = "saw"
    elif table_pos < 0.5:
        waveform = "square"
    elif table_pos < 0.75:
        waveform = "triangle"
    else:
        waveform = "sine"
    
    return {
        "preset_name": meta.get("presetName", "Untitled"),
        "sound_type": sound_type,
        "oscillator_a": {
            "waveform": waveform,
            "octave": int(osc0.get("kParamOctave", 0)),
            "detune": float(osc0.get("kParamDetune", 0.0)),
            "unison_voices": int(osc0.get("kParamUnisonVoices", 1)),
            "unison_detune": float(osc0.get("kParamUnisonDetune", 0.0)),
            "gain": float(osc0.get("kParamGain", 0.8)),
            "table_pos": int(table_pos * 256) if isinstance(table_pos, float) else 0,
            "warp_type": osc0.get("kParamWarpMenu", "kPD_OSC"),
        },
        "oscillator_b": {
            "enabled": bool(osc1.get("kParamEnable", False)),
            "waveform": waveform,
            "octave": int(osc1.get("kParamOctave", 0)),
            "detune": float(osc1.get("kParamDetune", 0.0)),
            "gain": float(osc1.get("kParamGain", 0.5)),
        },
        "filter": {
            "enabled": bool(filt0.get("kParamEnable", True)),
            "type": filt0.get("kParamType", "LP12"),
            "cutoff": float(filt0.get("kParamFreq", 0.7)),
            "resonance": float(filt0.get("kParamReso", 0.2)),
            "drive": float(filt0.get("kParamDrive", 0.0)),
        },
        "envelope_amp": {
            "attack": float(env0.get("kParamAttack", 0.0)),
            "decay": float(env0.get("kParamDecay", 0.3)),
            "sustain": float(env0.get("kParamSustain", 0.7)),
            "release": float(env0.get("kParamRelease", 0.3)),
        },
        "envelope_filter": {
            "enabled": bool(env1) and len(env1) > 0,
            "attack": float(env1.get("kParamAttack", 0.0)),
            "decay": float(env1.get("kParamDecay", 0.2)),
            "sustain": float(env1.get("kParamSustain", 0.0)),
            "release": float(env1.get("kParamRelease", 0.2)),
            "amount": 0.3,
        },
        "fx": {
            "reverb": {
                "enabled": bool(fx0.get("kParamEnable", False)),
                "wet": float(fx0.get("kParamWet", 0.0)),
                "size": float(fx0.get("kParamSize", 0.5)),
                "damping": float(fx0.get("kParamDamping", 0.5)),
            },
            "distortion": {
                "enabled": bool(fx1.get("kParamEnable", False)),
                "drive": float(fx1.get("kParamDrive", 0.0)),
                "wet": float(fx1.get("kParamWet", 0.5)),
            },
            "delay": {
                "enabled": bool(fx2.get("kParamEnable", False)),
                "wet": float(fx2.get("kParamWet", 0.0)),
                "feedback": float(fx2.get("kParamFeedback", 0.4)),
            },
        },
    }


def guess_sound_type(preset_name: str) -> str:
    """Guess sound type from preset name."""
    name = preset_name.upper()
    if name.startswith("LD") or "LEAD" in name:
        return "lead"
    if name.startswith("BS") or "BASS" in name or "SUB" in name:
        return "bass"
    if name.startswith("PD") or "PAD" in name:
        return "pad"
    if name.startswith("PL") or "PLUCK" in name:
        return "pluck"
    if name.startswith("CH") or "CHORD" in name:
        return "chords"
    if name.startswith("AR") or "ARP" in name:
        return "arp"
    return "lead"  # default


def render_preset_to_audio(params: dict, output_path: str, freq: float = 440.0):
    """Render preset to a 2-second audio file."""
    try:
        from synth_renderer import render_preset
        audio = render_preset(params, freq=freq, duration=2.0)
        sf.write(output_path, audio, 44100)
        return True
    except Exception as e:
        print(f"  ! Render failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Build training dataset from Serum presets")
    parser.add_argument("--input", required=True, help="Folder with .SerumPreset files")
    parser.add_argument("--output", required=True, help="Output dataset folder")
    parser.add_argument("--limit", type=int, default=0, help="Max presets to process (0=all)")
    args = parser.parse_args()
    
    input_dir = pathlib.Path(args.input)
    output_dir = pathlib.Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    preset_files = list(input_dir.rglob("*.SerumPreset"))
    if args.limit > 0:
        preset_files = preset_files[:args.limit]
    
    print(f"Found {len(preset_files)} presets to process")
    
    successful = 0
    failed = 0
    pairs = []
    
    for i, preset_path in enumerate(preset_files, 1):
        name = preset_path.stem
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
        print(f"[{i}/{len(preset_files)}] {name}")
        
        # Extract preset
        preset_data = extract_preset(str(preset_path))
        if not preset_data:
            failed += 1
            continue
        
        # Guess sound type from name
        sound_type = guess_sound_type(preset_data["meta"].get("presetName", name))
        
        # Convert to Claude format
        params = preset_to_claude_format(preset_data, sound_type)
        
        # Save params JSON
        json_path = output_dir / f"{safe_name}.json"
        with open(json_path, "w") as f:
            json.dump(params, f, indent=2)
        
        # Render audio
        wav_path = output_dir / f"{safe_name}.wav"
        if render_preset_to_audio(params, str(wav_path)):
            pairs.append({
                "name": name,
                "sound_type": sound_type,
                "params_file": str(json_path.name),
                "audio_file": str(wav_path.name),
            })
            successful += 1
        else:
            failed += 1
    
    # Save manifest
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump({
            "total": len(preset_files),
            "successful": successful,
            "failed": failed,
            "pairs": pairs,
        }, f, indent=2)
    
    print()
    print(f"Done! {successful} pairs, {failed} failed")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
