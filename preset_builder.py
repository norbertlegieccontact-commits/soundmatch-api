"""
SoundMatch Preset Builder v2
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Generates valid Serum 2 .SerumPreset files.

Key discoveries:
- Custom CBOR encoder (float32 only, preserve key order)
- hash in JSON = MD5 of compressed CBOR
- FX entries need: {FXClassName: {plainParams:...}, kUIParamMixOrGain: 0.0, type: N}
- Correct param names: kParamUnison (not Voices), kParamVolume (not Gain), kParamDetuneWid
"""

import struct, json, hashlib
import zstandard as zstd
import cbor2


# ══════════════════════════════════════════
# CUSTOM CBOR ENCODER — float32, key order
# ══════════════════════════════════════════

def cbor_encode(obj):
    if obj is None: return b'\xf6'
    elif obj is True: return b'\xf5'
    elif obj is False: return b'\xf4'
    elif isinstance(obj, int) and not isinstance(obj, bool):
        if obj >= 0:
            if obj <= 23: return bytes([obj])
            elif obj <= 255: return bytes([0x18, obj])
            elif obj <= 65535: return b'\x19' + struct.pack('>H', obj)
            elif obj <= 4294967295: return b'\x1a' + struct.pack('>I', obj)
            else: return b'\x1b' + struct.pack('>Q', obj)
        else:
            val = -1 - obj
            if val <= 23: return bytes([0x20 + val])
            elif val <= 255: return bytes([0x38, val])
            elif val <= 65535: return b'\x39' + struct.pack('>H', val)
            elif val <= 4294967295: return b'\x3a' + struct.pack('>I', val)
            else: return b'\x3b' + struct.pack('>Q', val)
    elif isinstance(obj, float):
        return b'\xfa' + struct.pack('>f', obj)
    elif isinstance(obj, str):
        e = obj.encode('utf-8')
        return _cbor_len(3, len(e)) + e
    elif isinstance(obj, bytes):
        return _cbor_len(2, len(obj)) + obj
    elif isinstance(obj, list):
        return _cbor_len(4, len(obj)) + b''.join(cbor_encode(i) for i in obj)
    elif isinstance(obj, dict):
        items = b''.join(cbor_encode(k) + cbor_encode(v) for k, v in obj.items())
        return _cbor_len(5, len(obj)) + items
    else: raise TypeError(f"Cannot CBOR encode: {type(obj)}: {obj!r}")

def _cbor_len(major, length):
    m = major << 5
    if length <= 23: return bytes([m | length])
    elif length <= 255: return bytes([m | 24, length])
    elif length <= 65535: return bytes([m | 25]) + struct.pack('>H', length)
    else: return bytes([m | 26]) + struct.pack('>I', length)

def fix_floats(obj):
    """Convert all float64 → float32 precision."""
    if isinstance(obj, dict): return {k: fix_floats(v) for k, v in obj.items()}
    elif isinstance(obj, list): return [fix_floats(v) for v in obj]
    elif isinstance(obj, float): return struct.unpack('f', struct.pack('f', obj))[0]
    return obj


# ══════════════════════════════════════════
# FX TYPE MAP
# ══════════════════════════════════════════

FX_TYPES = {
    "FXDistortion": 0,
    "FXPhaser": 2,
    "FXChorus": 3,
    "FXDelay": 4,
    "FXComp": 5,
    "FXReverb": 6,
    "FXEQ": 7,
    "FXFilter": 8,
}


# ══════════════════════════════════════════
# WAVEFORM → WAVETABLE POSITION MAP
# ══════════════════════════════════════════

WAVEFORM_TO_POS = {
    "saw": 0.0,
    "square": 64.0,
    "triangle": 128.0,
    "sine": 192.0,
}


# ══════════════════════════════════════════
# BUILD PRESET FROM CLAUDE PARAMS
# ══════════════════════════════════════════

def build_preset_from_params(params: dict, template_path: str):
    """
    Convert Claude AI parameters into a valid .SerumPreset binary.
    
    Returns: (preset_bytes, preset_name)
    """
    with open(template_path, "rb") as f:
        template = f.read()
    
    # Parse template
    json_len = struct.unpack_from("<I", template, 9)[0]
    json_start = 17
    meta = json.loads(template[json_start:json_start + json_len])
    cbor_offset = json_start + json_len
    cbor_version = struct.unpack_from("<I", template, cbor_offset + 4)[0]
    preset = cbor2.loads(zstd.ZstdDecompressor().decompress(template[cbor_offset + 8:]))
    
    # Helper: set plainParams on a block
    def set_params(block_name, new_params):
        if block_name not in preset:
            return
        pp = preset[block_name].get("plainParams", {})
        if pp == "default":
            pp = {}
        pp.update(new_params)
        preset[block_name]["plainParams"] = pp
    
    # ── Oscillator A ──
    osc_a = params.get("oscillator_a", {})
    osc_params = {}
    # Volume (Claude may use "gain", "volume", or direct "kParamVolume")
    if "kParamVolume" in osc_a:
        osc_params["kParamVolume"] = float(osc_a["kParamVolume"])
    elif "volume" in osc_a or "gain" in osc_a:
        osc_params["kParamVolume"] = float(osc_a.get("volume", osc_a.get("gain", 0.8)))
    else:
        osc_params["kParamVolume"] = 0.8  # always set a volume
    if "octave" in osc_a:
        osc_params["kParamOctave"] = float(osc_a["octave"])
    if "detune" in osc_a:
        osc_params["kParamDetune"] = float(osc_a["detune"])
    if "unison_voices" in osc_a or "unison" in osc_a:
        osc_params["kParamUnison"] = float(osc_a.get("unison_voices", osc_a.get("unison", 1)))
    if "unison_detune" in osc_a or "detune_width" in osc_a:
        osc_params["kParamDetuneWid"] = float(osc_a.get("detune_width", osc_a.get("unison_detune", 0.0)))
        # Scale: if value is 0-1, convert to 0-100 range
        if osc_params["kParamDetuneWid"] <= 1.0:
            osc_params["kParamDetuneWid"] *= 100.0
    if "pan" in osc_a:
        osc_params["kParamPan"] = float(osc_a["pan"])
    # Also handle direct Serum param names (from v4 engine)
    for k, v in osc_a.items():
        if k.startswith("kParam") and k not in osc_params:
            osc_params[k] = float(v) if isinstance(v, (int, float)) else v
    # FORCE ENABLE Oscillator0 (template may have it disabled)
    osc_params["kParamEnable"] = 1.0
    if osc_params:
        set_params("Oscillator0", osc_params)
    
    # DO NOT touch Oscillator1/2/3 - they are internal Serum 2 routing slots
    # (Noise oscillator, Sub, etc.) that need to keep template values for signal flow.
    
    # WTOsc params
    wt_params = {}
    # Map waveform name to wavetable position
    waveform = osc_a.get("waveform", "saw")
    if "table_pos" in osc_a and float(osc_a["table_pos"]) > 0:
        wt_params["kParamTablePos"] = float(osc_a["table_pos"])
    elif waveform in WAVEFORM_TO_POS:
        wt_params["kParamTablePos"] = WAVEFORM_TO_POS[waveform]
    if "warp" in osc_a:
        wt_params["kParamWarp"] = float(osc_a["warp"])
    if "warp_type" in osc_a:
        wt_params["kParamWarpMenu"] = osc_a["warp_type"]
    # Also handle "wt" key with direct Serum params
    wt_direct = params.get("wt", {})
    for k, v in wt_direct.items():
        if k.startswith("kParam"):
            wt_params[k] = float(v) if isinstance(v, (int, float)) else v
    if wt_params:
        wt = preset["Oscillator0"]["WTOsc0"]
        pp = wt.get("plainParams", {})
        if pp == "default": pp = {}
        pp.update(wt_params)
        wt["plainParams"] = pp
    
    # ── Wavetable PATH (v5 engine: load actual factory wavetable) ──
    wt_path = wt_direct.get("wavetable_path") or osc_a.get("wavetable_path") or osc_a.get("wavetable")
    if wt_path and isinstance(wt_path, str):
        wt_block = preset["Oscillator0"]["WTOsc0"]
        # Remove any embeddedWTData (would conflict with relativePathToWT)
        if "embeddedWTData" in wt_block:
            del wt_block["embeddedWTData"]
        wt_block["relativePathToWT"] = wt_path
        # Also reset flex curve to clean state (template may have weird curve)
        if "flex" in wt_block:
            wt_block["flex"] = {"curveVals": [0.5, 0.5], "numPoints": 1, "xVals": [0.0, 1.0], "yVals": [1.0, 0.0]}
    
    # ── Oscillator B ──
    osc_b = params.get("oscillator_b", {})
    if osc_b.get("enabled"):
        osc_b_params = {}
        if "volume" in osc_b or "gain" in osc_b:
            osc_b_params["kParamVolume"] = float(osc_b.get("volume", osc_b.get("gain", 0.5)))
        if "octave" in osc_b:
            osc_b_params["kParamOctave"] = float(osc_b["octave"])
        if "detune" in osc_b:
            osc_b_params["kParamDetune"] = float(osc_b["detune"])
        if "unison_voices" in osc_b or "unison" in osc_b:
            osc_b_params["kParamUnison"] = float(osc_b.get("unison_voices", osc_b.get("unison", 1)))
        if osc_b_params:
            set_params("Oscillator1", osc_b_params)
    
    # ── Filter ──
    filt = params.get("filter", {})
    if filt.get("enabled", True):
        filt_params = {
            "kParamEnable": 1.0,      # FORCE enable
            "kParamWet": 1.0,         # FORCE 100% wet (no dry bypass)
        }
        # Handle generic names
        if "cutoff" in filt:
            filt_params["kParamFreq"] = float(filt["cutoff"])
        # Handle direct Serum names
        for k, v in filt.items():
            if k.startswith("kParam"):
                filt_params[k] = float(v) if isinstance(v, (int, float)) else v
        if "resonance" in filt:
            res = float(filt["resonance"])
            # Scale: if 0-1, convert to 0-100 range
            filt_params["kParamReso"] = res * 100.0 if res <= 1.0 else res
        if "type" in filt:
            filt_params["kParamType"] = filt["type"]
        if filt_params:
            set_params("VoiceFilter0", filt_params)
    
    # ── Amp Envelope ──
    env = params.get("envelope_amp", params.get("env", {}))
    env_params = {}
    for k in ["attack", "decay", "sustain", "release"]:
        if k in env:
            env_params[f"kParam{k.capitalize()}"] = float(env[k])
    # Handle direct Serum names
    for k, v in env.items():
        if k.startswith("kParam"):
            env_params[k] = float(v) if isinstance(v, (int, float)) else v
    if env_params:
        set_params("Env0", env_params)
    
    # ── FX Chain ──
    fx = params.get("fx", {})
    fx_list = []
    
    # ALWAYS clear template FX first - we add only what AI requested
    if "FXRack0" in preset and isinstance(preset["FXRack0"], dict):
        preset["FXRack0"]["FX"] = []
    
    # If fx is already a list (v5 direct format), use it directly
    if isinstance(fx, list):
        # Normalize each FX entry - add REQUIRED inner slots (lfo, lfophasor, etc.)
        for fx_entry in fx:
            if not isinstance(fx_entry, dict): continue
            ft = fx_entry.get("type")
            # Find inner FX class
            inner_class = None
            for k in fx_entry:
                if k.startswith("FX") and isinstance(fx_entry[k], dict):
                    inner_class = k
                    break
            if not inner_class: continue
            inner = fx_entry[inner_class]
            
            # Add required slots per FX type
            if ft == 1 or ft == 2 or ft == 3:  # Flanger, Phaser, Chorus
                if "lfophasor" not in inner:
                    inner["lfophasor"] = 0.0
            elif ft == 9:  # HyperD - needs lfo (curve data)
                if "lfo" not in inner:
                    # 2 arrays of 8 floats each (LFO curves)
                    inner["lfo"] = [
                        [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                        [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                    ]
                # HyperD has TWO mix knobs (Hyper + DimE), normalize outer
                if "kUIParamMixOrGainDimE" not in fx_entry:
                    fx_entry["kUIParamMixOrGainDimE"] = 0.0
                if "kUIParamMixOrGainHyper" not in fx_entry:
                    fx_entry["kUIParamMixOrGainHyper"] = 0.0
                # Remove generic kUIParamMixOrGain if present (HyperD uses specific ones)
                if "kUIParamMixOrGain" in fx_entry:
                    del fx_entry["kUIParamMixOrGain"]
            elif ft == 11:  # Conv - needs IR data (we don't have IRs, so use a known factory one)
                if "relativePathToIR" not in inner:
                    inner["relativePathToIR"] = "Factory/Massive/Cathedral In The Sky.flac"
                if "numChannels" not in inner:
                    inner["numChannels"] = 2
                if "numFrames" not in inner:
                    inner["numFrames"] = 180611
                if "sampleRate" not in inner:
                    inner["sampleRate"] = 44100
            
            # All FX (except 7 EQ and 13 Split) need kUIParamMixOrGain at outer level
            if ft not in (7, 9, 13):  # 9 already handled above
                if "kUIParamMixOrGain" not in fx_entry:
                    fx_entry["kUIParamMixOrGain"] = 0.0
            
            # Ensure 'type' is LAST key (Serum expects this order)
            if "type" in fx_entry:
                t_val = fx_entry.pop("type")
                fx_entry["type"] = t_val
            
            # For Distortion (type 0), add flex at OUTER level (not inner)
            if ft == 0 and "flex" not in fx_entry:
                # Insert flex BEFORE kUIParamMixOrGain and type
                t_val = fx_entry.pop("type", None)
                ui_val = fx_entry.pop("kUIParamMixOrGain", 0.0)
                fx_entry["flex"] = [{}, {}]
                fx_entry["kUIParamMixOrGain"] = ui_val
                if t_val is not None:
                    fx_entry["type"] = t_val
        
        fx_list = fx
        preset["FXRack0"]["FX"] = fx_list
        fx = {}  # skip individual FX processing below
    
    # Reverb
    reverb = fx.get("reverb", {})
    if reverb.get("enabled"):
        reverb_params = {}
        if "wet" in reverb: reverb_params["kParamWet"] = float(reverb["wet"]) * (100 if float(reverb["wet"]) <= 1 else 1)
        if "size" in reverb: reverb_params["kParamSize"] = float(reverb["size"]) * (100 if float(reverb["size"]) <= 1 else 1)
        if "damping" in reverb: reverb_params["kParamFeedback"] = float(reverb["damping"]) * (100 if float(reverb["damping"]) <= 1 else 1)
        if "type" in reverb: reverb_params["kParamType"] = reverb["type"]
        else: reverb_params["kParamType"] = "kHall"
        reverb_params.setdefault("kParamWidth", 100.0)
        reverb_params.setdefault("kParamFreqB", 35.0)
        fx_list.append({
            "FXReverb": {"plainParams": reverb_params},
            "kUIParamMixOrGain": 0.0,
            "type": 6,
        })
    
    # Delay
    delay = fx.get("delay", {})
    if delay.get("enabled"):
        delay_params = {}
        if "wet" in delay: delay_params["kParamWet"] = float(delay["wet"]) * (100 if float(delay["wet"]) <= 1 else 1)
        if "feedback" in delay: delay_params["kParamFeedback"] = float(delay["feedback"]) * (100 if float(delay["feedback"]) <= 1 else 1)
        delay_params.setdefault("kParamTimeL", 0.06)
        delay_params.setdefault("kParamTimeR", 0.06)
        delay_params.setdefault("kParamMode", 1.0)
        delay_params.setdefault("kParamBW", 3.0)
        fx_list.append({
            "FXDelay": {"plainParams": delay_params},
            "kUIParamMixOrGain": 0.0,
            "type": 4,
        })
    
    # Distortion
    dist = fx.get("distortion", {})
    if dist.get("enabled"):
        dist_params = {}
        if "drive" in dist: dist_params["kParamDrive"] = float(dist["drive"]) * (100 if float(dist["drive"]) <= 1 else 1)
        if "wet" in dist: dist_params["kParamWet"] = float(dist["wet"]) * (100 if float(dist["wet"]) <= 1 else 1)
        fx_list.append({
            "FXDistortion": {"plainParams": dist_params},
            "kUIParamMixOrGain": 0.0,
            "type": 0,
        })
    
    # Chorus
    chorus = fx.get("chorus", {})
    if chorus.get("enabled"):
        chorus_params = {}
        if "wet" in chorus: chorus_params["kParamWet"] = float(chorus["wet"]) * (100 if float(chorus["wet"]) <= 1 else 1)
        if "depth" in chorus: chorus_params["kParamDepth"] = float(chorus["depth"]) * (100 if float(chorus["depth"]) <= 1 else 1)
        fx_list.append({
            "FXChorus": {"plainParams": chorus_params},
            "kUIParamMixOrGain": 0.0,
            "type": 3,
        })
    
    if fx_list:
        preset["FXRack0"]["FX"] = fx_list
    
    # ── Metadata ──
    preset_name = params.get("preset_name", "SoundMatch AI")
    meta["presetName"] = preset_name
    meta["artist"] = "SoundMatch AI"
    meta["comments"] = params.get("reasoning", "")[:200]
    
    # ── Encode ──
    new_json = json.dumps(meta, separators=(",", ":")).encode("utf-8")
    custom_cbor = cbor_encode(fix_floats(preset))
    compressed = zstd.ZstdCompressor(level=3).compress(custom_cbor)
    
    # ⭐ CRITICAL: Update hash = MD5 of compressed CBOR
    meta["hash"] = hashlib.md5(compressed).hexdigest()
    new_json = json.dumps(meta, separators=(",", ":")).encode("utf-8")
    
    result = (
        template[:9] +
        struct.pack("<II", len(new_json), 0) +
        new_json +
        struct.pack("<II", len(custom_cbor), cbor_version) +
        compressed
    )
    
    return result, preset_name
