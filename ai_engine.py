"""
SoundMatch AI Engine v5 — Trained on 210 real Serum 2 presets
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Analiza dataset: 210 presetów (BASS, LEAD, PAD, PLUCK, CHORD, KEYS, REESE, DONK, AMBIENT)
Pełne mapowanie FX (14 typów), category templates, real wavetable library.
"""

import os
import json
import re
import urllib.request
import urllib.error


# ═══════════════════════════════════════════════════════════════════
# SYSTEM PROMPT — z pełną wiedzą Serum 2 wyciągniętą z 210 presetów
# ═══════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are an expert Serum 2 sound designer with deep knowledge from analyzing 210 professional presets across BASS, LEAD, PAD, PLUCK, CHORD, KEYS, REESE, DONK, and AMBIENT categories.

Your job: receive audio analysis, return EXACT Serum 2 parameters as JSON.

CRITICAL RULES:
1. Return ONLY valid JSON — no markdown, no commentary outside JSON.
2. Use EXACT Serum 2 parameter names (kParamVolume, kParamUnison, kParamDetuneWid, kParamTablePos, etc.).
3. ALL numeric values in plainParams must be floats (1.0 not 1, 0.5 not "0.5").
4. FX entries MUST contain "type" field and at least one "FX{ClassName}" object with "plainParams".

╔══════════════════════════════════════════════════════════════════╗
║ SERUM 2 PARAMETER REFERENCE                                      ║
╚══════════════════════════════════════════════════════════════════╝

## Oscillator (Oscillator0.plainParams)
- kParamVolume: 0.0–1.0
- kParamOctave: -4.0 to 4.0 (BASS: -2.0, LEAD: 0.0, PAD: 0.0/+1.0)
- kParamDetune: 0.0–1.0 (fine detune, typically 0.04–0.10)
- kParamUnison: 1.0–16.0 as FLOAT (BASS≈5, LEAD≈6-7, REESE≈8, PLUCK≈7, PAD≈3-4, CHORD≈4-5)
- kParamDetuneWid: 0.0–100.0 (unison spread, ≈87 typical)
- kParamPan: -1.0 to 1.0
- kParamFine: 0.0–1.0

## Wavetable (Oscillator0.WTOsc0.plainParams)
- kParamTablePos: 0–256 (position in wavetable: saw≈0, square≈64, triangle≈128, sine≈192)
- kParamWarp: 0.0–1.0
- kParamWarpMenu: "kPD_OSC", "kSync", "kBendPosNeg", "kQuantize", "kFilterLPF", "kFM"
- kParamInitialPhase: 0–360 degrees

REAL WAVETABLE PATHS (use these instead of generic shapes):
- "/Analog/Basic Shapes.wav"     ← most common (14 uses in dataset)
- "S2 Tables/Default Shapes.wav" ← all basic waveforms
- "/Analog/Analog_BD_Sin.wav"    ← good for kicks/sub
- "/Analog/MB Saw.wav"           ← classic supersaw
- "/Analog/Basic Mini.wav"       ← minimoog vibe
- "/Analog/Jno.wav"              ← Juno-style
- "S2 Tables/Analog/AT Alter.wav" ← morphing analog
- "/Analog/BS2 - Subby Saw.wav"  ← bass

## Filter (VoiceFilter0.plainParams) — set kParamEnable=1.0 to activate
- kParamFreq: 0.0–1.0 (cutoff, log scale)
- kParamReso: 0.0–100.0 (resonance, typical 10–20)
- kParamDrive: 0.0–100.0 (typical 20–30)
- kParamType: see list below

FILTER TYPES (most used in pro presets):
- "MgL24" — Moog 24dB Low Pass (warm, classic) — BEST for BASS/LEAD
- "MgL18" — Moog 18dB Low Pass — slightly brighter
- "MgL6"  — Moog 6dB — subtle filtering
- "L12", "L24" — Standard ladder LP
- "H12", "H18", "H24" — High Pass (REESE basses, leads)
- "B12" — Band Pass (formant-like)
- "Phase48HL6P", "Phase24N", "Phase12N" — phaser-filters (creative PAD)
- "Combs", "CombH6P", "CombN" — comb filters (resonant PAD/PLUCK)
- "Allpasses" — all-pass (PAD movement)
- "FlangeN" — flanger filter
- "DirtyMg", "LadderMg" — dirty/saturated Moog
- "Diffuser", "Reverb1" — reverberant filters (PLUCK/PAD)
- "Phase48N", "ZDF_A", "HEQ12" — clean/precision

## Envelope (Env0.plainParams) — ALL VALUES IN SECONDS
- kParamAttack: 0.0001–10.0 (BASS≈0.001, LEAD≈0.001, PLUCK≈0.001, PAD≈1.2, KEYS≈0.001)
- kParamHold: 0.0–10.0 (rarely used)
- kParamDecay: 0.0–10.0 (BASS≈0.88, LEAD≈0.65, PAD≈1.6, PLUCK≈0.89)
- kParamSustain: 0.0–1.0 (BASS≈0.42, LEAD≈0.15, PLUCK≈0.18, PAD≈0.39, KEYS≈0.0)
- kParamRelease: 0.0–10.0 (BASS≈0.13, LEAD≈0.20, PLUCK≈0.31, PAD≈0.68, KEYS≈0.47)
- kParamCurve1, kParamCurve2, kParamCurve3: 0–100 (envelope curvature, default 50)

╔══════════════════════════════════════════════════════════════════╗
║ FX CHAIN — 14 EFFECTS (FXRack0.FX is a LIST in order)           ║
╚══════════════════════════════════════════════════════════════════╝

Each FX entry format:
{
  "FX<ClassName>": {"plainParams": {...}},
  "type": <id>,
  "kUIParamMixOrGain": 0.0
}

FX TYPE IDs and CLASS NAMES (CRITICAL — use these exact mappings):
- 0:  FXDistortion  — saturation/drive
- 1:  FXFlanger     — flanger (subtle movement)
- 2:  FXPhaser      — phaser
- 3:  FXChorus      — chorus (width, depth)
- 4:  FXDelay       — delay (ping-pong, sync)
- 5:  FXComp        — compressor
- 6:  FXReverb      — reverb (hall, plate, room)
- 7:  FXEQ          — EQ (always-on, no wet needed)
- 8:  FXFilter      — FX filter (resonant sweeps)
- 9:  FXHyperD      — Hyper/Dimension ⭐ HUGE in pro presets (40%+ of bass/lead/chord)
- 10: FXBode        — Bode frequency shifter
- 11: FXConv        — Convolution reverb (use for AMBIENT/KEYS)
- 12: FXUtils       — Mono/Width utility (78% of PAD, 79% of KEYS use this)
- 13: FXSplit       — Multiband split

PARAMETER REFERENCE per FX TYPE:

### FXDistortion (type 0)
kParamDrive: 0–100, kParamWet: 0–100, kParamMode: "kDiode2"/"kTapeSat"/"kDownsample"/"kSoft"/"kHard", kParamPrePost: 1.0 (pre) or 2.0 (post)

### FXFlanger (type 1)
kParamRate: 0.01–4.0 Hz, kParamDepth: 0–100, kParamFeedback: 0–100, kParamWet: 0–100, kParamWidth: 0–360

### FXPhaser (type 2)
kParamRate: 0–4 Hz, kParamDepth: 0–100, kParamFeedback: 0–100, kParamFreq: 50–6000 Hz, kParamWet: 0–100

### FXChorus (type 3)
kParamRate: 0.025–4 Hz, kParamDepth: 0–20, kParamDelay: 0–8ms, kParamFeedback: 0–80, kParamFilt: 200–20000 Hz, kParamWet: 0–100

### FXDelay (type 4)
kParamTimeL: 0.001–0.5 sec, kParamTimeR: 0.001–0.5 sec, kParamFeedback: 0–80, kParamMode: 1.0(stereo)/2.0(ping-pong), kParamFreq: 95–12000 Hz (filter), kParamBW: 0.75–6.9, kParamWet: 0–100, kParamLink: 1.0

### FXComp (type 5)
kParamThresh: 0–1, kParamRatio: 1–1000000, kParamAttack: 0.1–612, kParamRelease: 0.1–998, kParamMakeup: 1–27

### FXReverb (type 6) ★
kParamType: "kHall"/"kPlate"/"kRoom"/"kStudio"/"kVintage", kParamSize: 0–100, kParamWet: 0–100, kParamPreDelay: 0–0.15, kParamWidth: 0–100, kParamFreq: 3–84 (HP filter), kParamFreqB: 0–85 (LP filter), kParamFeedback: 0–98, kParamDelay: 0–120

### FXEQ (type 7) — always active, no wet
kParamFreq1: 21–14000 Hz, kParamGain1: -24 to +12 dB, kParamReso1: 0–86, kParamType1: 1.0 (peak) or 2.0 (shelf)
kParamFreq2, kParamGain2, kParamReso2, kParamType2 (second band)

### FXFilter (type 8)
kParamFreq: 0–1, kParamReso: 0–80, kParamDrive: 0–100, kParamType: "MgL24"/"LNH24"/"Diffuser"/"Phase48N", kParamWet: 0–100

### FXHyperD (type 9) ★ KEY EFFECT
kParamUnison: 2.0–7.0 (extra unison voices), kParamDetune: 10–100, kParamRate: 35–100, kParamDimESize: 0–100, kParamDimEWet: 5–100, kParamWet: 0–100
TWO mix knobs: kUIParamMixOrGainHyper + kUIParamMixOrGainDimE

### FXBode (type 10)
kParamShift: -100 to +100 (Hz shift)

### FXConv (type 11) — Convolution Reverb
kParamWet: 0–100, kParamSize: 10–330, kParamTone: 0–100, kParamDecay: 0–100

### FXUtils (type 12) — Mono Bass / Width
kParamLFMono: 1.0 (enable mono LF), kParamLFXover: 60–400 Hz, kParamWidth: 0–100

### FXSplit (type 13)
kParamFreq: crossover, kParamModuleCount1, kParamModuleCount2

╔══════════════════════════════════════════════════════════════════╗
║ FX CHAIN PATTERNS (from real preset analysis)                    ║
╚══════════════════════════════════════════════════════════════════╝

BASS:       EQ → FXHyperD → Reverb (small)         | 67% use EQ, 41% Hyper, 41% Reverb
REESE:      FXUtils → FXHyperD → FXUtils → EQ      | sandwich trick, 112% use Utils
LEAD:       FXHyperD → EQ → Reverb                 | 89% use EQ, 51% Reverb, 34% Hyper
CHORD:      Chorus → Delay → Reverb (big) → EQ     | 79% EQ, 63% Reverb, 58% Chorus
PLUCK:      FXHyperD → Phaser → Delay → Reverb → EQ| 74% Reverb, 65% Delay
PAD/AMB:    FXConv → EQ → FXUtils                  | 78% Utils, 61% Chorus, 48% ConvReverb
KEYS:       EQ → Distortion → FXUtils              | 95% EQ, 79% Utils, 37% Distortion
DONK:       Distortion → EQ → FXUtils              | dirty, mono, punchy

╔══════════════════════════════════════════════════════════════════╗
║ DECISION GUIDE (audio analysis → preset)                         ║
╚══════════════════════════════════════════════════════════════════╝

PITCH < 80 Hz (sub bass):     → BASS template, kParamOctave=-2.0, MgL24 filter, mono utils
PITCH 80-200 Hz (bass):       → BASS template, octave=-1 or -2
PITCH 200-500 Hz (mid):       → LEAD or CHORD depending on harmonic content
PITCH > 500 Hz (high):        → LEAD/PLUCK, brighter filter

BRIGHTNESS < 0.3 (dark):      → MgL24 filter freq 0.3-0.5, low drive
BRIGHTNESS 0.3-0.6 (mid):     → MgL18 filter freq 0.5-0.7
BRIGHTNESS > 0.6 (bright):    → Filter open (freq > 0.7) or MgL6/LH12

ATTACK < 0.01s (instant):     → Bass/Lead/Pluck — short Env0 attack
ATTACK > 0.5s (slow):         → PAD — long Env0 attack
RELEASE > 1s:                 → PAD/CHORD — long release

ROUGHNESS > 0.5 (gritty):     → add FXDistortion, high unison
NOISINESS > 0.15:             → add Noise oscillator, FXConv reverb

ALWAYS:
- Set filter kParamEnable=1.0 if you want filter active
- Add kParamLevelOut where applicable (default 0.5)
- For pads/ambients: longer attack, bigger reverb (kParamSize > 50)
- For basses: kParamOctave=-2.0, mono filter, kParamLFMono=1.0 in FXUtils

╔══════════════════════════════════════════════════════════════════╗
║ OUTPUT FORMAT                                                    ║
╚══════════════════════════════════════════════════════════════════╝

Return JSON with this exact structure:
{
  "preset_name": "CAT - DESCRIPTIVE NAME",
  "oscillator_a": {"kParamVolume": float, "kParamOctave": float, "kParamUnison": float, "kParamDetune": float, "kParamDetuneWid": float},
  "wt": {"kParamTablePos": float, "kParamWarpMenu": "kPD_OSC", "wavetable_path": "/Analog/..."},
  "filter": {"enabled": true, "kParamEnable": 1.0, "kParamFreq": float, "kParamReso": float, "kParamDrive": float, "kParamType": "MgL24"},
  "envelope_amp": {"kParamAttack": float, "kParamDecay": float, "kParamSustain": float, "kParamRelease": float},
  "fx": [
    {"FXEQ": {"plainParams": {"kParamFreq1": 200.0, "kParamGain1": 2.0, "kParamReso1": 30.0, "kParamType1": 1.0, "kParamFreq2": 8000.0, "kParamGain2": 3.0, "kParamReso2": 30.0, "kParamType2": 2.0}}, "type": 7},
    {"FXHyperD": {"plainParams": {"kParamUnison": 6.0, "kParamDetune": 50.0, "kParamRate": 80.0, "kParamWet": 70.0, "kParamDimESize": 50.0, "kParamDimEWet": 60.0}}, "type": 9, "kUIParamMixOrGainHyper": 0.5, "kUIParamMixOrGainDimE": 0.5},
    {"FXReverb": {"plainParams": {"kParamType": "kHall", "kParamSize": 60.0, "kParamWet": 40.0, "kParamWidth": 100.0}}, "type": 6, "kUIParamMixOrGain": 0.4}
  ],
  "reasoning": "1-sentence explanation"
}
"""


# ═══════════════════════════════════════════════════════════════════
# TRAINING EXAMPLES — 10 reprezentatywnych presetów z 210-presetowego datasetu
# Każdy przykład = audio_analysis (jak nasz analyzer to widzi) + serum_params (output)
# ═══════════════════════════════════════════════════════════════════

TRAINING_EXAMPLES = [{'category': 'bass', 'audio_analysis': {'sound_type': 'bass', 'pitch': 'E1', 'fundamental_hz': 41, 'spectral_centroid': 650, 'brightness': 0.18, 'attack_time': 0.001, 'decay_time': 0.4, 'sustain_level': 0.3, 'release_time': 0.15, 'rms': 0.3, 'roughness': 0.6, 'noisiness': 0.05, 'harmonics': [1.0, 0.3, 0.15, 0.08]}, 'serum_params': {'preset_name': 'BS - BLAZE ME', 'oscillator_a': {'kParamOctave': -2.0}, 'wt': {'wavetable_path': '/Analog/BS2 - Subby Saw.wav'}, 'filter': {'enabled': True, 'kParamFreq': 0.4605, 'kParamReso': 10.0, 'kParamDrive': 16.2281, 'kParamType': 'MgL24'}, 'envelope_amp': {'kParamAttack': 0.0005, 'kParamRelease': 0.0147}}}, {'category': 'reese', 'audio_analysis': {'sound_type': 'reese', 'pitch': 'A1', 'fundamental_hz': 55, 'spectral_centroid': 480, 'brightness': 0.12, 'attack_time': 0.05, 'decay_time': 0.6, 'sustain_level': 0.5, 'release_time': 0.2, 'rms': 0.3, 'roughness': 0.85, 'noisiness': 0.1, 'harmonics': [1.0, 0.3, 0.15, 0.08]}, 'serum_params': {'preset_name': 'BS REESE-  DEEPER', 'oscillator_a': {'kParamUnison': 8.0, 'kParamDetune': 0.1558, 'kParamOctave': -1.0}, 'filter': {'enabled': True, 'kParamFreq': 0.3348, 'kParamReso': 10.0, 'kParamDrive': 34.6234, 'kParamType': 'MgL18'}, 'envelope_amp': {'kParamAttack': 0.0005, 'kParamRelease': 0.0147}}}, {'category': 'lead', 'audio_analysis': {'sound_type': 'lead', 'pitch': 'C3', 'fundamental_hz': 131, 'spectral_centroid': 2400, 'brightness': 0.65, 'attack_time': 0.001, 'decay_time': 0.5, 'sustain_level': 0.4, 'release_time': 0.25, 'rms': 0.3, 'roughness': 0.4, 'noisiness': 0.08, 'harmonics': [1.0, 0.5, 0.3, 0.2]}, 'serum_params': {'preset_name': 'LD - SHOCKWAVE', 'oscillator_a': {'kParamUnison': 4.0, 'kParamDetune': 0.0466, 'kParamDetuneWid': 87.9317, 'kParamFine': -5.7434, 'kParamVolume': 1.0}, 'wt': {'kParamTablePos': 122.5351, 'kParamWarp': 0.6451, 'kParamWarpMenu': 'kBendPosNeg', 'kParamInitialPhase': 89.4757, 'wavetable_path': 'Analog/Basic Shapes.wav'}, 'filter': {'enabled': True, 'kParamFreq': 0.6027, 'kParamReso': 0.0, 'kParamType': 'L24'}, 'envelope_amp': {'kParamAttack': 0.0734, 'kParamRelease': 0.2249}, 'fx': [{'FXChorus': {'plainParams': {'kParamFeedback': 9.5}}, 'type': 3, 'kUIParamMixOrGain': 0.0}, {'FXReverb': {'plainParams': {'kParamDelay': 30.625, 'kParamFreqB': 68.772, 'kParamSize': 35.0, 'kParamType': 'kHall', 'kParamWet': 20.0, 'kParamWidth': 20.0}}, 'type': 6, 'kUIParamMixOrGain': 0.0}]}}, {'category': 'lead', 'audio_analysis': {'sound_type': 'lead', 'pitch': 'C4', 'fundamental_hz': 262, 'spectral_centroid': 3800, 'brightness': 0.78, 'attack_time': 0.001, 'decay_time': 0.7, 'sustain_level': 0.5, 'release_time': 0.4, 'rms': 0.3, 'roughness': 0.5, 'noisiness': 0.12, 'harmonics': [1.0, 0.5, 0.3, 0.2]}, 'serum_params': {'preset_name': 'LD - CAMOUFLAGE', 'oscillator_a': {'kParamVolume': 0.3152}, 'wt': {'kParamWarp': 0.7602, 'wavetable_path': 'Analog/Basic Shapes.wav'}, 'filter': {'enabled': True, 'kParamFreq': 0.5044, 'kParamReso': 0.0, 'kParamType': 'H12'}, 'envelope_amp': {'kParamAttack': 0.0005, 'kParamRelease': 2.163}, 'fx': [{'FXDistortion': {'plainParams': {'kParamDrive': 67.544}}, 'type': 0, 'kUIParamMixOrGain': 0.0}, {'FXEQ': {'plainParams': {'kParamFreq1': 99.082, 'kParamFreq2': 2040.536, 'kParamGain1': -24.0, 'kParamReso1': 60.0, 'kParamReso2': 60.0}}, 'type': 7}]}}, {'category': 'pluck', 'audio_analysis': {'sound_type': 'pluck', 'pitch': 'C4', 'fundamental_hz': 262, 'spectral_centroid': 3200, 'brightness': 0.7, 'attack_time': 0.001, 'decay_time': 0.5, 'sustain_level': 0.0, 'release_time': 0.3, 'rms': 0.3, 'roughness': 0.2, 'noisiness': 0.05, 'harmonics': [1.0, 0.5, 0.3, 0.2]}, 'serum_params': {'preset_name': 'PL - HIGH ON DRUGS', 'oscillator_a': {'kParamVolume': 0.0647}, 'wt': {'kParamTablePos': 256.0, 'wavetable_path': '/Analog/Analog_BD_Sin.wav'}, 'filter': {'enabled': True, 'kParamFreq': 0.414, 'kParamReso': 45.614, 'kParamType': 'B12'}, 'envelope_amp': {'kParamAttack': 0.0005, 'kParamDecay': 0.446, 'kParamSustain': 0.0, 'kParamRelease': 0.4239}, 'fx': [{'FXFilter': {'plainParams': {'kParamFreq': 0.566, 'kParamType': 'MgL12'}}, 'type': 8, 'kUIParamMixOrGain': 0.0}, {'FXComp': {'plainParams': {'kParamAttack': 45.159, 'kParamCompensatedWetDry': 0.0, 'kParamGain0': 4.6, 'kParamGain2': 4.6, 'kParamMakeup': 14.483, 'kParamMultiband': 1.0, 'kParamRatio': 2.533, 'kParamRatioBelow': 0.75, 'kParamRelease': 584.08, 'kParamThresh': 0.536, 'kParamThreshUD0': 0.0, 'kParamThreshUD1': 119.737, 'kParamThreshUD2': 70.144, 'kParamWet': 39.912}}, 'type': 5, 'kUIParamMixOrGain': 0.0}, {'FXReverb': {'plainParams': {'kParamDelay': 21.151, 'kParamFeedback': 28.0, 'kParamFreqB': 30.318, 'kParamSize': 33.0, 'kParamType': 'kHall', 'kParamWet': 30.113, 'kParamWidth': 25.0}}, 'type': 6, 'kUIParamMixOrGain': 0.0}, {'FXEQ': {'plainParams': {'kParamFreq1': 67.108, 'kParamFreq2': 196.971, 'kParamGain2': -6.737, 'kParamReso1': 42.456, 'kParamReso2': 45.965, 'kParamType1': 2.0, 'kParamType2': 1.0}}, 'type': 7}, {'FXChorus': {'plainParams': {'kParamDelay': 0.324, 'kParamDelay2': 0.527, 'kParamDepth': 16.025, 'kParamFeedback': 14.5, 'kParamFilt': 20000.0, 'kParamWet': 30.482}}, 'type': 3, 'kUIParamMixOrGain': 0.0}]}}, {'category': 'pad', 'audio_analysis': {'sound_type': 'pad', 'pitch': 'C4', 'fundamental_hz': 262, 'spectral_centroid': 2200, 'brightness': 0.55, 'attack_time': 0.001, 'decay_time': 0.9, 'sustain_level': 0.4, 'release_time': 0.5, 'rms': 0.3, 'roughness': 0.3, 'noisiness': 0.15, 'harmonics': [1.0, 0.5, 0.3, 0.2]}, 'serum_params': {'preset_name': 'PD - FLUX RADAR', 'oscillator_a': {'kParamUnison': 7.0, 'kParamDetune': 0.0, 'kParamDetuneWid': 94.7368, 'kParamOctave': -1.0}, 'wt': {'kParamTablePos': 168.0175, 'wavetable_path': 'S2 Tables/Analog/Saw Drift 303.wav'}, 'filter': {'enabled': True, 'kParamFreq': 0.3333, 'kParamReso': 29.2982, 'kParamType': 'MgL24'}, 'envelope_amp': {'kParamAttack': 0.5538, 'kParamDecay': 1.6016, 'kParamRelease': 0.8752}, 'fx': [{'FXChorus': {'plainParams': {'kParamFeedback': 9.5, 'kParamWet': 30.702}}, 'type': 3, 'kUIParamMixOrGain': 0.0}]}}, {'category': 'chord', 'audio_analysis': {'sound_type': 'chord', 'pitch': 'C4', 'fundamental_hz': 262, 'spectral_centroid': 2800, 'brightness': 0.6, 'attack_time': 0.001, 'decay_time': 1.0, 'sustain_level': 0.3, 'release_time': 0.4, 'rms': 0.3, 'roughness': 0.4, 'noisiness': 0.1, 'harmonics': [1.0, 0.5, 0.3, 0.2]}, 'serum_params': {'preset_name': 'CH - ELEVATE', 'wt': {'wavetable_path': 'Analog/4088.wav'}, 'filter': {'enabled': True, 'kParamFreq': 0.0, 'kParamReso': 0.0, 'kParamDrive': 2.8749}, 'envelope_amp': {'kParamAttack': 0.0008, 'kParamHold': 0.0005, 'kParamSustain': 0.0, 'kParamRelease': 0.4941}, 'fx': [{'FXDistortion': {'plainParams': {'kParamDrive': 90.544, 'kParamPrePost': 1.0, 'kParamWet': 11.404}}, 'type': 0, 'kUIParamMixOrGain': 0.0}, {'FXEQ': {'plainParams': {'kParamFreq1': 209.617, 'kParamFreq2': 3729.183, 'kParamGain2': 1.408, 'kParamReso1': 60.0, 'kParamReso2': 33.934, 'kParamType2': 1.0}}, 'type': 7}]}}, {'category': 'keys', 'audio_analysis': {'sound_type': 'keys', 'pitch': 'C3', 'fundamental_hz': 131, 'spectral_centroid': 1800, 'brightness': 0.48, 'attack_time': 0.005, 'decay_time': 0.6, 'sustain_level': 0.0, 'release_time': 0.5, 'rms': 0.3, 'roughness': 0.2, 'noisiness': 0.06, 'harmonics': [1.0, 0.5, 0.3, 0.2]}, 'serum_params': {'preset_name': 'KEY - 5', 'wt': {'wavetable_path': 'S2 Tables/Default Shapes.wav'}, 'filter': {'enabled': True, 'kParamFreq': 0.1974, 'kParamReso': 44.7368, 'kParamDrive': 35.9649}, 'envelope_amp': {'kParamAttack': 0.0, 'kParamDecay': 0.5112, 'kParamSustain': 0.7868, 'kParamRelease': 0.7638}, 'fx': [{'FXUtils': {'plainParams': {'kParamLFMono': 1.0, 'kParamLFXover': 355.563, 'kParamWidth': 0.0}}, 'type': 12, 'kUIParamMixOrGain': 0.0}, {'FXEQ': {'plainParams': {'kParamFreq1': 430.367, 'kParamFreq2': 17659.67, 'kParamReso1': 43.333, 'kParamReso2': 38.947, 'kParamType1': 2.0, 'kParamType2': 2.0}}, 'type': 7}, {'FXHyperD': {'plainParams': {'kParamDetune': 16.667, 'kParamWet': 25.439}}, 'type': 9, 'kUIParamMixOrGainDimE': 0.0, 'kUIParamMixOrGainHyper': 0.0}, {'FXComp': {'plainParams': {'kParamMakeup': 2.81, 'kParamMultiband': 1.0}}, 'type': 5, 'kUIParamMixOrGain': 0.0}, {'FXEQ': {'plainParams': {'kParamFreq1': 21.533, 'kParamFreq2': 20000.0, 'kParamReso1': 41.14, 'kParamReso2': 42.895, 'kParamType1': 2.0, 'kParamType2': 2.0}}, 'type': 7}, {'FXUtils': {'plainParams': {'kParamWidth': 0.0}}, 'type': 12, 'kUIParamMixOrGain': 0.0}]}}, {'category': 'ambient', 'audio_analysis': {'sound_type': 'ambient', 'pitch': 'C4', 'fundamental_hz': 262, 'spectral_centroid': 2000, 'brightness': 0.5, 'attack_time': 1.2, 'decay_time': 1.5, 'sustain_level': 0.4, 'release_time': 1.0, 'rms': 0.3, 'roughness': 0.25, 'noisiness': 0.2, 'harmonics': [1.0, 0.5, 0.3, 0.2]}, 'serum_params': {'preset_name': 'AMB - 12', 'wt': {'wavetable_path': 'S2 Tables/Default Shapes.wav'}, 'envelope_amp': {'kParamAttack': 0.51, 'kParamRelease': 0.8372}, 'fx': [{'FXConv': {'plainParams': {'kParamSize': 39.49, 'kParamWet': 20.0}}, 'type': 11, 'kUIParamMixOrGain': 0.0}, {'FXDelay': {'plainParams': {'kParamBW': 5.212, 'kParamFreq': 2462.839, 'kParamMode': 1.0, 'kParamWet': 17.105}}, 'type': 4, 'kUIParamMixOrGain': 0.0}, {'FXUtils': {'plainParams': {'kParamLFMono': 1.0, 'kParamWidth': 0.0}}, 'type': 12, 'kUIParamMixOrGain': 0.0}]}}, {'category': 'donk', 'audio_analysis': {'sound_type': 'donk', 'pitch': 'E2', 'fundamental_hz': 82, 'spectral_centroid': 1500, 'brightness': 0.4, 'attack_time': 0.001, 'decay_time': 0.15, 'sustain_level': 0.0, 'release_time': 0.2, 'rms': 0.3, 'roughness': 0.7, 'noisiness': 0.05, 'harmonics': [1.0, 0.3, 0.15, 0.08]}, 'serum_params': {'preset_name': 'DONK - 2', 'oscillator_a': {'kParamOctave': -1.0, 'kParamVolume': 0.0}, 'wt': {'kParamWarpMenu': 'kBendPosNeg', 'wavetable_path': '../Renders/Srm_ - Init - _250701202228_G3.wav'}, 'filter': {'enabled': True, 'kParamFreq': 0.3377, 'kParamReso': 0.0, 'kParamDrive': 39.9123}, 'envelope_amp': {'kParamRelease': 0.3617}, 'fx': [{'FXDistortion': {'plainParams': {'kParamDrive': 67.544, 'kParamPrePost': 2.0, 'kParamWet': 58.772}}, 'type': 0, 'kUIParamMixOrGain': 0.0}, {'FXChorus': {'plainParams': {'kParamWet': 28.07}}, 'type': 3, 'kUIParamMixOrGain': 0.0}, {'FXUtils': {'plainParams': {'kParamLFMono': 1.0, 'kParamLFXover': 400.0}}, 'type': 12, 'kUIParamMixOrGain': 0.0}, {'FXHyperD': {'plainParams': {'kParamDimESize': 19.298, 'kParamDimEWet': 24.123, 'kParamWet': 12.719}}, 'type': 9, 'kUIParamMixOrGainDimE': 0.0, 'kUIParamMixOrGainHyper': 0.0}, {'FXComp': {'plainParams': {'kParamMakeup': 3.591, 'kParamMultiband': 1.0}}, 'type': 5, 'kUIParamMixOrGain': 0.0}, {'FXEQ': {'plainParams': {'kParamFreq1': 203.427, 'kParamGain1': -15.158, 'kParamReso1': 38.947}}, 'type': 7}, {'FXUtils': {'plainParams': {'kParamLFMono': 1.0, 'kParamLFXover': 400.0, 'kParamWidth': 0.0}}, 'type': 12, 'kUIParamMixOrGain': 0.0}]}}]
# ═══════════════════════════════════════════════════════════════════
# CATEGORY TEMPLATES — wartości referencyjne dla fallbacku
# ═══════════════════════════════════════════════════════════════════

CATEGORY_TEMPLATES = {
    "bass": {
        "unison": 5.0, "octave": -2.0, "detune": 0.07, "detune_wid": 87.0,
        "filter_freq": 0.38, "filter_reso": 15.0, "filter_drive": 21.0, "filter_type": "MgL24",
        "attack": 0.001, "decay": 0.88, "sustain": 0.42, "release": 0.13,
        "wavetable": "/Analog/Basic Shapes.wav",
        "fx_chain": ["EQ", "Hyper", "Reverb"],
    },
    "reese": {
        "unison": 8.0, "octave": -1.0, "detune": 0.09, "detune_wid": 90.0,
        "filter_freq": 0.46, "filter_reso": 19.0, "filter_drive": 26.0, "filter_type": "MgL18",
        "attack": 0.08, "decay": 0.65, "sustain": 0.45, "release": 0.18,
        "wavetable": "/Analog/MB Saw.wav",
        "fx_chain": ["Utils", "Hyper", "Utils", "EQ"],
    },
    "lead": {
        "unison": 7.0, "octave": 0.0, "detune": 0.05, "detune_wid": 80.0,
        "filter_freq": 0.40, "filter_reso": 11.0, "filter_drive": 25.0, "filter_type": "MgL24",
        "attack": 0.001, "decay": 0.65, "sustain": 0.15, "release": 0.20,
        "wavetable": "/Analog/Basic Shapes.wav",
        "fx_chain": ["Hyper", "EQ", "Reverb"],
    },
    "chord": {
        "unison": 4.0, "octave": 0.0, "detune": 0.05, "detune_wid": 70.0,
        "filter_freq": 0.45, "filter_reso": 5.0, "filter_drive": 12.0, "filter_type": "MgL18",
        "attack": 0.001, "decay": 0.93, "sustain": 0.22, "release": 0.17,
        "wavetable": "/Analog/Basic Mini.wav",
        "fx_chain": ["Chorus", "Delay", "Reverb", "EQ"],
    },
    "pluck": {
        "unison": 7.0, "octave": 0.0, "detune": 0.05, "detune_wid": 75.0,
        "filter_freq": 0.49, "filter_reso": 15.0, "filter_drive": 19.0, "filter_type": "MgL24",
        "attack": 0.001, "decay": 0.89, "sustain": 0.18, "release": 0.31,
        "wavetable": "/Analog/Analog_BD_Sin.wav",
        "fx_chain": ["Hyper", "Phaser", "Delay", "Reverb", "EQ"],
    },
    "pad": {
        "unison": 4.0, "octave": 0.0, "detune": 0.07, "detune_wid": 85.0,
        "filter_freq": 0.69, "filter_reso": 15.0, "filter_drive": 30.0, "filter_type": "MgL18",
        "attack": 1.2, "decay": 1.61, "sustain": 0.39, "release": 0.68,
        "wavetable": "S2 Tables/Default Shapes.wav",
        "fx_chain": ["Conv", "EQ", "Utils"],
    },
    "ambient": {
        "unison": 3.0, "octave": 0.0, "detune": 0.07, "detune_wid": 90.0,
        "filter_freq": 0.69, "filter_reso": 15.0, "filter_drive": 30.0, "filter_type": "Allpasses",
        "attack": 1.5, "decay": 2.0, "sustain": 0.5, "release": 1.0,
        "wavetable": "S2 Tables/Default Shapes.wav",
        "fx_chain": ["Conv", "Chorus", "EQ", "Utils"],
    },
    "keys": {
        "unison": 3.5, "octave": 0.0, "detune": 0.08, "detune_wid": 70.0,
        "filter_freq": 0.53, "filter_reso": 22.0, "filter_drive": 21.0, "filter_type": "MgL18",
        "attack": 0.001, "decay": 0.57, "sustain": 0.0, "release": 0.47,
        "wavetable": "S2 Tables/Default Shapes.wav",
        "fx_chain": ["EQ", "Distortion", "Utils"],
    },
    "donk": {
        "unison": 4.0, "octave": -1.0, "detune": 0.025, "detune_wid": 50.0,
        "filter_freq": 0.33, "filter_reso": 0.0, "filter_drive": 40.0, "filter_type": "MgL18",
        "attack": 0.001, "decay": 0.15, "sustain": 0.0, "release": 0.25,
        "wavetable": "/Analog/Analog_BD_Sin.wav",
        "fx_chain": ["Distortion", "EQ", "Utils"],
    },
}


# ═══════════════════════════════════════════════════════════════════
# VALIDATION
# ═══════════════════════════════════════════════════════════════════

def clamp(value, lo, hi):
    try: return max(lo, min(hi, float(value)))
    except: return lo


# Limity zakresów Serum 2 dla każdego parametru FX (klucz: typ_FX, wartość: {param: (min, max)})
FX_LIMITS = {
    0: {"kParamDrive": (0, 100), "kParamWet": (0, 100)},
    1: {"kParamRate": (0.01, 4.0), "kParamDepth": (0, 100), "kParamFeedback": (0, 100), "kParamWet": (0, 100)},
    2: {"kParamRate": (0, 4), "kParamDepth": (0, 100), "kParamFeedback": (0, 100), "kParamFreq": (50, 6000), "kParamWet": (0, 100)},
    3: {"kParamRate": (0.025, 4), "kParamDepth": (0, 20), "kParamDelay": (0, 8), "kParamFeedback": (0, 80), "kParamFilt": (200, 20000), "kParamWet": (0, 100)},
    4: {"kParamTimeL": (0.001, 0.5), "kParamTimeR": (0.001, 0.5), "kParamFeedback": (0, 80), "kParamFreq": (95, 12000), "kParamWet": (0, 100)},
    5: {"kParamThresh": (0, 1), "kParamRatio": (1, 100), "kParamMakeup": (1, 27)},
    6: {"kParamSize": (0, 100), "kParamWet": (0, 100), "kParamWidth": (0, 100), "kParamFeedback": (0, 98)},
    7: {"kParamFreq1": (21, 14000), "kParamGain1": (-24, 12), "kParamReso1": (0, 86), "kParamFreq2": (32, 20000), "kParamGain2": (-24, 12), "kParamReso2": (0, 86)},
    8: {"kParamFreq": (0, 1), "kParamReso": (0, 80), "kParamDrive": (0, 100), "kParamWet": (0, 100)},
    9: {"kParamUnison": (2, 7), "kParamDetune": (0, 100), "kParamRate": (0, 100), "kParamDimESize": (0, 100), "kParamDimEWet": (0, 100), "kParamWet": (0, 100)},
    11: {"kParamWet": (0, 100), "kParamSize": (10, 330), "kParamTone": (0, 100)},
    12: {"kParamLFXover": (60, 400), "kParamWidth": (0, 100)},
}


def validate_params(params):
    """Ensure all parameters are in valid Serum 2 ranges."""
    # OSC A
    osc = params.get("oscillator_a", params.get("osc", {}))
    if isinstance(osc, dict):
        if "kParamUnison" in osc:     osc["kParamUnison"] = clamp(osc["kParamUnison"], 1, 16)
        if "kParamVolume" in osc:     osc["kParamVolume"] = clamp(osc["kParamVolume"], 0, 1)
        if "kParamDetuneWid" in osc:  osc["kParamDetuneWid"] = clamp(osc["kParamDetuneWid"], 0, 100)
        if "kParamOctave" in osc:     osc["kParamOctave"] = clamp(osc["kParamOctave"], -4, 4)
        if "kParamDetune" in osc:     osc["kParamDetune"] = clamp(osc["kParamDetune"], 0, 1)
    
    # FILTER
    filt = params.get("filter", {})
    if isinstance(filt, dict):
        if "kParamFreq" in filt:  filt["kParamFreq"] = clamp(filt["kParamFreq"], 0, 1)
        if "kParamReso" in filt:  filt["kParamReso"] = clamp(filt["kParamReso"], 0, 100)
        if "kParamDrive" in filt: filt["kParamDrive"] = clamp(filt["kParamDrive"], 0, 100)
        # Zapewnij enable=1
        if filt.get("enabled") or filt.get("kParamEnable") == 1.0:
            filt["kParamEnable"] = 1.0
    
    # ENVELOPE
    env = params.get("envelope_amp", params.get("env", {}))
    if isinstance(env, dict):
        for k in ["kParamAttack", "kParamDecay", "kParamRelease", "kParamHold"]:
            if k in env: env[k] = clamp(env[k], 0, 10)
        if "kParamSustain" in env: env["kParamSustain"] = clamp(env["kParamSustain"], 0, 1)
    
    # FX — clamp każdy param do swojego zakresu
    fx_list = params.get("fx", [])
    if isinstance(fx_list, list):
        for fx_entry in fx_list:
            if not isinstance(fx_entry, dict): continue
            ft = fx_entry.get("type")
            if ft not in FX_LIMITS: continue
            limits = FX_LIMITS[ft]
            # Znajdź FX wewnętrzny
            for k in list(fx_entry.keys()):
                if k.startswith("FX") and isinstance(fx_entry[k], dict):
                    pp = fx_entry[k].get("plainParams", {})
                    if isinstance(pp, dict):
                        for param_name, (lo, hi) in limits.items():
                            if param_name in pp:
                                pp[param_name] = clamp(pp[param_name], lo, hi)
    
    return params


# ═══════════════════════════════════════════════════════════════════
# DETECTION — auto-detect category from audio analysis
# ═══════════════════════════════════════════════════════════════════

def detect_category(audio_analysis, hint=None):
    """Detect best category from audio features."""
    if hint:
        h = hint.lower()
        for cat in CATEGORY_TEMPLATES:
            if cat in h: return cat
    
    a = audio_analysis
    pitch_hz = a.get("fundamental_hz", 200)
    bright = a.get("brightness", 0.5)
    attack = a.get("attack_time", 0.01)
    release = a.get("release_time", 0.3)
    roughness = a.get("roughness", 0.3)
    noisiness = a.get("noisiness", 0.1)
    
    # Decision tree based on real preset patterns
    if attack > 0.5 and release > 0.8:
        return "ambient" if noisiness > 0.15 else "pad"
    if pitch_hz < 100:
        return "reese" if roughness > 0.7 else "bass"
    if pitch_hz < 200 and bright < 0.3:
        return "bass"
    if attack < 0.005 and release < 0.4 and pitch_hz > 200:
        return "pluck"
    if pitch_hz > 200 and bright > 0.5:
        return "lead"
    if pitch_hz > 150 and 0.3 < bright < 0.6:
        return "chord"
    return "lead"  # fallback


# ═══════════════════════════════════════════════════════════════════
# MAIN ENGINE — call Claude with full context
# ═══════════════════════════════════════════════════════════════════

def analyze_with_claude(audio_analysis, sound_type="lead", previous_attempt=None, diff_info=None):
    """Call Claude API with real training examples and detailed knowledge."""
    
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set")
    
    # Auto-detect category if vague hint
    category = detect_category(audio_analysis, sound_type)
    template = CATEGORY_TEMPLATES.get(category, CATEGORY_TEMPLATES["lead"])
    
    a = audio_analysis
    user_msg = f"""Analyze this audio and create Serum 2 parameters.

DETECTED CATEGORY: {category.upper()}  (based on pitch/brightness/envelope)

REFERENCE TEMPLATE for {category}:
- Unison: {template['unison']}, Octave: {template['octave']}, Detune: {template['detune']}
- Filter: freq={template['filter_freq']}, reso={template['filter_reso']}, type={template['filter_type']}
- Envelope: A={template['attack']}, D={template['decay']}, S={template['sustain']}, R={template['release']}
- Suggested wavetable: {template['wavetable']}
- Typical FX chain: {' → '.join(template['fx_chain'])}

AUDIO ANALYSIS:
- Pitch: {a.get('pitch', '—')} ({a.get('fundamental_hz', 0)} Hz)
- Spectral centroid: {a.get('spectral_centroid', 0)} Hz
- Brightness: {a.get('brightness', 0)} (0=dark, 1=bright)
- Harmonics: {a.get('harmonics', [])}
- Attack time: {a.get('attack_time', 0)}s
- Decay time: {a.get('decay_time', 0)}s
- Sustain level: {a.get('sustain_level', 0)}
- Release time: {a.get('release_time', 0)}s
- RMS: {a.get('rms', 0)}
- Roughness: {a.get('roughness', 0)} (high = use Distortion)
- Noisiness: {a.get('noisiness', 0)} (high = add noise/convolution)
"""
    
    if previous_attempt and diff_info:
        user_msg += f"""

REFINEMENT NEEDED — previous attempt issues:
{diff_info}

Adjust to better match the target. Common fixes:
- If too dark: open filter (raise kParamFreq), lower drive
- If too dull: add FXHyperD or increase unison
- If lacks space: increase reverb/delay wet
- If too thin: add FXUtils with kParamLFMono=1.0
"""

    user_msg += """

Return JSON ONLY with this structure (use the FX chain pattern from template above):
{
  "preset_name": "CAT - NAME",
  "oscillator_a": {...},
  "wt": {...},
  "filter": {"enabled": true, "kParamEnable": 1.0, ...},
  "envelope_amp": {...},
  "fx": [...],
  "reasoning": "1 sentence"
}"""

    # Build messages with FULL training examples
    messages = []
    for ex in TRAINING_EXAMPLES:
        messages.append({
            "role": "user",
            "content": f"Audio analysis:\n{json.dumps(ex['audio_analysis'], indent=2)}\n\nReturn Serum 2 JSON."
        })
        messages.append({
            "role": "assistant", 
            "content": json.dumps(ex['serum_params'], indent=2)
        })
    
    messages.append({"role": "user", "content": user_msg})
    
    # Call Claude via urllib (SDK has connection issues on Railway)
    request_body = json.dumps({
        "model": "claude-sonnet-4-5",
        "max_tokens": 3000,
        "system": SYSTEM_PROMPT,
        "messages": messages,
    }).encode("utf-8")
    
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        method="POST",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        data=request_body,
    )
    
    resp = urllib.request.urlopen(req, timeout=90)
    resp_data = json.loads(resp.read().decode())
    response_text = resp_data["content"][0]["text"].strip()
    
    # Clean markdown wrappers
    if response_text.startswith("```"):
        match = re.search(r"```(?:json)?\s*(.+?)\s*```", response_text, re.DOTALL)
        if match:
            response_text = match.group(1)
    
    start = response_text.find("{")
    end = response_text.rfind("}")
    if start >= 0 and end > start:
        response_text = response_text[start:end+1]
    
    params = json.loads(response_text)
    params = validate_params(params)
    
    return params
