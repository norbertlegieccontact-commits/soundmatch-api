"""
SoundMatch AI Engine v4 — Trained on REAL Serum 2 presets
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Few-shot examples from actual VoxTune preset pack.
"""

import os
import json
import re
import urllib.request
import urllib.error


SYSTEM_PROMPT = """You are an expert Serum 2 sound designer. You receive audio analysis data and return EXACT Serum 2 parameters as JSON.

CRITICAL RULES:
1. Return ONLY valid JSON — no markdown, no explanation outside JSON
2. Use EXACT Serum 2 parameter names (kParamVolume, kParamUnison, kParamDetuneWid, etc.)
3. Base your decisions on the harmonic content, brightness, envelope shape, and pitch
4. ALL float values in plainParams must be floats (not integers, not booleans)
5. FX entries MUST have "type" and "kUIParamMixOrGain" fields

SERUM 2 PARAMETER REFERENCE (from real presets):

## Oscillator (Oscillator0.plainParams)
- kParamVolume: 0.0-1.0 (volume)
- kParamOctave: -4.0 to 4.0
- kParamDetune: 0.0-1.0 (fine detune)
- kParamUnison: 1.0-16.0 (unison voice count, as FLOAT)
- kParamDetuneWid: 0.0-100.0 (unison spread width)
- kParamPan: -1.0 to 1.0

## Wavetable (Oscillator0.WTOsc0.plainParams)  
- kParamTablePos: 0-256 (position in wavetable: saw≈0, square≈64, triangle≈128, sine≈192)
- kParamWarp: 0.0-1.0
- kParamWarpMenu: "kPD_OSC", "kSync", "kBendPosNeg", "kQuantize", "kFilterLPF", "kFM"

## Filter (VoiceFilter0.plainParams)
- kParamFreq: 0.0-1.0 (cutoff frequency, log scale)
- kParamReso: 0.0-100.0 (resonance)
- kParamType: "LP12", "LP24", "HP12", "HP24", "BP", "Notch"

## Envelope (Env0.plainParams)
- kParamAttack: 0.0-1.0 (0=instant, 1=10sec)
- kParamDecay: 0.0-1.0
- kParamSustain: 0.0-1.0
- kParamRelease: 0.0-1.0

## FX (in FXRack0.FX list — EACH entry needs type + kUIParamMixOrGain)
FX Types: FXDistortion=0, FXPhaser=2, FXChorus=3, FXDelay=4, FXComp=5, FXReverb=6, FXEQ=7, FXFilter=8

Example FX entry:
{"FXReverb": {"plainParams": {"kParamWet": 30.0, "kParamSize": 55.0, "kParamType": "kHall"}}, "kUIParamMixOrGain": 0.0, "type": 6}

SOUND TYPE PATTERNS (from real presets):
- BASS: Low pitch, dark (brightness<0.15), 1-9 unison, low cutoff, short release, distortion common
- LEAD: Mid-high pitch, bright (0.2-0.5), 2-7 unison, medium cutoff, moderate FX  
- PLUCK: Fast attack+decay, low sustain, triangle/sine wave, reverb/delay common
- PAD: Slow attack (>0.3), high sustain, wide unison, heavy reverb+chorus
"""

# Real training pairs from VoxTune preset pack
TRAINING_EXAMPLES = [
  {
    "audio_analysis": {
      "pitch": "C\u266f2",
      "fundamental_hz": 70.1,
      "spectral_centroid": 437.0,
      "brightness": 0.05,
      "rms": 0.1263,
      "noisiness": 1.0,
      "attack_time": 0.4144,
      "harmonics": [
        1.0,
        0.284,
        0.042,
        0.437,
        0.173,
        0.019
      ]
    },
    "serum_params": {
      "name": "BS-  Deep HOUSE",
      "osc": {
        "kParamDetune": 0.15,
        "kParamOctave": -1.0,
        "kParamUnison": 9.0,
        "kParamVolume": 0.41
      },
      "wt": {
        "kParamTablePos": 121.79
      },
      "filter": {
        "kParamEnable": 1.0,
        "kParamFreq": 0.1,
        "kParamReso": 10.0,
        "kParamType": "MgL18"
      },
      "env": {
        "kParamAttack": 0.0,
        "kParamCurve1": 40.0,
        "kParamCurve2": 60.0,
        "kParamCurve3": 60.0,
        "kParamDecay": 1.28,
        "kParamRelease": 0.14,
        "kParamSustain": 0.51
      },
      "fx": [
        {
          "FXHyperD": {
            "kParamDimESize": 15.79,
            "kParamRate": 40.0,
            "kParamUnison": 4.0,
            "kParamWet": 0.0
          },
          "type": 9
        },
        {
          "FXDistortion": {
            "kParamDrive": 18.15,
            "kParamFreq": 0.47,
            "kParamMode": "kDownsample",
            "kParamWet": 0.0
          },
          "type": 0
        },
        {
          "FXReverb": {
            "kParamDelay": 0.0,
            "kParamFreq": 53.51,
            "kParamFreqB": 56.05,
            "kParamSize": 0.0,
            "kParamType": "kHall",
            "kParamWet": 10.99,
            "kParamWidth": 20.0
          },
          "type": 6
        }
      ]
    }
  },
  {
    "audio_analysis": {
      "pitch": "G\u266f2",
      "fundamental_hz": 102.6,
      "spectral_centroid": 278.0,
      "brightness": 0.03,
      "rms": 0.3031,
      "noisiness": 0.0,
      "attack_time": 0.0401,
      "harmonics": [
        1.0,
        3.498,
        0.309,
        0.606,
        0.262,
        0.02
      ]
    },
    "serum_params": {
      "name": "BS - KNOCK KNOCK",
      "osc": {
        "kParamOctave": -2.0,
        "kParamVolume": 0.57
      },
      "wt": {
        "kParamRandomPhase": 0.0,
        "kParamWarp": 0.45,
        "kParamWarpMenu": "kPD_OSC"
      },
      "filter": {
        "kParamDrive": 15.79,
        "kParamEnable": 1.0,
        "kParamFreq": 0.16,
        "kParamReso": 10.0
      },
      "env": {
        "kParamAttack": 0.0,
        "kParamCurve1": 40.0,
        "kParamCurve2": 60.0,
        "kParamCurve3": 60.0,
        "kParamDecay": 1.29,
        "kParamRelease": 0.07,
        "kParamSustain": 0.16
      },
      "fx": [
        {
          "FXChorus": {
            "kParamFeedback": 9.5,
            "kParamFilt": 1233.96,
            "kParamWet": 0.0
          },
          "type": 3
        },
        {
          "FXReverb": {
            "kParamDelay": 1.7,
            "kParamFreq": 25.44,
            "kParamFreqB": 81.93,
            "kParamSize": 0.0,
            "kParamType": "kHall",
            "kParamWet": 0.0,
            "kParamWidth": 20.0
          },
          "type": 6
        }
      ]
    }
  },
  {
    "audio_analysis": {
      "pitch": "D3",
      "fundamental_hz": 146.8,
      "spectral_centroid": 984.0,
      "brightness": 0.12,
      "rms": 0.0535,
      "noisiness": 0.0,
      "attack_time": 1.0762,
      "harmonics": [
        1.0,
        189.587,
        0.812,
        64.106,
        1.123,
        36.984
      ]
    },
    "serum_params": {
      "name": "BS - Brassline",
      "osc": {
        "kParamDetune": 0.1,
        "kParamDetuneWid": 70.09,
        "kParamUnison": 3.0
      },
      "wt": {
        "kParamTablePos": 41.26,
        "kParamWarp": 0.28,
        "kParamWarpMenu": "kDistTube"
      },
      "filter": {
        "kParamDrive": 28.42,
        "kParamEnable": 1.0,
        "kParamFreq": 0.48
      },
      "env": {
        "kParamAttack": 0.07,
        "kParamCurve1": 50.0,
        "kParamCurve2": 66.6,
        "kParamCurve3": 66.6,
        "kParamDecay": 1.15,
        "kParamRelease": 0.36,
        "kParamSustain": 0.41
      },
      "fx": [
        {
          "FXDistortion": {
            "kParamDrive": 26.75,
            "kParamLevelOut": 0.61
          },
          "type": 0
        },
        {
          "FXUtils": {
            "kParamWidth": 0.0
          },
          "type": 12
        },
        {
          "FXDelay": {
            "kParamBW": 1.66,
            "kParamFeedback": 46.67,
            "kParamFreq": 964.27,
            "kParamMode": 1.0,
            "kParamOffsetL": 0.94,
            "kParamOffsetR": 1.06,
            "kParamTimeL": 0.05,
            "kParamTimeR": 0.05,
            "kParamWet": 20.53
          },
          "type": 4
        },
        {
          "FXConv": {
            "kParamSize": 181.82,
            "kParamTone": 57.89,
            "kParamWet": 27.37
          },
          "type": 11
        }
      ]
    }
  },
  {
    "audio_analysis": {
      "pitch": "D3",
      "fundamental_hz": 146.8,
      "spectral_centroid": 1724.0,
      "brightness": 0.22,
      "rms": 0.145,
      "noisiness": 0.0,
      "attack_time": 1.2798,
      "harmonics": [
        1.0,
        458.458,
        1.033,
        0.201,
        0.523,
        18.903
      ]
    },
    "serum_params": {
      "name": "LD - CAMOUFLAGE",
      "osc": {
        "kParamVolume": 0.32
      },
      "wt": {
        "kParamWarp": 0.76
      },
      "filter": {
        "kParamEnable": 1.0,
        "kParamFreq": 0.5,
        "kParamReso": 0.0,
        "kParamType": "H12"
      },
      "env": {
        "kParamAttack": 0.0,
        "kParamCurve1": 49.95,
        "kParamCurve2": 66.6,
        "kParamCurve3": 66.6,
        "kParamRelease": 2.16
      },
      "fx": [
        {
          "FXHyperD": {
            "kParamDimESize": 0.0,
            "kParamDimEWet": 69.3,
            "kParamRate": 40.0,
            "kParamUnison": 4.0,
            "kParamWet": 0.0
          },
          "type": 9
        },
        {
          "FXDistortion": {
            "kParamDrive": 67.54
          },
          "type": 0
        },
        {
          "FXDelay": {
            "kParamBW": 6.75,
            "kParamFeedback": 40.0,
            "kParamMode": 1.0,
            "kParamWet": 0.0
          },
          "type": 4
        },
        {
          "FXReverb": {
            "kParamDelay": 30.62,
            "kParamFreqB": 35.0,
            "kParamSize": 35.0,
            "kParamType": "kHall",
            "kParamWet": 0.0,
            "kParamWidth": 20.0
          },
          "type": 6
        },
        {
          "FXFilter": {
            "kParamFreq": 0.46,
            "kParamReso": 44.74,
            "kParamType": "MgL12",
            "kParamWet": 0.0
          },
          "type": 8
        },
        {
          "FXEQ": {
            "kParamFreq1": 99.08,
            "kParamFreq2": 2040.54,
            "kParamGain1": -24.0,
            "kParamReso1": 60.0,
            "kParamReso2": 60.0
          },
          "type": 7
        }
      ]
    }
  },
  {
    "audio_analysis": {
      "pitch": "D3",
      "fundamental_hz": 146.8,
      "spectral_centroid": 1926.0,
      "brightness": 0.24,
      "rms": 0.0661,
      "noisiness": 0.0,
      "attack_time": 2.4058,
      "harmonics": [
        1.0,
        458.39,
        1.419,
        120.109,
        3.105,
        91.093
      ]
    },
    "serum_params": {
      "name": "LD - AFTERLIFE",
      "osc": {
        "kParamVolume": 0.16
      },
      "wt": {
        "kParamInitialPhase": 180.94,
        "kParamRandomPhase": 0.0,
        "kParamTablePos": 136.15
      },
      "filter": {
        "kParamEnable": 1.0,
        "kParamFreq": 0.71,
        "kParamReso": 0.0,
        "kParamType": "H12"
      },
      "env": {
        "kParamAttack": 0.03,
        "kParamCurve1": 50.16,
        "kParamCurve2": 64.2,
        "kParamCurve3": 60.0,
        "kParamDecay": 0.24,
        "kParamRelease": 0.35,
        "kParamSustain": 0.0
      },
      "fx": [
        {
          "FXEQ": {
            "kParamFreq1": 209.62,
            "kParamFreq2": 1653.56,
            "kParamGain1": -2.72,
            "kParamGain2": 1.82,
            "kParamReso1": 36.87,
            "kParamReso2": 41.85,
            "kParamType2": 1.0
          },
          "type": 7
        },
        {
          "FXFilter": {
            "kParamFreq": 0.43,
            "kParamType": "MgL18"
          },
          "type": 8
        },
        {
          "FXHyperD": {
            "kParamDimESize": 13.6,
            "kParamRate": 40.0,
            "kParamUnison": 4.0,
            "kParamWet": 0.0
          },
          "type": 9
        },
        {
          "FXDistortion": {
            "kParamDrive": 36.55
          },
          "type": 0
        },
        {
          "FXDelay": {
            "kParamBW": 4.01,
            "kParamFeedback": 40.0,
            "kParamFreq": 1006.8,
            "kParamMode": 1.0,
            "kParamWet": 24.56
          },
          "type": 4
        },
        {
          "FXReverb": {
            "kParamDelay": 61.23,
            "kParamFeedback": 18.67,
            "kParamFreq": 23.75,
            "kParamFreqB": 61.31,
            "kParamSize": 43.21,
            "kParamType": "kHall",
            "kParamWet": 32.89,
            "kParamWidth": 87.57
          },
          "type": 6
        }
      ]
    }
  },
  {
    "audio_analysis": {
      "pitch": "D3",
      "fundamental_hz": 146.8,
      "spectral_centroid": 1955.0,
      "brightness": 0.24,
      "rms": 0.0552,
      "noisiness": 0.0,
      "attack_time": 0.7599,
      "harmonics": [
        1.0,
        380.494,
        1.871,
        155.649,
        5.469,
        125.157
      ]
    },
    "serum_params": {
      "name": "PL - ETERNITY",
      "osc": {},
      "wt": {},
      "filter": {
        "kParamEnable": 1.0,
        "kParamFreq": 0.76,
        "kParamReso": 10.0
      },
      "env": {
        "kParamAttack": 0.0,
        "kParamCurve1": 40.0,
        "kParamCurve2": 60.0,
        "kParamCurve3": 60.0,
        "kParamDecay": 0.88,
        "kParamRelease": 0.45,
        "kParamSustain": 0.69
      },
      "fx": [
        {
          "FXHyperD": {
            "kParamDimESize": 7.02,
            "kParamDimEWet": 37.28,
            "kParamRate": 40.0,
            "kParamUnison": 4.0,
            "kParamWet": 21.49
          },
          "type": 9
        },
        {
          "FXPhaser": {
            "kParamFeedback": 82.19,
            "kParamWet": 15.35
          },
          "type": 2
        },
        {
          "FXDelay": {
            "kParamBW": 0.75,
            "kParamFeedback": 40.0,
            "kParamFreq": 1395.47,
            "kParamTimeR": 0.13,
            "kParamWet": 36.58
          },
          "type": 4
        },
        {
          "FXReverb": {
            "kParamDelay": 27.63,
            "kParamFreq": 3.07,
            "kParamFreqB": 35.0,
            "kParamSize": 71.4,
            "kParamType": "kHall",
            "kParamWet": 20.0,
            "kParamWidth": 20.0
          },
          "type": 6
        },
        {
          "FXEQ": {
            "kParamFreq1": 209.62,
            "kParamFreq2": 2040.54,
            "kParamReso1": 60.0,
            "kParamReso2": 60.0
          },
          "type": 7
        }
      ]
    }
  },
  {
    "audio_analysis": {
      "pitch": "D3",
      "fundamental_hz": 146.8,
      "spectral_centroid": 896.0,
      "brightness": 0.11,
      "rms": 0.075,
      "noisiness": 0.0,
      "attack_time": 1.316,
      "harmonics": [
        1.0,
        393.579,
        3.62,
        433.965,
        15.574,
        298.547
      ]
    },
    "serum_params": {
      "name": "PL - HEARTBREAK",
      "osc": {
        "kParamDetune": 0.07,
        "kParamUnison": 3.0,
        "kParamVolume": 0.06
      },
      "wt": {
        "kParamTablePos": 65.87,
        "kParamWarp": 0.32,
        "kParamWarpMenu": "kSync",
        "kParamWarpVar": 1.0
      },
      "filter": {
        "kParamEnable": 1.0,
        "kParamFreq": 0.54,
        "kParamReso": 0.0,
        "kParamType": "LadderMg"
      },
      "env": {
        "kParamAttack": 0.0,
        "kParamCurve1": 49.95,
        "kParamCurve2": 69.54,
        "kParamCurve3": 66.6,
        "kParamDecay": 1.4,
        "kParamRelease": 1.35,
        "kParamSustain": 0.18
      },
      "fx": [
        {
          "FXHyperD": {
            "kParamDetune": 25.44,
            "kParamDimESize": 0.0,
            "kParamRate": 40.0,
            "kParamUnison": 4.0,
            "kParamWet": 20.61
          },
          "type": 9
        },
        {
          "FXDistortion": {
            "kParamBW": 0.07,
            "kParamDrive": 16.67,
            "kParamFreq": 0.58,
            "kParamLPHP": 46.5,
            "kParamMode": "kDiode2",
            "kParamPrePost": 2.0,
            "kParamWet": 4.82
          },
          "type": 0
        },
        {
          "FXPhaser": {
            "kParamDepth2": 0.07,
            "kParamFeedback": 80.0,
            "kParamWet": 0.0
          },
          "type": 2
        },
        {
          "FXChorus": {
            "kParamFeedback": 9.5,
            "kParamWet": 11.84
          },
          "type": 3
        },
        {
          "FXDelay": {
            "kParamBW": 1.48,
            "kParamFeedback": 50.96,
            "kParamFreq": 1027.02,
            "kParamMode": 1.0,
            "kParamTimeL": 0.04,
            "kParamTimeR": 0.05,
            "kParamWet": 2.81
          },
          "type": 4
        },
        {
          "FXReverb": {
            "kParamDelay": 21.46,
            "kParamFreq": 25.88,
            "kParamFreqB": 35.0,
            "kParamSize": 30.61,
            "kParamType": "kHall",
            "kParamWet": 38.86,
            "kParamWidth": 20.0
          },
          "type": 6
        },
        {
          "FXEQ": {
            "kParamFreq1": 499.95,
            "kParamFreq2": 1704.67,
            "kParamGain1": -20.84,
            "kParamGain2": 11.56,
            "kParamReso1": 30.61,
            "kParamReso2": 40.26,
            "kParamType2": 2.0
          },
          "type": 7
        },
        {
          "FXComp": {
            "kParamAttack": 90.09,
            "kParamCompensatedWetDry": 0.0,
            "kParamGain0": 4.6,
            "kParamGain2": 4.6,
            "kParamMakeup": 8.77,
            "kParamRatioBelow": 0.75,
            "kParamRelease": 39.67,
            "kParamThreshUD0": 28.57,
            "kParamThreshUD1": 0.0,
            "kParamThreshUD2": 0.0,
            "kParamWet": 87.28
          },
          "type": 5
        },
        {
          "FXFilter": {
            "kParamDrive": 9.65,
            "kParamFreq": 0.42,
            "kParamType": "H18"
          },
          "type": 8
        }
      ]
    }
  }
]

# Full dataset summary for reference
DATASET_SUMMARY = [
  {
    "name": "LD-CAMOUFLAGE",
    "pitch": "D3",
    "brightness": 0.22,
    "attack": 1.2798,
    "osc_params": {
      "kParamVolume": 0.32
    },
    "wt_pos": 0,
    "fx_count": 6
  },
  {
    "name": "BS-DEEP_HOUSE",
    "pitch": "C\u266f2",
    "brightness": 0.05,
    "attack": 0.4144,
    "osc_params": {
      "kParamOctave": -1.0,
      "kParamUnison": 9.0,
      "kParamVolume": 0.41
    },
    "wt_pos": 121.79,
    "fx_count": 3
  },
  {
    "name": "BS-_KNOCK_KNOCK",
    "pitch": "G\u266f2",
    "brightness": 0.03,
    "attack": 0.0401,
    "osc_params": {
      "kParamOctave": -2.0,
      "kParamVolume": 0.57
    },
    "wt_pos": 0,
    "fx_count": 2
  },
  {
    "name": "bs-_brassline",
    "pitch": "D3",
    "brightness": 0.12,
    "attack": 1.0762,
    "osc_params": {
      "kParamDetuneWid": 70.09,
      "kParamUnison": 3.0
    },
    "wt_pos": 41.26,
    "fx_count": 4
  },
  {
    "name": "BS-DONK",
    "pitch": "C2",
    "brightness": 0.02,
    "attack": 0.3913,
    "osc_params": {
      "kParamOctave": -2.0,
      "kParamVolume": 0.83
    },
    "wt_pos": 0,
    "fx_count": 4
  },
  {
    "name": "BS-RIOT",
    "pitch": "C2",
    "brightness": 0.46,
    "attack": 0.7796,
    "osc_params": {
      "kParamOctave": -2.0
    },
    "wt_pos": 0,
    "fx_count": 4
  },
  {
    "name": "BS-SHAKE_IT",
    "pitch": "C2",
    "brightness": 0.14,
    "attack": 1.0151,
    "osc_params": {
      "kParamOctave": -2.0
    },
    "wt_pos": 210.14,
    "fx_count": 6
  },
  {
    "name": "BS-SLOW_DRIFT",
    "pitch": "C2",
    "brightness": 0.12,
    "attack": 1.8602,
    "osc_params": {
      "kParamOctave": -2.0,
      "kParamUnison": 5.0
    },
    "wt_pos": 0,
    "fx_count": 4
  },
  {
    "name": "LD-AFTERLIFE",
    "pitch": "D3",
    "brightness": 0.24,
    "attack": 2.4058,
    "osc_params": {
      "kParamVolume": 0.16
    },
    "wt_pos": 136.15,
    "fx_count": 6
  },
  {
    "name": "LD-ROCKET",
    "pitch": "D4",
    "brightness": 0.26,
    "attack": 0.0359,
    "osc_params": {
      "kParamOctave": 1.0
    },
    "wt_pos": 161.89,
    "fx_count": 5
  },
  {
    "name": "LD-WHISTLE",
    "pitch": "D4",
    "brightness": 0.12,
    "attack": 1.311,
    "osc_params": {
      "kParamOctave": 1.0,
      "kParamUnison": 7.0,
      "kParamVolume": 0.0
    },
    "wt_pos": 87.12,
    "fx_count": 8
  },
  {
    "name": "PL-_ETERNITY",
    "pitch": "D3",
    "brightness": 0.24,
    "attack": 0.7599,
    "osc_params": {},
    "wt_pos": 0,
    "fx_count": 5
  },
  {
    "name": "PL-_HEARTBREAK",
    "pitch": "D3",
    "brightness": 0.11,
    "attack": 1.316,
    "osc_params": {
      "kParamUnison": 3.0,
      "kParamVolume": 0.06
    },
    "wt_pos": 65.87,
    "fx_count": 9
  },
  {
    "name": "PL-_MAGE",
    "pitch": "D4",
    "brightness": 0.15,
    "attack": 2.1446,
    "osc_params": {
      "kParamDetuneWid": 45.61,
      "kParamOctave": 1.0,
      "kParamUnison": 7.0,
      "kParamVolume": 0.0
    },
    "wt_pos": 63.63,
    "fx_count": 6
  },
  {
    "name": "PL-_PORTA",
    "pitch": "D3",
    "brightness": 0.23,
    "attack": 0.7646,
    "osc_params": {
      "kParamVolume": 0.43
    },
    "wt_pos": 0,
    "fx_count": 6
  },
  {
    "name": "PL-_SOFTY",
    "pitch": "D3",
    "brightness": 0.31,
    "attack": 0.7742,
    "osc_params": {
      "kParamDetuneWid": 44.3,
      "kParamUnison": 4.0,
      "kParamVolume": 0.2
    },
    "wt_pos": 0,
    "fx_count": 7
  }
]



def clamp(value, lo, hi):
    try: return max(lo, min(hi, float(value)))
    except: return lo


def validate_params(params):
    """Ensure all parameters are in valid Serum 2 ranges."""
    osc = params.get("oscillator_a", params.get("osc", {}))
    if "kParamUnison" in osc:
        osc["kParamUnison"] = clamp(osc["kParamUnison"], 1, 16)
    if "kParamVolume" in osc:
        osc["kParamVolume"] = clamp(osc["kParamVolume"], 0, 1)
    if "kParamDetuneWid" in osc:
        osc["kParamDetuneWid"] = clamp(osc["kParamDetuneWid"], 0, 100)
    
    filt = params.get("filter", {})
    if "kParamFreq" in filt:
        filt["kParamFreq"] = clamp(filt["kParamFreq"], 0, 1)
    if "kParamReso" in filt:
        filt["kParamReso"] = clamp(filt["kParamReso"], 0, 100)
    
    env = params.get("envelope_amp", params.get("env", {}))
    for k in ["kParamAttack", "kParamDecay", "kParamSustain", "kParamRelease"]:
        if k in env:
            env[k] = clamp(env[k], 0, 1)
    
    return params


def analyze_with_claude(audio_analysis, sound_type="lead", previous_attempt=None, diff_info=None):
    """Call Claude API with real training examples."""
    
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set")
    
    # Build user message
    a = audio_analysis
    user_msg = f"""Analyze this audio and create Serum 2 parameters.

AUDIO ANALYSIS:
- Sound type: {sound_type}
- Pitch: {a.get('pitch', '—')} ({a.get('fundamental_hz', 0)} Hz)
- Spectral centroid: {a.get('spectral_centroid', 0)} Hz
- Brightness: {a.get('brightness', 0)} (0=dark, 1=bright)
- Harmonics: {a.get('harmonics', [])}
- Attack time: {a.get('attack_time', 0)}s
- Decay time: {a.get('decay_time', 0)}s
- Sustain level: {a.get('sustain_level', 0)}
- Release time: {a.get('release_time', 0)}s
- RMS: {a.get('rms', 0)}
- Roughness: {a.get('roughness', 0)}
- Noisiness: {a.get('noisiness', 0)}
"""

    if previous_attempt and diff_info:
        user_msg += f"""
REFINEMENT: Your previous attempt had these issues:
{diff_info}

Previous params: {json.dumps(previous_attempt, indent=2)[:500]}
Adjust to better match the target.
"""

    user_msg += """
Return a JSON object with these EXACT keys using REAL Serum 2 parameter names:
{
  "preset_name": "XX - NAME",
  "oscillator_a": {"kParamVolume": float, "kParamOctave": float, "kParamDetune": float, "kParamUnison": float, "kParamDetuneWid": float},
  "wt": {"kParamTablePos": float, "kParamWarpMenu": "string"},
  "filter": {"enabled": bool, "kParamFreq": float, "kParamReso": float, "kParamType": "string"},
  "envelope_amp": {"kParamAttack": float, "kParamDecay": float, "kParamSustain": float, "kParamRelease": float},
  "fx": [
    {"FXClassName": {"plainParams": {...}}, "kUIParamMixOrGain": 0.0, "type": int},
    ...
  ],
  "reasoning": "brief explanation"
}
Return ONLY JSON."""

    # Build messages with REAL few-shot examples
    messages = []
    for ex in TRAINING_EXAMPLES:
        messages.append({
            "role": "user",
            "content": f"AUDIO ANALYSIS:\n{json.dumps(ex['audio_analysis'], indent=2)}\n\nReturn Serum 2 preset JSON."
        })
        messages.append({
            "role": "assistant", 
            "content": json.dumps(ex['serum_params'], indent=2)
        })
    
    messages.append({"role": "user", "content": user_msg})
    
    # Call Claude via urllib (SDK has connection issues on Railway)
    request_body = json.dumps({
        "model": "claude-sonnet-4-5",
        "max_tokens": 2000,
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
    
    resp = urllib.request.urlopen(req, timeout=60)
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
