# SoundMatch API

AI-powered audio-to-Serum-2-preset engine.

## Endpoints
- `POST /api/analyze` — Upload audio → get analysis + preset
- `GET /api/download/{id}` — Download .SerumPreset
- `GET /api/download/{id}/wavetable` — Download wavetable
- `GET /api/health` — Health check

## Deploy on Railway
1. Push to GitHub
2. Connect repo on railway.app
3. Auto-deploys with Dockerfile
