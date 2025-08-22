# 🎵 PunjabiGen — AI-Generated Bhangra & Punjabi Beats

Generate brand-new Punjabi-style instrumentals (dhol, tumbi, synths, claps) using Meta MusicGen (Transformers).

## Quickstart
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r punjabi_gen/requirements.txt
python punjabi_gen/generate_track.py --prompt "High-energy Punjabi wedding bhangra with loud dhol, tumbi riffs, bright synth, claps, 132 BPM" --seconds 20 --temp 1.2 --seed 123 --wav_out punjabi_gen/examples/bhangra.wav
afplay punjabi_gen/examples/bhangra.wav  # macOS player
```
