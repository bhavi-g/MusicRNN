# 🎵 PunjabiGen — AI-Generated Bhangra & Punjabi Beats

PunjabiGen generates brand-new Punjabi-style instrumentals (dhol, tumbi riffs, synths, claps) using Meta’s MusicGen via Hugging Face Transformers.

## Quickstart

<pre>
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python generate_track.py --prompt "High-energy Punjabi wedding bhangra with loud dhol, tumbi riffs, bright synth, claps, 132 BPM" --seconds 30 --temp 1.2 --seed 123 --wav_out chorus.wav
ffmpeg -y -i chorus.wav -ar 44100 -b:a 192k chorus.mp3
</pre>

## Features
- Text-to-music prompts
- Adjustable duration, temperature, seed
- WAV output (MP3 via ffmpeg)

## Requirements
- Python 3.9+
- ffmpeg for MP3 conversion (optional)

## License
MIT
