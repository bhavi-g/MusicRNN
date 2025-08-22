# 🎶 MusicRNN — A Small Collection of AI Music Experiments

This repo contains two self-contained projects:
- **classical_rnn/** — character-level LSTM that generates ABC-notation tunes.
- **punjabi_gen/** — Punjabi/Bhangra style text-to-music generator built on MusicGen (Transformers).

## Repo Map
- `classical_rnn/src/` entrypoints: `main.py`, `train_and_generate.py`
- `punjabi_gen/generate_track.py` main generator script
- Separate `requirements.txt` per module

## Quickstart (pick one)
```bash
# Classical RNN
python3 -m venv .venv && source .venv/bin/activate
pip install -r classical_rnn/requirements.txt
python classical_rnn/src/main.py
python classical_rnn/src/train_and_generate.py
```
```bash
# PunjabiGen (MusicGen)
python3 -m venv .venv && source .venv/bin/activate
pip install -r punjabi_gen/requirements.txt
python punjabi_gen/generate_track.py --prompt "Fast bhangra with powerful dhol and catchy tumbi, bright synth lead, 132 BPM" --seconds 20 --temp 1.2 --seed 123 --wav_out punjabi_gen/examples/bhangra.wav
afplay punjabi_gen/examples/bhangra.wav
```
