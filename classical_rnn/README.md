# 🎼 Classical RNN — Character-level LSTM (ABC Notation)

Minimal LSTM that learns simple ABC-style tunes; includes a quick smoke test and a short trainer/sampler.

## Quickstart
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r classical_rnn/requirements.txt
python classical_rnn/src/main.py
python classical_rnn/src/train_and_generate.py
head -40 classical_rnn/out/generated_tune.abc
```
