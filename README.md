# MusicRNN — Character‑level LSTM (ABC‑style)

A compact, modular character‑level LSTM that learns on simple ABC‑style sequences.
The repo includes:
- `src/main.py` — quick smoke test that runs a forward/backward step on a tiny toy corpus.
- `notebooks/exploration.ipynb` — exploration notebook (original work used for reference).

## Quick start
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/main.py
## Notes
- The quickcheck uses an inline toy corpus (no external downloads), so it runs fast and offline.
- You can extend `src/main.py` to load a real ABC dataset later.
- The notebook is included only as exploration; the runnable entrypoint is `src/main.py`.
