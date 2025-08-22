import os, numpy as np, torch, torch.nn as nn

# --- Small built-in ABC corpus (several short tunes) ---
ABC_CORPUS = """
X:1
T:Toy Reel
M:4/4
K:D
DFA d2|AFD A2|d2 fd a2|afd e2|
d2 fd a2|afd e2|d2 fd a2|dAF D2||

X:2
T:Simple Jig
M:6/8
K:G
GAG BAG|dBG A3|GAG BAG|dBG G3|
G2 A B2|d2 B A2|GAG BAG|dBG G3||

X:3
T:Easy March
M:4/4
K:C
C2 E G2 c|BAGF E2 C2|C2 E G2 c|BAGF E4||

X:4
T:Air
M:3/4
K:Am
A2 c e2|a2 g e2|c2 B A2|E4 E2|
A2 c e2|a2 g e2|c2 B A2|E6||

X:5
T:Polka
M:2/4
K:F
F A c A|G F E D|C F A c|B A G F|
F A c A|G F E D|C F A c|F2 F2||
""".strip()

# --- Vocab, vectorization (with fallback for unknown chars) ---
vocab = sorted(set(ABC_CORPUS))
char2idx = {ch: i for i, ch in enumerate(vocab)}
idx2char = np.array(vocab)
FALLBACK_ID = 0  # graceful fallback

def vec(s: str) -> np.ndarray:
    return np.array([char2idx.get(c, FALLBACK_ID) for c in s], dtype=np.int64)

data = vec(ABC_CORPUS)

def get_batch(vec_data, seq_len, batch_size):
    n = len(vec_data) - 1 - seq_len
    idx = np.random.choice(n, batch_size, replace=True)
    X = np.stack([vec_data[i:i+seq_len] for i in idx])
    Y = np.stack([vec_data[i+1:i+seq_len+1] for i in idx])
    return torch.from_numpy(X), torch.from_numpy(Y)

# --- Model ---
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, emb=128, hid=256, layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb)
        self.lstm = nn.LSTM(emb, hid, num_layers=layers, batch_first=True)
        self.fc = nn.Linear(hid, vocab_size)
    def forward(self, x, state=None):
        x = self.embedding(x)
        out, state = self.lstm(x, state)
        return self.fc(out), state

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = LSTMModel(len(vocab)).to(device)
opt    = torch.optim.Adam(model.parameters(), lr=3e-3)
crit   = nn.CrossEntropyLoss()

# --- Quick train ---
seq_len = 120
batch   = 64
steps   = 600   # increase (1500-3000) if you want nicer tunes

model.train()
for step in range(1, steps+1):
    X, Y = get_batch(data, seq_len, batch)
    X, Y = X.to(device), Y.to(device)
    opt.zero_grad()
    logits, _ = model(X)
    loss = crit(logits.reshape(-1, logits.size(-1)), Y.reshape(-1))
    loss.backward()
    opt.step()
    if step % 100 == 0:
        print(f"[step {step:4d}] loss={loss.item():.4f}")

@torch.no_grad()
def sample(model, start_text, length=1000, temperature=0.9):
    model.eval()
    ids = torch.tensor([char2idx.get(c, FALLBACK_ID) for c in start_text],
                       dtype=torch.long, device=device).unsqueeze(0)
    _, state = model(ids)
    last_id = ids[:, -1:]
    out_chars = list(start_text)
    for _ in range(length):
        logits, state = model(last_id, state)
        logits = logits[:, -1, :] / temperature
        probs  = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        last_id = next_id
        out_chars.append(idx2char[next_id.item()])
    return "".join(out_chars)

# SAFE PREFIX (no digits other than those in corpus; unknowns handled anyway)
prefix = "X:1\nT:Generated\nM:4/4\nK:C\n"
generated = sample(model, prefix, length=1200, temperature=0.9)

os.makedirs("out", exist_ok=True)
out_path = "out/generated_tune.abc"
with open(out_path, "w") as f:
    f.write(generated)

print("\n--- SAMPLE (first 400 chars) ---")
print(generated[:400])
print(f"\nSaved full tune to {out_path}")
