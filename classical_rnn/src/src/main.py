import numpy as np
import torch
import torch.nn as nn

# --- Tiny toy "music-like" corpus so we don't need mitdeeplearning or matplotlib ---
toy_corpus = """
X:1
T:ToyTune
M:4/4
K:C
CDEF GABc|cBAG FEDC|
X:2
T:Another
M:4/4
K:G
GABc d2 cB|AGFE DCBA|
""".strip()

vocab = sorted(set(toy_corpus))
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

def vectorize_string(s):
    return np.array([char2idx[c] for c in s], dtype=np.int32)

vectorized = vectorize_string(toy_corpus)

def get_batch(vec, seq_length, batch_size):
    n = vec.shape[0] - 1
    idx = np.random.choice(n - seq_length, batch_size)
    input_batch = [vec[i:i+seq_length] for i in idx]
    target_batch = [vec[i+1:i+seq_length+1] for i in idx]
    return np.array(input_batch), np.array(target_batch)

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def init_hidden(self, batch_size, device):
        return (torch.zeros(1, batch_size, self.hidden_size, device=device),
                torch.zeros(1, batch_size, self.hidden_size, device=device))

    def forward(self, x, state=None, return_state=False):
        x = self.embedding(x)
        if state is None:
            state = self.init_hidden(x.size(0), x.device)
        out, state = self.lstm(x, state)
        out = self.fc(out)
        return (out, state) if return_state else out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

embedding_dim = 128
hidden_size   = 256
model = LSTMModel(vocab_size=len(vocab), embedding_dim=embedding_dim, hidden_size=hidden_size).to(device)

seq_length = 40
batch_size = 8
x_np, y_np = get_batch(vectorized, seq_length, batch_size)
x = torch.tensor(x_np, dtype=torch.long, device=device)
y = torch.tensor(y_np, dtype=torch.long, device=device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

model.train()
optimizer.zero_grad()
y_hat = model(x)  # [B, T, V]
loss = criterion(y_hat.reshape(-1, y_hat.size(-1)), y.reshape(-1))
loss.backward()
optimizer.step()

print("Quickcheck OK ✅")
print(f"vocab_size={len(vocab)}, batch={batch_size}x{seq_length}, loss={loss.item():.4f}")
