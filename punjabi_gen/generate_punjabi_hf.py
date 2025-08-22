import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration
from scipy.io.wavfile import write as wavwrite
import numpy as np

# Model + processor
MODEL_ID = "facebook/musicgen-small"
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = MusicgenForConditionalGeneration.from_pretrained(MODEL_ID)

# Force CPU for widest compatibility; set to "cuda" if you have a GPU
device = "cpu"
model = model.to(device)

# Prompt — tweak to taste
prompt = ("Punjabi bhangra with energetic dhol, punchy tumbi riffs, "
          "bright synth lead, claps, 128 BPM, dance floor vibe")

# Prepare inputs
inputs = processor(text=[prompt], padding=True, return_tensors="pt").to(device)

# Generate ~30s (adjust 'max_new_tokens' to change length; ~10 tokens ≈ ~1 sec)
# keep it conservative on CPU; raise for longer clips
audio_tokens = model.generate(
    **inputs,
    do_sample=True,
    temperature=1.2,           # raise for wilder; lower for safer
    max_new_tokens=300,        # ~30 sec
)

# Decode to audio (shape: [batch, channels, samples])
audio = processor.batch_decode(audio_tokens, device=device, sampling_rate=32000)[0]
# Convert to int16 WAV
audio_np = np.clip(audio * 32767.0, -32768, 32767).astype(np.int16)

out_path = "punjabi_bhangra.wav"
wavwrite(out_path, 32000, audio_np.T)  # transpose to [samples, channels] if needed

print(f"✅ Saved: {out_path}")
