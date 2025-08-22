import argparse, numpy as np, torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration
from scipy.io.wavfile import write as wavwrite

def main():
    ap = argparse.ArgumentParser(description="Generate Punjabi-style music with MusicGen (Transformers).")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--seconds", type=int, default=30)
    ap.add_argument("--temp", type=float, default=1.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", default="facebook/musicgen-small")
    ap.add_argument("--wav_out", default="clip.wav")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(args.model)
    model = MusicgenForConditionalGeneration.from_pretrained(args.model).to(device)

    inputs = processor(text=[args.prompt], padding=True, return_tensors="pt").to(device)
    max_new_tokens = int(args.seconds * 10)
    tokens = model.generate(**inputs, do_sample=True, temperature=float(args.temp), max_new_tokens=max_new_tokens)

    audio = processor.batch_decode(tokens, device=device, sampling_rate=32000)[0]
    audio_i16 = np.clip(audio * 32767.0, -32768, 32767).astype(np.int16)
    wavwrite(args.wav_out, 32000, audio_i16.T)
    print(args.wav_out)

if __name__ == "__main__":
    main()
