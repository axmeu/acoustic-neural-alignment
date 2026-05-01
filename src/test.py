import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
from utils import load_audio

wav_path = "src/sd_fra_list1_FRcorp1.wav"
audio, sr = load_audio(wav_path)
print(f"audio: {len(audio)/sr:.2f}s")
print(f"audio samples: {len(audio)}")

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
    "facebook/wav2vec2-large-xlsr-53"
)
model = Wav2Vec2Model.from_pretrained(
    "facebook/wav2vec2-large-xlsr-53",
    output_hidden_states=True,
    use_safetensors=True
)
model.eval()

inputs = feature_extractor(audio, sampling_rate=sr, return_tensors="pt", padding=False)
print(f"input_values shape: {inputs.input_values.shape}")
print(f"model dtype: {next(model.parameters()).dtype}")

with torch.no_grad():
    out = model(inputs.input_values)

print(f"hidden shape: {out.hidden_states[4].shape}")