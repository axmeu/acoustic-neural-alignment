import pandas as pd
import numpy as np
import torch
import argparse
from tqdm import tqdm
from transformers import WhisperProcessor, WhisperModel
from neural_utils import load_audio, get_frame_indices, save_npz


def load_model(model_name, device):
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperModel.from_pretrained(
        model_name,
        output_hidden_states=True,
        torch_dtype=torch.float32)

    model.eval().to(device)
    return processor, model


def extract_file(wav_path, group, processor, model, layer, device):
    audio, sr = load_audio(wav_path)

    audio_duration_s = len(audio) / sr

    inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
    input_features = inputs.input_features.to(device)

    with torch.no_grad():
        outputs = model.encoder(input_features, output_hidden_states=True)

    hidden = outputs.hidden_states[layer]
    hidden = hidden.squeeze(0).cpu().numpy()
    n_frames = hidden.shape[0]

    results = []
    for _, row in group.iterrows():
        t0, t1 = get_frame_indices(
            row["onset_s"], row["offset_s"], n_frames, audio_duration_s
        )
        if t0 >= t1:
            vec = np.full(hidden.shape[-1], np.nan)
        else:
            vec = hidden[t0:t1].mean(axis=0)
        results.append((row["phoneme_id"], vec))

    return results


def extract_neural_whisper(table_path, output_path, model_name, layer, device):
    df = pd.read_csv(table_path)

    processor, model = load_model(model_name, device)
    print(f"Model : {model_name}  |  layer : {layer}  |  device : {device}")

    hidden_dim = model.config.d_model

    all_ids, all_vecs = [], []

    for wav_path, group in tqdm(df.groupby("wav_path"), desc="Whisper extraction"):
        try:
            results = extract_file(wav_path, group, processor, model, layer, device)
            for pid, vec in results:
                all_ids.append(pid)
                all_vecs.append(vec)
        except Exception as e:
            print(f"[WARN] {wav_path}: {e}")
            for _, row in group.iterrows():
                all_ids.append(row["phoneme_id"])
                all_vecs.append(np.full(hidden_dim, np.nan))

    ids_arr = np.array(all_ids)
    vecs_arr = np.stack(all_vecs)

    save_npz(output_path, ids_arr, vecs_arr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--table",   required=True)
    parser.add_argument("--output",  default="outputs/features_whisper.npz")
    parser.add_argument("--model",   default="openai/whisper-large-v3")
    parser.add_argument("--layer",   type=int, default=8)
    parser.add_argument("--device",  default="cuda")
    args = parser.parse_args()

    extract_neural_whisper(
        table_path=args.table,
        output_path=args.output,
        model_name=args.model,
        layer=args.layer,
        device=args.device,
    )
