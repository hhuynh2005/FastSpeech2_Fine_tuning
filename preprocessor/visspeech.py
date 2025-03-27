import os

import librosa
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm

from text import _clean_text
def prepare_align(config):
    in_dir = config["path"]["corpus_path"]
    out_dir = config["path"]["raw_path"]
    sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
    max_wav_value = config["preprocessing"]["audio"]["max_wav_value"]
    cleaners = config["preprocessing"]["text"]["text_cleaners"]
    #Tên cuat người tạo giọng nói
    speaker = "ViSSpeech"

    print(f"Input directory: {in_dir}")
    print(f"Output directory: {out_dir}")
    
    # Kiểm tra sự tồn tại của metadata.csv
    metadata_path = os.path.join(in_dir, "metadata.csv")
    if not os.path.exists(metadata_path):
        print(f"Error: metadata.csv not found at {metadata_path}")
        return
    
    # Đọc metadata.csv
    with open(metadata_path, encoding="utf-8") as f:
        for line in tqdm(f):
            parts = line.strip().split("|")
            base_name = parts[0].split('.')[0] 
            text = parts[1]
            text = _clean_text(text, cleaners)

            # Kiểm tra sự tồn tại của tệp wav
            wav_path = os.path.join(in_dir, "wavs", "{}.wav".format(base_name))
            if os.path.exists(wav_path):
                print(f"Processing: {base_name}.wav")
                os.makedirs(os.path.join(out_dir, speaker), exist_ok=True)
                
                # Đọc và chuẩn hóa âm thanh
                wav, _ = librosa.load(wav_path, sr=sampling_rate)
                wav = wav / max(abs(wav)) * max_wav_value
                wavfile.write(
                    os.path.join(out_dir, speaker, "{}.wav".format(base_name)),
                    sampling_rate,
                    wav.astype(np.int16),
                )
                # Ghi tệp .lab
                with open(
                    os.path.join(out_dir, speaker, "{}.lab".format(base_name)),
                    "w", encoding='utf-8'
                ) as f1:
                    f1.write(text)
                print(f"Created: {base_name}.wav and {base_name}.lab")
            else:
                print(f"Warning: File {wav_path} does not exist!")
