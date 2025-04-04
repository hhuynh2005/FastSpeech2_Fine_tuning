'''
import os
import random
import json

import tgt
import librosa
import numpy as np
import pyworld as pw
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import audio as Audio


class Preprocessor:
    def __init__(self, config):
        self.config = config
        self.in_dir = config["path"]["raw_path"]
        self.out_dir = config["path"]["preprocessed_path"]
        self.val_size = config["preprocessing"]["val_size"]
        self.sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
        self.hop_length = config["preprocessing"]["stft"]["hop_length"]

        assert config["preprocessing"]["pitch"]["feature"] in [
            "phoneme_level",
            "frame_level",
        ]
        assert config["preprocessing"]["energy"]["feature"] in [
            "phoneme_level",
            "frame_level",
        ]
        self.pitch_phoneme_averaging = (
            config["preprocessing"]["pitch"]["feature"] == "phoneme_level"
        )
        self.energy_phoneme_averaging = (
            config["preprocessing"]["energy"]["feature"] == "phoneme_level"
        )

        self.pitch_normalization = config["preprocessing"]["pitch"]["normalization"]
        self.energy_normalization = config["preprocessing"]["energy"]["normalization"]

        self.STFT = Audio.stft.TacotronSTFT(
            config["preprocessing"]["stft"]["filter_length"],
            config["preprocessing"]["stft"]["hop_length"],
            config["preprocessing"]["stft"]["win_length"],
            config["preprocessing"]["mel"]["n_mel_channels"],
            config["preprocessing"]["audio"]["sampling_rate"],
            config["preprocessing"]["mel"]["mel_fmin"],
            config["preprocessing"]["mel"]["mel_fmax"],
        )

    def build_from_path(self):
        os.makedirs((os.path.join(self.out_dir, "mel")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "pitch")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "energy")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "duration")), exist_ok=True)

        print("Processing Data ...")
        out = list()
        n_frames = 0
        pitch_scaler = StandardScaler()
        energy_scaler = StandardScaler()

        # Compute pitch, energy, duration, and mel-spectrogram
        speakers = {}
        for i, speaker in enumerate(tqdm(os.listdir(self.in_dir))):
            speakers[speaker] = i
            for wav_name in os.listdir(os.path.join(self.in_dir, speaker)):
                if ".wav" not in wav_name:
                    continue

                basename = wav_name.split(".")[0]
                tg_path = os.path.join(
                    self.out_dir, "TextGrid", speaker, "{}.TextGrid".format(basename)
                )
                #print("trace 0", tg_path)
                if os.path.exists(tg_path):
                    #print("trace 1", tg_path)
                    ret = self.process_utterance(speaker, basename)
                    if ret is None:
                        continue
                    else:
                        info, pitch, energy, n = ret
                    out.append(info)

                if len(pitch) > 0:
                    pitch_scaler.partial_fit(pitch.reshape((-1, 1)))
                if len(energy) > 0:
                    energy_scaler.partial_fit(energy.reshape((-1, 1)))

                n_frames += n

        print("Computing statistic quantities ...")
        # Perform normalization if necessary
        if self.pitch_normalization:
            pitch_mean = pitch_scaler.mean_[0]
            pitch_std = pitch_scaler.scale_[0]
        else:
            # A numerical trick to avoid normalization...
            pitch_mean = 0
            pitch_std = 1
        if self.energy_normalization:
            energy_mean = energy_scaler.mean_[0]
            energy_std = energy_scaler.scale_[0]
        else:
            energy_mean = 0
            energy_std = 1

        pitch_min, pitch_max = self.normalize(
            os.path.join(self.out_dir, "pitch"), pitch_mean, pitch_std
        )
        energy_min, energy_max = self.normalize(
            os.path.join(self.out_dir, "energy"), energy_mean, energy_std
        )

        # Save files
        with open(os.path.join(self.out_dir, "speakers.json"), "w") as f:
            f.write(json.dumps(speakers))

        with open(os.path.join(self.out_dir, "stats.json"), "w") as f:
            stats = {
                "pitch": [
                    float(pitch_min),
                    float(pitch_max),
                    float(pitch_mean),
                    float(pitch_std),
                ],
                "energy": [
                    float(energy_min),
                    float(energy_max),
                    float(energy_mean),
                    float(energy_std),
                ],
            }
            f.write(json.dumps(stats))

        print(
            "Total time: {} hours".format(
                n_frames * self.hop_length / self.sampling_rate / 3600
            )
        )

        random.shuffle(out)
        out = [r for r in out if r is not None]

        # Write metadata
        with open(os.path.join(self.out_dir, "train.txt"), "w", encoding="utf-8") as f:
            for m in out[self.val_size :]:
                f.write(m + "\n")
        with open(os.path.join(self.out_dir, "val.txt"), "w", encoding="utf-8") as f:
            for m in out[: self.val_size]:
                f.write(m + "\n")

        return out

    def process_utterance(self, speaker, basename):
        wav_path = os.path.join(self.in_dir, speaker, "{}.wav".format(basename))
        text_path = os.path.join(self.in_dir, speaker, "{}.lab".format(basename))
        tg_path = os.path.join(
            self.out_dir, "TextGrid", speaker, "{}.TextGrid".format(basename)
        )

        # Get alignments
        textgrid = tgt.io.read_textgrid(tg_path)
        phone, duration, start, end = self.get_alignment(
            textgrid.get_tier_by_name("phones")
        )
        text = "{" + " ".join(phone) + "}"
        if start >= end:
            return None

        # Read and trim wav files
        wav, _ = librosa.load(wav_path)
        wav = wav[
            int(self.sampling_rate * start) : int(self.sampling_rate * end)
        ].astype(np.float32)

        # Read raw text
        with open(text_path, "r", encoding='utf-8') as f:
            raw_text = f.readline().strip("\n")

        # Compute fundamental frequency
        pitch, t = pw.dio(
            wav.astype(np.float64),
            self.sampling_rate,
            frame_period=self.hop_length / self.sampling_rate * 1000,
        )
        pitch = pw.stonemask(wav.astype(np.float64), pitch, t, self.sampling_rate)

        pitch = pitch[: sum(duration)]
        if np.sum(pitch != 0) <= 1:
            return None

        # Compute mel-scale spectrogram and energy
        mel_spectrogram, energy = Audio.tools.get_mel_from_wav(wav, self.STFT)
        mel_spectrogram = mel_spectrogram[:, : sum(duration)]
        energy = energy[: sum(duration)]

        if self.pitch_phoneme_averaging:
            # perform linear interpolation
            nonzero_ids = np.where(pitch != 0)[0]
            interp_fn = interp1d(
                nonzero_ids,
                pitch[nonzero_ids],
                fill_value=(pitch[nonzero_ids[0]], pitch[nonzero_ids[-1]]),
                bounds_error=False,
            )
            pitch = interp_fn(np.arange(0, len(pitch)))

            # Phoneme-level average
            pos = 0
            for i, d in enumerate(duration):
                if d > 0:
                    pitch[i] = np.mean(pitch[pos : pos + d])
                else:
                    pitch[i] = 0
                pos += d
            pitch = pitch[: len(duration)]

        if self.energy_phoneme_averaging:
            # Phoneme-level average
            pos = 0
            for i, d in enumerate(duration):
                if d > 0:
                    energy[i] = np.mean(energy[pos : pos + d])
                else:
                    energy[i] = 0
                pos += d
            energy = energy[: len(duration)]

        # Save files
        dur_filename = "{}-duration-{}.npy".format(speaker, basename)
        np.save(os.path.join(self.out_dir, "duration", dur_filename), duration)

        pitch_filename = "{}-pitch-{}.npy".format(speaker, basename)
        np.save(os.path.join(self.out_dir, "pitch", pitch_filename), pitch)

        energy_filename = "{}-energy-{}.npy".format(speaker, basename)
        np.save(os.path.join(self.out_dir, "energy", energy_filename), energy)

        mel_filename = "{}-mel-{}.npy".format(speaker, basename)
        np.save(
            os.path.join(self.out_dir, "mel", mel_filename),
            mel_spectrogram.T,
        )

        return (
            "|".join([basename, speaker, text, raw_text]),
            self.remove_outlier(pitch),
            self.remove_outlier(energy),
            mel_spectrogram.shape[1],
        )

    def get_alignment(self, tier):
        sil_phones = ["sil", "sp", "spn"]

        phones = []
        durations = []
        start_time = 0
        end_time = 0
        end_idx = 0
        for t in tier._objects:
            s, e, p = t.start_time, t.end_time, t.text

            # Trim leading silences
            if phones == []:
                if p in sil_phones:
                    continue
                else:
                    start_time = s

            if p not in sil_phones:
                # For ordinary phones
                phones.append(p)
                end_time = e
                end_idx = len(phones)
            else:
                # For silent phones
                phones.append(p)

            durations.append(
                int(
                    np.round(e * self.sampling_rate / self.hop_length)
                    - np.round(s * self.sampling_rate / self.hop_length)
                )
            )

        # Trim tailing silences
        phones = phones[:end_idx]
        durations = durations[:end_idx]

        return phones, durations, start_time, end_time

    def remove_outlier(self, values):
        values = np.array(values)
        p25 = np.percentile(values, 25)
        p75 = np.percentile(values, 75)
        lower = p25 - 1.5 * (p75 - p25)
        upper = p75 + 1.5 * (p75 - p25)
        normal_indices = np.logical_and(values > lower, values < upper)

        return values[normal_indices]

    def normalize(self, in_dir, mean, std):
        max_value = np.finfo(np.float64).min
        min_value = np.finfo(np.float64).max
        for filename in os.listdir(in_dir):
            filename = os.path.join(in_dir, filename)
            values = (np.load(filename) - mean) / std
            np.save(filename, values)

            max_value = max(max_value, max(values))
            min_value = min(min_value, min(values))

        return min_value, max_value
'''


import os
import random
import json
import logging

import tgt
import librosa
import numpy as np
import pyworld as pw
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import audio as Audio

# Thiết lập logging để dễ debug
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Preprocessor:
    def __init__(self, config):
        self.config = config
        self.in_dir = config["path"]["raw_path"]
        self.out_dir = config["path"]["preprocessed_path"]
        self.val_size = config["preprocessing"]["val_size"]
        self.sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
        self.hop_length = config["preprocessing"]["stft"]["hop_length"]

        # Kiểm tra cấu hình pitch và energy
        assert config["preprocessing"]["pitch"]["feature"] in ["phoneme_level", "frame_level"], \
            "pitch.feature must be either 'phoneme_level' or 'frame_level'"
        assert config["preprocessing"]["energy"]["feature"] in ["phoneme_level", "frame_level"], \
            "energy.feature must be either 'phoneme_level' or 'frame_level'"

        self.pitch_phoneme_averaging = config["preprocessing"]["pitch"]["feature"] == "phoneme_level"
        self.energy_phoneme_averaging = config["preprocessing"]["energy"]["feature"] == "phoneme_level"
        self.pitch_normalization = config["preprocessing"]["pitch"]["normalization"]
        self.energy_normalization = config["preprocessing"]["energy"]["normalization"]

        # Khởi tạo STFT để tính mel-spectrogram
        self.STFT = Audio.stft.TacotronSTFT(
            config["preprocessing"]["stft"]["filter_length"],
            config["preprocessing"]["stft"]["hop_length"],
            config["preprocessing"]["stft"]["win_length"],
            config["preprocessing"]["mel"]["n_mel_channels"],
            config["preprocessing"]["audio"]["sampling_rate"],
            config["preprocessing"]["mel"]["mel_fmin"],
            config["preprocessing"]["mel"]["mel_fmax"],
        )

    def build_from_path(self):
        # Tạo các thư mục đầu ra
        os.makedirs(os.path.join(self.out_dir, "mel"), exist_ok=True)
        os.makedirs(os.path.join(self.out_dir, "pitch"), exist_ok=True)
        os.makedirs(os.path.join(self.out_dir, "energy"), exist_ok=True)
        os.makedirs(os.path.join(self.out_dir, "duration"), exist_ok=True)

        logging.info("Processing Data ...")
        out = []
        n_frames = 0
        pitch_scaler = StandardScaler()
        energy_scaler = StandardScaler()

        # Xử lý từng speaker và file âm thanh
        speakers = {}
        for i, speaker in enumerate(tqdm(os.listdir(self.in_dir))):
            if not os.path.isdir(os.path.join(self.in_dir, speaker)):
                continue
            speakers[speaker] = i
            for wav_name in os.listdir(os.path.join(self.in_dir, speaker)):
                if not wav_name.endswith(".wav"):
                    continue

                basename = wav_name.split(".")[0]
                tg_path = os.path.join(self.out_dir, "TextGrid", speaker, f"{basename}.TextGrid")
                
                if not os.path.exists(tg_path):
                    logging.warning(f"TextGrid file not found: {tg_path}")
                    continue

                # Xử lý từng utterance
                ret = self.process_utterance(speaker, basename)
                if ret is None:
                    logging.warning(f"Skipping {basename} from speaker {speaker}: processing failed")
                    continue
                
                info, pitch, energy, n = ret
                out.append(info)

                # Chỉ xử lý pitch và energy khi chúng hợp lệ
                if pitch is not None and len(pitch) > 0:
                    pitch_scaler.partial_fit(pitch.reshape((-1, 1)))
                if energy is not None and len(energy) > 0:
                    energy_scaler.partial_fit(energy.reshape((-1, 1)))

                n_frames += n

        logging.info("Computing statistic quantities ...")
        # Chuẩn hóa pitch và energy nếu cần
        pitch_mean = pitch_scaler.mean_[0] if self.pitch_normalization and hasattr(pitch_scaler, 'mean_') else 0
        pitch_std = pitch_scaler.scale_[0] if self.pitch_normalization and hasattr(pitch_scaler, 'scale_') else 1
        energy_mean = energy_scaler.mean_[0] if self.energy_normalization and hasattr(energy_scaler, 'mean_') else 0
        energy_std = energy_scaler.scale_[0] if self.energy_normalization and hasattr(energy_scaler, 'scale_') else 1

        # Chuẩn hóa các file pitch và energy
        pitch_min, pitch_max = self.normalize(os.path.join(self.out_dir, "pitch"), pitch_mean, pitch_std)
        energy_min, energy_max = self.normalize(os.path.join(self.out_dir, "energy"), energy_mean, energy_std)

        # Lưu thông tin speakers và thống kê
        with open(os.path.join(self.out_dir, "speakers.json"), "w") as f:
            json.dump(speakers, f)

        with open(os.path.join(self.out_dir, "stats.json"), "w") as f:
            stats = {
                "pitch": [float(pitch_min), float(pitch_max), float(pitch_mean), float(pitch_std)],
                "energy": [float(energy_min), float(energy_max), float(energy_mean), float(energy_std)],
            }
            json.dump(stats, f)

        logging.info(f"Total time: {n_frames * self.hop_length / self.sampling_rate / 3600:.2f} hours")

        # Xáo trộn và chia dữ liệu thành train/val
        random.shuffle(out)
        out = [r for r in out if r is not None]

        with open(os.path.join(self.out_dir, "train.txt"), "w", encoding="utf-8") as f:
            for m in out[self.val_size:]:
                f.write(m + "\n")
        with open(os.path.join(self.out_dir, "val.txt"), "w", encoding="utf-8") as f:
            for m in out[:self.val_size]:
                f.write(m + "\n")

        return out

    def process_utterance(self, speaker, basename):
        wav_path = os.path.join(self.in_dir, speaker, f"{basename}.wav")
        text_path = os.path.join(self.in_dir, speaker, f"{basename}.lab")
        tg_path = os.path.join(self.out_dir, "TextGrid", speaker, f"{basename}.TextGrid")

        # Đọc TextGrid và lấy thông tin căn chỉnh
        try:
            textgrid = tgt.io.read_textgrid(tg_path)
            phone, duration, start, end = self.get_alignment(textgrid.get_tier_by_name("phones"))
        except Exception as e:
            logging.error(f"Error reading TextGrid {tg_path}: {e}")
            return None

        text = "{" + " ".join(phone) + "}"
        if start >= end:
            logging.warning(f"Invalid alignment in {basename}: start ({start}) >= end ({end})")
            return None

        # Đọc và cắt file âm thanh
        try:
            wav, _ = librosa.load(wav_path, sr=self.sampling_rate)
            wav = wav[int(self.sampling_rate * start):int(self.sampling_rate * end)].astype(np.float32)
        except Exception as e:
            logging.error(f"Error loading WAV file {wav_path}: {e}")
            return None

        # Đọc văn bản thô
        try:
            with open(text_path, "r", encoding='utf-8') as f:
                raw_text = f.readline().strip("\n")
        except Exception as e:
            logging.error(f"Error reading text file {text_path}: {e}")
            return None

        # Tính tần số cơ bản (pitch)
        try:
            pitch, t = pw.dio(
                wav.astype(np.float64),
                self.sampling_rate,
                frame_period=self.hop_length / self.sampling_rate * 1000,
            )
            pitch = pw.stonemask(wav.astype(np.float64), pitch, t, self.sampling_rate)
            pitch = pitch[:sum(duration)]
            if np.sum(pitch != 0) <= 1:
                logging.warning(f"Invalid pitch data in {basename}: too few non-zero values")
                return None
        except Exception as e:
            logging.error(f"Error computing pitch for {basename}: {e}")
            return None

        # Tính mel-spectrogram và energy
        try:
            mel_spectrogram, energy = Audio.tools.get_mel_from_wav(wav, self.STFT)
            mel_spectrogram = mel_spectrogram[:, :sum(duration)]
            energy = energy[:sum(duration)]
        except Exception as e:
            logging.error(f"Error computing mel-spectrogram for {basename}: {e}")
            return None

        # Trung bình pitch theo phoneme nếu cần
        if self.pitch_phoneme_averaging:
            nonzero_ids = np.where(pitch != 0)[0]
            if len(nonzero_ids) > 0:
                interp_fn = interp1d(
                    nonzero_ids,
                    pitch[nonzero_ids],
                    fill_value=(pitch[nonzero_ids[0]], pitch[nonzero_ids[-1]]),
                    bounds_error=False,
                )
                pitch = interp_fn(np.arange(0, len(pitch)))

                pos = 0
                for i, d in enumerate(duration):
                    if d > 0:
                        pitch[i] = np.mean(pitch[pos:pos + d])
                    else:
                        pitch[i] = 0
                    pos += d
                pitch = pitch[:len(duration)]

        # Trung bình energy theo phoneme nếu cần
        if self.energy_phoneme_averaging:
            pos = 0
            for i, d in enumerate(duration):
                if d > 0:
                    energy[i] = np.mean(energy[pos:pos + d])
                else:
                    energy[i] = 0
                pos += d
            energy = energy[:len(duration)]

        # Lưu các file kết quả
        try:
            np.save(os.path.join(self.out_dir, "duration", f"{speaker}-duration-{basename}.npy"), duration)
            np.save(os.path.join(self.out_dir, "pitch", f"{speaker}-pitch-{basename}.npy"), pitch)
            np.save(os.path.join(self.out_dir, "energy", f"{speaker}-energy-{basename}.npy"), energy)
            np.save(os.path.join(self.out_dir, "mel", f"{speaker}-mel-{basename}.npy"), mel_spectrogram.T)
        except Exception as e:
            logging.error(f"Error saving files for {basename}: {e}")
            return None

        return (
            "|".join([basename, speaker, text, raw_text]),
            self.remove_outlier(pitch),
            self.remove_outlier(energy),
            mel_spectrogram.shape[1],
        )

    def get_alignment(self, tier):
        sil_phones = ["sil", "sp", "spn"]
        phones = []
        durations = []
        start_time = 0
        end_time = 0
        end_idx = 0

        for t in tier._objects:
            s, e, p = t.start_time, t.end_time, t.text
            if phones == [] and p in sil_phones:
                continue
            if p not in sil_phones:
                phones.append(p)
                end_time = e
                end_idx = len(phones)
            else:
                phones.append(p)

            durations.append(
                int(np.round(e * self.sampling_rate / self.hop_length) -
                    np.round(s * self.sampling_rate / self.hop_length))
            )
            if phones and start_time == 0:
                start_time = s

        # Loại bỏ các khoảng lặng ở cuối
        phones = phones[:end_idx]
        durations = durations[:end_idx]

        return phones, durations, start_time, end_time

    def remove_outlier(self, values):
        values = np.array(values)
        p25 = np.percentile(values, 25)
        p75 = np.percentile(values, 75)
        lower = p25 - 1.5 * (p75 - p25)
        upper = p75 + 1.5 * (p75 - p25)
        normal_indices = np.logical_and(values > lower, values < upper)
        return values[normal_indices]

    def normalize(self, in_dir, mean, std):
        if not os.path.exists(in_dir) or not os.listdir(in_dir):
            return 0, 0

        max_value = np.finfo(np.float64).min
        min_value = np.finfo(np.float64).max
        for filename in os.listdir(in_dir):
            filepath = os.path.join(in_dir, filename)
            try:
                values = (np.load(filepath) - mean) / std
                np.save(filepath, values)
                max_value = max(max_value, max(values))
                min_value = min(min_value, min(values))
            except Exception as e:
                logging.error(f"Error normalizing file {filepath}: {e}")
        return min_value, max_value