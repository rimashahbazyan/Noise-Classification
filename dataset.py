import argparse
import csv
from functools import partial
import glob
import os

import torch
import torchaudio
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from transformers import Wav2Vec2FeatureExtractor



class NoiseClassesDataset(Dataset):
    def __init__(self, dataset_dir, dataset_csv_path, sample_rate=16000, audio_length=2, is_hubert=False):
        self.dataset_audio_paths = glob.glob(os.path.join(dataset_dir, '**', '*.wav'), recursive=True)
        self.dataset_csv_path = dataset_csv_path
        self.sample_rate = sample_rate
        self.audio_length = audio_length
        if not is_hubert:
            self.feature_extraction = self.get_mel_spectrogram
        else:
            self.hubert_fe = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
            self.feature_extraction = partial(self.hubert_fe, sampling_rate=16000, return_tensors="pt")
        
    def __len__(self):
        return len(self.dataset_audio_paths)

    def __getitem__(self, idx):
        audio_path = self.dataset_audio_paths[idx]
        audio_data = self.load_audio(audio_path)
        if len(audio_data) < self.sample_rate * self.audio_length:
            audio_data = torch.cat([audio_data, torch.zeros(self.sample_rate * self.audio_length - len(audio_data))])
        audio_data = self.get_cut_waveform(audio_data)
        data = self.feature_extraction(audio_data).input_values.squeeze(0)
        label = self.get_class_from_path(audio_path)
        return data, label

    def get_class_mapping(self):
        # CSV head is ['slice_file_name', 'fsID', 'start', 'end', 'salience', 'fold', 'classID', 'class']
        with open(self.dataset_csv_path, 'r') as f:
            reader = csv.reader(f)
            csv_data = [r for r in reader]
        classes = dict()
        for entry in csv_data[1:]:
            if int(entry[6]) not in classes:
                classes[int(entry[6])] = entry[7]
        classes = dict(sorted(classes.items(), key=lambda item: item[0]))
        return classes

    def load_audio(self, audio_path):
        audio_data, sample_rate = torchaudio.load(audio_path, normalize=True)
        # if audio_data.shape[0] != 1:
        #     audio_data = torch.mean(audio_data, dim=0)
        audio_data = torch.squeeze(audio_data)
        # if sample_rate != self.sample_rate:
        #     audio_data = torchaudio.functional.resample(audio_data, orig_freq=sample_rate, new_freq=self.sample_rate)
        return audio_data

    def get_cut_waveform(self, audio_data):

        segment_length = self.audio_length * self.sample_rate
        start = torch.randint(0, len(audio_data) - segment_length + 1, (1,))
        return audio_data[start: start + segment_length]

    def get_mel_spectrogram(self, audio):
        mel_transfrom = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=1024,
            win_length=1024,
            hop_length=256,
            n_mels=80,
        )
        mel_data = mel_transfrom(audio)
        return mel_data

    @staticmethod
    def get_class_from_path(audio_path):
        tmp = os.path.basename(audio_path)
        tmp = os.path.splitext(tmp)[0]
        tmp = tmp.split('-')[1]
        return int(tmp)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, help="Path to dataset audios directory",
                        default='/home/rshahbazyan/Downloads/UrbanSound8K/audio16k')
    parser.add_argument("--dataset_csv_path", type=str, help="Path to dataset csv",
                        default='/home/rshahbazyan/Downloads/UrbanSound8K/metadata/UrbanSound8K.csv')
    args = parser.parse_args()
    noise_classes_dataset = NoiseClassesDataset(args.dataset_dir, args.dataset_csv_path)
    class_mapping = noise_classes_dataset.get_class_mapping()

    data_loader = DataLoader(noise_classes_dataset, batch_size=64)
    for inputs, labels in data_loader:
        print(inputs[0].size(), class_mapping[int(labels[0])])
        #break


if __name__ == '__main__':
    main()
