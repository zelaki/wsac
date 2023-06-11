
import torch
import librosa
import h5py
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader




class AACEvalDataset(Dataset):

    def __init__(self, config, dataset_name):

        if dataset_name == 'clotho':
            self.h5_path = '/home/theokouz/data/clotho/hdf5/evaluation.h5'
        
        elif dataset_name == 'audiocaps':
            self.h5_path = '/home/theokouz/data/audiocaps/hdf5s/test/test.h5'


        with h5py.File(self.h5_path, 'r') as hf:
            self.audio_names = [audio_name.decode() for audio_name in hf['audio_name'][:]]
            self.captions = [caption for caption in hf['caption'][:]]
        self.sr = config.wav.sr
        self.window_length = config.wav.window_length
        self.hop_length = config.wav.hop_length
        self.n_mels = config.wav.n_mels

        self.caption_field = ['caption_{}'.format(i) for i in range(1, 6)]

    def __len__(self):
        return len(self.audio_names)

    def __getitem__(self, index):
        with h5py.File(self.h5_path, 'r') as hf:
            waveform = self.resample(hf['waveform'][index])
        audio_name = self.audio_names[index]
        captions = self.captions[index]

        target_dict = {}
        for i, cap_ind in enumerate(self.caption_field):
            target_dict[cap_ind] = captions[i].decode()

        return waveform, target_dict, audio_name

    def resample(self, waveform):
        """Resample.
        Args:
          waveform: (clip_samples,)
        Returns:
          (resampled_clip_samples,)
        """
        if self.sr == 32000:
            return waveform
        elif self.sr == 16000:
            return waveform[0:: 2]
        elif self.sr == 8000:
            return waveform[0:: 4]
        else:
            raise Exception('Incorrect sample rate!')




def collate_fn_eval(batch):

    # feature = [torch.tensor(i[0]) for i in batch]
    # feature_tensor = torch.nn.utils.rnn.pad_sequence(feature, batch_first=True)
    feature_tensor = [i[0] for i in batch]

    file_names = [i[2] for i in batch]
    target_dicts = [i[1] for i in batch]

    return feature_tensor, target_dicts, file_names

def get_clotho_loader(config, dataset_name):
    dataset = AACEvalDataset(config, dataset_name)
    return DataLoader(dataset=dataset, batch_size=config.data.batch_size,
                        shuffle=False, drop_last=False,
                        num_workers=config.data.num_workers, collate_fn=collate_fn_eval)





