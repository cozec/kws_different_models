import os
import os.path
import torch
import torchaudio
import soundfile as sf
import json
import numpy

AUDIO_EXTENSIONS = '.wav'
AUDIO_PATH = '/Users/adam/interviews/kws/KWS_pytorch/data/mobvoi_hotword_dataset/'

# Shared STFT front-end: n_fft=320 -> 161 freq bins, hop 160 (=10ms @16kHz).
# This matches the input dims the committed models expect (161 x T).
_SPECTROGRAM = torchaudio.transforms.Spectrogram(
    n_fft=320, win_length=320, hop_length=160, power=2.0
)

def find_classes():
    # Deep KWS (Chen et al., ICASSP 2014) label convention: index 0 = filler
    # (non-keyword), keyword labels follow.
    classes = ["filler", "HIW", "NHWW"]  # filler, (hi xiaowen), (nihao wenwen)
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

def make_dataset(dir, class_to_idx):
    paths = []
    file = open(dir, 'r', encoding='utf-8')
    data = json.load(file)
    for i in range(len(data)):
        if data[i]["utt_id"] == "0":
            continue
        path = data[i]["utt_id"]
        path = AUDIO_PATH + path + AUDIO_EXTENSIONS
        # try:
        #     path = os.path.expanduser(path)
        #     y, sr = torchaudio.load(path)
        # except RuntimeError:
        #     data[i]["utt_id"] = "0"
        #     continue
        # keyword_id: -1 = negative, 0/1 = hotwords. Shift by +1 so that
        # label 0 = filler (paper convention), 1 = HIW, 2 = NHWW.
        target = data[i]["keyword_id"] + 1
        item = (path, target)
        paths.append(item)
    # with open(dir, "w", encoding='utf-8') as jsonFile:
    #     json.dump(data, jsonFile)
    return paths

def spect_loader(path, normalize, max_frames=101):
    # STFT log-power spectrogram, shape (161, max_frames). CPU only.
    # Load via soundfile (avoids the torchcodec/ffmpeg backend requirement).
    try:
        path = os.path.expanduser(path)
        y_np, sr = sf.read(path, dtype='float32', always_2d=True)  # (n, ch)
        y = torch.from_numpy(y_np.T)                               # (ch, n)
    except Exception:
        return torch.zeros((161, max_frames))

    if sr != 16000:
        y = torchaudio.functional.resample(y, sr, 16000)
    if y.shape[0] > 1:                       # mix down to mono
        y = y.mean(0, keepdim=True)

    spect = _SPECTROGRAM(y).squeeze(0)       # (161, T)
    spect = torch.log1p(spect)               # log power

    # fix the number of frames to max_frames (pad/truncate along time)
    if spect.shape[1] < max_frames:
        pad = torch.zeros((spect.shape[0], max_frames - spect.shape[1]))
        spect = torch.cat([spect, pad], 1)
    elif spect.shape[1] > max_frames:
        spect = spect[:, :max_frames]

    # 特征归一化
    if normalize:
        mean = spect.mean()
        std = spect.std()
        if std != 0:
            spect = (spect - mean) / std
    return spect                             # (161, max_frames)

class CommandLoader(torch.utils.data.Dataset):
    def __init__(self, root, normalize=True, max_frames=101):
        classes, class_to_idx = find_classes()
        paths = make_dataset(root, class_to_idx)
        if len(paths) == 0:
            print("Dataset is None!")
            raise (RuntimeError)
        
        self.root = root
        self.paths = paths
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.loader = spect_loader
        self.normalize = normalize
        self.max_frames = max_frames

    def __getitem__(self, index):
        path, target = self.paths[index]
        spect = self.loader(path, self.normalize, self.max_frames)
        return spect, target
    
    def __len__(self):
        return len(self.paths)

# 用于测试可以正确加载数据
if __name__ == '__main__':
    train_dataset = CommandLoader('./mobvoi_hotword_dataset_resources/p_test.json', max_frames=200)
    for i in range(21282):
        print(train_dataset[i])
            # 200 * 1640