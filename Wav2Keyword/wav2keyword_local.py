"""Wav2Keyword ported off the bundled 2020 fairseq fork onto torchaudio.

Faithful to downstream_kws.py:
- Encoder: pretrained Wav2Vec 2.0 base, no fine-tuning (torchaudio's
  WAV2VEC2_BASE bundle == the README's wav2vec_small.pt: LibriSpeech 960 h).
- Decoder head, optimizer param groups (encoder 1e-5 / head 5e-4, wd 1e-5),
  22 classes, batch augmentation (loudest-section, shift, time-mask, noise
  mixing) copied from the original.
Deviations (all forced or bug fixes, see summary):
- fairseq RawAudioDataset -> plain torch Dataset (clips are all exactly 1 s,
  so collate is a stack).
- Original applied Softmax before CrossEntropyLoss in training (double
  softmax); here the model always returns raw logits.
- CUDA -> MPS/CPU; 3 epochs instead of 100.
"""
import os
import random
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.utils.data as data
import torchaudio
from tqdm import tqdm

ROOT = 'data/speech_commands_v1'
CLASSES = 'unknown, silence, yes, no, up, down, left, right, on, off, stop, go, zero, one, two, three, four, five, six, seven, eight, nine'.split(', ')
BATCH = 64
EPOCHS = 3
DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
SR = 16000


class KWS(nn.Module):
    def __init__(self, n_class=22, encoder_hidden_dim=768):
        super().__init__()
        self.w2v_encoder = torchaudio.pipelines.WAV2VEC2_BASE.get_model()
        out_channels = 112
        self.decoder = nn.Sequential(
            nn.Conv1d(encoder_hidden_dim, out_channels, 25, dilation=2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, 1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, n_class, 1),
        )

    def forward(self, x):
        feats, _ = self.w2v_encoder.extract_features(x)
        output = feats[-1]                    # (B, T, 768), final layer
        output = output.transpose(1, 2)       # (B, 768, T)
        return self.decoder(output).squeeze(-1)   # (B, n_class); logits


class SpeechCommandsDataset(data.Dataset):
    def __init__(self, mode, root, loudest_section=True,
                 noise_prob=0.5, noise_level=0.7, shift_prob=0.5,
                 mask_prob=0.5, mask_len=0.1):
        self.mode = mode
        self.root = root
        self.loudest_section = loudest_section
        self.data = []
        for c in CLASSES:
            cdir = os.path.join(root, mode, c)
            for f in sorted(os.listdir(cdir)):
                if f.endswith('.wav'):
                    self.data.append((os.path.join(cdir, f), c))
        print(f'{mode} data number: {len(self.data)}', flush=True)

        if mode == 'training':
            noise_path = os.path.join(root, '_background_noise_')
            samples = []
            for f in sorted(os.listdir(noise_path)):
                if f.endswith('.wav'):
                    wav, _ = sf.read(os.path.join(noise_path, f))
                    samples.append(wav)
            samples = np.hstack(samples)
            r = len(samples) // SR
            self.noise_data = samples[:r * SR].reshape(-1, SR)
            self.noise_prob = noise_prob
            self.noise_level = noise_level
            self.shift_prob = shift_prob
            self.mask_prob = mask_prob
            self.mask_len = mask_len

    def extract_loudest_section(self, wav, win_len=30):
        wav_len = len(wav)
        temp = abs(wav)
        st, et, max_dec = 0, 0, 0
        for ws in range(0, wav_len, win_len):
            cur_dec = temp[ws:ws + SR].sum()
            if cur_dec >= max_dec:
                max_dec = cur_dec
                st, et = ws, ws + SR
            if ws + SR > wav_len:
                break
        return wav[st:et]

    def __getitem__(self, idx):
        f_path, cmd = self.data[idx]
        wav, _ = sf.read(f_path, dtype='float32')
        if wav.ndim == 2:
            wav = wav.mean(1)
        if self.loudest_section and cmd != 'silence':
            wav = self.extract_loudest_section(wav)
        if len(wav) < SR:
            pad = SR - len(wav)
            wav = np.pad(wav, (round(pad / 2) + 1, round(pad / 2) + 1), 'constant')
        mid, cut = len(wav) // 2, SR // 2
        wav = wav[mid - cut:mid + cut].copy()

        if self.mode == 'training':
            if random.random() < self.shift_prob:
                d = int(SR * random.uniform(-self.shift_prob, self.shift_prob))
                wav = np.roll(wav, d)
                if d > 0:
                    wav[:d] = 0
                else:
                    wav[d:] = 0
            if random.random() < self.mask_prob:
                t = int(self.mask_len * SR)
                t0 = random.randint(0, SR - t)
                wav[t0:t0 + t] = 0
            if random.random() < self.noise_prob:
                noise = random.choice(self.noise_data)
                pct = random.uniform(0, 1) if cmd == 'silence' else random.uniform(0, self.noise_level)
                wav = wav * (1 - pct) + noise * pct

        return torch.from_numpy(wav).float(), CLASSES.index(cmd)

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    torch.manual_seed(1234)
    random.seed(1234)
    print('device:', DEVICE, flush=True)

    train_ds = SpeechCommandsDataset('training', ROOT)
    test_ds = SpeechCommandsDataset('testing', ROOT)
    train_dl = data.DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=2)
    test_dl = data.DataLoader(test_ds, batch_size=BATCH, shuffle=False, num_workers=2)

    model = KWS(n_class=len(CLASSES)).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('trainable params:', n_params, flush=True)

    optimizer = torch.optim.Adam([
        {'params': model.w2v_encoder.parameters(), 'lr': 1e-5},
        {'params': model.decoder.parameters(), 'lr': 5e-4},
    ], weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    os.makedirs('checkpoint/w2k_v1', exist_ok=True)
    best_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        for step, (x, y) in enumerate(tqdm(train_dl, desc=f'train {epoch}')):
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            total += y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
        print('epoch %d train loss %.4f acc %.2f%%' % (
            epoch, loss_sum / (step + 1), 100 * correct / total), flush=True)

        model.eval()
        total, correct = 0, 0
        with torch.no_grad():
            for x, y in tqdm(test_dl, desc=f'test  {epoch}'):
                x, y = x.to(DEVICE), y.to(DEVICE)
                logits = model(x)
                total += y.size(0)
                correct += (logits.argmax(1) == y).sum().item()
        acc = 100 * correct / total
        print('epoch %d TEST acc %.2f%% (%d/%d)' % (epoch, acc, correct, total), flush=True)
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), 'checkpoint/w2k_v1/best_model.pth')
            print('saved best', flush=True)

    print('BEST TEST ACC: %.2f%%' % best_acc, flush=True)
    print('========== DONE ==========', flush=True)
