"""Build the training/testing/<class> layout Wav2Keyword expects from
Google Speech Commands v0.01 (already downloaded for the earlier project).

- 20 command words are symlinked into training/<word> or testing/<word>
  according to testing_list.txt (validation_list.txt files are excluded from
  training for hygiene; the repo ignores validation entirely).
- 'unknown' = the 10 non-command words, subsampled to ~10% of each split
  (the TF AudioProcessor convention this repo emulates; keeping all 10 words
  unsubsampled would swamp training 10:1).
- 'silence' = 1 s clips cut at random offsets from _background_noise_.
- _background_noise_ is symlinked at the root for training-time augmentation.
"""
import os
import random
import numpy as np
import soundfile as sf

SRC = '/Users/adam/interviews/kws/Spoken-Keyword-Spotting/input/speech_commands/train'
DST = '/Users/adam/interviews/kws/Wav2Keyword/data/speech_commands_v1'
COMMANDS = 'yes, no, up, down, left, right, on, off, stop, go, zero, one, two, three, four, five, six, seven, eight, nine'.split(', ')
UNKNOWN_FRAC = 0.10
random.seed(1234)

testing = set(open(os.path.join(SRC, 'testing_list.txt')).read().splitlines())
validation = set(open(os.path.join(SRC, 'validation_list.txt')).read().splitlines())

words = [d for d in os.listdir(SRC)
         if os.path.isdir(os.path.join(SRC, d)) and d != '_background_noise_']
unknown_words = sorted(set(words) - set(COMMANDS))
print('unknown words:', unknown_words)

split_files = {'training': {}, 'testing': {}}   # split -> class -> [src paths]
for word in words:
    cls = word if word in COMMANDS else 'unknown'
    for f in os.listdir(os.path.join(SRC, word)):
        if not f.endswith('.wav'):
            continue
        rel = f'{word}/{f}'
        if rel in testing:
            split = 'testing'
        elif rel in validation:
            continue
        else:
            split = 'training'
        split_files[split].setdefault(cls, []).append(os.path.join(SRC, rel))

counts = {}
for split in ['training', 'testing']:
    n_cmd = sum(len(v) for c, v in split_files[split].items() if c != 'unknown')
    # subsample unknown to ~10% of the command total
    unk = split_files[split]['unknown']
    random.shuffle(unk)
    split_files[split]['unknown'] = unk[:int(UNKNOWN_FRAC * n_cmd)]

    for cls, paths in split_files[split].items():
        d = os.path.join(DST, split, cls)
        os.makedirs(d, exist_ok=True)
        for p in paths:
            link = os.path.join(d, os.path.basename(os.path.dirname(p)) + '_' + os.path.basename(p))
            if not os.path.islink(link):
                os.symlink(p, link)
    counts[split] = {c: len(v) for c, v in sorted(split_files[split].items())}

# silence clips: random 1 s cuts from _background_noise_
noise = []
noise_dir = os.path.join(SRC, '_background_noise_')
for f in os.listdir(noise_dir):
    if f.endswith('.wav'):
        wav, sr = sf.read(os.path.join(noise_dir, f))
        noise.append((wav, sr))

for split, n_sil in [('training', 2300), ('testing', 250)]:
    d = os.path.join(DST, split, 'silence')
    os.makedirs(d, exist_ok=True)
    for i in range(n_sil):
        wav, sr = random.choice(noise)
        off = random.randint(0, len(wav) - sr - 1)
        sf.write(os.path.join(d, f'silence_{i:05d}.wav'), wav[off:off + sr], sr)
    counts[split]['silence'] = n_sil

# background noise at root for augmentation
bg = os.path.join(DST, '_background_noise_')
if not os.path.islink(bg) and not os.path.isdir(bg):
    os.symlink(noise_dir, bg)

for split in ['training', 'testing']:
    total = sum(counts[split].values())
    print(f'{split}: total={total}')
    print('  ', counts[split])
