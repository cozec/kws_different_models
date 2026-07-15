"""Streaming keyword detection with Wav2Keyword.

Modes:
  python stream_demo.py sim   # (a) simulate a stream from test-set clips,
                              #     score detections against ground truth
  python stream_demo.py mic   # (b) live microphone detection (sounddevice)

Detector = sliding 1 s window over a ring buffer, re-scored every HOP_MS,
posterior smoothing over the last SMOOTH windows, threshold TAU on the best
keyword posterior, REFRACTORY_S dead-time after each firing (Deep KWS-style
decision rule at window rate).
"""
import os
import sys
import random
import numpy as np
import soundfile as sf
import torch
from wav2keyword_local import KWS, CLASSES, DEVICE

SR = 16000
WINDOW = SR                  # 1 s model window
HOP_MS = 100
HOP = SR * HOP_MS // 1000
SMOOTH = 5                   # windows averaged (Deep KWS Eq. 2, window-rate)
TAU = 0.85                   # keyword posterior threshold
REFRACTORY_S = 1.0
KEYWORD_IDS = [i for i, c in enumerate(CLASSES) if c not in ('unknown', 'silence')]

ROOT = 'data/speech_commands_v1'


def load_model():
    model = KWS(n_class=len(CLASSES)).to(DEVICE)
    sd = torch.load('checkpoint/w2k_v1/best_model.pth', map_location=DEVICE)
    model.load_state_dict(sd)
    model.eval()
    return model


class StreamingDetector:
    def __init__(self, model):
        self.model = model
        self.buf = np.zeros(WINDOW, dtype=np.float32)   # ring buffer (1 s)
        self.recent = []                                # last SMOOTH posteriors
        self.last_fire_t = -1e9
        self.t = 0.0                                    # stream clock (s)

    def push(self, chunk):
        """Feed a chunk of audio; returns (keyword, prob, t) if fired else None."""
        self.buf = np.concatenate([self.buf[len(chunk):], chunk])
        self.t += len(chunk) / SR

        x = torch.from_numpy(self.buf).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            probs = torch.softmax(self.model(x), dim=-1)[0].cpu().numpy()
        self.recent.append(probs)
        if len(self.recent) > SMOOTH:
            self.recent.pop(0)
        smoothed = np.mean(self.recent, axis=0)
        self.smoothed = smoothed          # exposed for monitoring/verbose mode

        kw_probs = smoothed[KEYWORD_IDS]
        best = int(np.argmax(kw_probs))
        prob = float(kw_probs[best])
        if prob > TAU and (self.t - self.last_fire_t) > REFRACTORY_S:
            self.last_fire_t = self.t
            return CLASSES[KEYWORD_IDS[best]], prob, self.t
        return None


# ---------------------------------------------------------------- (a) sim ---
def build_sim_stream(n_keywords=8, seed=42):
    """Concatenate test clips: keyword events separated by unknown words,
    silence clips, and background noise. Returns (audio, ground_truth)."""
    rng = random.Random(seed)
    tdir = os.path.join(ROOT, 'testing')
    kw_classes = [c for c in CLASSES if c not in ('unknown', 'silence')]

    def clip(cls):
        d = os.path.join(tdir, cls)
        wav, _ = sf.read(os.path.join(d, rng.choice(sorted(os.listdir(d)))), dtype='float32')
        if len(wav) < SR:
            wav = np.pad(wav, (0, SR - len(wav)))
        return wav[:SR]

    audio, truth = [np.zeros(SR, dtype=np.float32)], []   # 1 s lead-in silence
    t = 1.0
    for _ in range(n_keywords):
        # distractor between keywords: unknown word / silence / quiet noise
        kind = rng.choice(['unknown', 'silence', 'unknown'])
        dis = clip(kind)
        audio.append(dis)
        t += len(dis) / SR

        kw = rng.choice(kw_classes)
        w = clip(kw)
        audio.append(w)
        truth.append((t + 0.5, kw))          # keyword centered ~t+0.5
        t += len(w) / SR
    audio.append(np.zeros(SR, dtype=np.float32))
    return np.concatenate(audio), truth


def run_sim():
    model = load_model()
    audio, truth = build_sim_stream()
    print('stream length: %.1f s, %d keyword events' % (len(audio) / SR, len(truth)))
    print('ground truth :', ', '.join('%.1fs=%s' % (t, k) for t, k in truth))
    print()

    det = StreamingDetector(model)
    firings = []
    for i in range(0, len(audio) - HOP + 1, HOP):
        out = det.push(audio[i:i + HOP])
        if out:
            kw, p, t = out
            firings.append((t, kw, p))
            print('  %5.1fs  FIRE  %-8s p=%.3f' % (t, kw, p))

    # score: hit if fired same keyword within +-0.7 s of event center
    hits, used = 0, set()
    for tt, tk in truth:
        for j, (ft, fk, _) in enumerate(firings):
            if j not in used and fk == tk and abs(ft - tt) <= 0.7:
                hits += 1
                used.add(j)
                break
    fa = len(firings) - len(used)
    print('\nresult: %d/%d keywords detected, %d false alarms, %d total firings'
          % (hits, len(truth), fa, len(firings)))


# ---------------------------------------------------------------- (b) mic ---
def run_mic(verbose=False):
    import sounddevice as sd
    model = load_model()
    det = StreamingDetector(model)
    print('listening (Ctrl-C to stop)… say one of:')
    print('  ' + ', '.join(CLASSES[2:]))
    if verbose:
        print('verbose: printing top hypothesis whenever it changes '
              '(unknown/silence never fire — they are rejection classes)')
    state = {'last_top': None}

    def cb(indata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        out = det.push(indata[:, 0].astype(np.float32).copy())
        if verbose:
            top = int(np.argmax(det.smoothed))
            if CLASSES[top] != state['last_top']:
                state['last_top'] = CLASSES[top]
                print('  %6.1fs    top: %-8s p=%.2f' % (det.t, CLASSES[top], det.smoothed[top]), flush=True)
        if out:
            kw, p, t = out
            print('  %6.1fs  🔔 %-8s p=%.3f' % (t, kw, p), flush=True)

    with sd.InputStream(samplerate=SR, channels=1, blocksize=HOP, callback=cb):
        try:
            while True:
                sd.sleep(1000)
        except KeyboardInterrupt:
            print('\nstopped.')


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'sim'
    if mode == 'mic':
        run_mic(verbose='-v' in sys.argv)
    else:
        run_sim()
