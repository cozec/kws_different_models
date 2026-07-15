"""Deep KWS reproduction with a filler class (Chen, Parada & Heigold, ICASSP 2014).

Labels follow the paper's convention: 0 = filler (non-keyword), 1 = HIW
("Hi Xiaowen"), 2 = NHWW ("Nihao Wenwen"). The filler class is trained from
the MobvoiHotwords negative manifests (n_*.json), mirroring the paper's use of
negative voice-search queries.

Evaluation reports argmax accuracy plus the paper-style detection rule
(Section 2.3, adapted to clip level): a keyword fires only if its posterior
exceeds a confidence threshold, swept to trade false alarms vs false rejects.
"""
import os
import numpy as np
import torch
from command_loader import CommandLoader
from model import CNN
from train import train, test

RES = 'data/mobvoi_hotword_dataset_resources'
BATCH = 100
EPOCHS = 5
LR = 0.005
PATIENCE = 5
CUDA = False


def make_loader(name, shuffle):
    ds = CommandLoader(os.path.join(RES, name))
    return torch.utils.data.DataLoader(ds, batch_size=BATCH, shuffle=shuffle, num_workers=0)


def collect_posteriors(loader, model):
    """Run the model over a loader; return (probs, labels) arrays."""
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for data, target in loader:
            probs.append(model(data).exp())
            labels.append(target)
    return torch.cat(probs).numpy(), torch.cat(labels).numpy()


if __name__ == '__main__':
    torch.manual_seed(1234)

    train_loader = make_loader('pn_train_local.json', True)
    valid_loader = make_loader('pn_dev_local.json', False)
    test_loader = make_loader('pn_test_local.json', False)
    print('sizes  train=%d  dev=%d  test=%d' % (
        len(train_loader.dataset), len(valid_loader.dataset), len(test_loader.dataset)), flush=True)

    model = CNN(num_classes=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    os.makedirs('checkpoint', exist_ok=True)
    best_valid_loss = np.inf
    iteration = 0
    epoch = 1
    while (epoch < EPOCHS + 1) and (iteration < PATIENCE):
        train(train_loader, model, optimizer, epoch, CUDA)
        valid_loss = test(valid_loader, model, CUDA)
        if valid_loss > best_valid_loss:
            iteration += 1
            print('Loss was not improved, iteration {0}'.format(iteration), flush=True)
        else:
            iteration = 0
            best_valid_loss = valid_loss
            torch.save(model, 'checkpoint/cnn_filler_epoch{}'.format(epoch))
            print('Saving model (epoch {})...'.format(epoch), flush=True)
        epoch += 1

    print('========== FINAL TEST ==========', flush=True)
    probs, labels = collect_posteriors(test_loader, model)
    preds = probs.argmax(1)

    names = ['filler', 'HIW', 'NHWW']
    print('argmax accuracy: %.2f%% (%d/%d)' % (
        100.0 * (preds == labels).mean(), (preds == labels).sum(), len(labels)), flush=True)
    print('confusion matrix (rows=true, cols=pred) [filler, HIW, NHWW]:')
    for t in range(3):
        row = [(preds[labels == t] == p).sum() for p in range(3)]
        print('  %-6s %s' % (names[t], row), flush=True)

    # Paper-style detection (Sec 2.3, clip level): fire the argmax keyword only
    # if its posterior exceeds threshold tau; otherwise output filler.
    kw_conf = probs[:, 1:].max(1)          # confidence = best keyword posterior
    kw_pick = probs[:, 1:].argmax(1) + 1   # which keyword would fire
    is_neg = labels == 0
    print('\nthreshold sweep (FA = negatives that fire; FR = keywords not detected correctly):')
    print('  tau    FA rate     FR rate')
    for tau in [0.3, 0.5, 0.7, 0.9, 0.95, 0.99]:
        fired = kw_conf > tau
        fa = (fired & is_neg).sum() / max(is_neg.sum(), 1)
        detected = fired & ~is_neg & (kw_pick == labels)
        fr = 1.0 - detected.sum() / max((~is_neg).sum(), 1)
        print('  %.2f   %6.3f%%   %7.3f%%' % (tau, 100 * fa, 100 * fr), flush=True)
    print('========== DONE ==========', flush=True)
