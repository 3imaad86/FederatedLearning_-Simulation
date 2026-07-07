"""Boucles train/test partagees par tous les algos FL du repo.

  * train(...)                     : SGD multi-epoch (FedAvg/FedProx/FedNova/...)
  * train_scaffold(...)            : SGD avec correction (c_global - c_local) (SCAFFOLD)
  * test(...)                      : eval (loss, accuracy)
  * test_with_class_accuracies(...): eval + per-class + macro recall/F1
"""

import numpy as np
import torch
import torch.nn as nn

from .metrics import class_accuracies_from_preds, macro_recall_f1_from_preds


# ---------------------------------------------------------------------------
# 1) Entrainement local
# ---------------------------------------------------------------------------

def train(net, loader, epochs, lr, device, mu=0.0, global_params=None,
          momentum=0.0):
    """Entrainement multi-epoch SGD.

    - Si mu > 0 et global_params fourni : ajoute le terme proximal FedProx
          total = ce_loss + (mu / 2) * sum (p - gp)^2
    - Le `train_loss` retourne est la CE pure (sans terme proximal) pour
      que les courbes FedAvg vs FedProx restent comparables. Le terme prox
      contribue uniquement au gradient d'optimisation.
    - momentum : 0.0 par defaut (vanilla SGD), aligne sur les pyproject.toml
      pour permettre une comparaison fair entre tous les algos. FedNova et
      SCAFFOLD exigent momentum=0 (formules theoriques basees sur SGD vanilla).
    """
    net.to(device)
    crit = nn.CrossEntropyLoss().to(device)
    opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum)
    net.train()

    use_prox = (mu > 0 and global_params is not None)
    tot_loss, tot_ex, steps = 0.0, 0, 0
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            ce_loss = crit(net(x), y)
            if use_prox:
                prox = sum(((p - gp) ** 2).sum()
                           for p, gp in zip(net.parameters(), global_params))
                total_loss = ce_loss + (mu / 2.0) * prox
            else:
                total_loss = ce_loss
            total_loss.backward()
            opt.step()

            bs = y.size(0)
            # On log la CE PURE (pas le terme prox) -> courbes comparables.
            tot_loss += ce_loss.item() * bs
            tot_ex += bs
            steps += 1
    return tot_loss / max(tot_ex, 1), steps


def train_scaffold(net, loader, epochs, lr, device, c_global_sd, c_local_sd,
                   momentum=0.0):
    """Entrainement SCAFFOLD (Karimireddy et al. 2020).

    Update local : y = y - lr * (grad + c_global - c_local)
    Le terme (c_global - c_local) corrige le drift du gradient local en non-IID.

    NB sur le momentum :
      - Theoriquement le papier suppose vanilla SGD (momentum=0).
      - En pratique on peut utiliser momentum=0.9 pour une comparaison juste
        contre FedAvg-with-momentum (la correction reste qualitativement utile,
        meme si la formule exacte est une approximation).

    Returns: (avg_loss, num_steps)
    """
    net.to(device)
    crit = nn.CrossEntropyLoss().to(device)
    opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum)
    net.train()

    # Convert control variates en tensors sur le bon device
    cg = {name: c_global_sd[name].to(device) for name in c_global_sd}
    cl = {name: c_local_sd[name].to(device) for name in c_local_sd}

    tot_loss, tot_ex, steps = 0.0, 0, 0
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(net(x), y)
            loss.backward()

            # correction SCAFFOLD : grad <- grad + (c_global - c_local).
            # on verifie que name est dans cg et cl : si c_local persiste
            # depuis un run anterieur avec un modele different, la cle
            # pourrait manquer.
            for name, p in net.named_parameters():
                if p.grad is not None and name in cg and name in cl:
                    p.grad.add_(cg[name] - cl[name])

            opt.step()

            bs = y.size(0)
            tot_loss += loss.item() * bs
            tot_ex += bs
            steps += 1
    return tot_loss / max(tot_ex, 1), steps


# ---------------------------------------------------------------------------
# 2) Evaluation (forward sans grad)
# ---------------------------------------------------------------------------

def _forward_eval(net, loader, device, collect_preds=False):
    """Forward pass sans grad. Retourne (tot_loss, tot_ok, tot_ex, ys, ps).

    Si collect_preds=False, ys/ps sont vides (economie memoire).
    """
    net.to(device)
    crit = nn.CrossEntropyLoss().to(device)
    net.eval()

    tot_loss, tot_ok, tot_ex = 0.0, 0, 0
    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = net(x)
            bs = y.size(0)
            tot_loss += crit(out, y).item() * bs
            preds = out.argmax(1)
            tot_ok += (preds == y).sum().item()
            tot_ex += bs
            if collect_preds:
                ys.append(y.cpu().numpy())
                ps.append(preds.cpu().numpy())
    return tot_loss, tot_ok, tot_ex, ys, ps


def test(net, loader, device):
    """Eval simple. Retourne (loss_moyen, accuracy)."""
    tot_loss, tot_ok, tot_ex, _, _ = _forward_eval(net, loader, device)
    return tot_loss / max(tot_ex, 1), tot_ok / max(tot_ex, 1)


def test_with_class_accuracies(net, loader, device, num_classes=10):
    """Eval complete : loss, accuracy globale, accuracies par-classe, macro recall/F1."""
    tot_loss, _, tot_ex, ys, ps = _forward_eval(net, loader, device, collect_preds=True)
    if tot_ex == 0:
        return 0.0, 0.0, [0.0] * num_classes, 0.0, 0.0

    y_true = np.concatenate(ys)
    y_pred = np.concatenate(ps)
    overall_acc = float((y_true == y_pred).sum() / tot_ex)
    class_accs = class_accuracies_from_preds(y_true, y_pred, num_classes=num_classes)
    macro_recall, macro_f1 = macro_recall_f1_from_preds(y_true, y_pred)
    return tot_loss / tot_ex, overall_acc, class_accs, macro_recall, macro_f1
