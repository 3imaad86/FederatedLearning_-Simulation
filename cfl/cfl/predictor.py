"""Accuracy Predictor pour CFL (Wang et al. 2023).

Petit MLP entraine en ligne : il apprend la relation width -> accuracy
observee a partir des samples (w_k, acc_k) renvoyes par les clients, et
sert ensuite au search pour choisir le submodel de chaque client.
"""

import torch
import torch.nn as nn


class AccuracyPredictor(nn.Module):
    """MLP 4 couches : (w) -> accuracy ∈ [0, 1].

    Convention :
      - w : ratio de largeur du submodel ∈ (0, 1].
      - sortie : prob (sigmoid) interpretee comme accuracy attendue ∈ [0, 1].
    """

    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        # 4 couches lineaires (Sec III.B 1 du papier), input dim = 1 (width)
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def _features(self, w):
        """Construit le tensor de features (w,) shape (B, 1), aligne sur le
        device du predictor pour eviter un mismatch CPU/GPU."""
        device = next(self.parameters()).device
        if not torch.is_tensor(w):
            w = torch.as_tensor(w, dtype=torch.float32, device=device)
        else:
            w = w.to(device)
        x = w.float().flatten().unsqueeze(-1)
        return x

    def forward(self, w):
        """Predit l'accuracy attendue pour un width donne (scalaire ou batch).

        Retourne un tensor de meme batch que l'entree, valeurs ∈ [0, 1].
        """
        x = self._features(w)
        out = self.net(x).squeeze(-1)
        return out


class AccuracyPredictorTrainer:
    """Trainer minimal du predictor : un epoch SGD par round (Algo 2).

    Le predictor est attache a la strategie serveur ; cette classe encapsule
    l'optimiseur + la boucle d'entrainement pour eviter de polluer la
    strategy avec ces details.
    """

    def __init__(self, predictor: AccuracyPredictor, lr: float = 0.01):
        self.predictor = predictor
        self.opt = torch.optim.Adam(self.predictor.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        # Buffer de samples (w, acc) collectes au fil des rounds. On les
        # garde sur un horizon long pour stabiliser l'apprentissage du
        # predictor (sinon 1-2 samples/round, training tres bruite).
        self._samples = []
        self._max_buffer = 1024  # plafond pour eviter la croissance memoire
        # Diagnostic
        self.last_n_samples = 0
        self.last_loss = 0.0

    def add_sample(self, w: float, acc: float):
        """Ajoute un sample observe (Algo 2 ligne 5)."""
        self._samples.append((float(w), float(acc)))
        # Rolling buffer : on garde les `_max_buffer` plus recents.
        if len(self._samples) > self._max_buffer:
            self._samples = self._samples[-self._max_buffer:]

    def train_one_epoch(self) -> float:
        """Une epoch d'entrainement sur le buffer (Algo 2).

        Retourne la loss MSE moyenne sur le batch. 0.0 si buffer vide.
        """
        if not self._samples:
            self.last_n_samples = 0
            return 0.0

        device = next(self.predictor.parameters()).device
        ws = torch.tensor(
            [s[0] for s in self._samples], dtype=torch.float32, device=device)
        accs = torch.tensor(
            [s[1] for s in self._samples], dtype=torch.float32, device=device)

        self.predictor.train()
        self.opt.zero_grad()
        preds = self.predictor(ws)
        loss = self.loss_fn(preds, accs)
        loss.backward()
        self.opt.step()
        self.predictor.eval()

        self.last_n_samples = len(self._samples)
        self.last_loss = float(loss.item())
        return self.last_loss

    @torch.no_grad()
    def predict(self, w: float) -> float:
        """Predit l'accuracy pour un width donne. Scalaire en sortie."""
        self.predictor.eval()
        device = next(self.predictor.parameters()).device
        out = self.predictor(torch.tensor([float(w)], device=device))
        return float(out.item())
