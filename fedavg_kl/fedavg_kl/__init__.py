"""FedAvg+KL : agregation FedAvg + distillation KL via proxy partage.

Variante hybride inspiree de la partie proxy/KMeans-DRE d'EdgeFD
(Liu et al. 2025, arXiv:2508.14769) : l'agregation FedAvg des poids reste
le mecanisme principal, la distillation joue un role de regularisation.

Composants :
  - client_app.py : entrainement local CE + KL distill, KMeans local pour
                    le DRE, filtrage logits ID/OOD sur proxy
  - server_app.py : agregation FedAvg des poids + agregation des logits
  - strategy.py   : FedAvgKLStrategy (sous-classe de FedAvgStrategy)
  - kmeans_dre.py : KMeans + seuil ID adaptatif
"""
