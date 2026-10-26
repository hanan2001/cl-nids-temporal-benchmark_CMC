"""
Réseau neuronal pour la détection d'intrusions.

CORRECTIONS vs version précédente:
- Un SEUL enregistrement de hook par couche linéaire (le double-hook est supprimé).
- La clé d'activation utilise le PRÉFIXE DE PARAMÈTRE ('param_network.<idx>')
  afin que gpm.py puisse relier chaque couche à son tenseur de poids
  'network.<idx>.weight' sans décalage de clés.
"""

import torch
import torch.nn as nn


class CyberNetwork(nn.Module):
    """
    Réseau feed-forward pour la détection d'intrusions.
    Sans BatchNorm (petits batchs de classes rares).
    """

    def __init__(self, input_size, hidden_sizes, output_size):
        super(CyberNetwork, self).__init__()

        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, output_size))

        self.network = nn.Sequential(*layers)
        self.activations = {}
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        """
        Un hook par couche linéaire. La clé est 'param_network.<idx>' où <idx>
        est l'indice du module dans self.network. Le tenseur de poids
        correspondant est nommé 'network.<idx>.weight' par PyTorch, ce qui
        permet à la GPM de faire le lien exact (voir gpm.py).
        """
        def hook_fn(name):
            def hook(module, inp, out):
                if isinstance(out, tuple):
                    out = out[0]
                # entrée de la couche linéaire = représentation à projeter
                # inp est un tuple (x,) ; on garde l'entrée, pas la sortie,
                # car GPM projette dans l'espace des ENTRÉES de la couche.
                if isinstance(inp, tuple) and len(inp) > 0 and isinstance(inp[0], torch.Tensor):
                    self.activations[name] = inp[0].detach()
                else:
                    self.activations[name] = out.detach()
            return hook

        for name, module in self.network.named_children():
            if isinstance(module, nn.Linear):
                h = module.register_forward_hook(hook_fn(f"param_network.{name}"))
                self.hooks.append(h)

    def forward(self, x):
        self.activations = {}
        return self.network(x)

    def get_representations_for_gpm(self):
        """Ne renvoie que les activations des couches linéaires."""
        return {n: a for n, a in self.activations.items() if n.startswith('param_')}