import argparse
import numpy as np
import os

from utils.grassmann import grassmann_repr, grassmann_repr_full, compute_distances_on_grassmann_mdf
from utils.glvq import *
from lvq.prototypes import PrototypeLayer


LOW_BOUND_LAMBDA = 0.001


class Model(nn.Module):
    def __init__(self,
                 num_classes: int,
                 feature_extractor: torch.nn.Module,
                 args: argparse.Namespace,
                 add_on_layers: nn.Module = nn.Identity(),
                 device='cpu'
                 ):
        super().__init__()
        assert num_classes > 0

        self._num_classes = num_classes
        self._loss_function = args.cost_fun
        self._metric_type = args.metric_type
        self._num_prototypes = args.num_of_protos
        # self._feature_dim = args.depth_of_net_last_layer
        self._feature_dim = args.num_features
        self._subspace_dim = args.dim_of_subspace
        self._coef_subspace_dim = args.coef_dim_of_subspace

        # Set the feature extraction network
        self.feature_extractor = feature_extractor
        self._add_on = add_on_layers


        # Create the prototype layers
        self.prototype_layer = PrototypeLayer(
            num_prototypes=self._num_prototypes,
            num_classes=self._num_classes,
            feature_dim=self._feature_dim,
            subspace_dim=self._subspace_dim,
            metric_type=self._metric_type,
            device=device,
        )

    @property
    def prototypes_require_grad(self) -> bool:
        return self.prototype_layer.xprotos.requires_grad

    @prototypes_require_grad.setter
    def prototypes_require_grad(self, val: bool):
        self.prototype_layer.xprotos.requires_grad = val

    @property
    def features_require_grad(self) -> bool:
        return any([param.requires_grad for param in self._net.parameters()])

    @features_require_grad.setter
    def features_require_grad(self, val: bool):
        for param in self._net.parameters():
            param.requires_grad = val

    @property
    def add_on_layers_require_grad(self) -> bool:
        return any([param.requires_grad for param in self._add_on.parameters()])

    @add_on_layers_require_grad.setter
    def add_on_layers_require_grad(self, val: bool):
        for param in self._add_on.parameters():
            param.requires_grad = val

    def forward(self,
                inputs: torch.Tensor,
                **kwargs):
        """
        Forward pass of the model.

        Args:
            inputs (torch.Tensor): A batch of input data.

        Returns:
            tuple: Distances and Qw.
        """
        # Perform forward pass through the feature extractor
        features = self.feature_extractor(inputs)
        features = self._add_on(features)

        # Get Grassmann representation of the features
        subspaces = grassmann_repr(features, self._coef_subspace_dim * self._subspace_dim)

        # Compute distance between subspaces and prototypes
        distance, Qw = self.prototype_layer(
            subspaces)  # SHAPE: (batch_size, num_prototypes, D: dim_of_data, d: dim_of_subspace)

        return distance, Qw

    def forward_partial(self,
                        inputs: torch.Tensor):
        # Perform forward pass through the feature extractor
        features = self.feature_extractor(inputs)
        features = self._add_on(features)

        # Get Grassmann representation of the features
        subspaces, Vh, S = grassmann_repr_full(features, self._subspace_dim)

        output = compute_distances_on_grassmann_mdf(
            subspaces,
            self.prototype_layer.xprotos,
            self._metric_type,
            self.prototype_layer.relevances,
        )

        return features, subspaces, Vh, S, output

    def save(self, directory_path: str):
        if not os.path.isdir(directory_path):
            os.mkdir(directory_path)

        with open(directory_path + "/model.pth", 'wb') as f:
            torch.save(self, f)

    def save_state(self, directory_path: str):
        if not os.path.isdir(directory_path):
            os.mkdir(directory_path)

        with open(directory_path + "/model_state.pth", 'wb') as f:
            torch.save(self.state_dict(), f)

    @staticmethod
    def load(directory_path: str):
        return torch.load(directory_path + '/model.pth', map_location=torch.device('cpu'))




def return_model(fname):
    with np.load(fname + '.npz', allow_pickle=True) as f:
        xprotos, yprotos = f['xprotos'], f['yprotos']
        lamda = f['lamda']
        print(f"train accuracy: {f['accuracy_of_train_set'][-1]}, "
              f"\t validation accuracy: {f['accuracy_of_validation_set'][-1]} ({np.max(f['accuracy_of_validation_set'])})")

    return xprotos, yprotos, lamda
