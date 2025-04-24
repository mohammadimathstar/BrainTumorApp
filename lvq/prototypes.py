import torch.nn as nn
from lvq.prototypes_gradients import *
from utils.grassmann import init_randn

import torch


class PrototypeLayer(nn.Module):
    def __init__(self,
                 num_prototypes,
                 num_classes,
                 feature_dim,
                 subspace_dim,
                 metric_type='chordal',
                 dtype=torch.float32,
                 device='cpu'
                ):
        """
        Initialize the PrototypeLayer.

        Args:
            num_prototypes (int): Number of prototypes.
            num_classes (int): Number of classes.
            feature_dim (int): Dimension of data features.
            subspace_dim (int): Dimension of subspaces.
            metric_type (str, optional): Type of metric to use. Defaults to 'geodesic'.
            dtype (torch.dtype, optional): Data type of the tensors. Defaults to torch.float32.
            device (str, optional): Device to use ('cpu' or 'cuda'). Defaults to 'cpu'.
        """
        super().__init__()
        self._feature_dim = feature_dim
        self._subspace_dim = subspace_dim
        self._metric_type = metric_type

        # Initialize prototypes
        self.xprotos, self.yprotos, self.yprotos_mat, self.yprotos_comp_mat = init_randn(
            self._feature_dim,
            self._subspace_dim,
            num_of_protos=num_prototypes,
            num_of_classes=num_classes,
            device=device,
        )

        self._number_of_prototypes = self.yprotos.shape[0]

        # Initialize relevance parameters
        self.relevances = nn.Parameter(
            torch.ones((1, self.xprotos.shape[-1]), dtype=dtype, device=device) / self.xprotos.shape[-1]
        )

        # Define a mapping from metric type to the corresponding layer
        self.metric_type_to_layer = {
            'geodesic': GeodesicPrototypeLayer,
            'chordal': ChordalPrototypeLayer
        }
        self.distance_layer = self.metric_type_to_layer.get(self._metric_type)
        if self.distance_layer is None:
            raise ValueError(f"Unsupported metric type: {self.metric_type}")

    def forward(self, xs_subspace: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the PrototypeLayer.

        Args:
            xs_subspace (torch.Tensor): Input subspaces.

        Returns:
            torch.Tensor: Output from the GeodesicPrototypeLayer.
        """

        return self.distance_layer.apply(
            xs_subspace,
            self.xprotos,
            self.relevances
        )

