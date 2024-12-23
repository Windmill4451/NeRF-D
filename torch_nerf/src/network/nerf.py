"""
Pytorch implementation of MLP used in NeRF (ECCV 2020).
"""

from typing import Tuple
from typeguard import typechecked

from jaxtyping import Float, jaxtyped
import torch
import torch.nn as nn


class NeRF(nn.Module):
    """
    A multi-layer perceptron (MLP) used for learning neural radiance fields.

    For architecture details, please refer to 'NeRF: Representing Scenes as
    Neural Radiance Fields for View Synthesis (ECCV 2020, Best paper honorable mention)'.

    Attributes:
        pos_dim (int): Dimensionality of coordinate vectors of sample points.
        view_dir_dim (int): Dimensionality of view direction vectors.
        feat_dim (int): Dimensionality of feature vector within forward propagation.
    """

    def __init__(
        self,
        pos_dim: int,
        view_dir_dim: int,
        feat_dim: int = 256,
    ) -> None:
        """
        Constructor of class 'NeRF'.
        """
        super().__init__()

        # TODO
        
        self.pos_dim = pos_dim
        self.view_dir_dim = view_dir_dim
        self.feat_dim = feat_dim
        
        self.linear1 = nn.Linear(pos_dim, 256)
        self.linear2 = nn.Linear(256, 256)
        self.linear3 = nn.Linear(256, 256)
        self.linear4 = nn.Linear(256, 256)
        self.linear5 = nn.Linear(256, 256)
        self.linear6 = nn.Linear(256+pos_dim, 256)
        self.linear7 = nn.Linear(256, 256)
        self.linear8 = nn.Linear(256, 256)
        self.linear9 = nn.Linear(256, feat_dim+1)
        self.linear10 = nn.Linear(feat_dim+view_dir_dim, 128)
        self.linear11 = nn.Linear(128, 3)

    @jaxtyped
    @typechecked
    def forward(
        self,
        pos: Float[torch.Tensor, "num_sample pos_dim"],
        view_dir: Float[torch.Tensor, "num_sample view_dir_dim"],
    ) -> Tuple[Float[torch.Tensor, "num_sample 1"], Float[torch.Tensor, "num_sample 3"]]:
        """
        Predicts color and density.

        Given sample point coordinates and view directions,
        predict the corresponding radiance (RGB) and density (sigma).

        Args:
            pos: The positional encodings of sample points coordinates on rays.
            view_dir: The positional encodings of ray directions.

        Returns:
            sigma: The density predictions evaluated at the given sample points.
            radiance: The radiance predictions evaluated at the given sample points.
        """

        # TODO
        
        relu = nn.ReLU()
        sigmoid = nn.Sigmoid()
        
        x = relu(self.linear1(pos))
        x = relu(self.linear2(x))
        x = relu(self.linear3(x))
        x = relu(self.linear4(x))
        x = relu(self.linear5(x))
        
        x = torch.concat([x, pos], 1)
        
        x = relu(self.linear6(x))
        x = relu(self.linear7(x))
        x = relu(self.linear8(x))
        x = self.linear9(x)
        
        sigma = relu(x[:, :1])
        x = x[:, 1:]
        x = torch.concat([x, view_dir], 1)
        
        x = relu(self.linear10(x))
        radiance = sigmoid(self.linear11(x))
        
        return sigma, radiance
