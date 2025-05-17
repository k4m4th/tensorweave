import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional

# --- Neural Field Module ---

class NeuralFourierField(nn.Module):
    """
    A neural field module that interpolates tensors using random Fourier features.
    
    This model first encodes the input coordinates using random Fourier features,
    then passes the encoded features through a multilayer perceptron (MLP) to produce
    a scalar field (potential). The field can be used in applications such as implicit
    neural representations.
    """
    
    def __init__(self,
                 input_dim: int = 3,
                 num_fourier_features: int = 32,
                 harmonic: bool = True,
                 distribution: str = "Normal",
                 dist_variance: float = "1",
                 potential_scale: float = 1e2,
                 length_scales: List[float] = [1e2, 5e2, 1e3],
                 decay_scales: List[float] = [1e1, 1e2, 1e3],
                 learnable: bool = True,
                 hidden_layers: List[int] = [512],
                 output_dim: int = 1,
                 activation: nn.Module = nn.SiLU(),
                 seed: int = 404,
                 device: Optional[str] = None) -> None:
        """
        Initializes the NeuralFourierField.
        
        Args:
            input_dim (int): Dimension of the input coordinates.
            num_fourier_features (int): Number of Fourier features per length scale.
            potential_scale (float): Scaling factor applied to the final output.
            length_scales (List[float]): List of length scales for the random Fourier features.
            decay_scales (List[float]): List of decay scales for the decaying dimension.
            hidden_layers (List[int]): Sizes of hidden layers in the MLP.
            output_dim (int): Dimension of the output.
            activation (nn.Module): Activation function to use between layers.
            seed (int): Seed for reproducibility.
            device (Optional[str]): Device on which to run the model.
        """
        super().__init__()
        # Set device: if not provided, use cuda if available
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_rff = num_fourier_features > 0  # Flag to use random Fourier features
        self.activation = activation
        self.harmonic = harmonic
        
        # -------------------- Random Fourier Features Initialization -------------------- #
        if self.use_rff:
            # Use a dedicated generator for reproducibility
            gen = torch.Generator(device=self.device)
            gen.manual_seed(seed)
            
            if harmonic:
                if distribution == "Normal":
                    # Create a random projection matrix for spatial dimensions (all but last)
                    self.k_xy = torch.randn(input_dim - 1, num_fourier_features, device=self.device, generator=gen)
                elif distribution == "Cauchy":
                    self.k_xy = torch.distributions.Cauchy(0, dist_variance).sample((input_dim - 1, num_fourier_features)).to(device)
                # self.k_xy = self.k_xy / torch.norm(self.k_xy, dim=0, keepdim=True)
                # Compute the norm along the projection directions for decay computation
                self.k_z = torch.norm(self.k_xy, dim=0, keepdim=True)
                
            else:
                self.k_xy = torch.randn(input_dim, num_fourier_features, device=self.device, generator=gen)
            
            # Random bias for Fourier features
            self.bias_vector = 2 * torch.pi * torch.randn(num_fourier_features, device=self.device, generator=gen)
            
            # Store log-transformed length and decay scales for numerical stability
            if learnable:
                self.potential_scale = nn.Parameter(torch.log10(torch.tensor(potential_scale, dtype=torch.float32, device=self.device)))
                self.length_scales = nn.Parameter(torch.log10(torch.tensor(length_scales, device=self.device)))
                self.decay_scales = nn.Parameter(torch.log10(torch.tensor(decay_scales, device=self.device)))
            else:
                self.potential_scale = torch.log10(torch.tensor(potential_scale, dtype=torch.float32, device=self.device))
                self.length_scales = torch.log10(torch.tensor(length_scales, device=self.device))
                self.decay_scales = torch.log10(torch.tensor(decay_scales, device=self.device))
            
        # -------------------- MLP Construction -------------------- #
        # Determine the input dimension to the MLP based on whether RFF is used
        if self.use_rff:
            mlp_input_dim = 2 * num_fourier_features * len(length_scales)
        else:
            mlp_input_dim = input_dim
        
        # Build layer dimensions list: input, hidden layers, then output
        dims = [mlp_input_dim] + hidden_layers + [output_dim]
        layers = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1], device=self.device))
            if self.activation is not None:
                layers.append(self.activation)
        # Final layer without activation
        layers.append(nn.Linear(dims[-2], dims[-1], device=self.device))
        self.mlp = nn.Sequential(*layers)
        
        # Xavier (Glorot) initialization for all Linear layers in the MLP
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (N, input_dim).
            
        Returns:
            torch.Tensor: Scaled scalar field output of shape (N, output_dim).
        """
        if self.use_rff:
            features = self._encode_rff(x)
        else:
            features = x
        # Scale the output potential
        return self.mlp(features) * (10 ** self.potential_scale)
        
    def _encode_rff(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Encodes input coordinates using random Fourier features.
        
        For each provided scale, the spatial coordinates (all except the last dimension)
        are projected, phase-shifted, and modulated by an exponential decay based on the last dimension.
        
        Args:
            coords (torch.Tensor): Input tensor of shape (N, input_dim).
            
        Returns:
            torch.Tensor: Encoded tensor of shape (N, 2 * num_fourier_features * num_scales).
        """
        encoded_features = []
        num_scales = len(self.length_scales)
        
        for i in range(num_scales):
            scale = 10 ** (self.length_scales[i])
            d_scale = 10 ** (self.decay_scales[i])
            
            # Instead of directly using coords, incorporate the learnable transform T:
            transformed_coords = coords[:, :-1]
            proj = transformed_coords @ (self.k_xy / scale)
            decay = torch.exp(- coords[:, -1].unsqueeze(-1) @ (self.k_z / d_scale))
            cos_features = torch.cos(proj) * decay
            sin_features = torch.sin(proj) * decay
            encoded_features.append(torch.cat([cos_features, sin_features], dim=-1))
            
        # Concatenate all encoded features along the feature dimension
        return torch.cat(encoded_features, dim=-1)


# --- Gradient and Hessian Computations ---

def compute_gradient(model: nn.Module,
                     coords: torch.Tensor,
                     normalize: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the scalar field and its gradient with respect to the input coordinates.
    
    Args:
        model (nn.Module): The neural network model.
        coords (torch.Tensor): Input tensor of shape (N, input_dim).
        normalize (bool): If True, the gradient is normalized.
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple (scalar_field, grad_field) where:
            - scalar_field: Tensor of shape (N, output_dim).
            - grad_field: Tensor of shape (N, input_dim) representing the gradient.
    """
    coords = coords.requires_grad_(True)
    scalar_field = model(coords)
    # Compute gradients with create_graph=True to enable higher order derivatives
    grad_field = torch.autograd.grad(outputs=scalar_field,
                                     inputs=coords,
                                     grad_outputs=torch.ones_like(scalar_field),
                                     create_graph=True,
                                     retain_graph=True)[0]
    if normalize:
        # Avoid division by zero by adding a small constant
        norm = torch.norm(grad_field, dim=-1, keepdim=True) + 1e-8
        grad_field = grad_field / norm
    return scalar_field, grad_field


def compute_hessian(model: nn.Module,
                    coords: torch.Tensor,
                    chunk_size: int = 1024,
                    device: Optional[str] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes the scalar field, its gradient, and the Hessian matrix for the input coordinates.
    
    This function processes the input in chunks to manage memory consumption.
    
    Args:
        model (nn.Module): The neural network model.
        coords (torch.Tensor): Input tensor of shape (N, input_dim).
        chunk_size (int): Number of points to process per chunk.
        device (Optional[str]): Device on which to perform computations.
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - scalar_fields: Tensor of shape (N, output_dim).
            - gradient_field: Tensor of shape (N, input_dim).
            - hessian_matrix: Tensor of shape (N, input_dim, input_dim).
    """
    device = device if device is not None else (model.device if hasattr(model, "device") else "cpu")
    scalar_fields_list, gradients_list, hessians_list = [], [], []
    
    # Process the input coordinates in chunks
    for start_idx in range(0, coords.shape[0], chunk_size):
        end_idx = start_idx + chunk_size
        x_chunk = coords[start_idx:end_idx]
        # Compute scalar field, gradient, and Hessian for the current chunk
        s, g, h = _compute_hessian_chunk(model, x_chunk)
        scalar_fields_list.append(s)
        gradients_list.append(g)
        hessians_list.append(h)
    
    # Concatenate results from all chunks along the batch dimension
    scalar_fields = torch.cat(scalar_fields_list, dim=0)
    gradient_field = torch.cat(gradients_list, dim=0)
    hessian_matrix = torch.cat(hessians_list, dim=0)
    
    return scalar_fields, gradient_field, hessian_matrix


def _compute_hessian_chunk(model: nn.Module, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes the Hessian (second-order derivatives) of the scalar field for a chunk of inputs.
    
    Args:
        model (nn.Module): The neural network model.
        x (torch.Tensor): Input tensor of shape (batch_size, input_dim).
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - scalar_field: Tensor of shape (batch_size, output_dim).
            - grad_vector_field: Tensor of shape (batch_size, input_dim).
            - hessian_matrix: Tensor of shape (batch_size, input_dim, input_dim).
    """
    x.requires_grad_(True)
    scalar_field = model(x)
    # Compute the first derivative (gradient)
    grad_vector_field = torch.autograd.grad(outputs=scalar_field,
                                            inputs=x,
                                            grad_outputs=torch.ones_like(scalar_field),
                                            create_graph=True,
                                            retain_graph=True)[0]
    hessian_rows = []
    # Compute second derivatives row-wise
    for i in range(x.shape[1]):
        grad_i = grad_vector_field[:, i]
        # Compute the gradient of each component of the gradient vector
        hessian_row = torch.autograd.grad(outputs=grad_i,
                                          inputs=x,
                                          grad_outputs=torch.ones_like(grad_i),
                                          create_graph=True,
                                          retain_graph=True)[0]
        hessian_rows.append(hessian_row.unsqueeze(-1))
    hessian_matrix = torch.cat(hessian_rows, dim=-1)
    return scalar_field, grad_vector_field, hessian_matrix


def compute_hessian_eval(model: nn.Module,
                         coords: torch.Tensor,
                         chunk_size: int = 1024,
                         device: Optional[str] = None
                         ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    device = device if device else (model.device if hasattr(model, "device") else "cpu")
    scalar_fields_list, gradients_list, hessians_list = [], [], []

    model.to(device).eval()
    
    with torch.no_grad():

        for start_idx in range(0, coords.shape[0], chunk_size):
            end_idx = start_idx + chunk_size
            # Explicitly clone & detach chunk input to avoid reference leaks
            x_chunk = coords[start_idx:end_idx].detach().clone().to(device).requires_grad_(True)
            
            with torch.enable_grad():
                s_chunk, g_chunk, h_chunk = _compute_hessian_chunk_eval(model, x_chunk)

            # Immediately move tensors to CPU & detach completely
            scalar_fields_list.append(s_chunk.cpu().detach())
            gradients_list.append(g_chunk.cpu().detach())
            hessians_list.append(h_chunk.cpu().detach())

            # Explicitly delete GPU tensors to break any hidden references
            del s_chunk, g_chunk, h_chunk, x_chunk
            torch.cuda.empty_cache()

        scalar_fields = torch.cat(scalar_fields_list, dim=0)
        gradient_field = torch.cat(gradients_list, dim=0)
        hessian_matrix = torch.cat(hessians_list, dim=0)

    return scalar_fields, gradient_field, hessian_matrix

def _compute_hessian_chunk_eval(model: nn.Module, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    
    scalar_field = model(x)

    grad_vector_field = torch.autograd.grad(
        outputs=scalar_field,
        inputs=x,
        grad_outputs=torch.ones_like(scalar_field),
        create_graph=True,
        retain_graph=True,  # Required for Hessian
    )[0]

    hessian_rows = []
    for i in range(x.shape[1]):
        grad_i = grad_vector_field[:, i]
        hessian_row = torch.autograd.grad(
            outputs=grad_i,
            inputs=x,
            grad_outputs=torch.ones_like(grad_i),
            retain_graph=True,  # Necessary for multiple Hessian rows
            create_graph=False,
        )[0]
        hessian_rows.append(hessian_row.unsqueeze(-1))

        # Free intermediate GPU memory immediately
        del hessian_row, grad_i
        torch.cuda.empty_cache()

    hessian_matrix = torch.cat(hessian_rows, dim=-1)

    return scalar_field, grad_vector_field, hessian_matrix
