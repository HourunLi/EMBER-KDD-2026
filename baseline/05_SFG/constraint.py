import torch
import numpy as np
from typing import Dict, Any, List

class Callback():
    """Base class for callbacks."""

    def __init__(self):
        """Initialize the Callback."""
        pass

    def on_train_begin(self):
        """Called when the training begins."""
        pass

    def on_train_end(self):
        """Called when the training ends."""
        pass

    def on_epoch_begin(self):
        """Called at the beginning of each epoch."""
        pass

    def on_epoch_end(self):
        """Called at the end of each epoch."""
        pass

    def on_batch_begin(self):
        """Called at the beginning of each batch."""
        pass

    def on_batch_end(self):
        """Called at the end of each batch."""
        pass

    def on_loss_begin(self):
        """Called before the computation of the loss."""
        pass

    def on_loss_end(self):
        """Called after the computation of the loss."""
        pass

    def on_step_begin(self):
        """Called before the optimizer step."""
        pass

    def on_step_end(self):
        """Called after the optimizer step."""
        pass

    
class Constraint_SAGE(Callback):
    """Callback for applying constraints to the SAGE model."""

    def __init__(self, model: torch.nn.Module, parameters: Dict[str, Any], device: torch.device, with_constraint: bool):
        """Initialize the Constraint_SAGE callback.

        Args:
            model (torch.nn.Module): The SAGE model.
            parameters (dict): Dictionary with parameters for the model and the FISTA accelerated DBF algorithm.
            device (torch.device): Device to be used for computations.
            with_constraint (bool): Whether to apply the constraint or not.
            constraint_type (str): Type of constraint to apply.
        """
        self.with_constraint = with_constraint
        self.model = model
        self.device = device
        self.num_layers = parameters['num_layers']
        self.neurons_per_layer = parameters['neurons_per_layer']
        if (with_constraint):
            self.nit = parameters['nit']
            self.rho = parameters['rho']
            self.criterion = parameters['criterion']
            self.cnst = parameters['cnst']
            self.alpha = parameters['alpha']
    
    def Constraint_Fista(self, A: torch.Tensor, B: torch.Tensor, layer_index: int, generator_weights: torch.Tensor) -> torch.Tensor:
        """Apply the constraint using the FISTA algorithm.

        Args:
            A (torch.Tensor): Input matrix A.
            B (torch.Tensor): Input matrix B.
            layer_index (int): Index of the current layer.

        Returns:
            torch.Tensor: Updated weights.
        """
        w = torch.cat((self.model.gcn_stack[layer_index].lin_r.weight.data, self.model.gcn_stack[layer_index].lin_l.weight.data))

        Y0 = torch.zeros((self.neurons_per_layer[-1], self.neurons_per_layer[0]), device=self.device)
        Y = Y0
        Yold = Y0
        gam = (1 / ((torch.linalg.matrix_norm(A, ord=2) * torch.linalg.matrix_norm(B, ord=2) + 
                     torch.finfo(torch.float32).eps) ** 2)).item()

        I = torch.eye(A.shape[1], device=self.device)
        I = torch.cat((I, I), dim=1)
        A = A @ I

        for i in range(self.nit):
            eta = i / (i + 1 + self.alpha)
            Z = Y + eta * (Y - Yold)
            Yold = Y
            
            w_new = w - A.T @ Z @ B.T
            w_new[w_new < 0] = 0.0
            
            # keep a closed convex set
            # use detach to do in-palce operate
            mask = generator_weights.detach() < 0
            generator_weights.data[mask] = 0.0
            
            epsilon = 1e-8
            min_val = torch.min(generator_weights)
            max_val = torch.max(generator_weights)
            generator_weights = (generator_weights - min_val) / (max_val - min_val + epsilon)   
            T = A @ w_new @ B * generator_weights.unsqueeze(0) # * for broadcast
            # T = A @ w_new @ B
            s = torch.linalg.svdvals(T)

            criterion = torch.linalg.matrix_norm(w_new - w, ord='fro').item()
            constraint = torch.linalg.vector_norm(s[s > self.rho] - self.rho, ord=2).item()

            Yt = Z + gam * T
            [u1, s1, v1] = torch.linalg.svd(Yt / gam, full_matrices=False)

        
            s1 = torch.clamp(s1, min=0, max=self.rho)   # important! can set a extrmely small rho
            Y = Yt - gam * (u1 * s1) @ v1

            if (criterion < self.criterion and constraint < self.cnst):
                return w_new
        return w_new

    def on_batch_end(self, previous_weights: List[torch.Tensor], generator_weights: torch.Tensor):
        """Callback called at the end of each batch.

        Args:
            previous_weights (List[torch.Tensor]): List of previous weights of the form [num_layers x 2] (w0 and w1).
        """
        if self.with_constraint:
            # Initialize B as an identity matrix of shape = length of input of the first layer
            B = torch.eye(self.neurons_per_layer[0], device=self.device)

            for layer_index in range(self.num_layers):
                # Initialize A as an identity matrix of shape = length of output of the last layer
                A = torch.eye(self.neurons_per_layer[-1], device=self.device)

                # Compute A
                for layer_index_A in range(self.num_layers - 1, -1, -1):
                    if layer_index_A > layer_index:
                        prev_weight = previous_weights[layer_index_A][0] + previous_weights[layer_index_A][1]
                        A = A @ prev_weight

                # Apply constraint on w
                w = self.Constraint_Fista(A, B, layer_index, generator_weights)

                # Write new weights
                dim_w0 = int(w.shape[0] / 2)
                w0 = w[:dim_w0, :]
                w1 = w[dim_w0:, :]
                self.model.gcn_stack[layer_index].lin_r.weight.data = w0
                self.model.gcn_stack[layer_index].lin_l.weight.data = w1

                B = (w0 + w1) @ B
        
        else:
            pass

                
                
class Constraint_GCN(Callback):
    def __init__(self, model: torch.nn.Module, parameters: Dict[str, Any], device: torch.device, with_constraint: bool, constraint_type: str):
        """Initialize the Constraint_GCN callback.

        Args:
            model (torch.nn.Module): The SAGE model.
            parameters (dict): Dictionary with parameters for the model and the FISTA accelerated DBF algorithm.
            device (torch.device): Device to be used for computations.
            with_constraint (bool): Whether to apply the constraint or not.
            constraint_type (str): Type of constraint to apply.
        """
        self.with_constraint = with_constraint
        self.constraint_type = constraint_type
        self.model = model
        self.device = device
        self.num_layers = parameters['num_layers']
        self.neurons_per_layer = parameters['neurons_per_layer']
        if (with_constraint):
            self.nit = parameters['nit']
            self.rho = parameters['rho']
            self.criterion = parameters['criterion']
            self.cnst = parameters['cnst']
            self.alpha = parameters['alpha']

    def SpectralNorm_Constraint(self, w):
        """Spectral normalization.

        Args:
            w (torch.Tensor): weight matrix.

        Returns:
            torch.Tensor: Updated weight.
        """
        norm = torch.linalg.matrix_norm(w, ord = 2)
        w_new = (w / norm) * self.rho ** (1/self.num_layers)
        return w_new
    
    
    def Constraint_Fista(self, A: torch.Tensor, B: torch.Tensor, layer_index: int) -> torch.Tensor:
        """Apply the constraint using the FISTA algorithm.

        Args:
            A (torch.Tensor): Input matrix A.
            B (torch.Tensor): Input matrix B.
            layer_index (int): Index of the current layer.

        Returns:
            torch.Tensor: Updated weights.
        """
        w = self.model.gcn_stack[layer_index].lin.weight.data
        
        Y0 = torch.zeros((self.neurons_per_layer[-1], self.neurons_per_layer[0]), device=self.device)
        Y = Y0
        Yold = Y0
        gam = (1 / ((torch.linalg.matrix_norm(A, ord=2) * torch.linalg.matrix_norm(B, ord=2) + torch.finfo(torch.float32).eps) ** 2)).item()

        I = torch.eye(A.shape[1], device=self.device)
        A = A @ I        
        
        for i in range(self.nit):
            eta = i / (i + 1 + self.alpha)
            Z = Y + eta * (Y - Yold)
            Yold = Y
            
            w_new = w - A.T @ Z @ B.T
            w_new[w_new < 0] = 0.0
            
            T = A @ w_new @ B
            s = torch.linalg.svdvals(T)
            
            criterion = torch.linalg.matrix_norm(w_new - w, ord='fro').item()
            constraint = torch.linalg.vector_norm(s[s > self.rho] - self.rho, ord=2).item()
            
            Yt = Z + gam * T
            [u1, s1, v1] = torch.linalg.svd(Yt / gam, full_matrices=False)
            
            s1 = torch.clamp(s1, min=0, max=self.rho)
            Y = Yt - gam * (u1 * s1) @ v1
            
            if (criterion < self.criterion and constraint < self.cnst):
                return w_new
        return w_new
    

    def on_batch_end(self, previous_weights: List[torch.Tensor]):
        """Callback called at the end of each batch.

        Args:
            previous_weights (List[torch.Tensor]): List of previous weights.
        """
        if self.with_constraint:
            B = torch.eye(self.neurons_per_layer[0], device=self.device)

            for layer_index in range(self.num_layers):
                A = torch.eye(self.neurons_per_layer[-1], device=self.device)

                for layer_index_A in range(self.num_layers - 1, -1, -1):
                    if layer_index_A > layer_index:
                        prev_weight = previous_weights[layer_index_A]
                        A = A @ prev_weight

                w = self.Constraint_Fista(A, B, layer_index)
                self.model.gcn_stack[layer_index].lin.weight.data = w

                B = w @ B
        
        else:
            pass
