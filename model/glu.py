import torch.nn as nn
import math


class GLU(nn.Module):
    """        
    Args:
        input_dim (int): Dimension of the input features
        intermediate_dim (int): Dimension of the intermediate representation (hidden dimension)
        output_dim (int): Dimension of the output features
        activation (nn.Module, optional): Activation function for the gate path. 
                                          Defaults to nn.SiLU (Swish activation).
        use_bias (bool, optional): Whether to include bias terms in the linear projections.
                                  Defaults to False.
        dropout_p (float, optional): Dropout probability applied after gating. Defaults to 0.0.
    """
    def __init__(self, 
                 input_dim, 
                 intermediate_dim, 
                 output_dim, 
                 activation=nn.SiLU,
                 use_bias=False,
                 dropout_p=0.0):
        super().__init__()

        self.input_dim = input_dim
        self.intermediate_dim = intermediate_dim
        self.output_dim = output_dim
        self.activation = activation()
        self.dropout = nn.Dropout(dropout_p)

        # Fused projection for both gate and content paths
        self.fused_gate_up_proj = nn.Linear(input_dim, 2 * intermediate_dim, bias=use_bias)
        nn.init.xavier_uniform_(self.fused_gate_up_proj.weight, gain=math.sqrt(2.0))
        # Project gated representation to output dimension
        self.down_proj = nn.Linear(intermediate_dim, output_dim, bias=use_bias)
        
    def forward(self, x):
        fused = self.fused_gate_up_proj(x)
        gate, up = fused.chunk(2, dim=-1)
        gate = self.activation(gate)
        
        gated_output = up * gate
        gated_output = self.dropout(gated_output)
        
        out = self.down_proj(gated_output)
        return out
    