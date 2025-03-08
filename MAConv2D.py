import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MAConv2D(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 kernel_size: int, 
                 stride: int = 1, 
                 padding: int = 0,
                 groups: int = 1, 
                 bias: bool = False, 
                 freq_bands: int = 6, 
                 manifold_dims: int = 2, 
                 manifold_groups: int = 4,
                 eps: float = 1e-4):
        super().__init__()
        # Dimension validation
        assert in_channels % groups == 0, "in_channels must be divisible by groups"
        
        # Special handling: when groups equals out_channels (depthwise conv), auto-adjust manifold_groups
        if groups == out_channels:
            # For depthwise convolution, set manifold_groups equal to groups
            manifold_groups = groups
        else:
            # For normal convolution, ensure manifold_groups is a multiple of groups
            assert manifold_groups % groups == 0, "manifold_groups must be a multiple of groups"
            # Also ensure out_channels is divisible by groups*manifold_groups
            assert out_channels % (groups * manifold_groups) == 0, "out_channels must be divisible by groups*manifold_groups"
        
        assert manifold_dims in [2], "Only 2D manifold transformation is supported"

        # Core parameters
        self.in_channels = in_channels          
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.manifold_groups = manifold_groups
        self.manifold_dims = manifold_dims
        self.freq_bands = freq_bands
        self.eps = eps
        
        # Dimension splitting calculation - special handling for depthwise convolution
        self.out_per_group = out_channels // groups  # Output channels per group
        
        # Special handling for depthwise convolution
        if groups == out_channels:
            self.manifold_per_group = 1  # Simplified handling
            self.out_per_manifold = 1    # Simplified handling
        else:
            self.manifold_per_group = manifold_groups // groups  # Manifold groups per convolution group
            self.out_per_manifold = self.out_per_group // self.manifold_per_group  # Output channels per manifold group

        # Convolution weights [out_channels, in_per_group, k, k]
        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels//groups, kernel_size, kernel_size)
        )

        # Manifold transformation parameters
        self.manifold_scale = nn.Parameter(
            torch.Tensor(manifold_groups, self.out_per_manifold, manifold_dims)
        )
        self.manifold_rot = nn.Parameter(
            torch.Tensor(manifold_groups, self.out_per_manifold, 
                        manifold_dims*(manifold_dims-1)//2)
        )

        # Learnable dilation coefficient (initial γ=1)
        self.dilation_log = nn.Parameter(
            torch.full((manifold_groups,), math.log(1.0))  # Initialize uniformly for all manifold groups
        )   

        # Spectral modulation
        self.freq_weights = nn.Parameter(
            torch.Tensor(manifold_groups, self.out_per_manifold, freq_bands)
        )
        self.register_buffer('freq_base', torch.linspace(0.5, 2.0, freq_bands))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        # Initialization
        self._init_parameters()
        self._precompute_bases()

    def _init_parameters(self):
        # Manifold parameters
        nn.init.uniform_(self.manifold_scale, -0.4, 0.4)
        nn.init.uniform_(self.manifold_rot, -math.pi/12, math.pi/12)
        
        # Frequency weights
        nn.init.normal_(self.freq_weights, mean=0.0, std=0.1)
        with torch.no_grad():
            self.freq_weights[..., 0] += 0.5

        # Convolution weights
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        # Bias
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def _precompute_bases(self):
        k = self.kernel_size
        u = torch.linspace(-(k-1)/2, (k-1)/2, k)
        v = torch.linspace(-(k-1)/2, (k-1)/2, k)
        grid = torch.stack(torch.meshgrid(u, v, indexing='ij'), dim=-1)
        self.register_buffer('base_grid', grid)  # [k, k, d]

    def _construct_pd_matrix(self, scale_params, rot_params):
        diag = torch.exp(scale_params * 0.5)
        triu = torch.tanh(rot_params)
        
        L = torch.diag_embed(diag)
        if self.manifold_dims == 2:
            L[..., 0, 1] = triu.squeeze(-1)
        else:
            idx = 0
            for i in range(3):
                for j in range(i+1,3):
                    L[..., i, j] = triu[..., idx]
                    idx += 1
        return L

    def _manifold_transform(self):
        MG = self.manifold_groups
        k = self.kernel_size
        d = self.manifold_dims

        # Construct positive definite matrix [MG, out_per_manifold, d, d]
        L = self._construct_pd_matrix(self.manifold_scale, self.manifold_rot)

        # Apply dilation [MG, k, k, d]
        gamma = torch.exp(self.dilation_log).view(MG,1,1,1)
        dilated_grid = self.base_grid * gamma

        # Manifold transformation [MG, out_per_manifold, k, k, d]
        grid_flat = dilated_grid.view(MG,1,k*k,d)  # [MG,1,k^2,d]
        transformed = torch.matmul(grid_flat, L.transpose(-1,-2))  # [MG, out_per_manifold, k^2, d]
        return transformed.view(MG, self.out_per_manifold, k, k, d)

    def _spectral_modulation(self, w):
        G = self.groups
        MG = self.manifold_groups
        k = self.kernel_size
        
        # Reshape weights [groups, manifold_per_group, out_per_manifold, in_per_group, k, k]
        w_reshape = w.view(
            G,
            self.manifold_per_group,
            self.out_per_manifold,
            self.in_channels // G,
            k,
            k
        )

        # Calculate dynamic spectral basis
        gamma = torch.exp(self.dilation_log).view(MG,1,1,1)
        dilated_grid = self.base_grid * gamma  # [MG, k, k, d]
        r = torch.norm(dilated_grid, dim=-1)  # [MG, k, k]
        r_norm = r / (r.amax(dim=(1,2), keepdim=True) + self.eps)
        
        # Generate basis functions [MG, freq_bands, k, k]
        spectral_base = torch.exp(-(r_norm.unsqueeze(1) * self.freq_base.view(1,-1,1,1))**2)
        
        # Modulate weights [MG, out_per_manifold, k, k]
        modulation = torch.einsum('mbf,mfkh->mbkh', 
                                self.freq_weights, 
                                spectral_base)  # [MG, out_per_manifold, k, k]
        
        # Apply modulation [groups, manifold_per_group, out_per_manifold, 1, k, k]
        return w_reshape * modulation.view(G, self.manifold_per_group, 
                                         self.out_per_manifold, 1, k, k)

    def forward(self, x):
        B, C, H, W = x.shape
        G = self.groups
        k = self.kernel_size

        manifold = self._manifold_transform()  

        MG = self.manifold_groups
        opm = self.out_per_manifold
        mpg = self.manifold_per_group

        manifold = manifold.view(G, mpg, opm, k, k, self.manifold_dims)

        manifold = manifold.permute(0, 2, 1, 3, 4, 5)
        
        manifold = manifold.reshape(G, opm * mpg, k, k, self.manifold_dims)

        in_per_group = self.in_channels // G
        manifold = manifold.unsqueeze(2).expand(-1, -1, in_per_group, -1, -1, -1)

        manifold = manifold.reshape(self.out_channels * in_per_group, k, k, self.manifold_dims)

        w = self.weight.view(G, self.out_per_group, C // G, k, k)
        w_modulated = self._spectral_modulation(w).view_as(self.weight)

        sampled = F.grid_sample(
            w_modulated.view(-1, 1, k, k),
            manifold,  
            align_corners=True
        ).view(self.out_channels, C // G, k, k)

        return F.conv2d(x, sampled, self.bias, self.stride, self.padding, groups=G)