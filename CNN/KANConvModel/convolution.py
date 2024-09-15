import torch
import numpy as np
from typing import List, Tuple, Union


def calc_out_dims(matrix, kernel_side, stride, dilation, padding):
    batch_size,n_channels,n, m = matrix.shape
    h_out = np.floor((n + 2 * padding[0] - kernel_side - (kernel_side - 1) * (dilation[0] - 1)) / stride[0]).astype(int) + 1
    w_out = np.floor((m + 2 * padding[1] - kernel_side - (kernel_side - 1) * (dilation[1] - 1)) / stride[1]).astype(int) + 1
    b = [kernel_side // 2, kernel_side// 2]
    return h_out,w_out,batch_size,n_channels

def kan_conv2d(matrix: Union[List[List[float]], np.ndarray], 
             kernel, 
             kernel_side,
             stride= (1, 1), 
             dilation= (1, 1), 
             padding= (0, 0),
             device= "cuda"
             ) -> torch.Tensor:
    h_out, w_out,batch_size,n_channels = calc_out_dims(matrix, kernel_side, stride, dilation, padding)
    
    matrix_out = torch.zeros((batch_size,n_channels,h_out,w_out)).to(device)
    unfold = torch.nn.Unfold((kernel_side,kernel_side), dilation=dilation, padding=padding, stride=stride)


    for channel in range(n_channels):
        conv_groups = unfold(matrix[:,channel,:,:].unsqueeze(1)).transpose(1, 2)
        for k in range(batch_size):
            matrix_out[k,channel,:,:] = kernel.forward(conv_groups[k,:,:]).reshape((h_out,w_out))
    return matrix_out

def multiple_convs_kan_conv2d(matrix, 
             kernels, 
             kernel_side,
             stride= (1, 1), 
             dilation= (1, 1), 
             padding= (0, 0),
             device= "cuda"
             ) -> torch.Tensor:
    h_out, w_out,batch_size,n_channels = calc_out_dims(matrix, kernel_side, stride, dilation, padding)
    n_convs = len(kernels)
    matrix_out = torch.zeros((batch_size,n_channels*n_convs,h_out,w_out)).to(device)
    unfold = torch.nn.Unfold((kernel_side,kernel_side), dilation=dilation, padding=padding, stride=stride)
    conv_groups = unfold(matrix[:,:,:,:]).view(batch_size, n_channels,  kernel_side*kernel_side, h_out*w_out).transpose(2, 3)
    for channel in range(n_channels):
        for kern in range(n_convs):
            matrix_out[:,kern  + channel*n_convs,:,:] = kernels[kern].conv.forward(conv_groups[:,channel,:,:].flatten(0,1)).reshape((batch_size,h_out,w_out))
    return matrix_out


def add_padding(matrix: np.ndarray, 
                padding: Tuple[int, int]) -> np.ndarray:
    n, m = matrix.shape
    r, c = padding
    
    padded_matrix = np.zeros((n + r * 2, m + c * 2))
    padded_matrix[r : n + r, c : m + c] = matrix
    
    return padded_matrix