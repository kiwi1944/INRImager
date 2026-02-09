import numpy as np
import torch
import torch.nn as nn
import math
from torch.nn.parameter import Parameter

class PhysicalModel(object): # defines how the image projects to the measurements
    def __init__(self, used_channel, value_scale):
        self.used_channel = used_channel
        self.value_scale = value_scale
        self.wavelen = used_channel['wavelen']
        self.RIS_RCS = (4 * math.pi * (used_channel['RIS_element_size'][0] * used_channel['RIS_element_size'][1]).pow(2) / (used_channel['wavelen']).pow(2)).pow(0.5)
        self.voxel_RCS = (4 * math.pi * (used_channel['ROI_element_size'][0] * used_channel['ROI_element_size'][1]).pow(2) / (used_channel['wavelen']).pow(2)).pow(0.5)
        # the above RCS is sqrt RCS in fact

        self.h_tx_ris = used_channel['h_tx_ris']
        self.h_tx_roi = used_channel['h_tx_roi']
        self.h_ris_rx = used_channel['h_ris_rx']
        self.h_roi_rx = used_channel['h_roi_rx']
        self.h_roi_ris = used_channel['h_roi_ris']
        self.RIS_phase = used_channel['RIS_phase']


    def illuminate(self, img):

        img = img * self.voxel_RCS
        RIS_phase = self.RIS_phase * self.RIS_RCS
        h_tx_ris_rx = (self.h_tx_ris.unsqueeze(0).expand(RIS_phase.shape[0], self.h_tx_ris.shape[0], self.h_tx_ris.shape[1]) * RIS_phase.unsqueeze(1)) @ self.h_ris_rx
        h_tx_roi_rx = self.h_tx_roi * img @ self.h_roi_rx
        h_tx_ris_roi_rx = ((self.h_tx_ris.unsqueeze(0).expand(RIS_phase.shape[0], self.h_tx_ris.shape[0], self.h_tx_ris.shape[1]) * RIS_phase.unsqueeze(1)) @ self.h_roi_ris.t()) * img @ self.h_roi_rx
        h_tx_roi_ris_rx = (self.h_tx_roi * img @ self.h_roi_ris).unsqueeze(0).expand(RIS_phase.shape[0], self.h_tx_roi.shape[0], self.h_roi_ris.shape[1]) * RIS_phase.unsqueeze(1) @ self.h_ris_rx
        mea = h_tx_roi_rx + h_tx_ris_rx + h_tx_ris_roi_rx + h_tx_roi_ris_rx  # num_glimpses, antenna_num_tx, antenna_num_rx
        
        mea = mea.view(mea.shape[0], -1) # num_glimpses, antenna_num_tx * antenna_num_rx
        mea = torch.cat((mea.real, mea.imag), 1) # num_glimpses, antenna_num_tx * antenna_num_rx * 2
        mea = mea * self.value_scale

        return mea


class PositionalEncoder(): # Input Gaussian Positional Encoding
    def __init__(self, config):
        self.pe_type = 'gauss'
        self.B = torch.randn((config.pe_embedding_size, config.pe_indim)) * config.pe_value_scale
        self.B = self.B.to(config.device)
        self.device = config.device

    def embedding(self, x):
        x_embedding = (2. * np.pi * x) @ self.B.t()
        x_embedding = torch.cat([torch.sin(x_embedding), torch.cos(x_embedding)], dim=-1)
        return x_embedding.to(device=self.device)


class SirenLayer(nn.Module):
    def __init__(self, in_f, out_f, w0=30, is_first=False, is_last=False): # frequency of sin function is given by 30
        super().__init__()
        self.in_f = in_f
        self.w0 = w0
        self.linear = nn.Linear(in_f, out_f)
        self.is_first = is_first
        self.is_last = is_last
        self.init_weights()

    def init_weights(self):
        b = 1 / self.in_f if self.is_first else np.sqrt(6 / self.in_f) / self.w0
        with torch.no_grad():
            self.linear.weight.uniform_(-b, b)

    def forward(self, x):
        x = self.linear(x)
        return x if self.is_last else torch.sin(self.w0 * x)


class SIREN(nn.Module):
    def __init__(self, config):
        super(SIREN, self).__init__()

        num_layers = config.net_depth
        hidden_dim = config.net_width
        input_dim = config.net_indim
        output_dim = config.net_outdim

        layers = [SirenLayer(input_dim, hidden_dim, is_first=True)]
        for i in range(1, num_layers - 1):
            layers.append(SirenLayer(hidden_dim, hidden_dim))
        layers.append(SirenLayer(hidden_dim, output_dim, is_last=True))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        out = self.model(x)
        out = torch.sigmoid(out)
        return out

