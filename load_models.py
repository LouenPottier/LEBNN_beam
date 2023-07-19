# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 11:24:35 2023

@author: LP263296
"""

import torch
from models import EnergyLatNN, FCLatNN, FCNN, EncoderF, DecoderF





fcnn=FCNN(128)
fcnn.load_state_dict(torch.load('fcnn_00003L1'))

fclat=FCLatNN(2, 128)
fclat.load_state_dict(torch.load('fclat_0001L1'))

enerlat=EnergyLatNN(2, 128, 32)
enerlat.load_state_dict(torch.load('enerlat_0001'))

enc_f=EncoderF()
dec_f=DecoderF()


H_fclat = torch.load('H_fclat')
H_enerlat = torch.load('H_enelat')

