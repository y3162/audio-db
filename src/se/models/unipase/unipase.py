# Copyright 2025 Cisco Systems, Inc. and its affiliates
# Adapted under Apache-2.0
# Source: https://github.com/cisco-open/pase/blob/main/models/pase.py
# License included under licenses/LICENSE_pase.

import torch
import torch.nn as nn
from .wavlm.feature_extractor_plc import WavLM_feat as Encoder
from .adapter.vocos.adapter import VocosAdapter as Adapter
from .vocoder.vocos.vocoder import VocosVocoder as Decoder
from .postnet.tfgrid_cws_res import TFGridNet as PostNet
from torchaudio.functional import resample


class UniPASE(nn.Module):
    def __init__(
        self, 
        dewavlm_ckpt_path="/data/hdd0/xiaobin.rong/pretrained/UniPASE/DeWavLM-Omni.pt",
        adapter_ckpt_path="/data/hdd0/xiaobin.rong/pretrained/UniPASE/Adapter.pt",
        vocoder_ckpt_path="/data/hdd0/xiaobin.rong/pretrained/UniPASE/Vocoder_DWO-L1.pt",
        postnet_ckpt_path="/data/hdd0/xiaobin.rong/pretrained/UniPASE/PostNet.pt",
    ):
        super().__init__()
        self.encoder = Encoder(dewavlm_ckpt_path, output_layer=[1,24])
        self.adapter = Adapter.from_pretrained(adapter_ckpt_path)
        self.decoder = Decoder.from_pretrained(vocoder_ckpt_path)
        self.postnet = PostNet.from_pretrained(postnet_ckpt_path)

    @torch.no_grad()
    def forward(self, x, sr_in, sr_out=None, enable_plc=True):
        """
        Args:
            x (torch.Tensor): noisy speech with shape of (B, L) or (B, 1, L)
            sr_in (int): sampling rate of input speech
            sr_out (int): sampling rate of output speech
            enable_plc (bool): whether to perform PLC
        Return:
            y (torch.Tensor): enhanced speech with shape of (B, L).
        """
        n_samples = x.shape[-1]
        
        noisy_feat_a, enh_feat_p = self.encoder(x, sr=sr_in, mask=enable_plc)
        enh_feat_a = self.adapter(noisy_feat_a, enh_feat_p)
                
        enh_wav_16k = self.decoder(enh_feat_a.transpose(1, 2))  # (B, 1, T)
        
        if sr_out is None:
            sr_out = sr_in
        
        if sr_out < 16000:
            enh_wav = resample(enh_wav_16k, orig_freq=16000, new_freq=sr_out)
        elif sr_out == 16000:
            enh_wav = enh_wav_16k
        else:
            enh_wav = self.postnet.infer(enh_wav_16k, sr_in=16000, sr_out=sr_out)  # (B, 1, T)
 
        y = enh_wav.squeeze(1)  # (B, T)
        
        if sr_out == sr_in:
            out_samples = n_samples
        else:
            out_samples = int(n_samples * sr_out / sr_in)
        
        if y.shape[-1] < out_samples:
            y = torch.nn.functional.pad(y, (0, out_samples-y.shape[-1]))
        else:
            y = y[..., :out_samples]
 
        return y


    
if __name__ == "__main__":
    
    model = UniPASE()

    x = torch.randn(2, 16000*4)
    
    y = model(x, sr_in=16000, sr_out=24000, enable_plc=True)
    print(y.shape)
    
    y = model(x, sr_in=16000, sr_out=44100, enable_plc=True)
    print(y.shape)
    
    y = model(x, sr_in=16000, sr_out=48000, enable_plc=True)
    print(y.shape)

    # from ptflops import get_model_complexity_info
    
    # with torch.inference_mode():
    #     macs, params = get_model_complexity_info(model, (16000,), print_per_layer_stat=False)
    
    # params = 0
    # for p in model.parameters():
    #     params += p.numel()
    # print(macs, f"{params / 1e6:.2f} M")

