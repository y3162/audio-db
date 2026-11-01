"""Experimental external Mamba mixer operator; not a production backend."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import torch
from mamba_ssm.ops.selective_scan_interface import mamba_inner_fn
from src.se.models.semambapp.network import SelectiveSSM


def full_mixer(self, hidden):
    batch, length, channels = hidden.shape
    xz = (self.in_proj.weight @ hidden.reshape(-1, channels).T).reshape(-1, batch, length).permute(1, 0, 2)
    return mamba_inner_fn(xz, self.conv1d.weight, self.conv1d.bias, self.x_proj.weight,
                          self.dt_proj.weight, self.out_proj.weight, self.out_proj.bias,
                          -torch.exp(self.A_log.float()), D=self.D.float(),
                          delta_bias=self.dt_proj.bias.float(), delta_softplus=True)

if __name__ == '__main__':
    SelectiveSSM.forward = full_mixer
    sys.argv = [sys.argv[0], 'full', *sys.argv[1:]]
    exec(compile((Path(__file__).parent / 'measure_semamba_backend_step.py').read_text(),
                 str(Path(__file__).parent / 'measure_semamba_backend_step.py'), 'exec'))
