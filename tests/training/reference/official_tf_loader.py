"""Load the actual ESPnet TF-GridNet encoder and decoder from its Git tree."""
import subprocess
import sys
import tempfile
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ESP = ROOT / 'src/se/official/espnet'
_TEMP = None


def official_tf_gridnet():
    global _TEMP
    _TEMP = tempfile.TemporaryDirectory()
    paths = ['espnet2/enh/encoder', 'espnet2/enh/decoder', 'espnet2/enh/separator',
             'espnet2/enh/layers', 'espnet2/layers', 'espnet2/legacy/nets/pytorch_backend',
             'espnet2/torch_utils']
    archive = subprocess.Popen(['git', '-C', str(ESP), 'archive', 'HEAD', *paths], stdout=subprocess.PIPE)
    subprocess.run(['tar', '-x', '-C', _TEMP.name], stdin=archive.stdout, check=True)
    archive.stdout.close()
    if archive.wait() != 0:
        raise RuntimeError('ESPnet git archive failed')
    sys.path.insert(0, _TEMP.name)
    tc = types.ModuleType('torch_complex'); tc.__path__ = []
    tensor = types.ModuleType('torch_complex.tensor')
    class UnusedLegacyComplex: pass
    tensor.ComplexTensor = UnusedLegacyComplex
    functional = types.ModuleType('torch_complex.functional')
    tc.tensor, tc.functional = tensor, functional
    sys.modules.update({'torch_complex': tc, 'torch_complex.tensor': tensor,
                        'torch_complex.functional': functional})
    typeguard = types.ModuleType('typeguard'); typeguard.typechecked = lambda fn: fn
    sys.modules['typeguard'] = typeguard
    from espnet2.enh.separator.tfgridnet_separator import TFGridNet
    return TFGridNet
