# audio-db

## 音声強調

`available_enhancers()` は MP-SENet、TF-GridNet、DeepFilterNet3、MossFormerGAN、SEMamba++、UniPASE、StuPASE を返します。単一ファイルのモデルはチェックポイントパスを渡し、UniPASE と StuPASE は部品名からパスへの辞書を渡します。公式波形との対応と、CLI のファイル出力など波形以外で未確認の範囲は `docs/se/VALIDATION.md` にあります。

```python
from src.se import available_enhancers, build_model, load_enhancer

print(available_enhancers())
model = build_model('mp_senet')  # 未学習の nn.Module
enc = load_enhancer('mp_senet', checkpoint='/path/to/g_best_vb', device='cpu')
enhanced = enc.enhance(noisy, sample_rate=16000)
```

設定は辞書または JSON で渡せます。詳細と未対応事項は [設計](docs/se/ARCHITECTURE.md)、[モデル](docs/se/MODELS.md)、[検証](docs/se/VALIDATION.md) を参照してください。

TF-GridNet は以下の variant を選択できます（現時点では検証途中です）。

```python
model = build_model('tf_gridnet', variant='urgent24')
enhancer = load_enhancer('tf_gridnet', checkpoint='/path/to/21epoch.pth', variant='urgent24')
model_ms = build_model('tf_gridnet', variant='ms-snsd')
```

可変長バッチは `enhancer.enhance_batch(padded_waveforms, 16000, lengths)` で処理できます。

```python
enc = load_enhancer('unipase', {
    'dewavlm': '/path/to/DeWavLM-Omni.pt',
    'adapter': '/path/to/Adapter.pt',
    'vocoder': '/path/to/Vocoder_DWO-L1.pt',
    'postnet': '/path/to/PostNet.pt',
})
enc = load_enhancer('stupase', {
    'dewavlm': '/path/to/DeWavLM-R.pt',
    'cfm': '/path/to/CFM.pt',
    'vocoder': '/path/to/Vocoder_Mel-16k.pt',
})
```
