# 学習スモークセット

登録コード `src/data/training_sets.py` は、`resource.duckdb` を読み取り専用で開き、`data/staging/training/` に書いてから `data/db/training.duckdb` へ登録する。`resource.duckdb` へは書かない。この二つのパスがいま在るかは [配置と記録](INVENTORY.md)。

| セット | 件数 | 音声 | 雑音 | SNR |
|---|---|---|---|---|
| `libri_demand_smoke_train` | 8 | LibriSpeech train-clean-100。話者 103, 1034, 1040, 1069, 1081, 1088, 1098, 1116 から 1 発話 | DEMAND `OOFFICE`, `OMEETING`, `OHALLWAY`, `DLIVING` の `ch01` | 0, 5, 10, 15 dB |
| `libri_demand_smoke_valid` | 4 | LibriSpeech dev-clean。話者 1272, 1462, 1673, 174 から 1 発話 | DEMAND `NPARK`, `NFIELD` の `ch01` | 0, 5, 10, 15 dB |

出力は 16 kHz、最長 2 秒、教師は `dry_clean`。`noise_start` と SNR はレシピに保存する。学習と検証で、発話パス、話者、雑音ファイルは重ならない。LibriSpeech test-clean と CHiME3 et05 は入っていない。

このスモークセットは VoiceBank-DEMAND ではない。VoiceBank の分割は [コーパス](CORPORA.md)、ファイルの場所は [配置と記録](INVENTORY.md)。

登録のやり直し:

```bash
.venvs/se/bin/python -m src.data.training_sets
```

同じ内容の再実行は行を増やさない。
