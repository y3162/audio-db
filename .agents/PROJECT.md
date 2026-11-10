# プロジェクトと責務

固定した ASR に SE の出力を通し、WER で比べる。PESQ / STOI / SI-SDR は clean 参照があるときだけの補助指標。ASR の重みは更新しない。

## モジュール

| 場所 | 責務 | 入口 |
|---|---|---|
| `src/se/` | SE の推論モデル。MP-SENet、TF-GridNet、TF-Locoformer、DeepFilterNet3、MossFormerGAN、SEMamba++ | `load_enhancer()`、`build_model()` |
| `src/asr/` | ASR の推論。Parakeet、Whisper、Wav2Vec2、HuBERT、WavLM | `load_recognizer()` → `transcribe()` |
| `src/evaluation/` | 名前付き評価セット → 必要なら波形生成 → SE → ASR → 指標 | `python -m src.evaluation.cli` |
| `src/data/` | コーパス走査、レシピ、ステージング、評価用 DuckDB | `make db-init`、`make db-ingest` |
| `src/reproduction/` | 再現用 DB、スナップショット、オンデマンド描画。学習ループは持たない | `python -m src.reproduction.cli` |
| `src/experiments/` | 唯一の学習基盤。YAML から組み立て、`audio-db-experiment-v1` を保存して再開する | `python -m src.experiments.cli` |
| `src/experiments/calc/` | 損失、STFT、scheduler、参照ステップ。CLI ではない | テストと Trainer が呼ぶ |

`src/training/` と `configs/training/` は無い。旧形式 `audio-db-training-v1` の checkpoint は再開しない。

## 推論と学習

- `transcribe()` は `eval()`、全パラメータ `requires_grad=False`、`torch.inference_mode()`。
- `enhance()` は推論専用。`build_model()` は学習可能な `nn.Module` を返すだけで、損失も更新順も含まない。
- 学習は `enhance()` と `transcribe()` を呼ばない。
- `src/se/` と `src/asr/` は、他の `src/` を import しない。
- `src/evaluation/` は `src/se`、`src/asr`、`src/data` を使う。`src/reproduction` は import しない。
- `src/reproduction/` は `src/data/recipe` を使う。`src/se`、`src/asr`、`src/evaluation`、`src/experiments` は import しない。
- `src/experiments/` は `src/se.build_model` と `src/reproduction` の描画関数を使う。学習中に DuckDB を開かない。
