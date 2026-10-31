# プロジェクトと責務

## 研究目的

固定した ASR（重みを更新しない）を使い、音声認識性能を上げる音声強調（SE）を研究する。SE の出力を固定 ASR に通して比べる。

- 主指標は WER（`wer-norm-v1`、micro 集計）。PESQ / STOI は補助で、clean 参照があるときだけ意味を持つ。
- 推論と評価に加え、`src/training/` に SE の学習基盤がある。ASR は学習しない。学習の仕様は `docs/training/MP_SENET.md`。

## モジュールの責務

| 場所 | 責務 | 入口 |
|---|---|---|
| `src/se/` | 独立した SE モデルと共通 API。MP-SENet、TF-GridNet、DeepFilterNet3、MossFormerGAN、SEMamba++、UniPASE、StuPASE | `load_enhancer()`、`build_model()` |
| `src/asr/` | 独立した ASR モデルと共通 API。Parakeet、Whisper、Wav2Vec2、HuBERT、WavLM | `load_recognizer()` → `transcribe()` |
| `src/evaluation/` | 評価パイプライン（入力 → レシピがあれば波形生成 → SE → ASR → 指標 → 結果保存）。SE と ASR は共通 API だけを呼ぶ | `python -m src.evaluation.cli` |
| `src/data/` | コーパスの走査、レシピの検証とケース ID、永続ステージング、DuckDB（`corpora`・`eval` スキーマ）の管理 | `make db-ingest`、`make db-init` |
| `src/training/` | SE の学習。共通の Trainer・バッチ計画・checkpoint と、MP-SENet の損失・Discriminator。推論 API は呼ばない | `python -m src.training.se.cli` |

モデルの一覧と検証状況は `docs/se/`、`docs/asr/MODELS.md`、評価の仕様は `docs/evaluation/PIPELINE.md`。

## 依存方向

現状（import で確認）:

- `src/se/` と `src/asr/` は、他の `src/` モジュールを import しない。`src/asr` は `tests/asr/test_independence.py` が、evaluation・se・モデルライブラリを import しないことを確認している。
- `src/evaluation/` は `src/se`、`src/asr`（遅延 import）、`src/data`（列定義・スキーマ・レシピ検証）を使う。波形生成は `src/evaluation/audio/render.py` にあり、`src/data/` はそれを import しない。
- `src/data/` から `src/evaluation/` への依存もある。`src/data/init_eval.py` と注釈インポータ（`src/data/corpora/chime3_annotations.py`、`src/data/corpora/mpsenet_filelist.py`）が `EvaluationDB` と `AudioSample` を使う。data と evaluation は相互依存のままである。レシピの検証を `src/data/recipe.py` に置いたので、この循環は広がっていない。
- `src/training/` は `src/se`（`build_model`）、`src/data/recipe`、`src/evaluation`（レシピの波形化と名前付きセットの読み取り）を使う。`src/se` と `src/asr` は `src/training` を import しない。学習コードは `src/se/official/` を import しない。

方針:

- `src/se/` と `src/asr/` に、データ・評価・互いの知識を入れない。
- 評価は SE と ASR の内部実装を参照せず、共通 API 経由で使う。
- 現状の相互依存は広げない。新しい import は、既存の方向に足す。

## 推論と学習の責務分離

- `src.asr` の `transcribe()` は推論専用。モデルは `eval()`、全パラメータは `requires_grad=False`、実行は `torch.inference_mode()`。学習・fine-tuning・勾配を通す経路は無い。
- `src.se` の `load_enhancer()` → `enhance()` は推論専用。`build_model()` は勾配を保つ `nn.Module` を返すだけで、学習ループ・損失・discriminator は含まない。
- 学習は `src/training/` に置く。`transcribe()` と `enhance()` の意味は変えない。評価パイプラインは推論 API だけを呼ぶ。MP-SENet の Generator は `build_model()` のモジュールを学習し、学習済み重みは `generator` キーの checkpoint として `load_enhancer()` に渡せる。
