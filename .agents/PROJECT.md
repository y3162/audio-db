# プロジェクトと責務

## 研究目的

固定した ASR（重みを更新しない）を使い、音声認識性能を上げる音声強調（SE）を研究する。SE の出力を固定 ASR に通して比べる。

- 主指標は WER（`wer-norm-v1`、micro 集計）。PESQ / STOI は補助で、clean 参照があるときだけ意味を持つ。
- 推論と評価に加え、`src/training/` に SE の学習基盤がある。ASR は学習しない。学習の仕様は `docs/training/MP_SENET.md`。

## モジュールの責務

| 場所 | 責務 | 入口 |
|---|---|---|
| `src/se/` | 独立した SE モデルと共通 API。MP-SENet、TF-GridNet、TF-Locoformer、DeepFilterNet3、MossFormerGAN、SEMamba++ | `load_enhancer()`、`build_model()` |
| `src/asr/` | 独立した ASR モデルと共通 API。Parakeet、Whisper、Wav2Vec2、HuBERT、WavLM | `load_recognizer()` → `transcribe()` |
| `src/evaluation/` | 評価パイプライン（入力 → レシピがあれば波形生成 → SE → ASR → 指標 → 結果保存）。SE と ASR は共通 API だけを呼ぶ | `python -m src.evaluation.cli` |
| `src/data/` | コーパスの走査、レシピの検証とケース ID、永続ステージング、DuckDB（`corpora`・`eval` スキーマ）の管理 | `make db-ingest`、`make db-init` |
| `src/reproduction/` | 公式再現用の DB（`repro` スキーマ）、staging、監査、スナップショット、VoiceBank 固定ペアの準備、MS-SNSD の混合計画。学習ループは持たない。固定ペアの学習時読み込みは `pipeline/` がスナップショットだけを読む。`delivery: on_demand` の MS-SNSD は `pipeline/ondemand.py` が、DNS の TF-Locoformer は `pipeline/dns_ondemand.py` が、条件と元素材から PCM_16 を再構築する。P7 の `length_plan_no_sample_retry` は 16 kHz の区間読みのままである。P7.1 の `dns_pcm16_accept_v2` は受理済み区間を再構築し、16 kHz でない雑音だけを描画時にリサンプルする。公式規模の受理 Dataset は `tf_locoformer.dns.accept_recipe@1` で、P7 の `tf_locoformer.dns.recipe@1` とは別 ID である。VoiceBank のオンデマンド固定ペアは `prepare/resample.py` が 48 kHz を学習時に 16 kHz PCM_16 にする。DeepFilterNet3 のオンライン混合は `protocols/online_dfn3.py` がスナップショットの素材から 48 kHz・3 秒を描画し、混合 WAV は書かない。SEMamba++ のオンライン劣化は `protocols/online_semamba.py` がスナップショットの素材から 16 kHz・48,000 サンプルを描画し、劣化 WAV は書かない。公式素材の Dataset は blocked。検証は `docs/training/reproduction/P9_VERIFICATION.md`。`pipeline/pregenerated.py` はオンデマンド以外の chunk Dataset を拒否する。ESPnet chunk の複数 rank 計画は `pipeline/chunking.py` が、呼び出し側の発話順と長さの列から作り、DuckDB も音声も読まない | `python -m src.reproduction.cli` |
| `src/training/` | SE の学習。共通の Trainer・バッチ計画・checkpoint。モデルごとの損失と更新順は `src/training/se/`。推論 API は呼ばない | `python -m src.training.se.cli` |
| `src/experiments/` | 実験の組み立て。YAML のクラスを Factory が作り、Builder が明示参照を解決する。Phase 1 は CPU / FP32 の最小学習。Phase 2-A は `audio-db-experiment-v1` の保存と、同じディレクトリでの中断再開。Phase 2-B は CPU / Gloo の world size 2 DDP と、同じ world size の別プロセス再開。AMP、CUDA、FSDP、本格 SE は未実装。`src/training/` と `src/reproduction/` は import しない | `load_experiment()`、`build_experiment()`、`python -m src.experiments.cli` |

モデルの一覧と検証状況は `docs/se/`、`docs/asr/MODELS.md`、評価の仕様は `docs/evaluation/PIPELINE.md`。

## 依存方向

現状（import で確認）:

- `src/se/` と `src/asr/` は、他の `src/` モジュールを import しない。`src/asr` は `tests/asr/test_independence.py` が、evaluation・se・モデルライブラリを import しないことを確認している。
- `src/evaluation/` は `src/se`、`src/asr`（遅延 import）、`src/data`（列定義・スキーマ・レシピ検証）を使う。波形生成は `src/evaluation/audio/render.py` にあり、`src/data/` はそれを import しない。学習データの来歴は checkpoint 横の `data_provenance.json` を `src/evaluation/provenance.py` が読む。`src.reproduction` は import しない。
- `src/data/` から `src/evaluation/` への依存もある。`src/data/init_eval.py` と注釈インポータ（`src/data/corpora/chime3_annotations.py`、`src/data/corpora/mpsenet_filelist.py`）が `EvaluationDB` と `AudioSample` を使う。data と evaluation は相互依存のままである。レシピの検証を `src/data/recipe.py` に置いたので、この循環は広がっていない。
- `src/training/` は `src/se`（`build_model`）、`src/data/recipe`、`src/evaluation`（レシピの波形化、名前付きセットの読み取り、来歴 JSON のバイト列）を使う。再現経路では `src/reproduction.pipeline` の `TrainingData` だけを使う。checkpoint 保存時に rank 0 が `data_provenance.json` を書く。`src/se` と `src/asr` は `src/training` を import しない。学習コードは `src/se/official/` を import しない。学習は `src/reproduction.db` を開かない。`repro.runs` への登録は学習後の `register_training_run` である。
- `src/reproduction/` は `src/data/recipe`（正規化と SHA-256）を使う。`src/reproduction/db/` は DuckDB を使う。`src/reproduction/snapshot_reader.py` と `src/reproduction/pipeline/` は NumPy と JSON でスナップショットを読み、DuckDB を import しない。`src/reproduction` は `src/se`、`src/asr`、`src/evaluation`、`src/training` を import しない。`src/evaluation` は `src/reproduction` を import しない。
- `src/experiments/` は `src/training/core/rng.py` の `set_determinism`、`capture_rank_rng`、`restore_rank_rng` だけを使う。`src/se`、`src/asr`、`src/evaluation`、`src/data`、`src/reproduction` は import しない。`src/training/` と `src/reproduction/` は `src/experiments` を import しない。

方針:

- `src/se/` と `src/asr/` に、データ・評価・互いの知識を入れない。
- 評価は SE と ASR の内部実装を参照せず、共通 API 経由で使う。
- 現状の相互依存は広げない。新しい import は、既存の方向に足す。

## 推論と学習の責務分離

- `src.asr` の `transcribe()` は推論専用。モデルは `eval()`、全パラメータは `requires_grad=False`、実行は `torch.inference_mode()`。学習・fine-tuning・勾配を通す経路は無い。
- `src.se` の `load_enhancer()` → `enhance()` は推論専用。`build_model()` は勾配を保つ `nn.Module` を返すだけで、学習ループ・損失・discriminator は含まない。
- 学習は `src/training/` に置く。`transcribe()` と `enhance()` の意味は変えない。評価パイプラインは推論 API だけを呼ぶ。MP-SENet の Generator は `build_model()` のモジュールを学習し、学習済み重みは `generator` キーの checkpoint として `load_enhancer()` に渡せる。
