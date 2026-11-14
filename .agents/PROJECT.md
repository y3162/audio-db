# プロジェクトと責務

固定した ASR に SE の出力を通し、WER で比べる。PESQ / STOI / SI-SDR は clean 参照があるときだけの補助指標。ASR の重みは更新しない。

## モジュール

| 場所 | 責務 | 入口 |
|---|---|---|
| `src/se/` | SE の推論モデル。MP-SENet、TF-GridNet、TF-Locoformer、DeepFilterNet3、MossFormerGAN、SEMamba++ | `load_enhancer()`、`build_model()` |
| `src/asr/` | ASR の推論。Parakeet、Whisper、Wav2Vec2、HuBERT、WavLM | `load_recognizer()` → `transcribe()` |
| `src/evaluation/` | 名前付き評価セット → 必要なら波形生成 → SE → ASR → 指標 | `python -m src.evaluation.cli` |
| `src/data/` | コーパス走査、レシピ、ステージング、評価用 DuckDB | `make db-init`、`make db-ingest` |
| `src/experiments/` | 唯一の学習基盤。YAML から組み立て、`audio-db-experiment-v1` を保存して再開する | `python -m src.experiments.launcher`（起動と監視）、`python -m src.experiments.cli`（学習プロセス、preflight） |
| `src/experiments/core/` | コンポーネントの生成、依存の解決、状態の登録 | `builder` が使う |
| `src/experiments/execution/` | CPU/GPU/DDP、checkpoint（Latest / Archive）、Selector、Lifecycle、StepJournal、再開、中断、来歴、実行状態、プロファイリング | `runtime`、`checkpoint`、`lifecycle` |
| `src/experiments/training/` | 共通の Trainer、Loss Term、加重 Objective、意味が一致した終了・窓・停止。モデル固有の公式 Trainer は持たない | YAML の `class` |
| `src/experiments/stopping.py` | 指標イベント、独立した停止判定、Trainer と rank 間の制御 | YAML の `stopping` component |
| `src/experiments/metrics/` | 研究用の指標。Loss、SI-SDR、PESQ、STOI、ESTOI、WER。計算だけを行い、停止も保存もしない | YAML の `metric` |
| `src/experiments/evaluation/` | 研究用の波形評価。学習カーソル、重み、Optimizer を動かさない | YAML の `evaluator` |
| `src/experiments/selection/` | 研究用の best 判定。推論重みの書き出しは判定の外 | YAML の `other` と `state.save: true` |
| `src/experiments/data/` | 再現用 DB、スナップショット、サンプリング、オンデマンド描画。学習ループは持たない | `python -m src.experiments.data.cli` |
| `src/experiments/official/` | 6 モデルの公式 Model、Trainer、参照ステップ、モデル固有の検証統計と選択。他の実験から個別に import する | YAML の `class` |
| `src/experiments/calc/` | 手法を共有する STFT、スペクトル損失、scheduler、RNG。CLI ではない | テストと Trainer が呼ぶ |

`api.py`、`cli.py`、`launcher.py`、`config.py`、`errors.py` は `src/experiments/` 直下の入口である。`launcher.py` は学習プロセスを起動して終了状態を記録するだけで、ジョブの投入、GPU の選択、再試行はしない。`builder.py`、`factory.py`、`registry.py`、`state.py`、`runtime.py`、`checkpoint.py`、`lifecycle.py`、`selectors.py`、`provenance.py`、`signals.py`、`profile.py`、`results.py`、`trainers.py`、`losses.py` は公開 import パスで、実装は上の3ディレクトリにある。Experiment は build し、Lifecycle を注入し、クラス自身の `fit()` を一度呼び、資源を解放する。信号、`RUNNING`、再開と終了の判定は `Lifecycle.start` と `finish` が持ち、Trainer の `fit` がそれらを一度ずつ呼ぶ。直接比較は `run_direct` を使う。学習順は `fit` にある。YAML の `class` はこの公開パスのままである。公開モジュールは実装と同じオブジェクトを返す。クラスの identity は同じである。公開モジュール上の関数名を差し替えても、実装モジュールのグローバルは変わらない。

YAML から参照する公式モデル・Trainer の `official/<model>/model(s).py`・`trainer.py` と共有 scheduler の `calc/optim.py` は直接の公開モジュールである。公開名と checkpoint の識別規則は `docs/experiments/CHECKPOINT.md` を参照する。

`src/training/` と `configs/training/` は無い。旧形式 `audio-db-training-v1` の checkpoint は再開しない。研究用の実装を足すときは、公式再現の Model、Trainer、参照ステップを変えず、独立した領域に置く。

## 推論と学習

- `transcribe()` は `eval()`、全パラメータ `requires_grad=False`、`torch.inference_mode()`。
- `enhance()` は推論専用。`build_model()` は学習可能な `nn.Module` を返すだけで、損失も更新順も含まない。
- 学習は `enhance()` と `transcribe()` を呼ばない。
- `src/se/` と `src/asr/` は、他の `src/` を import しない。
- `src/evaluation/` は `src/se`、`src/asr`、`src/data` を使う。再現用 DB パッケージは import しない。
- テスト（`tests/`）は `data/` の下に書かない。本番データを作る処理は `src/` と Make ターゲットに置く。
- `src/experiments/data/` の DB と準備は `src/data/recipe` を使う。`src/se`、`src/asr`、`src/evaluation`、Trainer は import しない。
- 学習ランタイムは `src/se.build_model` と `src/experiments/data/` の描画関数を使う。学習中に DuckDB を開かない。
