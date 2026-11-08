# 移行計画

Phase 1 は `src/experiments/` を並行パッケージとして追加した。現行の公式再現コードは移していない。この文書は、後続 Phase が現行の公式再現を壊さずに新基盤へ移す順序である。

公式再現の数値的同等性は未検証のまま保証しない。Phase 4 の比較が記録されるまで、新経路の結果を現行経路の再現結果と呼ばない。

## 再利用、アダプタ、置換

| 現行 | 方針 | 理由 |
|---|---|---|
| `src.se.build_model` と各モデルクラス | 再利用 | 推論実装と学習 Generator を二重に持たない。`enhance()` は学習から呼ばない現行契約を維持する |
| `src/training/se/` の損失関数、STFT、Discriminator モジュール | 再利用 | 数値の本体である。Trainer から関数として呼ぶ。Task クラスの継承はしない |
| `src/training/se/optim.py` の `WarmupReduceLROnPlateau` | 再利用 | 公式の更新式がここにある。新 Scheduler として `class:` から直接指定できる |
| `src/training/core/trainer.py` の `Trainer` | 置換対象にしない。新経路は別クラス | 三つのループと Task 契約が研究用 API と衝突する。Phase 4 まで現行ループが公式再現の実行系である |
| `accumulation_scope` の仕様 | アダプタではなく Runtime が同じ契約を実装 | forward 前に `no_sync` を決める、というコメントが契約である |
| `src/training/core/rng.py` | 再利用 | rank 別 RNG の capture と restore は試験済みの実装である。新 Runtime がこの関数を呼ぶ |
| `src/training/core/checkpoint.py` の `save_atomic` の手順 | 手順を再利用してよい。形式文字列は共有しない | 原子的置換は同じで、payload の schema は別である |
| `src/training/core/distributed.py` | 手順を Runtime が再実装してよい。現行モジュールは残す | 新 API の `PreparePolicy` を現行関数に押し込むと、現行呼び出し側が変わる |
| `src/reproduction/pipeline/` 一式 | アダプタ | DataModule が `open_training_data`、`open_chunk_datasets`、online クラスを呼ぶ。式は移さない |
| `src/reproduction/protocols/` | 再利用。分割しない | オンライン混合の順序が仕様である |
| `RecipeDataset` と `epoch_indices` | アダプタ | legacy 経路の crop と順序を変えない。新研究の既定データ経路にはしない |
| `load_training_config` と `configs/training/**` | 残す | 公式再現 YAML は現行スキーマのまま実行できる必要がある |
| `src.asr.load_recognizer` | アダプタ | 勾配を通さない FrozenRecognizer だけが呼ぶ。`transcribe()` の意味は変えない |
| `data_provenance.json` と評価側の読取 | 再利用 | 評価は `src.reproduction` を import しない現行契約のまま、JSON だけを読む |
| ジョブ連鎖 `src/training/se/job_chain.py` | Phase 4 まで現行のまま | 新経路の長時間実行仕様が無い。30 分以上の投入条件は現行ルールのまま |

## 既存四モデルへの影響

Phase 0 から Phase 3 まで、四モデルの実行系は `python -m src.training.se.cli` のままである。設定ファイル、checkpoint 形式、DataModule アダプタの未接続、の三点により、新パッケージを import しても現行の 1 step の数値は変わらない。変えてはいけない。

Phase 4 で各モデルに新 Trainer を足すときの固定要件は [ARCHITECTURE.md](ARCHITECTURE.md) の調査節を移したものであり、緩和しない。

| モデル | 新 Trainer が守る現行要件 | 移行しないもの |
|---|---|---|
| MP-SENet | 公式損失の遅延 backward、Generator 勾配の明示平均、ExponentialLR は epoch の usable micro-batch 完了時だけ、検証 PESQ の -1 | `enhance()` への切替 |
| MossFormerGAN | G1/D4 の初期化順、PESQ 欠測で Optimizer を進めない、全 rank 同時 skip、増分平均、検証中 Discriminator は train、plateau の state | 公式 2 GPU の重み割れの再現 |
| TF-GridNet | `mr_l1_tfd`、RMS を掛けない公式モード、`espnet_chunk_v1`、最短 rank で停止、n-best は valid loss | chunk 計画の再実装 |
| TF-Locoformer | 時間重み 0.5、clip 5、warmup は step の後、最初の step は構築時学習率、研究用 si_snr は dB の最大 | `cv_synthetic_small` という名前の検証集合。現行決定で未解決のままである |

DeepFilterNet3 と SEMamba++ は、この四モデルの移行が数値比較を終えるまで新 Trainer を作らない。データアダプタの接口だけ Phase 3 で用意してよい。本学習の開始条件は現行の P10 系決定に従い、この移行計画は本学習の承認にならない。

## 並行運用

二つの入口を同時に残す。

```text
python -m src.training.se.cli --config configs/training/reproduction/...
python -m src.experiments.cli --config configs/experiments/...
```

`configs/experiments/` は Phase 1 で新しい YAML を置く場所である。Phase 0 ではディレクトリを作らない。

同じ `output.directory` を両方から使うことは、lock がファイルシステムロックなので拒否される。実験ディレクトリは経路ごとに分ける。

現行 checkpoint を新 CLI の `--resume` に渡すと、形式不一致で失敗する。これは不具合ではない。

回帰の間、公式再現の正本の数値は現行入口の実行結果だけである。

## 回帰検証

Phase 4 で各モデルについて、次を別々に記録する。

1. 同じ seed、同じスナップショット、同じ world size、CPU、`deterministic: true` で、現行 Task と新 Trainer の 1 optimizer update 後の重みの最大絶対誤差。
2. その条件で損失スカラーの最大絶対誤差。
3. GPU で行う場合は、決定的アルゴリズムの可否を記録し、非決定的カーネルが残るモデルはビット一致を主張しない。MossFormerGAN の `adaptive_max_pool2d` がこの例である。
4. 中断と再開のあと、もう 1 update した重みが、中断しなかった実行と CPU 決定的条件で一致すること。
5. 現行 `tests/training` と `tests/reproduction` の既存テストが、変更前と同じ合否であること。新テストの追加は許される。既存テストの期待値変更は、この移行の範囲では許されない。

比較の環境、device、seed、ステップ数を記録しない結果は、検証済みと書かない。Mock の成功は 1 から 4 の代替にならない。

## Phase の実装順序

依存のため、順序は次で固定する。巨大な骨格を先に作らない。

| Phase | 実装するもの | 完了条件 |
|---|---|---|
| 0 | このディレクトリの設計書 | 実装が無い。本 Phase は完了扱い |
| 1 | Factory、Builder、`Experiment`、CPU の `ProcessRuntime`（DDP も AMP も未実装で、`distribute: none` と `fp32` だけ）、`TensorPairData`、`StandardTrainer`、CLI の `--preflight` と短い `--config` 実行 | [ACCEPTANCE_CRITERIA.md](ACCEPTANCE_CRITERIA.md) の Phase 1 項目が単体テストで満たされる。スナップショットも GPU も使わない |
| 2 | DDP、AMP、StateRegistry、`audio-db-experiment-v1`、rank 別 RNG、原子的保存、resume | Phase 2 の受け入れ。まだ SE モデルを接続しない |
| 3 | DataModule アダプタ。固定ペア、chunk、オンライン混合を既存関数のまま包む。`src.se.build_model` で MP-SENet の Generator を 1 step 進める | 公式損失の数値一致はまだ要求しない。既存 pipeline テストが引き続き通る。アダプタが DuckDB を開かない |
| 4 | 四モデルの Trainer。現行入口と並走させ、上の回帰検証を記録する | 記録が `docs/training/experiment-framework/` の検証文書に残り、未測定を一致と書いていない |
| 5 | LoRA、難例 Sampler、複数 Expert の動的枠、Distillation、勾配を通す ASR が別クラスとして必要ならその設計の実装 | 各拡張が YAML の DSL ではなく Trainer クラスである。公式再現の入口が変わっていない |

Phase 1 のゴールは `torch.nn.Linear` の数 step である。GAN の `run`、実 ASR、SE の重みは Phase 1 の完了条件に入れない。それらの YAML は Builder の構築試験にだけ使う。

順序を前倒ししないもの:

- Phase 1 で checkpoint 形式を確定実装すること。仕様は Phase 0 で固定したが、保存の実装は Phase 2 である。Phase 1 の `StandardTrainer` はメモリ上の step だけで、ファイルへ resume しなくてよい。
- Phase 3 で四モデルの公式損失を移植し切ること。接続は 1 モデルの forward と backward が有限値であることまで。

## 各 Phase の完了条件の見方

完了は受け入れ表の該当行が、テスト名と結果の記録を持つことである。設計書が存在することは Phase 0 以外の完了条件にならない。

Phase 4 の完了は、四モデルすべてについて回帰記録があることである。一モデルだけの記録では Phase 4 を完了にしない。記録の中で一致しなかった項目は、完了ではなく差分として残す。差分が既知の決定（MossFormerGAN の 2 GPU 平均）なら、その決定番号を記録に書く。

## ロールバック

新パッケージは現行入口から import されない。ロールバックは次で足りる。

- `src/experiments/` と `configs/experiments/` と新テストを、その Phase の変更から外す。
- 現行 `src/training/`、`src/reproduction/`、`configs/training/` は、その Phase が制約を守っていれば差分が無い。
- 新形式 checkpoint は現行ローダが読まないので、現行実験ディレクトリに混ざっていても現行 `--resume` は形式エラーで止まる。削除は利用者の明示が無い限り行わない。`data/checkpoints/` の配布重みとは別ディレクトリに書く。

Phase 4 で現行 Task の数値を変える修正が必要になった場合、その修正は新基盤の作業ではなく現行のバグ修正として別にレビューする。移行の差分に混ぜない。

データファイル、`resource.duckdb`、`reproduction.duckdb`、スナップショット、`data/derived/` は、どの Phase でもこの移行のために削除も再作成もしない。

## 公式再現への影響

Phase 0 の成果物は文書だけである。実行中の公式再現ジョブ、既存 YAML、既存 checkpoint の解釈は変わらない。

後続 Phase でも、現行 CLI の既定動作を新 CLI へ切り替える変更を、回帰記録の前に入れない。切り替えの判断は Phase 4 の記録を見たあとの別決定である。この文書はその切り替えを採用しない。
