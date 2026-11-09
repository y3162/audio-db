# Phase 4-A: MP-SENet Generator の最小接続

判定: **GO**。MP-SENet Generator を Experiment Framework の `StandardTrainer` に接続し、CPU / FP32 で Waveform L1 の 1 回の Optimizer 更新が Parameter を変えるところまで確認した。

この GO は公式 MP-SENet の複合損失、GAN 学習、Validation、公式スコアの再現を意味しない。Phase 4 全体（四モデル）は完了ではない。

基準 Commit は `ad10fc1`（`ad10fc17b26eab1d746e81e832f24b2141f6ae22`）。ブランチは `feature/experiment-framework`。測定は CPU / FP32、`deterministic: true`、world size 1、蓄積 1。GPU は使っていない。

## 1. 変更ファイル

- `src/experiments/models/__init__.py`
- `src/experiments/models/mp_senet.py`
- `configs/experiments/mp_senet_l1.yaml`
- `tests/experiments/test_mp_senet_connection.py`
- `docs/training/experiment-framework/PHASE_4A_MP_SENET_CONNECTION.md`
- `docs/training/experiment-framework/ARCHITECTURE.md`
- `docs/training/experiment-framework/PUBLIC_API.md`
- `docs/training/experiment-framework/MIGRATION_PLAN.md`
- `docs/training/experiment-framework/ACCEPTANCE_CRITERIA.md`
- `docs/INDEX.md`
- `.agents/DECISIONS.md`
- `.agents/PROJECT.md`
- `.agents/DEVELOPMENT.md`

`src/training/` の Task と Trainer、`src/se/` の MPNet、Data Pipeline の数値式、checkpoint 形式 `audio-db-experiment-v1` は変えていない。Factory と Builder にモデル名の分岐は足していない。

## 2. Forward 契約

`MPNet.forward` は `(noisy_magnitude, noisy_phase)` を受け、`(enhanced_magnitude, enhanced_phase, complex_spectrum)` を返す。

`SnapshotDataModule` の `Batch.inputs` は `[batch, time]` の波形である。`StandardTrainer` は `model(batch.inputs)` を呼び、`loss(prediction, batch.targets)` を呼ぶ。Trainer に MP-SENet 専用の分岐は無い。

アダプタの `forward` は次だけを行う。

```text
[B, T]
  -> mag_pha_stft
  -> MPNet
  -> mag_pha_istft
  -> [B, T]
```

入力 rank が 2 でないとき、波形が reflect pad に足りないとき、iSTFT 長が入力長と違うとき、出力が非有限のときは `RuntimeFailure`。長さの不一致を crop や pad で隠さない。

入力と教師の sample rate は、固定ペアの rule が両方を同じ `sample_rate` で描く。アダプタは教師波形を受け取らない。構築時の `sampling_rate` が 16000 でないときは `BuildError`。試験は snapshot rule の `sample_rate` とアダプタの値がどちらも 16000 であることを確認した。

## 3. Model Adapter

クラスは `src.experiments.models.mp_senet.MPSENetWaveformAdapter`。`src/se/` に置かない。`src/se` は `src/training` を import できず、STFT 関数は `src.training.se.stft` にあるためである。

`src/experiments/` の他のモジュールは `src/se` を import しない。このファイルだけが `src.se.build_model` と `mag_pha_stft` / `mag_pha_istft` を呼ぶ。`src/se` は `src/experiments` を import しない。

アダプタは Loss、Optimizer、Scheduler、Checkpoint、データ取得を持たない。追加の RMS 正規化もしない。正規化は `FixedPairProtocol` の `noisy_rms_unit_v1` のままである。

## 4. Parameter Identity

`self.generator` が MPNet の唯一のインスタンスである。`parameters()` は Generator の Parameter と同一オブジェクトで、重複は無い。

`PreparedModel.parameters` と AdamW の param group は、その同じオブジェクトを参照する。`parameters_of` が module でない対象や、`trainable: false` の対象を指すときは、構築前に `BuildError` になる。

checkpoint の module state では Generator のキーが `generator.` で始まる。state の所有者はアダプタ 1 つである。再開は新しい Experiment を同じ YAML から組み、その Parameter へ state を載せる。実行中の module 設定を checkpoint の中身で差し替えない。

## 5. STFT と iSTFT

計算は次の既存関数だけを使う。窓や圧縮の式は新規定義していない。

- `src.training.se.stft.mag_pha_stft`
- `src.training.se.stft.mag_pha_istft`

現行 `MPSENetTask._generate` と同じ呼び方である。`center=True` の iSTFT は長さを明示しない。入力長が hop の倍数でないとき、出力長は `(length // hop) * hop` になり、アダプタはそれを拒否する。試験の長さ 64 は hop 8 の倍数で、出力長は 64 のままである。

## 6. 使用した Model Config

公開の `build_model("mp_senet", config)` が受け取る小さい合法構成を使った。ネットワークを別の簡易モデルへ置き換えていない。`dense_channel` は attention head 数 4 で割り切れる。

| 項目 | 値 |
|---|---|
| `dense_channel` | 4 |
| `num_tsconformers` | 1 |
| `n_fft` | 32 |
| `hop_size` | 8 |
| `win_size` | 32 |
| `compress_factor` | 0.3 |
| `beta` | 2.0 |
| `sampling_rate` | 16000 |
| dtype / device | float32 / cpu |

公式の `dense_channel` 64、`num_tsconformers` 4、`n_fft` 400 ではない。CPU の接続試験をこの大きさに限るためである。

`configs/experiments/mp_senet_l1.yaml` もこの構成を宣言する。snapshot は既存の `mp_senet.vb28.ondemand.safe@1` と相対パス `data/db/snapshots` で、存在しないパスは書いていない。受け入れ試験はこの VoiceBank snapshot を実行せず、同じ protocol の合成 snapshot を使う。

## 7. 入力波形と教師波形

合成 snapshot の algorithm は `fixed_pair.mp_senet`、正規化は `noisy_rms_unit_v1`、sample rate は 16000、window は 64、delivery は `on_demand`、profile は `leakage_safe`。各発話は 64 sample で、crop は窓いっぱい、pad は無い。

学習 step の batch は `[1, 64]`。数値比較のテンソルは `[2, 64]`。どちらも float32、CPU。

## 8. Forward 結果

合成 batch からの forward:

- 入力 `[1, 64]`、出力 `[1, 64]`
- 出力は有限
- `grad_fn` がある
- `load_enhancer` と `_MPEnhancer._enhance_one` は呼ばれない

## 9. Loss と Gradient

Loss は `torch.nn.functional.l1_loss`（Factory では `torch.nn.L1Loss`）。この構成の 1 batch で損失は有限だった。

Generator の Parameter は 127 個。この構成と波形長 64、batch 1 では、127 個すべてに勾配があり、すべて有限かつ非 0 だった。勾配が無い Parameter は 0 個である。全 Parameter に勾配があることは、より大きい公式構成では要求しない。この試験の構成では欠けるものは無かった。

## 10. Optimizer 更新

AdamW、`lr` 0.001、`weight_decay` 0.0、betas は PyTorch の既定 `(0.9, 0.999)`。`weight_decay` を 0 にしたので、更新は勾配による。

`StandardTrainer` の 1 logical step:

- `logical_step == 1`
- `optimizer_updates == {"optimizer": 1}`
- 非 0 勾配を持つ 127 個すべてが更新前と異なる
- 各 Parameter の Adam state に有限な `exp_avg` がある
- `metrics.jsonl` に 1 行あり、loss は有限

## 11. Checkpoint

形式は `audio-db-experiment-v1`。1 step で中断したファイルに、次があることを確認した。

- Generator state（キーは `generator.`）
- Optimizer state
- Trainer state（`logical_step` 1）
- Data state（`plan_hash` と `identity.snapshot_hash`）
- `rng_by_rank`（長さ 1）
- `dataset_identity`
- data の `plan_hash`

`max_logical_steps: 2` の連続実行と、1 step で中断して同じディレクトリから再開した実行は、2 step 目の module state と optimizer state が `torch.equal` で一致した。loss 列も一致した。`logical_step` と optimizer update はどちらも 2。

dataset identity を書き換えたファイルと、Generator tensor の形状を壊したファイルは `CheckpointError`。DataModule の identity だけをずらした `load_state_dict` は `StateError`。

## 12. 現行計算経路との数値比較

同じ MPNet 初期重み、同じ波形、同じ STFT で、アダプタの forward と `MPSENetTask._generate` の `audio` を比べた。比較したのは Waveform L1 の経路である。公式複合損失ではない。

| 項目 | 値 |
|---|---|
| 入力形状 | `[2, 64]` |
| 出力形状 | `[2, 64]` |
| モデル Config | 第 6 節の小さい構成 |
| dtype / device | float32 / cpu |
| 波形の最大絶対誤差 | 0 |
| L1 の差 | 0 |

同じ AdamW（`lr` 0.001、`weight_decay` 0）で 1 回更新したあとの Generator 重みの最大絶対誤差も 0 だった。現行 Task の構築時 Optimizer（betas `(0.8, 0.99)`）とは別に、同じハイパーパラメータの AdamW を両方に作って比べた。

未解決の数値差は無い。

## 13. 回帰テスト

変更前と変更後のコマンドは次である。`CUDA_VISIBLE_DEVICES=""`。

```bash
.venvs/se/bin/python -m unittest discover -s tests/experiments -v
.venvs/se/bin/python -m unittest discover -s tests/reproduction -v
```

Phase 3-F 終了時の基準は、experiments 92 件・失敗 0・既存 Skip 1、reproduction 135 件・既存 Error 1・Skip 2。

変更前（この作業ツリー、Commit `ad10fc1`）:

- `tests/experiments`: 92 件、OK、skipped 1、exit 0。ログ `/tmp/phase4a-baseline-experiments.txt`。467.233 秒。
- `tests/reproduction`: 135 件、errors 1、skipped 2、exit 1。ログ `/tmp/phase4a-baseline-reproduction.txt`。455.541 秒。Error は `test_one_deepfilternet_step_is_finite` の `GLIBC_2.33` で、`libdf_rfft.so` がこのマシンの libc を要求する既存の失敗である。

変更後:

- `tests/experiments`: 100 件、OK、skipped 1、exit 0。ログ `/tmp/phase4a-after-experiments.txt`。473.494 秒。増分 8 件は `tests.experiments.test_mp_senet_connection`。新規の失敗は無い。Skip は変更前と同じ `test_cuda_amp` の `NOT_RUN: CUDA is not visible`。
- `tests/reproduction`: 135 件、errors 1、skipped 2、exit 1。ログ `/tmp/phase4a-after-reproduction.txt`。461.211 秒。Error と Skip は変更前と同じ 1 件と 2 件で、GLIBC Error は修正していない。

Phase 4-A の試験 8 件は失敗 0。

## 14. 公式学習再現との差分

今回入っていないもの:

- Metric Discriminator
- PESQ Loss、Magnitude Loss、Phase Loss、Complex Loss、Consistency Loss
- 公式 Generator 複合損失と Discriminator Loss
- GAN Trainer
- ExponentialLR
- 公式の `dense_channel` 64、4 block、`n_fft` 400、2 秒窓
- Validation と公式スコア

Waveform L1 の更新が成功しても、公式学習アルゴリズムの再現完了とは判定しない。

## 15. Phase 4-B への残課題

Phase 4-B は TF-GridNet の単一 Optimizer 型の公式学習アルゴリズムを接続する。

Phase 4-A の残件は次である。

- MP-SENet の公式複合損失と GAN 更新順
- 公式サイズのモデルと VoiceBank 窓 32000 での学習
- Validation
- MossFormerGAN、SEMamba++、DeepFilterNet3 の接続
- 現行 checkpoint 形式 `audio-db-training-v1` からの移行
