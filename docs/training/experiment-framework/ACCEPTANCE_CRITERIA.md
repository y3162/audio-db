# 受け入れ基準

Phase 1、Phase 2-A の CPU checkpoint、Phase 2-B の CPU / Gloo DDP、Phase 2-C の CUDA / NCCL / AMP FP16 の試験は `tests/experiments/` にある。固定ペア Data System の Phase 3-B も同じディレクトリの `test_fixed_pair_data.py` にある。FSDP と AMP BF16 は未実装である。数値は [PHASE_2A_CHECKPOINT.md](PHASE_2A_CHECKPOINT.md)、[PHASE_2B_DDP.md](PHASE_2B_DDP.md)、[PHASE_2C_CUDA_AMP.md](PHASE_2C_CUDA_AMP.md)、[PHASE_3B_FIXED_PAIR.md](PHASE_3B_FIXED_PAIR.md)。音声 Data System の受け入れ表は [DATA_ACCEPTANCE_CRITERIA.md](DATA_ACCEPTANCE_CRITERIA.md)。Phase 3-B が測ったのはその表の固定ペア行である。

既存テストのパスは、この設計作業では実行していない。回帰行は将来の Phase で実行する条件である。

## 読み方

各項目の「期待」は機械的に判定できる条件だけを書く。比較は、同じプロセス条件で二回実行した値、または現行実装と新実装の値である。許容差を書いている行だけが、その差を許す。書いていない数値比較は完全一致である。

## Phase 1

### Factory が任意クラスを作る

- 方法: テスト用の空クラス `tests.experiments.support.Marker` を YAML の `class` に書き、`ComponentFactory.create` を呼ぶ。Factory のソースにそのクラス名が無いことを、テストが Factory ファイルのテキストを読んで確認する。
- 期待: 戻り値の型がそのクラスである。`Marker.created` が 1 である。Factory ソースに `Marker` という文字列は無い。
- Phase: 1

### 明示的な依存が解決される

- 方法: `inject: {loss: {ref: loss}}` を持つ Trainer スタブを構築する。
- 期待: スタブが受け取った loss オブジェクトが、`loss` コンポーネントそのものである（`is`）。
- Phase: 1

### 同一インスタンスの共有

- 方法: 実体 `loss` と別名 `loss_for_log: {ref: loss}` を置き、両方を Trainer スタブの別引数へ inject する。
- 期待: 二つの引数が `is` で真。`Marker` の構築回数が 1。
- Phase: 1

### 未定義参照

- 方法: `inject` が存在しない id `missing` を指す YAML を `build` する。
- 期待: `BuildError`。メッセージに `missing` が含まれる。インスタンスは残らない。
- Phase: 1

### 循環依存

- 方法: `a.inject.other` が `b`、`b.inject.other` が `a`、両方とも同じ Build 段の role である YAML を構築する。
- 期待: `BuildError`。メッセージに `a` と `b` が含まれる。
- Phase: 1

### 引数衝突

- 方法: 同じコンポーネントの `kwargs` と `inject` に `lr` を両方書く。
- 期待: `BuildError`。メッセージに `lr` が含まれる。
- Phase: 1

### 未知キー

- 方法: トップレベルに `train_loop:` を足した YAML を読む。
- 期待: `ConfigError`。学習は始まらない。
- Phase: 1

### Model と Optimizer の接続

- 方法: `torch.nn.Linear` を `parameters_of` で AdamW に渡して `build` する。準備後に、Optimizer のパラメータ id 集合と、ハンドルの `parameters` の id 集合を比較する。
- 期待: 集合が一致する。`prepare.trainable: false` の module を `parameters_of` に使う別 YAML は `BuildError`。
- Phase: 1

### 単一 Optimizer の学習

- 方法: 教師ありの YAML 例で `StandardTrainer` を CPU、`max_logical_steps: 2`、`accumulation: 1`、固定シードで実行する。
- 期待: `RunResult.logical_step == 2`。`optimizer_updates` がその Optimizer id で 2。モデルの重みが初期値と異なる。損失が有限。
- Phase: 1

### 複数 Optimizer の接続

- 方法: GAN の YAML 例で `build` まで行う。`run` はしない。
- 期待: 二つの Optimizer のパラメータ id 集合が交わり空である。Generator 側の集合が generator のパラメータと一致し、Discriminator 側も同様である。
- Phase: 1

### 共有しない Expert

- 方法: 複数 Expert の YAML 例で `build` する。
- 期待: `expert_a` と `expert_b` は `is` が偽。`optim_experts` のパラメータ集合は両方の和集合と一致する。`optim_router` との交差は空である。inject の Expert リスト順は YAML のリスト順と一致する。
- Phase: 1

## Phase 2

### CPU 単一プロセス

- 方法: world size 1、device cpu、`precision: fp32` で Phase 1 の教師あり実験を 4 logical step 実行し、2 step 目の checkpoint から新しいプロセスで `resume` する。
- 期待: 再開後の最初の logical step が 3 である。4 step まで通した重みと、中断せず 4 step 走らせた重みの最大絶対誤差が 0。データが返す `sample_id` の列が両実行で一致する。
- Phase: 2

### 勾配蓄積

- 方法: 同じデータと初期重みで、`accumulation: 1` かつ batch 4 の実行と、`accumulation: 2` かつ micro-batch 2 の実行を、CPU 決定的条件で 1 logical step 進める。`StandardTrainer` は損失を蓄積回数で割る。
- 期待: 両実行の更新後重みの最大絶対誤差が 0。logical step はどちらも 1。Optimizer update count はどちらも 1。
- Phase: 2

### DDP

- 方法: CPU gloo、world size 2、`distribute: ddp`、global に相当する batch が両 rank で 1 micro-batch、`accumulation: 2`。最後の micro-batch だけ `sync=True`。2 logical step 進める。
- 期待: 両 rank の重みの最大絶対誤差が 0。`update_applied` を rank 間で意図的にずらす試験用 Trainer は `RuntimeFailure` で止まり、その後のプロセスがハングしない（タイムアウト 30 秒以内に例外で終了する）。
- Phase: 2
- 記録: `tests/experiments/test_ddp.py` の `test_two_ranks_train_and_match_an_averaged_gradient` と `test_skip_disagreement_fails_without_hanging`。数値は [PHASE_2B_DDP.md](PHASE_2B_DDP.md)。

### AMP

- 方法: CUDA が無い環境、または見える GPU の使用メモリが 1024 MiB 以上のときは実行しない。`precision: amp_fp16` の `Linear` を CUDA で進める。非有限 loss は overflow skip と別の試験にする。有限 loss のまま scale を上げて勾配を溢れさせる試験と、world size 2 で片方の rank だけ勾配を無限大にする試験を分ける。
- 期待: 非有限 loss は backward の前に `RuntimeFailure`。有限 loss の overflow は全 rank で `applied` が False になり、その step では update count も step ごとの Scheduler も進まない。logical step は進む。続く有限 step は `applied` が True に戻る。Scaler の `update` は logical step ごとに 1 回で、Optimizer の個数とは一致しなくてよい。`optimizer.step` が `None` を返しても、呼ばれていれば `applied` は True。
- Phase: 2
- 記録: 2026-11-08 に [PHASE_2C_CUDA_AMP.md](PHASE_2C_CUDA_AMP.md) の C3 と C4 で測った。GPU が無い、または空きが無いときは `NOT_RUN` と書き、成功に数えない。

### Checkpoint の原子性

- 方法: `save` の `os.replace` 直前で例外を投げるテストダブルを使う。
- 期待: 置換前の完成ファイルのバイト列が変わらない。一時ファイルを `restore` しようとすると `CheckpointError`。
- Phase: 2

### 中断再開

- 方法: 4 logical step の実験を 2 step で保存してプロセスを終了し、別プロセスで resume して 4 まで進める。JSONL には step 3 の行を余分に書いてから resume する。
- 期待: 余分な行は切り捨てられる。最終の logical step は 4。State Restore として optimizer state の `step` 相当カウンタが、中断しなかった実行と一致する。
- Phase: 2

### Rank 別 RNG

- 方法: world size 2 の CPU 実験で、各 rank が `torch.rand` を 1 回消費したあとに保存し、両 rank を復元してからもう 1 回 `torch.rand` を取る。対照として、rank 0 の状態を rank 1 にコピーした復元は別テストで拒否されることを確認する。
- 期待: 正規の復元では、各 rank の 2 回目の乱数が、中断しなかった実行の 2 回目と一致する。`rng_by_rank` の長さが 1 のファイルを world size 2 で開くと `CheckpointError`。
- Phase: 2

### データ位置

- 方法: 純関数 Sampler の `TensorPairData` で、update index 1 を消費したあと保存する。先読み用に index 2 を生成して捨てる。resume 後に次に返る index を見る。
- 期待: 次の index は 2 である。捨てた先読みは位置に含まれない。`plan_hash` を書き換えたファイルは `CheckpointError`。
- Phase: 2

### resume と initialize の分離

- 方法: 2 step 学習した checkpoint を `initialize.allow` 無しのまま `--initialize-from` に渡す試験と、`label: init` で module だけを allow したファイルを新しい Optimizer で 1 step 学習する試験。
- 期待: 前者は `CheckpointError`。後者は module 重みが初期化元と一致し、Optimizer の state は空に近く、logical step は 0 から始まる。RNG は元の checkpoint の乱数と一致しない（新しい seed 列である）。
- Phase: 2

### パラメータの rank 間一致

- 方法: 保存直前に rank 1 の重みへ 1 を足す試験ダブル。
- 期待: `save` が `CheckpointError` でファイルを置換しない。
- Phase: 2

## Phase 3

### 固定 ASR の共有

- 方法: ダミー認識器クラスを `prepare.trainable: false` かつ `distribute: none` で構築し、Trainer と metric の両方が `ref` する。`differentiable: true` の YAML も読む。
- 期待: 二つの参照が `is` で真。Optimizer は存在してもそのパラメータを含まない。`requires_grad` はすべて False。`differentiable: true` は `ConfigError`。実 Whisper を使う試験は、チェックポイントが存在する環境だけが別項目として行い、欠けるときは skip と記録する。skip は成功ではない。
- Phase: 3

### 既存データ関数を包む

Phase 3-A で撤回した。代替の受け入れは [DATA_ACCEPTANCE_CRITERIA.md](DATA_ACCEPTANCE_CRITERIA.md)。旧文は次のとおりで、実装の指示には使わない。

- 方法: 合成スナップショットに対する既存 pipeline テストと同じ fixture で、DataModule アダプタの `train_epoch.batch(0)` を、直接 `open_training_data` した batch と比較する。対象は固定ペアのアダプタ 1 種。chunk と online は各 1 種の形状と `sample_id` の一致までを別テストにする。
- 期待: 波形の最大絶対誤差が 0。アダプタのプロセスが DuckDB を import して接続していないことを、開いたファイルの検査か、import 禁止の単体テストで示す。公式プロトコルのソースファイルはこの Phase の差分に含まれない。
- 撤回の理由: 供給クラスは `train_epoch` を持たず、chunk の再開状態は dataset の外にある。比較対象は `open_training_data` のインスタンスではなく、同じ純関数の波形である。

### SE モデルへの接続

- 方法: `src.se.build_model("mp_senet", ...)` を module として構築し、ランダムな短い波形で `reconstruction` 相当の L1 を 1 step、CPU で進める。公式の複合損失はこの項目に含めない。
- 期待: 損失が有限。Generator の少なくとも 1 パラメータが初期値と異なる。`enhance()` が呼ばれていないことを、テストが `enhance` を差し替えて呼び出し回数 0 で確認する。
- Phase: 3 の項として書いてあった。実装は Phase 4 の最初の接続へ移し、Phase 4-A で満たした。測定は [PHASE_4A_MP_SENET_CONNECTION.md](PHASE_4A_MP_SENET_CONNECTION.md)。公式複合損失の 1 update 比較は、下の Phase 4 の項のまま未了である。

## Phase 4

### 既存公式再現モデルとの数値比較

モデルごとに別記録にする。一括の合否にしない。

- 方法: 現行 Task と新 Trainer を、同じ初期化 seed、同じ 1 micro-batch、CPU、`deterministic: true` で 1 optimizer update 進める。MP-SENet は `loss: reconstruction` と公式損失を別記録にする。MossFormerGAN、TF-GridNet、TF-Locoformer は現行の公式損失で 1 update。
- 期待: 各記録に device、seed、ステップ数、最大絶対誤差が書いてある。CPU 決定的条件の重みの最大絶対誤差が 0。0 でない場合は完了にせず、差分の原因を記録する。GPU の結果をこの期待の代替にしない。
- Phase: 4
- 測定: TF-GridNet MS-SNSD は [PHASE_4B_TF_GRIDNET_TRAINING.md](PHASE_4B_TF_GRIDNET_TRAINING.md)。CPU / FP32、seed 0、1 micro-batch、小さい合法構成で、forward、loss、勾配、クリップ後勾配、Adam 更新後の重み、Adam state の最大絶対誤差は 0。TF-Locoformer DNS は [PHASE_4C_TF_LOCOFORMER_TRAINING.md](PHASE_4C_TF_LOCOFORMER_TRAINING.md)。同じ CPU / FP32 で、forward、loss、勾配、クリップ後勾配、AdamW 更新後の重み、AdamW state、warmup 後の 2 更新目の最大絶対誤差は 0。`step_batch` は更新のあと、`step_validation` は epoch 末で、warmup 中も呼ぶ。MP-SENet の Waveform L1 は Phase 4-A。公式 GAN は [PHASE_4D_MP_SENET_GAN.md](PHASE_4D_MP_SENET_GAN.md)。CPU / FP32、seed 7、公式 STFT の小さい Generator で、初期 G/D、スタブ PESQ の 1 更新と蓄積 2、実 PESQ の 16,000 サンプル 1 更新について、損失、勾配、更新後 Parameter、両 AdamW state の最大絶対誤差は 0。MossFormerGAN はこの項のまま未了である。
- 非目標: 公式論文のスコアとの一致。本項目は現行 audio-db 実装との一致である。公式公開実装との一致は、既存の検証記録が既に分けている範囲を超えたいときだけ別実験にする。

### 既存テストの回帰

- 方法: 次を、Phase 4 の差分を入れた作業ツリーで実行する。
  - `.venvs/se/bin/python -m unittest discover -s tests/training -v`
  - `.venvs/se/bin/python -m unittest discover -s tests/reproduction -v`
  - `.venvs/se/bin/python -m unittest discover -s tests/experiments -v`
- 期待: `tests/training` と `tests/reproduction` の失敗数が、移行差分を入れる直前の同じコマンドの失敗数を上回らない。スキップは成功に数えない。`tests/experiments` は Phase 1 から 4 の受け入れテストが失敗 0 である。
- Phase: 4
- 前提: 直前の失敗数が記録されていること。記録が無い状態で「回帰なし」と書かない。

### 再開の数値

- 方法: TF-GridNet の新 Trainer で CPU 決定的に 2 update 進め、1 update 後の checkpoint から再開した 2 update 目の重みを、連続実行と比較する。
- 期待: 最大絶対誤差が 0。logical step と、その単一 Optimizer の update count がどちらも 2。現行形式の checkpoint を新 CLI で resume すると `CheckpointError`。
- Phase: 4
- 測定: [PHASE_4B_TF_GRIDNET_TRAINING.md](PHASE_4B_TF_GRIDNET_TRAINING.md)。epoch 途中の 1 step 再開と、epoch 境界の `validation_pending` を別プロセスから閉じた実行は、連続実行と model、optimizer、scheduler、trainer、data、RNG、loss 列、learning rate 列が一致した。TF-Locoformer も [PHASE_4C_TF_LOCOFORMER_TRAINING.md](PHASE_4C_TF_LOCOFORMER_TRAINING.md) で、epoch 途中と `validation_pending` の別プロセス再開が連続実行と一致した。MP-SENet 公式 GAN も [PHASE_4D_MP_SENET_GAN.md](PHASE_4D_MP_SENET_GAN.md) で、epoch 途中と境界の再開が連続実行と一致し、CPU / Gloo の world size 2 でも短い再開が一致した。再開後の `step_num` と `last_epoch` は連続実行と同じで、境界の検証と plateau は一度だけ走る。`audio-db-training-v1` を新 CLI で読む経路は足していない。

## Phase 5

Phase 5 の個々の研究機能は、機能を足すときに受け入れを追加する。Phase 0 の時点で次だけを固定する。

- 学習手順を YAML の新しい制御構文で書いていないこと。テストは、未知のトップレベルキーが `ConfigError` になる Phase 1 の試験を引き続き実行する。
- 動的 Expert を足す場合、宣言していない state_id の resume が `CheckpointError` になるテストが先に存在する。
- `transcribe()` を backward のグラフに入れたテストは受け入れにしない。勾配を通す ASR は別クラスの単体テストで、`inference_mode` の外にあることを確認する。

## この Phase 0 で実施していないこと

- 上記テストの実装と実行。
- GPU ジョブ、学習の 1 step、既存 `unittest` の実行。
- 公式再現の数値比較。
