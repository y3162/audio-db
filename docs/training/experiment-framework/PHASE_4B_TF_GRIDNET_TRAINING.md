# Phase 4-B: TF-GridNet MS-SNSD の学習接続

判定: **GO**。公式 MS-SNSD の単一 Optimizer アルゴリズムを Experiment Framework へ移した。公式データ全量の学習と論文スコアの再現は意味しない。Phase 4 全体は完了ではない。

基準 Commit は `870e66a`（`870e66a96c950b5281e54c8836c7cd19f5cb87dc`、Phase 4-A）。ブランチは `feature/experiment-framework`。数値比較、epoch、validation、n-best、再開、CPU / Gloo の world size 2 は、すべて CPU / FP32、`deterministic: true` である。GPU は使っていない。

## 1. 変更ファイル

- `src/experiments/models/tf_gridnet.py`
- `src/experiments/losses.py`
- `src/experiments/tf_gridnet.py`
- `configs/experiments/tf_gridnet_ms_snsd.yaml`
- `tests/experiments/test_tf_gridnet_training.py`
- `tests/experiments/tf_gridnet_data.py`
- `tests/experiments/tf_gridnet_identity_worker.py`
- `docs/training/experiment-framework/PHASE_4B_TF_GRIDNET_TRAINING.md`
- `docs/training/experiment-framework/PUBLIC_API.md`
- `docs/training/experiment-framework/ARCHITECTURE.md`
- `docs/training/experiment-framework/MIGRATION_PLAN.md`
- `docs/training/experiment-framework/ACCEPTANCE_CRITERIA.md`
- `docs/training/experiment-framework/COMPONENT_LIFECYCLE.md`
- `docs/training/experiment-framework/PHASE_3C_CHUNK.md`
- `docs/INDEX.md`
- `.agents/DECISIONS.md`
- `.agents/PROJECT.md`
- `.agents/DEVELOPMENT.md`

`src/training/`、`src/reproduction/`、`src/se/` のモデル本体、`StandardTrainer`、`CheckpointManager` のファイル名規則は変えていない。Factory と Builder にモデル名の分岐は無い。checkpoint 形式は `audio-db-experiment-v1` のままである。旧 CLI はそのままである。

`configs/experiments/tf_gridnet_ms_snsd.yaml` は公式幅の宣言である。`max_logical_steps` は 1 なので、このファイルを実行しても epoch は終わらない。受け入れは小さい合法構成と合成 snapshot で測った。公式規模の学習は起動していない。

## 2. Model 構築・初期化

クラスは `src.experiments.models.tf_gridnet.TFGridNetModel`。コンストラクタは `seed_model_init(seed)` の中で `src.se.build_model("tf_gridnet", architecture, variant="ms-snsd")` を呼び、その直後に `apply_espnet_initialization(..., "xavier_uniform")` を呼ぶ。これは Builder の stage 0 なので、prepare と Optimizer より前である。初期化乱数は学習乱数へ戻す。

`variant` が `ms-snsd` 以外のときは `BuildError`。`urgent24` は作らない。Generator は `self.generator` の 1 組だけである。`parameters()`、`PreparedModel.parameters`、Adam の param group は同じ Parameter オブジェクトである。CPU の DDP prepare もその同一性を保ち、`state_module` は包む前の module である。

YAML 1.1 は小数点の無い `1e-5` を文字列にする。コンストラクタは `eps` と整数の architecture 項目を数値へ戻してから `build_model` へ渡す。公式宣言は `eps: 1.0e-5` である。

推論用 export は `separator.` を付けた state dict である。学習中に二つ目の Generator は作らない。

## 3. Forward 契約

入力は rank 2 の波形 `[batch, time]`。長さが `n_fft // 2` 以下のとき、出力形状が入力と違うとき、非有限のときは `RuntimeFailure`。crop も pad もしない。中身は `TFGridNet.forward` で、Hann STFT / iSTFT と発話ごとの標準偏差スケールはモデル本体のままである。このスケールは損失の分散正規化ではない。

試験の小さい構成は `n_fft` 16、`stride` 8、`n_layers` 1、`lstm_hidden_units` 4、`attn_n_head` 1、`attn_approx_qk_dim` 8、`emb_dim` 4、`emb_ks` 1、`emb_hs` 1、`activation` prelu、`eps` 1e-5。波形長 32。公式幅（`n_fft` 512、`n_layers` 4、`lstm_hidden_units` 128、`attn_n_head` 4、`emb_dim` 32）は構築できることを確認し、forward はしていない。

## 4. Loss

クラスは `src.experiments.losses.MultiResolutionL1Loss`。`forward(estimate, target)` は既存の `multi_resolution_l1(target, estimate, windows, time_domain_weight=0.5, normalize_variance=False)` を呼ぶ。窓は `(256, 512, 768, 1024)`。矩形 STFT と hop = window/2 はその関数のままである。戻り値は batch 平均である。

`normalize_variance=True` は `BuildError`。SI-SDR は損失に足さない。学習重みは 0 のままである。

## 5. Gradient

Trainer は `StandardTrainer` と同じく、最後以外の micro-batch を `sync=False`、最後を `sync=True` で包み、損失を蓄積回数で割ってから `runtime.backward` する。勾配クリップは `runtime.clip`、上限 5.0。クリップの実装は `clip_grad_norm_` で、現行 `clip_gradients` と同じである。

## 6. Optimizer 更新

Optimizer は Adam、`lr` 0.001、`betas` `(0.9, 0.999)`、`weight_decay` 0。`optimizer.step` は `runtime.step` だけが呼ぶ。`applied` が True のときだけ update count を増やす。logical step は `applied` が False でも 1 進む。

同一 seed、同一 architecture、同一波形で、現行 `TFGridNetTask` と次を比べ、最大絶対誤差は 0 だった。

- 初期重み
- forward
- `mr_l1_tfd` のスカラー
- backward 後の勾配
- クリップ後の勾配
- Adam 更新後の重み
- Adam の `exp_avg`、`exp_avg_sq`、`step`

## 7. Epoch 管理

1 epoch は `plan_chunk_epoch` が返す micro-batch 列を、`SnapshotDataModule.train_epoch` が尽きるまで消費することである。固定の Optimizer 回数ではない。`ChunkSampler` はこれまで通り `espnet_epoch=epoch+1` を計画へ渡す。`group_size` は 1 で、Trainer の `accumulation` と違うときは `RuntimeFailure`。

`max_logical_steps` が epoch の途中で尽きたときは、validation、scheduler、best 選択をしない。`stop_after_logical_steps` が epoch の途中なら `interrupted` で、`complete.json` は書かない。epoch の最後の更新の直後に `stop_after` へ達したときは、`validation_pending` を書いてから戻る。検証はまだ走っていない。

## 8. Scheduler

Scheduler は注入された `ReduceLROnPlateau`（`mode=min`、`factor` 0.7、`patience` 1）である。`StandardTrainer` のように更新のたびには進めない。検証損失が出たあと、`scheduler.step(loss)` を 1 回だけ呼ぶ。

PyTorch 2.6 の初期状態は `last_epoch=0`、`best=inf` である。patience 1 では、改善しない指標の 1 回目と 2 回目では学習率が 0.001 のままになり、3 回目で 0.0007 になる。最初の epoch で学習率が落ちることは要求しない。試験は `last_epoch` と `best` を見る。epoch 途中の 2 step では、記録した学習率がどちらも 0.001 で、pending の `last_epoch` は 0 のままである。閉じた境界では `last_epoch` が 1 で、`best` はその検証損失である。

## 9. Validation

`model.eval()`、`torch.no_grad()`、Optimizer は更新しない。学習 cursor と学習 RNG は検証の前後で同じである。forward は `state_module` で、DDP の forward の中では all-reduce しない。

重みは batch の波形数である。`multi_resolution_l1` の戻り値は batch 平均なので、rank 内では `平均 * 件数` を足す。world size が 2 以上のときは、各 rank の和と件数を `gather_objects` し、全 rank が `合計 / 総件数` を計算する。rank 平均の平均は使わない。

world size 1 の合成検証は件数 2 と 1 で、重み 3。world size 2 は rank 0 が 2 と 1、rank 1 が 2 で、重み 5。`(1*2 + 5*1 + 3*2) / 5 = 2.6` は、rank 平均 `(7/3 + 3) / 2` と違う。この差を試験で確認した。

短い最後の batch は、world size 1 の件数 1 と、合成 snapshot の chunk 計画の端数である。合成 snapshot の 1 epoch は `SnapshotDataModule` と `plan_chunk_epoch` で終わり、scheduler の `last_epoch` は 1 だった。`OnDemandChunkData` は作っていない。

## 10. DDP

CPU / Gloo、world size 2、小さい TF-GridNet、合成 chunk で 1 epoch（2 logical step）を走らせた。ハングはしていない。保存された checkpoint は `parameter_ranks_agree` が真で、optimizer update は 2、検証重みは 5、scheduler の `last_epoch` は 1 である。rank 0 と rank 1 の sample id 列は違う。

prepare の前後で Parameter の `id` は各 rank の中で一致し、Adam もその同じオブジェクトを掴む。`module` は DDP で包まれ、`state_module` は包む前の module である。

## 11. Checkpoint

形式は `audio-db-experiment-v1`。1 logical step につきファイルは 1 つで、`validation_pending` と閉じた `step` は同じ `step_XXXXXXXX.pt` を置き換える。ラベルは payload の中にある。

epoch 末の確定状態は二つだけである。

- `validation_pending`: Optimizer 更新は終わり、検証、scheduler、n-best はまだ。
- 置き換えた `step`: その三つが一度ずつ終わっている。

scheduler だけを先に保存すると、再開が `scheduler.step` を二度呼ぶ。メモリ上の `pending_scheduler` と `pending_selection` は試験用の hook で、別ファイルにはしない。

## 12. N-best

選択指標は `valid.loss` の最小だけである。`valid.si_snr` は書かないし、順位にも使わない。同点は epoch が小さい方を先にする。整列キーは `(loss, epoch)`。

rank 0 が `{espnet_epoch}epoch.pt` を書く。その epoch が最良なら `valid.loss.best.pt` と JSON も書く。`keep_nbest` 件に満たないときは `valid.loss.ave_1best.pt` を最良 epoch への symlink にし、`valid.loss.ave.pt` をその symlink へ向ける。平均ファイルは作らない。件数が揃ったら `average_checkpoint_files` で `valid.loss.ave_{keep}best.pt` を書く。浮動小数は平均、整数は和である。実験の `step_*.pt` は消さない。消すのは n-best から外れた `{digits}epoch.pt` だけである。

early stopping は現行 chunk 規則で、履歴に今の epoch を入れたあと、`espnet_epoch - best_epoch > patience` なら止める。既定の patience は None なので短い試験は止まらない。宣言 YAML は 5。停止理由は `completed` で、Trainer 状態の `early_stopped` が真になる。

## 13. Resume

`validation_pending` の再開は、学習ループの前に境界を一度だけ閉じる。Optimizer 更新はやり直さない。検証は最初からやり直す。途中までの検証 batch は保存しない。

比較した実行は次である。

- 連続して 1 epoch（2 logical step）を終える。
- 1 step で中断し、同じディレクトリから再開して 2 step まで進める。
- epoch 境界の `validation_pending` で中断し、別プロセスの `python -m src.experiments.cli --resume` で境界を閉じる。

連続実行と再開後で、model、optimizer、scheduler、trainer、data、RNG、検証履歴、`1epoch.pt`、`valid.loss.best.pt`、n-best の symlink、`metrics.jsonl` の loss と learning rate が一致した。`stop_after_logical_steps` は config hash に入らないので、上限だけを外した再開は同じアルゴリズムである。

## 14. 新旧数値比較

比較相手は `src.training.se.tf_gridnet.TFGridNetTask`。損失は `official`、初期化は `xavier_uniform`、Optimizer は Adam、scheduler は task 側では `none` である。実験側の 1 step は検証と scheduler まで進むが、それらは重みと Adam の moment を変えない。最大絶対誤差は 0。未解決の差は無い。

これは現行 audio-db 実装との一致であり、ESPnet の長時間ログや論文の WER との一致ではない。

## 15. GPU

使っていない。Factory、初期化、forward、loss、勾配、Adam、scheduler、checkpoint、再開、world size 2 の Gloo は CPU で判定できる。CUDA の損失一致と NCCL は、この位相の完了条件に入っていない。AMP FP16 の数値一致も要求していない。

## 16. 回帰テスト

変更前（`870e66a`、実装ファイルを import する前）:

| スイート | 結果 |
|---|---|
| `tests/experiments` | 100 件、失敗 0、Skip 1、EXIT 0 |
| `tests/training` | 297 件、失敗 1、Error 6、Skip 12、EXIT 1 |
| `tests/reproduction` | 135 件、Error 1、Skip 2、EXIT 1 |

`tests/training` の失敗 1 と Error 6、`tests/reproduction` の Error 1 は、いずれも `libdf_rfft.so` が要求する `GLIBC_2.33` である。今回は直していない。

変更後（同じコマンド、`CUDA_VISIBLE_DEVICES=""`）:

| スイート | 結果 | 変更前との差 |
|---|---|---|
| `tests/experiments` | 115 件、失敗 0、Skip 1、EXIT 0 | 新規 15 件はすべて成功。Skip は Phase 2-C の既存 1 件のまま |
| `tests/training` | 297 件、失敗 1、Error 6、Skip 12、EXIT 1 | 件数、失敗、Error、Skip は変更前と同じ。中身は `GLIBC_2.33` |
| `tests/reproduction` | 135 件、Error 1、Skip 2、EXIT 1 | 件数、Error、Skip は変更前と同じ。中身は `GLIBC_2.33` |

新規の Failure、Error、Skip は無い。既存の GLIBC Error は直していない。

## 17. Phase 4-C への残課題

TF-Locoformer は、この単一 Optimizer の Trainer を参照して、warmup と plateau を後で接続する。warmup は今回実装していない。`StandardTrainer` と `TFGridNetTrainer` の共通ループも、TF-Locoformer が実際に共有するまで切り出さない。

残る未接続は、MP-SENet の公式 GAN、MossFormerGAN、TF-Locoformer、DeepFilterNet3、SEMamba++、URGENT24 の TF-GridNet である。FSDP、sharded checkpoint、AMP BF16、`audio-db-training-v1` からの再開も未実装である。
