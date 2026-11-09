# Phase 4-D: MP-SENet 公式 GAN 学習の移植

Phase 4-A の Waveform L1 アダプタは残す。公式 GAN は別クラス `MPSENetGANTrainer` が持つ。計算式はピン留め済みの現行関数のままである。本格 VoiceBank 学習と論文スコアの再現はしていない。

## 1. 変更ファイル

追加:

- `src/experiments/models/mp_senet_gan.py`
- `src/experiments/mp_senet.py`
- `configs/experiments/mp_senet_gan.yaml`
- `tests/experiments/test_mp_senet_gan.py`
- `tests/experiments/mp_senet_gan_data.py`
- `tests/experiments/mp_senet_gan_worker.py`

変更:

- `src/experiments/runtime.py` に `broadcast_buffers` を足した。world size 1 は何もしない。既存の単一 Optimizer 経路は呼ばない。

触っていないもの:

- `src/training/`、`src/se/` の MPNet と `MetricDiscriminator`、公式 Loss
- `src/experiments/models/mp_senet.py` の L1 アダプタ
- `StandardTrainer`、`TFGridNetTrainer`、`TFLocoformerTrainer`、`GanTrainer`
- Factory と Builder のモデル名分岐
- DB と snapshot の再構築

## 2. G/D の構築と初期化

Generator は `MPSENetGANGenerator`。中身は `src.se.build_model("mp_senet")` の MPNet が 1 組だけである。Discriminator は `MPSENetMetricDiscriminator` で、中身は既存の `MetricDiscriminator` である。新しい畳み込みは作っていない。

現行 `MPSENetTask` は、一つの `seed_model_init` の中で Generator のあと Discriminator を作る。Builder の stage 0 はその順を保たない。各クラスは自分の `seed_model_init` に入る。Discriminator のコンストラクタは、同じ architecture の Generator を一度作り、そのオブジェクトは保持しない。これで Discriminator の乱数は、Generator の初期化の直後から始まる。捨てる Generator は State Registry に入らない。学習 Parameter の所有者は、登録された MPNet と `MetricDiscriminator` だけである。

CPU / FP32、seed 7、`dense_channel` 4、`num_tsconformers` 1、`n_fft` 400、`hop_size` 100、`win_size` 400 で、現行 Task との初期 Parameter と Buffer の最大絶対誤差は Generator も Discriminator も 0 だった。数値更新の比較は、この一致した初期状態から始めた。State のコピーは要らなかった。

Optimizer は別々の AdamW で、Parameter の共有は `BuildError` になる。

## 3. Forward 契約

1 micro-batch の Generator forward は 1 回である。戻り値は公式 `_generate` と同じ順のテンソル列で、Trainer が名前へ詰める。

- Clean / Noisy の magnitude と phase、Clean complex
- Enhanced magnitude、phase、complex
- Enhanced waveform
- その waveform から再計算した magnitude、phase、complex

学習中の iSTFT 長が波形長と違うときは `RuntimeFailure` で、crop も pad もしない。検証だけは現行 `validation_report` が公式 center 長を許す。モデル側で RMS 正規化はしない。固定ペアの `noisy_rms_unit_v1` がデータの側で行う。

DDP の forward は `PreparedModel.module` を通す。Generator の計算グラフは Discriminator の step のあとまで残す。試験は、Discriminator step の直後に enhanced waveform の `grad_fn` が生きていて、`mag_hat` が勾配を要求することを確認した。

## 4. 公式 Loss

次をそのまま呼ぶ。

- `generator_losses`
- `discriminator_loss`
- `metric_targets`
- `mag_pha_stft`
- `mag_pha_istft`

係数は Magnitude 0.9、Phase 0.3、Complex 0.1、Consistency 0.1、Metric 0.05、Time 0.2 である。Complex と Consistency の内部 2 倍は `generator_losses` のままである。Phase の anti-wrapping、group delay、angular frequency は変えていない。Waveform L1 へ置き換えていない。

## 5. PESQ

学習の教師は `metric_targets` である。16 kHz、wideband、`(PESQ - 1) / 3.5`。micro-batch のどれかが失敗すると `None` になり、Discriminator の fake 項だけを省く。Real 項は計算する。G/D の更新全体は skip しない。MossFormerGAN の skip とは違う。失敗を定数の教師で埋めない。

`pesq()` が `ImportError` を上げると、現行 `_pesq_wideband` が -1 を返し、`metric_targets` は `None` になる。この経路を、成功する 1 秒の正弦へパッチして確認した。

検証は shadow の `MPSENetTask.validation_report` を使う。公式損失では失敗を -1 として平均に残す。`library_status` が理由を返すときは `unavailable` で、値は `None` であり、0 でも -1 でもない。研究用の `na` 除外は使わない。

スタブの数値一致と、実 PESQ の 1 更新は別の試験である。長さ 1,600 の乱数波形では実 PESQ がすぐ `None` になる。長さ 16,000 の正弦では実 PESQ がテンソルを返し、その 1 更新も最大絶対誤差 0 だった。

## 6. G/D 更新順

現行 `_official` と同じである。

1. D と G の `zero_grad(set_to_none=True)`。D が先。
2. 各 micro-batch で Generator を 1 回 forward し、グラフを保持する。
3. Clean magnitude と detach した fake magnitude を Discriminator へ入れる。Real のあと Fake。
4. Discriminator loss を蓄積回数で割って backward する。
5. Discriminator の `runtime.step`。
6. Discriminator Parameter の `requires_grad` を外す。例外時も `finally` で戻す。
7. 保持したグラフと、更新後の unwrap した Discriminator で Generator の metric loss を計算する。fake magnitude は detach しない。
8. Generator 複合 loss を蓄積回数で割って backward する。この backward は Discriminator Parameter を更新しない。
9. DDP では Generator 勾配を `runtime.average_gradients` で 1 回だけ平均する。
10. Generator の `runtime.step`。
11. `runtime.end_step` を 1 回呼ぶ。
12. Discriminator を DDP の外で forward したあと、Buffer を rank 0 から配る。Parameter は変えない。

Checkpoint はこの列の途中では書けない。`_in_official_step` のあいだの保存は `RuntimeFailure` である。

`amp_fp16` は拒否する。現行の GradScaler は Optimizer ごとに overflow を見る。D を先に step したあとで G の overflow が分かっても、二つの更新を一つの原子操作には戻せない。FP32 の式は AMP のために変えていない。

## 7. Gradient Accumulation

D loss と G loss は、それぞれ micro-batch 数で割ってから backward する。蓄積が 2 でも Generator は micro-batch ごとに 1 回だけで、再 forward しない。

小さい公式 STFT（`n_fft` 400、`hop_size` 100、`win_size` 400、波形長 1,600、`dense_channel` 4、block 1、batch 1）で、蓄積 1 と 2 の現行 Task との最大絶対誤差は 0 だった。同じプロセスの `ru_maxrss` は、蓄積 1 のあと 641,548 KB、続けて蓄積 2 のあと 656,624 KB だった。これは高水位で、保持したグラフだけの増分ではない。

公式幅（`dense_channel` 64、block 4、波形長 32,000）は動かしていない。メモリ不足は起きていない。式を変えた回避もない。

## 8. DDP 同期

CPU / Gloo、world size 2。

Generator の forward と遅延 backward は `sync_context(..., sync=False)` の中にある。backward の時点で `require_backward_grad_sync` は False だった。そのあと `average_gradients` を logical step あたり 1 回だけ呼ぶ。DDP の自動 all-reduce と二重にはしない。

Discriminator は最後の micro-batch だけ `sync=True` で、その backward も同じスコープにある。Generator 側の metric forward は unwrap したモジュールで、DDP の all-reduce を足さない。

片 rank の非有限入力は、forward の前に `agree(..., policy="all")` で全 rank が落ちる。試験は "rejected its micro-batch" で終了し、ハングしなかった。片 rank だけ PESQ が `None` でも、両 rank は D と G を 1 回ずつ更新し、checkpoint の rank 一致は真だった。PESQ 失敗を全体 skip にはしていない。

進行中の all-reduce の途中で片 rank だけが落ちる場合は、process group の timeout が戻りの上限である。任意の故障でハングしないとは書かない。

更新後の G/D Parameter と Optimizer state は rank 間で一致した。Spectral norm の Buffer は unwrap した metric forward で rank ごとに変わるので、step のあと rank 0 から `broadcast_buffers` する。次の DDP forward が rank 0 の Buffer を配るのと同じ状態を、checkpoint の比較の前に作る。

## 9. Optimizer と Scheduler

G/D とも AdamW、`lr` 0.0005、`betas` `(0.8, 0.99)`、`weight_decay` 0.01。正常な FP32 では 1 logical step で両方が 1 回更新される。`end_step` は logical step の終わりに 1 回である。

Scheduler は両方 `ExponentialLR`、`gamma` 0.99。Optimizer 更新のたびには進めない。その epoch の使える micro-batch を終えたとき、G と D がそれぞれ 1 回進む。

検証間隔は別である。公式忠実の宣言は 5,000 optimizer 更新ごとで、1 epoch は 2,893 更新なので epoch 境界と一致しない。Leakage-safe の現行設定は 2,675 更新で、global batch 4 の 1 epoch と一致する。検証が起きたことだけでは Scheduler を進めない。同じ更新で両方が起きるときは、検証と best 選択のあとで両方の Scheduler を進める。

`experiment_claim` は `official_faithful` か `leakage_safe` である。データの `profile` と `selection_uses_test` が逆のときは `RuntimeFailure` で、二つの条件を一つの実験にしない。

## 10. Validation

意味は現行 `MPSENetTask.validation_report` である。Trainer は重みを shadow の Task へコピーし、その報告を集める。集計は batch 平均の平均で、現行 `_aggregate_validation` と同じである。損失成分、PESQ、STOI、波形長を報告に残す。欠損を 0 にしない。

MP-SENet の検証波形は rank 0 が全件持つ。`fixed_pair.mp_senet` の `validation_epoch` は rank 0 以外が空 iterator を返す。空は正常である。forward は DDP で包んだモジュールではなく `state_module` と shadow である。結果は `gather_objects` で全 rank が同じ要約を持つ。Scheduler と best 判断はそれで揃う。検証の前後で RNG を戻す。学習 cursor は進まない。

## 11. Best 選択

TF-GridNet の `valid.loss` n-best は使わない。現行 `_consider_best` の PESQ 規則である。

- `best_checkpoint_start_epoch` より前の epoch では best を確定しない。宣言は 40。epoch 番号はデータ epoch の 0 始まりで、現行 Trainer が渡す値と同じである。
- 開始以降は PESQ が厳密に高いときだけ置き換える。同点は前の checkpoint を残す。
- `status` が `ok` でない、値が `None`、非有限のときは best を更新しない。`unavailable` を 0 にしない。
- 公式の短い検証で PESQ が -1 のときは `status` が `ok` で、平均に -1 が入る。開始 epoch 以降なら、その -1 が最初の best になり得る。
- 選んだときは rank 0 が `best_generator.pt` と `best_generator.json` を書く。中身は Generator の推論 export である。履歴は Trainer state に残り、再開で一致する。

## 12. Checkpoint

形式は `audio-db-experiment-v1` のままである。別 owner は Generator、Discriminator、Optimizer G/D、Scheduler G/D、Trainer、Data、Runtime である。

保存できるのは、G/D の両 step と `end_step` のあとだけである。D のあと G の前には書かない。

検証か epoch 終了が残っている更新では、Optimizer 更新を state に入れたあと `validation_pending` を書く。再開はその検証、best 選択、必要な Scheduler 更新を一度だけ行い、同じ logical step の `step` で閉じる。完了済みの Optimizer 更新はやり直さない。未完了の G/D 途中は完成した logical step として残さない。壊れたファイルは `CheckpointError` で、再開元にしない。

RNG は既存の checkpoint が Python、NumPy、Torch CPU、CUDA 使用時の生成器を rank ごとに保存する。このフェーズの測定は CPU である。

## 13. Resume

CPU / FP32 で、連続実行と中断再開を比べた。

- epoch 途中: 4 micro-batch の epoch で 1 step 保存し、別の `Experiment.run(resume=...)` で 2 step まで続けた。
- epoch 境界: 2 step で epoch が終わり、検証も走る更新を `validation_pending` で止めた。別プロセスの CLI が検証、best、両 Scheduler を一度だけ完了した。

一致したものは、G/D の Parameter と Buffer、両 Optimizer、両 Scheduler、Trainer、Data、RNG、loss 列、学習率列、検証履歴、best 履歴である。最大差は checkpoint の完全一致で見ている。

CPU / Gloo の world size 2 でも、1 step のあと別プロセス群が 2 step まで再開し、`parameter_ranks_agree` は真だった。

## 14. 新旧の数値比較

条件は CPU / FP32、`deterministic: true`、seed 7、AdamW は上記の公式値。小さい構成は `dense_channel` 4、`num_tsconformers` 1、公式 STFT（400 / 100 / 400）。

| 項目 | 最大絶対誤差 |
|---|---:|
| 初期 Generator | 0 |
| 初期 Discriminator | 0 |
| スタブ PESQ の 1 更新（損失、D/G 勾配、更新後 Parameter、両 AdamW state） | 0 |
| 同じ条件の蓄積 2 | 0 |
| 実 PESQ の 16,000 サンプル正弦、蓄積 1 | 0 |
| 合成 snapshot の `noisy_rms_unit_v1` バッチ 1 更新 | 0 |

最初に差が出た項目は無かった。Loss だけでなく勾配、Parameter、Optimizer state も一致している。

スタブは全行 0.25 の決定的な教師である。実 PESQ の試験はライブラリを呼び、ターゲットが `None` でないことを別に見ている。

## 15. GPU

GPU は使っていない。CPU / Gloo の成功を、GPU / NCCL の同じ同期として扱わない。長時間の公式学習はしていない。

## 16. 回帰

変更前は基準 commit `7779c6d` で、Phase 4-C 終了時の件数と一致した。

| 対象 | 件数 | 結果 |
|---|---:|---|
| `tests/experiments` | 136 | OK、skipped 1 |
| `tests/training` | 297 | FAILED、failures 1、errors 6、skipped 12 |
| `tests/reproduction` | 135 | FAILED、errors 1、skipped 2 |

変更後:

| 対象 | 件数 | 結果 |
|---|---:|---|
| `tests/experiments` | 153 | OK、skipped 1。増分 17 は Phase 4-D |
| `tests/training` | 297 | FAILED、failures 1、errors 6、skipped 12 |
| `tests/reproduction` | 135 | FAILED、errors 1、skipped 2 |

失敗の名前は変更前と同じで、DeepFilterNet3 の `libdf_rfft.so` だけである。新規の Failure、Error、Skip は無い。

## 17. Phase 4-E への残課題

この節を書いた時点では、Phase 4-E の MossFormerGAN は未実装だった。D を先に更新して G のグラフを保持する順序は、MossFormerGAN の 3 forward と PESQ skip へ流用しない。実装後の記録は [PHASE_4E_MOSSFORMERGAN_TRAINING.md](PHASE_4E_MOSSFORMERGAN_TRAINING.md)。

残るものは、MossFormerGAN、DeepFilterNet3、SEMamba++、URGENT24 の TF-GridNet、FSDP、sharded checkpoint、AMP BF16、複数 Optimizer の AMP、`audio-db-training-v1` からの再開、MP-SENet の公式データ全量と論文スコアである。CPU / Gloo は NCCL の証明ではない。
