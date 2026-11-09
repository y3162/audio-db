# Phase 4-C: TF-Locoformer DNS の学習接続

判定: **GO**。TF-Locoformer の DNS 向け学習アルゴリズムを Experiment Framework へ移した。研究用 validation profile での選択規則までを含む。公式 DNS 2020 全量の学習と論文スコアの再現は意味しない。Phase 4 全体は完了ではない。

基準 Commit は `2f701a2`（`2f701a271aa6eba621a664aa29335cd082f5657d`、Phase 4-B）。ブランチは `feature/experiment-framework`。数値比較、warmup、plateau、epoch、validation、n-best、再開、CPU / Gloo の world size 2 は、すべて CPU / FP32、`deterministic: true` である。GPU は使っていない。

## 1. 変更ファイル

- `src/experiments/models/tf_locoformer.py`
- `src/experiments/tf_locoformer.py`
- `configs/experiments/tf_locoformer_dns.yaml`
- `tests/experiments/test_tf_locoformer_training.py`
- `tests/experiments/tf_locoformer_identity_worker.py`
- `docs/training/experiment-framework/PHASE_4C_TF_LOCOFORMER_TRAINING.md`
- `docs/training/experiment-framework/PUBLIC_API.md`
- `docs/training/experiment-framework/ARCHITECTURE.md`
- `docs/training/experiment-framework/MIGRATION_PLAN.md`
- `docs/training/experiment-framework/ACCEPTANCE_CRITERIA.md`
- `docs/training/experiment-framework/COMPONENT_LIFECYCLE.md`
- `docs/training/experiment-framework/PHASE_3C_CHUNK.md`
- `docs/training/experiment-framework/PHASE_4B_TF_GRIDNET_TRAINING.md`
- `docs/INDEX.md`
- `.agents/DECISIONS.md`
- `.agents/PROJECT.md`
- `.agents/DEVELOPMENT.md`

`src/training/` の Task と Trainer、`src/reproduction/` の数値式、`src/se/` のモデル本体は変えていない。`MultiResolutionL1Loss` と `TFGridNetTrainer` の式も変えていない。Factory と Builder にモデル名の分岐は無い。YAML に学習アルゴリズムの DSL は無い。checkpoint 形式は `audio-db-experiment-v1` のままである。

`configs/experiments/tf_locoformer_dns.yaml` は研究用 DNS レシピの宣言である。`max_logical_steps` は 1 なので、このファイルを実行しても epoch は終わらない。`global_batch` 4 と `micro_batch` 1 は研究用の world size 4 に対応する。受け入れは小さい合法構成と合成 snapshot で測った。公式規模の DNS 学習は起動していない。

## 2. Model 構築

クラスは `src.experiments.models.tf_locoformer.TFLocoformerModel`。コンストラクタは `seed_model_init(seed)` の中で `src.se.build_model("tf_locoformer", architecture, variant="dns")` を呼び、その直後に `apply_espnet_initialization(..., "xavier_uniform")` を呼ぶ。これは Optimizer を作る前である。Generator は `self.generator` の 1 組だけである。`parameters()`、`PreparedModel.parameters`、AdamW の param group は同じ Parameter オブジェクトである。CPU の DDP prepare もその同一性を保ち、`state_module` は包む前の module である。

公式幅は `n_fft` 256、`hop_length` 128、`normalize_variance` true、`n_layers` 6、`emb_dim` 128、`attention_dim` 128、`ffn_hidden_dim` `[384, 384]`、`eps` `1.0e-5` である。モデル内部の Hann STFT、`torch.std` による分散正規化、separator、波形長の復元は `src/se/models/tf_locoformer/model.py` のままである。無音と、分散が `1e-6` 未満の入力は、既存モデルと同じく拒否する。`normalize_variance` を false にした構成だけは、その拒否を外した既存の分岐に従う。

YAML 1.1 の `1e-5` は文字列になる。コンストラクタは `eps` と整数の architecture 項目を数値へ戻す。`variant` が `dns` 以外のときは `BuildError`。

## 3. 数値一致

比較相手は現行 `TFLocoformerTask` である。CPU / FP32、seed 0、小さい合法構成（`n_fft` 16、`hop_length` 8、`n_layers` 1、`emb_dim` 4、`attention_dim` 4、`num_groups` 4、`ffn_hidden_dim` `[8, 8]`）、1 micro-batch、AdamW（`lr` 0.001、`betas` `(0.9, 0.999)`、`weight_decay` 0.01）、勾配クリップ 5。

forward、`mr_l1_tfd`、勾配、クリップ後の勾配、AdamW 更新後の重み、AdamW state の最大絶対誤差は 0 である。2 更新目は warmup 後の学習率を使う。その重みと AdamW state の最大絶対誤差も 0 である。学習 loss に SI-SDR は足していない。ログの `si_sdr` は `si_sdr_loss` の値で、現行 Task の戻りと一致する。

## 4. Warmup

Scheduler は既存の `src.training.se.optim.WarmupReduceLROnPlateau` である。別実装は作っていない。

1 回の Optimizer 更新の順序は、forward、loss、backward、clip、`runtime.step`、`applied` が true のときだけ `step_batch`、その後に checkpoint 可能な境界、である。最初の更新が使う学習率はコンストラクタの 0.001 である。`step_batch` はその次の更新向けに `initial_lr * (warmup_steps ** -1) * step_num` を書く。

Scheduler 単体で、モデルを 4000 回学習せずに境界を確認した。`step_num` 1 の学習率は `0.001 / 4000`、3999 は `0.001 * 3999 / 4000`、4000 は `0.001`、4001 は `0.001` のままである。warmup を過ぎた `step_batch` は学習率を変えない。

`runtime.step` が `applied=False` のとき、`step_num` は進まない。この確認は CPU 上で `StepResult(applied=False)` を返した。実 GPU の AMP overflow は走らせていない。logical step と、epoch 末の `step_validation` は進む。

## 5. Plateau

Epoch 末の validation loss のあと、`step_validation` を 1 回呼ぶ。warmup 中でも呼ぶ。設定は mode `min`、factor 0.5、patience 3、threshold `1e-4`、threshold mode `rel`、cooldown 0、min lr 0、eps `1e-8` である。`step_validation` は `step_num` を変えない。

patience 3 では、最良値のあと悪い値が 4 回続いたときに学習率を factor 倍する。`num_bad_epochs` が patience を超えたときである。warmup 中に plateau が学習率を半分にしても、次の `step_batch` が warmup 式で上書きする。この上書きを Scheduler 単体で確認した。改善した validation は `best` を更新して `num_bad_epochs` を 0 にする。相対 threshold の内側は改善とみなさない。

`min_lr` 0.0008 と `cooldown` 1 でも、既存クラスが下限で止め、次の validation で cooldown を 1 減らして `num_bad_epochs` を 0 に戻すことを確認した。宣言 YAML の cooldown と min lr は既存の既定 0 である。

## 6. Epoch 管理

1 epoch は `plan_chunk_epoch` が `SnapshotDataModule.train_epoch` に出した micro-batch 列である。`espnet_epoch = trainer_epoch + 1`。`TFGridNetTrainer` から再利用したのは `rank_validation_loss` と `early_stop` だけである。epoch ループは共有クラスへ切り出していない。TF-GridNet の `ReduceLROnPlateau` は、Optimizer 更新のたびには進まないままである。

epoch 途中の `max_logical_steps` と `stop_after_logical_steps` は、validation、`step_validation`、n-best を走らせない。epoch の最後の更新では、`step_batch` のあとに label `validation_pending` の checkpoint を書く。その後に validation、`step_validation`、best / n-best、同じ logical step の label `step` への置換、である。

## 7. Validation

Validation は epoch 末だけである。`model.eval()` と `torch.no_grad()` で、forward は `state_module` である。学習 cursor と学習 RNG は進まない。

Loss は batch 平均に波形数を掛けた和を、全 rank の件数で割る。SI-SNR はバッチ内のサンプルごとの SI-SDR（dB）の和を、同じ件数で割る。world size 1 の合成 batch サイズ 2 と 1 では、重み 3 の平均は rank 内の batch 平均の単純平均と一致しなかった。world size 2 の重みは 5（rank 0 が 2+1、rank 1 が 2）である。

## 8. SI-SNR

学習ログの `si_sdr` は `si_sdr_loss` で、符号は負の SI-SDR である。Validation の `si_snr` はサンプルごとの SI-SDR（dB）で、その平均は `si_sdr_loss` の符号を反転したものと最大絶対誤差 0 で一致する。同じ波形でなければ、この二つを同じ数として比較しない。

選択に使う `si_snr` は高いほど良い。DNS レシピの `best_model_criterion` が名前として指す `si_snr` と、loss が stats に書く `si_snr_loss` は、この研究用の最大化と同一視しない。今回の選択は研究用 ranker に合わせた。

## 9. Best / N-best

`keep_nbest` は 5 である。`valid.loss` は最小、同点は早い epoch を優先する。`valid.si_snr` は最大、同点は早い epoch を優先する。epoch ファイルは両方の n-best の和集合を残す。Loss の平均は `average_state_dicts` で、浮動小数は平均、整数は和である。件数が `keep_nbest` 未満のときは `valid.loss.ave_1best.pt` を最良 epoch への symlink にする。SI-SNR の平均ファイルは現行 Trainer と同じく書かない。

JSON には `validation_profile` を書く。宣言 YAML の profile は `tf_locoformer.dns.accept_recipe.research_valid_fixed@1` である。これは研究用の 1,024 件であり、`cv_synthetic_small` ではない。受け入れ試験の合成データは profile 文字列 `scripted` を書く。この研究用 snapshot での長時間学習は起動していない。

early stopping は `valid.loss` の最良 epoch だけを見る。`si_snr` が良くても loss が悪ければ、既存の `early_stop` は止まる。宣言 YAML の patience は 10 である。

## 10. Checkpoint

学習 checkpoint は `audio-db-experiment-v1` である。推論用 export とは別ファイルである。epoch 末の `validation_pending` は、その epoch の `step_batch` が終わった状態を持つ。`step_num` は進んでおり、`last_epoch` はまだ 0 である。閉じた `step` ファイルでは `last_epoch` が 1 で、`step_num` は pending のときと同じである。

`WarmupReduceLROnPlateau.state_dict` は `optimizer` 以外の `__dict__` を写す。Builder が付けた `_experiment_component_id` もその辞書に入る。現行 Task の scheduler にはこの属性が無い。warmup の `step_num`、`best`、`num_bad_epochs`、`last_epoch`、学習率の比較では、この属性を除いた最大絶対誤差は 0 である。実験経路の再開は、両側に同じ属性がある checkpoint 同士を比較し、一致した。

## 11. Resume

epoch 途中の 1 step で止め、別の `Experiment.run(resume=...)` で 2 step まで進めた結果は、連続 2 step と checkpoint の state、RNG、loss 列、使った学習率の列、scheduled 学習率の列が一致した。2 step 目が使う学習率は、1 step 目の `step_batch` が書いた `0.001 / 4` である。試験の warmup step 数は 4 で、公式の 4000 は Scheduler 単体で確認した。

epoch 境界は `stop_after_logical_steps` で `validation_pending` のまま止め、別プロセスの CLI が validation、plateau、選択を 1 回だけ実行した。連続実行と、model、optimizer、scheduler、trainer、data、RNG、metrics、`1epoch.pt`、`valid.loss.best.pt`、`valid.si_snr.best.pt` が一致した。`step_num` は 2 のまま、`last_epoch` は 0 から 1 になる。選択の途中で例外を起こした場合も、残るファイルは `validation_pending` で、同じ別プロセス再開が連続実行と一致した。validation と plateau の二重実行は起きていない。

## 12. DDP

CPU / Gloo、world size 2、小さい TF-Locoformer、合成 chunk で 1 epoch（2 logical step）を走らせた。保存された checkpoint は `parameter_ranks_agree` が真で、optimizer update は 2、検証重みは 5、`step_num` は 2、`last_epoch` は 1 である。rank 0 と rank 1 の sample id 列は違う。Parameter identity は DDP の前後で同じオブジェクトのままである。NCCL は使っていない。

## 13. 推論用 Export

`export_for_inference` は Generator の `state_dict` である。`TFLocoformer` のパラメータは `separator` サブモジュールの下にあるので、キーはすでに `separator.` で始まる。これは現行 `TFLocoformerTask.export_for_inference` と同じ対象で、モデル全体の state であり、中身は separator だけである。

`src.se.models.tf_locoformer.inference.load` は、全キーが `separator.` で始まる辞書から接頭辞を外して `separator.load_state_dict(..., strict=True)` する。接頭辞の無い辞書はそのまま `separator` へ載る。両方を同じ小さいモデルで読み、separator の state の最大絶対誤差は 0 だった。学習 checkpoint の payload はこの export ではない。

## 14. GPU 使用状況

GPU は使っていない。比較と試験は CPU / FP32 である。`CUDA_VISIBLE_DEVICES=""` で unittest を実行した。実 GPU の AMP overflow は未測定である。`applied=False` のとき warmup を進めないことは、CPU 上の `StepResult` で確認した。

## 15. 回帰

変更前、`CUDA_VISIBLE_DEVICES=""`。

| スイート | 実行 | 結果 |
|---|---|---|
| `tests/experiments` | 115 | OK、skipped 1（Phase 2-C は CUDA 非表示で NOT_RUN） |
| `tests/training` | 297 | FAILED、failures 1、errors 6、skipped 12 |
| `tests/reproduction` | 135 | FAILED、errors 1、skipped 2 |

`tests/training` の failure 1 と error 6、`tests/reproduction` の error 1 は、いずれも DeepFilterNet3 の `libdf_rfft.so` がこの機械で読めない既存の失敗である。今回の差分では直していない。

変更後、同じ `CUDA_VISIBLE_DEVICES=""`。

| スイート | 実行 | 結果 |
|---|---|---|
| `tests/experiments` | 136 | OK、skipped 1（Phase 2-C は CUDA 非表示で NOT_RUN）。増分 21 は Phase 4-C |
| `tests/training` | 297 | FAILED、failures 1、errors 6、skipped 12 |
| `tests/reproduction` | 135 | FAILED、errors 1、skipped 2 |

失敗の名前は変更前と同じで、DeepFilterNet3 の `libdf_rfft.so` だけである。新規の Failure、Error、Skip は無い。TF-GridNet と MP-SENet の既存試験は通った。

## 16. Phase 4-D への残課題

Phase 4-D は MP-SENet の公式 GAN 学習を扱う。今回の単一 Optimizer と warmup scheduler を、その GAN の更新順へ流用しない。Phase 4-D は実装済みで、記録は [PHASE_4D_MP_SENET_GAN.md](PHASE_4D_MP_SENET_GAN.md)。

残る未接続は、MossFormerGAN、DeepFilterNet3、SEMamba++、URGENT24 の TF-GridNet である。FSDP、sharded checkpoint、AMP BF16、`audio-db-training-v1` からの再開、公式 DNS 全量の学習も未実装である。
