# Phase 4-F: 4モデル統合検証

判定: **GO（CPU / Gloo）。** MP-SENet、MossFormerGAN、TF-GridNet、TF-Locoformer の公式学習経路は、Experiment Framework 上で構築でき、CPU / FP32 の新旧数値一致と CPU / Gloo の短い DDP を維持している。公式データ全量の学習と論文スコアの再現はしていない。DeepFilterNet3 と SEMamba++ は対象外である。

GPU / NCCL は別記録である。MP-SENet の短い 2 GPU 更新は今回確認した。MossFormerGAN の CUDA / NCCL は [Phase 4-E](PHASE_4E_MOSSFORMERGAN_TRAINING.md) の 1 step のままで、今回は繰り返していない。TF-GridNet と TF-Locoformer の CUDA は world size 1 の 1 step である。NCCL は使っていない。

基準 commit は `5303e7d`。作業ツリーは Phase 4-F の検証差分を含む。

## 1. 検証マトリクス

4 つの公式 YAML は同じ schema `audio-db-experiment/v1` を Factory と Builder が読む。`src/experiments/factory.py` と `src/experiments/builder.py` にモデル名の分岐は無い。

| 項目 | MP-SENet | MossFormerGAN | TF-GridNet | TF-Locoformer |
|---|---|---|---|---|
| YAML | `configs/experiments/mp_senet_gan.yaml` | `configs/experiments/mossformergan_se_16k.yaml` | `configs/experiments/tf_gridnet_ms_snsd.yaml` | `configs/experiments/tf_locoformer_dns.yaml` |
| Trainer | `MPSENetGANTrainer` | `MossFormerGANTrainer` | `TFGridNetTrainer` | `TFLocoformerTrainer` |
| Optimizer | AdamW G/D、lr 0.0005 | AdamW G 0.0005 / D 0.001 | Adam 0.001 | AdamW 0.001 |
| Scheduler | ExponentialLR ×2 | 無し。plateau は Trainer | ReduceLROnPlateau | WarmupReduceLROnPlateau |
| データ | 固定ペア `official_faithful` | 固定ペア `official_faithful` | chunk、global batch 4 | chunk、global batch 4 |
| 必要 world size | 1 で構築できる | 1 で構築できる。prepare は ddp | 2。1 は拒否 | 4。1 は拒否 |
| Precision | fp32。`amp_fp16` は Trainer が拒否 | fp32。`amp_fp16` は Trainer が拒否 | fp32 | fp32 |
| 特殊状態 | D 先行、遅延 G backward | PESQ skip、部分蓄積、micro-batch clip | n-best は `valid.loss` | warmup、`valid.loss` 最小と `valid.si_snr` 最大 |
| Validation | rank 0 の PESQ。失敗は -1 | G eval / D train。欠測は分母 | 波形件数の重み付き loss | 波形件数の loss と SI-SNR |
| Checkpoint state | G、D、両 Optimizer、両 Scheduler、data、runtime、trainer | G、D、両 Optimizer、data、runtime、trainer | model、optimizer、scheduler、data、runtime、trainer | model、optimizer、scheduler、data、runtime、trainer |

`logical_step` は Optimizer が成立した更新である。MossFormerGAN の skip は data cursor と `official_step` だけを進め、`logical_step` は進めない。この意味は他の 3 モデルへ揃えていない。

## 2. CPU 数値一致

今回のセッションで、変更前の `tests/experiments` 166 件を CPU / FP32、`CUDA_VISIBLE_DEVICES=""` で再実行した。失敗 0、skip 1。skip は Phase 2-C の CUDA 非表示による NOT_RUN である。この 166 件が、各 Phase の初期 Parameter、Buffer、forward、loss、勾配、clip、Optimizer 更新後の Parameter と state の最大絶対誤差 0 を含む。

Phase 4-F は損失式、旧 Task、`src/reproduction` の式、DataModule の抽選を変えていない。変更後の再実行は第 13 節である。過去の Phase 文書に書いた 0 を、今回測り直していない単発の数値として再利用していない。再利用しているのは、今回の 166 件の再実行結果である。

## 3. DDP

CPU / Gloo の world size 2 は、変更前の 166 件に各モデルの短い DDP が含まれる。

今回追加した公式幅の短い実行:

- TF-GridNet。`torchrun --nproc_per_node=2`、CPU / Gloo、公式 YAML、`max_logical_steps=1`。22.86 秒。checkpoint の `world_size` は 2、`parameter_ranks_agree` は true、optimizer update は 1。rank 0 と rank 1 の sample id は違う。`phase` は `in_epoch` で、検証履歴は空である。
- TF-Locoformer。`torchrun --nproc_per_node=4`、CPU / Gloo、公式 YAML、`max_logical_steps=1`。222.59 秒。`world_size` は 4、`parameter_ranks_agree` は true、optimizer update は 1、`step_num` は 1。4 rank の sample id は互いに違う。使った学習率は 0.001 で、次の warmup 率は `2.5e-7` である。`phase` は `in_epoch` で、検証履歴は空である。

片 rank の例外で他 rank が待たない試験は、既存の `test_nan_rank_and_split_pesq_do_not_hang` と `test_skip_disagreement_fails_without_hanging` が今回の 166 件に含まれる。

## 4. GPU / NCCL

### MP-SENet（優先度 A）

2026-11-09。物理 GPU index 1 と 2。NVIDIA A100-PCIE-40GB。PyTorch 2.6.0+cu124。world size 2、NCCL、FP32。小さい Generator（`dense_channel` 4、`num_tsconformers` 1、公式 STFT）。`deterministic: false`。logical step 1。

- `average_calls` は両 rank で 1。Generator の明示平均は 1 回である。
- `synced_during_generator_backward` は両 rank で false。遅延 backward の間に DDP の自動同期は走っていない。
- 両 Optimizer の update は 1。
- checkpoint の `parameter_ranks_agree` は true。形式は `audio-db-experiment-v1`。

CPU とのビット一致は主張しない。

### MossFormerGAN（優先度 B）

今回は追加していない。Phase 4-E の A100×2、accumulation 1、logical step 1、SyncBatchNorm 8、`deterministic: false` が CUDA の記録である。`adaptive_max_pool2d_backward_cuda` に決定的実装が無い制約は残る。PESQ skip 混在の NCCL は未実施である。

### TF-GridNet と TF-Locoformer（優先度 C）

world size 1、CUDA / FP32、`deterministic: false`、小さい合法構成、1 logical step。損失は有限で、checkpoint の `parameter_ranks_agree` は true。TF-GridNet の loss は 1812.6871337890625、TF-Locoformer の loss は 2234.3916015625 である。これは未学習の小さい網の値で、公式スコアではない。NCCL は使っていない。CPU とのビット一致は主張しない。

## 5. Checkpoint と Resume

`tests/experiments/test_phase4f_integration.py` の `test_interrupted_resume_keeps_one_metric_line_per_logical_step` が、4 モデルを小さい構成で 1 step 止めてから 2 step まで再開した。

- 形式は `audio-db-experiment-v1`。`provenance_sha256` は None。
- `code_identity` は `git_commit` と `dirty` を持つ。
- `dataset_identity` と data の `plan_hash` がある。
- 1 step の checkpoint バイト列は、再開後も変わらない。
- `complete.json` があるディレクトリの resume は `CheckpointError`。metrics と checkpoint のバイト列は変わらない。
- 他モデルの checkpoint を出力先だけ合わせて resume すると `ExperimentError`。元ファイルは変わらない。循環は MP-SENet → TF-GridNet → TF-Locoformer → MossFormerGAN → MP-SENet である。
- TF-GridNet で、plan hash、dataset identity、world size、壊したバイト列は resume を拒否する。元の step 1 ファイルは変わらない。

MP-SENet は D 更新と G 更新の途中では保存しない。この拒否は Phase 4-D の試験が持っていて、今回の 166 件が再実行した。MossFormerGAN の部分蓄積、skip 後の cursor、同じ logical step の置換、plateau の復元は Phase 4-E の試験が持っていて、同じ 166 件が再実行した。

## 6. Validation

公式 YAML の `max_logical_steps=1` は epoch の途中で終わる。今回の公式幅 1 step は検証も scheduler の epoch 更新も走らせていない。TF-GridNet の検証履歴は空、TF-Locoformer の `history` は空、`phase` は `in_epoch` である。

集計式はモデルごとに残している。

- MP-SENet。rank 0。PESQ 失敗は -1。best は選択開始 epoch 以降。validation interval と ExponentialLR は別である。
- MossFormerGAN。Generator は eval、Discriminator は train。欠測は分母に残る。rank 平均は等重み。5 回目で Generator の学習率だけを半分にし、10 回目で止める。
- TF-GridNet。波形件数の重み。`valid.loss` の最小。n-best。ReduceLROnPlateau。
- TF-Locoformer。波形件数の loss と SI-SNR。`valid.loss` の最小と研究用 `valid.si_snr` の最大。Warmup のあと plateau。最初の更新はコンストラクタの学習率を使い、そのあと `step_batch` が warmup 率を置く。

これらを一つの平均へ置き換えていない。

## 7. Scheduler と Plateau

公式幅の 1 step では epoch が終わらない。TF-Locoformer の `step_num` は 1 で、sample log の使用学習率は 0.001、`scheduled_lr` は `2.5e-7`（`0.001 / 4000`）である。これは 1 更新のあとの warmup 率で、検証による plateau ではない。

MossFormerGAN の plateau と MP-SENet の ExponentialLR は、epoch 末の既存試験が担当する。Phase 4-F はそれらを共通 Scheduler にマージしていない。

## 8. Best / N-best

公式幅の 1 step は best を書いていない。小さい CUDA の TF-GridNet と TF-Locoformer は `updates=1` かつ `max_epochs=1` なので epoch が閉じ、`valid.loss.best.pt` を書いた。TF-Locoformer は `valid.si_snr.best.pt` も書いた。TF-GridNet の実行ディレクトリに `valid.si_snr` は無い。選択規則の単体は既存試験のままである。

## 9. 推論用 Export

`test_inference_exports_round_trip_without_a_discriminator` が、既存の `load_enhancer` で読み戻した。

| モデル | Export のキー | Loader | Forward の最大絶対誤差 |
|---|---|---|---|
| MP-SENet | `generator` だけ | `load_enhancer("mp_senet")` | 重み 0。同じ magnitude / phase の MPNet forward は 0。波形 800 サンプルの `enhance` は有限で長さが一致 |
| MossFormerGAN | `model` だけ。num_channel 64 | `load_enhancer("mossformergan", variant="se_16k")` | 重み 0。同じ 2400 サンプルの推論チャンクは 0。`enhance` の長さは 2400 |
| TF-GridNet | すべて `separator.` | `load_enhancer("tf_gridnet", variant="ms-snsd")` | 波形 forward は 0 |
| TF-Locoformer | すべて `separator.` | `load_enhancer("tf_locoformer", variant="dns")` | 波形 forward は 0 |

Discriminator の state は export に入れていない。学習 checkpoint の payload はこの export ではない。推論スコアは測っていない。

num_channel 4 の MossFormerGAN は attention の view が割れ、公式の 64 では割れない。小さい channel を公式構成として扱っていない。

## 10. Dataset Identity と Lineage

公式幅の checkpoint が持っていた identity:

| モデル | dataset_id | profile | order |
|---|---|---|---|
| TF-GridNet | `13c6058881ef86fe7506f705b1bd5a5bc68b313181410d8fdaaee20c33ef3fc6` | `official_faithful` | `espnet_chunk_v1` |
| TF-Locoformer | `e4a53415122aba24055f4ba7eef6367a7e41d062485d74de3d7ce94c07d7bdd0` | `official_faithful` | `espnet_chunk_v1` |

TF-Locoformer の id は `tf_locoformer.dns.accept_recipe.research_valid_fixed` である。`cv_synthetic_small` ではない。Trainer の `validation_profile` もその名前である。学習 split と研究用 validation profile は同じ snapshot 名を指すが、今回の 1 step は train split の 1 micro-batch だけで、validation は走っていない。

plan hash は checkpoint の data state にある。`code_identity.git_commit` は `5303e7dc8a17d2a9c0b5a6f86b2aa126558c34bd` で、`dirty` は true である。dirty は Phase 4-F の作業ツリーを隠していない。

`data_provenance.json` は書いていない。checkpoint の `provenance_sha256` は None のままである。None 以外は resume が `CheckpointError` にする。評価の `load_provenance` はファイル欠落を `"data_provenance.json is missing"` とし、独立判定は UNKNOWN になる。欠落は独立の証明にならない。公式の材料ハッシュを持たない sidecar を足すと、評価が別の来歴だと誤読し得る。今回は sidecar を足していない。`SnapshotDataModule.lineage()` は呼べるが、実験 checkpoint はそれをファイルにしていない。

## 11. ファイルとログ

公式幅の 1 step は `status.json` が `COMPLETED`、`complete.json` の logical step が 1、`metrics.jsonl` が 1 行である。MP-SENet と MossFormerGAN の `checkpoint_interval` は 5000 と 5786 なので、step 1 の `.pt` は無い。TF-GridNet と TF-Locoformer の interval は 1 なので `step_00000001.pt` がある。rank 別 sample log は world size だけの本数である。

小さい 4 モデルの中断再開では、metrics の `logical_step` が 1 のあと 2 で、重複は無い。完了後の resume は metrics を足さない。

## 12. 実装変更

損失、旧 Task、reproduction、DataModule の抽選、Factory、Builder は変えていない。

`configs/experiments/tf_gridnet_ms_snsd.yaml` と `configs/experiments/tf_locoformer_dns.yaml` の module prepare を `distribute: none` から `distribute: ddp` に変えた。

変更前は、world size 1 が `global_batch` 不一致で拒否され、宣言どおりの world size 2 と 4 は `distribute: none` のため拒否された。どちらの world size でも公式 YAML を構築できなかった。`distribute: ddp` のあと、world size 1 は従来どおり batch 不一致で拒否される。world size 2 と 4 の CPU / Gloo では preflight が PASS し、1 logical step が完了した。world size 1 は DDP で包まない。

この変更は損失式でも Optimizer の更新順でもない。`max_logical_steps` は 1 のままである。

## 13. 回帰テスト

変更前。`CUDA_VISIBLE_DEVICES=""`。基準は Phase 4-E 終了時の件数と一致した。

| Suite | 件数 | 結果 |
|---|---:|---|
| `tests/experiments` | 166 | OK、skip 1 |
| `tests/training` | 297 | failure 1、error 6、skip 12 |
| `tests/reproduction` | 135 | error 1、skip 2 |

失敗は DeepFilterNet3 の `libdf_rfft.so` が要求する `GLIBC_2.33` だけである。今回は直していない。

変更後の `tests/experiments` は 171 件、785.195 秒、OK、skip 1、失敗 0 だった。増加の 5 件は Phase 4-F の横断試験である。skip は変更前と同じ Phase 2-C の NOT_RUN である。`tests/training` と `tests/reproduction` は `src/training` と `src/reproduction` を変えていない。上の実行がこのセッションの結果である。

Phase 4-F の新規試験は `tests/experiments/test_phase4f_integration.py` の 5 件である。公式 YAML の契約、chunk の world size、4 モデルの中断再開と取り違え拒否、推論 export の往復を見る。モデルごとの数値試験の複製ではない。

## 14. 未検証と既知の制約

- 公式データ全量の学習と論文スコア。
- DeepFilterNet3、SEMamba++、URGENT24 の TF-GridNet。
- FSDP、sharded checkpoint、AMP BF16、GAN の複数 Optimizer AMP。
- `audio-db-training-v1` からの resume。world size を変える resume。
- 実験 checkpoint の `data_provenance.json`。評価は UNKNOWN になる。
- MossFormerGAN の CUDA で PESQ skip を混ぜた NCCL。Phase 4-E の 1 step を超えていない。
- TF-GridNet と TF-Locoformer の NCCL。今回の CUDA は world size 1 である。
- CUDA のビット一致。MP-SENet の NCCL と、Grid / Locoformer の CUDA は `deterministic: false` である。
- MP-SENet と MossFormerGAN の公式 YAML は `checkpoint_interval` が大きいので、1 step では `.pt` を書かない。metrics と `complete.json` は書く。

## 15. Phase 4 の完了判定

4 モデルについて、公式経路の構築、CPU 数値一致の維持、CPU / Gloo の短い DDP、checkpoint と resume、validation と scheduler の意味、推論 export、identity の拒否、ログの整合、既存試験の退行なし、未検証範囲の記録を満たした。Phase 4-F は GO である。

この GO は、この 4 モデルを今後の学習実験で Experiment Framework から起動できることの検証である。公式全量の完走でも、論文スコアの再現でも、DeepFilterNet3 と SEMamba++ の移植でもない。

## 16. 次段階

DeepFilterNet3 と SEMamba++ の Trainer は、この 4 モデルの Trainer をマージせず、別段階で接続する。`data_provenance.json` を実験 checkpoint へ出す場合は、`provenance_sha256` の resume 契約を 4 モデル全部で変える独立した変更にする。旧 CLI は残す。実験ディレクトリを旧 CLI と共用しない。
