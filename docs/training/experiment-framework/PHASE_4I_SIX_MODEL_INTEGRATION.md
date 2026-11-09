# Phase 4-I: 6 モデル統合検証

判定: **CONDITIONAL GO。** MP-SENet、MossFormerGAN、TF-GridNet、TF-Locoformer、SEMamba++ は、この文書に書いた範囲で Experiment Framework から構築でき、CPU / FP32 の現行 Task との一致を維持している。DeepFilterNet3 の学習アルゴリズムも同じ枠で動く。全体を無条件 GO にしない理由は、追跡中の `libdf_rfft.so` がこの glibc 2.31 では読めないこと、CUDA の bit 一致を主張しないこと、MossFormerGAN の PESQ skip 混在 NCCL をまだ測っていないことである。

この判定は公式データ全量の学習でも論文スコアの再現でもない。`dfn3.online.official@1` と `semambapp.gsr.official@1` は blocked のままである。`data_provenance.json` は足していない。

## 1. 6 モデルの検証マトリクス

セルは今回の再実行、過去 Phase の再実行、未測定を分けてある。未測定は PASS にしていない。

| 対象 | MP-SENet | MossFormerGAN | TF-GridNet | TF-Locoformer | DeepFilterNet3 | SEMamba++ |
|---|---|---|---|---|---|---|
| Factory / Builder | 今回。名前分岐なし | 今回。名前分岐なし | 今回。名前分岐なし | 今回。名前分岐なし | 今回。名前分岐なし | 今回。名前分岐なし |
| 公式モデル構築 | 今回。公式幅 YAML の preflight PASS。学習はしていない | 今回。公式幅 preflight PASS。学習はしていない | 今回。公式幅、world size 2 の preflight PASS。学習はしていない | 今回。公式幅、world size 4 の preflight PASS。学習はしていない | 今回。公式幅、world size 1 の preflight PASS。学習はしていない | 今回。公式幅、reference、world size 1 の preflight PASS。学習はしていない |
| DataModule | 公式 YAML は VoiceBank 固定ペア。短い試験は合成 | 公式 YAML は VoiceBank 固定ペア。短い試験は合成 | 公式 YAML は MS-SNSD chunk。短い試験は合成 | 公式 YAML は DNS chunk。短い試験は合成 | 今回。smoke snapshot の 1 step | 公式 YAML は smoke。数値試験は合成 2,048 |
| CPU 数値一致 | 既存試験をこのセッションで再実行。最大絶対誤差 0 の主張は Phase 4-D のまま | 既存試験を再実行。Phase 4-E のまま | 既存試験を再実行。Phase 4-B のまま | 既存試験を再実行。Phase 4-C のまま | 既存試験を再実行。同じ FFT 経路で 0。native と fallback は別経路 | 既存試験を再実行。reference、蓄積 1 と 2 で 0 |
| 公式 Loss | D 先行、遅延 G backward、PESQ | G のあと D、PESQ skip、増分平均 | `mr_l1_tfd` | `mr_l1_tfd` | multi-resolution、local SNR、deep-filter cosine | G/MRD/CQT。式は変えていない |
| Optimizer / Scheduler | AdamW 2 つ、epoch 末 ExponentialLR | AdamW 2 つ。Scheduler なし。独自 plateau | Adam、epoch 末 Plateau | AdamW、warmup のあと plateau | AdamW、cosine。index は epoch 内 | AdamW 2 つ、epoch 末 ExponentialLR |
| Validation | PESQ。開始 epoch 前は選ばない。1 step では未実行 | rank 平均、欠測は分母。1 step では未実行 | `valid.loss` の n-best。1 step では未実行 | `valid.loss` 最小と SI-SNR 最大。1 step では未実行 | 発話の等重み。1 step では未実行 | waveform L1、SI-SDR、PESQ、STOI。best generator は無し |
| Checkpoint / Resume | 既存の中断再開を再実行。間隔 5000 の 1 step は `.pt` なしで COMPLETED | 既存の中断再開を再実行。公式間隔 5786 は 1 step では `.pt` を書かない | 既存の中断再開を再実行 | 既存の中断再開を再実行 | 今回。plan、identity、world、component、破損を拒否 | 既存の中断再開を再実行。NCCL の再開は下の CUDA 節 |
| CPU/Gloo | 既存 world size 2 を再実行 | 既存 world size 2 を再実行 | 既存 world size 2 を再実行 | 既存 world size 2 を再実行 | 既存 world size 2 を再実行 | 既存 world size 2、蓄積 1 と 2 を再実行 |
| CUDA 単一 GPU | Phase 4-F の短い NCCL に含まれる。今回は再実行していない | Phase 4-E の 1 step。今回は再実行していない | 今回は world size 2 のみ。単一 GPU は Phase 4-F の 1 step のまま | 同じ | Phase 4-G の world size 1、1 更新。今回は再実行していない | Phase 4-H の reference と fused。今回は再実行していない |
| NCCL 複数 GPU | Phase 4-F の短い 2 GPU。今回は再実行していない | Phase 4-E の 1 step、SyncBatchNorm。PESQ skip 混在は未検証 | 今回。小さい構成、world size 2、1 logical step。rank の parameter と buffer は一致 | 今回。同じ | 今回。小さい構成、world size 2、1 logical step。`df_fc_a` は残る。SyncBatchNorm にはしていない | Phase 4-H の reference/fused、蓄積 1/2。今回は reference の 2 step 再開。bit 一致ではない |
| 推論 Export | Phase 4-F の往復を再実行 | Phase 4-F の往復を再実行 | Phase 4-F の往復を再実行 | Phase 4-F の往復を再実行 | 今回。`df_fc_a` を含む DfNet state の strict load | 今回。reference と fused の重みは strict load できる。CPU の fused 設定は reference を実行する |
| データ来歴 | sidecar なし。評価は UNKNOWN | 同じ | 同じ | 同じ | smoke は ready な代替。official は blocked。sidecar なし | smoke は ready な代替。official は blocked。sidecar なし |

小さい合法構成の 1 更新と、公式幅の構築は別である。公式幅で学習はしていない。

## 2. 環境条件

2026-11-09。ホストの glibc は `ldd (Ubuntu GLIBC 2.31-0ubuntu9.18) 2.31`。Python は `.venvs/se`（torch 2.6.0+cu124）。

DeepFilterNet3 の 3 経路:

| 経路 | ファイル | SHA-256 | 結果 |
|---|---|---|---|
| 追跡中の native | `src/se/models/deepfilternet3/libdf_rfft.so`（1,641,616 bytes） | `122fafe319df0eb7b10ba8fd27cf97ffb8542710d57016ad3a1b7af38991e6e8` | `dlopen` は失敗する。必要シンボルに `GLIBC_2.33` と `GLIBC_2.34` がある。ファイルが存在するので PyTorch fallback には落ちず、`RuntimeError` になる |
| ローカル再ビルド | `build/df-rfft/release/libdf_rfft.so`（4,908,712 bytes） | `e1eb0815e5242191e98ca5785ca2a868a01b421d91be7b0fc6e79b994dc77e77` | ABI 1 で読める。このセッションの学習試験は `DF_RFFT_LIBRARY` でこのパスを明示した |
| PyTorch fallback | 存在しないパスを `DF_RFFT_LIBRARY` に指定 | なし | kernel は None。`torch.fft.rfft` に戻る |

追跡ファイルは置換していない。二つの SHA-256 は違う。native と fallback の 1 フレーム差は Phase 4-G の既知差のままである。同じ native 経路で旧 Task と新 Trainer が一致することと、環境変数なしで追跡バイナリから学習できることは別である。後者はこのホストでは成立しない。

CUDA を使った試験は物理 GPU 6 と 7（A100-PCIE-40GB）である。`deterministic: false`。FP32。AMP FP16 は有効にしていない。

## 3. Factory / Builder

`src/experiments/factory.py` と `src/experiments/builder.py` に、6 モデルの名前分岐は無い。YAML のクラスパスをそのまま作る。

6 本の公式 YAML はすべて `audio-db-experiment/v1`、`entry: trainer`、`ProcessRuntime`、`device: cpu`、`precision: fp32`、`max_logical_steps: 1` である。`runtime.world_size` は YAML に書いていない。world size は起動プロセスが決める。

| YAML | 所有 | prepare | world の制約 | checkpoint 間隔 |
|---|---|---|---|---|
| `mp_senet_gan.yaml` | `optim_g` が generator、`optim_d` が metric discriminator | `distribute: none` | world size 1 で構築できる | 5000 |
| `mossformergan_se_16k.yaml` | `optim_g` が generator、`optim_d` が discriminator。scheduler は無い | `ddp` と `sync_batchnorm` | world size 1 は DDP で包まない | 5786 |
| `tf_gridnet_ms_snsd.yaml` | Adam が model。Plateau | `ddp` | `global_batch` 4、`micro_batch` 2 なので world size 2 | 1 |
| `tf_locoformer_dns.yaml` | AdamW が model。WarmupReduceLROnPlateau | `ddp` | `global_batch` 4、`micro_batch` 1 なので world size 4 | 1 |
| `deepfilternet3.yaml` | AdamW が model。`find_unused_parameters: true` | `ddp`。SyncBatchNorm ではない | `batch_schedule` は world size 1 だけ | 1 |
| `semambapp_gsr.yaml` | `optim_g` が generator。`optim_d` が mrd と cqt | `ddp` | world size 1 で構築できる。scan は reference | 1 |

`--preflight` は 6 本とも `status: PASS`、`training_executed: false`、`logical_step: 0` だった。出力先は `/tmp` のコピーで、リポジトリの `runs/` は書いていない。TF-GridNet は CPU / Gloo の world size 2、TF-Locoformer は world size 4、ほかは 1 プロセスである。これは公式幅の構築であり、1 更新の学習ではない。

## 4. DataModule

| モデル | snapshot | claim | profile 欄 | 状態 |
|---|---|---|---|---|
| MP-SENet | `mp_senet.vb28.ondemand.official` | `official_reproduction` | `official_faithful` | ready |
| MossFormerGAN | `mossformergan.vb28.ondemand.official` | `official_reproduction` | `official_faithful` | ready |
| TF-GridNet | `tf_gridnet.ms_snsd.espnet_recipe` | `espnet_public_se_recipe_reproduction` | `official_faithful` | ready |
| TF-Locoformer | `tf_locoformer.dns.accept_recipe.research_valid_fixed` | `research_checkpoint_selection` | `official_faithful` | ready。選択集合の名前であり、`cv_synthetic_small` ではない |
| DeepFilterNet3 | `dfn3.online.smoke` | `protocol_conformant_substitute_materials` | `official_faithful` | ready。train 8、valid 4 は Phase 4-G のまま |
| SEMamba++ | `semambapp.gsr.smoke` | `protocol_conformant_substitute_materials` | `official_faithful` | ready。VCTK 0.80、DNS2020 noise、OpenSLR28 RIR |

`reproduction-p8.duckdb` の `dfn3.online.official` と `reproduction-p9.duckdb` の `semambapp.gsr.official` は、どちらも `declared_status = blocked`、`claim_label = official_reproduction` である。snapshot ディレクトリは無い。smoke と同一視しない。

TF-Locoformer の dataset JSON の `profile` 欄は `official_faithful` である。研究用であることは `claim_label` と Trainer の `validation_profile` が持っている。

## 5. CPU 数値一致

正本は CPU / FP32、`deterministic: true` の既存試験である。この Phase は式を変えていない。回帰の `tests/experiments` が、Phase 4-B から 4-H の最大絶対誤差 0 の試験を再実行する。

モデル固有の更新は残している。

- MP-SENet は D を先に更新し、Generator の backward を遅らせ、勾配を手で平均する。
- MossFormerGAN は G1/D4 から始まり、PESQ 失敗では Optimizer を進めず、micro-batch ごとに clip し、増分平均する。
- TF-GridNet は Adam と epoch 末の Plateau である。
- TF-Locoformer は AdamW で、warmup の `step_batch` が plateau より先である。
- DeepFilterNet3 は同じ FFT 経路での一致である。追跡バイナリが読めない環境での一致は、ローカル native または fallback の一致であり、libdf との bit 一致ではない。
- SEMamba++ は reference scan である。蓄積 1 は Generator forward 1 回、蓄積 2 は preview 2 回と再 forward 2 回である。

公式幅・実データ全量での一致は測っていない。

## 6. DDP

CPU / Gloo、world size 2 は 6 モデルとも既存試験にある。TF-Locoformer の公式 YAML が要求する world size は 4 で、Gloo の受け入れ試験は world size 2 の小さい構成である。公式 YAML の world size 4 は preflight の構築だけを今回確認した。

DeepFilterNet3 の `batch_schedule` は world size 1 以外を `RuntimeFailure` にする。world size 2 の Gloo と今回の NCCL は、schedule を外した小さい YAML である。出荷 schedule の複数 GPU は未検証である。

## 7. CUDA / NCCL

2026-11-09。物理 GPU 6 と 7。world size 2。FP32。`deterministic: false`。

DeepFilterNet3。小さい `DfNet`（`conv_ch` 16）、`dfn3.online.smoke@1`、`find_unused_parameters: true`、`global_batch` 2、`micro_batch` 1、logical step 1。両 rank の state dict の最大絶対誤差は 0 だった。`SyncBatchNorm` は 0 個である。checkpoint の model state に `df_fc_a` がある。損失は有限だった。CPU との bit 一致は主張しない。BatchNorm は rank 0 の buffer を配る既存の動作のままで、SyncBatchNorm にはしていない。

SEMamba++。狭い Generator（`hid_feature` 8、`num_tfmamba` 1）、合成 2,048 サンプル、reference、蓄積 1、logical step 2。1 step で止めたプロセスを別の起動で 2 step まで再開した。両 rank の parameter と buffer は再開後も一致した。checkpoint には generator、mrd、cqt、`optim_g`、`optim_d`、両 ExponentialLR、RNG がある。`scan_configured` と `scan_executed` はどちらも reference だった。連続実行との最大絶対誤差は、1 回目が 1.219e-3、確認の再実行が 4.384e-4 だった。同じ 2 step を最初から 2 回走らせた自己再実行差は 1.611e-3 だった。再開差は自己再実行差より小さいが、厳密な数値同等性の保証には使わない。`reflection_pad1d` の CUDA backward は決定的でない。fused の NCCL は Phase 4-H の 1 step のままで、今回は再開していない。CPU で `configured=fused` を指定すると `executed=reference` になり、これは fused GPU 試験に数えない。

TF-GridNet と TF-Locoformer。既存の小さい合成構成、world size 2、logical step 1。両 rank の state dict の最大絶対誤差は 0 だった。`SyncBatchNorm` は 0 個である。公式幅ではない。CPU との bit 一致は主張しない。

MossFormerGAN の PESQ skip 混在 NCCL は未検証のままである。MP-SENet の NCCL は Phase 4-F の短い 2 GPU を再実行していない。

3 秒、batch 8 の SEMamba++ は Phase 4-H で OOM した条件のままである。今回は成功扱いにしていない。公式幅 8,000 サンプルの 1 更新も再実行していない。

checkpoint の `parameter_ranks_agree` は、保存時に rank を比較せず True を書く。上の一致は、各 rank が自分の state dict を書いたファイルの比較である。フラグだけでは証明にしない。

## 8. Checkpoint / Resume

形式は `audio-db-experiment-v1` のままである。`provenance_sha256` は None である。

6 モデルの既存の連続実行と中断再開は、回帰の既存試験が再実行する。今回足した横断試験は次である。

- DeepFilterNet3 と SEMamba++ の checkpoint を、6 モデルの別設定へ渡すと `ExperimentError` で拒否する。元のファイルは変えない。
- DeepFilterNet3 の同じディレクトリで、plan hash、dataset identity、world size、component id、破損バイトを拒否する。
- `complete.json` があるディレクトリの resume は拒否する。metrics は変えない。
- `max_logical_steps: 1` かつ `checkpoint_interval: 5000` で、epoch の途中なら `step_*.pt` は無い。`metrics.jsonl` は 1 行、`status.json` は COMPLETED、`complete.json` はある。DeepFilterNet3 の smoke と、小さい MP-SENet（epoch を 1 step で閉じない設定）で確認した。epoch 末の境界は、間隔に関係なく checkpoint を書く。MP-SENet の公式 YAML は間隔 5000 で、1 step では epoch が終わらないので `.pt` は出ない。MossFormerGAN の公式間隔は 5786 で、同じ余り規則である。本番 YAML は変えていない。

GAN の片側だけを完成 checkpoint にしない規則は、各 Trainer の既存実装のままである。SEMamba++ に `best_generator.pt` は足していない。

## 9. Validation

集計規則は変えていない。公式 YAML の `max_logical_steps: 1` は epoch を閉じないので、validation、scheduler、best、n-best、plateau 復元は走らない。小さい試験で epoch を閉じたときの規則は Phase 4-B から 4-H のままである。

SEMamba++ の validation は waveform L1 であり、GAN loss ではない。best generator は選ばない。

## 10. 推論 Export

学習 checkpoint と推論 export は別ファイルである。

DeepFilterNet3。小さい `DfNet` の `export_for_inference` は Discriminator を含まず、`df_fc_a` を含む。`load_enhancer("deepfilternet3", ...)` は strict load し、state の最大絶対誤差は 0 だった。4,800 サンプル、48 kHz の出力長は入力と同じで、有限だった。解析 FFT は、プロセスが読めた kernel に従う。追跡バイナリだけの環境では、この export を使う forward も `RuntimeError` になる。

SEMamba++。同じ seed の reference と fused の初期重みの最大絶対誤差は 0 だった。どちらの export も `{"generator": state_dict}` で、Discriminator は入っていない。`load_enhancer` は `scan_backend: reference` でも `fused` でも strict load できる。CPU では fused 設定の実行 kernel は reference である。800 サンプル、16 kHz の出力長は入力と同じで、有限だった。

既存 4 モデルの往復は Phase 4-F の試験を再実行する。

## 11. Dataset Identity

上の表の snapshot を、`data/db/snapshots/*/dataset.json` と読み取り専用の `reproduction-p8.duckdb`、`reproduction-p9.duckdb` で確認した。DB も snapshot も再構築していない。

resume は dataset identity、plan hash、config identity、component id、world size が違うと拒否する。今回の破損試験は DeepFilterNet3 で行い、4 モデル間の取り違えは Phase 4-F の試験を再実行する。

## 12. Provenance

実験 checkpoint は `data_provenance.json` を書かない。`provenance_sha256` は None である。None 以外の resume は既存の `CheckpointError` のままである。

`judge_checkpoint` は sidecar 欠落を `"data_provenance.json is missing"` とし、`independent` は `UNKNOWN`、`confirmed_independent` は false だった。UNKNOWN は独立の証明ではない。sidecar は足していない。本格的な接続は Phase 4-J に残す。

## 13. ログ・ファイル整合性

短い完走は `metrics.jsonl` の logical step が 1 から順に並び、`status.json` が COMPLETED、`complete.json` がある。中断は INTERRUPTED で `complete.json` が無い。これは既存試験と、今回の間隔試験で見た。

`max_logical_steps: 1` でも、間隔が 1 なら `step_00000001.pt` がある。間隔が 5000 で epoch の途中なら `.pt` は無い。完了と重み保存は同じ条件ではない。

既存の完了済み実験ディレクトリは変更していない。

## 14. 修正内容

学習アルゴリズム、公式 Loss、DataModule の生成、GAN の更新順、旧 Task、Factory / Builder、checkpoint 形式、CLI の既定は変えていない。

足したものは試験だけである。

- `tests/experiments/test_phase4i_integration.py`。6 YAML、3 つの FFT 経路、blocked dataset、export、checkpoint の取り違え、広い checkpoint 間隔。
- `tests/experiments/phase4i_nccl_worker.py` と `tests/experiments/test_cuda_amp.py` の `Phase4ICudaTests`。CUDA を隠すと module ごと NOT_RUN のままなので、CPU 回帰の skip 件数は増えない。

`parameter_ranks_agree` を実際の比較に変える修正は入れていない。フラグは比較していない、という記録だけを残す。

## 15. 回帰テスト

コマンドは次である。`DF_RFFT_LIBRARY` はローカル再ビルドを指す。外すと追跡バイナリの glibc 問題が残る。

```bash
CUDA_VISIBLE_DEVICES="" .venvs/se/bin/python -m unittest discover -s tests/experiments -v
CUDA_VISIBLE_DEVICES="" .venvs/se/bin/python -m unittest discover -s tests/training -v
CUDA_VISIBLE_DEVICES="" .venvs/se/bin/python -m unittest discover -s tests/reproduction -v
```

Phase 4-H の基準は experiments 199 OK / skip 1、training 297 OK / skip 12、reproduction 135 OK / skip 2 である。

2026-11-10 00:04–00:43（JST）。上の 3 コマンドを、`DF_RFFT_LIBRARY` にローカル再ビルド `build/df-rfft/release/libdf_rfft.so` を指定し、`CUDA_VISIBLE_DEVICES=""` で順に実行した。

| suite | 結果 | 基準との差 |
|---|---|---|
| experiments | Ran 209、OK、skipped=1、973.671s、exit 0 | OK は 199 から 209。増分 10 件は今回の CPU 試験（宣言 4、export 2、checkpoint 4）。skip は 1 のまま。CUDA を隠した `test_cuda_amp` が module ごと NOT_RUN になる |
| training | Ran 297、OK、skipped=12、924.708s、exit 0 | 基準と同じ |
| reproduction | Ran 135、OK、skipped=2、452.710s、exit 0 | 基準と同じ |

失敗、エラー、新規 skip は無い。`DF_RFFT_LIBRARY` を外すと、追跡中の `libdf_rfft.so` は glibc 2.31 で読めず、DeepFilterNet3 の native 経路は `RuntimeError` になる。この回帰の成立は、ローカル再ビルドを明示した条件に限る。

## 16. 各モデルの最終判定

| モデル | 判定 | 範囲 |
|---|---|---|
| MP-SENet | GO | CPU / FP32 の公式 GAN 一致、CPU / Gloo、Phase 4-F の短い NCCL。公式全量と論文スコアは含まない |
| MossFormerGAN | GO | CPU / FP32 と CPU / Gloo、Phase 4-E の NCCL 1 step。PESQ skip 混在の NCCL は未検証のまま GO の外に置く |
| TF-GridNet | GO | CPU / FP32 の一致。今回の NCCL は小さい構成の 1 step で rank が一致した。公式幅の NCCL ではない |
| TF-Locoformer | GO | 同じ。研究用 validation profile は `tf_locoformer.dns.accept_recipe.research_valid_fixed@1` |
| DeepFilterNet3 | CONDITIONAL GO | 同じ FFT 経路での CPU 一致と、小さい構成の NCCL 1 step は成立した。追跡 native library はこのホストで読めない。`batch_schedule` の複数 GPU は未検証。`dfn3.online.official@1` は blocked |
| SEMamba++ | GO | CPU / FP32、reference、蓄積 1 と 2 の一致。CUDA は bit 一致ではない。今回の NCCL 再開は rank が一致し、連続実行との差は自己再実行差の内側だった。`semambapp.gsr.official@1` は blocked。best generator は無い |

## 17. Phase 4 全体の最終判定

**CONDITIONAL GO。** 6 モデルは同じ Experiment Framework から構築でき、検証した CPU 条件では現行 Task と一致し、短い学習、checkpoint、既存 Loader への export が成立する。DeepFilterNet3 の追跡 native library が未解決なので、全体を無条件 GO とは書かない。CUDA の bit 一致、公式全量、論文スコア、`data_provenance.json` もこの GO に含めない。

## 18. Phase 4-J への残課題

- 追跡中の `libdf_rfft.so` を、この glibc で読める成果物へ配備する。追跡ファイルの無断置換はしない。
- 実験 checkpoint の `data_provenance.json` と `provenance_sha256`。欠落を独立と読まない契約を維持したまま接続する。
- `parameter_ranks_agree` を、保存前の rank 比較にする。
- MossFormerGAN の PESQ skip 混在 NCCL。
- DeepFilterNet3 の `batch_schedule` を付けた複数 GPU。
- SEMamba++ fused の NCCL 再開。今回の再開は reference だけである。
- 長時間学習の運用、中断、ログ。30 分以上の投入は別の承認が要る。
- world size を変える resume、AMP BF16、複数 Optimizer の AMP FP16、FSDP、sharded checkpoint。
