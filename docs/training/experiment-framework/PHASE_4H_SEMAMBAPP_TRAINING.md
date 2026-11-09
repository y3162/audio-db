# Phase 4-H: SEMamba++

判定: **GO。** 公式 GAN の更新順、損失、AdamW、epoch 末の ExponentialLR、online SEMamba++ 供給、checkpoint を Experiment Framework に接続した。CPU / FP32、reference selective scan、狭い Generator で、現行 `SEMambaTask` との損失、勾配、更新後 Parameter、両 AdamW state の最大絶対誤差は 0 だった。

この判定は公式 VCTK-GSR 全量の学習でも論文スコアの再現でもない。`semambapp.gsr.official@1` は blocked のままである。使った ready な snapshot は `semambapp.gsr.smoke@1` で、`claim_label` は `protocol_conformant_substitute_materials` である。素材は VCTK 0.80、DNS2020 noise、OpenSLR28 RIR であり、VCTK 0.92、WHAM!、Arni RIR、公式話者分割ではない。

基準 commit は `70146ef`。作業ツリーは Phase 4-H の差分を含む。

## 1. 変更ファイル

- `src/experiments/models/semambapp.py` — Generator、MRD、CQT Discriminator
- `src/experiments/semambapp.py` — `SEMambaPPTrainer`
- `configs/experiments/semambapp_gsr.yaml` — 公式幅の宣言。`max_logical_steps` は 1
- `tests/experiments/test_semambapp.py` — CPU 試験
- `tests/experiments/semamba_cuda_probe.py` — CUDA の 1 step 計測。discover 対象外
- `tests/experiments/test_cuda_amp.py` — CUDA / NCCL 試験。GPU を隠すと module ごと NOT_RUN

Factory と Builder にモデル名の分岐は足していない。`src/training/` と `src/reproduction/` の数値計算は変えていない。Generator と Discriminator は再実装していない。

## 2. Generator

`SEMambaPPGenerator` は `src.se.build_model("semambapp")` の `SEMambaPP` を 1 組持つ。STFT は既存の `semamba_mag_phase_stft` / `semamba_mag_phase_istft` である。

- 16 kHz、FFT 400、hop 100、window 400、magnitude compression 0.3
- clean / noisy の STFT は `addeps=False`
- 再計算 complex spectrum は `addeps=True`
- `_match_bins` と `_fit_time` は現行 `_generate` と同じ条件
- `scan_backend` は `reference` または `fused`。別アーキテクチャにはしない

宣言 YAML の公式幅は `hid_feature=48`、`num_tfmamba=4`、`d_state=16`、`d_conv=4`、`expand=2`、`norm_epsilon=1e-5` である。CPU の数値比較は `hid_feature=8`、`num_tfmamba=1`、`d_state=4`、`d_conv=2`、長さ 2,048 である。

## 3. MRD / CQT

再利用したクラスは `MultiResolutionDiscriminator` と `MultiScaleSubbandCQTDiscriminator` である。解像度は `(1024, 120, 600)`、`(2048, 240, 1200)`、`(512, 50, 240)`。CQT は hop 512 / 256 / 256、octave 8、bins per octave 12 / 24 / 36、nnAudio `CQT2010v2`、16 kHz から 32 kHz への resample、内部幅 128 のままである。

Discriminator 用 AdamW は 1 つで、MRD と CQT を別 param group として持つ。学習率、betas、weight decay は同じなので、現行 Task の 1 group と同じ Parameter 更新になる。CPU で Adam state の最大絶対誤差は 0 だった。

## 4. 初期化順序

現行 Task は一つの `seed_model_init` で Generator、MRD、CQT の順に作る。Builder の stage 0 はその順を保たない。各クラスは同じ seed で入り、捨てるモジュールの乱数だけを消費する。捨てたモジュールは登録せず、Optimizer の対象にしない。

seed 1234 の狭い構成で、Builder が CQT、Generator、MRD の順に作っても、現行 Task の Parameter と Buffer の最大絶対誤差は 0 だった。

## 5. 公式 Loss

再利用した関数は `least_squares_discriminator`、`least_squares_generator`、`feature_matching`（`scale=2.0`、`reduction="sum"`）、`compressed_complex_losses`（`phase_mode="matrix"`）、`summed_slaney_log_mel` である。

Generator は adversarial + feature matching + 0.1 mel + 0.9 magnitude + 0.3 phase + 0.1 complex + 0.1 consistency である。Waveform L1 は学習損失に足さない。Discriminator は MRD と CQT の least squares の和で、各 scale の和は現行のままである。独立の学習重みは足していない。

CPU で個別に差 0 だった項は adversarial、feature matching、mel、magnitude、phase、complex、consistency、generator total、discriminator total である。

## 6. Accumulation 1 / 2

Accumulation 1 は Generator を 1 回 forward し、その計算グラフを Discriminator 更新のあとまで保持する。再 forward はしない。

Accumulation 2 以上は、各 micro-batch を `no_grad` で preview し、Discriminator を更新したあと、同じ Tensor で Generator を forward し直す。DataModule は再消費しない。

両方とも損失を accumulation で割ってから backward する。Generator 損失の backward 中は MRD と CQT の Parameter を freeze し、`finally` で戻す。例外を起こした試験でも `requires_grad` は戻った。

`grad_clip` が null のときは clip しない。値が正のときは MRD と CQT を別々に clip し、Generator も別である。一括 clip にはしていない。宣言 YAML は null で、現行 Task がキー無しのときと同じである。

## 7. Generator Forward 回数

狭い CPU 構成で、1 logical step あたりの回数は次である。

| accumulation | graph forward | preview | 更新後の再 forward |
|---|---|---|---|
| 1 | 1 | 0 | 0 |
| 2 | 0 | 2 | 2 |

Gloo world size 2 の checkpoint に書いた回数も同じだった。

## 8. DDP Gradient 同期

3 module を DDP にする。Discriminator の Optimizer は 1 つである。

Accumulation 1 では、Generator forward のとき `require_backward_grad_sync` を true のままにする。PyTorch 2.6 はその場で reducer を準備するので、Discriminator 更新のあとの backward が all-reduce する。明示の勾配平均は足していない。

Accumulation 2 以上では、preview は unwrap した module の `no_grad` である。再 forward と backward は最後の micro-batch だけ同期する `sync_context` の中で行う。Discriminator も最後の micro-batch だけ同期する。

片 rank の非有限入力は、forward の前に全 rank が `agree` してから失敗する。Gloo の試験はハングせず、`rejected its micro-batch` で戻った。

保存の前に 3 module の buffer を rank 0 から配る。rank ローカルの損失列は trainer state に入れず、rank 0 のスカラーを共有する。

## 9. Selective Scan Backend

設定名と、その device で実行される kernel を分ける。

- `reference` は CPU でも CUDA でも reference scan である。学習中の reference scan は既存の gradient checkpoint を使う。
- `fused` かつ CUDA かつ `mamba-ssm` を import できるときだけ fused kernel である。
- `fused` を CUDA で指定し、`mamba-ssm` が無いときは reference に落とさず `RuntimeFailure` にする。
- CPU で `fused` を指定すると、既存の `SelectiveSSM` は reference を実行する。Trainer は `configured=fused`、`executed=reference` と記録する。設定を reference へ書き換えない。

`.venvs/se` には `mamba-ssm` が無い。fused の CUDA 試験は `.venvs/semambapp`（`mamba-ssm==2.3.2.post1`）で実行し、`scan_executed` は `fused` だった。

## 10. GPU メモリ

NVIDIA A100-PCIE-40GB、batch 1、FP32、`deterministic: false`。`reflection_pad1d` の CUDA backward に決定的実装が無く、`deterministic: true` では現行 Task の backward が失敗する。

| 構成 | backend | 実行 kernel | step 秒 | peak allocated |
|---|---|---|---|---|
| 幅 8、1 層、2,048 サンプル | reference | reference | 2.90 | 361,862,144 |
| 幅 8、1 層、2,048 サンプル | fused | fused | 3.33 | 505,116,160 |
| 公式幅、4 層、8,000 サンプル | reference | reference | 8.94 | 1,815,268,864 |
| 公式幅、4 層、8,000 サンプル | fused | fused | 6.13 | 1,485,900,288 |

3 秒・batch 8 は実行していない。過去の研究記録では、その条件は約 45.56 GiB を確保したあと Discriminator で OOM している。micro 4 × accumulation 2 を公式 batch 8 の代替とは呼ばない。

## 11. Optimizer / Scheduler

G と D は AdamW、学習率 0.0002、betas `(0.8, 0.99)`、weight decay 0.01、eps 1e-8、amsgrad なしである。違う値は構築時に拒否する。

G と D のそれぞれに `ExponentialLR(gamma=0.99)` がある。epoch の途中では進めない。epoch が終わった logical step で、両方を 1 回ずつ進める。validation interval だけでは進めない。狭い CPU で、epoch 途中の学習率は 0.0002 のままで、2 micro-batch の epoch 末は両方とも `0.0002 * 0.99` だった。

`amp_fp16` は拒否する。2 つの AdamW を現行の GradScaler で一つの原子的更新にできない。更新順は変えていない。BF16、FSDP、sharded checkpoint は対象外である。

## 12. Online Data

既存の `SnapshotSource`、`OrderSampler`、`SemambaProtocol`、`MicroBatcher`、`SnapshotDataModule` を使う。アルゴリズムは `online.semambapp_gsr`。教師は `dry_clean`。劣化の式は変えていない。

`semambapp.gsr.smoke@1` で確認したこと:

- `claim_label` は `protocol_conformant_substitute_materials`
- 波形は 16 kHz、48,000 サンプル
- 同じ index の再 materialize は同じ waveform、sample id、realization id
- plan hash は同じ epoch で不変
- `group_size` 3 は端数を捨て、更新数は 3 の倍数で素材数より小さい
- `group_size` と Trainer の accumulation が違うと、学習の前に拒否する
- materialize は data cursor を進めない。Generator の再 forward も進めない

`.venvs/se` の `resampy` は 0.4.3 で、`requirements/se.in` に既にある。この Phase では依存を足していない。smoke の 1 micro-batch は、この環境で描画して学習まで進んだ。

## 13. Validation

Rank 0 だけが `SemambaProtocol` の valid を見る。他 rank は空である。集計は現行の `_aggregate_validation` である。指標は waveform L1、SI-SDR、PESQ、STOI、入力 / 出力 / 参照の長さである。L1 は GAN の学習損失ではない。

検証の前に RNG を保存し、あとで戻す。学習 cursor は進まない。`best_checkpoint_start_epoch` が 0 でも `best_generator.pt` は書かない。履歴には `semambapp_does_not_select_best_generator` を残す。

## 14. Checkpoint

形式は `audio-db-experiment-v1` である。`provenance_sha256` は None のままである。`data_provenance.json` は書かない。

保存する状態は Generator、MRD、CQT、Optimizer G、Optimizer D、Scheduler G、Scheduler D、Trainer、Data、Runtime である。D の更新後、G の更新前には保存しない。保存は両 Optimizer が終わった logical step のあとだけである。

epoch 末は label `validation_pending` を書いてから、検証と Scheduler を一度だけ行う。中断した再開は、その境界を連続実行と同じ結果にする。

## 15. Resume

狭い CPU、world size 1 で、連続 2 更新と、1 更新で止めて別プロセスから再開した 2 更新目を比べた。Parameter、Buffer、両 AdamW、両 ExponentialLR、epoch、data cursor、RNG、損失列、学習率列は一致した。

epoch 境界は、`validation_pending` のまま止めて再開すると、検証と Scheduler が一度だけ走り、連続実行の checkpoint と一致した。破損ファイルと dataset identity の不一致は再開しない。

Accumulation 1 と 2 の Gloo world size 2 でも、保存された parameter は rank 間で一致し、再開に使う state が割れていない。

## 16. 新旧数値一致

CPU / FP32、reference、seed 1234、同じ初期状態、同じ 2,048 サンプルである。現行 `SEMambaTask.train_update` との最大絶対誤差は 0 だった。

比べた段階は、初期 Generator / MRD / CQT、各損失、Discriminator 勾配、D 更新後の Parameter と Adam state、Generator 勾配、G 更新後の Parameter と Adam state である。Accumulation 1 と 2 の両方である。最初に差が出た段階は無かった。

## 17. GPU / NCCL

CUDA の bit 一致は主張しない。`reflection_pad1d_backward_out_cuda` に決定的実装が無い。同じ Task を同じ重みと入力で 2 回走らせた自己再実行の差は、損失 8.61e-6、Generator 勾配 2.21e-4、更新後 Generator 3.89e-4 だった。

狭い CUDA、Task と Trainer の差は、その自己再実行の範囲に収まった。

| 比較 | 損失 | Generator 勾配 | 更新後 Generator |
|---|---|---|---|
| reference Task 対 Trainer | 3.32e-5 | 7.36e-5 | 3.84e-4 |
| fused Task 対 Trainer | 2.96e-6 | 7.57e-5 | 4.00e-4 |
| reference 対 fused の更新後 Generator | — | — | 4.00e-4 |

公式幅、8,000 サンプルでも、Task 対 Trainer の更新後 Generator は reference 3.9985e-4、fused 3.9989e-4 だった。reference 対 fused の更新後 Generator は 4.0001e-4 で、この形状では backend 差を自己再実行より大きく分離できていない。

NCCL world size 2 は、reference と fused のそれぞれで accumulation 1 と 2 を 1 logical step 実行した。`parameter_ranks_agree` は true で、forward 回数は上の表と一致した。長い resume は CPU で確認し、NCCL の長い resume はしていない。

## 18. 回帰テスト

変更前、`DF_RFFT_LIBRARY` をローカル再ビルドの `build/df-rfft/release/libdf_rfft.so` にし、`CUDA_VISIBLE_DEVICES=""` で実行した。

- `tests/experiments`: 181、Failure 0、Error 0、Skip 1
- `tests/training`: 297、Failure 0、Error 0、Skip 12
- `tests/reproduction`: 135、Failure 0、Error 0、Skip 2

変更後も同じ環境変数で実行した。

- `tests/experiments`: 199、Failure 0、Error 0、Skip 1。増分 18 は Phase 4-H の CPU 試験である。Skip 1 は CUDA を隠した Phase 2-C の module skip のままである。
- `tests/training`: 297、Failure 0、Error 0、Skip 12
- `tests/reproduction`: 135、Failure 0、Error 0、Skip 2

新規の Failure、Error、Skip は無い。追跡中の `libdf_rfft.so` はこの glibc 2.31 では読めない。その問題は Phase 4-G から残っている。

## 19. 未検証・既知の差分

- 公式素材の `semambapp.gsr.official@1` は blocked である。代替素材を公式 GSR と呼ばない。
- 3 秒、batch 8、1 forward は未実行である。過去の OOM 記録を、この Phase の成功で上書きしない。
- CUDA は決定的 backward が使えず、bit 一致ではない。差は現行 Task の自己再実行と同じ桁である。
- CPU の `scan_backend: fused` は reference kernel を実行する。これは既存ネットワークの動作で、fused の検証とは分けて記録する。
- `.venvs/se` だけでは fused kernel は import できない。
- AMP FP16、BF16、FSDP、sharded checkpoint は未実装である。
- NCCL の複数 epoch resume は未実施である。

## 20. Phase 4-I への引き継ぎ

次は DeepFilterNet3 と SEMamba++ を含む 6 モデルの統合検証である。SEMamba++ の Trainer は他の GAN Trainer を継承していない。`group_size` と accumulation を一致させる。`best_generator.pt` は作らない。fused を reference の成功として数えない。公式 snapshot は blocked のままにする。
