# 学習基盤の確認

2026-10-31、ホスト `turing`。Python 3.13.9、PyTorch 2.6.0+cu124、CUDA 12.4、NumPy 2.5.4、pesq 0.0.4。GPU は Quadro RTX 5000。使用した装置は `nvidia-smi` で使用メモリ 10 MiB だった物理 GPU 1 と 2 である。物理 GPU 0 は 9522 MiB 使用中だったので使っていない。

コマンドは `.venvs/lab/bin/python -m unittest discover -s tests/training -v`。30 件、失敗 0。これは CPU である。CUDA の追試は下に分けて書く。

## CPU（単体テスト）

Seed はテスト内の値。再現実験の代表は seed 7、再構成損失、`global_batch_size=4`、`micro_batch_size_per_gpu=2`（micro 2、accumulation 2）、segment 160、`n_fft=32`、`dense_channel=4`、`num_tsconformers=1`。

| 確認 | 結果 |
|---|---|
| 同じ seed の Generator 初期化が一致し、seed を変えると変わる | PASS |
| 同じ seed の Discriminator 初期化が一致し、seed を変えると変わる | PASS |
| 同じ seed のサンプル順が一致し、別 seed で変わる | PASS |
| 同じレシピと seed の入出力波形が一致する | PASS |
| `dry_clean` と `reverberant_clean` の教師が異なる | PASS |
| `reference: none` を拒否する | PASS |
| 名前付きセットの `case_id` がレシピハッシュである | PASS |
| `data/db/rebuild-a.duckdb` を読み、11,108 ケースのうち LibriSpeech test-clean と CHiME3 et05 のセットを学習に使わず拒否する | PASS。セットは 6 件（`libri_demand_snr0`、`libri_testclean_full_smoke`、`chime3_et05_real_ch0`、`chime3_et05_real_ch5`、`chime3_et05_simu_ch5`、`chime3_tr05_org`） |
| 同じ条件の 2 回の学習で Generator・Optimizer・Scheduler が一致する | PASS |
| 1 step で保存し、3 step まで再開した結果が、中断しない 3 step と一致する | PASS |
| 再構成損失の 20 step で時間損失が初期値より下がる（1.226830 → 0.569280） | PASS。これは公式損失ではない |
| 公式の Discriminator 更新のあと Generator を更新し、Generator の勾配が有限で、Discriminator の更新が Generator の重みを変えない | PASS |
| 学習後の `generator.pt` を `load_enhancer` で読み、長さ 1000 の推論が有限値で完了する | PASS |
| Global 64 / GPU 4 / micro 上限 8 → micro 8、accumulation 2。Global 48 / GPU 3 / 上限 6 → micro 4、accumulation 4。Global 50 / GPU 4 はエラー | PASS |
| 線形モデルで、full batch と勾配蓄積の勾配が `rtol=1e-5`, `atol=1e-6` で一致する | PASS |
| gloo の 2 プロセスで、micro-batch 同期と蓄積のどちらも全バッチ勾配と一致する | PASS |
| `ExponentialLR` が 2 回の Optimizer 更新で `0.001 * 0.5**2` になる（micro-batch 数ではない） | PASS |
| 学習 STFT の振幅が `torch.stft(center=True, pad_mode="reflect")` と一致する | PASS |

再開実験の Generator state dict の SHA-256 は `a851db743971f0a4c201e07bc3cd06a6c86538c31530b5924b501d8348338e9e`。2 回目の連続学習も同じハッシュだった。

## CUDA（単体テストの外）

`CUDA_VISIBLE_DEVICES=1`、`CUBLAS_WORKSPACE_CONFIG=:4096:8`、`deterministic: true`。

| 確認 | 結果 |
|---|---|
| 再構成損失を 2 step、2 回実行し Generator が一致する | PASS |
| 公式損失を 1 step、2 回実行し Generator と Discriminator が一致する。`n_fft=64`、hop 16、segment 512、batch 2 | PASS |
| 物理 GPU 1 と 2、`torch.distributed.run --nproc_per_node=2`。global 4、micro 2、accumulation 1。再構成 2 step を 2 回実行し、rank 0 の `generator.pt` が一致する。step 1 の時間損失はどちらも 1.010923、step 2 は 0.999492 | PASS |
| 同じ 2 GPU で公式損失を 1 step（global 2、micro 1、accumulation 1、`n_fft=64`、segment 512）。2 回とも Discriminator 損失 0.437902、Generator 損失 1.743332。Generator と Discriminator の state dict が一致する | PASS |

2 GPU 実験の Generator SHA-256 は `4edd7bbf232fe1a6d5f9ea0d135d0ddf3f9ce95cd76e9e2590adfc908569a26a`。このファイルは `/tmp` にあり、リポジトリには残していない。

`AdaptiveMaxPool2d` の CUDA 逆伝播と `reflection_pad1d` の CUDA 逆伝播は、決定的アルゴリズムを有効にすると実行できない。順伝播の値が公式と同じになる `amax` と index reflect に置き換えたうえで、上の一致を確認した。非決定的カーネルへ黙って戻してはいない。

VoiceBank-DEMAND での長時間学習と、公式チェックポイントとの学習曲線の比較は行っていない。

## 公式 1 step（L3）と他モデル

2026-10-31、同じホスト `turing`、同じ lab 環境。CPU。公式 MP-SENet と独立実装を、`dense_channel=4`、`n_fft=64`、hop 16、長さ 2048、batch 2、eval（Dropout なし）で 1 step 比べた。

| 項目 | 最大絶対誤差 |
|---|---|
| STFT 振幅 | 0 |
| Generator 振幅 | 0 |
| Generator 位相 | 0 |
| Discriminator 出力 | 0 |
| 損失合計 | 0 |
| Generator 勾配 | 6.43e-8 |
| Discriminator 勾配 | 0 |
| AdamW 更新後の重み | 4.05e-4 |

重みの差は、6e-8 の勾配差が AdamW の epsilon 1e-8 を通ったあとである。

同じ日、物理 GPU 0（使用メモリ 18 MiB）で公式損失を 1 step、2 回実行した。損失はどちらも Generator 1.705775、Discriminator 0.437144。Generator の SHA-256 はどちらも `3f27a11466229721003b99d14e838f7a4a48d891e61468b02f307b0e1dba6f9e`。物理 GPU 0 と 1（どちらも使用メモリ 13–18 MiB）の `torchrun --nproc_per_node=2` を 2 回実行し、rank 0 の SHA-256 はどちらも `7ff01c656bf9bcd87852ddba9476a9b1402b21c26ef5ea2caf482e21366d3078`。Generator 損失はどちらも 1.704373。ファイルは `/tmp` にあり、リポジトリには残していない。

TF-GridNet、SEMamba++、MossFormerGAN、DeepFilterNet3 の 1 step は `tests/training/test_model_tasks.py` が CPU で確認した。レベルは [OTHER_MODELS.md](OTHER_MODELS.md)。SEMamba++ の `A_log` は未初期化だと selective scan が非有限になる。S6 の `A_i = i` で初期化する。CUDA の `mamba_ssm` 融合カーネルとの学習ステップ比較はしていない。

## MP-SENet の AdamW 更新差（2026-10-31）

同じ縮小条件（`dense_channel=4`、`n_fft=64`、長さ 2048、batch 2、eval）で公式モジュールと 1 step を比較した。STFT、Generator、Discriminator、損失の最大絶対誤差は 0。Discriminator 勾配の最大絶対誤差は 0。Generator 勾配の最大絶対誤差は 6.43e-8。AdamW（lr `5e-4`、betas `(0.8, 0.99)`、weight decay `0.01`、epsilon `1e-8`、初期モーメント 0）のあとの重みの最大絶対誤差は 4.052668809890747e-4。

初回ステップは `param * (1 - lr * weight_decay) - lr * grad / (|grad| + eps)` である。独立実装の更新後重みは、この式と最大絶対誤差 1e-6 未満で一致した。二つのモデルの重み差も、同じ式を二つの勾配に適用した差と 1e-6 未満で一致した。weight decay は同じ初期重みに掛かるので、重み差には残らない。

最大の重み差が出た要素は `dense_encoder.dense_block.dense_block.3.1.bias` の 1 要素である。独立実装の勾配は -8.90e-9、公式の勾配は 5.14e-9 で、符号は逆、絶対差は 1.40e-8 である。どちらも epsilon より小さく、`grad / (|grad| + eps)` は符号付きで約 -0.47 と 0.34 になる。その差に lr `5e-4` を掛けると 4.05e-4 になる。勾配の絶対値が 1e-6 を超える 4930 要素では、符号の不一致は 0 だった。したがって、重みの最大差は「勾配の最大誤差 6e-8 がそのまま増幅された」のではなく、epsilon 未満で符号が違う要素が初回 AdamW で O(lr) まで広がった結果である。学習式は変えていない。

## DeepFilterNet3、TF-GridNet（2026-10-31）

ホスト `turing`。lab 環境。CPU の `tests/training` は 47 件、`tests/se` は 28 件、`tests/evaluation` は 81 件、`tests/asr` は 5 件。いずれも失敗 0。

| 確認 | 結果 |
|---|---|
| DeepFilterNet3 公式幅（`conv_ch=64`、48 kHz、長さ 4800）の 1 step。損失と勾配は有限。checkpoint を `load_enhancer` で読み、同じ発話の 2 回は一致 | PASS（CPU） |
| 同じ公式幅を物理 GPU 0（使用メモリ 15 MiB）で 2 回。`deterministic` と `CUBLAS_WORKSPACE_CONFIG=:4096:8`。損失はどちらも 38.875637。重みの SHA-256 はどちらも `89595b78d09bd7ec36be47e2e68576f3f2de7f0cee57f0ec35c3a5435d477d46`。学習後重みの推論は 2 回一致 | PASS |
| 物理 GPU 0 と 1 の `torchrun --nproc_per_node=2`。両 rank に同じ 1 発話を置いた 1 step を 2 回。rank 0 の損失と SHA-256 は単一 GPU と同じ | PASS。これは分割バッチの勾配一致ではない。両 rank の入力が同じなので、平均勾配は単一 GPU と一致する |
| TF-GridNet `ms-snsd` の公式幅（4 層、`n_fft=512`、stride 256、`emb_dim=32`、LSTM 128）で 1 step し、`load_enhancer` で長さ 1024 を処理 | PASS（CPU） |


## 学習挙動の同等性（2026-10-31）

判定は **GO**。MP-SENet と SEMamba++ の公式幅で、同じ重み・同じ波形に対する forward、損失、勾配方向、1 step の更新が揃い、20 step の損失が同じ水準を追う。参照ツリーは学習コードから import していない。SEMamba++ のソースはコピーしていない。

事前に置いた許容値は、forward と損失成分 `atol=1e-5`、パラメータ勾配 `atol=1e-4` かつ `|grad|` の最大が `1e-5` を超えるテンソルの cosine `>= 0.999`、である。`|grad|` がそれより小さい要素の cosine は判定に使わない。測定後にこの値は緩めていない。NaN や Inf は 0 にしていない。

### 環境

| 項目 | 値 |
|---|---|
| リポジトリ HEAD | `a81c14cac67e62c37ef53133d86435b39ce67f45`（`dev`）。この検証の差分は未 commit |
| MP-SENet 公式 | `src/se/official/mp-senet` commit `89932cfe90d1dacb8e170e4a331d762462c21792` |
| SEMamba++ 公式 | `src/se/official/semambapp` commit `cd369c89a8106072c077b468417b765b227afd74` |
| lab | Python 3.13.9、PyTorch 2.6.0+cu124、CUDA 12.4、`nnAudio` 0.3.3 |
| semamba | Python 3.10.20、PyTorch 2.6.0+cu124。`mamba_ssm` と `causal_conv1d` の cubin は sm_75 のみ |
| 決定性 | `CUBLAS_WORKSPACE_CONFIG=:4096:8` |

GPU の使い分け:

- MP-SENet の公式幅は shannon の物理 GPU 5（A100 80GB、使用メモリ 13 MiB、`CUDA_VISIBLE_DEVICES=5` なのでプロセス内は `cuda:0`）。
- SEMamba++ の融合 Mamba と比較学習は turing の物理 GPU 0（Quadro RTX 5000、使用メモリ 15 MiB）。sm_75 の cubin は A100 でロードできない。
- Global batch の 2 GPU は turing の物理 GPU 1 と 2（どちらも使用メモリ 10 MiB）。

### MP-SENet

公式幅は `dense_channel=64`、`num_tsconformers=4`、`n_fft=400`、hop 100、win 400。パラメータ数 2,263,372。入力は batch 2、長さ 8000、seed 11。初期化 seed は 7。Optimizer は AdamW、lr `5e-4`、betas `(0.8, 0.99)`、weight decay `0.01`。

CUDA 上の Hann 窓をその場で作ると、CPU で作ってから移した窓と約 1e-7 違い、位相が 2π 反転して Generator 出力が崩れた。学習 STFT と iSTFT は、公式と同じく CPU で窓を作ってからデバイスへ移す。修正後の公式幅:

| 項目 | eval 最大絶対誤差 | train 最大絶対誤差 |
|---|---:|---:|
| STFT 振幅 | 0 | 0 |
| Generator 振幅 | 0 | 0 |
| Generator 位相 | 0 | 0 |
| Discriminator | 0 | 0 |
| 波形 | 0 | 0 |
| 損失 | 0 | 0 |

Generator 勾配の最大絶対誤差は 3.36e-5（`dense_encoder.dense_conv_1.0.weight`）。`|grad|` 最大が 1e-5 を超えるテンソルの最小 cosine は 0.9999998。それより小さいテンソルには cosine が負のものがある。Discriminator 勾配の最大絶対誤差は 9.39e-8。大きな勾配の cosine は 1 に近い。

1 step 後の重みの最大絶対誤差は 9.86e-4。同じ入力の次の eval forward では、損失差 2.38e-7、波形の最大絶対誤差 7.41e-3 である。位相の最大差 6.28 は 2π の反転で、損失にはほとんど残らない。

同じ入力で 20 step した公式側の損失は 2.337 から 0.343 へ単調に近い形で下がった。独立実装との損失差は最初の 2 step が 2e-7、その後 0.006 から 0.015 の範囲に留まる。20 step 目の損失 0.343 に対する差 0.0056 は約 1.6% である。最終重みの最大絶対誤差は 9.85e-3。学習は有限のままである。

PESQ / STOI はこのガウス雑音の比較バッチでは計測していない（NOT TESTED）。0.5 秒の非音声に対する PESQ を一致の証拠にはしていない。検証 API は短い入力で PESQ を出せないとき `na` のままにする。

### SEMamba++

参照 commit の Discriminator を、同じモジュール名で独立に実装した。`nnAudio` 0.3.3 の `CQT2010v2`（`sr=32000`、`output_format=Complex`、`pad_mode=constant`）を使い、ホップ `(512, 256, 256)`、8 octaves、1 octave あたり `(12, 24, 36)` bin、チャネル 128 である。設定辞書の幅 32 はコンストラクタが層へ渡していない。STFT Discriminator の解像度は `(1024, 120, 600)`、`(2048, 240, 1200)`、`(512, 50, 240)`。パラメータ数は CQT 4,453,974、STFT 280,902 で、公式と一致し、state dict のキーも一致する。長さ 8192、batch 2 の eval と train で、スコアと中間特徴の最大絶対誤差は 0 である。

Mamba の比較は `d_model=48`、`d_state=16`、`d_conv=4`、`expand=2`、長さ 64、batch 2、物理 GPU 0。独立実装は左パディングの因果畳み込みと逐次 scan で、公式は融合カーネルである。

| 項目 | 最大絶対誤差 | cosine |
|---|---:|---:|
| 出力 | 3.73e-8 | 1 |
| 入力勾配 | 5.22e-8 | 1.000 |
| `A_log` 勾配 | 5.59e-9 | 1.000 |
| `in_proj.weight` 勾配 | 1.19e-6 | 1 |
| `conv1d.weight` 勾配 | 4.77e-7 | 1 |
| `out_proj.weight` 勾配 | 7.15e-7 | 1 |

`A_log` の `A_i = i` 初期化は変えていない。

Generator の state dict は公式幅（`hid_feature=48`、4 block、2,655,521 パラメータ）でキーが一致する。同じ重みでの eval は、振幅の最大絶対誤差 1.19e-6、位相 6.74e-5、入力勾配 2.89e-6、パラメータ勾配の最大 2.30e-5。cosine はいずれも 1 に近い。

学習ステップの損失は、特徴整合をマップ平均の和の 2 倍（スケール数では割らない）、mel を Slaney フィルタの log10 L1 の 4 スケール和、位相を最終ビンに `-1` を含む差分行列、整合 STFT を `1e-10` 入りの `atan2` にした。同じガウス波形、長さ 8000、batch 1、AdamW lr `2e-4`、betas `(0.8, 0.99)`、weight decay `0.01`。

公式幅の step 0:

| 項目 | 差 |
|---|---:|
| Generator 損失 | 0（公式値 50.308） |
| Discriminator 損失 | 0（公式値 5.749） |
| 振幅、位相、mel、敵対 | 0 |
| 波形 | 最大絶対誤差 8.3e-8、cosine 1 |
| CQT 勾配 | 最大 3.7e-9、大きな勾配の cosine 1.000 |
| STFT Discriminator 勾配 | 最大 1.2e-7、cosine 1.000 |
| Generator 勾配 | 最大 6.4e-3（`dense_encoder.dense_conv_1.0.weight`）。`|grad|>1e-5` の最小 cosine は 0.9999996 |
| Generator 重み | 最大 4.00e-4 |
| Discriminator 重み | 最大 3.2e-7 |

Generator 勾配のピーク差 6.4e-3 は、単一 Mamba ブロックに置いた `atol=1e-4` を超える。方向は cosine 0.9999996 で揃っており、4 ブロックの float32 累積として扱った。判定のあとで許容値は変えていない。

20 step の公式 Generator 損失は 50.3 から 62.6 を経て 52.0 に戻る。独立実装との差の最大は 0.104（step 13）で、損失 53 に対して約 0.2% である。Discriminator 損失は 5.75 から 1.5 前後へ下がり、差の最大は 0.052 である。20 step 目の波形 cosine は 0.972、最大絶対誤差は 0.605 である。損失の水準は追従し、重みの完全一致はしていない。`|grad|` が 1e-5 を超えるテンソルの最小 cosine は step 0 で 1 に近く、step 19 では -0.30 まで落ちる。これは初回の 4e-4 の重み差が GAN の更新で広がったあとであり、step 0 の構造差ではない。

Slaney mel と公式 `MultiScaleMelSpectrogramLoss` の同じ波形での差は 1.19e-7 である。

### Global batch と checkpoint

構成 A（1 GPU、micro 4、蓄積 1）を全バッチの基準にし、構成 B（2 GPU、micro 2、蓄積 1）と構成 C（2 GPU、micro 1、蓄積 2）を比べた。各 rank の入力は異なる。モデルは `Linear-ReLU-Linear` で、`src.training.core.trainer.backward` を使う。事前の許容は `rtol=1e-4`、`atol=1e-5`。B と C の勾配の最大絶対誤差はどちらも 2.98e-8。PASS。これは SE モデルの分割バッチではない。MP-SENet の CPU 上の蓄積と gloo は既存の `tests/training/test_accumulation.py` にある。

checkpoint の保存、Optimizer を含む再開、`load_enhancer` による推論は、この変更のあと `.venvs/lab` の `tests/training` 47 件、`tests/se` 28 件が失敗 0 で通った。SEMamba++ の 1 step テストは CQT Discriminator を含み、推論用 `generator.pt` を `load_enhancer` で読む。

### 単体テスト

| スイート | 結果 |
|---|---|
| `tests/training` | 47 件、失敗 0 |
| `tests/se` | 28 件、失敗 0 |
| `tests/evaluation` | 81 件、失敗 0 |
| `tests/asr` | 5 件、失敗 0 |

公式比較は次のスクリプトで、通常の `unittest discover` には入らない。公式ツリーが無い環境ではこれらのスクリプトは動かず、その結果を PASS には数えない。

```bash
# shannon 物理 GPU 5 が空いているとき
ssh shannon 'cd /net/spring/work/seki/exp/audio-db && export CUDA_VISIBLE_DEVICES=5 CUBLAS_WORKSPACE_CONFIG=:4096:8 && .venvs/lab/bin/python tests/training/reference/compare_mp_senet.py'

# turing。融合 Mamba は sm_75 が必要
export CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
.venvs/semamba/bin/python tests/training/reference/compare_semamba_disc.py
.venvs/semamba/bin/python tests/training/reference/compare_semamba_mamba.py
.venvs/semamba/bin/python tests/training/reference/compare_semamba_generator.py
.venvs/semamba/bin/python tests/training/reference/compare_semamba_step.py official

# turing 物理 GPU 1 と 2
export CUDA_VISIBLE_DEVICES=1,2 CUBLAS_WORKSPACE_CONFIG=:4096:8
.venvs/lab/bin/python -m torch.distributed.run --nproc_per_node=2 tests/training/reference/compare_global_batch.py
```

VoiceBank-DEMAND 全件の学習と論文スコアの再現はしていない。commit と push はしていない。

## 公式学習順と 4 モデルの短時間比較（2026-11-01）

独立実装の作業ツリーは `dev` の `1cd910f8d4743f5cd4063e066eae7c13246f4287` に、この節の未 commit 差分を足したものである。許容値は測定前に固定した。forward と損失成分は `atol=1e-5`。パラメータ勾配は `atol=1e-4`、かつ `|grad|` の最大が `1e-5` を超えるテンソルの cosine は `>= 0.999`。測定後にこの値は緩めていない。NaN は 0 にしていない。

音声は `data/db/training.duckdb` の `libri_demand_smoke_train`（8）と `libri_demand_smoke_valid`（4）を、学習と同じレシピ再生と切り出しでテンソル化した `data/results/training/official-compare-20261101/speech_cache.pt` である。`tensor` の torchaudio は FLAC を読めないため、再生は `turing` で行い、比較プロセスは同じテンソルを読む。検証発話は学習に混ぜていない。

`CUBLAS_WORKSPACE_CONFIG=:4096:8`。`tensor` は物理 GPU 0、6、7（いずれも A100 40GB、使用メモリ 13 MiB、プロセス内は `cuda:0`）。PyTorch 2.6.0+cu124。SEMamba++ だけは `turing` の物理 GPU 0（Quadro RTX 5000、sm_75、使用メモリ 15 MiB）。融合カーネルは `mamba-ssm==2.3.2.post1` と `causal-conv1d==1.7.0` で、cubin は sm_75 だけである。A100 ではロードできない。PyTorch の scan を公式カーネルの代替としては使っていない。

### MP-SENet

公式 commit `89932cfe90d1dacb8e170e4a331d762462c21792`。公式幅、AdamW lr `5e-4`、betas `(0.8, 0.99)`、weight decay `0.01`。更新は公式 `train.py` と同じく、Discriminator（実ペアは 1、偽ペアは正規化 PESQ、強調振幅は detach）のあと Generator である。ExponentialLR はこの 20 更新では進めない。比較バッチは 2。公式 `config.json` の batch は 4。

ガウス入力の eval と train で、STFT、振幅、位相、Discriminator、波形、損失の各成分の差は 0 だった。実音声の step 0 は、波形差 0、PESQ 目標差 0、Generator 損失差 0、Discriminator 損失差 0、Generator 勾配の最大絶対誤差 `3.50e-5`、大きな勾配の cosine `0.99999988`。更新後重みの最大絶対誤差は `9.95e-4` で、以前記録した初回 AdamW の epsilon 増幅と同じ形である。

20 step 後の Generator 損失は公式 1.018、独立実装 1.000（差 0.018）。検証発話の波形 cosine は 0.981、PESQ は 1.178 と 1.180、STOI は 0.889 と 0.895。step 19 の大きな勾配の cosine は -0.56 まで落ちる。step 0 の構造差ではなく、epsilon 未満の勾配差が GAN 更新で広がったあとである。本番 `MPSENetTask` の 1 step は、手動の独立実装に対して Discriminator 重みの最大差 `1.79e-7`、Generator は同じ epsilon 増幅で `9.96e-4` だった。

### SEMamba++

公式 commit `cd369c89a8106072c077b468417b765b227afd74`。公式幅、AdamW lr `2e-4`、betas `(0.8, 0.99)`、weight decay `0.01`。融合 Mamba を使った。

ガウス波形、長さ 8000 での切り分け:

| 比較 | 最大絶対誤差 | 備考 |
|---|---:|---|
| 4 つの Mamba block 出力 | `8.58e-6` | cosine 1 |
| 波形 | `6.91e-6` | cosine 1 |
| スペクトルだけの損失 | 0 | 勾配最大 `7.95e-5`（許容内）、大きな勾配の cosine 1 |
| mel だけの損失 | 0 | 勾配最大 `1.56e-6` |
| 敵対と特徴整合 | 0 | 勾配最大 `5.88e-4`。最大要素の `|grad|` は 5.90、相対誤差 `1.0e-4`、その cosine は 1 |
| 全損失 | 0 | 勾配最大 `4.32e-4`（`dense_encoder.dense_conv_1.0.weight`）。相対誤差 `7.5e-5`、cosine 1 |

事前の `atol=1e-4` は、全損失と敵対項の絶対値では超える。超える要素は勾配そのものが大きい先頭畳み込みで、方向は cosine 1 である。`|grad|` が `1e-8` 前後の bias には cosine が負のものがある。これは公式融合カーネルと逐次 scan の float32 累積で、損失式の取り違えではない。CQT / MRD を外したスペクトル損失は許容内に収まる。

実音声 8000 サンプル、20 step。step 0 の Generator 損失差は `1.91e-6`、波形差 `3.48e-7`、勾配最大 `5.89e-3`、`|grad|>1e-5` の最小 cosine は 0.9981（しきい値 0.999 をわずかに下回る）。最小 cosine -0.47 のテンソルは `|grad|` 最大 `7.7e-9` の bias である。公式 Generator 損失は 14.6 から始まり、step 19 は 15.19、独立実装との差は 0.110。波形 cosine は 0.810、重みの最大差は `4.36e-3`。固定検証発話の PESQ / STOI は測っていない（NOT TESTED）。`.venvs/semamba` に pesq が無く、学習後の波形も保存していない。

### TF-GridNet ms-snsd

ESPnet commit `0d136c5456d403cb2c0a4c13e76e8136e32e9b7a`。`train_enh_tfgridnet.yaml` の ms-snsd。4 層、`n_fft=512`、stride 256、`emb_dim=32`、LSTM 128。Adam lr `1e-3`、betas `(0.9, 0.999)`、eps `1e-8`、weight decay 0。YAML は `grad_clip` を書かないので、ESPnet の既定 5.0 を使った。SI-SNR の重みは 0。ReduceLROnPlateau はこの 20 更新では進めない。比較は batch 1、長さ 16000。公式 chunk は 48000、batch 4。cuDNN は deterministic、TF32 はオフ。`torch.use_deterministic_algorithms` は LSTM の一部カーネルが拒否するため使っていない。

同じ波形に対する `mr_l1_tfd` の差は 0。step 0 から step 2 まで、公式と独立実装の損失差は 0、波形差も 0。step 0 の勾配最大絶対誤差は `5.62e-3`。別の 1 step では最大が `conv.1.bias` の `8.30e-3` で、その `|grad|` は 6812、相対誤差は `1.2e-6`、cosine は 1。cosine が 0.12 まで落ちるのは `|grad|` が `2.3e-5` の `beta` で、`1e-5` の判定線のすぐ上である。本番 `TFGridNetTask` と手動更新の重み差は 0。`load_enhancer` で読み直した波形差も 0。

step 19 の損失は公式 42581、独立実装 41373（差 1208、約 2.8%）。検証波形の cosine は 0.929、PESQ は 1.0347 と 1.0351、STOI は 0.666 と 0.658。同じ公式波形を両方の損失に入れると差は最後まで 0 なので、損失定義は一致したまま、初回の相対誤差 `1e-6` が更新で広がっている。`urgent24` の学習比較はしていない（NOT TESTED）。

### DeepFilterNet3

公式 commit `d375b2d8309e0935d165700c91da9de862a99c31`。`config.ini` の SHA-256 は `415eb925d44990d938fb739f514aa3662c1ec0ea836cff044fa1291b82cb4290`。公式幅 `conv_ch=64`、48 kHz、パラメータ数 2,135,484。16 kHz のスモークは `torchaudio.functional.resample` の既定 sinc（低域通過つき）で 48 kHz にしてから両方へ渡した。切り出しは 8000 サンプル（16 kHz で 0.5 秒、48 kHz で 24000 サンプル）。公式学習は元から 48 kHz である。

`steps_per_epoch=8`、`epochs=120` の出荷 cosine と、タスクの `deepfilter_cosine_schedules` は、最初の 20 値で学習率も weight decay も差 0。step 0 の学習率は `1e-4`、warmup 最終値は `1e-3`、weight decay の初期値は `1e-12`。AdamW は betas `(0.9, 0.999)`、`amsgrad=True`、勾配クリップ 1。

公式 `MultiResSpecLoss` と独立実装の差は 0。local SNR 目標の差は 0。公式 `DfNet` に同じ state dict を `strict=True` で読むと、同じ特徴での forward 最大絶対誤差は 0。`libdf` は入っていないので、公式の解析 STFT と ERB は実行していない（NOT TESTED）。

step 0 の損失差は `7.63e-6`、勾配最大 `3.08e-5`、大きな勾配の cosine `0.99999988`、重み差 `1.75e-4`。step 19 の損失は 18.493 と 18.475。スペクトル式の差は `1.34e-5`、local SNR 差は 0。検証波形の cosine は 0.9988、最大絶対誤差は `9.37e-3`。PESQ は 1.0422 と 1.0425、STOI は 0.198 と 0.200。STOI の絶対値は 0.5 秒を 16 kHz に戻して測ったための低さで、二つのモデルの差ではない。学習後の checkpoint は `load_enhancer` で 8000 サンプルの推論が有限値で完了した。step 19 で `|grad|>1e-5` の最小 cosine が -0.92 になるパラメータは `df_dec.df_gru.gru.weight_hh_l1` で、step 0 では揃っている。

### 再現

```bash
# turing で波形を作ったあと
# tensor の空いている GPU
export CUDA_VISIBLE_DEVICES=<free> CUBLAS_WORKSPACE_CONFIG=:4096:8
.venvs/lab/bin/python tests/training/reference/compare_mp_senet.py
.venvs/lab/bin/python tests/training/reference/compare_tf_gridnet.py
.venvs/lab/bin/python tests/training/reference/compare_deepfilternet.py

# turing。sm_75 の融合カーネル
export CUDA_VISIBLE_DEVICES=<free> CUBLAS_WORKSPACE_CONFIG=:4096:8
.venvs/semamba/bin/python tests/training/reference/compare_semamba_step.py diagnose
.venvs/semamba/bin/python tests/training/reference/compare_semamba_step.py speech
```

結果 JSON は `data/results/training/official-compare-20261101/`。MossFormerGAN の NISQA Discriminator は実装も比較もしていない。論文スコアの完全再現は対象外である。

## SEMamba++ の短時間学習と品質（2026-11-01）

判定は **PARTIAL**。事前に固定した許容は、発話ごとの絶対差で PESQ `0.10`、STOI `0.02`、SI-SDR `1.0 dB` である。測定後にこの値は変えていない。判定点は更新 20 回後の検証 4 発話である。そこでは PESQ と STOI は帯の内、SI-SDR は 4 発話とも帯の外だった。100 更新では SI-SDR の差は帯の内に戻ったが、これは事前の判定点ではなく、学習済みモデルの最終品質でもない。

比較したのは、同じ初期重みから同じ 8,000 サンプルの学習順で 20 回、および別プロセスで 100 回更新した公式幅である。検証音声は学習に使った 0.5 秒の切り出しではない。`libri_demand_smoke_valid` の 4 発話を、元の clean と雑音から 3〜10 秒に伸ばしたレシピで再生した。長さは 5.855 秒（SNR 0 dB）、10 秒（5 dB）、10 秒（10 dB）、4.53 秒（15 dB）である。レシピ JSON は実験ディレクトリにあり、DuckDB には書いていない。本番 `resource.duckdb` と `official-compare-20261101/` は変更していない。

### 条件

| 項目 | 値 |
|---|---|
| 初期化 | seed 1234。構築後に公式の state dict を独立実装へコピーし、`torch.equal` とソートした SHA-256 が一致 |
| 学習 | 前回の `semamba_waves.pt` の 8 区間を順に循環。AdamW、lr `2e-4`、betas `(0.8, 0.99)`、weight decay `0.01`。勾配クリップはしない |
| Scheduler | 公式と同じ ExponentialLR、gamma `0.99`。コンストラクタ後の `last_epoch` は 0 で、lr は `2e-4` のまま。20 回でも 100 回でも `step()` は呼ばない |
| Dropout | モジュール数 0。更新中は両方 `train()`、品質の forward は両方 `eval()` |
| 入力レベル | レシピ再生のあと、学習と同じエネルギー正規化。クリップしない。再サンプリングしない。16 kHz |
| カーネル | 公式 Generator の Mamba 24 個は `use_fast_path=True`。`mamba_inner_fn` と `selective_scan_cuda`。`mamba-ssm==2.3.2.post1`、`causal-conv1d==1.7.0` |
| 指標 | `.venvs/lab` の pesq `0.0.4`（wideband）、pystoi `0.4.1`、リポジトリの SI-SDR。`.venvs/semamba` のダミー `pesq` は損失モジュールの import 用で、値は読んでいない |
| GPU | `turing`。Phase A は物理 GPU 0、Phase B は物理 GPU 1。どちらも Quadro RTX 5000、開始時の使用メモリは 15 MiB と 10 MiB。ピークは約 8368 MiB。`CUBLAS_WORKSPACE_CONFIG=:4096:8`。TF32 はオフ。融合カーネルを止める `use_deterministic_algorithms` は使っていない |

Generator のパラメータ数は 2,655,521。コピー前の Generator ハッシュは公式と独立実装で違った。CQT と MRD は seed 1234 の時点で一致した。コピー後は 3 つとも一致した。

公式 iSTFT は 5.855 秒と 4.53 秒の 2 発話で独立実装より 80 サンプル短い。指標は重なった区間で計算し、`trimmed_samples` に 80 を残している。長さが一致する 10 秒の 2 発話でも SI-SDR の差は同程度なので、この 80 サンプルが差の原因ではない。

### 更新 20 回（判定点）

Phase A。4 発話の平均。noisy は強調前。

| 指標 | noisy | 公式 | 独立 | 公式−独立の絶対値 |
|---|---:|---:|---:|---:|
| PESQ | 1.674 | 1.043 | 1.045 | 0.0026 |
| STOI | 0.942 | 0.584 | 0.592 | 0.0083 |
| SI-SDR (dB) | 7.51 | -25.31 | -27.34 | 2.03 |
| 波形 cosine | | | | 0.906 |

発話ごとの SI-SDR 差は 1.82、1.72、2.11、2.47 dB で、許容 1.0 dB を全部超えた。PESQ の発話ごと差は最大 0.007、STOI は最大 0.009 で、こちらの許容内だった。どちらも noisy より悪い。PESQ は約 0.63 下がり、STOI は約 0.35 下がり、SI-SDR は約 33 dB 下がる。20 更新は強調を獲得していない。

### 100 更新での推移

Phase B は同じ seed の別プロセスで 100 更新した。20 更新時点の学習波形 cosine は Phase A の 0.903 に対して 0.898 で、ビット一致ではない。融合カーネルの非決定性と、再開実験の重み差 `6.3e-5` と同じ種類のずれである。

検証 4 発話の平均絶対差:

| 更新後 | PESQ | STOI | SI-SDR (dB) | 波形 cosine | 公式 SI-SDR | 独立 SI-SDR |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.010 | 0.000 | 0.005 | 1.000 | -39.48 | -39.48 |
| 20 | 0.148 | 0.005 | 1.38 | 0.891 | -25.40 | -26.78 |
| 50 | 0.003 | 0.008 | 0.73 | 0.883 | -15.09 | -15.82 |
| 100 | 0.004 | 0.004 | 0.59 | 0.849 | -11.61 | -12.20 |

SI-SDR の実装間差は 20 更新で広がり、50 更新と 100 更新では 1 dB の帯に入った。波形 cosine はその間も 0.89 から 0.85 へ緩やかに下がった。学習セグメントの cosine の最小は 0.860（forward 93）、Generator 損失差の最大は 0.184（forward 78）である。100 更新後も SI-SDR は noisy の +7.5 dB に対して約 -12 dB で、強調としての改善にはなっていない。この 100 更新は収束も最終性能の一致も示さない。

更新 20 回の PESQ 平均差 0.148 は、1 発話の全区間 PESQ 1.538 が押し上げている。同じ波形の 1 秒区間は平均 1.061 だった。Phase A の更新 10 回にも、全区間 PESQ 4.407 に対して 1 秒区間が 1.03〜1.09 の発話がある。SI-SDR は約 -39 dB、clean との cosine は 0.01 なので、この全区間 PESQ は品質の証拠に使っていない。判定点の Phase A 更新 20 回は、全区間と 1 秒区間がどちらも約 1.04 で揃っていた。

### cosine 0.810 との対応

前回の 0.810 は、seed を固定する前の実行で、更新 19 回後の 0.5 秒学習区間の波形 cosine である。その波形は残っていないので、同じペアの PESQ は再計算できない。今回の同じ位置は Phase A で cosine 0.903、PESQ 1.064 と 1.059（差 0.005）、SI-SDR -27.3 dB と -28.6 dB（差 1.28 dB）だった。0.5 秒は品質の正本ではない。同じ重みの 4.5〜10 秒では cosine 0.906、PESQ 差 0.003、SI-SDR 差 2.0 dB である。cosine が 0.9 前後でも PESQ はほとんど動かない。差の本体は、両方に共通する大きな歪みの上の 1〜2.5 dB の SI-SDR である。

### 再開

Phase A の更新 10 回の checkpoint から 1 更新した Generator は、連続実行の更新 11 回と最大絶対誤差 `6.29e-5`（公式）と `5.95e-5`（独立）だった。ビット一致ではない。

### テスト

`.venvs/lab`。失敗 0。

| スイート | 結果 |
|---|---|
| `tests/training` | 56 件。品質比較の 8 件を含む |
| `tests/se` | 28 件 |
| `tests/evaluation` | 81 件 |
| `tests/asr` | 5 件 |

### 再現

```bash
# turing。空いている sm_75 GPU。PESQ はここでは計算しない
export CUDA_VISIBLE_DEVICES=<free> CUBLAS_WORKSPACE_CONFIG=:4096:8
.venvs/lab/bin/python tests/training/reference/semamba_quality.py prepare \
    --out data/results/training/semamba-quality-20261101/phase-a
.venvs/semamba/bin/python tests/training/reference/compare_semamba_quality.py \
    --out data/results/training/semamba-quality-20261101/phase-a --steps 20
.venvs/lab/bin/python tests/training/reference/semamba_quality.py score \
    --run data/results/training/semamba-quality-20261101/phase-a
```

波形と checkpoint は `data/results/training/semamba-quality-20261101/phase-a/` と `phase-b/`。`scores.json` が指標の一次記録である。commit と push はしていない。

## SEMamba++ 再現性と品質の追加調査（2026-11-01）

今回の判定は **NO-GO**。前節の 20 更新の PARTIAL 判定は変更しない。今回の比較では条件ごとに評価区間が異なり、PESQ の長さ依存も未解決で、品質指標を同等性判定の根拠として信頼できない。強調性能を獲得したモデルの品質同等性も確認していない。

`seed_model_init` の `torch.manual_seed` は CPU だけでなく CUDA の seed も変更していた。CPU の `default_generator` のみを seed するよう修正し、CPU 学習時にも CUDA を seed しないようにした。CPU 状態の復元と CUDA seed 呼び出しをテストした。これは融合カーネルの勾配差を解消する修正ではない。品質許容判定は空の発話リストを失敗とする。

`turing` の Quadro RTX 5000、PyTorch 2.6.0+cu124、`CUBLAS_WORKSPACE_CONFIG=:4096:8`、cuDNN deterministic、benchmark と TF32 オフで、同じ入力と重みを使い融合 Mamba を反復した。forward は一致したが backward は一致せず、入力勾配の最大差は `2.85e-13`、パラメータ勾配の最大差は `2.92e-11` だった。独立した演算の検査では、`causal_conv1d` の重み・バイアス勾配と `selective_scan` の A/B/C/D 勾配に差が再現した。`torch.use_deterministic_algorithms(True)` では causal convolution の今回の試験は一致したが、selective scan の差は残った（A 勾配最大 `1.91e-6`）。これらの値は指定入力の測定値であり、全形状や学習全体に対する上限ではない。CUDA RNG の差がなくても backward に差が生じることを示すが、学習後の差をこれだけで定量説明したわけではない。

前回 phase-a の step 10 波形を再採点した。最初の発話の clean / noisy は 93,680 サンプル、公式は 93,600、独立は 93,680。公式 iSTFT は `length` を指定しないため、hop 100 の倍数に切り下げた 93,600 を返す。独立の全文 PESQ 4.407 は同じ clean と独立波形を共通 prefix 93,600 サンプルにすると 1.070、SI-SDR は約 -38.8 dB のまま。全文 PESQ は共通ゲイン 1、0.1、0.01 でほぼ不変で、末尾 20 サンプルを削っても 4.407、80 サンプル削ると 1.070 だった。原因を PESQ 内部の特定処理まで同定できていない。clean/noisy はエネルギー正規化後にピークが 1 を超えるが、クリップはしていない。比較側の既存スコアは各条件で個別に短い方へ切るため、公式と独立が同じ区間とは限らない。共通 prefix と、公式末尾をゼロで埋めた全文の両方を記録し、この相違を隠さない。末尾ゼロ埋めを公式出力仕様の修正とはみなさない。

データ候補として、既存の resource DB は読み取り専用で使い、`data/staging/semamba-medium-20261101/` にレシピとセットを保存し、別 DB `data/db/semamba-medium-20261101.duckdb` に登録した。LibriSpeech train-clean-100 の 64 話者 64 ケース、dev-clean の別 8 話者 8 ケース、各 2 秒、SNR 0/5/10/15 dB。学習の DEMAND 4 シーンと検証 2 シーンは別で、最終テストは使用していない。各セット先頭の再生成一致を確認した。これは公式 VoiceBank-DEMAND ではなく、品質を獲得するのに十分な量であることも未確認である。公式 VoiceBank+DEMAND の train/test wav ディレクトリは存在するが、そのレシピ登録と学習は未実施。

再現用コードは `tests/training/reference/probe_semamba_kernel.py`、`probe_semamba_components.py`、`audit_semamba_waves.py`、`compare_semamba_checkpoints.py`、`prepare_semamba_medium.py`。結果は `data/results/training/semamba-repro-20261101/`。ステージングと DB の生成スクリプトは既存出力があれば上書きを拒否する。既存結果と本番 DB は変更していない。

追加の短時間再実行 `run1` は以前と同じ 8 個の 0.5 秒学習区間を循環し、同じ 4 発話を検証した。判定点 20 更新の平均は noisy / 公式 / 独立の順に PESQ 1.674 / 1.052 / 1.056、STOI 0.942 / 0.597 / 0.592、SI-SDR +7.51 / -25.71 / -26.37 dB。公式と独立の平均絶対差は 0.0095 / 0.0054 / 0.661 dB、波形 cosine は 0.894。SI-SDR は 1 発話で 1.381 dB 差となり、発話ごとの許容 1.0 dB を超えた。前述のアラインメント問題があるため、この指標は品質同等性の証拠としない。0→20 更新で平均 SI-SDR は公式 -39.48→-25.71 dB、独立 -39.48→-26.37 dB と増えたが noisy を下回る。run1 は品質学習ではない。

run1 の step 10 から 1 更新する既存のプローブでは、連続 step 11 との差は Generator 最大 `8.06e-5`（公式）、`5.84e-5`（独立）。実験時間は 277.1 秒（検証を含む）、最大 GPU 割当量は `run1/train_summary.json` に記録した。テストは training 61 件、SE 28 件、evaluation 81 件、ASR 5 件が成功。training は最初サンドボックスのローカル通信制限で 2 エラーとなり、権限付き環境で再実行して成功した。テストの skip と、fixture が意図的に報告する未登録データの skip は別である。

同じ物理 GPU 1 の別プロセスで 20 更新した `run1` / `run2` は、更新前 checkpoint の Generator、CQT、MRD、Optimizer、Scheduler、RNG が一致した。更新前検証波形は独立側で一致、公式側では最大 `4.60e-6` の差が出た。1 更新後の Generator 最大差は公式 `4.00e-4`、独立 `3.12e-4`、20 更新後は公式 `4.73e-3`、独立 `3.46e-3`。20 更新時の CQT / MRD 最大差は公式 `3.88e-4` / `1.77e-4`、独立 `4.74e-4` / `1.55e-4`。Optimizer の state にも差がある一方、Scheduler と Python / NumPy / CPU / CUDA RNG state は一致した。両モデルを同一プロセスに併置する既存比較の 2 実行であり、4 系統それぞれを単独プロセスにした試験ではない。run1 の開始後に CPU 用 seed の修正を行ったが、どちらも学習前に同じ training seed を明示し、step 0 の状態は一致していた。forward の最初の差を演算単位まで同定していない。

run2 の step 10 から全モデルと Optimizer / Scheduler / RNG を復元し、同じ入力順で 20 まで再開した。連続実行との差は Generator 公式 `7.79e-4`、独立 `6.23e-4`、CQT 公式 `2.22e-5`、独立 `7.35e-6`、MRD 公式 `2.50e-6`、独立 `3.91e-6`。Optimizer state は一致せず、Scheduler と RNG は一致した。再開に 64.5 秒。再開後の検証波形は保存していない。詳細は `rerun-*.json`、`resume-0020-*.json` と `resumed/`。

run2 の step 20 の 4 発話平均は noisy / 公式 / 独立で PESQ 1.674 / 1.056 / 1.043、STOI 0.942 / 0.588 / 0.597、SI-SDR +7.51 / -27.10 / -26.03 dB。平均絶対差はそれぞれ 0.0137、0.0102、1.068 dB、波形 cosine 0.898。SI-SDR は 2 発話で 1 dB 超、別の 2 発話は長さ不一致。各波形のサンプル数をスコアに保存し、長さ不一致は許容差判定を失敗とするよう変更した。数値を参考記録として残し、過去結果を上書きしない。

再現実行の例（出力先は未使用の新規ディレクトリを指定する）:

```bash
# 空き GPU を確認したうえで、融合 CUDA カーネルの単体調査
CUDA_VISIBLE_DEVICES=<free> CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 \
  .venvs/semamba/bin/python tests/training/reference/probe_semamba_components.py
# run2 から step 10 → 20。出力先は存在しないこと
CUDA_VISIBLE_DEVICES=<same-gpu> CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venvs/semamba/bin/python tests/training/reference/resume_semamba_pair.py \
  data/results/training/semamba-repro-20261101/run2 <new-output-directory>
.venvs/lab/bin/python tests/training/reference/compare_semamba_checkpoints.py \
  data/results/training/semamba-repro-20261101/run2/checkpoints/step_0020_official.pt \
  <new-output-directory>/checkpoints/step_0020_official.pt <new-comparison.json>
```

未実施は、4 系統の単独プロセス比較、step ごとの勾配保存と最初の差の演算特定、中規模候補を使う数百更新以上の学習、最終テスト、信頼できる PESQ と厳密に同じ長さ・時刻の比較である。`selective_scan` の観測差を学習差の唯一の原因と決めつけず、これらを実施してから再判定する。

## SEMamba++ 学習済み重みの診断（2026-11-01、途中）

**総合 PARTIAL。推論品質 GO、学習品質 GO とも未判定。** 次の診断は既存の拡張レシピを再生した4発話で、登録済み名前付き評価セットそのものではなく、最終テストの代替ではない。本番 DB は変更していない。公式テストとの重複も未証明である。

公式ソース `src/se/official/semambapp` commit `cd369c89a8106072c077b468417b765b227afd74`、公式 README の [配布重み](https://huggingface.co/yongjoonlee/semambapp/resolve/main/semambapp.pth) に対応するローカル `data/checkpoints/semambapp/semambapp.pth` SHA-256 `ccb9d1c49146e915c427e5724fec36825196f6b7a8fe23414c26a5224b39af72`。両 Generator に `strict=True` で読み、欠落・余剰はともに 0。幅 48、4 block、state 16、conv 4、expand 2、eval mode。配布元のファイルとローカルのバイト一致は今回照合していない。公式 README は 2026 年 8 月の再学習重みと記載するが、同梱 train/validation JSON は空で、VoiceBank test 発話や話者の非重複を確認できない。

入力は既存の LibriSpeech/DEMAND 検証レシピ由来の 16 kHz 浮動小数点波形で、今回リサンプリングなし。公式 `infer.py` と同様に noisy peak + 1e-9 で割り、出力に peak を掛ける。Hann、center reflect、FFT 400、hop 100、圧縮 0.3。公式 iSTFT は `length` 未指定、独立は入力長指定。フレーム数は `1 + floor(N/100)`、公式出力長は `100 * floor(N/100)`、開始位置はいずれも入力先頭である。差 80 は `N % 100 == 80` のときで、末尾に由来する。診断の採点は全条件で共通 prefix を使用し、差を隠す padding はしない。これを正式な評価契約の承認や発話全体の評価とはみなさない。

| case ID 先頭 | 入力 / frame / 公式 / 独立 | 波形 cosine（float64） | 最大絶対差 | noisy / 公式 / 独立 SI-SDR dB |
|---|---|---:|---:|---|
| 6a9b80200b27 | 93680 / 937 / 93600 / 93680 | 0.999999984 | 0.002296 | 0.044 / 10.027 / 10.027 |
| 7be04d42ba9e | 160000 / 1601 / 160000 / 160000 | 1.000000000 | 0.000594 | 4.992 / 18.740 / 18.740 |
| bf30dca8ea46 | 160000 / 1601 / 160000 / 160000 | 0.999999953 | 0.004282 | 10.003 / -6.610 / -6.611 |
| 464cf1d482e8 | 72480 / 725 / 72400 / 72480 | 0.999999734 | 0.011689 | 15.015 / -20.364 / -20.365 |

4件の公式 PESQ は 1.773、2.790、1.711、2.009、STOI は 0.8893、0.9859、0.9189、0.8615。公式と独立の同指標差は詳細 JSON を参照。PESQ 異常の根本原因が未解明なので、これらを品質同等性の証拠にはしない。少なくとも後半2件の SI-SDR は noisy より悪化しており、公式モデルがこの診断で常に改善するとはいえない。

既存 step 10 の異常例を再測定した。clean / 独立波形はいずれも float32、93680 samples、NaN/Inf なし。clean peak 4.418、enhanced peak 0.814、先頭100ms RMS は 0.0105 / 0.0420、末尾100ms RMS は 0.0136 / 0.0487。16 kHz、`pesq==0.0.4` の `pesq(16000, ref, est, 'wb')`、wrapper は float32 経由 float64、外部 resampling なし。93620 samples で PESQ 1.07015、93640 では 4.40658、93680 では 4.40710、SI-SDR は全て約 -38.84 dB。共通ゲインを変えても異常は残る（既存 `pesq-sensitivity.json`）。PESQ 内部の padding / alignment のどの処理が原因か、別実装との照合を含め未特定。基本的な同一波形・スケールの回帰テストが通ることはこの異常の解決を意味しない。

再実行例（出力先は必ず新規）:

```bash
CUDA_VISIBLE_DEVICES=<空きGPU> CUBLAS_WORKSPACE_CONFIG=:4096:8 .venvs/semamba/bin/python tests/training/reference/probe_semamba_pretrained.py <新規出力先>
.venvs/lab/bin/python tests/training/reference/audit_semamba_waves.py <新規出力先>/waves.pt <新規出力先>/audit.json
```

実行環境は turing Quadro RTX 5000 sm_75、driver 535.274.02、Python 3.10.20 / torch 2.6.0+cu124 / mamba-ssm 2.3.2.post1 / causal-conv1d 1.7.0、採点は lab Python 3.13、pesq 0.0.4、pystoi 0.4.1、numpy 2.5.4、scipy 1.18.1。結果は `data/results/training/semamba-audit-20261101/`。公式融合 selective_scan backward の非決定性等は前節の既存測定を参照。今回の4独立プロセス、resume 10→20、十分な品質獲得学習は未実施。以前の2系統併置結果では代替できない。以前の 0.5 秒・2系統併置で 20 step 275 秒、最大 8366 MiB、学習済み checkpoint は1系統あたり約87 MiB。公式3秒バッチ8や長時間学習にその数字を外挿して容量・所要時間を保証できず、別途実測と予算の確認が必要である。

回帰テストは `.venvs/lab/bin/python -m unittest discover -s tests/{training,se,evaluation,asr} -v` を各ディレクトリで実行し、順に 61 / 28 / 82 / 5 件、失敗 0、skip 0。個別のログは上記結果ディレクトリにある。今回の判定は、波形長・時間対応 PARTIAL、指標信頼性 PARTIAL、重み互換性 GO、推論波形同等性 PARTIAL、推論品質同等性 PARTIAL、同一実装の学習再実行性 PARTIAL、checkpoint 再開再現性 NO-GO（前節の併置実験で不一致、独立プロセスで未再確認）、学習による品質獲得 NOT TESTED。正式な比較には、独立した一時 DB と永続 staging に VoiceBank のペアを登録し、重複を証明できるデータ出典を確保し、評価区間・失敗時の扱い・発話ごとの許容差と集計を実行前に固定する必要がある。

## SEMamba++ 追加検証（2026-11-01、進行中）

PESQ の問題例を 93,600〜93,680 サンプルの全長で再採点した。93,631 では 1.070、93,632 では 4.407 となり、逆順の呼び出しでも一致した。両波形を 93,600、93,620、93,640 で切り、その後 93,680 までゼロで埋めるといずれも約 4.407〜4.408。`pesq==0.0.4` の wrapper は共通ピークで正規化して float32 に変換し、ネイティブコードは長さを Downsample 単位で整数除算して VAD を計算する。境界はその窓に一致するが、窓計算とスコア跳躍の因果関係は未確定である。異常スコアを品質判定に使わない。結果は `data/results/training/semamba-boundary-20261101/boundary.json`、再現コマンドは `.venvs/lab/bin/python tests/training/reference/probe_pesq_boundary.py <新規JSON>`。

無音や非有限値の入力、STOI の発話不足時のライブラリ警告と固定返却値、SI-SDR のゼロ推定・直交推定を数値として集計しないよう修正した。弱雑音・強雑音・時間ずれ・長さ違い・短時間・無音・非有限値の回帰試験を追加した。実音声での PESQ の異常解決を意味しない。

公式 VoiceBank-DEMAND test 全824件は clean/noisy の名前・サンプル数・サンプリング周波数・チャネル数を照合し、`data/staging/semamba-voicebank-test-20261101-v2/` と別 DB `data/db/semamba-voicebank-test-20261101-v2.duckdb` に名前付きセット `voicebank_demand_official_test` として登録した。2話者 p232 / p257、波形全体、レシピにファイルの SHA-256 を保存。元データへの新しいシンボリックリンクは `data/corpora/VoiceBank-DEMAND`。公式重みの学習データとの非重複は未証明。初回は kind 制約で登録失敗し、元の staging と DB を保持した。推論用 Python 3.10 では DuckDB 2.0 を使用できないため、DuckDB 1.4.3 を追加して同 staging から `data/db/semamba-voicebank-test-20261101-compat.duckdb` を再構築した。最初の推論は新形式 DB を読めず失敗し、互換 DB で再実行中。これらの結果だけで品質 GO とはしない。

回帰テストは `.venvs/lab/bin/python -m unittest discover -s tests/<suite> -v`、`OMP_NUM_THREADS=1` で training 61、se 28、evaluation 84、asr 5 件、すべて成功。ログは `data/results/training/semamba-boundary-20261101/test-logs/`。

同じ PESQ 境界検査を別プロセスで再実行し、全 81 長・正逆順・ゼロ padding の結果が完全に一致した（`boundary-process2.json`）。公式単独プロセス 2 系統（同一物理 GPU 0、同じ初期 checkpoint と seed、0.5 秒区間、20 更新）では初期のモデル・optimizer・scheduler・RNG が一致。初回 forward は一致したが、保存した Generator 勾配の最大差は 1.09e-6、1 更新後 Generator 最大差 2.94e-4、20 更新後 4.48e-3。step 10 から別プロセスで再開した公式 20 更新後の Generator 最大差は 9.67e-4、scheduler と RNG は一致した。これだけでは checkpoint の全復元状態の完全性や非決定的カーネルとの因果関係は証明できない。一次結果は `data/results/training/semamba-single-20261101/`、診断の入口は `tests/training/reference/train_semamba_single.py`。独立実装側も実行中。推論の最初の46発話を保存後に処理を中断し、残りを区分して 3 GPU で再実行中であり、全件の採点は未完了。

PESQ のネイティブ実装を変更せず、`gdb` で `pesq_psychoacoustic_model` の関数入口の `ERROR_INFO` を観測した。93,631 サンプルでは分割数 8、93,632 では 11 となり、後者の `Utt_Delay` には約 -60,000 samples（16 kHz で約 -3.75 秒）の値がある。入力は同じ時点からの prefix であり、この極端な遅延は波形自体の時間ずれを反映しない。問題は心理音響計算より前の発話区分・遅延推定に現れる。個別のコード行が不連続の唯一の原因か、他の音声で安全かは未証明。生の観測は `data/results/training/semamba-boundary-20261101/gdb-entry-93631b.log` と `gdb-entry-93632b.log`。

独立実装の単独 A/B も初期モデル・optimizer・scheduler・RNG が一致した。初回 forward は一致し Generator 勾配差最大 1.27e-6、初回更新後 Generator 差 3.70e-4、20 更新後 3.33e-3。公式と独立の初回 forward 最大差は 5.89e-7。保存した Discriminator 勾配は Generator backward 後の値であり、Discriminator backward の単独比較と混同しない。両独立実装の再開比較は実行中。

公式配布 URL の HTTP 応答の `x-linked-etag` は `ccb9d1c49146e915c427e5724fec36825196f6b7a8fe23414c26a5224b39af72`、ローカル checkpoint の SHA-256 と一致する（`curl -sI https://huggingface.co/yongjoonlee/semambapp/resolve/main/semambapp.pth`）。公式 README は修正版を同じ設定で再学習したと記載するが、今回取得できた情報では学習発話一覧を示していないため、test との非重複は依然として未証明。

名前付き `voicebank_demand_official_test` の 824 発話について、同じ配布重みを両実装に strict load し、16 kHz、公式 peak 正規化の推論波形を個別保存した。46 件の先行処理の後、残りは同じ設定で 3 GPU に分割した。採点は clean/noisy/両出力の全てに共通する先頭区間で行い、元の長さを別に記録した。807/824 件は公式 iSTFT と独立 API の出力長が違う。float64 で求めた共通区間の cosine 最小 0.999998763、最大絶対差 0.001393。STOI の公式・独立差の最大 0.000208（許容 0.02 超 0 件）、SI-SDR 最大 0.0321 dB（許容 1 dB 超 0 件）。PESQ 差の最大は 0.00365 以下だが、上述のアラインメント異常があるため合否には使わない。noisy / 公式 / 独立の平均 STOI は 0.9211 / 0.9492 / 0.9492、平均 SI-SDR は 8.449 / 16.457 / 16.456 dB。公式重みはこの test セットで平均改善している。checkpoint と test の訓練時非重複は未証明であり、共通 prefix の結果を全文一致や独立 held-out にすり替えない。全発話の指標と分布は `data/results/training/semamba-boundary-20261101/voicebank-scores.json`、float64 波形比較は `voicebank-wave64.json`。`voicebank-scores.json` の既存 float32 cosine 集計は丸めにより 1 を僅かに超えるため、cosine は float64 の記録を使う。

VoiceBank 28 話者公式 train のペア 11,572 件から p286 / p287 の 872 件を validation、残り 10,700 件を train として `data/staging/semamba-voicebank-train-20261101/` と `data/db/semamba-voicebank-train-20261101.duckdb` に保存・登録した。公式 test p232 / p257 は train / validation と話者が重ならない。作成は `.venvs/lab/bin/python tests/training/reference/prepare_voicebank_train.py`、test は `prepare_voicebank_test.py`。生成処理は既存の出力を上書きしない。

実 Trainer の 64/8 ケース・4 更新パイロットは損失が有限で checkpoint と validation L1 を保存した。validation の PESQ / STOI は現状未計算。CUDA checkpoint を `map_location=cuda` で開くと CPU RNG state も GPU に移り、`torch.random.set_rng_state` が拒否して再開できない不具合を再現した。`restore_rng` は CPU / CUDA の RNG state を CPU byte tensor に戻してから復元するよう修正した。修正後 step 2 → 4 を別ディレクトリで再開でき、データ位置・バッチ計画・RNG は一致したが、task の optimizer state に最大差 0.000696 が残る。保存時の CUDA RNG は公式・独立とも変化しなかった。詳細は `data/results/training/semamba-pilot-20261101-v2/`、`semamba-pilot-resume-20261101/`（修正前失敗）、`semamba-pilot-resume-20261101-v2/`。実 Trainer の再開成功とビット一致を区別する。

追加した回帰試験を含め、修正後の training 62 件と evaluation 84 件が成功。SE 28 件と ASR 5 件は同じ実行で成功。テストログは `data/results/training/semamba-boundary-20261101/test-logs/`。

PESQ の因果確認として、問題の 93,632 samples に限り、ネイティブ実装の心理音響モデル入口で `Utt_Delay` の各値をデバッガから 0 に変えて続行したところ、4.40668 から 1.03765 に変わった。これは推定遅延が高スコアに直接関与することを示す介入実験であり、変更後の数字を正式な PESQ として使用しない。ログは `gdb-zero-93632.log`。問題の例では PESQ を品質判定から除く。

追加した SEMamba の STFT/iSTFT 長さ回帰では 401〜680 samples の hop 境界を検証した。center reflect、FFT / window 400、hop 100 で STFT は `1 + floor(N/100)` frame、公式と同じ length 未指定の iSTFT は `100 * floor(N/100)`、独立 API の再合成は `N` samples を返す。identity spectrum で入力と再合成の時間対応も確認した。SE の再実行は 29 件成功。

VoiceBank 実ペアの 4 更新パイロットは `data/results/training/semamba-voicebank-pilot-20261101/` に config、ログ、checkpoint を保存し、有限な損失で終了した。これは品質獲得を証明しない。この checkpoint から 100 更新までの継続と、固定した validation 8 件（各検証話者の登録順先頭 4 件、診断用）の複数 checkpoint 評価を実行中。評価用 subset は全 validation や test の代わりではない。

公式・独立を別プロセス、同じ物理 GPU 2 で、同一初期 checkpoint、VoiceBank train の同一 seed / 順列 / 0.5 秒 crop を使って各 4 更新した。初期の Generator / CQT / MRD、optimizer、scheduler、全 RNG は一致し、4 件の `case_id` 順も一致。公式 / 独立の Generator 損失は step 1 で 19.015274 / 19.015274、step 4 で 17.987616 / 17.990313。step 4 の Generator 最大差は 0.001016。結果は `data/results/training/semamba-voicebank-paired-{official,independent}-20261101/` と `data/results/training/semamba-boundary-20261101/voicebank-train-*.json`。これは実データでの短時間パイプライン診断であり、品質獲得の証拠ではない。互換 DB は `data/staging/semamba-voicebank-train-20261101/` を DuckDB 1.4.3 で再構築した。

長時間実行の規模を、実 VoiceBank ペア・公式幅・0.5 秒・batch 1 の上記 4 更新で実測した。checkpoint 書き込みを含み公式 9.57 秒、独立 25.96 秒。10,700 更新への単純外挿は順に約 7.1 / 19.3 GPU 時間（合計約 26.4 GPU 時間）で、full validation、test、起動・異なる長さ、再実行、収束に必要な追加 epoch を含まない。各学習済み checkpoint は約 87 MiB。`tests/training/reference/train_semamba_voicebank.py` は両実装を別プロセスで同じ named train set と順列・crop に適用し、1000 更新ごとに保存する。4 更新の実動作を確認したが、長時間試験の完了や収束は確認していない。

VoiceBank の 100 更新と固定 validation 8 件の全予定点が終了。step 4 / 20 / 40 / 60 / 80 / 100 の平均 SI-SDR は -41.30 / -22.60 / -14.53 / -8.04 / -7.13 / -5.75 dB、平均 STOI は 0.614 / 0.722 / 0.730 / 0.709 / 0.719 / 0.669。noisy の平均は SI-SDR 6.30 dB、STOI 0.799。PESQ は異常のため参考値のみ。SI-SDR の上昇傾向はあるが、100 更新後も noisy を下回り、品質を獲得したとは判定しない。データ順や音声は `data/results/training/semamba-voicebank-100-validation-20261101/`、学習曲線・checkpoint は `data/results/training/semamba-voicebank-100-20261101/`。実行は同ディレクトリの `config.yaml` と `tests/training/reference/validate_voicebank_pilot.py` に記録した条件である。

VoiceBank 学習用の公式・独立 checkpoint をそれぞれ step 4 から別プロセスで step 6 に再開できた。再開直後のモデル・optimizer・scheduler・RNG は保存 checkpoint と完全一致した（`voicebank-{official,independent}-restore.json`）。継続実行の step 6 との比較はしておらず、長時間の厳密再現を保証しない。

## Dataset、Mock Trainer、SEMamba++ 200 step（2026-11-01）

ホスト `turing`。学習比較は物理 GPU 0（Quadro RTX 5000、使用メモリ 15 MiB、sm_75）。融合カーネルの cubin は sm_75 だけなので、この GPU を恒久の指定にはしない。lab は Python 3.13、semamba は Python 3.10。どちらも PyTorch 2.6.0+cu124。`CUBLAS_WORKSPACE_CONFIG=:4096:8`。`torch.use_deterministic_algorithms(True)` は融合 selective scan の backward が実行できないため、200 step には使っていない。cudnn の benchmark は切ってある。

本番 `data/db/resource.duckdb` は開いていない。学習セットは既存の `data/db/semamba-voicebank-train-20261101-compat.duckdb`（train 10,700、validation 872）。test は使っていない。

### 修正

1. DDP は forward のときに all-reduce を予約する。`no_sync` を `loss.backward()` だけに掛けると、非最終 micro-batch でも同期する。`accumulation_scope` は forward を包む。等サイズの micro-batch では、毎 micro-batch の平均でも最後の 1 回の平均でも global batch の勾配は一致する。不要な同期を外した。gloo の 2 プロセスで、同期フラグは full batch が `[1]`、蓄積が `[0, 1]`、コンテキストを出た直後の backward が `[1]`、遅延 backward の手動平均が `[0, 0]` で、勾配はいずれも `rtol=1e-5`, `atol=1e-6` で全バッチと一致した。
2. SEMamba++ の Generator は micro-batch ごとに 1 回だけ forward し、Discriminator はその detach、Generator 損失は同じ波形を使う。公式 `train.py` と同じ順序である。DDP かつ蓄積が 2 以上のときは Reducer の hook を使わず、Generator backward のあとで勾配を平均する。`enhance()` と checkpoint のキーは変えていない。

### Phase A

実 `RecipeDataset`、`epoch_indices`、`Trainer`。モデルは載せていない。CPU。

VoiceBank の結果は `data/results/training/semamba-supply-20261101/supply.json`。

| 確認 | 結果 |
|---|---|
| train / validation 件数 | 10,700 / 872 |
| seed 1234 を 2 回。バッチ列、crop、clean/noisy の SHA-256 が一致。seed 1235 で先頭バッチの case 順が変わる | PASS |
| 切り出しが CPU / NumPy / Python のグローバル乱数を消費しない | PASS。`rng_draws=0` |
| clean と noisy が同じ crop で、レシピハッシュが `case_id` と一致 | PASS。16 行すべて |
| epoch 0、global batch 8。使用 10,696、末尾 4 件だけ欠落、使用内の重複 0 | PASS |
| 1 GPU micro 8、1 GPU micro 2 × 蓄積 4 のサンプル順が一致。2 GPU micro 2 × 蓄積 2 は集合が一致し順序は違う | PASS。先頭 5 optimizer step |
| global 8 を world 3 で割る構成 | 拒否 |
| 2 step で保存し、別プロセスで再開したサンプル列が連続実行と一致 | PASS。単体テスト |

### Phase B

実 `Trainer` に `Linear(4, 4)` の Generator と `Linear(4, 1)` の Discriminator を渡した。損失は MSE。CPU の決定的実行では再開がビット一致。

| 確認 | 結果 |
|---|---|
| 更新順が Discriminator、Generator。相手の重みは変わらない。Discriminator backward が Generator 勾配を書かない | PASS |
| micro 4 と micro 2 × 蓄積 2 の 1 step。重み、Optimizer、Scheduler が `rtol=1e-5`, `atol=1e-6` | PASS |
| 2 プロセス gloo が、micro 2 と micro 1 × 蓄積 2 のどちらも単一プロセスの 1 step と上記許容で一致 | PASS |
| 8 サンプル、global batch 4、4 step。Scheduler は 2 epoch 分。学習率 `0.001 * 0.5**2` | PASS |
| 連続 20 step と、10 step 保存から別プロセスで 20 step。Generator、Discriminator、Optimizer、Scheduler、epoch、cursor、CPU RNG が一致 | PASS |
| 同じ再開を CUDA、`deterministic`、`CUBLAS_WORKSPACE_CONFIG=:4096:8` で実行。重み、Optimizer、CPU RNG、CUDA RNG の最大差 0 | PASS。`data/results/training/mock-cuda-resume-20261101/report.json` |

### Phase C

4 プロセスを物理 GPU 0 で順に実行した。公式 A 14:34–14:40、公式 B 14:40–14:46、独立 A 14:46–15:07、独立 B 15:07–15:29（各 200 step）。初期重みは `data/results/training/semamba-repro-20261101/run2/checkpoints/step_0000_official.pt`。Optimizer は新しい AdamW（lr `2e-4`、betas `(0.8, 0.99)`、weight decay `0.01`）。ExponentialLR の gamma は `0.99` で、10,700 step に満たないため 200 step では進まない。学習率は終始 `2e-4`。勾配クリップはしない。

公式 `config.yaml` の batch 8、segment 48,000 は、この 16 GB GPU で Generator の計算グラフを 8 本保持する実行にしていない。比較は batch 1、segment 8,000（0.5 秒）。損失に入る項は実行される公式ループと同じで、時間 L1 と metric 係数はコメントアウトのまま合計に入っていない。

入力は 4 系統で 200 step すべて同じ `case_id` と clean/noisy SHA-256。損失はすべて有限。

step 1 の追跡:

| 項目 | 公式 A と公式 B | 公式 A と独立 A |
|---|---:|---:|
| 学習バッチの波形、最大絶対誤差 | 0 | 7.08e-7 |
| Generator 損失 | 0（どちらも 19.015274047851562） | 0 |
| Discriminator 損失 | 0 | 4.77e-7 |
| 振幅、位相 | 0 | 0 |
| 敵対、特徴、mel、整合 | 0 | 4.8e-7、9.5e-7、2.4e-7、6.0e-8 |
| Generator 勾配、最大絶対誤差 | 1.44e-6 | 1.52e-4 |
| `|grad|` 最大が 1e-5 を超えるテンソルの最小 cosine | 0.99999940 | 0.99999946 |
| それより小さいテンソルの最小 cosine | -0.038 | -1.0（bias、`|grad|` 最大 1.8e-8） |
| Generator 重み | 3.01e-4 | 4.00e-4 |
| CQT 重み | 0 | 2.04e-8 |

最初に 1e-4 を超える Generator 損失差は step 4 の 0.00272。同じ step の公式再実行差は 8.2e-5。差は forward ではなく、1.5e-4 の勾配差が AdamW を通ったあとに広がる。大きな勾配の方向は cosine 0.999999 で揃っている。

200 step の Generator 損失差の最大と、step 200 の重みの最大絶対誤差:

| 比較 | 損失差の最大（step） | step 200 の損失差 | step 200 の重み |
|---|---:|---:|---:|
| 公式 A / 公式 B | 0.354（196） | 0.122 | 0.0203 |
| 独立 A / 独立 B | 0.401（91） | 0.0097 | 0.0205 |
| 公式 A / 独立 A | 0.376（49） | 0.067 | 0.0197 |
| 公式 B / 独立 B | 0.383（97） | 0.065 | 0.0188 |

実装間の損失差と重み差は、公式どうしの再実行差と同じ規模に収まる。独立実装だけが発散してはいない。

固定 validation は `voicebank_28_valid` の登録順先頭 8 件。checkpoint 選択には使っていない。全 872 件ではない。PESQ は前回の遅延推定の不連続があるため判定に使っていない。平均 SI-SDR（dB）と STOI:

| step | noisy SI-SDR | 公式 A | 独立 A | 公式 A STOI | 独立 A STOI |
|---:|---:|---:|---:|---:|---:|
| 0 | 6.53 | -42.20 | -42.20 | 0.449 | 0.449 |
| 1 | 6.53 | -43.86 | -43.88 | 0.539 | 0.539 |
| 20 | 6.53 | -31.73 | -28.30 | 0.629 | 0.631 |
| 50 | 6.53 | -19.31 | -17.17 | 0.663 | 0.650 |
| 100 | 6.53 | -14.80 | -12.55 | 0.706 | 0.699 |
| 200 | 6.53 | -12.68 | -10.85 | 0.695 | 0.708 |

step 200 の出力波形 cosine の最小は、公式再実行 0.838、独立再実行 0.887、公式 A と独立 A 0.838（float64）。SI-SDR は 8 件の平均で約 30 dB 上がるが、noisy の 6.53 dB より低い。200 step では強調品質を獲得していない。これは失敗条件にしていない。

### 回帰テスト

`.venvs/lab/bin/python -m unittest discover -s tests/<suite> -v`

| スイート | 結果 |
|---|---|
| training | 73 件、失敗 0 |
| se | 29 件、失敗 0 |
| evaluation | 84 件、失敗 0 |
| asr | 5 件、失敗 0 |

ログは `data/results/training/semamba-200step-20261101/test-logs/`。

### 再実行

```bash
.venvs/lab/bin/python tests/training/reference/audit_voicebank_supply.py \
    data/results/training/semamba-supply-20261101
CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    .venvs/lab/bin/python tests/training/reference/mock_cuda_resume.py \
    data/results/training/mock-cuda-resume-20261101
CUDA_VISIBLE_DEVICES=<空きGPU> CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    .venvs/semamba/bin/python tests/training/reference/train_semamba_200.py \
    official|independent <新規ディレクトリ> 200
.venvs/lab/bin/python tests/training/reference/compare_semamba_200.py \
    data/results/training/semamba-200step-20261101
```

既存の出力ディレクトリは上書きしない。

### 判定

| 項目 | 判定 |
|---|---|
| Dataset / DataLoaderの正確性 | GO |
| seedによるデータ再現性 | GO |
| global batch・勾配蓄積 | GO |
| DDPの正確性 | GO |
| 更新順・Optimizer・Scheduler | GO |
| Mockによるcheckpoint再開 | GO |
| SEMamba++の200step再実行性 | PARTIAL |
| SEMamba++の公式・独立学習挙動 | GO |
| 長時間学習への移行可否 | GO |

再実行性を PARTIAL にしたのは、step 1 の Generator 損失は公式どうしも実装間も一致するが、200 step の重みは公式どうしでも最大 0.020 違うからである。差の大きさは融合カーネルの再実行と独立実装の再実行で同じで、独立実装だけが大きくずれてはいない。

長時間学習への GO は、この学習ステップとデータ供給を続けてよい、という意味である。200 step の SI-SDR が noisy を超えたことでも、公式の batch 8・3 秒セグメントをこの GPU で実行したことでもない。

### 残る課題

- 公式の batch 8、segment 48,000 は未実行。メモリが足りるかは別途確認する。
- 200 step では強調品質は得られていない。1 epoch は 10,700 step で、今回の実測（公式約 6 分、独立約 21 分 / 200 step、validation 8 件を含む）をそのまま 1 epoch に伸ばすと、validation を増やさない場合でも公式約 5 時間、独立約 19 時間のオーダーになる。
- DDP 上の SEMamba++ 蓄積は線形モデルと手動平均の単体テストまでで、実モデルの複数 GPU は回していない。
- PESQ は遅延推定の不連続が残るため、学習判定に戻していない。

## 公式条件に近い検証（2026-11-01）

前節の 200 step は batch 1、0.5 秒のままである。ここでは公式 `config.yaml` の batch 8、segment 48,000（3 秒、16 kHz）に寄せた。既存の 200 step 成果物は再実行していない。本番 `resource.duckdb` も変更していない。

ホストは `turing`。GPU は Quadro RTX 5000（16 GB、sm_75）。公式の融合カーネルの cubin は sm_75 だけなので、この GPU を恒久の指定にはしない。`CUBLAS_WORKSPACE_CONFIG=:4096:8`、cuDNN benchmark オフ、TF32 オフ。`torch.use_deterministic_algorithms(True)` は融合 selective scan の backward が実行できないため使っていない。

### 公式設定との差

| 項目 | 公式 `train.py` / `config.yaml` | この比較 |
|---|---|---|
| batch | 8。複数 GPU では `batch // n_gpus` を各 GPU が 1 回で処理する。勾配蓄積は無い | 16 GB では native batch 2 まで。global 8 は micro-batch 1 を 8 回蓄積 |
| segment | 48,000 | 48,000 |
| データ | オンライン劣化。clean と degraded を別々に peak 正規化し、切り出しは `random` | VoiceBank ペア。noisy のエネルギーで clean と共通正規化し、切り出しは `case_id` と seed |
| 損失 | 実行される和は敵対、特徴、mel、振幅、位相、複素、整合。時間 L1 はコメントアウト。metric 係数は設定にあるがこの和には入っていない | 同じ和 |
| Optimizer | AdamW、lr `2e-4`、betas `(0.8, 0.99)`。weight decay は引数に無いので PyTorch 既定の `0.01`。勾配クリップは無い | 同じ |
| Scheduler | ExponentialLR、gamma `0.99`、エポック終了時に 1 回 | 同じ。今回の step 数では進まない |
| 更新順 | Discriminator backward と step のあと、同じ Generator 波形で Generator | 蓄積 1 は同じ波形を再利用。蓄積 2 以上はグラフを保持せず、Discriminator step のあと Generator を forward し直す。dropout は実装に無い |
| 非決定 | `cudnn.benchmark = True`。融合 scan の backward は再実行で一致しない | benchmark はオフ。再実行差は下に記録した |

### 3 秒、batch、蓄積

入力は `voicebank_28_train` の epoch 0、seed 1234 の先頭 64 crop。各 48,000 サンプル。clean と noisy は同じ区間で、どちらも有限。結果は `data/results/training/semamba-phaseA-20261101/`。

| 実行 | 結果 |
|---|---|
| 公式 native batch 1 | 成功。ピーク 7.13 GB。Generator 損失 15.682699 |
| 公式 native batch 2 | 成功。ピーク 14.08 GB |
| 公式 native batch 4 と 8 | OOM。batch 4 の時点で確保 15.45 GB |
| 独立、再計算なしの batch 1 | OOM。参照 scan の計算グラフで 15.25 GB |
| 独立、学習時だけ scan を再計算した batch 1 | 成功。ピーク 9.08 GB。約 115 秒。Generator 損失 15.682700 |
| 独立 batch 2 | OOM |

3 秒・batch 1 の公式と独立（`step-compare.json`）:

| 項目 | 差 |
|---|---|
| Generator 損失 | 9.5e-7（15.682699 と 15.682700） |
| Discriminator 損失 | 4.8e-7 |
| 波形の最大絶対誤差 | 1.10e-6 |
| Discriminator 勾配 | 最大 3.65e-7、cosine 1.000 |
| Generator 勾配 | 最大 4.71e-3、cosine 0.999890 |
| 更新後の Generator 重み | 最大 4.00e-4、cosine 1.000 |

公式の native batch 2 を、同じ 2 サンプルの micro-batch 1 × 2 と比べた。損失の差は 2e-6、波形の最大差は 3.3e-6、Discriminator 勾配の最大差は 1.7e-6。Generator 勾配の最大差は 0.0465 で、同じ native batch 2 を 2 回走らせた再実行差 1.58e-5 より大きい。融合 scan の backward が batch 形状に依存する。損失が平均であることと、global batch が同じであることは、この勾配まで等しいことを意味しない。更新後の重みの最大差は 4.0e-4 で、1 step の AdamW がこの勾配差を通した大きさである。

蓄積で損失を割らない実装だったため、micro-batch の平均勾配が足し合わされ、公式の batch 平均より大きい step になっていた。平均損失を蓄積回数で割るように直し、CPU の小さい SEMamba で batch 2 と micro-batch 1 × 2 の勾配が一致する回帰テストを足した。

8 個の計算グラフを同時に持つ公式の蓄積は OOM になった（`short-official/` は step 0 だけ残っている）。Discriminator step のあと Generator を 1 micro-batch ずつ forward し直すと、公式は global batch 8 を 8 step 実行できた。所要 132 秒。損失はすべて有限で、Discriminator 損失は 6.32 から 4.48 へ下がった。step 4 の重みを別プロセスで復元すると最大差 0。そこから 1 step 進めた損失は連続実行の step 5 と表示上一致し、重みの最大差は 1.12e-4 だった。

独立の global batch 8 は 4 step まで終了した。所要 3,723 秒で、1 step あたり約 15 分である。Generator 損失は step 1 から 4 まで 17.814797、20.603335、17.376713、19.063525。同じ 8 サンプルの公式は 17.814772、20.604648、17.381723、19.066991。step 1 の差は 2.4e-5、step 4 の差は 0.0035。4 step とも有限で、Discriminator 損失も公式と同じく 6.32 から 5.97 へ動いた。

3 秒の実モデルを複数 GPU の DDP では回していない。蓄積 2 以上は、遅延 backward の手動 all-reduce ではなく、再 forward を `accumulation_scope` で包む。

### 64 発話の validation

`voicebank_28_valid` 872 件、話者 p286 / p287。test 話者 p232 / p257 は入っていない。train との `case_id` の重なりは 0。雑音は log の 10 種類、SNR は 0 / 5 / 10 / 15 dB で 80 セルある。名前順で最後の `station` と `traffic` を除き、残り 8 雑音 × 4 SNR × 2 話者 = 64。セル内の発話は SHA-256(`1234:話者:雑音:SNR`) で 1 件選んだ。スコアを見て選び直していない。一覧は `data/results/training/semamba-valid64-20261101/selection.json`。

前の 200 step の step 0 / 50 / 100 / 200 を、公式 A/B と独立 A/B に同じ 64 発話で通した。採点は共通区間の SI-SDR、STOI、PESQ、波形 cosine。noisy の平均 SI-SDR は 6.292 dB。

| step | 公式 A SI-SDR | 公式 B | 独立 A | 独立 B | 公式 A STOI | 独立 A STOI |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | -40.63 | -40.63 | -40.65 | -40.65 | 0.440 | 0.440 |
| 50 | -19.87 | -19.45 | -17.82 | -18.42 | 0.656 | 0.643 |
| 100 | -14.49 | -14.12 | -12.54 | -13.02 | 0.707 | 0.693 |
| 200 | -12.02 | -10.13 | -10.42 | -10.96 | 0.692 | 0.706 |

4 系統とも SI-SDR は上がるが、200 step でも noisy を下回る。発散や非有限値は無い。公式 A の step 200 を雑音と SNR で分けると、8 雑音は同程度で、SNR が高いほど noisy との差が大きい。セルあたり 2 発話なので、雑音ごとの優劣は付けない。

step 200 の発話別絶対差の平均は、公式 A/B が SI-SDR 1.89 dB、STOI 0.016、波形 cosine 最小 0.779。独立 A/B は 0.54 dB、0.009、0.856。公式 A と独立 A は 1.60 dB、0.015、0.781。実装間の差は、公式の再実行差より大きくない。64 発話・200 step では、実装差と再実行差を分離できる精度には届いていない。step 0 の公式どうし、独立どうしの波形 cosine は 1.000 である。

### PESQ

`pesq==0.0.4` は ITU-T P.862 / P.862.2（2005 年 10 月）の参照 C コードの wrapper である。16 kHz の `Downsample` は 64、`SEARCHBUFFER` は 75 で、読み込み時に両端へ足す 9,600 サンプルは 64 の倍数である。VAD 長は `(N + 9600) // 64` なので、N が 64 の倍数をまたぐと 1 増える。既知の失敗例は 93,631 で 1.070、93,632 で 4.407。前回のデバッガでは後者の発話数が 8 から 11 へ増え、推定遅延に約 -60,000 サンプルがあった。SI-SDR は約 -38.8 dB のままなので、高い PESQ は時間の合った品質を表していない。遅延を 0 に書き換える修正は入れていない。C コードも変更していない。

同じ境界を、時間の合った VoiceBank 16 対（2 話者 × 4 雑音 × 2 SNR、16 kHz へ resample）で測ると、境界とその 1 サンプル前の PESQ 差の最大は 0.0138、中央値は 1.2e-5 だった。0.1 をこの間に置き、全文スコアと直前の VAD bin の差が 0.1 を超えるときは `na`（`pesq_utterance_split_discontinuity`）にする。生の値は `raw_value` に残し、平均には入れない。別の PESQ 実装は追加していない。入っているライブラリが参照実装そのもので、同じ C コードの別 wrapper は独立した照合にならない。

64 発話では、この条件で `na` になったのは 1 checkpoint あたり 0 から 5 件。採用された PESQ の平均は 1.07 前後で、SI-SDR が noisy を下回ることと矛盾しない。4.4 のような不連続値は平均に入っていない。

### 回帰テスト

`.venvs/lab/bin/python -m unittest discover -s tests/<suite> -v`。CUDA は隠した。

| スイート | 結果 |
|---|---|
| training | 74 件、失敗 0 |
| se | 30 件、失敗 0 |
| evaluation | 86 件、失敗 0 |
| asr | 5 件、失敗 0 |

ログは `data/results/training/semamba-phaseA-20261101/test-logs/`。前回より増えたのは、蓄積と scan 再計算の一致、PESQ の境界である。

### 再現

```bash
# 3 秒 crop。出力先が無いこと
.venvs/semamba/bin/python tests/training/reference/prepare_voicebank_train.py   # 既存セットは上書きしない
CUDA_VISIBLE_DEVICES=<空き> CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  .venvs/semamba/bin/python tests/training/reference/run_semamba_official_batch.py \
  prepare <新規ディレクトリ>
# 64 発話の選択
.venvs/lab/bin/python tests/training/reference/select_voicebank_valid64.py <新規.json>
```

既存の `semamba-phaseA-20261101` と `semamba-valid64-20261101` は上書きしない。

### 長時間学習へ進むときの設定

この GPU での公式に近い実行は、segment 48,000、global batch 8、micro-batch 1、蓄積 8、AdamW lr `2e-4`、betas `(0.8, 0.99)`、weight decay `0.01`、勾配クリップなし、ExponentialLR gamma `0.99`。公式 8 step は 132 秒だった。10,700 発話を global 8 で 1 epoch とすると約 1,338 step で、同じ速度なら約 6 時間である。起動と validation を含まない外挿であり、1 epoch の実測ではない。

独立実装の参照 scan は、3 秒 batch 1 の学習 step が約 115 秒、global 8 の 1 step が約 15 分だった。同じ外挿では 1 epoch が約 2 週間になる。融合カーネルは sm_75 以外ではロードできない。長時間の独立学習を始める前に、この所要時間を予算として確認する。

checkpoint は 50 から 100 optimizer step ごと。検証は固定 64 発話を 100 step ごと。test 824 件は条件調整に使わない。PESQ は上記の `na` 条件のまま使い、SI-SDR と STOI を主にする。200 step の SI-SDR が noisy を超えていないので、この間隔は品質獲得の証明ではなく発散の監視である。

native batch 8 と micro-batch 1 × 8 の Generator 勾配は、上の 0.0465 と再実行差 1.58e-5 の範囲では一致していない。4 GPU が空いていれば、公式と同じく GPU あたり batch 2（1 GPU で 14 GB）が published の分割に近い。この構成は今回は実行していない。

### 判定

| 項目 | 判定 |
|---|---|
| 公式の 3 秒セグメント学習 | GO |
| global batch 8 | PARTIAL |
| 勾配蓄積・DDP との整合性 | PARTIAL |
| 公式・独立の学習挙動 | GO |
| 拡張 validation の妥当性 | GO |
| PESQ 評価の信頼性 | GO |
| checkpoint・再現性 | GO |
| 長時間学習への移行可否 | GO |

global batch 8 を PARTIAL にしたのは、native batch 8 が 16 GB に入らず、入る native batch 2 と micro-batch 1 × 2 の Generator 勾配が再実行差を超えるからである。損失と Discriminator 勾配と波形は一致している。DDP は 3 秒の実モデルでは未実行である。

移行の GO は、この手順で学習を始めてよい、という意味である。長時間学習のあと公式と独立の品質が一致することは、まだ証明していない。

## SEMamba++ fused scan・tensor 移行（2026-11-01、途中経過）

判定は保留である。`scan_backend: fused` を指定すると CUDA 上の選択的 scan のみを外部 `mamba-ssm` に委譲し、CPU では reference scan に戻す。既定は reference で、モデルの構造と checkpoint キーを変えない。外部 `mamba-ssm` は Apache-2.0、公式側が使う `causal-conv1d` は BSD 3-Clause。公式 SEMamba++ のコードは独立実装へコピーしていない。`tests/training/reference/run_semamba_official_batch.py` の discriminator 更新は生成波形を detach せず、同一 Generator graph の 2 度目の backward で失敗したため修正した。

`tensor` は driver 535.183.01、A100 PCIe 40 GB 7 枚と 80 GB 1 枚、全て compute capability 8.0。確認時 1、5、6、7 は 13 MiB 使用、0、2、3、4 は他ジョブ使用中。既存 `.venvs/semamba` は Python 3.10.20、PyTorch 2.6.0+cu124、`mamba-ssm==2.3.2.post1`、`causal-conv1d==1.7.0`。既存拡張は sm_75 のほか、turing の glibc 2.35 でビルドされており、tensor の glibc 2.31 では import できない。tensor 上の GCC 9.4 と `.venvs/cuda-nvcc` の nvcc 12.9.86 を使い、別ディレクトリで sm_80 向けのビルドを試行した。元ソースアーカイブ SHA-256 は causal `3202758494eaa7b597ce1c282dfa188889506bfcb92cad3c407d26736bfcd32b`、mamba `104cc47e9101e5401a675fa2b784f2952b9b037f3b1dd83b5ac544394e95d028`。ビルドログとコマンドは `data/results/training/semamba-tensor-20261101/` の `build-tensor-*-ninja.log` にある。CUDA Toolkit 12.9 と PyTorch の CUDA 12.4 はマイナーバージョンが異なり、動作確認が必要。

融合版導入前と同じ reference 経路の Generator 単体を、tensor の A100、3 秒相当の合成 STFT、batch 1、初回実行で同期して測定した。forward 6.146 秒、backward 31.731 秒、peak allocated 7,685,064,192 bytes。これは GAN 全体の step、実データ取得、定常状態を測った数字ではない。再現コード `tests/training/reference/benchmark_semamba_scan.py`、結果 `generator-reference.json`。

`turing` の空き Quadro RTX 5000 で小さい mixer を同一重み・入力で比較した。事前に出力 `atol=2e-5, rtol=2e-4`、勾配 `atol=2e-5, rtol=3e-4` を設定し、reference / fused / 公式 Mamba の全組み合わせが通過した。最大相対 L2 誤差 1.03e-6、最小 cosine 0.9999999999994。実行コード `tests/training/reference/compare_semamba_backends.py`、一次結果 `mixer-turing.json`。CUDA 専用 unit test も turing で成功した。

`turing` の同一 3 秒 VoiceBank ペア・同一初期重みの GAN 1 更新で、公式 / fused の Generator 損失は 15.682699 / 15.682700、波形差は最大 8.70e-7、Generator 勾配は最大差 0.00525、相対 L2 差 0.01549、cosine 0.999881、更新重みの最大差 0.000400。Discriminator 勾配の相対 L2 差は 1.58e-7。これは初回 1 更新であり、非決定性や微小勾配の影響を切り分けていないので学習全体の同等性 GO にはしない。2 実装の入力 SHA-256 は一致。ログと比較は `update-{official,fused}-turing*.log`、`compare-official-fused-turing.json`。所要時間は公式 5.71 秒、fused 2.49 秒だが初回実行順やドライバー差を含み、定常速度比較ではない。

通常 unit test は `OMP_NUM_THREADS=1` で training 74、SE 32（CUDA 専用 1 skip）、evaluation 86、ASR 5、全て成功。ログは同結果ディレクトリの `test-*.log`。CUDA 専用 test の turing 結果は `test-fused-turing.log`。既存本番 DB は変更していない。

追加: sm_80 の `selective_scan_cuda` / `causal_conv1d_cuda` のビルドは完了し、`.venvs/semamba-sm80/` に置いた。既存 `.venvs/semamba` を変更せず、このディレクトリを `PYTHONPATH` の先頭にして利用する。tensor の GPU 5 上で CUDA 単体 forward/backward test が成功した（`test-fused-tensor-v3.log`）。ソース全体を `PYTHONPATH` に追加すると未使用 Mamba3 と既存 Triton の組合せが import に失敗するため、コンパイル済み `.so` のみを先頭に置く。構築手順は各元アーカイブを `.venvs/semamba-sm80-build/` 等に展開し、`setup.py` の CUDA `cc_flag` を `['-gencode', 'arch=compute_80,code=sm_80']` に限定したうえ、tensor 上で各ディレクトリで次を実行した（`TASK_ROOT` はこの repository の絶対パス）。

```sh
CUDA_HOME="$TASK_ROOT/.venvs/cuda-nvcc" \
PATH="$TASK_ROOT/.venvs/semamba/bin:$TASK_ROOT/.venvs/cuda-nvcc/bin:$PATH" \
MAX_JOBS=3 "$TASK_ROOT/.venvs/semamba/bin/python" setup.py build_ext --inplace
# 各ディレクトリの selective_scan_cuda*.so と causal_conv1d_cuda*.so を
# 新規 .venvs/semamba-sm80/ にコピーする。
```

同一の 3 秒 VoiceBank ペア・初期重みでの turing 1 更新では reference 113.71 秒、fused 2.49 秒（45.7 倍）。これは初回 1 更新の観測値であり tensor の step 速度ではない。公式・reference・fused の 3 組はすべて入力 SHA-256 が一致。Generator 勾配の相対 L2 差は公式/reference 1.23%、公式/fused 1.55%、reference/fused 1.72%。同一公式の別プロセス再実行でも 1.24%、同一 fused では 2.04% あった。同一実装の変動と同程度であるが、20 step や DDP の一致を示すものではない。閾値 `max(1e-8, 各 tensor 最大絶対勾配の 0.1%)` より大きい勾配にも符号差があり、単に極小勾配の符号差とは断定しない。3 組の詳細は `compare-*-turing-v2.json`、同一実装の再実行は `repeat-turing/compare-*.json`。

公式配布 checkpoint を `load_enhancer` に strict load し、実 noisy 発話の先頭 47,993 samples について turing CUDA で `enhance` の reference / fused を比べた。両方とも入力長を返し、最大波形差 5.36e-6、相対 L2 6.35e-7、cosine 0.999999999999799（`api-turing.json`）。CPU fallback と checkpoint キーは通常 unit test でも確認した。824 発話の fused 再評価は未実施。

### tensor 実データ・2 GPU（中間結果）

Tensor GPU 5 で実ペア 3 秒・batch 1 の GAN 1 更新を比較した。前の 1 更新では既定の cuDNN TF32 が有効だったため、既存比較と同じ `cudnn.benchmark=False`、`cudnn.deterministic=True`、matmul / cuDNN TF32 無効、`CUBLAS_WORKSPACE_CONFIG=:4096:8` で別実行した（完全決定性は要求していない）。1 更新の reference / fused / 公式は 39.01 / 2.67 / 6.21 秒、peak allocated は 9,495,474,176 / 9,663,924,736 / 7,208,575,488 bytes。初回実行なので定常速度ではないが、reference に対する fused の高速化は約 14.6 倍。入力ハッシュは 3 者で一致。reference / fused の波形最大差 1.98e-6、Generator 勾配の相対 L2 差 2.06%、cosine 0.999800、公式 / fused はそれぞれ 1.81e-6、1.82%、0.999841。Discriminator 勾配の相対 L2 差は 1.62e-7 以下。同一実装の別プロセス再実行では Generator 勾配の相対 L2 差は fused / 公式とも約 3.9e-6 であり、実装間差を単なる同一実装の再実行差とは扱えない。結果は `a100-controlled/` と `a100-controlled-repeat/`。TF32 有効の別観測では reference / fused 波形最大差 0.00126 と勾配相対 L2 差 5.81% で、TF32 の条件を混ぜて結論を出さない。

位相損失だけを出力位相に関して微分した診断では、reference / fused の位相最大差 0.000353 rad、位相損失値はどちらも 4.5484676、位相勾配の相対 L2 差 3.05%、96,681 要素中 31 の符号反転を観測した（`a100-controlled/phase-probe.json`、`tests/training/reference/probe_semamba_phase_sensitivity.py`）。非滑らかな位相損失による微小な forward 差の増幅を示唆するが、Generator 勾配差の全てを説明する実験ではない。許容範囲を事後に拡大せず、この差の影響を引き続き調べる。

A100 40 GB の空き 2 枚で 48,000 samples、各 rank batch 4、蓄積 1、global batch 8 の独立・公式を各 20 更新した。64 件の保存済み VoiceBank crop を 8 件ずつ使い、8 更新ごとに繰り返す短時間検証であって epoch 学習ではない。rank ごとの入力 SHA-256 は全 20 step で一致。実験の両 Discriminator と Generator は DDP で包み、保存した全更新で rank 間重み最大差 0。初回を除く mean step 時間は公式 2.918 秒、独立 3.129 秒。ピーク allocated は公式 28,307,454,464 bytes、独立 37,543,592,448 bytes、OOM なし。GPU 40 GiB のため独立版の allocated ベース余裕は約 5.0 GB だが、実際の reserved / 他プロセスを含めると余裕はさらに小さい。20 step の Generator 重み最大差は公式 / 独立 0.003939、同一独立の再実行間 0.003894。公式 / 独立の rank 0 Generator 損失は step 1: 17.35006 / 17.35008、step 4: 20.03000 / 20.03136、step 20: 16.69110 / 16.73986。全 rank・全損失成分・速度は `a100/ddp-comparison.json`、checkpoint 比較は `compare-ddp-official-independent-{1,4,20}.json` と `compare-ddp-repeat-{1,4,20}.json`。

各 rank batch 2 × 蓄積 2 の独立 DDP 1 更新も完了し、Generator と両 Discriminator の勾配・重みは rank 間で一致。ピーク allocated 約 18.92 GB。native batch 4 と比べて Generator 更新後の重み最大差 0.000398、MRD 最大差 0.000178。batch の分割により演算と丸め順が変わり、非滑らかな損失と AdamW の更新が差を増幅し得るが、内訳の完全な因果は未特定。`a100/ddp-accum/`、`a100/compare-accum1.json`。1 step checkpoint から step 4 の再開も行い、Optimizer・モデルを復元したが、Generator 重み最大差 0.000840 が残った（`a100/ddp-resume/`、`compare-resume4.json`）。この最初の再開 probe は Trainer 全体や rank ごとの RNG 復元を検証したものではない。RNG を含む再開 probe の初回はスクリプトの必須引数漏れで失敗し、修正後を別ディレクトリで実行した。

RNG 込み task checkpoint の再開 probe は引数漏れを直して再実行し、step 4 まで完了した。連続実行と再開後の CPU / CUDA RNG state は一致したが、Generator 勾配の相対 L2 差 20.4%、cosine 0.9792、重み最大差 0.000933 が残った。Discriminator 勾配の相対 L2 差は 4.53e-6。`a100/ddp-resume-rng-v2/`、`compare-resume-rng4*.json`。これは診断用 task checkpoint の結果で、通常 Trainer の checkpoint・rank ごとの RNG を全て検証したという意味ではない。

公式 / 独立 DDP の保存勾配は、Generator の相対 L2 差 / cosine が step 1: 0.0357 / 0.99937、step 4: 0.394 / 0.9206、step 20: 1.804 / -0.681。両 rank の入力 hash、rank 間勾配・重み同期、checkpoint の CPU / CUDA RNG は一致。公式を別プロセスで再実行した場合も step 4 / 20 の Generator 勾配 cosine は 0.964 / 0.859 と変動するが、公式 / 独立の step 20 差はそれより大きい。`compare-ddp-gradients-{1,4,20}.json` と `compare-official-repeat-gradients-{1,4,20}.json`。step 20 の逆向き勾配には phase decoder 層が大きく寄与する（`gradient-layer-diagnosis.json`）。この不一致の原因・更新への長期影響は未解決である。native batch と蓄積の step 1 では Generator 勾配の相対 L2 差 4.88e-5、cosine 0.9999999989、Discriminator は相対 L2 差 2.13e-6 だった（`compare-ddp-accum-gradients-1.json`）。

20 step の GPU 常駐 crop から、10,700 件 / global batch 8 を最終不足分も含む 1,338 更新として単純外挿すると、公式 / 独立 DDP の経過時間は 100 更新 4.86 / 5.22 分、1 epoch 1.08 / 1.16 時間、3 epoch 3.25 / 3.49 時間、10 epoch 10.84 / 11.63 時間。2 GPU の GPU 時間はそれぞれ経過時間の 2 倍。データ読み出し、最後の不足 batch の取り扱い、validation、checkpoint、起動・障害復旧は含まず、保証値ではない。1 GPU の global batch 8 は tensor で実測していない。実行へ進む場合の暫定候補は 2 GPU、global batch 8、各 GPU batch 4、蓄積 1、16 kHz / 48,000 samples、seed 1234、AdamW lr 2e-4・betas (0.8, 0.99)・weight decay 0.01、ExponentialLR gamma 0.99（epoch ごと）、`voicebank_28_train`、checkpoint 1000 更新ごとである。公式 train から 2 話者を validation に分けたためデータ構成は公式と異なる。validation 間隔と本実験で測っていない PESQ / STOI / SI-SDR は長時間実行前に決めて検証する。PESQ の新規採点をしていないため今回の NA 件数は未計測であり、0 とみなさない。

独立実装の 20 更新を勾配保存付きで別プロセス再実行した。Generator 勾配 cosine は同一独立間で step 1 / 4 / 20 = 1.000 / 0.799 / 0.855、同一公式間では 1.000 / 0.964 / 0.859。公式 / 独立間では 0.999 / 0.921 / -0.681 で、step 20 の方向不一致は今回観測した同一実装間の変動より大きい。CPU / CUDA RNG state、全 rank の入力と学習率は一致した。`a100/compare-independent-repeat-gradients-{1,4,20}.json`。ここで比較したのは限られた seed・crop のみであり、原因は位相損失以外も含めて未確定。したがって長時間学習の移行判定は NO-GO とする。

追加診断として、独立ネットワークはそのまま、外部 `mamba-ssm` の mixer 全体の融合演算を使う試験コード `tests/training/reference/probe_semamba_full_mamba.py` を実行した。A100 の同一 3 秒ペア 1 更新で公式に対する Generator 勾配の相対 L2 差は 0.874%、cosine 0.999962 となり、scan のみ融合の 1.82% より縮小したが、同一公式再実行の約 0.00038% より大きく、位相・更新の相違も残る。`a100-full-mixer/compare-official-full.json`。この経路は原因切り分け用であり、本番バックエンドに採用していない。拡張の元アーカイブは `sources/` に保存し、両拡張のビルド後 SHA-256 は causal `7329b400fcaf4631b94fe5a29ed44f58689836ff92772a58e956bb4291489d49`、scan `dc2ea4b44f44f6d3c31b4235794d99d605a5d42f50bf8bbd31500425ccd266f0`。tensor で両 `.so` の import と公式 Mamba の fast path を確認した。

## SEMamba++ fused mixer・tensor 最終判定（2026-11-01）

前節の NO-GO は、selective scan だけを融合し softplus と因果畳み込みを PyTorch 側に残した経路の結果である。その経路は公式 fast path と backward がずれ、20 step の Generator 勾配が公式再実行より大きく離れた。`scan_backend: fused` は CUDA 上で外部 `mamba-ssm` の `mamba_inner_fn` と Triton RMSNorm を使うように変えた。ネットワーク構造と checkpoint キーは変えていない。CPU は reference のままである。公式 SEMamba++ のソースはコピーしていない。

比較条件は `CUBLAS_WORKSPACE_CONFIG=:4096:8`、cuDNN benchmark オフ、deterministic オン、matmul / cuDNN の TF32 オフ。ホストは tensor、GPU は物理 5 と 6（A100-PCIE-40GB、compute capability 8.0、開始時の使用メモリ 13 MiB）。拡張は既存の `.venvs/semamba-sm80`（sm_80）を `PYTHONPATH` の先頭に置いた。結果は `data/results/training/semamba-tensor-20261101/a100-inner/`。

同一重みの mixer では、公式 `Mamba` と fused の出力最大差は 0 だった（`mixer.json`）。3 秒・batch 1 の GAN 1 更新では、出力波形の最大差も 0、Generator 勾配の相対 L2 差は 1.36e-5、cosine は 0.9999999999、最大絶対差は 2.38e-6 だった。入力 SHA-256 は一致。初回 1 更新の所要時間は公式 4.47 秒、fused 4.46 秒、peak allocated は 7.21 GB と 7.49 GB（`compare-official-fused.json`）。同じ入力の reference は 38.93 秒、peak allocated 9.50 GB だった。reference と fused の波形最大差は 1.32e-6、Generator 損失は同じ値、勾配の相対 L2 差は 1.13%、cosine は 0.999936 である（`compare-reference-fused.json`）。小さい mixer の事前許容は CUDA 単体テストで満たしたが、3 秒 GAN の勾配最大差 0.0058 はその許容を超える。方向は一致している。1 更新の 4.46 秒は起動を含む。定常速度は下の 2 GPU 測定を使う。

2 GPU、各 rank batch 4、蓄積 1、global batch 8、48,000 samples を 20 更新した。64 件の保存 crop を繰り返す短時間検証であり、epoch 学習ではない。初回を除く平均 step は公式 2.901 秒、fused 3.136 秒（公式の 1.08 倍）。peak allocated は公式 26.36 GiB、fused 26.70 GiB。OOM は無い。全保存 step で rank 間の重みと勾配の最大差は 0、入力 hash と学習率は一致、保存した CPU / CUDA RNG も一致した。

Generator 勾配の cosine は、公式対 fused で step 1 / 4 / 20 が 1.000 / 0.944 / -0.626、同じ公式実装の別プロセス再実行で 1.000 / 0.942 / -0.607 だった。step 20 の相対 L2 は 1.63 と 1.72 である。損失は step 20 で公式 16.708、再実行 16.739、fused 16.637。逆向きの勾配スナップショットは公式自身の再実行でも起きる。重みの相対 L2 は公式対 fused の step 20 で 0.00126、cosine は 0.9999992 だった（`ddp-compare.json`、`ddp-official-repeat-compare.json`）。

fused の step 1 checkpoint から step 4 まで再開した。連続実行との Generator 損失は 20.02990 と 20.02962、重みの相対 L2 は 6.60e-5、勾配 cosine は 0.985、CPU / CUDA RNG は一致、rank 間最大差は 0 だった（`ddp-resume/`）。公式再実行の step 4 勾配 cosine 0.942 より近い。

10,700 件を global batch 8 の 1,338 更新として、初回を除く平均 step で外挿すると、公式 / fused の経過時間は 100 更新 4.84 / 5.23 分、1 epoch 1.08 / 1.17 時間、3 epoch 3.23 / 3.50 時間、10 epoch 10.78 / 11.66 時間。2 GPU の GPU 時間は経過時間の 2 倍。データ読み出し、不足 batch、validation、checkpoint、起動は含まない。1 GPU の global batch 8 は未計測で、batch 4 がすでに約 27 GiB allocated なので 40 GB 1 枚への収まりは確認していない。

PESQ は今回採点していない。NA 件数は未計測であり、0 とみなさない。

通常 unit test は `OMP_NUM_THREADS=1` の lab で training 74、SE 32（CUDA 専用 1 件 skip）、evaluation 86、ASR 5、失敗 0。CUDA 専用の forward/backward 比較は tensor GPU 5 で成功した。ログは `a100-inner/test-*.log`。

| 検証項目 | 判定 |
|---|---|
| tensor環境での公式実装 | GO |
| tensor環境での独立実装 | GO |
| 独立scanの高速化 | GO |
| referenceとのforward一致 | GO |
| referenceとのbackward一致 | PARTIAL |
| 公式との学習挙動同等性 | GO |
| 実モデルDDP | GO |
| global batch 8 | GO |
| checkpoint再開 | GO |
| 長時間学習への移行可否 | GO |

この GO は、fused で 3 秒・global batch 8・2 GPU の学習が公式と同じ時間規模で動き、1〜20 step の差が公式再実行の範囲に収まる、という意味である。長時間学習のあと品質が公式と一致することは証明していない。前節の scan のみ融合の NO-GO は、この fused mixer には当てはめない。


## SEMamba++ validation の実装途中

独立タスクの validation は既存の SI-SDR、STOI、PESQ 評価関数を呼び、波形長と発話単位の結果を記録する。PESQ の境界診断を含む失敗はそのまま記録する。SEMamba++ では PESQ による best 保存を無効化している。SI-SDR による best 選択、完全な学習 checkpoint、再開と公式実装を含む同条件の実音声検証は未完成で、長時間学習の開始条件を満たしていない。

### 長時間比較の暫定見積もり（実行不可）

既存の A100 40 GiB ×2、各 rank batch 4、16 kHz・48,000 samples の 20 更新実測では、公式 2.901 秒、独立 fused 3.136 秒/更新、peak allocated 26.36 / 26.70 GiB だった。現在の Trainer は端数を落とすので 10,700 件・global batch 8 では 1 epoch 1,337 更新、学習部分だけの単純外挿は約 1.08 / 1.16 時間である。既報の 1,338 更新は端数を含めた外挿で、現在の Trainer の処理とは異なる。データ供給、872 件の validation、保存、起動、復旧を含まず、実行時間の見積もりとしては不足している。保存時間・容量、CPU RAM、実データでの validation 時間と GPU reserved memory の測定も必要。

候補は両実装で同じ初期値、順序、crop、global batch 8、AdamW 2e-4・betas (0.8, 0.99)・weight decay 0.01、epoch ごとの ExponentialLR 0.99。比較前に、同じ端数処理、validation の固定集合と間隔、checkpoint 間隔、SI-SDR の best 選択、公式側の Trainer 接続と再開を検証する。検証セットの全 872 件で指標が安定し、noisy より改善することと、事前に固定する許容差での両者の品質同等性を別々に判定する。許容差と停止規則は未決定。test 824 件はモデル選択後に固定条件で評価し、その後に固定 ASR の WER を調べる。上記の未完成項目があるため、長時間学習用コマンドや時間予算を実行可能なものとして提示しない。

## 5 モデル再検証の着手記録（2026-11-01、未完了）

`dev` の `7d434c6169c2dee67d28201d73dd8b26bc5fe0a3` で開始した。開始時の作業ツリーはクリーン。共有 GPU 4 枚は全て別プロセスが使用率 100% で使用していたため、今回の GPU 比較は行っていない。既存の公式比較スクリプトには既存結果ディレクトリへ固定出力するものがあり、そのまま実行していない。過去の比較値は今回の結果ではない。

TF-GridNet ms-snsd について、既存比較の事前許容（forward / loss 1e-5、勾配 1e-4 および大きい勾配の cosine 0.999）を使い、補助診断として CPU 上の同一プロセスで公式と独立の 1 更新を比較した。seed 3 の 2,048 サンプル合成波形、batch 1 であり、実音声や別プロセス 4 系統の比較ではない。公式重みを独立実装に strict load した。Forward 最大差 0、損失差 0、勾配の要素最大差 0.001953125、層単位の最小 cosine -0.17649、更新後の重み最大差 0.00012050。勾配差は事前許容を超え、原因と最初に差の生じる演算は未特定。層単位 cosine の最小値を全体の cosine と解釈しない。記録は `data/results/training/five-model-audit-20261101-WV8BjA/tf-gridnet-cpu.json`。次のコマンドは新規ファイルを指定して再実行する（既存ファイルは上書きしない）。

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venvs/lab/bin/python tests/training/reference/probe_tf_gridnet_cpu.py data/results/training/NEW_DIRECTORY/tf-gridnet-cpu.json
```

この補助診断だけではモデルの GO 判定はできない。5 モデルの公式 A/B・独立 A/B、実入力の一致、GPU 上の 1 step・短時間学習、checkpoint 再開は今回未検証。MossFormerGAN の現行 Discriminator は公開版と構造が異なり、公式 GAN 学習の再現は未達である。

今回の lab 環境回帰テストは data 12/12、training 76/76、evaluation 86/86、SE 27/28（CUDA extension 専用 1 skip）、ASR 5/5 が成功した。training の最初のサンドボックス内実行はローカルソケットの `PermissionError` により 8 エラーとなり、制限外で再実行して成功した。これらは上記 4 系統の公式比較の代わりではない。

### TF-GridNet CPU backward 追加診断

同一 seed・入力・初期重みで上記の試験を再実施した。保存先は `data/results/training/tf-gridnet-gradient-20261101-KguXGL/`（合成音声、同一プロセスの補助診断）。修正前の全勾配 norm は公式 74192.1856、独立 74192.1771、相対 L2 差 2.36e-7、cosine 0.999999999999976、符号不一致 1696 要素。最大差 0.001953125 は `blocks.0.attn_conv_Q_3.0.weight[0,0,0,0]`（公式 7643.586914、独立 7643.584961）。最小 cosine -0.17649 は `blocks.0.attn_conv_K_2.2.beta`、norm は公式 5.63e-5、独立 3.64e-5。公式勾配絶対値 1e-4 以上の要素では符号不一致はなかった。

同一の公式損失を両モデルに適用するとモデル勾配差は 0。一方、同一の出力波形に両損失を適用すると波形勾配に最大 1.53e-5 の差があった。操作を段階的に比較すると、損失の `estimate * scale` を時間損失と全 STFT 損失で一つの計算ノードとして共有することが差を生んだ。公式は各分岐で乗算を別々に計算するため、backward の浮動小数点加算順が異なる。公式と同じ分岐に修正するとこの条件の波形、損失、全パラメータ勾配と Adam 更新後重みの最大差は 0。事前許容を変更していない。CPU の単一入力の結果であり、GPU や別プロセスでの一般的な一致を意味しない。`probe_tf_gridnet_cpu.py` の `--legacy` は修正前の計算グラフを再現する。

## MossFormerGAN Discriminator と GAN 1 step（2026-11-01）

参照は ClearerVoice-Studio commit `6b3774dc79c46ae8bed2a4fa5f706f0ac8c75c61` の `train/speech_enhancement/models/mossformer_gan/`、`losses/loss.py`、`solver.py`、`train.py`。独立側の実行時は参照ツリーを import しない。Apache-2.0 の公開コードに含まれる AdaptCNN / PoolAttFF の出典 NISQA は MIT、Conformer の出典 lucidrains/conformer も MIT。著作権・ライセンスと改変箇所は `src/training/se/mossformer_discriminator/NOTICE.md` と同ディレクトリの LICENSE に記載した。

| 公式の実行経路 | 学習側の対応 |
|---|---|
| 圧縮済み clean / 生成 magnitude `[B,1,F,T]` を 15 frame、hop 4 で切り、2 channel に連結 | `segment_specs`、`Discriminator.forward` |
| 6 convolution・BatchNorm・adaptive max pool・Dropout2d、384 次元 | `AdaptCNN` |
| 384 次元、4 head、kernel 31 の Conformer 2 層、相対位置埋め込み | `ConformerBlock` ×2 |
| feed-forward attention pooling、`sigmoid(slope * x)`、出力 (0,1) | `PoolAttFF`、`LearnableSigmoid` |
| G 用 fake、D 用 fake、D 用 real の順に forward。G backward → G clip → D backward → D clip → G/D step | `MossFormerGANTask.train_update`（clip は各 microbatch） |
| AdamW の既定 betas (0.9, 0.999)、weight decay 0.01、G 5e-4、D 1e-3 | サンプル設定。betas と学習率は設定可能 |

公開の `ndf` / `in_channel` は実際の層幅に使われない。Discriminator のパラメータは 8,121,331 個、state dict は 117 キーで、公式から strict load でき、shape と値は一致した。入力は波形ではなく、STFT・圧縮後の magnitude pair である。入力 STFT は Hamming periodic=False・center=True・圧縮 0.3。公式の時間損失は clean 実部と生成虚部を組み合わせ、center=False の iSTFT を使う。

CPU float32・単一スレッド、seed 77、合成入力で比較した。単体 Discriminator は batch 1、201 bin、25 frame、training mode、同じ RNG と初期 state。CNN、Conformer、pool、sigmoid の各出力は最大差 0。敵対損失 `(score - 1)^2` の全パラメータ勾配の norm は両者 17.0850190、入力勾配 norm 1.1830269、生成側入力勾配 norm 1.0604418。いずれも最大絶対差・相対 L2 差 0、cosine は丸めて 1。AdamW 1 更新後も最大絶対差・相対 L2 差 0。

学習タスク比較は公式幅 Generator と Discriminator、batch 1・8,000 samples の同じ合成 clean/noisy、accumulation 1。Generator は両側で同じ独立モデルと初期 state を使い、片側は公式の前処理と公式損失関数、もう片側は実際の `MossFormerGANTask.train_update` を呼んだ。公式 `batch_pesq` は CUDA を固定するため、公式 `pesq_loss` と同じ正規化を CPU 上で呼ぶアダプタに差し替えた。PESQ 自体は両側で実計算した。G/D 損失、Generator の 2 出力、両モデルの clip 後勾配、AdamW 更新後の両モデルの重みはすべて最大絶対差・相対 L2 差 0。G/D 勾配 norm はそれぞれ 6.1821041 / 4.2400952、重み cosine はそれぞれ 0.999999999999949 / 0.999999999999898。これは単一プロセス・単一入力・1 更新の結果であり、GPU、別プロセス、公式設定の accumulation 2、実音声、長時間学習の一致を意味しない。

修正前は時間波形再合成で `angle` を二度呼んでいた。公式のように位相を共有すると、同じ出力でも backward の加算順が変わる。2,000 samples の診断では修正前の Generator 勾配最大差 2.05e-7、更新後の重み最大差 7.65e-4 だったが、共有後は両方 0。独立した再合成勾配テストも追加した。

再実行には上記 commit の公式ツリーが必要。比較用環境には公式 import の `einops`、`joblib`、`librosa`、`pesq` が必要で、今回 `.venvs/lab` に不足分を導入した。比較スクリプトは結果を標準出力に出し、既存の実験結果を上書きしない。

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venvs/lab/bin/python tests/training/reference/compare_mossformer_cpu.py
```

回帰テストは `.venvs/lab`、上記と同じスレッド設定で training 79/79、SE 27/28（CUDA extension 専用 1 skip）、evaluation 86/86、ASR 5/5。失敗・エラー 0。MossFormerGAN の公式幅 1 更新、有限勾配、Generator の推論用 export/load も training テストに含む。

## 4 独立プロセス比較（2026-11-01）

`tests/training/four_process/compare.py` は、公式 A/B・独立 A/B を順番に別 OS プロセスで起動する。各プロセスはモデル、Adam、StepLR、乱数と出力を独立に持ち、終了コード、timeout、標準出力・標準エラーを保存する。出力先は新規ディレクトリに限る。`report` はプロセス状態と保存ファイルを読み直すので、中断・欠落・不正な結果も確認できる。`valid` は実行と条件照合の成否であり、実モデルの数値差の許容判定ではない。数値差は別に確認する。

準備時に初期 state dict と固定の合成 clean/noisy tensor を一度保存する。全プロセスが strict load し、ロード後の state、Optimizer、Scheduler、RNG と波形の内容ハッシュを照合する。合成波形は DB のレシピではないので case_id と crop は NA、順序は `[0]`。16 kHz、batch 1、accumulation 1、float32、AMP/TF32 無効。Discriminator は対象外。各 step に forward、clip 前勾配、更新後重みの snapshot、損失、勾配 norm、学習率、状態ハッシュを保存する。最終 Optimizer、Scheduler、RNG の状態も保存する。報告は最大絶対差、相対 L2、cosine、損失差、勾配 norm 差を出す。cosine は norm の積が 1e-24 以下なら null とし、基準勾配絶対値で分けた集計も出す。

CPU の TF-GridNet ms-snsd は既存 `compare_tf_gridnet.py` の ESPnet モデル・公式損失・独立モデル・独立損失を利用した。seed 3、2048 samples、Adam 0.001、clip 5、3 更新、StepLR を毎更新する短い合成入力の試験であり、公式レシピの epoch scheduler や実音声の検証ではない。保存先 `data/results/training/four-process-compare-20261101-cpu/`。PID は公式 A/B 3176526/3176600、独立 A/B 3176651/3176705。4 比較の全 3 step で forward、loss、勾配、更新後重みの最大絶対差は全て 0。これはその環境・入力での結果である。

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venvs/lab/bin/python tests/training/four_process/compare.py run data/results/training/NEW-FOUR-PROCESS --model tf_gridnet --steps 3
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venvs/lab/bin/python tests/training/four_process/compare.py report data/results/training/NEW-FOUR-PROCESS
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venvs/lab/bin/python -m unittest tests.training.test_four_process -v
```

2026-11-01 時点の実モデル接続は TF-GridNet のみだった。スナップショットは全 step 保存するため、長い実行には向かない。保存した PyTorch ファイルは信頼できるローカル結果だけを読み込む。

GPU 1 step は空き確認後 Quadro RTX 5000 の物理 GPU 1（実行時の `cuda:0`）を使用し、`CUBLAS_WORKSPACE_CONFIG=:4096:8` とした。`data/results/training/four-process-compare-20261101-gpu/` に保存。公式 A/B と独立 A/B はそれぞれ forward・loss・勾配・重みの最大差 0。公式/独立は forward 最大差 0、損失差 0.001953125、勾配最大差 0.00146484375、勾配相対 L2 2.7537e-7、更新後重み最大差 9.1635e-5。損失と勾配の絶対差は既存の許容値 1e-5 と 1e-4 を超え、当時は原因未調査だった。このディレクトリは上書きしていない。原因と再測定は次節。

Mock の 4 プロセス比較は CPU の決定的な線形モデルで、順序を逆に変えても比較値が一致した。入力・重み・Optimizer・勾配の改変、例外、結果欠落、非有限損失をそれぞれ検出した。保存内容の破損も検出する。別の Mock checkpoint 試験では 4 更新の連続実行と 2 更新後に別プロセスで再開した実行のモデル・Optimizer・Scheduler・RNG・データ位置が一致した。

モデル固有処理を `adapters.py` に分けた後、CPU 1 step を `data/results/training/four-process-compare-20261101-cpu-adapter/` で再実行した。4 組の loss・勾配・更新後重み最大差は全て 0。

公式 TF-GridNet の読み込みには既存の `reference_v1()` を使い、ESPnet の encoder/decoder を比較用の PyTorch STFT/iSTFT ラッパーに置き換えている。したがって ESPnet 全体のデータ入力や学習ランチャーの再現ではない。

GPU 試験の再実行例（直前に `nvidia-smi` で空きメモリを確認し、空いている物理 GPU 番号を指定する）:

```bash
CUDA_VISIBLE_DEVICES=FREE_GPU CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venvs/lab/bin/python tests/training/four_process/compare.py run data/results/training/NEW-GPU-FOUR-PROCESS --model tf_gridnet --steps 1 --device cuda:0
```

## 4 独立プロセス比較の再開・GAN・実レシピ（2026-11-02）

比較 worker は `checkpoint.pt` を step ごとに原子的保存する。中身は Generator、Discriminator（無いモデルは持たない）、各 Optimizer、各 Scheduler、global step、epoch、データ位置、Python `random`、NumPy、PyTorch CPU、CUDA を使ったときだけ CUDA RNG、学習設定、実行名、入力 SHA-256 である。`resume` は同じ設定と初期重みの実行だけを継続する。seed、入力ファイル、初期重み、fault が違う checkpoint は拒否する。Mock の連続 4 step と、2 step 後に別プロセスで再開した 4 step は、損失、重み、Optimizer、Scheduler、3 種類の RNG、データ位置、global step が一致した。RNG を別の妥当な状態に差し替えると Mock の数値不一致として検出する。TF-GridNet ms-snsd の CPU 2 step でも、連続実行と 1 step 後の別プロセス再開は損失・重み・RNG・Scheduler が一致した。保存先は `data/results/training/four-process-tfgridnet-20261102-cpu-cont/` と `four-process-tfgridnet-20261102-cpu-resume/`。

MP-SENet は縮小構成（`dense_channel=16`、TS-Conformer 1、8,000 samples）で接続した。更新順は公式と同じで、Discriminator（PESQ 目標、detach した magnitude）のあと Generator である。Optimizer は AdamW が 2 つ。この probe は ExponentialLR を毎 step 進める。本学習は epoch ごとであり、その間隔はこの短い比較では再現していない。CPU 1 step の保存先は `data/results/training/four-process-mpsenet-20261102-cpu/`。公式 A/B と独立 A/B は forward、Generator 損失、Discriminator 損失、勾配、重みの最大差 0。公式/独立は forward と両損失の差 0、勾配最大差 3.87e-7、更新後の Discriminator 重み最大差 0、Generator 重み最大差 7.24e-4（近零勾配の bias）。同じ初期重みから 2 step 進めた `four-process-mpsenet-20261102-cpu-cont/` でも公式どうし・独立どうしは全項目の最大差 0 である。公式/独立は 2 step 目の Generator 損失差 0、Discriminator 損失差 8.61e-9、勾配最大差 2.44e-4、重み最大差 1.26e-3 で、1 step 目の Generator 重み差が次の更新に入っている。連続 2 step と 1 step 後の再開は `four-process-mpsenet-20261102-cpu-resume/` で、同じ実装の損失・両重み・RNG・Scheduler が一致した。

`voicebank_28_train` を `data/db/voicebank.duckdb` から読み、本番 `resource.duckdb` は開いていない。48 kHz から 16 kHz への変換は `RecipeDataset` のロード時リサンプリングである。TF-GridNet の 1 step、segment 2048 は `data/results/training/four-process-voicebank-20261102-cpu/`。4 プロセスは同じ case `03dbca80e082f9b44aa2c7fa693f33d3f6d262c454c6744f3f7416bff963f8da`、crop 24853、resample 後の長さ 50666、clean/noisy の波形ハッシュが一致し、forward・損失・勾配・重みの最大差は 0 だった。

TF-GridNet の GPU 差は、強化波形が一致したあとの 256 点矩形 STFT にあった。PyTorch 2.6 の CUDA では `torch.stft(window=None)` と `torch.ones` が 256 点だけ最大約 1e-5 違い、損失和が float32 の 1 ulp である 0.001953125 動いた。TF32 は無効、cuDNN deterministic は有効、AMP は無効だった。独立損失を ESPnet と同じ `window=None`（pad は従来の明示 reflect）にすると、同じ GPU 上の 1 step で公式/独立の forward・損失・勾配・更新後重みの最大差は 0 になった。保存先は `data/results/training/four-process-compare-20261102-gpu/`（物理 GPU 0、実行時 `cuda:0`、`CUBLAS_WORKSPACE_CONFIG=:4096:8`）。2026-11-01 の測定ファイルは残してある。

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venvs/lab/bin/python tests/training/four_process/compare.py run data/results/training/NEW --model mp_senet --steps 1
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venvs/lab/bin/python tests/training/four_process/compare.py resume data/results/training/EXISTING --steps 4
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venvs/lab/bin/python tests/training/four_process/compare.py run data/results/training/NEW-VB --model tf_gridnet --steps 1 --dataset voicebank_28_train --database data/db/voicebank.duckdb --segment-size 2048 --sample-rate 16000
```

## 5 モデルの 4 プロセス比較（2026-11-02）

SEMamba++、MossFormerGAN、DeepFilterNet3 を `tests/training/four_process/adapters.py` から起動できるようにした。公式プロセスは公式モジュールと公式損失、独立プロセスは `src/training` を使う。初期重みは独立モデルで一度作り、両方へ strict load する。TF-GridNet と MP-SENet のアダプタは変えていない。本番 `resource.duckdb` は開いていない。

実行は `turing`。GPU は Quadro RTX 5000（16 GB）。使用前の使用メモリは 10–18 MiB。SEMamba++ の GPU 実行中に見えた使用量は 277 MiB で、ステップ中のピークは別途測っていない。PyTorch 2.6.0+cu124、CUDA 12.4。worker は TF32 を切り、cuDNN deterministic を有効にし、AMP は使わない。`CUBLAS_WORKSPACE_CONFIG=:4096:8`。

| モデル | CPU 1 step | GPU 1 step | 公式/独立 | 同一実装の再実行 | checkpoint 再開 | 実データ | 20 step | 判定 |
|---|---|---|---|---|---|---|---|---|
| TF-GridNet | PASS（既存、最大差 0） | PASS（既存、最大差 0） | PASS | PASS | PASS（CPU 2 step） | PASS（VoiceBank、segment 2048） | NOT TESTED | GO |
| MP-SENet | PASS（既存、縮小幅） | NOT TESTED | 損失 0、勾配最大差 3.87e-7 | PASS | PASS（CPU 2 step） | NOT TESTED | NOT TESTED | GO（CPU の縮小 probe） |
| SEMamba++ | 公式は NOT TESTED。独立 A/B の損失は一致 | PASS（fused、下記） | 1 step の損失差 4.39e-4 | GPU 損失 0、重み最大差 3.22e-4 | 再開は成功。step 4 の損失差 3.81e-6 | PASS（VoiceBank、16 kHz） | NOT TESTED（4 step まで） | PARTIAL |
| MossFormerGAN | PASS（最大差 0） | 損失差 2.38e-7、勾配最大差 4.43e-4 | CPU は 0。GPU は再実行より勾配差が大きい | CPU は 0。GPU 重み最大差 8.90e-4 | PASS（CPU 4 step、重みと RNG が一致） | PASS（VoiceBank、最大差 0） | NOT TESTED（4 step まで） | GO（CPU）。GPU は再実行差つき |
| DeepFilterNet3 | 損失・forward 0、勾配最大差 2.86e-6 | 損失・forward 0、勾配最大差 3.11e-5 | 1 step は小さい。20 step で損失差 0.195 | CPU 20 step は 0。GPU 1 step の勾配最大差 2.86e-6 | PASS（CPU 4 step、完全一致） | PASS（48 kHz、損失 0、勾配最大差 5.67e-5） | 同一実装は PASS。公式/独立は PARTIAL | PARTIAL |

SEMamba++ の probe は hid_feature 8、TF-Mamba 1、d_state 4、d_conv 2、長さ 8,000、16 kHz である。公開幅は hid_feature 48、4 block、d_state 16、d_conv 4 である。Discriminator は公開の CQT と 3 解像度 STFT のまま省いていない。損失係数と、Discriminator のあと Generator という更新順も公開どおりである。ExponentialLR はこの probe では毎 step 進める。本学習は epoch ごとである。公式コードの `mamba_ssm` Block は Triton RMSNorm を呼び、CPU tensor で `ValueError: Pointer argument cannot be accessed from Triton` になる。CPU の公式 4 プロセスは実行できない。独立どうしの CPU reference 1 step は損失 46.39616012573242 で一致した。保存先 `data/results/training/four-process-semambapp-20261102-cpu/` は公式プロセスが失敗しているので `valid` は false である。

GPU は物理 GPU 0、実行時 `cuda:0`、`.venvs/semamba`、`scan_backend: fused`。1 step の保存先は `four-process-semambapp-20261102-gpu/`。公式 A/B は forward 最大差 0、損失差 0、勾配最大差 3.80e-7、更新後重み最大差 3.22e-4。独立 A/B は勾配最大差 2.61e-7、重み最大差 2.85e-4。公式/独立は forward 最大差 1.49e-7、Generator 損失差 4.39e-4、Discriminator 損失差 4.77e-7、勾配最大差 1.04e-4、相対 L2 6.32e-5、cosine 0.999999998、重み最大差 3.96e-4。同じ初期重みで CPU reference と GPU fused の独立実装を 1 step 比べると、forward 最大差 0.0728、相対 L2 0.178、cosine 0.984、勾配最大差 0.151、相対 L2 0.0690、cosine 0.9976、損失差 0.0111、重み最大差 4.00e-4 だった。fused どうしの再実行より、reference と fused の波形差の方が大きい。

4 step の連続実行は `four-process-semambapp-20261102-gpu-cont/`、2 step 後の別プロセス再開は `four-process-semambapp-20261102-gpu-split/`。再開後の global step、データ位置、Scheduler、RNG ハッシュは連続実行と一致した。step 4 の公式損失差は 3.81e-6、独立損失差は 7.63e-6 で、重みハッシュは一致しなかった。checkpoint の読み込みは成功し、fused カーネルの数値は再現していない。4 step 終了時の公式/独立の勾配最大差は 0.0230、独立どうしの再実行は 0.00740 だった。公開幅の 20 step は、1 step が約 30 秒で勾配スナップショットが 1 プロセスあたり約 95 MB なので、20 step × 4 プロセスで約 40 分、約 8 GB になる。今回は行っていない。

VoiceBank は `voicebank_28_train`、16 kHz、segment 8,000。ファイルは 48 kHz で、リサンプリングは `RecipeDataset` のロード時である。case `03dbca80e082f9b44aa2c7fa693f33d3f6d262c454c6744f3f7416bff963f8da`、crop 14288、長さ 50666。4 プロセスの case と波形ハッシュは一致した。保存先 `four-process-semambapp-20261102-voicebank/`。公式/独立の forward 最大差 2.31e-7、損失差 1.91e-6、勾配最大差 1.82e-5。`.venvs/semamba` の DuckDB 1.4.3 は `voicebank.duckdb` を開けない。準備は `.venvs/lab`、worker は `FOUR_PROCESS_PYTHON=.venvs/semamba/bin/python` とした。worker は準備済み波形を使い、DB を開き直さない。

MossFormerGAN は公開幅 `num_channel=64`、8,000 samples、16 kHz。公式側は公式 `SyncANet` と公式 `Discriminator`、公式 `loss_mossformergan_se_16k` である。公式 `batch_pesq` は CUDA tensor を固定で返すため、同じ PESQ 正規化の結果を step の device へ返すように差し替えた。PESQ が取れないバッチは止める。更新は Generator backward、clip 10、Discriminator backward、clip 10、そのあと両方 step。Discriminator の学習率は 0.001、Generator は 0.0005。AdamW betas `(0.9, 0.999)`、weight decay 0.01。公開ループに per-step scheduler は無い。probe は学習率を変えない counter を checkpoint に入れる。CPU 1 step と 4 step は 4 比較とも forward、両損失、勾配、重みの最大差 0。保存先は `four-process-mossformergan-20261102-cpu/` と `four-process-mossformergan-20261102-cpu-cont/`。連続 4 step と、2 step 後に再開した 4 step（`four-process-mossformergan-20261102-cpu-split/`）は損失、重み、Optimizer、Scheduler、RNG が一致した。

GPU は物理 GPU 1。保存先 `four-process-mossformergan-20261102-gpu/`。公式どうしは損失差 0、forward 0、勾配最大差 2.68e-7、重み最大差 8.90e-4。公式/独立は損失差 2.38e-7、Discriminator 損失差 4.47e-7、forward 最大差 9.42e-6、勾配最大差 4.43e-4、相対 L2 8.17e-4、cosine 0.999999666、重み最大差 0.00200。勾配の公式/独立差は再実行差より大きい。VoiceBank は同じ case、crop 14288、48 kHz から 16 kHz。CPU の 4 比較は最大差 0。保存先 `four-process-mossformergan-20261102-voicebank/`。20 step は 4 step の結果が 1 プロセスあたり 477 MB なので、20 step × 4 で約 10 GB になる。今回は 4 step で止めた。

DeepFilterNet3 は出荷幅、48 kHz、長さ 4,800。公式 `DfNet` と公式 `MultiResSpecLoss`、`LocalSnrLoss` を使う。解析 STFT と ERB は両側とも独立 `DFStream` で、step ごとに reset する。`libdf` は実行していない。AdamW は上記の出荷設定。cosine は step の前に学習率と weight decay を書き、step のあとで index を進める。CPU 1 step（`four-process-deepfilternet3-20261102-cpu/`）は公式どうしも独立どうしも最大差 0。公式/独立は損失差 0、forward 0、勾配最大差 2.86e-6、相対 L2 2.78e-7、cosine は 1 に丸まる、重み最大差 5.13e-6。連続 4 step と再開（`four-process-deepfilternet3-20261102-cpu-cont/` と `cpu-split/`）は損失、重み、Optimizer、Scheduler、RNG が一致した。

CPU 20 step は `four-process-deepfilternet3-20261102-cpu-20/`。公式どうしと独立どうしは 20 step すべて損失・勾配・重みの最大差 0。公式/独立は step 1 の損失差 0、step 12 の勾配相対 L2 1.01、cosine 0.784、step 16 の損失差 0.175、勾配最大差 20.5、cosine 0.242、step 20 の損失差 0.195、forward 最大差 0.00311、重み最大差 0.00857。同じ実装の再実行は 0 なので、この増加は GPU の非決定性ではなく、公式損失と独立損失の更新が step を重ねて離れた結果である。2026-11-01 の `official-compare-20261101/deepfilternet3.json` でも 20 step の損失差は 0 ではなかった。

GPU は物理 GPU 0。保存先 `four-process-deepfilternet3-20261102-gpu/`。公式どうしは損失・forward 0、勾配最大差 2.86e-6、重み最大差 9.18e-6。公式/独立は損失・forward 0、勾配最大差 3.11e-5、相対 L2 2.64e-6、重み最大差 9.10e-6。VoiceBank は 48 kHz のまま、segment 4,800。case は同じ `03dbca80…`、crop 95991、ファイル長 151996。公式/独立の損失・forward は 0、勾配最大差 5.67e-5。保存先 `four-process-deepfilternet3-20261102-voicebank/`。

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venvs/lab/bin/python tests/training/four_process/compare.py run data/results/training/NEW-MOSS --model mossformergan --steps 1
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venvs/lab/bin/python tests/training/four_process/compare.py run data/results/training/NEW-DF --model deepfilternet3 --steps 1
CUDA_VISIBLE_DEVICES=FREE_GPU CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venvs/semamba/bin/python tests/training/four_process/compare.py run data/results/training/NEW-SEMAMBA --model semambapp --steps 1 --device cuda:0 --scan-backend fused
FOUR_PROCESS_PYTHON=.venvs/semamba/bin/python CUDA_VISIBLE_DEVICES=FREE_GPU CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venvs/lab/bin/python tests/training/four_process/compare.py run data/results/training/NEW-SEMAMBA-VB --model semambapp --steps 1 --device cuda:0 --scan-backend fused --dataset voicebank_28_train --database data/db/voicebank.duckdb --sample-rate 16000 --segment-size 8000
```

## 最終受入（2026-11-02）

総合は **PARTIAL**。学習処理の公式再現について、5 モデルとも更新順・損失式・optimizer の対応は取れている。長時間学習後の音質同等性は対象外で、未判定である。DeepFilterNet3 の 20 step 勾配、SEMamba++ の公開幅、MossFormerGAN の GPU 複数 step は、1 step の一致だけでは GO にしない。

HEAD は `2af91a3dfc0f95c0c18770bf94890b2faf2c3d77`。この節の計測スクリプトと MP-SENet probe の `--scheduler-step` は作業ツリーにあり、未 commit である。許容値はこれまでの forward / 損失 `atol=1e-5`、勾配 `atol=1e-4` かつ大きい勾配の cosine `>= 0.999` のままである。

| モデル | Forward | Backward | Optimizer | 4プロセス | 再開 | 実データ | 総合 |
|---|---|---|---|---|---|---|---|
| MP-SENet | GO | GO（1 step） | GO | GO（縮小幅、epoch scheduler） | GO | GO（1 step） | GO（縮小幅）。公開幅の再測はしていない |
| SEMamba++ | PARTIAL | PARTIAL | GO | PARTIAL（縮小幅、CUDA） | PARTIAL | GO（縮小幅、1 step） | PARTIAL |
| TF-GridNet | GO | GO | GO | GO | GO（CPU 2 step） | GO | GO |
| DeepFilterNet3 | GO（1 step） | PARTIAL（20 step） | GO | GO（同一実装） | GO | GO（1 step） | PARTIAL |
| MossFormerGAN | GO | PARTIAL（GPU） | GO | GO（CPU） | GO（CPU 4 step） | GO（CPU） | PARTIAL |

### MP-SENet

probe に `--scheduler-step epoch` を足した。このモードは ExponentialLR を step 内で進めない。公開 `train.py` は epoch の終わりにだけ `scheduler_g.step()` と `scheduler_d.step()` を呼ぶ。4 step の学習率は公式・独立とも 5e-4 のまま、scheduler ハッシュも一致した。幅は `dense_channel=16`、TS-Conformer 1 で、公開幅の 64 / 4 ではない。

CPU、`four-process-mpsenet-20261102-epoch/`。公式どうし・独立どうしは 4 step すべて損失・forward・勾配・重みの最大差 0。公式/独立は step 1 の両損失差 0、勾配最大差 3.87e-7、重み最大差 7.24e-4。step 4 の Generator 損失差は 5.48e-6（損失 2.799 に対して）、勾配最大差 0.0133、相対 L2 0.0665、cosine 0.9978、重み最大差 1.71e-3。step 1 の重み差は、以前と同じく AdamW の epsilon 1e-8 が微小勾配を O(lr) まで広げる分である。連続 4 step と、2 step 後に別プロセスで再開した 4 step（`four-process-mpsenet-20261102-epoch-split/`）は、公式・独立のどちらも損失、重み、optimizer、scheduler、RNG が一致した。

VoiceBank は `voicebank_28_train`、16 kHz、segment 8,000。case `03dbca80e082f9b44aa2c7fa693f33d3f6d262c454c6744f3f7416bff963f8da`、crop 14288、長さ 50666。保存先 `four-process-mpsenet-20261102-voicebank-epoch/`。公式どうしは最大差 0。公式/独立は両損失差 0、forward 0、勾配最大差 1.04e-7、相対 L2 5.74e-8、cosine は 1 に丸まる、重み最大差 5.58e-4。

### TF-GridNet

2026-11-02 に tensor の物理 GPU 0（A100-PCIE-40GB、開始時 932 MiB、`CUBLAS_WORKSPACE_CONFIG=:4096:8`）で 1 step をやり直した。保存先 `four-process-tfgridnet-20261102-tensor/`。公式どうし、独立どうし、公式/独立のすべてで forward、損失、勾配、更新後重みの最大差は 0。損失の乗算順と、矩形 STFT の `window=None` は、この A100 でも差を戻していない。CPU の VoiceBank 1 step と CPU の再開は前節の記録のままである。

### DeepFilterNet3

20 step の損失差 0.195 は、別の損失式ではなかった。保存済み入力で公式と独立を同じプロセスに並べると、step 1 の波形差は 0、損失はどちらも 46.343246、勾配の最大差 2.86e-6 は最大勾配約 19.7 の float32 で 1 ulp 程度、相対 L2 は 2.78e-7 である。同じ公式モデルの重み 1 要素だけを `nextafter` で 1 ulp 動かした対照も、step 16 で `df_dec.df_convp.1.weight` の勾配 norm が 111 と 30 に割れた。独立実装との差は、その 1 要素の摂動より早く広がる。step 12 で独立側の同じ層の勾配 norm は 44.7、公式は 20.5 である。step 16 の損失は公式 28.939、独立 29.114、1 ulp 対照 28.939。step 20 の既存 4 プロセスでは損失 24.351 と 24.156（差 0.195、約 0.8%）で、重み cosine は 0.999995 のまま、勾配 cosine は step 16 に 0.24 まで落ちている。同一実装の再実行は 20 step ずっと差 0 なので、実行時の非決定性ではない。原因は、bit 一致する forward のあとに残る 1 ulp 前後の backward 差が AdamW で増幅し、急な勾配に片側だけ当たることである。学習式は変えていない。`libdf` の特徴抽出は引き続き未実行で、両側とも独立 `DFStream` である。

診断は `tests/training/reference/diagnose_df_step1.py` と `diagnose_df_horizon.py`。入力は既存の `four-process-deepfilternet3-20261102-cpu-20/input.pt` で、その結果ディレクトリは上書きしていない。

### SEMamba++ と MossFormerGAN

SEMamba++ の公式 CPU 比較は Triton RMSNorm が CPU tensor を拒否するため N/A のままである。CUDA の縮小幅（hid_feature 8、1 block）の 1 step、4 step、再開、VoiceBank は前節の数値のままである。公開幅の fused 学習 step は tensor で動き、コスト節に書いた。公開幅の公式との 20 step 数値比較は今回もしていない。再開後に fused の重みハッシュが連続実行と一致しない点も前節のままで、チェックポイントの読み込み自体は成功している。

MossFormerGAN の公開設定は `accu_grad: 1`、`effec_batch_size: 2`、`batch_size: 1` である。学習タスクはその蓄積（損失を 2 で割り、両 backward のあと両 optimizer を 1 回）を使う。4 プロセス probe は蓄積 1 のままである。CPU の 4 step と再開は一致し、GPU 1 step の公式/独立勾配差は再実行差より大きい、という前節の判定は変えていない。20 step のスナップショットは容量のため今回も取っていない。

### 学習コスト

数値と shannon の条件付き推定は [GPU_COST.md](GPU_COST.md)。tensor の物理 GPU 0 だけが 1024 MiB 未満だった。2 GPU と 4 GPU は測っていない。shannon では実行していない。

### 回帰テスト

2026-11-02、turing、`.venvs/lab`、`OMP_NUM_THREADS=1`。

| スイート | 結果 |
|---|---|
| `tests/data` | 12 件、失敗 0 |
| `tests/se` | 28 件、失敗 0、skip 1（CUDA 拡張） |
| `tests/evaluation` | 86 件、失敗 0 |
| `tests/asr` | 5 件、失敗 0 |
| `tests/training` | 87 件、失敗 0、skip 0。710 秒 |
| うち `tests.training.test_four_process` | 8 件、失敗 0。543 秒 |

過去の training 件数（30、47、61、68、74、76、79）は、その時点のテストメソッド数である。68 と 86 の差は失敗や skip ではなく、その後に 4 プロセス比較やモデル別試験が足された結果、現在は 87 件になっている。evaluation の 86 件は training の件数ではない。skip は失敗ではない。公式ツリーや CUDA 拡張が無い環境の skip を、公式一致の PASS には数えない。

### TSUBAME H100 追加診断（2026-11-02）

`data/results/training/final-h100-20261102-cpu/` に今回の入力、各プロセスの結果とジョブログを保存した。以下は短い合成入力を使った 1 更新の比較であり、実データの学習品質の比較ではない。MossFormerGAN は同じ0.5秒波形を2回使って蓄積し、各 backward 後にクリップした。

- MossFormerGAN: 公開 solver は Hamming 窓を CPU 上で作り GPU に移す。独立側が GPU 上で生成すると、H100 で公式/独立の forward 最大差 `1.29e-5`、勾配相対 L2 `0.00263` が生じた。CPU 生成に合わせた後、forward 最大差 0、勾配相対 L2 `3.95e-7`。公式 A/B は `3.91e-7`、独立 A/B は `4.03e-7`。更新後重みの最大差は公式/独立 `5.69e-4`、公式 A/B `4.77e-4`、独立 A/B `6.51e-4`。CPU の同一合成波形の蓄積2・1更新では forward、損失、勾配、重みが一致した。
- MP-SENet: dense_channel 64、TS-Conformer 4、epoch 単位 scheduler、公式 PESQ 目標。H100 の公式/独立勾配相対 L2 `1.11e-7`、最大 `2.09e-7`。重み最大差 `6.61e-4` に対して公式 A/B `5.75e-4`、独立 A/B `7.23e-4`。probe の入力は公開 segment 長ではない。
- TF-GridNet ms-snsd: H100 で forward、loss、勾配、更新後重みの最大差は全ペア 0。公式 separator と損失を比較し、separator の STFT は既存 probe のスタブである。
- DeepFilterNet3: H100 の forward と loss は一致、公式/独立勾配相対 L2 `1.37e-6`、最大 `1.52e-5`。公式 A/B は相対 L2 `1.62e-7`。公式 libdf 0.5.6 と独立 DFStream を同じ合成入力で比較したところ、ERB 幅は一致し、スペクトル・ERB・DF 特徴の相対 L2 差は順に `1.65e-7`、`1.32e-7`、`2.91e-7`。CPU の20 step対照は公式/独立 loss 差 `0.283`、公式/1 ULP 摂動公式差 `0.133`。1つの摂動対照だけで前者を説明したとは判断しない。
- SEMamba++: CUDA 12.3.2 で `sm_90` を含む `mamba-ssm==2.3.2.post1` と `causal-conv1d==1.7.0` を新規ビルドし、PyTorch 2.6.0+cu124、H100 MIG 3g.47gb で実行した。パッケージの Mamba3 自動 import は Triton 3.2 に存在しない `set_allocator` を要求するため、専用環境でその import のみ外した。Mamba1 の fused CUDA や RMSNorm の実装は変更していない。単体 Mamba は `use_fast_path=True` で forward/backward が動作し、再実行の forward 最大差 0、入力勾配最大差 `2.27e-13`。公式幅（48、4 block）の4プロセス1更新は公式/独立 forward 差 0、勾配最大差 `3.44e-6`、相対 L2 `4.55e-6`、cosine 約1。公式 A/B の勾配相対 L2 は `9.97e-7`。GAN の probe は短い入力と1件バッチであり、公開バッチと epoch 単位 scheduler の完全再現ではない。

`--official-width` で MP-SENet と SEMamba++ の probe 幅を選べる。DeepFilterNet3 の参照設定は `DF_REFERENCE_CONFIG` でも指定でき、存在しない設定を暗黙に既定値で代用しない。CPU テストでは VoiceBank DB がこの環境に無いため該当テスト1件が失敗した。CUDA の MossFormerGAN 窓の回帰テストは H100 で成功した。

追加の再検証: MossFormerGAN は上記と同じ合成波形の蓄積2で CPU 4 step を実行し、公式 A/B、独立 A/B、公式/独立の forward・勾配・更新後重みが全 step で一致した。2 step 後に再開した4プロセスは、それぞれ連続実行と loss、モデル、optimizer、scheduler、RNG が一致した。これは異なる micro-batch による実データ検証ではない。

DeepFilterNet3 の spectral loss は公式に合わせ、複素数の再構成を複素指数で行い、Hann 窓を CPU で生成してから実行デバイスに移すよう変更した。公式 spectral loss と異なる波形に対する損失・勾配の CPU 回帰テストは成功した。同じ20 step診断を再実行すると step 20 の公式/独立 loss 差は `0.212`、公式/1 ULP 摂動公式は `0.133` であり、差の意味は未確定である。

SEMamba++ 公式幅 fused の4 step 連続比較では、step 4 の勾配 cosine は公式 A/B `0.934`、独立 A/B `0.997`、公式 A/独立 A `0.990`、公式 B/独立 B `0.930`。公式 A/独立 A の forward 最大差は `0.0205`、loss 差は `0.0126`、勾配相対 L2 は `0.141`。同一実装にも大きな再実行差があるが、その事実だけを根拠に公式/独立の差を許容とは判定しない。4 step の scheduler は比較用に更新ごとに進めており、公開設定の epoch 単位ではない。

H100 再確認では、DeepFilterNet3 の損失修正後の1 step は公式/独立 forward と loss が一致し、勾配相対 L2 `1.81e-7`、公式 A/B は `1.39e-7`、独立 A/B は `1.61e-7`。同じ出力波形を両 spectral loss に渡した CPU 診断では loss 差 0、波形勾配最大差 `3.05e-5`、相対 L2 `5.32e-8` だった。20 step 差は引き続き未解決。

MossFormerGAN の H100 4 step（同じ合成波形を各更新で蓄積2）は、step 4 の公式 A/独立 A の勾配相対 L2 `0.00921`、cosine `0.999958`、最大差 `6.72e-4`。公式 A/B は `0.00857`、cosine `0.999963`、最大差 `9.09e-4`。最大差は事前許容値を超えたため、再実行差が近いことは原因分析の一部として記録し、実データでの同等性を保証しない。

SEMamba++ の H100 checkpoint 再開は、4プロセスすべてで scheduler と RNG が連続実行と一致した一方、step 4 の loss 差は `0.00029–0.00125`、重み最大差は `0.00074–0.00095` だった。checkpoint のロード成功と、ビット単位の再現は区別する。

その後、上記 DeepFilterNet3 の CPU 20 step 差をさらに切り分けた。既存の決定的 STFT は反射パディングをスライスと連結で表し、公式 `torch.stft(center=True)` と forward は同じだが CPU の backward 加算順が異なった。複素指数と Hann 窓を公式に合わせたうえで、通常経路を公式の STFT に変更した結果、同じ合成入力・初期重みの公式と独立は CPU 20 step の損失と全パラメータ勾配が一致した（`df-final-check.out`）。1 ULP 摂動対照は step 20 で loss が約 `0.133` 異なる。CUDA の厳密な決定的モードでは `reflection_pad1d` backward が使えないため、従来の明示パディングを維持する。この代替経路は backward の加算順が公式と異なり得るため、CPU の一致をそのまま決定的 CUDA 学習の一致とみなさない。

修正後の DeepFilterNet3 は CPU の独立4プロセス20 step（同じ合成波形、共通 DFStream）でも、公式 A/B・独立 A/B・公式/独立の loss、全勾配、更新後重みが全 step で一致した。H100 の非決定的許可経路での1 step は公式/独立 forward と loss の差 0、勾配相対 L2 `1.64e-7`、公式 A/B `2.84e-7`。厳密な決定的 CUDA の代替経路では有限勾配を確認したが、公式との4プロセス20 step 数値一致は確認していない。

SEMamba++ の H100 公式幅20 step は4独立プロセスで同一の初期状態・入力・RNG を使い、途中は loss と勾配 norm、最終 step だけ波形・全勾配・重みを保存した（`semamba20-compact/` は228 MB）。scheduler は比較用に毎 step 更新、入力は8,000サンプルの合成波形である。step 20 の勾配 cosine は公式 A/B `0.467`、独立 A/B `0.613`、公式 A/独立 A `0.034`、公式 B/独立 B `0.813`。loss 差は順に `0.121`、`0.0908`、`0.00256`、`0.0275`。公式自体の非決定性と更新差の増幅は確認できるが、公式/独立の勾配差をそれだけで説明できないため、長期学習挙動の同等性は未確定である。

最終の H100 1 step（`qrsh-h100-df-final/`、厳密な決定的アルゴリズムを強制しない probe）は公式/独立の forward と loss 差 0、勾配相対 L2 `1.64e-7`、公式 A/B `2.84e-7`、独立 A/B `3.00e-7`。決定的 CUDA の明示パディング経路は単体テストで有限な backward を確認した。

SEMamba++ の公式/独立 1 step における最大重み差 `3.78e-4` の箇所は `generator.dense_encoder.dense_block.dense_block.0.0.weight` で、対応勾配は公式 `-3.46e-7`、独立 `1.12e-7` と微小かつ符号が逆だった。AdamW の初回更新は epsilon に対して十分大きい勾配の符号を強く反映し、この局所的な重み差を説明する。ただし、この1箇所の分析は20 step の全勾配差の原因を証明しない。

20 step の勾配を分けると、公式 A/独立 A の cosine は generator `0.0242`、CQT discriminator `-0.307`、multi-resolution discriminator `0.99960`。公式 A/B でもそれぞれ `0.462`、`0.164`、`0.99969` となり、差は主に generator と CQT 側に集中した。

蓄積1で作成された旧 MossFormerGAN probe の checkpoint は、蓄積2のコードでそのまま再開すると別の更新に切り替わるため拒否する。旧結果の読み取りと report は引き続き可能で、新しい probe は別ディレクトリに作る。

GPU 資源の会計上の確保時間（各 H100 MIG 3g.47gb は同時利用なし）は、ジョブ本体の `ru_wallclock` ではなく開始処理も含む `wallclock` を使う。9232560: 74.605秒、9232572: 196.918秒、9232648: 18.098秒、9232699: 46.118秒、9232825: 54.016秒、9232848: 139.252秒、9232870: 241.844秒、9232910: 188.155秒、9232940: 97.705秒、9232964: 112.439秒、9233120: 111.968秒。ここまで合計 `1281.118秒`（21分21.118秒）。9232560 は qsub から qrsh への切り替えで途中停止し、9232825 は Mamba3 の import エラーで失敗した。CPU のビルド・診断は含まない。

再現例（出力先は存在しない新しいディレクトリを指定する）。CUDA の2コマンドは GPU を、CPU のコマンドは CPU を `qrsh` で確保して実行する。拡張の専用環境と参照ソース、上記の Mamba3 import 制約が必要である。

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="$PWD/.venvs/tsubame-h100-causal:$PWD/.venvs/tsubame-h100-build:$PWD/.venvs/tsubame-ref-extra:$PWD"
export DF_REFERENCE_CONFIG="$PWD/data/results/training/final-h100-20261102-cpu/df-config.ini"
.venvs/tsubame/bin/python tests/training/four_process/compare.py run data/results/training/NEW-SEMAMBA --model semambapp --steps 1 --device cuda:0 --scan-backend fused --official-width
.venvs/tsubame/bin/python tests/training/four_process/compare.py run data/results/training/NEW-MOSS --model mossformergan --steps 4 --device cuda:0
.venvs/tsubame/bin/python tests/training/four_process/compare.py run data/results/training/NEW-DF-CPU --model deepfilternet3 --steps 20 --device cpu
```

H100 で discriminator と optimizer を除き、公式・独立 generator の同じ入力に共通の単純なスペクトル損失を与えた同一プロセス内対照では、forward と loss の差は 0、勾配相対 L2 は公式 A/B `3.34e-7`、独立 A/B `3.35e-7`、公式/独立 `3.34e-7` だった。これは generator 単体の対照であり、GAN 更新の同等性を証明するものではない。

固定波形の H100 discriminator 単体対照では、公式 A/B と独立 A/B の勾配差は 0、公式/独立は loss 差 `4.77e-7`、勾配相対 L2 `1.16e-7` だった。この調査で、公式は CQT と multi-resolution の discriminator、adversarial、feature matching の損失をそれぞれ別に合算してから足すのに対し、独立側は連結したリストを一度に合算していたことを確認した。独立学習と4プロセス probe を公式と同じグループ・順序へ変更した。変更前に作った SEMamba++ probe は異なる計算を混ぜないよう再開を拒否する。

グループ別の損失合算順を修正した SEMamba++ の H100 公式幅1 step（`qrsh-h100-semamba-grouped1/`）では、公式/独立の discriminator loss 差は `4.77e-7` から 0 になった。forward と generator loss の差も 0、勾配相対 L2 `4.55e-6`、最大差 `3.60e-6`。公式 A/B の勾配相対 L2 は `9.98e-7` だった。上の20 step はこの合算順修正前の記録である。

追加の H100 MIG ジョブの確保時間（会計 `wallclock`）は、discriminator 対照 `9233186` が103.933秒、合算順修正後の1 step `9233218` が81.316秒、修正後の4 step・再開・20 step `9233238` が312.953秒。前述分と合わせた GPU 確保時間は `1779.320秒`（29分39.320秒）で、最大同時使用は1 MIG、累計上限2時間以内だった。

合算順修正後に公式幅 fused を再検証した。4 step の公式 A/B、独立 A/B、公式 A/独立 A の勾配相対 L2 は `0.0379`、`0.0875`、`0.218`、cosine は `0.99928`、`0.99617`、`0.97593`。step 4 の公式/独立 forward 最大差は `0.01899`、loss 差は `0.01395`。20 step は同じ初期状態・合成入力で4独立プロセスを実行し、途中は統計だけ、最終状態だけを保存した（約230 MB）。勾配 cosine は公式 A/B `0.669`、独立 A/B `0.242`、公式 A/独立 A `0.719`、公式 B/独立 B `0.588`。loss 差はそれぞれ `0.0213`、`0.1328`、`0.1157`、`0.00423`。4 step で公式/独立の差が自己再実行差を上回り、20 step では両側とも大きくばらつくため、融合版の学習同等性は引き続き PARTIAL。

さらに公式 solver は各サブ discriminator の real/fake を交互に呼ぶが、独立側は各系列の全 real の後に全 fake を呼んでいた。独立 discriminator にペア入力を追加し、学習と probe で公式順にした。既存の単一入力は従来どおり使える。呼び出し順修正後の H100 公式幅1 step は forward・loss 差 0、公式/独立の勾配相対 L2 `4.56e-6`、公式 A/B `1.00e-6`、独立 A/B `1.00e-6` で、1 step の generator 勾配差はほぼ変わらなかった。ペア順に対応しない古い probe checkpoint は再開を拒否する。

ペア呼び出し順修正後の H100 1 step（`9233293`）の会計 `wallclock` は `163.542秒`。それまでのジョブとの累計は `1942.862秒`（32分22.862秒）。

ペア順修正後の4 step・再開・20 step（`9233314`）の会計 `wallclock` は `386.305秒`。全 GPU ジョブの最終合計は `2329.167秒`（38分49.167秒）である。

最終の paired 順による H100 の4 step は、公式 A/B・独立 A/B・公式 A/独立 A の勾配相対 L2 が `0.116`・`0.150`・`0.233`。20 step では同順の cosine が `0.101`・`0.746`・`0.639`、公式 B/独立 B は `0.338`。loss 差は順に `0.0551`・`0.1861`・`0.1255`・`0.2564`。入力と初期条件は4プロセスで一致し、途中の統計と最終状態のみ保存した。公式自身の再実行差も大きく、公式/独立の差の意味は未確定である。paired 順でも GO ではない。

最終 paired 順の2 step 後再開は4プロセスとも成功し、scheduler と RNG はそれぞれの連続実行と一致した。loss・モデル・optimizer のハッシュは4プロセスとも一致しなかった。

追加監査: この連続実行と再開実行は、別々に開始した二つの実行である。保存済み `official_a` の各 step を照合すると、入力と RNG ハッシュは一致するが、モデルと optimizer のハッシュは再開前の step 1 から既に異なる。step 1 の forward 差は 0、最大勾配差は `1.07e-6`、更新後の最大重み差は `2.98e-4`。step 1〜3 の loss は記録上同じ、step 4 の loss 差は `0.001194000244140625`。したがってこの比較は再開操作自体が差を導入したかどうかを識別できない。同一の step 2 状態から連続側と別プロセス側を分岐させ、再開直後の状態と最初の forward/backward/update を比較する必要がある。

同日 CPU の固定乱数波形を用いた `tests/training/reference/compare_semamba_disc.py` の再実行では、公式と独立の CQT / multi-resolution discriminator の state-dict キーは一致し、train / eval とも score・各 feature map の最大絶対差は 0 だった。これは固定入力の forward 比較であり、GAN backward や CUDA の複数更新の一致を示さない。

H100 MIG 3g.47gb のジョブ `9236723` で固定入力の CUDA 演算を5回反復した（`data/results/training/followup-20261102-voicebank/semamba-ops.py` と `semamba-ops.json`）。今回の形状では causal_conv1d と Triton RMSNorm は forward と全入力勾配が一致。selective_scan は forward と u、delta、A、D の勾配が一致したが、B と C の勾配にそれぞれ最大 `2.98e-7`、`1.49e-7` の再実行差があった。演算単体の非決定性の証拠だが、GAN の複数更新差や公式/独立の差全体の原因を証明しない。

ジョブ `9236748` では4プロセスの連続実行が保存した同一の step 2 checkpoint から、各実装を別プロセスで分岐させた。合成入力、公式幅 fused、scheduler は毎更新。分岐時のモデル・optimizer・RNG・入力は一致し、再開後 step 3 の forward と loss も4系統全てで一致したが、勾配とその後の更新は一致しない。step 3 勾配最大差は `1.61e-6`〜`1.95e-6`、更新後重み最大差は `1.33e-4`〜`1.78e-4`。step 4 forward 最大差は `9.83e-7`〜`3.99e-6`。RNG と入力は一致した。最初に観測された差は同一状態での backward であり、上記 selective_scan の非決定性と整合するが、その演算がこの GAN 全体の差の唯一の原因とは特定していない。結果は `semamba-cont/run/` と `semamba-fork/run/`。
この連続側の step 4 勾配相対 L2 は公式 A/B `0.288`、独立 A/B `0.0257`、公式 A/独立 A `0.137`、公式 B/独立 B `0.289`。公式/独立の差を単に自己再実行差と同等と認定しない。
追加 seed 17 / 29 の公式幅 fused・4更新（`qrsh` ジョブ `9236800`）では、それぞれ勾配相対 L2 が公式 A/B `0.0516` / `0.0387`、独立 A/B `0.0566` / `0.0245`、公式 A/独立 A `0.0445` / `0.0381`、公式 B/独立 B `0.0437` / `0.0302`。各 seed の初期状態と入力は同じで、seed 間では異なる。自己再実行差と近い例が増えたが、GAN の差の原因と絶対勾配差は未確定である。
固定の合成 real/fake 波形・初期重みを使った CQT discriminator 単体の H100 `qrsh` ジョブ `9236852` では、公式 A/B・独立 A/B・公式/独立のスコア、畳み込み後の全 feature map、discriminator loss、入力勾配、パラメータ勾配、AdamW 更新後の重みの最大差は全て 0。同一プロセス内の対照であり、CQT の中間スペクトルそのものや GAN 全体を比較したものではない。スクリプトと結果は `cqt-control.py` / `cqt-control.json`。
SEMamba の consistency STFT について固定波形と上流勾配を使った `qrsh` ジョブ `9236861` では、公式の native 反射と独立の明示反射の forward は一致、波形勾配の公式/独立相対 L2 差は `3.31e-8`。各経路内の再実行差も約 `3e-8` だった。これは固定入力での局所対照であり、学習全体の同等性を示さない（`semamba-stft.py` / `semamba-stft.json`）。

固定初期状態・合成入力の generator 出力に対する損失成分別勾配（`qrsh` ジョブ `9236869`）では、公式/独立の mel 勾配相対 L2 差 `4.27e-6` が各自己再実行差約 `8e-8` を上回った。独立側の GPU 生成 Hann 窓、独自 Slaney basis、行列積と対数の演算順が公式と異なっていた。学習損失を CPU 生成 Hann、librosa Slaney basis、公式と同じ行列積と対数・加算順に合わせた。CPU の公式損失との回帰テストでは損失と波形勾配が一致。修正後の同条件 `qrsh` ジョブ `9236900` では mel 勾配相対 L2 は公式/独立 `8.38e-8`、公式 A/B `8.17e-8`、独立 A/B `8.56e-8`、mel loss 差 0 となった。Generator 合計損失の出力勾配相対 L2 は `8.34e-7` から `1.31e-7` へ減少した。独立実装は公式コードを import せず、学習時に librosa を用いる。

同じ seed・入力・公式幅 fused・毎更新 scheduler の修正後4プロセス比較（`qrsh` ジョブ `9236902`、`semamba-fixed-cont/run/`）では step 1 の公式/独立パラメータ勾配相対 L2 は修正前 `4.55e-6`、修正後 `4.54e-6` で、mel 項の局所修正は全モデル勾配の差を解消しなかった。step 4 は公式 A/B `0.515`、独立 A/B `0.0627`、公式 A/独立 A `0.491`、公式 B/独立 B `0.300`。CUDA のばらつきが大きく、固定 seed の1回の前後比較だけで差の改善・悪化を判断しない。同じ step 2 checkpoint からの分岐では step 3 の forward・loss・RNG は一致し、モデルと optimizer は再び分岐した。未解明の実装間差が残るので PARTIAL。

さらに multi-resolution discriminator の固定 real/fake 入力・初期重みで generator adversarial 項を切り分けたところ、公式/独立の入力勾配相対 L2 差は `5.77e-7`、自己再実行差は約 `6e-8` だった（`qrsh` ジョブ `9236920`）。独立側の CUDA 上での Hann 窓生成と明示反射を、通常経路では公式の CPU 生成窓と native 反射に合わせ、厳密な決定的 CUDA では明示反射を維持した。修正後の同条件 `qrsh` ジョブ `9236927` は、公式/独立の入力勾配相対 L2 差 `5.30e-8`、公式 A/B `5.36e-8`、独立 A/B `5.05e-8`。スコア、畳み込み feature map、パラメータ勾配の公式/独立差は 0。CPU の公式 spectrogram と入力勾配の回帰テストも成功した。これは固定入力の discriminator 単体対照であり、GAN 全体の一致とは区別する。

両修正後の公式幅 fused・合成入力・毎更新 scheduler の4プロセス再比較（`qrsh` ジョブ `9236930`）では、step 1 の公式/独立パラメータ勾配相対 L2 が修正前 `4.55e-6` から `1.01e-6` へ減り、公式 A/B `9.9e-7`、独立 A/B `1.02e-6` と同程度になった。step 4 は公式 A/B `0.196`、独立 A/B `0.337`、公式 A/独立 A `0.167`、公式 B/独立 B `0.369`。結果は `semamba-final-cont/run/` と `semamba-final-fork/run/`。複数更新のばらつきとその原因は引き続き検討する。
両修正後の損失成分固定入力対照（`qrsh` ジョブ `9237070`）では generator 出力への合計勾配相対 L2 が公式 A/B `9.11e-8`、独立 A/B `8.75e-8`、公式/独立 `1.19e-7`。adversarial は順に `1.22e-7`、`1.09e-7`、`1.31e-7`、mel は約 `8.4e-8`。この局所対照だけでは複数更新全体の一致を証明しない。
同一 step 2 checkpoint からの4プロセス分岐では step 3 の forward・loss・RNG が一致する一方でモデルが分岐し、step 4 forward 最大差は `1.01e-6`〜`1.51e-6`。修正後の追加 seed 17 / 29（scheduler は毎更新）の step 4 勾配相対 L2 は公式 A/B `0.0535` / `0.0299`、独立 A/B `0.0398` / `0.0380`、公式 A/独立 A `0.0533` / `0.0144`、公式 B/独立 B `0.0277` / `0.0383`。同じジョブ `9236951` の VoiceBank 実音声・4更新では step 1 が公式 A/B `1.25e-6`、独立 A/B `1.28e-6`、公式 A/独立 A `1.26e-6`、step 4 は順に `0.0640`、`0.0599`、`0.0599`（公式 B/独立 B `0.0613`）。全入力 case・crop は各比較で一致した。3 seed の短い比較と局所的な CUDA 非決定性の観測は、残差の唯一の原因や長時間学習の同等性を証明しない。

上記 SEMamba probe で `--scheduler-step epoch` が無視され、実際には scheduler が毎更新進んでいたことを発見した。probe を修正し、古い設定の checkpoint は新しい損失・scheduler での再開を拒否する。`qrsh` ジョブ `9237160` の再検証では4更新の学習率が全て `2e-4` と確認できた。公式幅・合成入力で step 4 勾配相対 L2 は公式 A/B `0.0501`、独立 A/B `0.0549`、公式 A/独立 A `0.0461`、公式 B/独立 B `0.0537`。これは epoch 終了前の4更新であり、epoch 終了時の scheduler 更新は別途 CPU 回帰テストの範囲である。
同じ step 2 checkpoint からの分岐では4系統とも step 3 の forward・loss・RNG が一致し、モデルは更新後に分岐した。step 4 forward 最大差は `1.16e-6`〜`1.55e-6`。これは同一状態での backward 以降に差が生じることを示すが、個別の演算への一意な帰属ではない。
同ジョブの20更新は同一の合成入力・初期状態、公式幅、epoch 終了前の固定学習率で4プロセスを実行し、途中は loss と勾配 norm、最終だけ全勾配と重みを保存した。step 20 の勾配 cosine は公式 A/B `0.479`、独立 A/B `0.676`、公式 A/独立 A `0.773`、公式 B/独立 B `0.592`、相対 L2 は順に `1.205`、`0.799`、`0.676`、`0.824`。同一実装にも大きな分岐があるが、実装間差の原因を完全に説明したとはしない。結果は `semamba20-epoch/`。
追加 seed 17 / 29 の同条件20更新（`qrsh` ジョブ `9237460`）でも初期状態と入力は4プロセスで一致した。step 20 勾配 cosine は順に、公式 A/B `0.605` / `0.662`、独立 A/B `0.511` / `0.843`、公式 A/独立 A `0.380` / `0.634`、公式 B/独立 B `0.422` / `0.849`。seed 17 では実装間差が自己再実行差を上回るため、ばらつきだけで実装間差を説明できると認定しない。結果は `semamba20-epoch-seed-17/` と `semamba20-epoch-seed-29/`。
同ジョブの4更新の step 4 勾配相対 L2 は seed 17 / 29 で公式 A/B `0.0567` / `0.0614`、独立 A/B `0.0480` / `0.0640`、公式 A/独立 A `0.0553` / `0.0707`、公式 B/独立 B `0.0471` / `0.0413`。結果は `semamba-epoch-seed-17/` と `semamba-epoch-seed-29/`。
seed 17 の step 20 を層群別に分けると、公式 A/独立 A の勾配 cosine は Generator `0.378`、CQT discriminator `0.133`、multi-resolution discriminator `0.999`。公式 A/B は順に `0.602`、`0.852`、`0.998`。固定入力・固定重みの CQT 対照は一致しているため、この集計だけで CQT 実装の不具合と断定できない。

DeepFilterNet3 の CPU 20 step を新しい出力先 `data/results/training/df-cpu-20-20261102-followup/` で再実行した。合成入力、共通 DFStream、公式損失と独立損失を使う4プロセスの全比較で loss、forward、勾配、更新後の重みの最大差は全 step で 0、report は valid だった。決定的 CUDA 経路の検証ではない。

TF-GridNet は復旧した VoiceBank の1区間（16 kHz、2048 samples）を使う CPU 4プロセス1更新で、公式同士・独立同士・公式/独立の forward、loss、勾配、更新後重みの最大差が 0 だった。結果は `data/results/training/followup-20261102-voicebank/tf-cpu1/`。比較する公式 separator は STFT スタブを使用しており、ESPnet の前処理全体の一致を証明するものではない。

TSUBAME の VoiceBank 学習 DB を公式 28 話者データから構築した。永続 staging は `data/staging/voicebank-current/`、DB は `data/db/voicebank.duckdb`。train 10,700 件・26 話者、valid 872 件・p286/p287 で、test 話者 p232/p257 および train/valid 話者の重複はない。全 11,572 件で transcript があり、元の 48 kHz と clean/noisy の同名・同形状を保持し、case ID はレシピの SHA-256 と一致した。staging から別 DB を再構築し cases / sets / set_items の全行が一致した。RecipeDataset の1件で 48→16 kHz、同長の clean/noisy と seed 1234・epoch 0 の crop 再現を確認した。全発話の音質検査ではない。

元データへの `data/corpora/VoiceBank-DEMAND` リンクを事前に用意し、出力が存在しないときの構築例は次のとおり。`build` は staging または DB が存在すると拒否する。既存の再構築元を消して再実行しない。

```bash
.venvs/tsubame/bin/python -c 'from pathlib import Path; from src.data.voicebank_sets import build; print(build(Path("data/corpora"), Path("data/staging/voicebank-current"), Path("data/db/voicebank.duckdb")))'
```

4プロセス probe の実音声再開では `--prepare-steps 4` のように将来の入力を先に用意できる。例えば `--steps 2 --prepare-steps 4` は、蓄積2の MossFormerGAN なら異なる8件を準備し、2更新で止めてから4更新まで再開できる。
また `--fork-step 2` は連続実行中の2更新後 checkpoint を別名で保持する。別プロセスをその同一状態から再開するための診断用であり、mock ではモデル・optimizer・scheduler・RNG が連続側と一致した。

MossFormerGAN の実音声 0.5 秒・蓄積2 の4更新ジョブ `9236673` は4プロセスとも3更新後、4更新目の PESQ 目標を作れず停止した。入力の最後の2区間は clean/noisy 間でも PESQ が `NoUtterancesError` を返した。欠損を 0 に置き換えず、未完了の結果を `data/results/training/followup-20261102-voicebank/moss-cont4/` に残した。同じ8発話を1秒で切り出すと clean/noisy PESQ は全件得られたため、その別条件で再試行する。

1秒・異なる8 micro-batch の H100 MIG ジョブ `9236698` は連続4更新と2更新後の再開を完了した。step 4 の勾配相対 L2 は公式 A/B `0.004066`、独立 A/B `0.001948`、公式 A/独立 A `0.002806`、公式 B/独立 B `0.003870`。公式 A/独立 A の勾配最大差は `1.67e-4`、重み最大差は `0.00191`。再開側と連続側は全4プロセスで入力と RNG ハッシュが一致したが、モデル・optimizer は step 1 から別の値であるため、再開操作単独の差はこの実験では識別できない。step 4 loss 差は `1.79e-6`〜`5.42e-6`。結果は `moss-1s-cont4/` と `moss-1s-split4/`。絶対勾配差の発生演算は未特定であり、自己再実行差のみを根拠に GO にしない。
同一 step 2 checkpoint から分岐した `qrsh` ジョブ `9237107` では4系統とも step 3 の forward・loss・RNG が一致し、モデルと optimizer はその更新後に分岐した。step 4 forward 最大差は `9.54e-7`〜`1.67e-6`、loss 差は最大 `5.96e-8`。入力は VoiceBank 1秒・蓄積2である。結果は `moss-shared-cont/run/` と `moss-shared-fork/run/`。
別の実音声連続実行 `moss-1s-cont4/` の step 1 で公式 A/独立 A の最大重み差 `0.000315` は `generator.dense_encoder.conv_1.0.bias` の1要素で、対応勾配は `4.31e-9` / `-4.89e-9`。AdamW の epsilon `1e-8` に近い微小勾配の符号差による局所的な更新差だが、step 4 の全勾配差の原因をこれだけで説明しない。

ジョブ `9236728` では DeepFilterNet3 の反射パディング単体を同一の固定入力・上流勾配で4回比較し、native CUDA と明示パディングの forward・入力勾配は全て一致した。この形状の単体結果であり、STFT と損失を通した勾配や決定的 CUDA の複数更新は未検証である。同じジョブの MP-SENet 公式幅・合成入力・epoch 単位 scheduler の4更新では、step 4 の勾配相対 L2 が公式 A/B `0.251`、独立 A/B `0.331`、公式 A/独立 A `0.255`。2更新後の再開も完了したが、連続側とは再開前の step 1 からモデルが違うため、再開操作だけの影響は識別できない。入力と RNG は一致し、step 4 loss 差は `0.000207`〜`0.000588`。結果は `mp-cont4/` と `mp-split4/`。

追加のジョブ `9236783` で固定合成波形に対する DeepFilterNet3 spectral loss を native CUDA と決定的経路で4回ずつ比較した。loss は同じ `1428.7008056640625`、波形勾配の native/決定的の最大差 `2.38e-7`・相対 L2 `5.51e-8`。native の自己再実行最大差は `1.19e-7`、決定的経路の自己再実行差は 0。STFT を通した加算順と native の非決定性を含む比較であり、両者の差を一つの演算だけに帰属させない。

続く `qrsh` ジョブ `9236792` では VoiceBank 48 kHz・4,800 samples の4区間を使い、公式は native CUDA、独立は厳密な決定的 CUDA にして4更新を比較した。両経路は決定性設定が異なる。step 4 勾配相対 L2 は公式 A/B `0.000391`、独立 A/B `0`、公式 A/独立 A `0.00468`、公式 B/独立 B `0.00501`。step 2 の同一 checkpoint から分岐した再開では、独立 A/B とも step 4 までモデル・optimizer・RNG が連続側と完全一致した。公式の native 経路は step 3 からモデル・optimizer が異なり、RNG は一致した。公式/独立の複数更新差は公式の自己再実行差より大きく、原因を一つに特定できていないため PARTIAL を維持する。結果は `df-cont/run/` と `df-fork/run/`。

MP-SENet は `qrsh` ジョブ `9237090` で VoiceBank の異なる1秒区間・公式幅・epoch 単位 scheduler の4更新を比較し、PESQ 目標も全 step で得た。step 4 勾配相対 L2 は公式 A/B `0.00790`、独立 A/B `0.0813`、公式 A/独立 A `0.0409`、公式 B/独立 B `0.0704`。同一 step 2 checkpoint からの別プロセス分岐は step 3 の forward・loss・RNG が一致し、その更新後のモデル・optimizer が分岐した。step 4 forward 最大差は `8.34e-7`〜`1.10e-6`。この probe は公式・独立の Generator と Discriminator を比較するが、両側の GAN 損失と PESQ 目標は共通の独立損失関数を使うため、公式損失コードを別途実行した比較ではない。結果は `mp-voicebank-cont/run/` と `mp-voicebank-fork/run/`。

今回の H100 MIG ジョブの会計 `wallclock` 秒: `9236673` 122.952、`9236698` 295.456、`9236723` 87.022、`9236728` 156.948、`9236748` 170.446、`9236783` 9.225、`9236792` 65.757、`9236800` 281.372、`9236852` 104.707、`9236861` 15.011、`9236869` 52.227、`9236900` 109.938、`9236902` 159.994、`9236920` 81.969、`9236927` 79.765、`9236930` 257.938、`9236951` 347.195、`9237070` 23.806、`9237090` 68.038、`9237107` 138.842、`9237160` 337.109、`9237460` 409.981。今回分 `3375.698秒`、前回分 `2329.167秒` を含む累計 `5704.865秒`（95分4.865秒）。同時に使用したのは最大1 MIG。`9236673` は PESQ の発話不足で失敗、他は終了コード 0。最初の6ジョブは短時間開発にもかかわらず誤って `qsub` を使い、ユーザーの指摘後は `qrsh -p -5` に切り替えた。既存の24時間対話ジョブ `115141` はユーザー指定によりこの枠外として扱うが、そこで CPU テストが自動実行した CUDA 専用テスト1件の使用も記録する。

CPU 回帰は data 16 件、training 102 件（skip 3）、SE 28 件（skip 19）、evaluation 86 件（skip 3）、ASR 5 件で、失敗はない。training は scheduler probe 修正後に全件を再実行した。training の skip は未配置の rebuild DB 1 件と CUDA 専用 2 件。SE / evaluation の skip は配布 checkpoint・参照結果の欠落と明示 CUDA テストである。実際のモデルとの比較結果と単体テストを混同しない。

## seed 17 の Backward 分岐と CUDA 経路（2026-11-03）

総合は引き続き **PARTIAL**。最低目標の4項目は、実装差を許容値の変更で吸収せずに切り分けた。長時間学習の品質同等性は対象外のままである。GPU は既存の対話ジョブ `115141`（H100 MIG 3g.47gb）を使い、新しい `qrsh` は出していない。この対話予約そのものは前回と同じく枠外とし、実験プロセスの wall clock だけを加算する。

### SEMamba++

保存済みの seed 17、公式幅、fused、epoch 単位 scheduler、合成 8,000 サンプル（`semamba-epoch-seed-17`）で、optimizer を進めずに Generator 損失の backward を公式2回と独立1回行った。24回の `selective_scan_cuda.fwd` は、入力も出力も公式再実行・公式/独立のどちらも全要素が一致した。公式は `mamba_simple.py` から、独立は `_fused_mixer` から、同じ `mamba_inner_fn` に入る。

最初に一致しなくなるのは scan の出力勾配ではない。波形に接続する損失項の backward である。公式再実行の波形勾配の最大絶対差は、合計 `3.73e-9`、mel `7.45e-9`、feature matching `1.86e-9`、consistency `9.31e-10`、adversarial `1.46e-11`。公式/独立も最大絶対差は同じ値で、相対 L2 は合計 `8.02e-8` と公式再実行 `3.81e-8` の同じ桁である。magnitude / phase / complex は波形に接続しない。この波形勾配の差は、最初の scan backward の入力 `dout`（相対 L2 約 `7.5e-7`）として届く。

同じ seed 17 の活性化で `selective_scan_cuda.bwd` を4回繰り返すと、入力が同一でも dB と dC の相対 L2 は約 `1.6e-7`、最大絶対差は `1e-11` 未満だった。長さ16と batch 1 でも dB/dC の再実行差は残る。dtype は float32 で、カーネルはこの精度である。

849個の同名パラメータ勾配の相対 L2 の中央値は、公式再実行 `7.90e-7`、公式/独立 `7.90e-7`。最大絶対差はどちらも約 `3.05e-6`。公式再実行では `mask_decoder.mask_conv.1.bias` の勾配ノルムが `2.03e-6` で cosine が `-1` になり、1要素の符号が反転した。AdamW の epsilon `1e-8` より小さい勾配の符号差は、初回更新で O(学習率) の重み差になる。step 1 の公式/独立に、公式再実行を超える実装差は出ていない。

seed 17 の20 step cosine（公式 A/独立 A `0.380`、公式 A/B `0.605`）は、この非決定性の蓄積の1本である。同じ条件の step 4 は公式 A/B `0.0567`、公式 A/独立 A `0.0553` で、実装間が自己再実行を上回っていない。seed 29 の20 step も実装間 cosine `0.634` が公式 A/B `0.662` に並ぶ。20 step をこのセッションで追加してはいない。融合カーネルを決定的な参照 scan に替えると公式の学習経路から外れるため、コードは変えていない。

結果は `data/results/training/seed17-backward-20261103/`、`seed17-backward-20261103-named/`、`seed17-backward-20261103-terms/`。

### DeepFilterNet3

同じ VoiceBank 入力（seed 3、48 kHz、4,800 samples、4更新）で経路を分けた。結果は `data/results/training/df-cuda-paths-20261103/`。

| 経路 | step 4 の勾配相対 L2 |
|---|---|
| 公式 native A/B | 0.00244 |
| 独立 native A/B | 0.000725 |
| 公式 native / 独立 native | 0.00159 と 0.000848 |
| 公式 native A/B（独立だけ決定的にした実行） | 0.000613 |
| 独立決定的 A/B | 0 |
| 公式 native / 独立決定的 | 0.00470 と 0.00529 |

両方を native CUDA（`torch.stft` の反射パディング）にすると、公式/独立は公式自身の再実行差を超えない。独立だけを厳密な決定的 CUDA にすると、明示反射の backward 加算順が公式の `reflection_pad1d` backward と異なり、4 step の相対 L2 は約 `0.005` まで開く。独立決定的どうしは重みまで一致する。

公式モデルに `torch.use_deterministic_algorithms(True)` を掛けた1 step は実行できない。失敗は `reflection_pad1d_backward_out_cuda does not have a deterministic implementation`。CPU の公式・独立20 step 一致は前節のままである。

### MossFormerGAN

実音声1秒・蓄積2の既存4 step（`moss-1s-cont4`）では、step 4 の勾配相対 L2 が公式 A/B `0.00407`、独立 A/B `0.00195`、公式 A/独立 A `0.00281`、公式 B/独立 B `0.00387`。実装間差は公式の再実行差より小さい。絶対値が `1e-8` 未満の勾配要素は、公式再実行でも cosine `0.325` である。

step 1 の最大重み差 `0.000315` は `generator.dense_encoder.conv_1.0.bias` で、勾配は `4.31e-9` と `-4.89e-9`。どちらも AdamW epsilon `1e-8` 未満。同じ初期重みで `torch.optim.AdamW`（lr `5e-4`、betas `(0.9, 0.999)`、weight decay `0.01`）にこの2勾配を別々に与えると更新差は `0.0003148`。同じ勾配を2つの optimizer に与えると更新差は 0。したがって差は optimizer 実装ではなく、epsilon 未満の勾配符号差が初回更新で増幅されたものである。

### MP-SENet

公式幅・VoiceBank 1秒・epoch 単位 scheduler の既存4 step（`mp-voicebank-cont`）では、step 1 の損失差は全ペア 0、勾配相対 L2 は `7.7e-8` から `8.7e-8`、勾配最大差は `6.0e-8` 以下、更新後重みの最大差は `3.4e-4` から `3.7e-4` で、公式再実行と公式/独立が同じである。

2026-10-31 に記録した要素（勾配 `-8.90e-9` と `5.14e-9`、lr `5e-4`、betas `(0.8, 0.99)`、epsilon `1e-8`）を同じ AdamW で再計算すると、更新差は `4.0519e-4` で、当時の実測 `4.0527e-4` との差は `7.5e-8`。閉形式 `lr * grad / (|grad| + eps)` との誤差は 0。同じ勾配を2つの optimizer に与えると更新差は 0。`|grad| < eps` のとき更新はほぼ `lr * sign(grad)` になり、符号差が O(lr) の重み差になる。

公式 `train.py` の損失は関数として分離されていない。probe は公式・独立の両方に独立実装の GAN 損失と PESQ 目標を使う。step 1 の重み差が公式再実行にも同じ大きさで出るので、この増幅は損失関数の取り違えではない。step 4 の勾配相対 L2 は公式 A/B `0.00790` に対し独立 A/B `0.0813`、公式/独立 `0.0409` と `0.0704` で、独立側の再実行の方が大きい。step 1 の説明は完了しているが、4 step 以降の増幅量は公式再実行と同一とはまだ言えない。

計算は `tests/training/reference/diagnose_adamw_epsilon.py`、結果は `data/results/training/adamw-epsilon-20261103/adamw.json`。

### 回帰テスト（2026-11-03、`.venvs/tsubame`）

失敗は無い。ログは `data/results/training/regression-20261103/`。

| スイート | 結果 |
|---|---|
| `tests/data` | 16件、失敗 0、skip 0 |
| `tests/training` | 102件、失敗 0、skip 1（再構築 DB が無い） |
| `tests/se` | 28件、失敗 0、skip 19（配布 checkpoint・参照結果の欠落、明示 CUDA 拡張テスト） |
| `tests/evaluation` | 86件、失敗 0、skip 3（PESQ 境界の保存例と MP-SENet checkpoint がこの環境に無い） |
| `tests/asr` | 5件、失敗 0、skip 0 |

`tests/training` の DeepFilterNet3 CUDA 決定的経路テストは、この MIG 上で実行され成功した。前回 skip だった CUDA 専用テストのうち、この環境で実行できたものが含まれている。スイート時間の大半は CPU なので、下の実験 wall clock には足していない。evaluation のログに出る CHiME3 注釈の skip は、テストが意図的に出す未登録データの記録であり、unittest の skip 3件とは別である。

### このセッションの GPU 時間

実験プロセスの wall clock は `135.428 + 136.045 + 87.600 + 35.526 + 336.158 + 29.475 + 68.309 = 828.541` 秒。失敗した損失項の切り分け `87.600` 秒と、形状を誤った公式決定的 CUDA の試行 `29.475` 秒を含む。前回累計 `5704.865` 秒に加えると `6533.406` 秒（108分53.406秒）。上限300分に対する残りは約191分。同時に使った MIG は1つ。対話ジョブ `115141` の24時間予約自体は加算していない。

### MP-SENet 公式損失経路の追加比較（2026-11-03）

HEAD `298b283`、公式 MP-SENet `89932cfe90d1dacb8e170e4a331d762462c21792`。4プロセス probe の公式側を、公式 `train.py` の損失式と公式 `batch_pesq` を実行するよう変更した。公式コードは checkout から import するため joblib の子プロセスでは `models` を解決できず、初回 CPU 試行 `mp-official-loss-cpu-20261103/` は失敗して中断した。再試行は公式の PESQ 関数と集計を維持し、joblib の backend のみ threading とした。固定テンソルの generator 全損失成分、対応する勾配、PESQ 成功・失敗時 discriminator 損失の単体比較は完全一致した（`tests.training.test_mp_official_loss_path`、1件成功）。

VoiceBank 実音声1秒・epoch scheduler・CPU 1更新 `mp-official-loss-cpu-thread-20261103/` は4系統の条件照合が成立した。step 1 の損失差は全ペア 0、勾配相対 L2 は公式 A/B と独立 A/B が 0、公式/独立が `4.03e-8`。更新後重み最大差は公式/独立 `6.97e-4`。

既存の対話ジョブ `115141`、H100 MIG、公式幅・VoiceBank 1秒・4更新 `mp-official-loss-gpu-20261103/` も4系統の条件照合が成立した。step 1 の損失差は全ペア 0、勾配相対 L2 は `7.71e-8`〜`8.50e-8`。step 4 の勾配相対 L2 は公式 A/B `0.00774`、独立 A/B `0.0702`、公式/独立 `0.0277` と `0.0750`。discriminator 損失差はそれぞれ `3.18e-5`、`3.05e-5`、`1.54e-4`、`1.56e-4` で、実装間差は自己再実行差より大きい。原因は未特定。probe は generator を discriminator と generator の各更新時に再計算し、PESQ 失敗時は中止するため、公式ループ全体や例外動作の実音声検証ではない。GO とはしない。step 2 checkpoint は保存したが再開比較は未実施。

GPU 実験のプロセス経過時間は `53.03` 秒（CPU 側集計・待機を含む保守的な計上）。対話予約時間は除外。累計 `6586.436` 秒、300分上限まで `11413.564` 秒。新規結果は上記ディレクトリを参照。

### TF-GridNet 公式 STFT 経路の追加比較（2026-11-03）

ESPnet commit `0d136c5456d403cb2c0a4c13e76e8136e32e9b7a` の sparse checkout に含まれない encoder・decoder を `git archive` で取り出し、公式 `STFTEncoder`・`STFTDecoder`・`Stft` を用いた。`torch_complex` の旧型と `typeguard` の import shim は計算に使わず、公式の built-in complex 経路を使う。既存の公式 separator・損失比較に加え、音声前処理も公式コードから実行した。`tests/training/reference/probe_tf_gridnet_official_stft.py` で再実行できる。

VoiceBank train case `03dbca80e082f9b44aa2c7fa693f33d3f6d262c454c6744f3f7416bff963f8da`、crop 4041、16 kHz 1秒、ms-snsd 公式幅、Adam と gradient clipping 5 の1更新で、CPU と H100 MIG の双方において forward・loss・362パラメータ勾配・更新後重み・optimizer 状態の最大差は 0。報告は `data/results/training/tf-official-stft-cpu-20261103/result.json` と `gpu-result.json`。この試行は個々の前処理テンソル、複数更新、再開を別途比較しておらず、これだけで全受入条件を GO としない。GPU process wall clock は `6.40` 秒、既存ジョブ `115141`。上記累計に加え `6592.836` 秒、残り `11407.164` 秒。

同じ実音声 CPU 1 step で、両実装の前処理後・最初の convolution 入力を forward hook で取得した追加結果 `preprocessing-result.json` は最大差 0。これにより当該入力の正規化と STFT を含む前処理出力が一致した。

4個の異なる VoiceBank 区間による CPU 連続4更新 `four-step.json` は、全 step の前処理後 convolution 入力、forward、loss、勾配、更新後重みの最大差が 0。checkpoint 再開はこの公式 STFT 経路では未検証である。

公式 STFT を用いた CPU 4更新の step 2 モデル・optimizer・RNG をメモリ上に保存し、新しいモデルと optimizer にロードして step 3〜4 を再実行した `four-step-resume.json` では、公式・独立それぞれ連続実行との最終重みおよび optimizer 状態の最大差は 0。同一プロセス内の再開である。

### 回帰試験の追加結果（2026-11-03）

CUDA を隠した CPU 実行: data 16件成功、training 103件成功（skip 4）、SE 28件成功（skip 19）、evaluation 86件成功（skip 3）、ASR 5件成功。training 実行中に MP-SENet 比較用 adapter を更新したため、このスイートだけでは更新後の全挙動を保証しない。training の skip は再構築 DB と公式 DeepFilterNet 設定の未指定、CUDA 専用2件。SE の skip は配布 checkpoint・参照と CUDA 拡張の未使用。evaluation の skip は保存例・checkpoint の欠落である。

初回 evaluation は `test_same_id_with_different_content_is_rejected` が1件失敗。テストが同じ浮動小数 WAV を再保存すると PEAK chunk の時刻が秒境界で変わり、ソース SHA-256 と case ID が変化するため、ケース衝突より先にセットの不一致となった。1.2秒間隔の再保存で差は byte 60 のみだった。テストを、音声を再保存せず transcript だけ変更するよう修正した。修正後 evaluation は86件成功、skip 3。

### MP-SENet 公式更新順の追加比較（2026-11-03）

公式側は実際の `train.py` のように generator の出力グラフを保持し、discriminator 更新後にその出力を使う。独立側は現行 task の no-grad 再計算経路を用いる。PESQ 失敗時は公式・独立とも fake loss を省く。VoiceBank 1秒・公式幅・epoch scheduler の4プロセス CPU 1 step `mp-official-loop-cpu-20261103/` は条件照合 valid、損失差 0、勾配相対 L2 は公式/独立 `4.03e-8`、自己再実行 0。

同条件の H100 MIG 4 step `mp-official-loop-gpu-20261103/` は条件照合 valid。step 1 損失差 0、勾配相対 L2 は `7.70e-8`〜`8.45e-8`。step 4 勾配相対 L2 は公式 A/B `0.01146`、独立 A/B `0.07490`、公式/独立 `0.07070` / `0.02226`。discriminator loss 差は順に `3.76e-5`、`1.48e-4`、`5.66e-5` / `1.67e-4`。step 4 の fake loss は 公式 A/B `0.53579545` / `0.53583264`、独立 A/B `0.53585148` / `0.53599924`、PESQ 目標は `0.0813806`〜`0.0813924`。この試行でも実装間差のすべてを公式自己再実行のばらつきと判定できない。step 2 の checkpoint は保存済みだが再開比較は未実施。GPU process wall clock `36.90` 秒（集計含む）、ジョブ `115141`。以前の `6592.836` 秒に加算して `6629.736` 秒、残り `11370.264` 秒（SEMamba++ 進行中分は未計上）。

H100 MIG の同じ4更新と同一プロセス内 step 2 再開 `four-step-resume-gpu.json` でも、4 step 全ての前処理・forward・loss・勾配・更新重み差および再開後のモデル・optimizer 差は 0。実行時間 `6.32` 秒、ジョブ `115141`。先の `6629.736` 秒に足して `6636.056` 秒、残り `11363.944` 秒（SEMamba++ 集計中の時間は未計上）。

### SEMamba++ seed 17 の新規20 step（2026-11-03）

HEAD `298b283`、H100 MIG、既存ジョブ `115141`、seed 17、公式幅・fused、epoch scheduler、合成16 kHz 8,000 samples。`semamba-seed17-repeat-20261103/` の4プロセスは条件照合 valid、全20 step の損失・勾配・重みを保存した。step 1 の勾配相対 L2 は全ペア `1.45e-6`〜`1.49e-6`。step 4 は公式 A/B `0.03755`、独立 A/B `0.04070`、公式/独立 `0.03680` / `0.04063`。step 20 の勾配 cosine は順に `0.7216`、`0.7107`、`0.6420` / `0.6779`、相対 L2 は `0.7959`、`0.7403`、`0.7902` / `0.7356`。前回の seed 17 の低い実装間 cosine を無視せず、この新規試行でも同一実装を下回ったことを記録する。

step 20 の generator cosine は同順に `0.7277`、`0.7094`、`0.6395` / `0.6736`。CQT discriminator は `0.1792`、`0.7866`、`0.9409` / `0.8909`、MRD は `0.9981`、`0.9976`、`0.9997` / `0.9938`。全6ペア・各群の norm、最大差、相対 L2、cosine は `group-summary.json`、再集計方法は `tests/training/reference/summarize_semamba_four_process.py`。単一反復では実装間差と自己再実行の分布の同等性は判定できない。step 4 の checkpoint は保存したが、同一 checkpoint からの別実行は未実施。総合は PARTIAL。

4 worker の process wall clock は `82.020 + 59.779 + 59.975 + 59.727 = 261.501` 秒。全実行・CPU 集計を含む経過 `386.40` 秒を保守的に上限へ計上する（この CPU 集計の一部は他の GPU 実験と重複する）。直前の `6636.056` 秒に加算して `7022.456` 秒、残り `10977.544` 秒。別途 MP-SENet 再開の失敗・再試行は未計上。

上記 MP-SENet 公式更新順の step 2 checkpoint をコピーし、各系統を別プロセスで step 4 まで再開した。最初の `mp-official-loop-resume-20261103/` はコピー先のディレクトリ名が異なるため checkpoint の run name 検査に拒否された（4系統失敗、19.49秒）。元と同じ末尾名を新しい親ディレクトリに置いた `mp-official-loop-resume2-20261103/mp-official-loop-gpu-20261103/` は valid、34.60秒。各系統とも step 3 の forward・generator/discriminator loss・RNG が連続側と一致し、その更新後の model・optimizer hash は異なる。step 4 forward 最大差は `7.93e-7`〜`1.08e-6`、loss 差は最大 `1.37e-6`。両時間を上記 `7022.456` 秒に加えて累計 `7076.546` 秒、残り `10923.454` 秒。既存ジョブ `115141`、予約時間自体は除外。

SEMamba++ 公式 A の step 4 checkpoint を4系統に配る最初の試行 `semamba-shared-fork-20261103/` は公式2系統が成功、独立2系統は AdamW の moment と勾配の shape が合わず失敗した。公式と独立の generator は名前849件が一致するが、登録順の240位置が異なる。PyTorch の optimizer state は位置でロードされるため、実装間で optimizer checkpoint をそのまま流用できないことが原因であり、同一実装の通常の checkpoint 再開が失敗するという意味ではない。

`tests/training/four_process/shared_fork.py --remap-semamba` は、optimizer moment をパラメータ名で並べ替えて別実装へコピーする。小さな optimizer state の並べ替えと名前不一致拒否は `test_shared_fork` で確認した。新規 `semamba-shared-remap-20261103/semamba-seed17-repeat-20261103/` は4系統 valid。4系統は同一 step 4 のモデル・optimizer・scheduler・RNG と入力から始め、worker 識別子と独立 optimizer の格納順のみ変更した。step 5 forward と両損失は全ペア 0、勾配相対 L2 は `1.73e-6`〜`1.76e-6`。step 6 の勾配相対 L2 は公式 A/B `1.55e-4`、独立 A/B `1.34e-4`、公式/独立 `3.91e-5` / `1.99e-4`。更新後の重みは step 5 から分岐する。これは共通状態からの2 step の結果で、先の独立20 step の差を解消した証拠ではない。

最初の失敗試行 `57.20` 秒、parameter 登録順の調査 `5.70` 秒、共通 checkpoint の名前対応作成 `9.89` 秒、再試行・集計 `96.82` 秒を保守的に GPU 枠へ加算。累計 `7246.156` 秒、残り `10753.844` 秒。ジョブはすべて既存 `115141`。別の20 step 再反復は進行中。

今回の公式 checkout HEAD: MP-SENet `89932cfe90d1dacb8e170e4a331d762462c21792`、SEMamba++ `cd369c89a8106072c077b468417b765b227afd74`、ESPnet `0d136c5456d403cb2c0a4c13e76e8136e32e9b7a`、DeepFilterNet `d375b2d8309e0935d165700c91da9de862a99c31`、ClearerVoice `6b3774dc79c46ae8bed2a4fa5f706f0ac8c75c61`。独立のモデル・損失コードは HEAD `298b283` から変更していない。

追加比較の実行例（出力先は新規であること）:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venvs/tsubame/bin/python tests/training/reference/probe_tf_gridnet_official_stft.py --input data/results/training/mp-official-loss-gpu-20261103/input.pt --output data/results/training/NEW-tf-stft.json --steps 4
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venvs/tsubame/bin/python tests/training/reference/summarize_semamba_four_process.py data/results/training/semamba-seed17-repeat-20261103 --output data/results/training/NEW-semamba-groups.json
```

TF-GridNet の蓄積補助診断 `accumulation-adam-diagnostic.json` は縮小モデル・合成2区間、公式 Adam lr `0.001`・betas `(0.9, 0.999)`・clip 5、全バッチと micro-batch 2回を比較する。clip 後の勾配 norm は両者約 5、勾配最大差 `1.13e-6`、相対 L2 `9.83e-7`。最大重み差 `0.00086475245` は `blocks.0.attn_conv_K_0.2.beta` で、勾配 `-7.70e-9` / `7.53e-9` と Adam epsilon `1e-8` から同じ Adam による更新差 `0.00086475248` を再現した。key に時刻共通の定数を足すと softmax の出力が不変になる単体 float64 対照では、出力最大差 `3.33e-16`、定数への勾配最大 `1.33e-15` だった。これは当該 beta の微小勾配差の構造的な説明であり、蓄積の実音声・公式幅・複数更新全体を検証したものではない。初回の AdamW・clip 無し試行 `accumulation-diagnostic.json` は公式設定ではないため区別して保存した。再実行は `tests/training/reference/diagnose_tf_accumulation.py`。

同じ蓄積診断を公式幅・VoiceBank 1秒の異なる2区間で実行した `accumulation-voicebank.json` は、clip 後の勾配相対 L2 `1.61e-6`、最大差 `1.80e-6`。最大重み差 `0.00011148387` は `blocks.0.attn_conv_K_1.2.beta` で、対応勾配 `-5.49e-10` / `6.32e-10` から Adam の初回更新差 `0.00011148386` を再現した。CPU の1更新であり、多数更新の蓄積経路は別途必要。

同じ seed 17・同じ入力ハッシュ `8b71d25461fce9e5ef919b56746b4a81b3f7628e7f0835590ec9d6b70b67520f` の20 step を再反復し、2試行・8系統の最終勾配の全28ペアを `summarize_semamba_distribution.py` で集計した（`semamba-seed17-repeat2-20261103/two-trial-distribution.json`）。cosine は公式どうし6ペア min/median/max `0.0753/0.2030/0.7216`、独立どうし6ペア `0.2364/0.3656/0.7107`、実装間16ペア `0.0153/0.4439/0.7312`。実装間分布は自己再実行と大きく重なる一方、最小値は自己再実行より低い。ペアは同じ系統を共有し統計的に独立ではなく、2試行だけで分布一致を証明しない。前回の seed 17 の `0.380/0.422` は除外せず、その別記録を上に残す。追加試行の全 step 比較と GPU 時間は集計終了後に記録する。

4プロセス probe の公式 TF-GridNet も `tests/training/reference/official_tf_loader.py` に切り替え、ESPnet の公式 encoder・decoder を実行するようにした。VoiceBank 1秒の CPU 1更新 `tf-official-stft-four-cpu-20261103/` と、H100 MIG・epoch scheduler（更新中は plateau scheduler を固定）4更新 `tf-official-stft-four-gpu-20261103/` はともに valid。公式 A/B・独立 A/B・公式/独立の全 step で forward・loss・勾配・更新重みの最大差は 0。GPU 全 process wall clock `38.85` 秒、既存ジョブ `115141`。直前の `7246.156` に加え `7285.006` 秒、残り `10714.994` 秒。SEMamba++ 再反復と TF-GridNet 別プロセス再開は未計上。

追加20 step `semamba-seed17-repeat2-20261103/` の4系統 report は valid。step 20 勾配 cosine は公式 A/B `0.5449`、独立 A/B `0.4274`、公式/独立 `0.3536` / `0.6329`、相対 L2 は `0.9035`、`0.9758`、`1.1942` / `0.8519`。全実行・CPU 集計込み process wall clock `373.78` 秒を保守的に加算し、直前の `7285.006` 秒から累計 `7658.786` 秒、残り `10341.214` 秒。CPU 集計時間の一部は TF-GridNet GPU 処理と重複する。既存ジョブ `115141`、予約時間自体は除外。

TF-GridNet の公式 STFT 経路 4プロセス GPU 4更新の step 2 checkpoint を各系統別にコピーし、別プロセスで4更新まで再開した `tf-official-stft-resume-20261103/tf-official-stft-four-gpu-20261103/` は valid。step 3・4 とも4系統で連続側と model・optimizer・scheduler・RNG の hash が一致した。所要 `34.43` 秒を加算し累計 `7693.216` 秒、残り `10306.784` 秒。ジョブは `115141`。

再反復の step 20 component cosine は公式 A/B・独立 A/B・公式 A/独立 A・公式 B/独立 B の順に、generator `0.5412/0.4224/0.3544/0.6285`、CQT discriminator `0.7555/0.7388/0.0464/0.9960`、MRD `0.9995/0.9974/0.9986/0.9994`。全6ペアの最大差・norm・相対 L2 は `semamba-seed17-repeat2-20261103/group-summary.json`。CQT の低い実装間 cosine は前回の seed 17 と同じ方向だが、別の今回試行では公式 A/B が `0.1792` で実装間の方が高い。系統差か非決定性のばらつきかは未解決。

MP-SENet は公式幅・VoiceBank 1秒の CPU 1 step で、公式 `train.py` の損失式・公式 PESQ と独立損失を別々に実行した単一プロセス対照 `mp-official-loop-cpu-20261103/waveform-gradient.json` を追加した。全 generator 損失成分差と PESQ 目標差は 0、generator 出力波形の勾配最大差 `5.82e-11`。これは波形勾配の1 step 対照で、4 step の分岐量の説明には代用しない。

### DeepFilterNet3 公式 libdf 前処理の追加診断（2026-11-03）

既存4プロセスの native CUDA 比較は公式・独立とも独立 DFStream を使う。別前処理の影響を見るため、公式側 libdf 0.5.6 と独立 DFStream を選べる probe を追加した。最初の CPU 4プロセス `df-libdf-four-cpu-20261103/` は probe 設定の初期化ミスで開始前に失敗し、修正後の `df-libdf-four-cpu-retry-20261103/` は VoiceBank 48 kHz・4,800 samples・1更新で valid。公式 A/B、独立 A/B は完全一致。公式/独立は forward 最大差 `2.09e-7`、loss 差 `0.0006294`、勾配最大差 `0.003249`・相対 L2 `0.0004976`、重み最大差 `0.0001966`。synthesis は両側とも独立の `_vorbis_istft` を使っており公式の全データ経路ではない。

別の同一入力・seed 3 の CPU 対照 `df-libdf-cpu-20261103/` で、モデルと損失を両側とも公式に固定し、libdf と DFStream の前処理だけを変更した。公式/独立モデルの対照と前処理のみの対照は共に loss 差 `0.0021400`、波形最大差 `1.30e-7`、勾配最大差 `0.003141`・相対 L2 `0.0003912`、重み最大差 `0.00199005` になった。最初に観測した差は前処理スペクトルで、最大 `2.24e-8`（実数・虚数の両方を比較）。seed 3・同じ公式モデル対照の最大重み差 `enc.df_conv0.2.weight` に対応する clip 後勾配 `4.24e-6` / `-1.31e-6` を同じ AdamW に与えると更新差 `0.00199005` を再現した。これはその要素の1更新差の説明であり、複数更新の差をすべて説明しない。

初期の探索結果 `df-cuda-paths-20261103/libdf-cpu1.json` は seed を固定しておらず、複素スペクトル差を実部のみで計算していたため、上の確定数値には使わない。`df-libdf-cpu-20261103/one-step.json` は seed 固定前の別初期値である。確定対照の再現コードは `tests/training/reference/probe_df_libdf.py`。公式 native CUDA と独立 native CUDA の前回の比較は共通 DFStream 条件であることを区別し、libdf を含む GPU 複数更新は未判定。

libdf と DFStream を別々に実行する CPU 4プロセス VoiceBank 4更新 `df-libdf-four-cpu4-20261103/` は valid。公式 A/B・独立 A/B は全 step で完全一致。公式/独立は step 1 の勾配相対 L2 `0.000498` から step 4 `0.10480`（cosine `0.99475`）へ増加し、step 4 の波形最大差 `4.15e-5`、loss 差 `0.0003891`、重み最大差 `0.0004047`。自己再実行との差は 0 である。1 step の frontend と AdamW の説明だけではこの4更新の増幅を説明しない。公式 native CUDA と独立 native CUDA の libdf 込み比較は未実施。

上の libdf 診断の当初版は alpha の計算に独立ヘルパーを使っていた。公式 `df/utils.py` の `get_norm_alpha` と `_calculate_norm_alpha` を直接実行するよう比較経路を修正し、同一 seed・モデル・入力の `official-alpha-seed3.json` で前処理、forward、loss、勾配、重みの値が先の対照と全て一致した。公式・独立の alpha の値はこの設定で同じである。

公式 libdf（alpha は公式コード）と独立 DFStream、両側 native CUDA の VoiceBank 4更新 `df-libdf-four-gpu-20261103/` は valid。step 1 勾配相対 L2 は公式 A/B `1.09e-6`、独立 A/B `1.56e-6`、公式/独立 `0.0002001` / `0.0001999`。step 4 は `0.003875`、`0.002891`、`0.007310` / `0.012201`、波形最大差は公式/独立 `0.0001362` / `0.0001364`。各モデルの spectral loss は native CUDA、逆変換は両側とも独立の `_vorbis_istft`。共通 DFStream の旧 native 比較より実装間差が大きい。所要 `37.88` 秒を `7693.216` 秒へ加算して `7731.096` 秒、残り `10268.904` 秒。SEMamba++ 3回目は未計上。

`official-fft-seed3.json` の先頭フレーム対照では、libdf の解析窓と独立の窓の最大差 0、ERB 幅も一致した。同一の窓付き波形に対し、libdf の解析 FFT と PyTorch float32 FFT・正規化の出力差は最大 `5.27e-9`。PyTorch float64 FFT を float32 へ戻した対照でも最大 `5.27e-9`。公式 Rust は `realfft`、独立は `torch.fft.rfft` であり、窓の取り違えではなく FFT・正規化を含む演算段階まで差を絞ったが、FFT のどの内部演算かは特定していない。独立実装を公式コードに実行時依存させる変更は行わない。

seed 17 の3回目 `semamba-seed17-repeat3-20261103/` も20 step・4系統 valid。step 20 の全勾配 cosine は公式 A/B `-0.0436`、独立 A/B `0.5348`、公式/独立 `0.6798` / `-0.1824`。generator は同順に `-0.0333/0.5379/0.6777/-0.2148`、CQT は `-0.7356/0.0864/0.7984/0.9433`、MRD は `0.9982/0.9981/0.9952/0.9960`。3試行の全ペア `three-trial-distribution.json` では cosine の min/median/max が公式内15ペア `-0.0436/0.2421/0.7216`、独立内15ペア `0.2364/0.4274/0.7107`、実装間36ペア `-0.1824/0.4058/0.7583`。同じ系統を共有するペアは独立標本ではなく、分布の重なりだけで実装差の不在を証明しない。初回 seed 17 の悪い結果も引き続き除外しない。process wall clock `374.88` 秒を保守的に加算し累計 `8105.976` 秒、残り `9894.024` 秒。既存ジョブ `115141`、予約自体は除外。

比較 adapter 更新後の training 再実行は `DF_REFERENCE_CONFIG` を指定して104件・skip 3、1 error。`test_trainer_matches_the_index_plan_for_one_and_two_ranks` の rank 0 が空きポート選択後に `EADDRINUSE` となり、rank 1 の待機が180秒で timeout した。該当テストを単独再試行すると2.615秒で成功した。ログは `/tmp/audio-db-training-recheck-20261103.log`。この再実行を全件成功とは扱わない。前の103件成功は adapter 変更中の実行である。

MP-SENet の公式 A・step 2 checkpoint を、optimizer state を名前で対応付けて4系統へ配った `mp-shared-loss-retry-20261103/mp-official-loop-gpu-20261103/` は valid。最初の `mp-shared-loss-20261103/` は直接実行時の `tests` import path の不備で準備に失敗し、path を修正して新しい出力先で再試行した。共通状態の step 3 は全ペアで forward・G/D loss 差 0、勾配相対 L2 `2.57e-8`〜`2.60e-8`。step 4 の勾配相対 L2 は公式 A/B `6.55e-6`、独立 A/B `2.88e-5`、公式/独立 `1.02e-5` / `2.90e-5`。この2更新の結果は初期状態を共有した局所比較であり、独立した4更新試行すべての差を単独で説明しない。GPU 所要 `64.83` 秒、既存ジョブ `115141`。直前 `8105.976` 秒から累計 `8170.806` 秒、残り `9829.194` 秒。

MP-SENet の同じ条件・同じ入力ハッシュの公式更新順4 step 再反復 `mp-official-loop-repeat-20261103/` も valid。step 4 勾配相対 L2 は公式 A/B `0.007866`、独立 A/B `0.07009`、公式/独立 `0.06985` / `0.005676`。2試行8系統・全28ペアの `two-trial-distribution.json` では相対 L2 の min/median/max が公式内 `0.00787/0.02268/0.02575`、独立内 `0.00627/0.07003/0.07514`、実装間 `0.00568/0.04733/0.07600`。実装間差は独立自己再実行の範囲と重なるが、公式自己再実行を超える例もあり、ペアは独立標本ではない。初回の小さい勾配の AdamW 増幅と共通 checkpoint からの局所分岐は確認したが、異なる複数 step すべての数値差を一意に説明したとはしない。所要 `37.02` 秒を加算し累計 `8207.826` 秒、残り `9792.174` 秒。既存ジョブ `115141`。

参考として、3試行12実行の cosine 行列に対し「同一ラベル内の平均 cosine − 実装間平均 cosine」を算出し、実装ラベルを6対6で置換する全924通りの探索的検定を行った（`exploratory-permutation.json`）。観測値 `-0.01512`、正方向の片側置換 p 値 `0.487`。この小標本では実装別の集まりを検出していないが、事後的な探索であり、p 値が大きいことは同等性の証明ではない。再現は `tests/training/reference/permutation_semamba.py`。

## 受入項目の現状（2026-11-03、今回の追加試験後）

以下の GO は記載した短時間・入力条件の学習処理についての判定であり、長時間学習後の品質を意味しない。PARTIAL は対応する処理が動かないという意味ではなく、同等性の根拠が足りない項目を含む。旧結果は独立のモデル・損失コードが `298b283` から変わっていないことと、上記の公式 checkout HEAD を確認したうえで使用する。

| 項目 | MP-SENet | SEMamba++ | TF-GridNet ms-snsd | DeepFilterNet3 | MossFormerGAN |
|---|---|---|---|---|---|
| 公式構造 | GO（公式幅、strict load） | GO（公式幅、strict load） | GO（公式幅、strict load） | GO（公式 DfNet、strict load） | GO（公式 G/D、strict load） |
| Forward | GO（実音声1 step） | GO（同一状態、fused） | GO（実音声、公式 STFT、4 step 差0） | PARTIAL（libdf 込み4 step 差あり） | GO（実音声、4 step） |
| 損失 | GO（公式式・PESQ、成分1 step 差0） | GO（成分別対照、共通状態） | GO（公式損失・4 step 差0） | PARTIAL（libdf 込み損失差、独立合成を共用） | GO（公式 loss と実 PESQ、device adapter） |
| Backward | PARTIAL（独立4 step の差の全量未特定） | PARTIAL（20 step 分布が広い） | GO（4 process 4 step 差0） | PARTIAL（libdf 込み、step 4 相対 L2 `0.0073/0.0122`） | GO（実音声4 step、相対 L2 `0.0028/0.0039` と公式 A/B `0.0041`） |
| Optimizer 更新 | PARTIAL（微小勾配1 step 説明、複数 step は未確定） | PARTIAL（共通状態では再実行差、20 step 未確定） | GO（4 step 差0） | PARTIAL（公式 FFT 差が増幅） | GO（微小勾配の AdamW 更新差を再現） |
| Scheduler・更新順 | GO（公式 D→G・epoch 単位、比較 probe） | GO（公式 D→G・epoch 単位、epoch 内） | GO（clip 5、epoch 中 plateau 固定） | GO（公式表に対応する per-step schedule） | GO（G/D backward・clip 後に両 step） |
| 勾配蓄積 | PARTIAL（共通 trainer と短い対照、公式幅実音声未完） | PARTIAL（DDP/蓄積対照あり、20 step との同時検証なし） | GO（公式幅実音声 CPU 1更新、勾配相対 L2 `1.61e-6` と Adam の差を説明） | PARTIAL（公式 batch と独立蓄積の十分な比較なし） | GO（実音声、蓄積2、4更新） |
| 公式 A/B 再実行 | GO（4 step、差を記録） | GO（seed 17 の3試行、差を記録） | GO（実音声4 step 差0） | GO（native・libdf、差を記録） | GO（実音声4 step、勾配相対 L2 `0.00407`） |
| 独立 A/B 再実行 | GO（4 step、差を記録） | GO（seed 17 の3試行、差を記録） | GO（実音声4 step 差0） | GO（native・DFStream、差を記録） | GO（実音声4 step、勾配相対 L2 `0.00195`） |
| 公式/独立比較 | PARTIAL（2試行・共通 checkpoint、公式 A/B より大きい例） | PARTIAL（3試行でも20 step の差の帰属未確定） | GO（公式 STFT、実音声4 step 差0） | PARTIAL（libdf と DFStream の差が自己再実行を超える） | GO（実音声4 step、公式 A/B より小さい勾配差） |
| checkpoint 再開 | GO（同一 checkpoint、別プロセス。CUDA 非決定性あり） | PARTIAL（同一 checkpoint から2 step、20 step 継続は未判定） | GO（実音声 GPU、別プロセス、状態 hash 一致） | GO（独立決定的経路は一致、公式 native は非決定的） | GO（同一 checkpoint からの別プロセス分岐） |
| VoiceBank 実音声 | GO（1秒、4更新） | GO（1秒未満、4更新；20 step は合成） | GO（1秒、4更新） | GO（48 kHz、0.1秒、4更新） | GO（1秒、蓄積2、4更新） |
| 総合 | PARTIAL | PARTIAL | GO | PARTIAL | GO |

TF-GridNet の GO は 4 step・実音声・公式前処理の範囲である。MossFormerGAN の GO は PESQ を CPU 上で計算して device に返す比較 adapter、実音声1秒・蓄積2・4 step の範囲である。DeepFilterNet3 の決定的 CUDA は独立実装の再現性用であり、公式 native とビット一致しないこと自体を NO-GO としていない。一方、公式 libdf 前処理の有無が4 step の差を増やすため総合 GO は保留する。

前処理以外を切り分ける CPU 対照 `df-shared-libdf-cpu4-20261103/` は、公式と独立の両モデルに公式 libdf の前処理を与えた VoiceBank 4プロセス4更新である。4系統すべての全 step で forward・loss・勾配・重みの最大差は 0。独立の標準 frontend に置き換えた前述の CPU 比較では step 4 勾配相対 L2 が `0.10480` となるため、この条件の差は入力前処理を変えたことに依存する。これは標準の独立 frontend を検証した結果ではない。

公式 libdf を両モデルに与える GPU native 対照 `df-shared-libdf-gpu4-20261103/` は valid。step 4 勾配相対 L2 は公式 A/B `0.001795`、独立 A/B `0.002870`、公式/独立 `0.001937` / `0.006365`。標準の独立 DFStream を使う対照の公式/独立 `0.007310` / `0.012201` と比べると小さいが、この1組の GPU 試行だけで全てを前処理差と断定しない。所要 `35.34` 秒を加算し、GPU 計上累計 `8243.166` 秒、上限まで `9756.834` 秒。既存ジョブ `115141`。

同条件の libdf 対 DFStream・native CUDA 4更新再反復 `df-libdf-four-gpu-repeat-20261103/` は valid。step 4 の勾配相対 L2 は公式 A/B `0.003645`、独立 A/B `0.000496`、公式/独立 `0.015192` / `0.012055`。実装間は引き続き自己再実行より大きい。所要 `33.58` 秒を加算し、GPU 計上累計 `8276.746` 秒（137分56.746秒）、残り `9723.254` 秒（162分3.254秒）。既存ジョブ `115141`、予約時間そのものは含めない。計上には CPU 集計・重複時間を保守的に含む。
