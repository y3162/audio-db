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

TF-GridNet、SEMamba++、MossFormerGAN、DeepFilterNet3、StuPASE、UniPASE の 1 step は `tests/training/test_model_tasks.py` が CPU で確認した。レベルは [OTHER_MODELS.md](OTHER_MODELS.md)。SEMamba++ の `A_log` は未初期化だと selective scan が非有限になる。S6 の `A_i = i` で初期化する。CUDA の `mamba_ssm` 融合カーネルとの学習ステップ比較はしていない。

## MP-SENet の AdamW 更新差（2026-10-31）

同じ縮小条件（`dense_channel=4`、`n_fft=64`、長さ 2048、batch 2、eval）で公式モジュールと 1 step を比較した。STFT、Generator、Discriminator、損失の最大絶対誤差は 0。Discriminator 勾配の最大絶対誤差は 0。Generator 勾配の最大絶対誤差は 6.43e-8。AdamW（lr `5e-4`、betas `(0.8, 0.99)`、weight decay `0.01`、epsilon `1e-8`、初期モーメント 0）のあとの重みの最大絶対誤差は 4.052668809890747e-4。

初回ステップは `param * (1 - lr * weight_decay) - lr * grad / (|grad| + eps)` である。独立実装の更新後重みは、この式と最大絶対誤差 1e-6 未満で一致した。二つのモデルの重み差も、同じ式を二つの勾配に適用した差と 1e-6 未満で一致した。weight decay は同じ初期重みに掛かるので、重み差には残らない。

最大の重み差が出た要素は `dense_encoder.dense_block.dense_block.3.1.bias` の 1 要素である。独立実装の勾配は -8.90e-9、公式の勾配は 5.14e-9 で、符号は逆、絶対差は 1.40e-8 である。どちらも epsilon より小さく、`grad / (|grad| + eps)` は符号付きで約 -0.47 と 0.34 になる。その差に lr `5e-4` を掛けると 4.05e-4 になる。勾配の絶対値が 1e-6 を超える 4930 要素では、符号の不一致は 0 だった。したがって、重みの最大差は「勾配の最大誤差 6e-8 がそのまま増幅された」のではなく、epsilon 未満で符号が違う要素が初回 AdamW で O(lr) まで広がった結果である。学習式は変えていない。

## DeepFilterNet3、TF-GridNet、段階学習（2026-10-31）

ホスト `turing`。lab 環境。CPU の `tests/training` は 47 件、`tests/se` は 28 件、`tests/evaluation` は 81 件、`tests/asr` は 5 件。いずれも失敗 0。

| 確認 | 結果 |
|---|---|
| DeepFilterNet3 公式幅（`conv_ch=64`、48 kHz、長さ 4800）の 1 step。損失と勾配は有限。checkpoint を `load_enhancer` で読み、同じ発話の 2 回は一致 | PASS（CPU） |
| 同じ公式幅を物理 GPU 0（使用メモリ 15 MiB）で 2 回。`deterministic` と `CUBLAS_WORKSPACE_CONFIG=:4096:8`。損失はどちらも 38.875637。重みの SHA-256 はどちらも `89595b78d09bd7ec36be47e2e68576f3f2de7f0cee57f0ec35c3a5435d477d46`。学習後重みの推論は 2 回一致 | PASS |
| 物理 GPU 0 と 1 の `torchrun --nproc_per_node=2`。両 rank に同じ 1 発話を置いた 1 step を 2 回。rank 0 の損失と SHA-256 は単一 GPU と同じ | PASS。これは分割バッチの勾配一致ではない。両 rank の入力が同じなので、平均勾配は単一 GPU と一致する |
| TF-GridNet `ms-snsd` の公式幅（4 層、`n_fft=512`、stride 256、`emb_dim=32`、LSTM 128）で 1 step し、`load_enhancer` で長さ 1024 を処理 | PASS（CPU） |
| StuPASE の縮小 DeWavLM（層 1）、縮小 Vocoder、縮小 CFM。UniPASE の縮小 Adapter、Vocoder、PostNet。各 1 step、checkpoint 再開、Generator の重みが変わる | PASS（CPU）。WavLM-Large は使っていない |

WavLM-Large の公開先は Microsoft の [unilm/wavlm](https://github.com/microsoft/unilm/tree/master/wavlm)（コードは MIT）。README の Azure URL への HEAD は 403 で、ファイルは取得していない。`DeWavLM.pt` と `Vocoder_L24.pt` は公開配布を確認できず、ローカルにも無い。あるのは StuPASE の `DeWavLM-R.pt`（24 層、dim 1024）、`CFM.pt`、`Vocoder_Mel-16k.pt` と、UniPASE の `DeWavLM-Omni.pt`、`Adapter.pt`、`Vocoder_DWO-L1.pt`、`Vocoder_WavLM-L24.pt`、`PostNet.pt` である。名前が近いファイルを WavLM-Large や `Vocoder_L24.pt` としては使っていない。

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

結果 JSON は `data/results/training/official-compare-20261101/`。MossFormerGAN の NISQA Discriminator は実装も比較もしていない。UniPASE、StuPASE、論文スコアの完全再現は対象外である。

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
