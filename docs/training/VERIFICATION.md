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
