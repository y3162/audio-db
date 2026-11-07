# P10-B.1 検証

ホストは `tensor`。GPU は A100-PCIE-40GB（ドライバ 535.183.01）。Python は `.venvs/se`（3.12、torch 2.6.0+cu124）。HEAD は `b9774e9`。GPU 計測は `AUDIO_DB_STEP_SYNC=1`。Run ID は `20261107-p10b1`。既存の `20261107-p10b-lab` は上書きしていない。

このホストの `data/db/snapshots/` は次の 2 件だけである。

| Dataset | dataset id | snapshot hash |
|---|---|---|
| `mossformergan.vb28.ondemand.official@1` | `153096c312f8027bba93fa635224bff8681170841bbe40164d189f923d16aebe` | `ba0c77c94babbbc2c3e70b149554f8a922a68cf6fbe71ccf833927dc71dff555` |
| `tf_locoformer.dns.accept_recipe@1` | `5469f0416329a8661df2cc91cd5bd9a149d6ea223c72e69ebdd5696bea6bdfbe` | `3343a55efb4c57d9ffef9179fb2f39c3e0186d36f587a0f4ad9f0aa14e9926f5` |

`mp_senet.vb28.ondemand.official@1` と `tf_gridnet.ms_snsd.espnet_recipe@1` は無い。`reproduction.duckdb` は無い。どちらも作っていない。

## CPU

`CUDA_VISIBLE_DEVICES=""`。

境界テスト 38 件、10.1 秒、失敗 0、SKIP 1（`torch_complex` が無く ESPnet 公式 loss の符号比較を skip。自前実装の符号テストは成功）。

確認したこと:

- `verification_max_updates` で `INTERRUPTED` になり、`complete.json` は無い。上限を上げると同じディレクトリで再開する。同じ上限は拒否する。
- 公式 `max_steps` は `COMPLETED` のまま再開できない。注入した例外は `FAILED` で、`complete.json` は無い。
- 2 更新で 1 epoch の合成データでは、検証間隔 3 と Checkpoint 間隔 2 を混ぜても、ExponentialLR は epoch 末だけ進む。中断再開と連続ランの `last_epoch` と LR は一致し、再開で余分に減衰しない。連続ランの Scheduler step は 3、再開後は 2。重みは一致した。
- MP-SENet 公式 YAML は間隔 5,000、`max_steps` 1,157,200、ゲート 40、最初の対象 step は 120,000。
- MossFormerGAN plateau は、悪化で最良値を上書きして stall を増やす。同値も stall。5 回目は最良の重みと Optimizer を戻し、G の LR だけを半分にする。10 回目は止め、もう一度は半分にしない。Checkpoint の往復で stall は残る。
- TF-Locoformer の warmup 4,000 と plateau は、公式に合わせた CPU テストと再開テストが成功した。`ave_5best` は float 平均と整数 Tensor の和、5 件未満の symlink を確認した。

`DF_RFFT_LIBRARY` を付け、`tearDown` が環境変数を戻す修正と TensorBoard の step 順比較を入れたあとの一括実行は 239 件、1,722.080 秒、失敗 0、エラー 0、SKIP 12 である。SKIP は、固定 ESPnet ソースが無い chunk 照合、`libdf` 未導入、`mamba_ssm` が無い公式 SEMamba++ の照合、`torch_complex` が無い公式 loss の照合、CUDA が要る 2 件、再構築 DB が無い 1 件である。修正前の同じコマンドは 239 件、1,663.763 秒、失敗 2、エラー 6、SKIP 12 で、失敗は環境変数を消したあとの glibc と event の wall time 順だった。

## GPU の中断と再開

`tests/training/p10b_gpu_job.sh`。`complete.json` は移動していない。

### MossFormerGAN

物理 GPU 1 と 2。world 2、global batch 4、micro 1、蓄積 2。`verification_max_updates` は 2、再開と連続ランは 3。所要 134 秒。3 フェーズとも exit 0。

| 項目 | work（中断して再開） | continuous |
|---|---|---|
| step 1–3 の case id | 一致 | 一致 |
| plan hash | `173195611c6208f05027819cb48a03a787bebcc13c181baceac7b91ac01fda6f` | 同じ |
| dataset id / snapshot | 上の表 | 同じ |
| world size | 2 | 2 |
| LR | G 0.0005、D 0.001。3 step とも不変 | 同じ |
| `complete.json` | 無し | 無し |
| 終了 | `INTERRUPTED` | `INTERRUPTED` |

step 2 の Checkpoint は 272,796,207 バイト、書き込み 4.80 秒。step 3 は work が 272,834,607 バイト、3.91 秒。peak allocated は 12,575,379,968 バイト、reserved は 13,199,474,688 バイト。

損失の生成器は step 1 で一致（0.9778822064399719）。step 2 は work 0.5989946722984314、continuous 0.5989925265312195。step 3 は work 0.4379703998565674、continuous 0.43797728419303894。`deterministic: false` の CUDA 差である。Tensor の最大差は step 2 で 1.53e-3、step 3 で 1.91e-3。どちらも `discriminator.cnn.conv1.bias`。データ計画と LR は一致している。

検証は rank 0 の公式順の先頭 2 batch。生成器損失は work 0.4566635191440582、continuous 0.4566858112812042。PESQ 指標は `na`（採用ゲートであり plateau の指標ではない）。`best_checkpoint` は null。epoch 0 は公開ゲート 40 より前である。824 件ではない。

### TF-Locoformer

物理 GPU 1、2、6、7。world 4、global batch 4、micro 1。所要 409 秒。3 フェーズとも exit 0。peak allocated 16,470,292,992、reserved 17,630,756,864。

| 項目 | 結果 |
|---|---|
| case id | step 1、2、3 で work と continuous が一致 |
| plan hash | `1e194ba01a26a77e7ea9c4d68eb46fbbfdaf0bca0076ddf53044792b02fff3cd` |
| world size | 4 |
| LR | step 1 / 2 / 3 が 2.5e-7、5e-7、7.5e-7。`0.001 / 4000 * step`。再開で余分な減衰は無い |
| Checkpoint | step 2 は 179,910,292 バイト、5.38 秒。step 3 の再開側は 2.27 秒 |
| 終了 | `INTERRUPTED`。`complete.json` は無い |

step 3 の保存位置は epoch index 1、`update_in_epoch` 1。学習ログの epoch は 1 始まりの 2。work と continuous で同じである。

生成器損失は step 1 が 7947.0107421875 と 7947.0107421875（SI-SDR のログだけ 17.255243 と 17.255245）。step 3 は 3992.987548828125 と 3992.97607421875。重みの最大差は step 2 と step 3 の両方で 1.74e-3（`generator.separator.blocks.0.freq_path.ffn.1.deconv1d.weight`）。

検証は rank あたり 1 chunk。重み合計 4。生成器 6875.3355712890625。SI-SDR −4.634373351931572 dB。所要 11.75 秒。この 11.75 秒には 36,000 発話の計画作成が入る。全件ではない。`1epoch.pt` はある。`ave_5best` は 1 epoch では作られない。

### MP-SENet と TF-GridNet

このホストでは再実行していない。P10-B の H100 成功ジョブ 9293476（MP-SENet、ジョブ全体 92.510 秒、peak allocated 11,389,884,416）と 9293478（TF-GridNet、95.910 秒、peak allocated 2,743,491,584）は、`complete.json` を改名して再開している。新しい `interrupted.json` 経路の GPU 証拠には使わない。共有 Trainer の状態遷移は CPU で通り、非 chunk 経路は MossFormerGAN、chunk 経路は TF-Locoformer で GPU を通した。

## 異常終了

MossFormerGAN だけ実施した。物理 GPU 1 と 2。`verification_max_updates: 4`、Checkpoint 間隔 2、検証なし。step 2 の Checkpoint が書き終わったあと、プロセスがまだ生きている状態でそのプロセスグループへ `SIGKILL` した。`complete.json` は無かった。同じディレクトリを `step_00000002.pt` から再開し、exit 0 で step 4 まで進んだ。終了は `INTERRUPTED`。step 3 の Checkpoint ファイルは間隔 2 のため無い。学習行は step 1 から 4 まである。

TF-Locoformer、MP-SENet、TF-GridNet の強制終了は NOT_TESTED。

## 速度

同期した更新時間。先頭の 1 step は定常から外す。

### MossFormerGAN（12 更新、検証 32 batch）

| 統計 | 秒 |
|---|---:|
| step 1 | 2.791 |
| 定常 11 step の平均 | 0.971 |
| 中央値 | 0.967 |
| p95 | 1.019 |
| 最短 / 最長 | 0.954 / 1.019 |
| `load_wait_seconds` | 約 0.021–0.025 |

検証 32 batch、5.425 秒。生成器損失 0.298467212356627。対象 824 のうち 32。1 batch あたり 0.170 秒。この時間には PESQ ゲート（`metric_targets`）が入る。ログ上の PESQ 指標は常に `na`。

データ待ちは更新の約 2% なので、ボトルネックは GPU の更新である。peak allocated は中断ランと同じ 12,575,379,968。Checkpoint は約 260 MiB、書き込み約 4–5 秒。

### TF-Locoformer（12 更新、検証 32 chunk/rank）

| 統計 | 秒 |
|---|---:|
| step 1 | 1.273 |
| 定常 11 step の平均 | 0.377 |
| 中央値 | 0.379 |
| p95 | 0.388 |
| 最短 / 最長 | 0.371 / 0.388 |
| `load_wait_seconds` | 平均 0.187（0.119–0.222） |

更新より待ちは小さいが、無視はできない。1 step の壁時計はおよそ 0.564 秒。

検証は 1 chunk/rank が 11.753 秒、32 chunk/rank が 14.019 秒。差の 2.266 秒が 31 chunk 分で、1 chunk あたり 0.073 秒。固定分 11.68 秒は計画作成が大半である。32 件の生成器損失は 7864.884853363037、重み合計 128、SI-SDR −1.4251097971427953 dB。126,000 chunk/rank ではない。

Checkpoint は 179,910,292 バイト、このランの書き込みは 2.23 秒。

### MP-SENet と TF-GridNet

同期した step 時間は NOT_TESTED。P10-B のジョブ全体の壁時計は、初期化と 3 フェーズを含むので、step 時間としては使わない。

## Scheduler と Checkpoint 平均

CPU の結果は上の境界テスト。GPU では、MossFormerGAN の LR が再開後も G 0.0005 / D 0.001 のままで、TF-Locoformer の warmup が step 番号に比例した。どちらも再開で余分な減衰は無い。

`ave_5best` の実ファイルは、GPU ランが 1 epoch 相当で止めたため作っていない。数値は CPU の合成 5 epoch で確認した。

## 残課題

- 修正済み: 公式間隔 5,000、中断ファイル、DDP の待ち時間、`sys.path`、trial100、plateau の同値、`ave_5best` の数値。
- 環境: 共有 `libdf_rfft.so` の glibc。DeepFilterNet3 は対象外。
- 未計測: MP-SENet と TF-GridNet の、このホストでの公式幅 GPU。TF-GridNet の 1 epoch 更新数。4 モデルの全件検証。Locoformer 以外の強制終了は MossFormerGAN だけ実施。
- 公式情報が必要: `cv_synthetic_small`、TF-GridNet の shape file 順、MP-SENet の公式 16 kHz 変換器。
