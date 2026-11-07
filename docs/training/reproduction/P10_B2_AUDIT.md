# P10-B.2 監査

対象は MP-SENet、MossFormerGAN、TF-GridNet、TF-Locoformer。HEAD は `7979d931a9696f7a2b323309c2f68313a7248e49`（`feature/reproduction`）。本格再現学習は開始していない。数値は [P10_B2_VERIFICATION](P10_B2_VERIFICATION.md)、開始計画は [P10_C_EXECUTION_PLAN](P10_C_EXECUTION_PLAN.md)。

リポジトリの学習コードは変更していない。修正可能な不具合は 0 件である。

## スナップショット

P10-B.1 の時点で `tensor` に無かった 2 つのスナップショットは、このセッションの開始時に既に `data/db/snapshots/` にあった。このセッションでは移送していない。内容アドレス、件数、元素材の SHA-256、オンデマンド再構築を確認した。`reproduction.duckdb` はこのホストに無い。学習はスナップショットだけを読むので、コピーはしていない。

| Dataset | dataset id | snapshot hash | 件数 | 元素材 |
|---|---|---|---|---|
| `mp_senet.vb28.ondemand.official@1` | `36ab4590b113d0aa00884c44041eaa97d65085cee57bcc9c9e06135741fc76d7` | `265f256b938a89746dcaa62cd44daa2c6bcca3bce4cc1b1719eb2170717d33d5` | train 11,572 / valid 824 / test 824 | 一意 24,792 ファイル。欠落 0、SHA-256 不一致 0 |
| `tf_gridnet.ms_snsd.espnet_recipe@1` | `13c6058881ef86fe7506f705b1bd5a5bc68b313181410d8fdaaee20c33ef3fc6` | `a69a477af70bc43dbcabdafbd5c97ec8a26aac4debd51158079545de1e4f0757` | train 7,345 / valid 1,835 / test 305 | 一意 24,354 ファイル。欠落 0、SHA-256 不一致 0 |

MP-SENet の valid と test は同じ case 集合で、`selection_uses_test` は true。train との重なりは 0。world size 2、global batch 4、micro 2、蓄積 1 の epoch 0 は 2,893 更新。同じ case の再構築は最大絶対誤差 0。検証の先頭は全文（例: 27,861 サンプル）で、2 秒 crop ではない。

TF-GridNet の train / valid / test の case 重なりは 0。`selection_uses_test` は false。長さはスナップショットの計画長で、train の最小は 160,138 サンプル、valid の最小は 160,016 サンプル。chunk 48,000 以下の発話は 0。同じ chunk の再構築は最大絶対誤差 0。

## TF-GridNet の 1 epoch

音声は読んでいない。計画は `espnet_chunk_v1`、seed 0、world size 2、global batch 4、rank あたり元発話 5,000、chunk 48,000、shift 0.5、cache は `max(1024 // 2, 2) = 512`。公式 YAML（ESPnet `0d136c5` の `egs2/ms_snsd/enh1/conf/tuning/train_enh_tfgridnet.yaml`）は `num_iters_per_epoch: 5000`、`chunk_length: 48000`、`batch_size: 4`、`max_epoch: 35`。shift と cache の既定は pinned ESPnet の `ChunkIterFactory` と `abs_task.py` の `num_cache_chunks // world_size` である。

| 項目 | 値 |
|---|---|
| rank あたり元発話 | 5,000。shard は rank 0 が 3,673、rank 1 が 3,672 で、5,000 は周回する |
| per-rank batch | 2。global batch 4 は world size 2 で割り切れる |
| epoch 1 の更新 | 15,806。raw batch は 15,806 と 15,832。長い rank の 26 batch は `iterator_stop` で捨てる。端数 chunk は 0 |
| epoch 1 の plan hash | `550fdeabd54e01ad62e5f75bcdeda7ea487d6ff26f6e2d7da8072a68adc43144` |
| 35 epoch の更新 | 15,768 から 15,851。合計 553,521 |
| `max_steps: 2000000` | 公式の総更新数ではない |

検証は shuffle なし、全 1,835 件。rank の発話は 918 と 917。raw batch は 2,887 と 2,908。短い rank で止まり、更新は 2,887。捨てる batch は 21。保持 chunk は 11,548。plan hash は `734a3f63a5b5d554b1ee3319d3378c5eed6d4a8d2094da1aff312473b91f46d9`。

shape file 順は、呼び出し側の case 順のままである。公式の shape ファイルは手元に無い。

## 公式 Validation

### MP-SENet

公開間隔は 5,000 更新。ExponentialLR は epoch 末（2,893 更新）だけ進む。5,000 の倍数と 2,893 の倍数は、1,157,200 更新の範囲では一致しない。検証は 231 回。best PESQ のゲートは epoch 40 で、最初にゲートを満たす保存点は step 120,000。`selection_uses_test` は true。

今回の 16 更新では epoch は終わっていない。Scheduler の `last_epoch` は 0、学習率は Generator も Discriminator も 0.0005 のままである。

全文 824 件の検証は実施した。所要 778.3 秒。PESQ は 824 件とも有限で、平均 1.198。未学習の初期重みなので、公式の学習済みスコアではない。パラメータと buffer の最大差は 0。Optimizer は進んでいない。

### MossFormerGAN

検証は rank 0、公式順、batch 1、2 秒 crop。824 件の形状はすべて `(1, 32000)`。端数 batch は無い。Discriminator は `train` のまま、Generator は `eval`。Generator 損失の分母は 824。PESQ が落ちた 1 件は 0 として分母に残る。RI、magnitude、time、GAN の成分は、その 1 件を分母に入れないので count は 823。加重和を 823/824 倍すると、記録した Generator 平均 1.13466 と揃う。Plateau に渡す値はこの Generator 平均である。

検証そのものはパラメータを変えない。Discriminator の BatchNorm running statistics と `num_batches_tracked` は、公式が検証中も `train` にするため変わる。Optimizer の学習率は Generator 0.0005、Discriminator 0.001 のまま。最初の plateau 判定は改善で、stall は 0。最良 snapshot を取る。生きている重みは変わらない。半減は 5 回目である。今回は 1 回なので半減しない。

### TF-GridNet

検証は epoch 末。1 回の全件は 2,887 更新、重み合計 11,548、集約損失 7,132.23。DDP の all-reduce は損失の和と件数である。同期した所要は約 592 秒。35 回が上限で、early stopping（patience 5）で減ることがある。

### TF-Locoformer

現行の 36,000 件は sorted uttid の末尾 1/10 であり、`cv_synthetic_small` ではない。長さはすべて 480,000 サンプル。chunk 64,000、shift 0.5 では 1 発話 14 chunk。world size 4 では rank あたり 9,000 発話、126,000 chunk、余り batch 0。plan hash は `b4f3c3cd43dcd53d00b045cb880274605575a9665be9cbe2a710d1bf0cbc880b`。

128、512、2,048 chunk/rank の壁時計は 1 chunk あたり 0.157 から 0.169 秒で揃う。2,048 chunk の最も遅い rank は 0.1602 秒。全件 1 回は 5.61 時間。150 epoch すべてで行うと 841 時間。P10-B.1 の 2.56 時間は、1 chunk と 32 chunk の差から出した値で、今回の長い接頭辞より短い。

## `cv_synthetic_small`

固定した公式は merlresearch/tf-locoformer `7a615460d347ff7334a13dbb831d16280da72cdc`（v1.1.0）と、README が指定する ESPnet `90eed8e53498e7af682bc6ff39d9067ae440d6a4`。audio-db の ESPnet pin `0d136c5` の DNS 準備スクリプトは `90eed8e` とバイト一致だった。

確認できたこと:

- 保存済み `config.yaml` の検証入力は `dump/raw/cv_synthetic_small/wav.scp` と `spk1.scp`。`collect_stats` は false。`utt2spk` は null。shape ファイルは `exp/enh_stats_16k/valid/speech_mix_shape`。その shape ファイルは checkout に無い。
- `best_model_criterion` は `valid/si_snr/max` と `valid/loss/min`。`keep_nbest_models` は 5。`max_epoch` は 150。`patience` は 10。Scheduler の criterion は `valid/loss`。検証は epoch ごとに 1 回（pinned trainer の epoch ループ）。
- `SISNRLoss` の統計名の既定は `si_snr_loss`。設定の `name: si_snr` はクラス選択で、統計キーには渡っていない。`valid/si_snr` は stats に無い。
- ESPnet `dns_data_prep.sh` は sorted uttid の先頭 9/10 を `tr_synthetic`、残りを `cv_synthetic` にする。`run.sh` の `valid_set` は `cv_synthetic`。
- `90eed8e` の全ファイルパス 13,735 件に `synthetic_small` は無い。v1.1.0 の checkout でこの文字列があるのは、保存済み DNS config の 2 行だけである。
- README は、論文が 2,700 h / 300 h であることと、300 h の検証は長いので縮小を勧めることだけを書く。縮小後の件数、seed、選び方は書いていない。

件数、選び方、shape ファイルの中身は UNKNOWN。36,000 件を `cv_synthetic_small` とは呼ばない。

## Validation の 3 案

既存の Dataset と学習 YAML は変えていない。

| 観点 | A. 現行 36,000 件を毎 epoch | B. 固定した小さい検証集合 | C. `cv_synthetic_small` |
|---|---|---|---|
| 公式との対応 | 公開 `dns_data_prep.sh` の `cv_synthetic`（末尾 1/10）に対応する。保存 config の名前とは違う | 公式の検証集合ではない | 保存 config のパス名と一致させたい案。中身は UNKNOWN なので実装できない |
| 実現 | 現行コードのまま | 新しい Protocol か Dataset variant が要る。train からのリーク検査、case id、dataset id、snapshot hash を残す | 公式に一覧もスクリプトも無い |
| 1 回の時間 | 5.61 時間 | 発話数に比例。0.160 秒/chunk、1 発話 14 chunk、4 rank | 件数が UNKNOWN |
| 150 epoch | 検証 841 時間。学習 384 時間と足すと約 1,226 時間。余裕 1.5 を検証に掛けると検証だけで約 1,262 時間 | 集合の大きさで決まる | 見積もれない |
| Checkpoint 選択 | 広い。公式の保存 config が使った集合とは未対応 | 選択用と最終 test を分けられる。公式再現とは書かない | 特定できれば保存 config に最も近い |
| リーク | 現行の valid は train の末尾分割。test 150 件は選択に使わない | 話者・素材・生成条件の検査が条件 | UNKNOWN |
| 論文への書き方 | 「公開データ準備の 9/10 分割。`cv_synthetic_small` ではない」 | 「研究用の固定部分集合。公式の縮小検証ではない」 | 中身が分かるまで書けない |

判断の順は、公式との整合、独立した検証集合、計算資源、再現性、モデル間の公平性である。C は中身が特定できない。A は公開データ準備には合うが、保存済み学習 config の検証名とは違う。B は公式再現の claim にはできない。

P10-C で採用する検証プロトコルは、このセッションでは決めない。推奨は次のとおり。

1. いまの YAML と Dataset は A のまま残す。
2. 150 epoch を A の全件検証で走らせる計算は、学習本体より検証が長い。開始前に、全件を毎 epoch 行うかを決める。
3. 縮小するなら B を別 Protocol として新しく作る。現行の 36,000 件からの固定部分集合にし、選択用と最終 test を分ける。`cv_synthetic_small` とは呼ばない。
4. C は、公式の一覧か shape ファイルが得られるまで UNKNOWN のままにする。

## モデル別の残差

| モデル | 公式との差で残すもの | P10-C |
|---|---|---|
| MP-SENet | Scheduler 状態を保存する。公式の再開は ExponentialLR を余分に進め得る。16 kHz 変換器は不明のまま soxr HQ | スナップショット、中断再開、step 時間、全件検証時間は揃った |
| MossFormerGAN | plateau の半減と 10 回停止は、CPU の合成履歴では確認済み。今回の GPU 全件検証は 1 回で、半減には達していない。初期化のビット一致は見ていない | 824 件の検証は通った |
| TF-GridNet | shape file 順は未確定。生成は seed 付きで、公式の未シード生成とは別。余り batch の公式挙動は、4 と 2 が割り切れるため未使用 | 1 epoch の更新数と全件検証時間は確定した |
| TF-Locoformer | 平均 checkpoint の拡張子は `.pt`。検証名は `cv_synthetic_small` ではない。`valid/si_snr` は公式 stats に無い | 検証プロトコルは未決。Dataset は変えていない |

DeepFilterNet3 と SEMamba++ の公式 Dataset は blocked のままである。今回の共通コードは変えていない。
