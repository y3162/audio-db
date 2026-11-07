# P10-C 実行計画

本格学習はこの計画では開始しない。GPU 台数は公式プロトコルの要求である。どのマシンのどの GPU を使うかは、実行時の別指示に従う。

時間は `AUDIO_DB_STEP_SYNC=1` の今回の計測から置く。1 step は同期した更新時間とデータ待ちを足した値である。安全余裕は、その学習側に 1.3 を掛ける。検証の実測が全件のときは 1.5 を掛ける。早く plateau や early stop で終わる場合、この時間は上限である。

計測の詳細は [P10_B2_VERIFICATION](P10_B2_VERIFICATION.md)。

## MP-SENet

| 項目 | 値 |
|---|---|
| 公式設定 | `configs/training/reproduction/mp_senet_vb28_official.yaml`。AdamW lr 0.0005、betas 0.8 / 0.99、weight decay 0.01。ExponentialLR gamma 0.99。間隔 5,000。best ゲート epoch 40 |
| Dataset | `mp_senet.vb28.ondemand.official@1` |
| Snapshot | `36ab4590b113d0aa00884c44041eaa97d65085cee57bcc9c9e06135741fc76d7` / `265f256b938a89746dcaa62cd44daa2c6bcca3bce4cc1b1719eb2170717d33d5`。train 11,572、valid 824 |
| 学習 | 400 epoch。1 epoch は計画上 2,893 更新。合計 1,157,200 更新 |
| GPU | 2。global batch 4、micro 2、蓄積 1。A100 40GB で peak allocated 11.39 GB。全件検証の reserved は 21.6 GB |
| 更新時間 | 定常 15 step。更新の中央値 0.454 秒、待ちの中央値 0.069 秒。足した平均 0.534 秒。1,157,200 更新で 171.7 時間 |
| Validation | 5,000 更新ごと。231 回。全件 824 発話の実測は 778.3 秒。231 回で 49.9 時間。5,000 の倍数と epoch 末 2,893 は、この予算の中では一致しない。Scheduler は epoch 末だけ進む |
| Checkpoint | 5,000 更新ごと、最大 231 個。1 個は 29,689,087 バイト。合計 6.39 GiB。書き込みは 1 回 0.5 から 0.9 秒。最初の best 対象は step 120,000 |
| 上限の合計 | 学習 171.7 時間に 1.3 で 223.2 時間。検証 49.9 時間に 1.5 で 74.9 時間。I/O は 1 時間未満。確保の目安は約 299 時間 |
| 中断 | `verification_max_updates` は外す。途中停止は `interrupted.json`。`complete.json` ができたランは再開しない。16 更新の中断再開は、case id、plan hash、学習率が連続ランと一致した。Scheduler 状態は保存するので、公式コードのような再開時の余分な減衰は起きない |
| 公式との差 | Scheduler を保存する。16 kHz 変換器は未確定のまま soxr HQ |
| 開始条件 | このスナップショットで、公式幅 2 GPU の中断再開と全件検証時間は確認した。`verification_max_updates` を外し、間隔 5,000 と `max_steps` 1,157,200 を維持する |

## MossFormerGAN

| 項目 | 値 |
|---|---|
| 公式設定 | `configs/training/reproduction/mossformergan_vb28_official_2gpu.yaml`。121 epoch、`max_steps` 350,053。検証と Checkpoint は 2,893 更新ごと。best の公開ゲートは epoch 40。Scheduler 名は `none`。plateau が G の LR を操作する |
| Dataset | `mossformergan.vb28.ondemand.official@1` |
| Snapshot | `153096c312f8027bba93fa635224bff8681170841bbe40164d189f923d16aebe` / `ba0c77c94babbbc2c3e70b149554f8a922a68cf6fbe71ccf833927dc71dff555`。train 11,572、valid 824 |
| GPU | 2。global 4、micro 1、蓄積 2。A100 40GB で peak allocated 12.58 GB |
| 更新時間 | 定常 11 step。更新の中央値 1.038 秒、待ちの中央値 0.008 秒。足した平均 1.089 秒。350,053 更新で 105.9 時間 |
| Validation | 上限 121 回。824 件の実測は 237.94 秒。121 回で 8.00 時間。PESQ ゲートを含む。1 件は 0 として Generator の分母 824 に残る |
| Checkpoint | 上限 121 個 × 136,454,907 バイト = 15.4 GiB。書き込み 1.86 秒 × 121 = 約 4 分 |
| 上限の合計 | 学習 105.9 時間に 1.3 で 137.6 時間。検証 8.00 時間に 1.5 で 12.0 時間。確保の目安は約 150 時間 |
| 中断 | 公式完了の前に止めるときは `interrupted.json`。12 更新の計測も `INTERRUPTED` で、`complete.json` は無かった |
| 公式との差 | plateau の半減と 10 回停止は CPU の合成履歴で確認し、今回の GPU 検証は 1 回なので半減していない。初期化が公式リポジトリの乱数とビット一致するとは見ていない |
| 開始条件 | このホストのスナップショットで開始できる。`verification_max_updates` と `validation_max_batches` と `max_updates_per_epoch` を外す。2 GPU のままにする |

10 回連続の非改善で 121 epoch より前に止まる。上の時間は止まらなかった場合の上限である。

## TF-GridNet

| 項目 | 値 |
|---|---|
| 公式設定 | 2 rank、global batch 4、micro 2、Adam lr 0.001、clip 5、`mr_l1_tfd`、ReduceLROnPlateau factor 0.7 patience 1。`num_samples_per_epoch` 5,000、chunk 48,000、shift 0.5、35 epoch。選択は `valid.loss` |
| Dataset | `tf_gridnet.ms_snsd.espnet_recipe@1`。train 7,345 / valid 1,835 / test 305 |
| Snapshot | `13c6058881ef86fe7506f705b1bd5a5bc68b313181410d8fdaaee20c33ef3fc6` / `a69a477af70bc43dbcabdafbd5c97ec8a26aac4debd51158079545de1e4f0757` |
| 更新総数 | 35 epoch の計画合計 553,521。1 epoch は 15,768 から 15,851。epoch 1 は 15,806。`max_steps: 2000000` は公式の更新数ではない |
| GPU | 2。A100 40GB で peak allocated 2.74 GB。全件検証の peak allocated は 0.53 GB |
| 更新時間 | 定常 15 step。更新の中央値 0.252 秒、待ちの中央値 0.117 秒。足した平均 0.391 秒。553,521 更新で 60.1 時間 |
| Validation | 1 回は 2,887 更新、重み 11,548、損失 7,132.23。同期した実測は約 592 秒。35 回で 5.76 時間。patience 5 の early stop で減ることがある |
| Checkpoint | epoch 末に残すと 35 個 × 31,071,895 バイト = 1.01 GiB。書き込みは 1 回約 0.63 秒。`keep_nbest` の epoch ファイルは別に平均 1 個 |
| 上限の合計 | 学習 60.1 時間に 1.3 で 78.2 時間。検証 5.76 時間に 1.5 で 8.63 時間。確保の目安は約 87 時間 |
| 中断 | 6 更新で `INTERRUPTED`。`complete.json` は無い。再開と連続ランの case id、plan hash、学習率 0.001 は一致した |
| 公式との差 | shape file 順は未確定。生成は seed 付き。余り batch の公式挙動は、公開レシピが割り切れるため未使用 |
| 開始条件 | このスナップショットで、1 epoch の更新数と公式幅 2 GPU の中断再開、全件検証時間は確認した。`max_updates` と `validation_max_updates` と `verification_max_updates` を外し、`max_epochs: 35` を維持する |

## TF-Locoformer

現行 Dataset の検証は案 A である。案 B と案 C の時間も併記する。YAML は切り替えていない。

| 項目 | 値 |
|---|---|
| 公式設定 | DNS medium。4 rank、global 4、micro 1。chunk 64,000、shift 0.5。rank あたり 1,000 発話。150 epoch。warmup 4,000。plateau factor 0.5、patience 3。early stopping patience 10。`keep_nbest` 5 |
| Dataset | `tf_locoformer.dns.accept_recipe@1`。`cv_synthetic_small` ではない |
| Snapshot | `5469f0416329a8661df2cc91cd5bd9a149d6ea223c72e69ebdd5696bea6bdfbe` / `3343a55efb4c57d9ffef9179fb2f39c3e0186d36f587a0f4ad9f0aa14e9926f5`。1 epoch は 14,000 更新。150 epoch で 2,100,000 更新。長さがすべて 480,000 で、offset の乱数幅が 0 なので、epoch ごとの chunk 数は変わらない |
| GPU | 4。A100 40GB で peak allocated 16.47 GB |
| 更新時間 | 定常 11 step。更新の中央値 0.374 秒、待ちの平均 0.272 秒。足した平均 0.659 秒。2,100,000 更新で 384.4 時間 |
| Checkpoint | epoch 末に 150 個残すと 150 × 179,910,164 バイト = 25.1 GiB。書き込みは 1 回 2.29 秒。`ave_5best` は上位 5 epoch の追加ファイル |
| 中断 | 12 更新は `INTERRUPTED`。`complete.json` は無い。学習率は step 番号に比例した。P10-B.1 の公式幅 4 GPU で、中断再開と連続ランの case id と plan hash は一致している |
| 公式との差 | 平均 Checkpoint の拡張子は `.pt`。検証セット名は `cv_synthetic_small` ではない。`valid/si_snr` は公式 stats に無い |
| 開始条件 | Dataset は変えていない。150 epoch を始める前に、下の検証案のどれで選択するかを決める。`max_updates` と `validation_max_updates` と `verification_max_updates` を外す場合、現行コードは案 A の全件を毎 epoch 行う |

### 検証案

1 chunk あたり 0.1602 秒。1 発話は 14 chunk。rank あたり 126,000 chunk が案 A の 1 回である。

| 案 | 1 回 | 150 epoch | 学習 384.4 時間と合わせた上限 | 余裕込みの目安 |
|---|---:|---:|---:|---:|
| A. 現行 36,000 件を毎 epoch | 5.61 時間 | 841 時間 | 1,226 時間 | 学習に 1.3 で 500 時間。検証に 1.5 で 1,262 時間。合計約 1,762 時間 |
| B. 固定した小さい集合 | 発話数に比例。4 rank で約 0.560 秒/発話 | その 150 倍 | 学習 384 時間に検証を足す | 集合を決めてから同じ 1.3 と 1.5 を掛ける |
| C. `cv_synthetic_small` | 件数 UNKNOWN | 見積もれない | 見積もれない | 公式の一覧が無いので開始条件にできない |

案 A のまま 150 epoch を始めると、検証が学習より長い。patience 10 の early stop で短くなることはある。案 B は別 Protocol で、公式再現とは呼ばない。案 C は実装しない。

## 共通の開始前チェック

- `complete.json` を改名しない。
- 既存の `data/db/snapshots/`、`resource.duckdb`、`20261107-p10b-lab`、`20261107-p10b1`、`20261107-p10b2` を上書きしない。P10-C は別の出力ディレクトリにする。
- 学習用 WAV と波形キャッシュは作らない。
- DeepFilterNet3 をこのホストで動かすときは `DF_RFFT_LIBRARY` にローカルビルドを指定する。共有 `.so` は置き換えない。
