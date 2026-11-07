# P10-C 実行計画

本格学習はこの計画では開始しない。GPU 台数は公式プロトコルの要求である。どのマシンのどの GPU を使うかは、実行時の別指示に従う。

時間は `AUDIO_DB_STEP_SYNC=1` の計測から置く。安全余裕は、計測した更新・データ待ち・Checkpoint に 1.3 を掛ける。検証は計測点が全件より小さいので、別に 1.5 を掛ける。早く plateau や early stop で終わる場合、この時間は上限である。

計測の詳細は [P10_B1_VERIFICATION](P10_B1_VERIFICATION.md)。

## MP-SENet

| 項目 | 値 |
|---|---|
| 公式設定 | `configs/training/reproduction/mp_senet_vb28_official.yaml`。AdamW lr 0.0005、betas 0.8 / 0.99、weight decay 0.01。ExponentialLR gamma 0.99。間隔 5,000。best ゲート epoch 40 |
| Dataset | `mp_senet.vb28.ondemand.official@1`。このホストにスナップショットは無い |
| 学習 | 400 epoch。1 epoch 2,893 更新。合計 1,157,200 更新 |
| GPU | 公式幅の確認は 2 GPU。global batch 4、micro 2、蓄積 1 |
| 学習時間 | 同期した step 時間は NOT_TESTED。推定不能 |
| Validation | 5,000 更新ごと。1,157,200 は 5,000 で割り切れないので 231 回（最終は step 1,155,000）。1 回は公式 test 824 発話の全文。時間は NOT_TESTED |
| Checkpoint | 5,000 更新ごと、最大 231 個。1 個のバイト数は NOT_TESTED。最初の best 対象は step 120,000 |
| ストレージ | 推定不能。検証とログを別に残す |
| 中断 | `verification_max_updates` は外す。途中停止は `interrupted.json`。`complete.json` ができたランは再開しない。Scheduler 状態は保存するので、公式コードのような再開時の余分な減衰は起きない |
| 公式との差 | Scheduler を保存する。16 kHz 変換器は未確定のまま soxr HQ |
| 開始条件 | スナップショットを置く。新しい中断経路を、そのスナップショットで公式幅 2 GPU の数 step だけ確認する。全件検証の時間を、少量の接頭辞から見積もってから 400 epoch を始める |

## MossFormerGAN

| 項目 | 値 |
|---|---|
| 公式設定 | `configs/training/reproduction/mossformergan_vb28_official_2gpu.yaml`。121 epoch、`max_steps` 350,053。検証と Checkpoint は 2,893 更新ごと。best の公開ゲートは epoch 40。Scheduler 名は `none`。plateau が G の LR を操作する |
| Dataset | `mossformergan.vb28.ondemand.official@1` |
| Snapshot | `153096c312f8027bba93fa635224bff8681170841bbe40164d189f923d16aebe` / `ba0c77c94babbbc2c3e70b149554f8a922a68cf6fbe71ccf833927dc71dff555`。train 11,572、valid 824 |
| GPU | 2。global 4、micro 1、蓄積 2。A100 40GB で peak allocated 12.6 GB |
| 更新時間 | 定常 0.971 秒 × 350,053 = 94.4 時間。データ待ち 0.023 秒 × 350,053 = 2.2 時間 |
| Validation | 121 回が上限。32 / 824 件が 5.425 秒なので、824 件は約 140 秒。121 回で 4.7 時間。発話長のばらつきで前後する。PESQ ゲートの時間は含まれる |
| Checkpoint | 上限 121 個 × 約 260 MiB = 約 31 GB。書き込み 約 4.5 秒 × 121 = 約 9 分 |
| 上限の合計 | 更新 + 待ち + 検証 + I/O ≈ 101.5 時間。1.3 を掛けた更新側と 1.5 を掛けた検証で、約 130 時間を確保の目安にする |
| 中断 | 公式完了の前に止めるときは `interrupted.json`。`SIGKILL` のあとでも、完成した step 2 から step 4 へ再開できた。書き込み途中のファイルからの復旧は未確認 |
| 公式との差 | plateau の半減と 10 回停止は CPU の合成履歴で確認し、GPU の長時間ランでは未到達。初期化が公式リポジトリの乱数とビット一致するとは見ていない |
| 開始条件 | このホストのスナップショットで開始できる。`verification_max_updates` と `validation_max_batches` と `max_updates_per_epoch` を外す。2 GPU のままにする |

10 回連続の非改善で 121 epoch より前に止まる。上の時間は止まらなかった場合の上限である。

## TF-GridNet

| 項目 | 値 |
|---|---|
| 公式設定 | 2 rank、global batch 4、micro 2、Adam lr 0.001、clip 5、`mr_l1_tfd`、ReduceLROnPlateau factor 0.7 patience 1。`num_samples_per_epoch` 5,000、chunk 48,000、shift 0.5、35 epoch。選択は `valid.loss` |
| Dataset | `tf_gridnet.ms_snsd.espnet_recipe@1`。train 7,345 / valid 1,835 / test 305。このホストにスナップショットは無い |
| 更新総数 | NOT_TESTED。スモーク YAML の `max_steps: 2000000` は公式の更新数ではない。発話長が無いと chunk 数は決まらない |
| GPU | 2 |
| 学習時間 | NOT_TESTED |
| Validation | 1,835 発話を chunk した件数。全件時間は NOT_TESTED。P10-B は rank あたり 1 batch だけ |
| Checkpoint | epoch 末に残す想定。1 個の大きさは NOT_TESTED。P10-B の初期化後パラメータ数の監査は 2,552,790。今回数え直していない |
| ストレージ | 推定不能 |
| 中断 | chunk 経路の `interrupted.json` は TF-Locoformer で GPU 確認済み。このモデルのスナップショットでは未確認 |
| 公式との差 | shape file 順は未確定。余り batch の公式挙動は、公開レシピが割り切れるため未使用 |
| 開始条件 | スナップショットを置く。1 epoch の更新数を、音声を読まずに計画だけ計算する。公式幅 2 GPU で数 step の中断再開を確認する。検証接頭辞の時間から 35 epoch を見積もってから始める |

## TF-Locoformer

| 項目 | 値 |
|---|---|
| 公式設定 | DNS medium。4 rank、global 4、micro 1。chunk 64,000、shift 0.5。rank あたり 1,000 発話。150 epoch。warmup 4,000。plateau factor 0.5、patience 3。early stopping patience 10。`keep_nbest` 5 |
| Dataset | `tf_locoformer.dns.accept_recipe@1`。`cv_synthetic_small` ではない |
| Snapshot | `5469f0416329a8661df2cc91cd5bd9a149d6ea223c72e69ebdd5696bea6bdfbe` / `3343a55efb4c57d9ffef9179fb2f39c3e0186d36f587a0f4ad9f0aa14e9926f5`。train 側の計画は 14,000 更新/epoch。150 epoch で 2,100,000 更新 |
| GPU | 4。A100 40GB で peak allocated 15.3 GB |
| 更新時間 | 定常 0.377 秒 × 2,100,000 = 219.8 時間。データ待ち 0.187 秒 × 2,100,000 = 109.1 時間。合計 328.9 時間 |
| Validation | 検証発話 36,000 はすべて 480,000 サンプル。rank あたり 126,000 chunk。1 回の点推定は 11.68 秒 + 126,000 × 0.073 秒 = 2.56 時間。150 epoch で 384 時間。計測は 1 chunk と 32 chunk だけなので、検証には 1.5 を掛けて 576 時間を見ておく |
| Checkpoint | epoch 末に残すと 150 個 × 172 MiB ≈ 25 GB。書き込みは 1 回 2.2 秒の実測がある。`ave_5best` は上位 5 epoch の追加ファイル |
| 上限の合計 | 更新と待ち 329 時間に 1.3 で 428 時間。検証 576 時間。I/O は 1 時間未満。確保の目安は約 1,000 時間。patience 10 の early stop で短くなることがある |
| 中断 | 公式幅 4 GPU で `interrupted.json` から再開し、連続ランと case id、plan hash、warmup LR が一致した。強制終了は NOT_TESTED |
| 公式との差 | 平均 Checkpoint の拡張子は `.pt`。検証セット名は `cv_synthetic_small` ではない |
| 開始条件 | このスナップショットで開始できる。`max_updates` と `validation_max_updates` と `verification_max_updates` を外し、`max_epochs: 150` のままにする。4 GPU を維持する。検証が 150 回とも全件なら、学習本体より検証が長い |

## 共通の開始前チェック

- `complete.json` を改名しない。
- 既存の `data/db/snapshots/`、`resource.duckdb`、`20261107-p10b-lab`、`20261107-p10b1` を上書きしない。P10-C は別の出力ディレクトリにする。
- 学習用 WAV と波形キャッシュは作らない。
- DeepFilterNet3 をこのホストで動かすときは `DF_RFFT_LIBRARY` にローカルビルドを指定する。共有 `.so` は置き換えない。
