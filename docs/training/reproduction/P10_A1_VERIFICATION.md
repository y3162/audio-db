# P10-A.1 検証（2026-11-07）

受理規則を適用した DNS 360,000 件を `tf_locoformer.dns.accept_recipe@1` として `reproduction.duckdb` に追記し、スナップショットを書いた。混合 WAV と波形キャッシュは作っていない。GPU 学習はしていない。P7 の `tf_locoformer.dns.recipe@1` と `tf_locoformer.dns.smoke@1` は長さ計画のままである。`tf_locoformer.dns.accept_smoke@1` は本番 DB に入れていない。

`cv_synthetic_small` は UNKNOWN のままである。36,000 件の validation は sorted uttid の末尾 1/10 であり、その名前では呼ばない。公式の共有カウンタとのビット一致、公式 librosa の版、公式の無制限外側ループとの一致は主張しない。

## 実行

全量計画と登録は `qrsh -g tga-shinoda -p -5 -l cpu_40=1 -l h_rt=10:00:00` で計算ノード `r5n2` に載せた。`qsub` は使っていない。ノード上の表示は 80 コア、メモリ 755 GB。計画の worker は 32。開始 2026-11-07 08:51:46、終了 09:31:31（+09:00）。

| 処理 | 時間 |
|---|---:|
| 全量計画（shard を新規に計算） | 548.96 秒（CPU 12,769 秒、読み込み 86,162,055,168 バイト） |
| 登録時の計画再読込（欠落 shard 0） | 14.70 秒 |
| DuckDB 登録を含む register | 1,351.09 秒 |
| うちスナップショット | 132.75 秒 |
| メタデータ監査 | 49.62 秒 |
| 波形監査 | 57.17 秒 |

事前のベンチマークは別ノード `r22n4`（`os.cpu_count()` は 384）。100 件と 1,000 件を worker 1 / 2 / 4 で計画し、文書ダイジェストは一致した。10,000 件の単独計測はしていない。

登録後の Trainer 接続と回帰は、既に確保されていた対話ジョブ `116458`（`r3n11`、48 コア）で実行した。検証プロセスの最大 RSS は 30,513,784 KB。

## Dataset

| 項目 | 値 |
|---|---|
| 名前 | `tf_locoformer.dns.accept_recipe@1` |
| dataset_id | `5469f0416329a8661df2cc91cd5bd9a149d6ea223c72e69ebdd5696bea6bdfbe` |
| protocol | `dns.tflocoformer.ondemand.accept` `cadda632ebd08a9066e5b7a81851920762cc2a5fa5bddc21f8488bf1a1630116` |
| 親 protocol | `dns.tflocoformer.formula` `cb81f473d53ef80ee535fd7b06ec3ca9b57f9ab5e0146e8f89635d79e5e592e1` |
| snapshot_hash | `3343a55efb4c57d9ffef9179fb2f39c3e0186d36f587a0f4ad9f0aa14e9926f5` |
| スナップショット | 731,799,561 バイト。`delivery: on_demand`、`cache: none` |
| 件数 | train 324,000 / valid 36,000 / test 150 |
| 長さ | train と valid はすべて 480,000。test は既存の `no_reverb` 150 ペア |
| 素材 | clean 65,348。雑音 65,302（16 kHz 65,114、44.1 kHz 94、48 kHz 88、96 kHz 6） |
| material set | speech `60ff4422036c6ba21939ecfef4a5a222dea74a6b2e6bbc2a0bef549dc05c7eda`、noise `d85753d24f326b8c11caa6fe7a8da9c00880fbb4ffb311476c8ee4cc6134f278` |
| test case set | `52e7fea8f0ddf391c6ab06a15ef1ce47294fe593384f445c55f3f55876de44cd`（recipe と smoke と同じ） |
| train / valid の case set | それぞれ別。P7 の case set とも違う |
| claim | `espnet_public_se_recipe_reproduction` |
| `selection_uses_test` | false |
| 生成 WAV | 0。`generated.json` は空 |
| seed | 17 |
| renderer | `dns_pcm16_accept_v2` |
| retry | `deterministic_case_stream_v1` |
| recipe build | `0c34775cb2067416b719600bc626e9762b4b59dea04f5816b527942f89076e91` |

train / valid / test の case_id 交差は 0。recipe と smoke も 0。`reproduction.duckdb` は 953,954,304 バイト（mtime 2026-11-07 09:29:08）。`resource.duckdb` の mtime は 2026-11-06 02:48:34 のまま。recipe のスナップショット `63501932…` の mtime は 2026-11-06 17:51、smoke は 17:52、accept_smoke は 23:16 で、今回の 09:30 より前である。recipe の snapshot_hash は `c05cebd2fbea458c369c0570964681d8e69063fc21c27f4a61ca6b9343885ed5`、smoke は `9435ef7852eec5c0f2a8c58c7aeb9fa9df34920a6b0b7bd4f9c5eee2b1466d31` で、[P10_A_VERIFICATION](P10_A_VERIFICATION.md) の記録と一致する。

`dfn3.online.official@1` と `semambapp.gsr.official@1` はそれぞれの DB で `blocked`、binding 0 のままである。本番 DB には入っていない。

## 受理

360,000 件すべてで clean activity の最小値は 0.6016666666666667。0.6 以下は無い。失敗件数は 0。外側の試行が 50 回に達した案件は無い。最大再試行は 8。

| 再試行 | 件数 |
|---:|---:|
| 0 | 266,142 |
| 1 | 69,406 |
| 2 | 18,179 |
| 3 | 4,723 |
| 4 | 1,225 |
| 5 | 252 |
| 6 | 57 |
| 7 | 13 |
| 8 | 3 |

受理までの探索で clean clip の棄却が 1 回以上あった案件は 1,048。noise clip の棄却が 1 回以上あった案件は 160,492。受理した文書のメタデータ監査では、標本 1,024 件の元素材 clip と混合後 clip は 0。波形監査 533 件の元素材 clip 不一致は 0、混合後 clip は 0、PCM の再構築誤差は 0。非 16 kHz 雑音を含む案件は 3,703（断片 3,709）。16,000 サンプル未満の断片は 11,923。clean の連結は 0。noise の連結は 360,000。

SNR の件数は −5…14、レベルの件数は −35…−16 にあり、どちらも空の整数は無い。uttid は 360,000 で重複は無い。

`leak_check` は WARN。train/valid と train/test の資産 ID と SHA-256 の交差は 0。話者 ID が無い素材があるため話者成分だけ WARN である。P7 と同じ扱いであり、FAIL ではない。

## 公式との差

| 項目 | 状態 |
|---|---|
| clean activity `> 0.6` | 実装し、全件の記録と抽出波形で確認 |
| `abs(x) > 0.99` | 公式 `is_clipped`。抽出 8 件の activity 誤差 0、mixer 誤差 0、clip 不一致 0 |
| 混合後の clean / noise / noisy | 不適合なら棄却。抽出監査の混合後 clip は 0 |
| 共有カウンタ | 再現しない。案件ごとの `Generator`。ビット一致は主張しない |
| 外側ループ | 公式は無制限。audio-db は 50 で停止する。今回の到達は 0 |
| librosa | 公式の版は UNKNOWN。描画は librosa 1.0.0 の `soxr_hq`。16 kHz 以外だけ |
| `cv_synthetic_small` | UNKNOWN。validation をその名前で呼ばない |
| 公式公開 checkpoint の学習集合 | 一致したとは扱わない |

非 16 kHz の 2 秒区間で、`soxr_hq` と `scipy.signal.resample` の最大絶対差は 44.1 kHz で 0.000779、48 kHz で 0.004623、96 kHz で 0.004029。同じ区間の `kaiser_best`（`.venvs/asr` の `resampy==0.4.3`）と `soxr_hq` の最大絶対差は 0.005737、0.021388、0.017649。SE 環境のプロセス内では Numba が NumPy 2.5 を拒否するため、`kaiser_best` は別プロセスで測った。公式の `res_type` は UNKNOWN のままである。

## 再現性

100 件と 1,000 件は worker 1 / 2 / 4 で文書ダイジェストが一致した。100 件の wall は 11.18 / 1.78 / 1.49 秒。1,000 件は 92.87 / 11.05 / 6.16 秒。1 worker の内訳は decode が支配的で、100 件は decode 6.36 秒、I/O 3.17 秒、activity 0.24 秒、clip 0.13 秒、resample 0。

全量は worker 32 で 1 回計画した。登録は完成した shard を再読し、同じヒストグラム（失敗 0、最大再試行 8、非 16 kHz 3,703）になった。360,000 件を別の worker 数で最初から計画し直す測定はしていない。案件ごとのストリームと、100 / 1,000 件の一致、単体テストの再開・欠落・重複検査が根拠である。

波形監査は 533 件を再構築し、PCM 誤差 0、activity 最小 0.6016666666666667、SNR 誤差の最大 0.004575 dB。Train と valid の先頭チャンクを 2 回再構築し、配列は一致した。

スナップショットを開く経路はファイル内容の論理ハッシュと `snapshot_hash` を検査する。Train 324,000 と valid 36,000 を開いた。

## 来歴

Git HEAD は `2593e29d71a6bb231ec86bc5b74136cecdfc1c9f`。作業ツリーは dirty なので、HEAD だけでは生成器を特定しない。`generator_code_sha256` は `6dd072d518ed8242fa8a8247ebe71c184a3aa4daa0825bc3b5ca4f8e3417a145`。`porcelain_sha256` は `ebde4f7a6e45848d946e07f13ca473ba1d712519d3dd2bd9f436e17c24c6e2a5`。どちらも `manifest.json` の provenance にある。

パッケージは Python 3.12.13、NumPy 2.5.4、SciPy 1.18.1、librosa 1.0.0、soxr 1.1.0、soundfile 0.14.0。コマンドは `python -m src.reproduction.cli dns --config configs/data/builds/dns_tflocoformer_accept_recipe.yaml --step plan` のあと `--step register`。

## Trainer

`configs/training/tf_locoformer_dns_accept_recipe.yaml` は Dataset を `tf_locoformer.dns.accept_recipe@1` と明示する。CPU、公式幅の TF-Locoformer、global batch 1、chunk 64,000、`num_samples_per_epoch` 1、`max_steps` 2。checkpoint は `data/results/se-repro/p10a1-tflocoformer-accept-recipe-smoke`。

| 確認 | 結果 |
|---|---|
| train / valid を開く | 324,000 / 36,000。長さ 480,000。renderer は `dns_pcm16_accept_v2` |
| train / valid の 64,000 サンプル | 有限。再構築は一致。ピークは 1 未満 |
| world 1 / 2 / 4 | `num_samples=1000`、global batch 4、seed 17、epoch 1。更新数 3,500 / 7,000 / 14,000。検査したチャンクは 14,000 / 28,000 / 56,000 で、境界違反は 0。再計画の hash は一致し、world 間の hash は違う |
| 30 秒・chunk 64,000・shift 0.5 | 開始は 0, 32000, …, 416000。1 発話は 14 チャンクで 480,000 を覆う |
| `data_state` | `espnet_chunk_v1`、epoch 0、world 1。step 1 の cursor は 1、再開後の step 2 は 2。plan hash は上の 1 発話計画と一致 |
| `data_provenance.json` | dataset_id、snapshot_hash、protocol、`dns_pcm16_accept_v2`、`selection_uses_test: false` |
| 1 step のあと同一設定で再開 | step 2 まで進んだ。損失は有限（step 1 の generator 24849.35、step 2 は 19838.98） |
| `virtual_world_size: 2` で再開 | 拒否。global batch 1 は world 2 で割り切れない |
| recipe Dataset で accept の checkpoint を再開 | 拒否。`checkpoint dataset identity does not match this run` |
| P7 recipe / smoke | 開ける。実行時 renderer は `dns_pcm16_v1`。identity JSON は、規則に renderer キーが無い P7 ではそのフィールドを持たない。受理 renderer ではない。1 チャンクの再構築は有限で、再実行は一致 |

検証 epoch は 14 更新で終わらないため、この 2 step では validation epoch を走らせていない。valid のチャンクは別途 1 件再構築した。

## 回帰

2026-11-07 09:39–09:50、`r3n11`、`.venvs/se`。失敗は 0。

| テスト | 結果 |
|---|---|
| `tests/reproduction/test_p1.py` | PASS 13 |
| `test_p2.py` | PASS 9 |
| `test_p3.py` | PASS 19 |
| `test_p4.py` | PASS 10 |
| `test_p6_ondemand.py` | PASS 3 |
| `test_p6_pipeline.py` | PASS 3 |
| `test_p6_ms_snsd.py` | PASS 5 |
| `test_p7_dns.py` | PASS 6 |
| `test_p71_dns.py` | PASS 8 |
| `test_p10a1_dns.py` | PASS 6 |
| `test_p8_dfn3.py` | PASS 14 |
| `test_p9_semamba.py` | PASS 18 |
| `tests/training/test_p5_chunk.py` | PASS 15 |
| `test_chunk_plan.py` | PASS 2 |
| `test_chunk_official.py`（`.venvs/se`） | SKIP 4。`typeguard` が無い |
| `test_chunk_official.py`（`.venvs/official`） | PASS 4 |
| `test_protocol.py` | PASS 4 |
| `test_resample.py` | PASS 5 |
| `test_legacy_adapter.py` | PASS 3 |
| `test_tf_locoformer_protocol.py` | PASS 6 |
| `test_ave_nbest.py` | PASS 7 |
| `test_dataset.py` | PASS 17、SKIP 1（rebuild database が無い） |
| `test_training_sets.py` | PASS 2 |
| `test_tf_locoformer.py` | PASS 1 |
| `tests/evaluation/test_provenance.py` | PASS 15 |

MP-SENet、MossFormerGAN、TF-GridNet の学習入口を変えるテストは通った。DeepFilterNet3 と SEMamba++ の公式 Dataset は blocked のままである。

## 残る UNKNOWN

- `cv_synthetic_small` の中身、seed、件数。
- 公式 DNS 生成が使った librosa の版と `res_type`。
- 公式の共有カウンタがこの 360,000 件のどのファイルを選んだか。
- 公式外側ループは無制限である。今回は 50 回に達していない。
- `resource.duckdb` に `eval` が無いため、評価セットとの話者交差は測っていない。話者 ID の欠落による WARN は残る。
