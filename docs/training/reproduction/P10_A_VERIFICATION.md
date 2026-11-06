# P10-A 検証（2026-11-07）

監査の結論は [P10_A_AUDIT](P10_A_AUDIT.md)。ここには計測と ID を残す。本番 DB と既存スナップショットは読み取りだけである。新しい WAV は `data/generated/`、`data/staging/`、`data/derived/` に書いていない。

## DB・Snapshot 整合性

`reproduction.duckdb` の学習用 Dataset と、ディレクトリ名が `dataset_id` のスナップショットを読み取り専用で対応させた。`manifest.json` の `snapshot_hash` はディレクトリ名とは別である。P7.1 の受理サンプルはスナップショットと `data/staging/reproduction-p71` にあり、本番 DB の `repro.datasets` には無い。

`resource.duckdb` の DNS カタログは、clean 65,348（すべて 16 kHz）、雑音 16 kHz 65,114、16 kHz 以外の雑音 188 である。P7 登録時の記録と一致する。`eval` スキーマは無く、leakage 監査は UNKNOWN のままである。

学習プロセスが DuckDB を開く経路は、この監査で足していない。新しい `execute` は `delivery: on_demand` 以外を拒否する。履歴スナップショットは `open_training_data` で読める。混合 WAV を書く `write_build` はテストが不在のまま通過した。

## Dataset

登録先が `reproduction.duckdb` のものを先に書く。status はすべて `ready`。protocol 名のうしろが protocol_id である。

| Dataset | dataset_id | snapshot_hash | 件数 | protocol | claim | selection_uses_test |
|---|---|---|---|---|---|---|
| `mp_senet.vb28.ondemand.official@1` | `36ab4590b113d0aa00884c44041eaa97d65085cee57bcc9c9e06135741fc76d7` | `265f256b938a89746dcaa62cd44daa2c6bcca3bce4cc1b1719eb2170717d33d5` | 11,572 / 824 / 824 | `mp_senet.fixed_pair.ondemand` `69dcd7c2bd50f353777cacb76a86ccf3c44d7cc133a3bc355995727b9f0269a7` | `official_reproduction` | true |
| `mp_senet.vb28.ondemand.safe@1` | `54ee0fc9b6b4675c685991ecc9f090e51a3dc8fba0da3cd21231f542c88232da` | `4528f73a317140cbe05e89d6d30e3a9d2058c28ad00002457551ceb2935d5c3a` | 10,700 / 872 / 824 | 同上 | `leakage_safe_comparison` | false |
| `mossformergan.vb28.ondemand.official@1` | `153096c312f8027bba93fa635224bff8681170841bbe40164d189f923d16aebe` | `ba0c77c94babbbc2c3e70b149554f8a922a68cf6fbe71ccf833927dc71dff555` | 11,572 / 824 / 824 | `mossformergan_se_16k.fixed_pair.ondemand` `f1b67cc1d8672693faee128dc420a15735e35c8ffc7b88208e628311267c06d5` | `official_reproduction` | true |
| `mossformergan.vb28.ondemand.safe@1` | `5748381b83c289f58485062ab62f099f48899ec98456d85a8e487bf09cb0d8b6` | `369bffdd754b096a16a59d5a3cac655a2f4ff851a15dd04c90800d03eccd7bbf` | 10,700 / 872 / 824 | 同上 | `leakage_safe_comparison` | false |
| `tf_gridnet.ms_snsd.espnet_recipe@1` | `13c6058881ef86fe7506f705b1bd5a5bc68b313181410d8fdaaee20c33ef3fc6` | `a69a477af70bc43dbcabdafbd5c97ec8a26aac4debd51158079545de1e4f0757` | 7,345 / 1,835 / 305 | `ms_snsd.tfgridnet.ondemand` `9f5ffa7c14a43b23b5db0a0c931c8a80e91cb666518aa10c826813c90abf7e97` | `espnet_public_se_recipe_reproduction` | false |
| `tf_gridnet.ms_snsd.ondemand@1` | `d8298f35a3f4da1fcf5a1966163716eb15460602f4767989f8ba86fbae880f92` | `78152930aa7b3ba5f0cfa061cc6d4afaa9c82cef016102ad9d2a9532c2dbe7a5` | 20 / 5 / 5 | 同上 | `official_recipe_scaled_down` | false |
| `tf_gridnet.ms_snsd.smoke@1` | `6a6335376cac2cde7659651438629a1cf444f3bff5cf8816684034b36a434f09` | `6e91b3913ad4b11adec9a0c8a4904c9471262e149b221ecbb40a1988f4713bf6` | 20 / 5 / 5 | `ms_snsd.tfgridnet.seeded` `4f2f821a44129d1a95f7a020064917feb4e7b48f7a0c21785b3925c8335de5c2` | `official_recipe_scaled_down` | false |
| `tf_locoformer.dns.recipe@1` | `63501932b52ce7b6736ae59b1acd752f61dc4656afa15b4fb29ac94174df35f8` | `c05cebd2fbea458c369c0570964681d8e69063fc21c27f4a61ca6b9343885ed5` | 324,000 / 36,000 / 150 | `dns.tflocoformer.ondemand` `94f211d2bccb7f484efd5367b15ba60f7303d05b70b8c3d1564e4bb57a774edc` | `espnet_public_se_recipe_reproduction` | false |
| `tf_locoformer.dns.smoke@1` | `2d52d457e0380ad2116fac5093b471f7d9e3b550b7afb4cc72422467ae799447` | `9435ef7852eec5c0f2a8c58c7aeb9fa9df34920a6b0b7bd4f9c5eee2b1466d31` | 7 / 1 / 150 | 同上 | `official_recipe_scaled_down` | false |

履歴の 16 kHz VoiceBank Dataset も本番 DB に残る。新しい学習入口ではない。

| Dataset | dataset_id | snapshot_hash | protocol |
|---|---|---|---|
| `mp_senet.vb28.official@1` | `1e78f60c1f8333930bf2b252ccd457b9e83e8739354fadeb3c906136f7845f5b` | `ebbe60d377dd93c8f43e641b25bb1ed57d7090f7a64f3ed192066b877953d3bb` | `mp_senet.fixed_pair` `b13c9ea2cfe6f666e3d680d76c70ee7cc8804604721cb7936362c9c9a7065c06` |
| `mp_senet.vb28.safe@1` | `4cb14ffcca40205fe0adf1672aeb78d1f826a772f2e0e889c788cfa52cea033b` | `ff6e05243a2ad20b3e772e907f049072112a921f986a69a75a47c5281f366087` | 同上 |
| `mossformergan.vb28.official@1` | `2481d52346c21e69b78662920f94197eb4e8473af2f96126e96b1ae765a3302c` | `23af02cefb602107dee5affc324158865e19189869178d4f888eeca71dde9263` | `mossformergan_se_16k.fixed_pair` `9c85466d9af2f904dfca1149384b65b141c7a8ae8ba5fe8fe358d2c6f5fab319` |
| `mossformergan.vb28.safe@1` | `431030cb2b7ed9558c9c54519e3775443079a991d7f56da435cc1080a9d249ef` | `4a0cec401a5a2bbba6a22eac4207883c4dac567c7ca275d4d58092b377f8def6` | 同上 |

本番 DB に無い受理サンプル:

| Dataset | dataset_id | snapshot_hash | 件数 | 置き場所 |
|---|---|---|---|---|
| `tf_locoformer.dns.accept_smoke@1` | `f75b3e7a4dbab8a68a6594487b3d605e949d96ef85fb8667b34417ea12101a39` | `db52950f22d0b5f9f947a341d1afdf1c2b38f30da375d3fe505e384f5816ec8a` | 230 / 26 / 150 | snapshot と `data/staging/reproduction-p71` |

親 protocol `dns.tflocoformer.formula` は `cb81f473d53ef80ee535fd7b06ec3ca9b57f9ab5e0146e8f89635d79e5e592e1`。子 protocol は `cadda632ebd08a9066e5b7a81851920762cc2a5fa5bddc21f8488bf1a1630116`。公式規模の受理 Dataset は ID を発行していない。

P8 と P9 は別ファイルのままである。

| DB | Dataset | status | claim | dataset_id |
|---|---|---|---|---|
| `reproduction-p8.duckdb` | `dfn3.online.official@1` | blocked | `official_reproduction` | `be8e03d6f583c14c5667e6a86a09f6d3cae26b510c5a1aa872bed106ee8f8f60` |
| 同上 | `dfn3.online.smoke@1` | ready | `protocol_conformant_substitute_materials` | `283030adc4a81a90d833338fbd2d9e8677ec387b0cd2b7f7d222efb3aa0340e8` |
| 同上 | `dfn3.online.substitute@1` | ready | 同上 | `ee6223405c19bef4e8f39c9609b830273c422d7ed32b17fbaad0b0554ce1328b` |
| `reproduction-p9.duckdb` | `semambapp.gsr.official@1` | blocked | `official_reproduction` | `63a0e6785851996743c1ed040ddfbabb811c00098f821d09e4c57ea36a1f995f` |
| 同上 | `semambapp.gsr.smoke@1` | ready | `protocol_conformant_substitute_materials` | `d3bc5dbea5b7a19c8495a0d2ce5d7d53267c96d6f82dc049c193fd14a04980eb` |
| 同上 | `semambapp.gsr.substitute@1` | ready | 同上 | `4acf30da286a36cb8c6edcdf1091456c979fd839d7898ecc7528467e4b04d76e` |

学習に使う公式規模の入口は次の 4 つに分かれる。MP-SENet は `mp_senet.vb28.ondemand.official@1`（safe は別 Dataset）。MossFormerGAN は `mossformergan.vb28.ondemand.official@1`。TF-GridNet は `tf_gridnet.ms_snsd.espnet_recipe@1`。TF-Locoformer の受理規則による公式規模 Dataset は未登録で、本番にある `tf_locoformer.dns.recipe@1` は長さ計画である。

## オンデマンド生成

対話プロセスで、既存スナップショットから 1 バッチまたは 1 クリップを再構築した。所要 33.3 秒。WAV は書いていない。

| 対象 | 結果 |
|---|---|
| MS-SNSD 実データ、clean 2 ファイル | 長さ 166,505。公式 `audioread` から `snr_mixer` までの clean / noise / noisy は最大絶対誤差 0。実測 SNR 10.0。再実行は一致 |
| DNS 実データ 30 秒、2 件 | 長さ 480,000。activity 0.6366666666666667 と 0.64。SNR 1 と 13、レベル −35 と −23。SNR 誤差 −2.349e-5 dB と −7.341e-6 dB。元素材 clip は無し。外側のやり直しは 1 と 0。再実行は一致 |
| `mp_senet.vb28.ondemand.official@1` | 件数 11,572 / 824 / 824。`delivery` は `on_demand`。epoch 0 の batch は `(4, 32000)`。再実行と再開読み戻しは一致 |
| `mossformergan.vb28.ondemand.official@1` | world size 2 で rank ごとの形状は `(2, 32000)`。rank 間の窓は違う。再実行は一致 |
| `tf_gridnet.ms_snsd.espnet_recipe@1` | train 7,345、valid 1,835。最初のクリップ長 239,868。再実行は一致 |
| `tf_locoformer.dns.smoke@1` | train 7、valid 1。文書に受理レンダラは無い |
| `tf_gridnet.ms_snsd.smoke@1` | 新しい学習は拒否 |

DNS の受理計画を、ピークが 0.99 以下の実ファイル（clean 2 本は 496,000 フレーム、noise 2 本は 160,000 フレーム）で 4 件測った。ピーク確認の直後なので、ディスクの初回読み込みは含まない。1 回目 0.132 秒、2 回目 0.125 秒、1 件あたり約 0.033 秒。外側の試行は 0、長さは 480,000。同じ条件の worker 数 1 と 2 は、合成データ 6 件で文書と外側試行数が一致した。16 kHz の区間読みは、同じファイルの全読みスライスと一致した。

`is_test_set = true` で 1,000 サンプルの clean を目標 400 に渡すと、区間長は 1,000 である。公式 `build_audio` を子プロセスで実行し、出力長は 1,000、使用ファイルは 1、clip リストは空だった。`is_test_set = false` の区間長は 400 である。登録済みの学習定義はこの旗を false にする。

## 公式コードとの数値比較

ラベルの定義は監査文書の表と同じである。この節は計測の出どころを固定する。

- MS-SNSD: `tests/reproduction/test_p6_ms_snsd.py` の連結テストが、合成 2 ファイルで公式 `audiolib` と `np.array_equal`。上の実データクリップは最大絶対誤差 0。
- DNS の式: `tests/reproduction/test_p7_dns.py` の `test_formula_matches_official_audiolib` が、公式 `audiolib.py` と activity、clip、`snr_mixer` の浮動小数を一致させた。activity の書き換え後にもう一度通し、0.338 秒の実行に含まれて成功した。
- DNS の長い test-set clean: 上の子プロセス比較で長さが一致。
- VoiceBank の公式変換器、DNS の librosa 版、`cv_synthetic_small` は測っていない。UNKNOWN のままである。

## Sampler・DDP・Trainer・再開

`tests/reproduction` の順序、計画、来歴、checkpoint テストが、固定 shuffle、rank 固定、chunk の world size、`data_state`、別 Dataset の再開拒否、同じ run の保持を通した。実スナップショット 4 件の来歴往復も通った。

MP-SENet の実バッチは上の表のとおり、再開読み戻しが一致した。MossFormerGAN の 2 rank は別窓である。TF-GridNet と DNS smoke は chunk 計画の再実行が一致した。P7.1 の小さい CPU Trainer は、同じ chunk の再開テストとしてスイートに含まれ、成功した。PESQ を計算しない検証ステップは `pesq=na` のままで、0 にはしていない。

shape ファイルが無いため、公式 `collect_stats` の発話順は UNKNOWN である。world size は PCM の中身を変えない。

## 回帰テスト

環境は `.venvs/se`。GPU ジョブは投入していない。

| 実行 | 結果 | 時間 |
|---|---|---|
| `unittest discover -s tests/reproduction -v`（test-set の長さ修正のあと） | 108 件 OK。FAIL / ERROR / SKIP は 0。`test_test_set_clean_keeps_a_file_longer_than_the_target` を含む | 510.719 秒 |
| 同じスイートを、長さ修正の前に実行 | 107 件 OK。FAIL / ERROR / SKIP は 0 | 617.251 秒 |
| 長さ修正の直後の 3 件（長い clean、worker 独立性、DNS 式） | 3 件 OK | 0.640 秒 |
| `tests.training` の `test_p5_chunk`、`test_chunk_plan`、`test_protocol`、`test_resample`、`test_legacy_adapter`、`test_tf_locoformer_protocol`、`test_ave_nbest`、`test_dataset` | 59 PASS、1 SKIP、FAIL / ERROR は 0。SKIP は `rebuild database is absent` | 14.985 秒 |
| `tests/training` の残り、`test_chunk_official`、libdf の `DF()` | NOT_TESTED | — |

108 件には P8 と P9 が含まれる。公式 Dataset が blocked のままであること、代替素材の再構築、1 step の有限性、checkpoint 再開を通した。libdf の失敗はこの実行には出ていない。出ていないことを、libdf 全体の PASS にはしない。

## 計算資源

対話プロセスだけを使った。qsub は使っていない。GPU は割り当てていない。

| 作業 | CPU 時間の目安 | 生成した学習 WAV |
|---|---|---|
| 再現スイート 108 件 | 511 秒 | 0 |
| 学習データの回帰 60 件 | 15 秒 | 0 |
| 実スナップショットの再構築 | 33 秒 | 0 |
| DNS 受理 4 件 × 2 回 | 0.132 秒と 0.125 秒 | 0 |

メモリのピークは、この監査の短時間計測では別に記録していない。P7.1 が 256 件の計画で記録した約 726 MB は、そのときの記録であり、今回の 4 件のピークではない。

360,000 件へ外挿すると、1 worker で約 0.033 秒 × 360,000 は約 3.3 時間である。これは直前に読んだファイルの速度で、冷たい読み込みと棄却のやり直しを含まない。P7 のハッシュ 357 秒と DuckDB 1,352 秒も別途かかる。合計は 30 分を超えるので、承認前に登録しない。

## 本番 Dataset の準備

| モデル | 公式規模の学習入口 | 状態 |
|---|---|---|
| MP-SENet | `mp_senet.vb28.ondemand.official@1` | ready。公式 16 kHz 変換器は UNKNOWN |
| MossFormerGAN | `mossformergan.vb28.ondemand.official@1` | ready。変換器は同じ UNKNOWN |
| TF-GridNet | `tf_gridnet.ms_snsd.espnet_recipe@1` | ready。連結前の −25 dBFS を公式関数と照合済み |
| TF-Locoformer | 受理規則の 360,000 件 | 未登録。本番の `tf_locoformer.dns.recipe@1` は長さ計画 |
| DeepFilterNet3 | `dfn3.online.official@1` | blocked。対象外 |
| SEMamba++ | `semambapp.gsr.official@1` | blocked。対象外 |

## P10-B に進む条件

MP-SENet、MossFormerGAN、TF-GridNet は、上のオンデマンド Dataset を一意に指定すれば、短い GPU 確認に進める。結果には、VoiceBank 変換器が UNKNOWN であること、MS-SNSD のファイル選択がシード付きであることを残す。公式 epoch までの学習は、この検証だけでは始めない。

TF-Locoformer の公式規模学習は、受理規則の Dataset を新しい ID で登録してからである。`tf_locoformer.dns.recipe@1` を公式の受理データとして学習しない。`cv_synthetic_small` が UNKNOWN の間、36,000 件の検証分割をその名前で扱わない。登録ジョブは 30 分を超える見込みなので、開始前の明示的な承認が要る。承認文に含める条件は、新しい Dataset 名と版、新しい snapshot、staging の保持、混合 WAV を書かないこと、既存の recipe Dataset を変えないこと、spawn と連続 `gen_index`、worker 数で文書が変わらないことの再確認である。
