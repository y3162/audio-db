# P7 検証（2026-11-06）

TF-Locoformer の DNS-Challenge（`interspeech2020/master` @ `70f19285c36cca4df2338f9248775ddc50980c6b`）を、混合 WAV を書かずに条件として登録し、学習時に PCM_16 を再構築する。ESPnet は `90eed8e53498e7af682bc6ff39d9067ae440d6a4`、モデルは TF-Locoformer Medium。論文の学習ランや公開 checkpoint との一致は主張しない。`cv_synthetic_small` は作成スクリプトも件数が無く、UNKNOWN のままである。

現行の段階名は S0（条件の計画）、S1（メモリ上の式比較）、S2（3000 h 相当の条件登録）、S3（小さい Dataset の Trainer）である。旧設計の S0〜S4（約 1.04 TB の事前生成 WAV）は履歴であり、この実装は使わない。

## 総合

判定は CONDITIONAL GO。条件の登録、スナップショット、式の照合、chunk 計画、CPU の実モデルスモーク、再開、合成 checkpoint の `ave_5best` は通った。公式生成は activity と clip で標本を捨ててやり直す。その再試行は計画に入れておらず、256 件の監査では clean activity が閾値以下のものが 60 件、混合後に clip したものが 13 件あった。公式 WAV とのビット一致は主張しない。このファイルは P7 の Dataset の記録である。受理と非 16 kHz のリサンプルは別 Protocol で、[P7_1_VERIFICATION](P7_1_VERIFICATION.md) を見る。

## 素材

`resource.duckdb` は読み取り専用。更新時刻は登録の前（2026-11-06 02:48）のままである。カタログの `DNS2020/datasets/` は clean 65,348（すべて 16 kHz、モノラル、496,000 フレーム）、雑音 16 kHz 65,114、16 kHz 以外の雑音 188（44.1 kHz が 94、48 kHz が 88、96 kHz が 6。いずれも約 10 秒）、`no_reverb` の clean 150 と noisy 150。`with_reverb`、real recordings、blind test は学習 Dataset に入れていない。RIR は学習生成に使わない。

16 kHz 以外の雑音は librosa でリサンプルせず除外した。公開ダンプの librosa 版は不明である。clean と test が 16 kHz でない場合は登録を止める。除外は `algorithm_change` として記録した。

clean は 65,348 × 31 秒で約 562.7 時間。混合条件は 360,000 × 30 秒で 3,000 時間。WAV にすると 1 ストリーム 345.6 GB、3 ストリームで約 1.04 TB だが、書いていない。

## 登録

コマンドは `.venvs/se/bin/python -m src.reproduction.cli dns --config configs/data/builds/dns_tflocoformer_ondemand.yaml --step register`。対話プロセスで、qsub は使っていない。所要 2,154 秒。

| 段階 | 秒 |
|---|---|
| カタログ列挙 | 124.3 |
| SHA-256（130,762 ファイル） | 357.2 |
| 計画（recipe 360,000 と smoke 8） | 15.4 |
| 監査 | 64.0 |
| DuckDB 登録 | 1,351.7 |
| スナップショット 2 件 | 166.9 |

`reproduction.duckdb` は追記後 471,609,344 バイト。recipe スナップショットは 585 MB、読み込み 43.6 秒、最大 RSS 約 4.1 GB。smoke スナップショットは 57 MB。全量の `plan.jsonl` は 10,000 件を超えるので省略し、再構築元は staging の case parquet である。

| Dataset | dataset_id | snapshot_hash | 件数 |
|---|---|---|---|
| `tf_locoformer.dns.recipe@1` | `63501932b52ce7b6736ae59b1acd752f61dc4656afa15b4fb29ac94174df35f8` | `c05cebd2fbea458c369c0570964681d8e69063fc21c27f4a61ca6b9343885ed5` | train 324,000 / valid 36,000 / test 150 |
| `tf_locoformer.dns.smoke@1` | `2d52d457e0380ad2116fac5093b471f7d9e3b550b7afb4cc72422467ae799447` | `9435ef7852eec5c0f2a8c58c7aeb9fa9df34920a6b0b7bd4f9c5eee2b1466d31` | train 7 / valid 1 / test 150（共有） |

両 Dataset は同じ素材集合 `dns.tflocoformer.speech` / `dns.tflocoformer.noise` と、同じ test case 集合を使う。`selection_uses_test` はどちらも false。`leak_check` は WARN。clean の話者はファイル名の `reader_(\d+)` から取り、雑音に話者 ID は無い。valid は素材集合を束縛しないので、雑音の共有は資産 ID の重複としては見えない。

SNR の度数は −5…14 の 20 値だけ（17,776〜18,185。期待値は 18,000）。レベルは −35…−16 dBFS の 20 値だけ（17,720〜18,185）。15 と −15 は出ていない。上端は公式の `randint` と同じ排他である。

recipe の監査は 256 件をメモリで再構築した。clean activity が 0.6 以下のものが 60 件、素材片の clip が 128、混合後の clip が 13、測定 SNR の最大誤差は 0.0044 dB。状態は PASS、`wav_files` は 0。smoke 8 件は全件で、activity 棄却 3、clip 0、最大 SNR 誤差 8.6e-5 dB。レンダラは再試行しない。

同じ `gen_index` の切り出し、SNR、レベル、uttid は smoke と recipe で一致する。`build_id` が文書に入るので `case_id` は違う。9/10 分割は件数で計算し直すので、smoke の train は recipe の train の接頭辞ではない。

## 公式の式

`tests/reproduction/test_p7_dns.py` は、DNS の `audiolib.py` をファイルパスから読み、`snr_mixer`、`activitydetector`、`is_clipped`、PCM_16 の `audiowrite` と比べた。MS-SNSD の `audiolib` が先に import されていても、別モジュール名で読む。短信号の測定 SNR は名目 −3 dB から 0.2 dB 以内だった。実行時の学習経路はこの公式ファイルを import しない。

公式のファイル割当は `random` と NumPy と multiprocessing の共有カウンタで、seed を固定しても公式ダンプとはビット一致しない。audio-db は整列した名前を `Random(seed)` で 1 回流し、案件ごとに `PCG64(mix_seed("dns", seed, gen_index))` の結果を文書へ保存する。

## Chunk

recipe の train 324,000 と valid 36,000 は、すべて 480,000 サンプルである。音声は読んでいない。公式設定（seed 17、元発話 1,000 / rank、chunk 64,000、shift 0.5、global batch 4、`num_cache_chunks` 1024）の epoch 1 は次のとおり。端数バッチと端数チャンクは 0。幅は 64,000。開始位置は 32,000 の倍数で、30 秒ではオフセット 0 である。hash は再計算で一致し、epoch 2 では変わる。

| world | updates / rank | per-rank batch | per-rank cache | plan hash |
|---|---|---|---|---|
| 1 | 3,500 | 4 | 1024 | `3b85cc3b12809c6d450416f0a64488823fac534b1c7e96b6b41f096f241de91f` |
| 2 | 7,000 | 2 | 512 | `5776d506814a328a8e4053ded8b964289c1faeccf14828ca57d92393832fa586` |
| 4 | 14,000 | 1 | 256 | `33f954dbcf3b43bff65870aaae4dcd2da4e385286c99b3f156f1a52dab78c2ce` |

world 4 の 14,000 は公式の rank あたり更新数である。これは 1 プロセスの計画であり、勾配の all-reduce と数値が一致するとは言っていない。`num_cache_chunks` 16 でも更新数は同じで、cache 内の並べ替えが変わるので hash は違う。smoke の学習 YAML は world 1、cache 16、元発話 1 であり、この公式計画ではない。valid が 1 件しか無いので world 4 では計画できない。

shape の順は `dns_data_prep.sh` の sorted uttid である。`collect_stats` の順は確認していない。

## Trainer（S3）

設定は `configs/training/tf_locoformer_dns_ondemand.yaml`。実モデルは variant `dns` の Medium。CPU。GPU は使っていない。VoiceBank の研究 YAML とは別ファイルである。

14 step で epoch 1 が終わり、validation は 14 チャンク（valid 1 件 × 14）。`data/results/se-repro/p7-tflocoformer-ondemand/step_00000014.pt` の `data_state` は epoch 1、`update_in_epoch` 0、plan hash `c02dfdb828c82dd17ac6794d23957df9933c19812d81b624d8c49426fa1db00d`。`data_provenance.json` がある。`valid.loss.best.pt` と `1epoch.pt` はある。`valid.loss.ave_5best.pt` は無い（完了 epoch が 1 件）。

同じディレクトリには `complete.json` があるため、その場での再開は拒否された。再開は `data/results/se-repro/p7-tflocoformer-ondemand-resume` へ、`max_steps` だけ 15 にして行った。chunk iterator は同じである。step 15 は epoch 2 の別発話で、plan hash は再開前と同じ、`update_in_epoch` は 1。所要約 52 秒。

smoke の先頭 train 案件を 2 回再構築すると PCM_16 は一致し、長さは 480,000。先頭 64,000 サンプルはチャンク切り出しと一致した。ディスクには書いていない。

## `valid.loss.ave_5best`

`.venvs/official` からピン留め ESPnet の `average_nbest_models` を呼んだ。損失 5, 1, 4, 2, 3, 9 の 6 epoch では、上位 5 の浮動小数の平均が 3、整数の累積が 15 で、`average_checkpoint_files` と一致した。履歴が 2 epoch のとき、公式は `valid.loss.ave_5best.pth` を書かず `ave_1best` のリンクを書く。audio-db は 5 件未満では平均ファイルも `ave_1best` も書かない。`.venvs/se` には typeguard が無く、同じ import は SKIP になる。手計算の単体テストは `.venvs/se` で通っている。

## 回帰

`unittest discover tests/reproduction` は 67 件、92.9 秒、OK。学習側の対象テスト 37 件は 15.0 秒で、失敗は `configs/training/tf_locoformer_voicebank.yaml` が無い 1 件だけである。このファイルは P7 の前からリポジトリに無い。スキップが 1 件ある。`tests/training` 全体はこのセッションでは走らせていない。
