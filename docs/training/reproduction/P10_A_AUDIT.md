# P10-A 監査（2026-11-07）

ブランチは `feature/reproduction`、開始時の HEAD は `6f600190e7aafc61465ca76661f2a6b1f3cd722a`。この監査の差分は未コミットである。`reproduction.duckdb`、`resource.duckdb`、既存スナップショット、コーパス、`data/derived/`、`data/checkpoints/`、`data/results/`、`data/staging/` は書き換えていない。新しい学習用 WAV は作っていない。360,000 件の DNS 受理登録と GPU 学習は始めていない。

## 監査対象

4 領域に限る。

1. 6 モデル共通の DB・Data Pipeline
2. オンデマンド音声生成
3. 公式データ生成規則との小規模照合
4. 公式学習 Dataset の準備

モデル固有の生成・Dataset・公式照合は、公式素材がある MP-SENet、MossFormerGAN、TF-GridNet、TF-Locoformer である。DeepFilterNet3 と SEMamba++ は、共通 Pipeline を変えたときの回帰だけを見る。公式素材の blocked 行は変えていない。

対象外は、本学習、公式 epoch、大規模 GPU、最終 WER / PESQ / STOI、アーキテクチャ、損失、Optimizer、Scheduler、初期化、checkpoint 選択、生成波形のメモリキャッシュとディスクキャッシュ、事前生成 WAV を読む新しい学習経路である。

## 総合

| 領域 | 判定 |
|---|---|
| 共通 DB・Data Pipeline | CONDITIONAL GO |
| オンデマンド音声生成 | CONDITIONAL GO |
| 公式規則との小規模照合 | CONDITIONAL GO |
| 公式学習 Dataset の準備 | BLOCKED |

修正可能な BUG と MISMATCH は 0 件である。残るのは、公式情報の不足、30 分を超える登録の未承認、今回の対象外である。詳細な計測は [P10_A_VERIFICATION](P10_A_VERIFICATION.md)。

共通 Pipeline を CONDITIONAL GO にした理由は、実装上の重大な不具合を直し、回帰が通ったうえで、`resource.duckdb` に `eval` が無く leakage 監査を PASS にできないためである。オンデマンド生成は 4 モデルの学習入口で再構築できた。TF-Locoformer の本番入口は受理規則を持たない。公式照合は、測った MS-SNSD の連結と DNS の混合式が公式関数と一致し、VoiceBank の公式 16 kHz 変換器、DNS の librosa 版、`cv_synthetic_small` は UNKNOWN のままである。Dataset 準備は、MP-SENet、MossFormerGAN、TF-GridNet の公式規模オンデマンド Dataset を一意に指せる。TF-Locoformer の受理規則による 360,000 件は未登録なので、この領域は BLOCKED である。

## 問題一覧

重大度は Critical / Major / Minor。修正後の判定は、この監査で測った範囲のラベルである。

| ID | 分類 | 重大度 | 場所 | 原因 | 修正 | 修正後 |
|---|---|---|---|---|---|---|
| P10A-01 | BUG / OFFICIAL_DIFF | Critical | `src/reproduction/build/ms_snsd.py` `_render_pieces` | オンデマンド描画が区間を生の振幅で連結していた。公式 `audioread(norm=True)` は連結前にファイル全体を −25 dBFS にする。clean の大半は 10 秒以下で連結される | ファイル全体を −25 dBFS にしてから区間を切る。RMS は epsilon なし。case 文書と snapshot は区間だけを持つので ID は変えていない | MATCH。合成 2 ファイルと実データの複数ファイルで公式関数と配列一致 |
| P10A-02 | PERFORMANCE | Major | `src/reproduction/build/dns2020.py` | `activitydetector` の Python `sum` と、16 kHz ファイル全体の読み込みが受理計画を遅くしていた | フレームエネルギーは `np.add.reduce`。clip 判定は `np.any`。16 kHz は区間読み。案件ごとの一時キャッシュ。`workers` は spawn で `gen_index` の連続区間を分ける | 計画は速くなった。360,000 件の登録は未実施で、P10A-08 のまま |
| P10A-03 | UNKNOWN | Major | TF-Locoformer の検証集合 | 保存 config は `dump/raw/cv_synthetic_small/{wav,spk1}.scp` を指す。件数、seed、作成スクリプトはこの checkout に無い | 推測で作らない。sorted uttid の末尾 36,000 件をその名前で呼ばない | UNKNOWN |
| P10A-04 | UNKNOWN | Major | VoiceBank 16 kHz 化 | 公式の事前変換器が不明 | soxr 1.1.0 HQ と PCM_16 を記録済みの方法として維持し、公式変換器との一致は主張しない | UNKNOWN |
| P10A-05 | UNKNOWN | Major | DNS の非 16 kHz | 公式 `librosa.resample(y, fs_in, fs_out)` の版と `res_type` が不明。この環境の librosa 1.0.0 は位置引数を受けず、`resampy` が無いので `kaiser_best` を実行できない | 描画時の `soxr_hq` を維持。元素材は書き換えない | UNKNOWN |
| P10A-06 | OFFICIAL_DIFF | Major | DNS のファイル選択 | 公式はプロセス間の共有カウンタ。添字は加算後の剰余で 1 から始まる。スケジュールに依存する | 案件ごとの `Generator(PCG64(mix_seed("dns", seed, gen_index)))` を維持。ビット一致は主張しない | INTENTIONAL_DIFFERENCE |
| P10A-07 | OFFICIAL_DIFF | Minor | DNS の外側ループ | 公式は `while True`。実装は 50 回で打ち切る | 別 Protocol の `algorithm_change` として残す。無制限にはしない | INTENTIONAL_DIFFERENCE |
| P10A-08 | DATA_GAP | Critical | 受理規則の公式規模 Dataset | 音声探索、ハッシュ、360,000 件の JSON 登録を合わせると 30 分を超える見込み | 登録していない。承認後に新しい Dataset 名、新しい snapshot、追記だけにする。`tf_locoformer.dns.recipe@1` は変えない | BLOCKED |
| P10A-09 | DATA_GAP / OUT_OF_SCOPE | Major | DeepFilterNet3 | 公式学習素材が無い | `dfn3.online.official@1` は `blocked` のまま | BLOCKED。モデル固有作業は対象外 |
| P10A-10 | DATA_GAP / OUT_OF_SCOPE | Major | SEMamba++ | 公式学習素材が無い | `semambapp.gsr.official@1` は `blocked` のまま | BLOCKED。モデル固有作業は対象外 |
| P10A-11 | UNKNOWN | Minor | ESPnet chunk の順序 | shape ファイルが無く、`collect_stats` の順を再計算できない | 呼び出し側の発話順から計画する現行仕様を維持 | UNKNOWN |
| P10A-12 | UNKNOWN | Major | leakage 監査 | `resource.duckdb` に `eval` が無い | PASS にしない | UNKNOWN |
| P10A-13 | OFFICIAL_DIFF | Minor | `_build_signal` | `is_test_set` の clean が目標より長いとき、公式はファイル全体を残す。成功条件が `remaining == 0` だと、その案件を失敗にしていた | 成功条件を `remaining <= 0` にした。登録済み Dataset は `is_test_set` を false のままにする。その経路では長さが目標と一致する | MATCH。公式 `build_audio` と長さ 1,000 が一致。学習 Dataset の文書は変わらない |
| P10A-14 | OFFICIAL_DIFF | Minor | MP-SENet の crop | 公式は共有 `random.Random`。実装は `mix_seed` から開始位置を引く。範囲は同じ包含区間 | 文書化済みの差として残す。世界サイズが件数を割り切るとき、drop_last と公式 pad の採る件数は一致する。11,572 は 2 と 4 で割り切れる | INTENTIONAL_DIFFERENCE |

修正したのは P10A-01、P10A-02、P10A-13 である。P10A-02 の速度改善は登録そのものを完了させていない。

## モデル別

### MP-SENet

学習入口は `mp_senet.vb28.ondemand.official@1` と `mp_senet.vb28.ondemand.safe@1`。official は train 11,572 / valid 824 / test 824、`selection_uses_test = true`。safe は train 10,700 / valid 872（p286、p287）/ test 824、`selection_uses_test = false`。case は 48 kHz を指す。正規化・窓・順序は Dataset の protocol を読む。`cases.protocol_id` は共有の `mp_senet.fixed_pair.ondemand` である。履歴の 16 kHz Dataset は残り、新しい学習の `execute` は `delivery: on_demand` 以外を拒否する。

実データの epoch 0、rank 0、world 1 で batch は `(4, 32000)`。同じ呼び出しの再実行は一致し、再開位置の読み戻しも一致した。公式 16 kHz 変換器は P10A-04 のまま UNKNOWN である。

### MossFormerGAN

学習入口は `mossformergan.vb28.ondemand.official@1` と `mossformergan.vb28.ondemand.safe@1`。件数は MP-SENet と同じ対応である。正規化は `independent_audio_norm_v1`、順序は `rank_mod_fixed_v1`。world size 2 の rank 0 と rank 1 は別の窓を返す。再実行は一致した。公式変換器は P10A-04 と同じ UNKNOWN である。

### TF-GridNet

公式規模の学習入口は `tf_gridnet.ms_snsd.espnet_recipe@1`（train 7,345 / valid 1,835 / test 305、`selection_uses_test = false`）。小規模オンデマンドは `tf_gridnet.ms_snsd.ondemand@1`（20 / 5 / 5）。`tf_gridnet.ms_snsd.smoke@1` は保存済み WAV の履歴で、新しい学習は拒否する。拒否文は `dataset tf_gridnet.ms_snsd.smoke@1 is not delivery on_demand` である。

P10A-01 のあと、実データの複数 clean を公式 `audioread`、連結、無音、雑音の先頭切り出し、`snr_mixer` と比べ、clean / noise / noisy の最大絶対誤差は 0、実測 SNR は 10.0（名目 20 の半分）だった。再実行は一致した。公式の未シード乱数とのビット一致は主張しない。保存済み smoke の WAV は再生成していない。

### TF-Locoformer

本番 DB の学習入口は `tf_locoformer.dns.recipe@1`（train 324,000 / valid 36,000 / test 150）と `tf_locoformer.dns.smoke@1`（train 7 / valid 1、test 150 は共有）である。両方とも `length_plan_no_sample_retry` で、activity と clip の再試行を計画に入れない。意味は変えていない。

受理規則は子 Protocol `dns.tflocoformer.ondemand.accept`（`deterministic_case_stream_v1`、`dns_pcm16_accept_v2`）。実測サンプル `tf_locoformer.dns.accept_smoke@1` は snapshot `f75b3e7a4dbab8a68a6594487b3d605e949d96ef85fb8667b34417ea12101a39` と `data/staging/reproduction-p71` にあり、`reproduction.duckdb` には無い。公式規模の受理 Dataset は P10A-08 で未登録である。

この監査で実データの 30 秒を 2 件受理した。長さは 480,000、clean activity は 0.6367 と 0.64、元素材 clip は無し、SNR 誤差は −2.35e-5 dB と −7.34e-6 dB、再実行は一致した。公式の共有カウンタとは P10A-06 の差がある。`cv_synthetic_small` は P10A-03 のままである。

### DeepFilterNet3 と SEMamba++

`reproduction-p8.duckdb` の `dfn3.online.official@1` と `reproduction-p9.duckdb` の `semambapp.gsr.official@1` は `blocked` のままである。ready なのは代替素材の smoke と substitute だけであり、公式 Dataset の準備完了ではない。共通 Pipeline の変更後、`tests/reproduction` の P8 と P9 は通った。

## 公式との差異

| 項目 | ラベル | 根拠 |
|---|---|---|
| MS-SNSD の連結前 −25 dBFS と `snr_mixer` | MATCH | 合成 2 ファイルは `np.array_equal`。実データの複数ファイルは最大絶対誤差 0 |
| MS-SNSD の名目 SNR の半分 | MATCH | 既存の格子テストと、実測 SNR 10.0 |
| MS-SNSD のファイル選択 | INTENTIONAL_DIFFERENCE | 公式は未シード。実装はクリップごとの `mix_seed` |
| DNS `snr_mixer`、`activitydetector`、`is_clipped` | MATCH | 公式 `audiolib.py` と浮動小数が一致。activity のエネルギー和は 30 信号で最大絶対誤差 0 |
| DNS の実 30 秒 2 件 | MATCH（式に対して） | 長さ、activity、clip、SNR 誤差。公式ランのビット列ではない |
| DNS の共有カウンタ | INTENTIONAL_DIFFERENCE | P10A-06 |
| DNS の外側 50 回 | INTENTIONAL_DIFFERENCE | P10A-07 |
| DNS test-set clean の長いファイル | MATCH | 公式 `build_audio` の出力長 1,000 と一致。登録済み学習 Dataset はこの旗を使わない |
| DNS librosa の版 | UNKNOWN | P10A-05 |
| `cv_synthetic_small` | UNKNOWN | P10A-03 |
| VoiceBank の公式 16 kHz 変換器 | UNKNOWN | P10A-04 |
| MP-SENet の正規化してから crop、末尾 0 埋め | MATCH | 既存の計画テストが公式順と一致 |
| MP-SENet の乱数系列 | INTENTIONAL_DIFFERENCE | P10A-14 |
| MossFormerGAN の独立 `audio_norm` と rank 固定 | MATCH | 既存の公式 sampler テストが通過 |
| shape ファイル順 | UNKNOWN | P10A-11 |
| DeepFilterNet3 / SEMamba++ の公式素材 | N/A | 今回のモデル固有照合の対象外。公式行は blocked |

STATISTICAL_MATCH は、この監査の新規計測には使っていない。分布が公式の式の範囲に入ることと、配列が一致することは分けて書いた。ビット一致を主張するのは MATCH の行だけである。

## 未解決

修正可能な問題は 0 件である。

外部情報が要るもの: P10A-03、P10A-04、P10A-05、P10A-11、P10A-12。

承認が要るもの: P10A-08。P7 の条件登録は音声探索なしで合計 2,154 秒だった（DuckDB 登録 1,351.7 秒、SHA-256 357.2 秒）。この監査の受理計画は、直前に読んだ 16 kHz の実ファイル 4 件・30 秒で 0.132 秒、続けて 0.125 秒、1 件あたり約 0.033 秒、外側の試行はすべて 0、長さは 480,000 だった。この速度が冷たい読み込みでも維持されるとは限らない。ハッシュと 360,000 件の挿入を足すと 30 分を超える。承認前に始めない。登録するときは新しい Dataset と新しい snapshot へ追記し、混合 WAV は書かず、`tf_locoformer.dns.recipe@1` は変えない。worker は spawn と連続した `gen_index` のままで、文書は worker 数に依存しない。

対象外: P10A-09、P10A-10、本学習、最終評価、損失と Optimizer。

## 再監査

P10A-01 のあと、MS-SNSD の公式関数との配列一致、DNS の式テスト、worker 数を変えても文書が同じこと、事前生成 WAV を書く関数が無いこと、smoke Dataset の拒否、P8 / P9 の公式行が blocked であることを確認した。P10A-13 のあと、`tests/reproduction` は 108 件 OK（510.719 秒）で、公式 `build_audio` との長さ一致を含む。新しい失敗は出ていない。
