# P9 検証（2026-11-07）

SEMamba++ のオンライン劣化を、ピン留め `Eddieyjlee/SEMambapp` `cd369c89a8106072c077b468417b765b227afd74` の `dataset.py`、`simulate_degradation.py`、`simulate_degradation_utils.py` と HEAD の `config.yaml` に合わせて実装した。論文の 24,000 サンプルと `config_orig.yaml` は対象にしていない。公式の VCTK 0.92、Arni RIR、WHAM! の採用リスト、空のマニフェストは手元に無い。ready な Dataset は代替素材で、公式素材の Dataset は `blocked` のままである。劣化波形の WAV は書いていない。

公式コードの起動不能箇所は直していない。乱数列のビット一致は主張しない。確率分布の一致とビット一致は分けて判定する。

## 1. 総合判定

| 水準 | 判定 |
|---|---|
| オンライン劣化生成基盤 | CONDITIONAL GO |
| 代替素材による学習プロトコル | CONDITIONAL GO |
| 公式データセットの再現 | BLOCKED |
| 公式学習結果の再現 | NO-GO |

劣化の式、順序、VAD 基準の SNR、独立 peak、48,000 サンプル、オンデマンド再構築、代替 Dataset の登録とスナップショット、Trainer の 1 step・検証・再開は通った。条件は、`.venvs/se` に `resampy` が無く、`kaiser_best` と `kaiser_fast` は `.venvs/asr` の `resampy==0.4.3` を `sys.path` に足したときだけ実行したことである。この 2 方式は HEAD `config.yaml` の `bandwidth_type` に入っている。`resampy` が無いと公式の `librosa.resample` と同じ `ModuleNotFoundError` になる。`.venvs/se` へは入れてない。

公式素材と公式の固定 validation は無い。公式の学習結果は出していない。P10 には進んでいない。

## 2. 公式コードの固定 commit

`src/se/official/semambapp` は `cd369c89a8106072c077b468417b765b227afd74`。設定の正本は同 commit の `config.yaml`。

確認した実行経路:

- 劣化順は `reverb` → `noise` → `clipping` → `bandwidth`。`packet_loss` は `degrad_types` に無い。`packet_loss_prob: 0.2` は辞書に残るだけで、ベルヌーイ抽選しない。
- `__getitem__` は `random_select_and_order` を 2 回呼び、1 回目を捨てる。`train.py` は `seed=None` を渡す。`if seed:` は偽なので、2 回とも CPython `random` とグローバル `np.random` が進む。
- 雑音の切り出しだけ `np.random.default_rng(seed=None)` で、プロセスのエントロピーを使う。
- 教師は `apply_degradation` が返す元の clean。RIR の正規化と遅延補正は無い。畳み込みは `mode="full"` のあと入力長へ切る。
- peak は劣化後、crop の前。clean と degraded を別々に `x / (max|x| + 1e-9)`。float32 にしてから割る。
- 窓は 48,000。短ければ右側を 0 で埋める。crop 開始は両端を含む。
- `bandwidth_limitation` は low-pass、`filtfilt`、downsample、upsample、入力長への切り出し。`cheby2` には `ripple`（既定 1）を `rs` として渡す。関数は `ripple` と `stop_atten` を引数で受けないので、既定 1 と 40 が使われる。

起動しない箇所は記録だけした。

- `train.py` の config 名は `semambapp_cfg.yaml`。そのファイルは無い。あるのは `config.yaml`。
- `dataset.py` は存在しない `utils.util` を import する。
- `simulate_degradation_utils.py` は import 時に `pyroomacoustics` と `torchaudio.io` を読む。`.venvs/se` では `pyroomacoustics` が無い。数値比較は、この 2 つを空モジュールで埋めてから関数を import した。公式ファイルは変更していない。
- `data/train_speech.json`、`data/val_clean.json` などは改行 1 バイト。`json.load` できない。公式の 100 ペア validation は実行できない。

## 3. 公式データセットの不足素材

| 項目 | 手元 | 扱い |
|---|---|---|
| VCTK 0.92 | README は 0.80。話者 108。`p280` はあり、`p315` は無い | 0.92 として登録しない。代替の speech は 0.80 |
| Arni RIR | 無し | 登録しない |
| OpenSLR28 `simulated_rirs` | small / medium / large room がある | 代替 RIR。Arni ではない |
| DNS5 `impulse_responses` | SLR26 と SLR28 | 束ねない。Arni でも公式の学習リストでもない |
| WHAM! | `WHAM48/high_res_wham` はある | 論文の部分集合と split が不明なので束ねない |
| DNS2020 `datasets/noise` | 直下の wav が 65,302 | 代替 noise。公式の部分集合ではない。DNS5 ではない |
| `train_*.json` / `val_*.json` | 改行のみ | 公式のファイルリストと 100 ペア validation は不明 |
| VCTK-GSR の 99/8 split | 話者の順序リストが無い | 代替はディレクトリ名の昇順で話者を分ける |

## 4. 公式と audio-db の対応

| 項目 | 公式 | audio-db | 判定 |
|---|---|---|---|
| 素材 | VCTK 0.92、WHAM!、Arni、空の JSON | VCTK 0.80、DNS2020 noise、OpenSLR28 simulated RIR。公式 Dataset はバインド無し | BLOCKED（公式）/ 代替は別 claim |
| 劣化順序 | reverb → noise → clipping → bandwidth | 同じ。選択集合をこの順に並べる | MATCH |
| packet loss | 確率 0.2 はあるが `degrad_types` に無い | 適用しない。選択集合に出たらエラー | MATCH |
| Reverb | 畳み込み、入力長、RIR 正規化なし、遅延補正なし。教師は dry clean | 同じ | MATCH |
| VAD | frame 1024、hop 512、boxcar、閾値 0.01、フレームパワー / 平均パワー | 同じ式。boxcar なので重なりフレームへの in-place 窓は値を変えない | MATCH |
| Noise | 短い雑音は wrap、長い雑音は切り出し、同じ長さは offset 無し。尺度は `10^(-snr/20)*sqrt(Ps)/sqrt(max(Pn,1e-10))`。Ps、Pn は VAD 区間 | 同じ式。offset の乱数器は下記 | MATCH（式）/ INTENTIONAL_DIFFERENCE（乱数器） |
| Clipping | 分位点。min は U[0, 0.1)、max は U[0.9, 1.0)。degraded に掛ける | 同じ。`np.clip(x,-1,1)` ではない | MATCH |
| Bandwidth | low-pass のあと resample down/up。`cheby2` は `rs=ripple` | 同じ。短い出力だけ 0 埋めする。公式のスライスは短いまま返し、後の長さ検査が落ちる | MATCH（フィルタと resample）/ INTENTIONAL_DIFFERENCE（短いときの 0 埋め） |
| Peak | 全長を別々に peak。そのあと同じ crop | `independent_peak_v1` | MATCH |
| Crop | 48,000。短ければ右 0 埋め。開始位置は両端を含む | 同じ | MATCH |
| 読み込み | `librosa.load(..., sr=16000)`。mono の既定はチャンネル平均 | 同じ呼び出し | MATCH（呼び出し）。別の reader とのビット一致は測っていない |
| 乱数 | プロセス共有の CPython `random`、`np.random`、offset は `default_rng(None)` | サンプルごとに 1 つの PCG64。2 回目の抽選を使う | INTENTIONAL_DIFFERENCE（分布は合わせ、ビットは合わせない） |
| DDP | rank ごとに未シード shuffle が 1 回。`DistributedSampler(shuffle=True, seed=0, drop_last=False)` のあと DataLoader `drop_last=True`。world size 1 は sampler 無し、`shuffle=False` で epoch ごとに並べ直さない。`batch_size // n_gpus` | 既存の `per_rank_independent_v1`。seed 付き。drop_last は切り詰め。world size 1 も epoch ごとに並べ直す。global batch は `micro * world * accumulation` と一致しなければならない。波形の種に rank は入らない | INTENTIONAL_DIFFERENCE（方針の中身は変えていない） |
| Validation | ソートした事前劣化ペアの先頭 100。全長。batch 1。rank 0。checkpoint 選択には使わない | 代替の valid/test を epoch に依らずオンラインで 3 秒にする。公式ペアは無い | BLOCKED（公式）/ INTENTIONAL_DIFFERENCE（代替） |
| Checkpoint | 5,000 step ごと。`strict=False`。best 指標では選ばない | 既存 Trainer。`semambapp` は best を書かない。`load_state_dict` は `strict=True` のまま | INTENTIONAL_DIFFERENCE（公式の `strict=False` には合わせない） |
| STFT | Dataset が波形と STFT を返す | 波形だけ返す。`SEMambaTask._generate` が STFT を 1 回計算する | INTENTIONAL_DIFFERENCE（二重 STFT はしていない） |

## 5. 劣化順序と信号処理

固定入力で、ピン留め関数と audio-db を同じプロセスで比べた。`pyroomacoustics` と、無いときの `torchaudio.io` だけを空にした。

| 処理 | 結果 |
|---|---|
| VAD マスク（長さ 1023、1024、8000） | 不一致 0 |
| 雑音が短い / 長い / 同じ / speech が 0 | 波形の最大絶対誤差 0。検査の閾値は 1e-12 |
| 畳み込み | 出力長は入力長。最大絶対誤差 0。RIR は割っていない |
| quantile clip | 最大絶対誤差 0。`np.clip(x,-1,1)` との最大絶対誤差は 0.5 より大きい |
| bandwidth | `scipy`/`butter`、`scipy`/`cheby2`、`polyphase`/`ellip`、`kaiser_fast`/`bessel`、および別プローブの `kaiser_best`/`butter`。長さは入力と同じ。最大絶対誤差 0（`kaiser_best`）または 1e-5 未満 |
| フィルタ格子 | butter / cheby1 / cheby2 / bessel / ellip、次数 2・4・6・8、`scipy` / `polyphase` / `kaiser_fast`。16,000 サンプルの正規乱数で、すべて有限、長さは入力と同じ、0 埋めは起きなかった |
| 短い resample | `librosa.resample` を半分の長さを返す関数に差し替えたときだけ、末尾を 0 で埋める。長い出力は入力長へ切る。この分岐は実フィルタでは起きなかった |

選択された劣化は、常に reverb、noise、clipping、bandwidth の部分列になる。noise の確率が 1 なので、空の選択は起きない。公式の空選択は、合計 2.5 の重みを `np.random.choice` に渡して失敗する。audio-db はその分岐で `ReproductionError` を出す。HEAD の設定ではどちらも通らない。

独立 peak では、次を合成波形で外した。clean の peak を degraded に使う実装、crop のあとで peak を取る実装、正規化後も目標 SNR が残るという仮定。2 本の全長 peak は 4 と 8 で違い、同じ区間を切った。crop 後の clean の最大値は 0.2 未満で、crop 後に peak を取り直した波形とは 0.5 より離れた。

## 6. VAD 基準 SNR の数値比較

目標 5 dB の固定例で、VAD パワーから戻した SNR は 5 から高々 1 ULP（表示は `5.000000000000001`）だった。speech が 0 のときは尺度 0、測定 SNR は無し、出力は 0。公式と一致した。

全波形 RMS で尺度を作った比較用の信号（無音が長く、中央だけ 0.2）では、RMS パワーと VAD パワーの相対差は 0.2 より大きく、尺度の差は 1e-3 より大きかった。audio-db の尺度は VAD 側と一致し、RMS 側とは 1e-3 より離れた。

長さ 24,000 の合成 16 例（seed `20261107` の続き）は、peak と右 0 埋めのあと 48,000 サンプルで有限、絶対値の最大は 1 以下だった。雑音を混ぜた直後の VAD SNR と目標 SNR の絶対誤差の最大は 1e-6 未満だった。peak のあとでは、同じ目標 0 dB の例が 1 dB よりずれた。全長の SNR は正規化後に保存しない。

## 7. 乱数と DDP

分布は実行前に固定した。seed `20261107`、`draw_once` を 4,000 回。窓は 4 シグマ相当で、実行後に広げていない。

| 項目 | 期待 | 窓 | 実測 |
|---|---|---|---|
| noise | 4,000 | ちょうど 4,000 | 4,000 |
| reverb | 2,000 | 1,872–2,128 | 2,021 |
| clipping | 2,000 | 1,872–2,128 | 1,984 |
| bandwidth | 2,000 | 1,872–2,128 | 2,056 |
| SNR 6 値 | 666.7 | 各 572–761 | −5: 678、0: 657、5: 648、10: 674、15: 653、20: 690 |
| 8 通りの組合せ | 500 | 各 416–584 | noise 482、noise+bandwidth 516、noise+clipping 487、noise+clipping+bandwidth 494、reverb+noise 485、reverb+noise+bandwidth 533、reverb+noise+clipping 490、4 種 513 |
| packet_loss の選択 | 0 | 0 | 0 |

clipping の min は [0, 0.1)、max は [0.9, 1.0) に入った。8 通りはすべて出た。これは使用した抽選の周辺分布である。公式のビット列ではない。

1 例の中では、捨てる抽選のあと使う抽選を行う。seed 0 から 29 のうち、選択集合が 2 回で違うものが少なくとも 1 つあった。公式はプロセス全体で乱数がつながる。audio-db はサンプルごとに生成器を作るので、他のサンプルの捨て抽選では進まない。

種は `mix_seed("semamba-gsr", protocol_id, snapshot_hash, seed, split, epoch, index)`。train だけ epoch を入れる。valid と test は 0。rank と worker は入れない。同じ index は world size 1 と 2 で同じ波形だった。worker 1 と 2 も同じだった。2 回目の配列オブジェクトは 1 回目と別で、波形キャッシュは持っていない。

`per_rank_independent_v1` の並べ方は変えていない。説明文だけ、公式 sampler の `drop_last=False` と DataLoader の `drop_last=True`、および world size 1 でも epoch ごとに並べ直すことを足した。

計画の確認:

- world size 1 / 2 / 4、件数 16、micro 2。再計画の hash は一致。epoch 0 と 1 の rank 0 は違った。
- world size 1 は 16 件を過不足なく使う。
- world size 2、seed 1、件数 8 は、既存テストと同じく重複または欠落が起きる。
- 件数 10、micro 4、world size 1 は末尾 2 件を捨てる。
- 蓄積 2 は、1 更新が micro-batch 2 つ。smoke の Trainer でも 1 更新が micro-batch 2 つだった。

公式の world size 1 は epoch ごとに並べ直さない。audio-db は並べ直す。公式は重複した発話でも rank ごとに雑音を引き直す。audio-db は同じ index なら同じ波形である。どちらも `deviations_json` に書いた。

## 8. Dataset と Snapshot

`data/db/reproduction.duckdb` と `data/db/resource.duckdb` は書いていない。登録先は `data/db/reproduction-p9.duckdb`、staging は `data/staging/reproduction-p9/`。スナップショットは `data/db/snapshots/<dataset_id>/` に新しいディレクトリだけ書いた。

| Dataset | 状態 | claim | dataset_id | snapshot_hash |
|---|---|---|---|---|
| `semambapp.gsr.substitute@1` | ready | `protocol_conformant_substitute_materials` | `4acf30da286a36cb8c6edcdf1091456c979fd839d7898ecc7528467e4b04d76e` | `1b9be5fe72e9522516aff29f9ad60f21a3fe4f2dbfe3912a08c42173f3f66d37` |
| `semambapp.gsr.smoke@1` | ready | `protocol_conformant_substitute_materials` | `d3bc5dbea5b7a19c8495a0d2ce5d7d53267c96d6f82dc049c193fd14a04980eb` | `dd57fd637dbc095fc77a082bd9f8a3e6c0e2a52ec00df89690f2717cdfb2802f` |
| `semambapp.gsr.official@1` | blocked | `official_reproduction` | `63a0e6785851996743c1ed040ddfbabb811c00098f821d09e4c57ea36a1f995f` | 無し |

親 Protocol `semambapp.gsr` は `02ca964721b488f26325092aa510e06e37f96818784e3bb321ce837be33975b7`（`official_effective`）。子 `semambapp.gsr.audio-db` は `ee7a448b95c7961d9243897bb24cde79aa9217407dbb53cb0698debd8158a970`（`algorithm_change`）。代替 Dataset は子、blocked の公式 Dataset は親を指す。数値の正本は `SEMAMBA_GSR_V1`。スナップショットへ戻した `rule_json.online` はこの辞書と等しかった。`1e-9` と `1e-10` は文字列で持ち、6 桁の recipe hash で 0 に潰さない。

`count_check`、`list_hash_check`、`leak_check` は 3 Dataset とも PASS。公式はバインドが空なので leak は空集合の PASS。2 回目の `register_p9` の挿入件数はすべて 0。staging とスナップショットに wav と npy は 0。ユニークな素材は 60。

| 役割 | substitute train / valid / test | smoke train / valid / test |
|---|---|---|
| speech | VCTK 0.80 の昇順話者。8 話者×2、4 話者×1、4 話者×1 | 16 本の接頭辞 8、valid/test の接頭辞 2 / 2 |
| noise | DNS2020 の昇順 wav。12 / 4 / 4 | 4 / 2 / 2 |
| RIR | smallroom 8、mediumroom 4、largeroom 4 | 2 / 2 / 2 |

話者 ID は、speech が話者ディレクトリ、noise がファイル名、RIR が `simulated_rirs` からの相対ディレクトリ（`smallroom/...`）である。split をまたぐ資産・話者・SHA-256 の重複は登録時に落ちなかった。

train の case は無い。valid と test の case は `online_snapshot` で、clean / noisy / noise の列は空。長さ 48,000、教師 `dry_clean`。学習はスナップショットだけを読む。

スナップショットのファイル合計は substitute が 56,222 バイト、smoke が 37,523 バイト。staging のファイル合計は 101,563 バイト。中身は索引である。`reproduction-p9.duckdb` は 11,808,768 バイト。

`selection_uses_test` は 3 つとも false。smoke の global batch 2 と 4 は、規則の 8 と違うので `batch_warning` が出る。identity には現在の batch を入れない。

## 9. Trainer、validation、再開

`open_training_data` は `online.semambapp_gsr` を `OnlineSemambaTrainingData` へ渡す。`online.dfn3` はそのまま。`run.py` の reproduction 経路は `semambapp` を `online.semambapp_gsr` に対応させた。モデル、損失、optimizer、scheduler は変えていない。

smoke、global batch 4、micro 2、蓄積 2、world size 1、seed 7。

- 1 step で micro-batch が 2 つ。波形は、同じ計画を先に描いた 1 更新目と一致した。
- validation が動いた。PESQ は stub が na。`best_generator.pt` は無い。`model.name` が `semambapp` のとき既存 Trainer は best を書かない。
- `step_00000001.pt` から再開し、2 更新目の波形が再開前の計画と一致した。完了後の `data_state` は epoch 1、update 0、`per_rank_independent_v1`。
- `data_provenance.json` の claim は `protocol_conformant_substitute_materials`、algorithm は `online.semambapp_gsr`。`metrics.jsonl` に train 記録がある。
- バッチに `mag` は無い。clean と noisy の形は同じ。教師は `dry_clean`。valid の epoch 0 と 3 は同じ波形。train の epoch 0 と 1 は違う。

実モデルの 1 step は CPU。`hid_feature` 8、`num_tfmamba` 1、`d_state` 4、`d_conv` 4、`expand` 2、`n_fft` 400、`scan_backend` reference。Discriminator は実装のまま。入力は smoke の 48,000 サンプル。損失は有限。単体テスト 307.036 秒、経過 312.03 秒、最大 RSS 3,339,504 KB。公式幅、fused CUDA、複数 step の GAN は NOT_TESTED。

`SelectiveSSM` の `dt_proj.bias` に `_no_reinit` は無い。`softplus` は 0.1 を超える要素がある。公式 `mamba_ssm` 2.3.2 は bias を softplus の逆関数で `[dt_min, dt_max]`（既定 0.001 から 0.1）に置き、`_no_reinit` を付ける。`A_log` はこちらも `log(1..d_state)`。この初期化差は scratch 学習だけで、今回は直していない。P10 の項目である。

## 10. 実行したテスト

コマンドは `.venvs/se/bin/python -m unittest`。GPU ジョブと `qsub` は使っていない。P9 のテストは、無い `resampy` の代わりに `.venvs/asr` の site-packages を `sys.path` へ足す。

| 項目 | 結果 |
|---|---|
| `tests.reproduction.test_p9_semamba` の信号・分布・計画・Dataset・stub Trainer・`dt_proj`（17 件。48,000 の実 step を除く） | PASS。19.184 秒、経過 23.73 秒、最大 RSS 609,448 KB |
| 同じファイルの短い resample の 0 埋め | PASS（上の 17 件に含まれる） |
| 縮小 generator の 48,000・1 step | PASS。307.036 秒、経過 312.03 秒、最大 RSS 3,339,504 KB |
| 公式幅・fused・GPU の学習 step | NOT_TESTED |
| P1〜P8、`test_p5_chunk`、`test_batch_schedule`、`test_protocol`、DeepFilterNet のスペクトルと比較、checkpoint、mock trainer、`test_df_libdf_frontend` | 126 件中 124 PASS、2 ERROR。277 秒、最大 RSS 1,324,280 KB |
| `test_df_libdf_frontend` の 2 件 | ERROR。`libdf.DF()` が引数を取らない。P9 の変更箇所ではない。P8 の記録と同じ |

17 件の再実行は、短い resample のテストを足したあとに行った。分布の窓は最初の成功した実行の前にファイルへ書いた。

## 11. 計算資源

CPU だけ。`nvidia-smi` の使用メモリは作業の前後で 1,218 MiB のままで、このプロセスは GPU を確保していない。`qsub` は出していない。30 分を超えるジョブは無い。一番長い単体は 307 秒の CPU step。

ディスクに足したのは `data/db/reproduction-p9.duckdb`（11,808,768 バイト）、staging のファイル合計 101,563 バイト、スナップショットのファイル合計 56,222 バイトと 37,523 バイト。劣化 WAV は 0。

## 12. 公式との未解決差分

- 公式のビット列。CPython `random`、グローバル `np.random`、`default_rng(None)` を、サンプルごとの PCG64 に置き換えている。
- 捨てる抽選が後続サンプルへ影響しない。
- world size 1 でも epoch ごとに並べ直す。公式の 1 GPU は最初の未シード shuffle のまま。
- sampler は切り詰める。公式は pad してから DataLoader が短い batch を捨てる。
- 重なった index の波形は rank で同じ。公式は rank ごとに引き直す。
- bandwidth が入力より短いときだけ 0 埋め。実測のフィルタ格子では起きなかった。
- 代替 validation はオンラインの 3 秒。公式の全長 100 ペアではない。
- STFT は Task の中。Dataset は波形だけ。
- checkpoint の `strict=True`。公式の `strict=False` と二重 scheduler の読み戻しには合わせない。
- `dt_proj.bias` の scratch 初期化。今回は未変更。
- `.venvs/se` に `resampy` が無い。`kaiser_best` と `kaiser_fast` は、検証時だけ `.venvs/asr` の 0.4.3 を使った。
- 公式の `batch_size // n_gpus`。端数は捨てる。audio-db は global batch を割り切れない構成を拒否する。smoke は規則の 8 より小さく、警告だけ残す。
- 最小起動補修（config 名、`utils.util`、空 JSON、`pyroomacoustics` の import）は実装していない。

## 13. P10 へ引き継ぐもの

P10 は未着手。次に要るものは次のとおり。

- VCTK 0.92、Arni、公式の WHAM! リスト、空でない train/val JSON、VCTK-GSR の話者順。これらが無い間、公式 Dataset は blocked のまま。
- `.venvs/se` で `kaiser_best` と `kaiser_fast` を動かすなら `resampy`。検証は asr 環境の 0.4.3 で通した。
- `dt_proj.bias` を公式の softplus 逆関数初期化に合わせるかは、scratch 学習の判断。今回のモデルは変えていない。
- 公式幅の GAN、fused CUDA、複数 step、5,000 step の checkpoint 間隔。CPU の縮小 1 step は有限だったが、公式の学習結果ではない。
- 公式 validation の全長 100 ペア。ファイルが無いので未実施。
- 30 分を超える学習と GPU ジョブは、別の承認が要る。

P10 の開始判断は NO-GO。素材、環境、初期化、計算資源が揃っていない。
