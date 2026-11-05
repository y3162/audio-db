# DeepFilterNet3 公式再現プロトコル監査

調査日 2026-11-06。調査と文書化のみで、コード・設定・DB は変更していない。修正案は提案として書く。

記法:

- `OFF/` は `src/se/official/deepfilternet`（固定 commit `d375b2d8309e0935d165700c91da9de862a99c31`）。読み取り専用で、履歴調査は `/tmp/audit-dfn/dfn`（全履歴の別 clone）で行った。
- 確度ラベル: **[確認:コード]** 該当ファイルを読み、`rg -n` / `cat -n` で行番号を確かめた。**[確認:論文]** 論文本文。**[確認:ファイル]** `ls`・ハッシュ・`git log` で確かめた。**[導出]** 確認済みの事実からの算術・論理で、実行していない。**[推測]** 根拠が間接的。**[UNKNOWN]** 閉じられていない。**[既存記録]** 既存 docs の記述で、今回は再実行していない。
- 「検証済み」は、今回または既存記録で実行して確かめた範囲だけに使う。

## 0. 要約

**結論**: 現行の audio-db の DeepFilterNet3（DFN3）学習は、公式の「損失・optimizer・scheduler・batch schedule・早期停止」に相当する部分をコードとして持つ。一方、公式の核であるオンライン混合データ基盤は無く、VoiceBank 28 話者の固定ペア 10,700 件で学習する。公式プロトコルの再現にはなっていない。

**選んだ variant**: 出荷モデル `models/DeepFilterNet3.zip`（48 kHz、非 low-latency、`config.ini` の実効値）。論文は Interspeech 2023 の 4 ページのデモ論文（arXiv 2305.08227v1）で、学習条件はほぼ書かれていない。学習の実効動作は `OFF/` のコードと出荷 `config.ini` から読む。

**データ方式の分類**

| 対象 | 方式 | 根拠 |
|---|---|---|
| 学習（train） | オンライン混合（Rust `libdfdata`、発話ごとに seed 付き RNG） | `OFF/libDF/src/dataset.rs:1211-1379`、`dataloader.rs:270-278` |
| 学習中の検証・最後の test | オンライン混合だが seed は発話 index のみで、毎回同じ結果になる擬似固定 | `dataloader.rs:272-277`、`pyDF-data/src/lib.rs:162-171` |
| 論文 Table 1 の評価 | 固定ペア（VoiceBank+DEMAND test 824 件の収録済み noisy） | `OFF/DeepFilterNet/df/scripts/test_voicebank_demand.py:26-27` |
| 事前生成 | 公式には無い（HDF5 は素材のみ。混合済みを保存しない） | `prepare_data.py` の type は speech / noise / rir（`noisy` は読み込み側の拡張。生成に使われない） |

**差異の上位**（詳細は §9）

1. DFN-D01 学習データ分布が別物（公式は DNS4 全体＋PTDB/VCTK×10、audio-db は VoiceBank 固定ペア）。
2. DFN-D02 オンライン混合（SNR リストに −100、複数雑音の加算、gain、5% 合成雑音、各種 augmentation）が未実装。
3. DFN-D03 残響（p_reverb 0.1、RIR の再サンプル・減衰・DRR 混合ターゲット）が未実装。
4. DFN-D04 audio-db は既定で発話 RMS 正規化を掛ける。公式には無い。
5. DFN-D05 公式は 3 秒窓を「別発話の連結」で埋める。audio-db は短い発話を零パディングする。
6. DFN-D06 検証が別物（公式は 3 秒のオンライン混合窓の batch 平均、audio-db は発話全長の 1 件ずつの平均）。
7. DFN-D07 epoch の端数 batch（公式は ceil で残す、audio-db は切り捨て）。10,700 件では更新回数が 22,831 と 22,711 で 120 回違う。
8. DFN-D11 評価（PESQ/STOI のリサンプラ、CSIG/CBAK/COVL 未実装、出荷モデルの Table 1 再現は未実施）。

**レシピ JSON v1 の判定**: オンライン混合の記述には**不足**。v1 は 1 件の固定ケース（雑音 1 本・SNR 1 値・RIR 1 本）の実現値しか書けない。公式の確率分布・複数雑音・連結・ガードを表せない（§10）。オンライン生成のプロトコルは、レシピとは別の層で持つ必要がある。

**ローカルで再現できない点**: DNS4 の多言語 clean 音声、PTDB、公式の `dataset.cfg`・ファイルリスト・件数は、この環境に無いか公開されていない（§12）。DNS5 は雑音 63,810 件と RIR のみで、clean 音声は無い [確認:ファイル]。

## 1. 調査メタデータ

| 項目 | 値 |
|---|---|
| 調査日 | 2026-11-06 |
| audio-db | branch `feature/reproduction`、HEAD `3bee9ad`。調査開始時の `git status` は clean [確認:ファイル] |
| 公式 commit（固定） | `d375b2d8309e0935d165700c91da9de862a99c31`。author 日 2024-09-25、committer 日 2024-10-17、subject `channel`。`git describe` は `v0.5.6-89-gd375b2d`、`pyproject.toml:3` は `0.5.7-pre`。`origin/main` と一致（clone 時点）[確認:ファイル] |
| 公式タグ | v0.5.0 `a76dbcb`（2023-05-26、DFN3 を含む最初の release）、v0.5.6 `978576a`（2023-08-31）。固定 commit は v0.5.6 の 89 commit 後 [確認:ファイル] |
| 出荷モデルの履歴 | `models/DeepFilterNet3.zip` は `84d57ec`（2023-05-23 "Update final DeepFilterNet3 model"）と `82b0c7a`（同日 "Fix zip structure"）[確認:ファイル] |
| 重み | `requirements/official-assets.toml` の zip sha256 `49c52edc8947…22284d2`（7,986,207 B）。`/tmp` の clone で zip と、`config.ini`（sha256 `415eb925…4290`）の一致を確かめた。`model_120.ckpt.best` は toml 記載の sha256 `23b92884…6e003`（8,714,073 B）[確認:ファイル] |
| 配置 | `data/checkpoints/deepfilternet3/DeepFilterNet3/{checkpoints/model_120.ckpt.best,config.ini}` が存在 [確認:ファイル] |
| 論文 | DFN3: arXiv 2305.08227（v1 のみ確認、2023-05-14 投稿）。DFN2: arXiv 2205.05474。DFN1: arXiv 2110.05588。本文は arXiv HTML / ar5iv で読み、`/tmp/audit-dfn` に保存 |
| ツール版 | `.venvs/se`: `pesq 0.0.4`、`pystoi 0.4.1`、`torchaudio 2.6.0+cu124`。公式 `pyproject.toml:37-38` は `pystoi>=0.3,<0.5`、`pesq>=0.0.3,<0.0.5` [確認:ファイル] |
| 今回実行していないもの | 公式コード・学習・評価の実行、audio-db のテスト・学習・評価、GPU 使用、DB 書き込み |
| この作業ツリーの環境差 | `data/db/` は `.gitkeep` のみで、`resource.duckdb`、`voicebank.duckdb`、`training.duckdb`、`backups/` が無い。`data/staging/` も空。`data/corpora/` は 14 本のシンボリックリンク。`.agents/DATA.md` が書く DB の実在はこの環境では確かめられない |

## 2. A. 公式資料

### 2.1 資料の一覧

| 資料 | 場所 | 内容 | 学習プロトコルの記述 |
|---|---|---|---|
| DFN3 論文 | arXiv 2305.08227v1（Interspeech 2023 デモ、4 ページ）`dfn3.txt:115-140` | 構造（48 kHz、20 ms 窓・10 ms hop、先読み 2 フレーム、ERB 32、DF 次数 5）、リアルタイム実装（RTF 0.19）、Table 1 | 1 文のみ: 「多言語 DNS4 全体で学習、PTDB と VCTK を 10 倍オーバーサンプル、VCTK/DEMAND test で評価」 [確認:論文] |
| DFN2 論文 | arXiv 2205.05474（IWAENC 2022）`2205.05474.txt:62-65,80-82,120-123` | AdamW、warmup 3 epoch＋cosine、weight decay の cosine 増加、batch 8→96 の schedule、損失（λ_spec=1e3、λ_MR=5e2、c=0.3、窓 5〜40 ms）、DNS4 英語部分、split 70/15/15、100 epoch、valid loss で best | DFN3 の出荷設定とは値が違う（§11） |
| 公式 repo | `OFF/`（Rust `libDF` / `pyDF-data` / Python `DeepFilterNet/df`） | 学習コード・dataloader・評価スクリプト | コードが一次資料 |
| 出荷 config | `models/DeepFilterNet3.zip` → `DeepFilterNet3/config.ini`（2,067 B） | 学習時の実効設定（§6） | 値はある。データは書かれていない |
| README | `OFF/README.md:177-273`（`DeepFilterNet/README.md` はシンボリックリンク） | HDF5 の作り方、`dataset.cfg` の形式、`train.py` の起動 | データ構成の指針のみ。DFN3 の具体は無い |
| データ作成補助 | `scripts/download_process_dns4.sh:22-75`、`df/scripts/{prepare_data,split_hdf5,trim_silence_hdf5}.py` | DNS4 の取得と HDF5 化の手順 | DNS4 を `TRAIN`（本体）と `VALID`（`dev_testset`）に分ける。`TEST` と `dataset.cfg` は作らない |
| テスト用 | `OFF/assets/dataset.cfg`、`assets/clean.hdf5` など | 形式の例（各 split に clean.hdf5 を factor 100、noise.hdf5 を 10） | テスト用。学習の実データではない |

### 2.2 variant の選択と理由

選択: **DFN3 出荷モデル（`models/DeepFilterNet3.zip`、非 ll）**。

理由: audio-db が strict load している重みであること（`docs/se/MODELS.md`）、`requirements/official-assets.toml` で固定していること、実効設定（`config.ini`）が手元にあること。

候補と違い [確認:ファイル]:

| 候補 | 内容 | 採用しない理由 |
|---|---|---|
| DFN3-ll | `models/DeepFilterNet3_ll_onnx.tar.gz`（ONNX のみ） | PyTorch の checkpoint と学習設定が無い |
| DFN2 | `models/DeepFilterNet2.zip` | 論文は別（IWAENC 2022）。epoch・batch・損失の値が DFN3 の出荷設定と違う |
| DFN1 | `models/DeepFilterNet.zip` | 別構造・別データ（DNS3） |
| v0.5.0 のコード | `a76dbcb`（2023-05-26） | モデル更新（2023-05-23）の直後でコードは近いが、再現対象の「学習コミット」は特定できない（§12 U02） |

### 2.3 版の違い（固定 commit は学習時のコードとは限らない）

出荷モデル（2023-05-23）の後、固定 commit までに、学習・データに関わる次の変更がある [確認:ファイル。`git log` / `git diff --ignore-cr-at-eol v0.5.0 d375b2d`]。

| 変更 | commit・日付 | 学習結果への影響 |
|---|---|---|
| `DATALOADER_GAINS` の追加 | `935e2cc` 2023-06-29 | 出荷 ini に `dataloader_gains` が無い。既定 [-6,0,6]。それ以前は gain の設定自体が無かった可能性がある [UNKNOWN] |
| LocalSnrTarget のクランプ ±5 → ±1 | `4ec757b` 2023-06-29 | 学習時の局所 SNR 目標が違った可能性。現行 audio-db は ±1 |
| FFT dataset の `min_nb_erb` が 1 固定だった不具合の修正 | `e71c8ba` 2023-07-05 | 出荷 ini の `min_nb_erb_freqs=2` が学習で効いていたか [UNKNOWN] |
| 合成雑音のレベル修正 | `b50ee3d` 2023-08-28 | 合成雑音（5%）の振幅分布が学習時と違う |
| 残響の初期遅延 5 ms → 20 ms、畳み込みを fftconvolve に | `2484334`、`16afb10` 2023-09-14 | 学習時の残響ターゲットが固定 commit と違う |
| `RandResample` の確率の環境変数名の不具合（`DF_P_LFILT` を誤用）の修正 | v0.5.0 〜 固定 commit の間 | 学習時の resample augmentation の確率が違った可能性 |
| `DeepFilterNet/df/deepfilternet3.py` +63 行ほか | v0.5.0 〜 固定 commit | 出荷重みは固定 commit のコードで strict load できる [既存記録] |
| `train.py`・`lr.py`・`checkpoint.py` | v0.5.0 〜 固定 commit で差は `gains` の 1 行のみ | 学習ループは同一 [確認:ファイル] |
| `evaluation_utils.py`・`test_voicebank_demand.py` | 2023-07 以降は torchaudio 互換・複数データセット対応のみ | 評価の計算式の変更は見つからない [確認:ファイル] |

したがって「固定 commit の挙動 ＝ 出荷モデルを学習した挙動」とは言えない。データ側（残響・合成雑音・gain・min_nb_erb）の再現では、この差を差異として扱う。

## 3. B. 学習データセット

### 3.1 論文・公式が示すこと

| 項目 | DFN3 の記述 | 出典・確度 |
|---|---|---|
| 学習データ | 多言語 DNS4 の全体、PTDB と VCTK を 10 倍オーバーサンプル | [確認:論文] `dfn3.txt:115` |
| 評価データ | VoiceBank+DEMAND の test（VCTK/DEMAND） | [確認:論文] 同上 |
| clean 音声の量・言語・話者数 | 書かれていない | [UNKNOWN] |
| 雑音・RIR の構成 | DFN3 には無い。DFN2 は「英語の DNS4: clean 500 h 超、雑音 約 150 h、実 RIR 150 と模擬 RIR 60,000（論文は "HRTFs" と書く）」 | [確認:論文] `2205.05474.txt:120`。DFN3 に同じとは言えない |
| 学習・検証・テスト分割 | DFN3 には無い。DFN2 は 70/15/15、VoiceBank は話者排他 | [確認:論文] `2205.05474.txt:121-122` |
| オーバーサンプル | `dataset.cfg` の `sampling_factor`。整数でなければ epoch ごとに key を再生成 | [確認:コード] `OFF/libDF/src/dataset.rs:164-167,586-588,1397-1414,1429-1431` |

### 3.2 公式が公開するデータ作成手順

[確認:コード]

- DNS4: `scripts/download_process_dns4.sh:22-75`。blob ごとに、名前に `noise` を含めば vorbis の `noise`、`impulse_responses` を含めば flac の `rir`、それ以外は flac の `speech`。`dev_testset` を含む blob は `VALID`、他は `TRAIN`。トップ階層のディレクトリごとに `<dir>_<SPLIT>.hdf5`。`TEST` と `dataset.cfg` は作らない。
- HDF5 化: `df/scripts/prepare_data.py`。属性 `sr`・`max_freq`・`dtype`・`codec`・`db_id`、key は相対パスの `/` を `_` に置換（`:84`）、`n_samples` 属性（`:89`）、`--sr` への再サンプルは `kaiser_best`（`:125`）、`--mono` は平均（`:129`）。短すぎる音声は警告のみ（`:78-79`。コメントは 100 ms だが式は `sr/100`＝10 ms）。
- 分割: `df/scripts/split_hdf5.py:35-60`。`TRAIN.hdf5` を 70/15/15 にファイル単位で分ける。`np.random.permutation` は seed 無し。話者単位ではない。
- 無音除去: `df/scripts/trim_silence_hdf5.py:31-54`（窓 sr/10、hop sr/20、先頭 −120 dB・末尾 −100 dB を基準）。DFN3 で使われたかは [UNKNOWN]。
- `dataset.cfg`: `{"train":[[file,factor,...]],"valid":...,"test":...}`（`OFF/README.md:213-260`、`dataset.rs:234`）。`dataset.cfg` 自体に HDF5 の key キャッシュとハッシュが書き戻される（`pyDF-data/src/lib.rs:119,186`）。

### 3.3 DFN3 の学習データの確定状況

| 項目 | 状態 |
|---|---|
| clean（speech）の集合・件数・時間 | [UNKNOWN]。公開されていない |
| 雑音の集合・件数 | [UNKNOWN] |
| RIR の集合・件数 | [UNKNOWN]（出荷 ini は `p_reverb=0.1`。RIR があったことは確実） |
| 分割・件数 | [UNKNOWN]（DNS4 の `TRAIN`/`VALID` blob 分けは手順で分かる。test は VoiceBank/DEMAND 評価で使う） |
| VCTK の test 話者（p232、p257）が学習側の VCTK に入っていたか | [UNKNOWN] |
| 公開 `dataset.cfg`・ファイルリスト | 無い |

### 3.4 この環境にある素材（公式データの代替になるか）

[確認:ファイル。`ls`、`docs/db/CORPORA.md`]

| 役割 | あるもの | 公式との関係 |
|---|---|---|
| clean | LibriSpeech、LibriTTS、VCTK（48 kHz）、VoiceBank 28/56 話者、DNS2020 `clean`、OpenSLR など | DNS4 の clean 全体と PTDB は無い。**公式データの再現ではなく代替** |
| 雑音 | DNS5 `noise_fullband`（63,810 ファイル）、DNS2020 `noise`、DEMAND、WHAM48、musan、MS-SNSD | DNS4 の雑音と同一とは言えない |
| RIR | DNS5 `impulse_responses`（SLR26、SLR28）、OpenSLR26 `sim_rir_16k` | 公式の RIR と同一とは言えない |
| 評価ペア | VoiceBank+DEMAND test 824 件（`clean_testset_wav`、`noisy_testset_wav`）。train の SNR は 0/5/10/15 dB、test は 2.5/7.5/12.5/17.5 dB（`log_trainset_28spk.txt`、`log_testset.txt` を集計） | 論文 Table 1 と同じデータ |

audio-db の学習データは `voicebank_28_train`（10,700 件・26 話者）と `voicebank_28_valid`（872 件・p286、p287）。test 話者 p232、p257 と重ならない [既存記録: `docs/db/CORPORA.md`、`docs/training/VERIFICATION.md:613`]。

## 4. C. データ生成・前処理

公式の学習サンプルは、次の順序で毎回生成される。処理の「いつ」は、起動時 / split の初回 / epoch / sample / batch / step で分ける。

| 順 | いつ | 処理 | 根拠 |
|---|---|---|---|
| 0 | 起動時 1 回 | main thread の RNG を `seed_from_u64(42)` で固定（設定の seed とは無関係）。`dataset.cfg` の key キャッシュ読み込み。HDF5 を開く | `pyDF-data/src/lib.rs:110,119` |
| 1 | split ごとの初回 `start_epoch`（および非整数 factor のとき毎 epoch） | `generate_keys`: RNG を `epoch_seed` で再設定、HDF5 ごとに（train のみ）key を shuffle し、`cycle().take(round(n×factor))` で speech / noise / RIR の key 列を作る | `dataloader.rs:309-326`、`dataset.rs:1397-1451` |
| 2 | epoch ごと | 発話 index `0..ds_len` を main RNG で shuffle（train/valid/test とも）。この RNG は epoch ごとに再設定されない | `dataloader.rs:335-342` |
| 3 | sample ごと | worker が `seed_from_u64(self.seed + seed)`。train は `seed = epoch_seed + sample_idx`、valid/test は `sample_idx`。以降の乱数はすべてこの thread-local RNG | `dataloader.rs:270-278`、`dataset.rs:1214-1216` |
| 3a | sample | SNR を `snrs` から、gain を `gains` から一様に 1 つずつ選ぶ | `dataset.rs:1218-1219` |
| 3b | sample | `snr <= -100` なら speech は零（1, max_samples）で、残りは雑音のみの学習。そうでなければ `load_aug_speech` | `dataset.rs:1227-1239` |
| 3c | sample | `load_aug_speech`: `sp_keys[idx]` を最大 `1.1×max_samples − cur_len` だけ読み（元が長ければランダムな開始位置）、先頭チャンネルのみ、HDF5 の sr が違えば 48 kHz へ再サンプル。RMS が 1e-10 未満なら別 key を引き直す。speech augmentation（DC 除去 0.25、LFilt 0.25、Biquad 0、Resample 0.1）。長さが `max_samples` に届くまで `sp_keys` から**ランダムに別発話を選んで連結**。超えた分は `uniform(0, len−max)` で切り出す | `dataset.rs:1100-1175`、`_read_from_hdf5` `:972-1016`、augmentation `:644-659` |
| 3d | sample | 速度・ピッチ変更（`RandResample`）は speech と noise の両方で確率 0.1、倍率 0.9〜1.1（500 Hz 単位に丸め）| `dataset.rs:655-658,700-703`、`augmentations.rs`（`RandResample`） |
| 3e | sample | 雑音の本数 `n_noises = uniform(2, 6)`（2〜5）。各雑音は `load_aug_noise`: train のみ 5% で合成雑音（べき乗スペクトルの色付き雑音、傾き `uniform(-2,2)`、gain は {−24,−12,−6,0} から）。それ以外は雑音 key を一様に選び、長さ 100 未満・最大絶対値 1e-10 未満は引き直し。noise augmentation（LFilt 0.25、Biquad 0、Resample 0.1、train のみ Clipping 0.1）。`max_samples` に切り詰め。雑音ごとの gain を `gains` から選ぶ | `dataset.rs:1177-1207,692-710,720-730,1254-1261`、`augmentations.rs:730-808` |
| 3f | sample | `combine_noises`: 雑音が speech より短ければ自身を連結して倍にしていき、長ければランダム開始で切り出す。チャンネル数を合わせる。各雑音に `10^(g/20)` を掛けて**加算**（`/ ch` のみで、本数での平均は取らない） | `dataset.rs:1979-2023` |
| 3g | sample | 残響（RIR key があるときのみ）: speech と noise にそれぞれ独立に確率 p で適用。1 本の RIR を共有。RIR を確率 p で 0.8〜1.2 倍に再サンプル（500 Hz 単位）、確率 max(p, 0.5) で rt60 `uniform(0.2,1)` の減衰、−80 dB でトリム、L2 正規化。ノイジー用 speech は全残響。**ターゲット**は 0.3×乾いた speech ＋ 0.7×（後期反射を `max_idx + 20 ms` 以降 rt60 0.5 で減衰した RIR を畳み込んだもの）で、RMS を元の speech に戻す | `augmentations.rs:810-1066`、`dataset.rs:711-718,1271-1283` |
| 3h | sample | 時間領域の歪み・帯域制限・周波数領域の歪み・割り込み話者。出荷 ini では p_clipping、p_zeroing、p_air_absorption、p_bandwidth_ext、p_interfer_sp がすべて 0 のため、実質何もしない | `dataset.rs:1285-1346`。ini は §6 |
| 3i | sample | `mix_audio_signal`: `g = 10^(gain/20)` を clean に掛け、`noise *= mix_f(clean_out, noise, snr)`（全区間の二乗和の比。無音区間を含み、VAD や P.56 は無い）。混合は `clean_mix + noise`。`clean_out`、`noise`、`mixture` の最大絶対値が 1 を超えたら、3 つを `1/(max+1e-10)` でまとめて縮小 | `dataset.rs:2047-2074`、`transforms.rs:58-63` |
| 3j | sample | `FftDataset`: `DFState` で STFT（speech は reset 無し、noisy は reset 有り）。ERB 特徴は `erb_norm`、DF 特徴は `unit_norm`（減衰 alpha は `norm_tau=1` から）。フレーム数は `ceil(T/hop)`＝3 秒で 300 | `dataset.rs:863-919`、`transforms.rs:156-189` |
| 4 | batch ごと | worker の出力を順序 index で並べ直す。最後の端数 batch も返す（`drop_last` は未指定で偽）。`collate` は `max_sample_len`（フレーム）へ揃える | `dataloader.rs:363-445`、`torch_dataloader.py:76` |
| 5 | step ごと | lr・weight decay の表を引く（`it = epoch × loader.len + i`）。`zero_grad` → forward → loss → backward → `clip_grad_norm_(1.0, error_if_nonfinite=True)` → `step` | `train.py:352-420` |

注意:

- 3b の雑音のみのサンプル（`snr=-100`）では clean が零で、`mix_f` の式は `e_clean`（1e-10 の下駄）と 10^(−snr/10) の積を雑音エネルギーで割る形になる。このため雑音は全区間の二乗和が約 1 になるよう規格化される（3 秒で RMS 約 −52 dBFS）。[導出] `transforms.rs:58-63` の式を読んだのみで、実行していない。
- `p_fill_speech` / `p_sample_full_speech(1.0)` は保存されるだけで使われない（`dataset.rs:641,753,958`、`pyDF-data/src/lib.rs:172`）。連結による埋めは**常に**行われる [確認:コード]。
- augmentation（3c〜3d の speech 側、3f の gain、残響、−100 dB の雑音のみ）は valid/test にも適用される。無効なのは train 限定の項目（合成雑音 5%、noise の Clipping、TD/FD 歪み）だけ [確認:コード `dataset.rs:662-691,705-710,720-730`、`lib.rs:162-171`]。
- 「各 sample の内容」は `(self.seed, epoch_seed, sample_idx, key 列, 環境変数)` の関数である。worker の並びには依存しない [導出: 各 sample の冒頭で thread-local RNG を再設定している]。
- 「どの sample がどの batch に入るか」は main RNG の shuffle による。この RNG は起動時に 42 で初期化され、train・valid の `start_epoch` で順に消費される。resume すると先頭から始まるため、順序は連続側と違う [導出]。
- 環境変数（`DF_P_REMVOE_DC`、`DF_P_LFILT`、`DF_P_BIQUAD`、`DF_P_RESAMPLE`、`DF_P_CLIPPING`、`DF_P_CLIPPING_NOISE`、`DF_P_ZEROING`、`DF_P_AIR_AUG`、`DF_P_NOISE_GEN`、`DF_REVERB_DRR`、`DF_REVERB_RT60`、`DF_REVERB_OFFSET_LATE`）が実効値を変える（`dataset.rs:644-730`）。学習時に設定されていたかは [UNKNOWN]。

### 4.1 audio-db の現行の前処理（読み取り結果）

[確認:コード]

1. `RecipeDataset._load`: レシピを `render_recipe` で描画（`src/training/data/dataset.py:181-221`）。VoiceBank はレシピの `output.sample_rate` と学習 `sampling_rate` が 48 kHz なので再サンプルは起きない（`_match_rate`、`:331-347`。差があれば `torchaudio.functional.resample` の既定）。
2. `rms_normalize` は既定が真（`dataset.py:70,215`、`config.py:376`）。`configs/training/deepfilternet3.yaml` の `data:` は指定していないので、noisy の RMS が 1 になるよう noisy と clean に同じ係数を掛ける（`_normalize`、`dataset.py:359-364`）。
3. 学習時のみ 144,000 サンプル（3 秒、48 kHz）を切り出す。開始位置は `mix_seed("crop", seed, epoch, case_id) % (length−segment+1)`（`src/training/data/sampler.py:37-46`）。短い発話は clean・noisy とも**末尾を零で埋める**（`_segment`、`dataset.py:367-380`）。
4. 並び順は `torch.randperm`（`mix_seed("order", seed, epoch)`）で epoch ごとに決まる（`sampler.py:15-34`）。
5. 検証は `segment_size=None`、`training=False` で発話全長を 1 件ずつ（`run.py:111-120`、`trainer.py:633-636`）。
6. 特徴は `DFStream`: 各発話で `reset` し、末尾に 960 サンプルの零を足してから解析する（`src/training/se/deepfilternet.py:309-315`）。フレーム数は 3 秒で 302（[導出]）。公式は 300。

## 5. D. Dataset / DataLoader / Sampler

### 5.1 公式

| 項目 | 実効動作 | 根拠 |
|---|---|---|
| データセットの長さ | `ds_len = Σ_{speech HDF5} max(round(keys × sampling_factor), 1)`。雑音と RIR は含まない。`global_ds_sampling_f`（出荷 1）を factor に掛ける | `dataset.rs:586-606` |
| sampler | `0..ds_len` を main RNG で shuffle。epoch ごとの再設定は無い | `dataloader.rs:339-340` |
| sample と key | 各 index が `sp_keys[idx]` の発話を先頭に使う。train の key 列は初回の `generate_keys` で shuffle され、factor が整数なら以後固定。そのため整数 factor では各 epoch で全 key がちょうど 1 回、先頭に使われる | `dataset.rs:1100-1101,1397-1431` |
| batch | `drop_last` は偽。端数 batch を返す。`dataloader_len = ceil(ds_len / bs)` | `dataloader.rs:212-219`、`torch_dataloader.py:76` |
| worker | `num_workers`（出荷 16）の rayon thread。先読みは `batch_size × num_workers` | `torch_dataloader.py:101`、`dataloader.rs:259-263` |
| batch size | train は `batch_size_train`（schedule で epoch ごとに `set_batch_size`）、valid/test は `batch_size_eval`（出荷 64） | `train.py:233-246`、`dataloader.rs:221-227` |
| 分散 | `train.py` に DDP・分散処理は無い（単一プロセス） | `rg DistributedDataParallel train.py` が空 |
| 最後の batch | 端数を含むため、最後の batch は小さくなる | `dataloader.rs:430-445` |

### 5.2 epoch の定義の表（4 つの意味を混ぜない）

| 定義 | 公式（config の値） | 公式（実効） | audio-db 現行 |
|---|---|---|---|
| 発話 | `ds_len`（speech key 数×factor） | 各 epoch で整数 factor なら全 key が 1 回ずつ先頭になる。ただし 1 sample 内で連結される別発話が加わる（[導出] 3 秒に満たない先頭発話のとき） | VoiceBank 10,700 件が 1 回ずつ |
| chunk（切り出し） | 無い（1 sample＝1 つの 3 秒窓） | 同左 | 発話ごとに 144,000 サンプルの 1 窓。`chunk_plan.py` は TF-GridNet 用で DFN3 では使わない [確認:コード `trainer.py:348-349`] |
| batch | `ceil(ds_len / bs)` | 端数を含む | `len(indices) // micro`（端数を捨てる）（`trainer.py:256-257`） |
| optimizer 更新 | batch 数と同じ（蓄積なし） | 同左 | `usable_batches // accumulation`（蓄積 1 なので batch 数と同じ） |

audio-db の VoiceBank 10,700 件で、公式 schedule（`min(schedule, 64)`）を当てた場合 [導出]:

| epoch | 実効 batch | 公式 batch 数/epoch（ceil） | audio-db（floor） | 差 |
|---|---|---|---|---|
| 0〜1 | 16 | 669 | 668 | +1 |
| 2〜4 | 24 | 446 | 445 | +1 |
| 5〜9 | 32 | 335 | 334 | +1 |
| 10〜119 | 64（128 と 9999 は 64 に切り上げ制限） | 168 | 167 | +1 |
| 120 epoch 合計 | | 2×669 + 3×446 + 5×335 + 110×168 = **22,831** | 2×668 + 3×445 + 5×334 + 110×167 = **22,711** | 120 |

`configs/training/deepfilternet3.yaml:27` の `max_steps: 22711` は後者（floor）[確認:コード]。

lr・weight decay の表の長さは `max_epochs × len(dataloader)` で、`len` は現在の batch size での値。batch size が変わる epoch で表を作り直し、`it = epoch × loader.len + i` で引く（`train.py:194-195,233-246,352-362`）。結果として lr は小数 epoch `epoch + i/len` の関数で、batch size に依らない。audio-db は `retarget_deepfilter_schedule` と `note_schedule_position` で同じ構造を再現している（`deepfilternet.py:187-209`、`trainer.py:273,278,559-567`）。

## 6. E. モデル・学習条件

### 6.1 config の値と実効値

出荷 `config.ini` の値と、コードの既定・ハードコードを分ける。[確認:ファイル/コード]

| 項目 | config の値 | 実効値・補足 | 根拠 |
|---|---|---|---|
| seed | `[train] seed = 43` | コードの既定は 42（`train.py:92`）。`check_manual_seed` が `np`・`random`・`torch` を設定（`utils.py:130-136`）。dataloader の `seed` にも渡る（`train.py:161`）。main RNG の 42 は別（`lib.rs:110`） | |
| batch | `batch_size = 64`、`batch_size_eval = 64` | train の実効は `min(schedule, 64)`: 16→24→32→64。valid/test は 64 | `train.py:135-136,236-239` |
| batch schedule | `0/16,2/24,5/32,10/64,20/128,40/9999` | 128 と 9999 は `min(b, 64)` で 64 | `train.py:176-183,239` |
| max_epochs | 120 | `model_120.ckpt.best`＝最終 epoch が best（検証損失が最小）。早期停止には至らなかった [導出: ファイル名] | |
| early stopping | `early_stopping_patience = 25`（既定は 5） | 検証損失が最良を更新しない epoch が 25 回続いたら停止（`check_patience`、`checkpoint.py:156-171`）。`.patience` ファイルに状態を保存 | |
| validation | `validation_criteria = loss`（rule は既定 `min`） | `write_cp(..., metric=val_loss, cmp="min")` で best を更新 | `train.py:206-211,285-287` |
| 3 秒窓 | `max_sample_len_s = 3.0`（既定は 5.0） | `round(3.0×48000)=144,000`。valid/test にも同じ窓（`build_td_dataset` が共通） | `train.py:147`、`dataset.rs:545` |
| SNR | `dataloader_snrs = -100,-5,0,5,10,20,40` | 7 値の一様選択。−100 は雑音のみ（`dataset.rs:1227`）。valid/test にも同じリスト（`lib.rs:153-155`、clone 前） | |
| gain | ini に無し | 既定 `[-6, 0, 6]` | `train.py:166` |
| 残響 | `[distortion] p_reverb = 0.1`（コード既定 0.2） | `RandReverbSim::new(0.1)`: 適用確率が speech・noise とも 0.1、RIR 再サンプル 0.1、減衰 0.5（`max(p,0.5)`）、DRR 0.3、rt60 0.5、遅延 20 ms | `train.py:153`、`dataset.rs:715-718`、`augmentations.rs:1051-1065` |
| その他の歪み | `p_bandwidth_ext = p_clipping = p_air_absorption = p_zeroing = p_interfer_sp = 0` | 実質無効 | |
| DF | `sr 48000`、`fft 960`、`hop 480`、`nb_erb 32`、`nb_df 96`、`norm_tau 1`、`lsnr_max 35`、`lsnr_min -15`、`min_nb_erb_freqs 2`、`pad_mode output`、`df_order 5`、`df_lookahead 2` | `config.py:15-39` の既定は `df_lookahead 0`、`pad_mode input`。出荷値が実効 | |
| モデル | `conv_lookahead 2`、`conv_ch 64`、`emb_hidden_dim 256`、`emb_num_layers 3`、`enc_linear_groups 32`、`linear_groups 16`、`df_hidden_dim 256`、`df_num_layers 2`、ほか | コードの既定（`deepfilternet3.py:30-77`）は `conv_lookahead 0`、`conv_ch 16`、`emb_num_layers 2`、`df_num_layers 3` など。出荷値が実効 | |
| 損失 | `[multiresspecloss] factor 500, factor_complex 500, gamma 0.3, fft_sizes 256,512,1024,2048`、`[localsnrloss] factor 1e-3`、他（mask、spectral、sdr、asr）は 0 | `MultiResSpecLoss`: `Stft(n_fft, hop=n_fft/4, hann, normalized=True)`、`clamp_min(1e-12).pow(gamma)`、**MSE**×factor と、複素項の MSE×factor_complex（`loss.py:95-135`）。局所 SNR は `LocalSnrTarget(ws=20, [lsnr_min−1, lsnr_max+1])` と MSE（`loss.py:666,771,787`） | |
| optimizer | `optimizer = adamw`、`lr = 1e-3`、`weight_decay = 1e-12` | `AdamW(betas=[0.9,0.999], amsgrad=True)`（`amsgrad` はハードコード、`train.py:495`）。`weight_decay_end = 0.01` のとき、wd は 1e-12 から 0.01 へ cosine で増加する表を毎 step 書き込む（正の wd のときのみ、`train.py:361-362,539-553`） | |
| lr schedule | `lr_min 1e-6`、`lr_warmup 1e-4`、`warmup_epochs 3` | `np.linspace(lr_warmup, lr, 3×len)` の後に 1 周期の cosine（`lr.py:6-60`、`train.py:515-536`）。1 step ごとに更新 | |
| 勾配クリップ | ini に無し | `clip_grad_norm_(…, 1.0, error_if_nonfinite=True)` をハードコード（`train.py:394`） | |
| 非有限値 | | 損失が NaN のとき batch を飛ばす。最大 50 回（`MAX_NANS`、`train.py:43,383-390,413-417`） | |
| 初期化 | ini に無し | 明示的な初期化は `GroupedLinearEinsum` の `kaiming_uniform_(a=√5)` のみ（`modules.py:763-764`）。他は PyTorch の既定。`torch.manual_seed(seed)` 後に構築 | |
| 精度・決定性 | ini に無し | AMP・`autocast`・`GradScaler`・`deterministic` の設定が無い（`rg` が空）。fp32。`model.to(memory_format=torch.channels_last)`（`model.py:23`） | |
| checkpoint | `n_checkpoint_history`（既定 3）、`n_best_checkpoint_history`（既定 5） | epoch ごとに model・opt を保存。best は検証損失が `prev_best` を厳密に下回ったときのみ | `checkpoint.py:107-134` |
| resume | | 最新の epoch から再開。scheduler の状態は保存されず、表を作り直す。`.patience` は保持。shuffle の RNG は 42 に戻る | `train.py:188-197,233` |
| test | | 学習後に best の checkpoint を読み、`test` split を 42 で評価（`iter_epoch("test", 42)`）。ただしこの test も 3 秒のオンライン混合 | `train.py:301-320` |

### 6.2 audio-db の現行設定（読み取り結果）

`configs/training/deepfilternet3.yaml` [確認:コード]:

| 項目 | 値 |
|---|---|
| seed / deterministic | 42 / false |
| batch | `global_batch_size 64`、`micro_batch_size_per_gpu 64`、`batch_size_limit 64`、`batch_schedule` は公式と同じ 6 組 |
| max_steps | 22711 |
| segment | 144000 |
| loss / grad_clip | `official` / 1.0 |
| optimizer | AdamW、lr 1e-3、betas [0.9, 0.999]、weight_decay 1e-12、amsgrad true |
| scheduler | `deepfilter_cosine`、`steps_per_epoch 668`（初期表。epoch ごとに作り直す）、`epochs 120`、`lr_min 1e-6`、`lr_warmup 1e-4`、`warmup_epochs 3`、`weight_decay_end 0.01` |
| early stopping | patience 25、`best_checkpoint_start_epoch: 0`（コード既定は 40、`config.py:105`） |
| データ | `data/db/voicebank.duckdb` の `voicebank_28_train` / `voicebank_28_valid`、`load_workers 4`、`rms_normalize` は未指定＝真 |

## 7. F. Validation / Test / 評価

### 7.1 学習中の検証と test

| 項目 | 公式 | audio-db |
|---|---|---|
| データ | `valid` split の HDF5 から、train と同じ生成器（3 秒窓、SNR・gain・残響を含む）。seed は `self.seed + sample_idx` で毎回同じ | `voicebank_28_valid` の 872 発話を全長のまま |
| 前処理 | 学習と同じ（augmentation も適用）。合成雑音 5%、noise の Clipping、TD/FD 歪みだけ train 限定 | RMS 正規化のみ（切り出しなし） |
| batch | 64（最後は端数） | 1 発話ずつ |
| 集計 | `torch.stack(l_mem).mean()`＝batch ごとの損失の非加重平均（`train.py:457`） | 発話ごとの損失の非加重平均（`trainer.py:622-650`、`_combine_metric`） |
| モード | `model.train(False)`、`set_grad_enabled(False)` | `eval_mode` |
| best / patience | `loss` の最小で更新。`check_patience` は `raise_=False` で、`False` なら epoch ループを抜ける（`train.py:285-296`） | `_consider_best` と `_batch_schedule_should_stop`（`trainer.py:527-542,737-760`）。判定は同じ向き |
| test | 学習後に best を読んで `test` split を評価 | 無い |

### 7.2 論文 Table 1 と公式スクリプトと `src/evaluation` の比較

| 項目 | 論文 Table 1 | 公式スクリプト（`test_voicebank_demand.py` → `evaluation_utils.py`） | `src/evaluation`（audio-db） |
|---|---|---|---|
| 値 | PESQ 3.17、CSIG 4.34、CBAK 3.61、COVL 3.77、STOI 0.944（DFN2: 3.08/4.30/3.40/3.699/0.9429、DFN1: 2.81/4.14/3.31/3.46/0.942）[確認:論文] | 同左の指標を出力。SI-SDR も出す | 論文値の再現実行は無い（`data/results/se-evaluation/` に DFN3 の記録は確認できない）[UNKNOWN] |
| データ | VoiceBank+DEMAND test | `clean_testset_wav`・`noisy_testset_wav` を `sorted(glob)` で組む（`test_voicebank_demand.py:26-27`）。824 件 | `voicebank_demand_test`（DB の名前付き評価セット）。評価時に Manifest は読まない |
| 入力の sr | 論文に無し | 48 kHz で読む。`sinc_fast`（torchaudio `sinc_interp_hann`、`lowpass_filter_width=16`）（`evaluation_utils.py:32,116-117`、`io.py:95`） | VoiceBank は 48 kHz のためモデル入力で再サンプルは起きない [既存記録]。評価の再サンプルは `torchaudio.functional.resample` の既定（`pesq.py:134`、`stoi.py:96`） |
| 推論 | 論文に無し | `enhance`（`evaluation_utils.py:63-75`）は**末尾パディングなし**で全長を解析・合成し、STFT 遅延（480 サンプル）を補正しない | `DFEnhancer._enhance_native`（`inference.py:125-134`）は末尾に 960 を足し、遅延 480 を切って原信号に揃える |
| 参照の扱い | | clean も `synthesis(analysis(clean))` を通し、enhanced と同じ遅延・長さにする（`:120`）。長さ不一致は assert（`:320`） | 参照は元の clean。長さが違えば短い方に切る（`_prepare`）。遅延の補正はしない |
| PESQ | 版・モードの記載なし | 48→16 kHz 後に `pesq(16000, clean, deg, "wb")`。`composite` 経由でも同じ（`sepm.py:499`） | 16 kHz なら `wb`。48 kHz は 16 kHz へ再サンプルして `wb`。長さ境界の判定で `na` にすることがある（`pesq.py:68-93`）。`na` は 0 にせず件数で表す |
| STOI | | 48→10 kHz を `sinc_fast` で 1 段、`pystoi.stoi(..., fs_sig=10000, extended=False)`（`evaluation_utils.py:557-563`） | 48→16 kHz を 1 段、`pystoi` が内部で 10 kHz へ再サンプル（`pystoi/stoi.py:52-55`）。2 段の再サンプル |
| SI-SDR | | `si_sdr_speechmetrics`（`evaluation_utils.py:599-619`）。平均除去なし、48 kHz のまま | `score_si_sdr`（`sisdr.py`）。平均を除去する。別定義 |
| CSIG / CBAK / COVL / SSNR | 報告あり | `composite`（`sepm.py:490-510`）。PESQ-wb、LLR、WSS、SegSNR から係数式で算出し、[1, 5] に制限 | **未実装** |
| 集計 | | ファイルごとの値の平均（`MPMetric.mean`、`:331-336`）。失敗ファイルは含まれない | 発話ごとに status を持つ。`na`・`unavailable` は 0 にしない |
| post-filter | 記載なし | 既定は無効（`init_df(post_filter=False)`、`args.pf`）。出荷 ini は `mask_pf` を持たない（既定 false） | `mask_pf` の既定は false |

- 論文 Table 1 の数値が、出荷モデル（2023-05-23）と同じ checkpoint から出たかは [UNKNOWN]。論文の投稿は 2023-05-14 で、出荷モデルの更新はその後である。
- 公式スクリプトの `PESQ` が論文の PESQ と同じ設定かは [UNKNOWN]（論文に記載がない）。
- WER は固定 ASR を使う別の実験で、PESQ/STOI/SI-SDR の公式再現とは混ぜない。DFN3 の公式側に WER はない。

## 8. 現行 audio-db との項目別比較表

判定: MATCH（一致）/ PARTIAL（一部一致）/ DIFF（不一致）/ UNKNOWN / N/A。audio-db 側は今回コードを読んで書いた。「検証」の語は既存記録の範囲を指し、今回は再実行していない。

| 項目 | 公式設定 | 公式実効動作 | audio-db現状 | 判定 | 根拠 |
|---|---|---|---|---|---|
| Dataset（train） | DNS4 全体＋PTDB/VCTK×10（論文） | `dataset.cfg` の HDF5 群。epoch 長は speech の key 数×factor。具体は非公開 | `voicebank_28_train` 10,700 固定ペア | DIFF | 論文 `dfn3.txt:115`、`dataset.rs:603`、`deepfilternet3.yaml:60` |
| Dataset（valid/test） | DNS4 `dev_testset` 等（手順）。test は評価時 VoiceBank | valid は seed 固定のオンライン混合 | valid は `voicebank_28_valid` 872 件、test は無し | DIFF | `download_process_dns4.sh:46-50`、`dataloader.rs:276` |
| 混合方式 | オンライン | 発話ごとに seed 付きで生成 | 固定ペア（レシピの描画） | DIFF | `dataset.rs:1211-1379`、`dataset.py:181-221` |
| 雑音選択 | 雑音 key を一様に 2〜5 本、各雑音に gain（{-6,0,6}）、5% は合成雑音 | `load_aug_noise`、`combine_noises` で加算 | 録音済みペアに入った雑音のみ | DIFF | `dataset.rs:1254-1263,1177-1207` |
| SNR | {-100,-5,0,5,10,20,40} dB の一様選択 | 全区間の二乗和比（無音を含む）。−100 は雑音のみ | VoiceBank 収録の SNR（train 0/5/10/15 dB） | DIFF | `dataset.rs:1218,2060`、`transforms.rs:58-63`、`log_trainset_28spk.txt` |
| gain | {-6,0,6} dB を clean と mixture に掛ける | 混合の絶対レベルが ±6 dB 動く（SNR は不変） | 無し | DIFF | `dataset.rs:2055-2058` |
| 残響・RIR | `p_reverb 0.1`。speech と noise に独立に適用。RIR 再サンプル・減衰・−80 dB トリム・L2 正規化。ターゲットは 0.3 乾き＋0.7 後期減衰 | `RandReverbSim` | 無し | DIFF | `augmentations.rs:925-1027` |
| 再サンプル | HDF5 を 48 kHz で作成（`kaiser_best`）。読み込み時に sr が違えば再サンプル。速度変更 augmentation 0.1 | | VoiceBank は 48 kHz のままで再サンプルなし。差があれば `torchaudio` 既定 | PARTIAL | `prepare_data.py:125`、`dataset.py:331-347` |
| 正規化 | 発話 RMS 正規化なし。gain とクリップガードのみ（最大絶対値が 1 を超えたら 3 信号を縮小） | | 既定で noisy の RMS を 1 にする | DIFF | `dataset.rs:2063-2072`、`dataset.py:215,359-364`、`config.py:376` |
| crop | 3.0 s（144,000）。連結＋ランダム切り出し | `uniform(0, len−max)` | 144,000。開始位置は `case_id` のハッシュ | PARTIAL | `dataset.rs:1170-1173`、`sampler.py:37-46` |
| padding | 零埋めしない（連結で埋める）。noise-only は零音声 | | 短い発話は clean・noisy とも零埋め | DIFF | `dataset.rs:1112-1166`、`dataset.py:367-380` |
| 長さ上限 | `max_sample_len_s 3.0`、valid/test にも同じ | | train のみ 144,000。valid は全長 | DIFF | `dataset.rs:545`、`run.py:111-120` |
| shuffle / sampler | main RNG で shuffle（epoch 間で再設定なし） | resume で順序が変わる | `randperm(mix_seed(seed, epoch))` | PARTIAL | `dataloader.rs:339-340`、`sampler.py:28-30` |
| epoch 定義 | speech key 数×factor 発話。batch は ceil | 端数 batch を残す | 10,700 発話。batch は floor | DIFF | `dataloader.rs:212-219`、`trainer.py:256-257` |
| batch | schedule `min(b,64)`。valid は 64 | | schedule は同じ。valid は 1 発話 | PARTIAL | `train.py:233-246`、`deepfilternet3.yaml:20-26` |
| DDP / 蓄積 | 単一プロセス、蓄積なし | | schedule は 1 プロセスのみ許可、蓄積 1 | MATCH | `trainer.py:548`、`train.py`（分散なし） |
| 損失 | MR spectral 500/500、γ0.3、FFT 256〜2048、局所 SNR 1e-3 | MSE。局所 SNR は ±1 クランプ | 同じ定数・同じ式の読み取り結果。数値一致は既存記録 | MATCH | `deepfilternet.py:49-54,322-342,375-408`、`loss.py:95-135,666` |
| optimizer | AdamW、lr 1e-3、wd 1e-12、betas(0.9,0.999)、amsgrad | | 同じ | MATCH | `optim.py:20-29`、`train.py:495` |
| scheduler（lr・wd） | per-step。warmup 3 epoch＋cosine。wd は 1e-12→0.01 | 表は loader 長で作り直し | 同じ式と作り直し。長さは floor | PARTIAL | `deepfilternet.py:57-84,187-209`、`lr.py:6-60` |
| 勾配クリップ | 1.0（`error_if_nonfinite`） | NaN は 50 回まで skip | 1.0。非有限は例外で停止 | PARTIAL | `train.py:394,383-390`、`optim.py:191-194`、`deepfilternet.py:411-413` |
| 初期化 | PyTorch 既定＋`kaiming_uniform_` 1 か所 | seed 43 | 同じ規則（独立実装）。値は一致しない（seed と構築順） | PARTIAL | `modules.py:763-764`、`model.py:114`、`rng.py:54-75` |
| 最大 epoch / step | 120 epoch（早期停止あり） | 更新回数は ceil 基準 | `max_steps 22711`（floor 基準） | PARTIAL | `deepfilternet3.yaml:27` |
| validation データ | オンライン混合（3 秒、固定 seed） | | VoiceBank 発話 全長 | DIFF | §7.1 |
| validation 前処理 | 学習と同じ | | RMS のみ | DIFF | §7.1 |
| validation batch | 64。batch 損失の平均 | | 1 発話。非加重平均 | DIFF | `train.py:457`、`trainer.py:622-650` |
| best / early stop | valid loss 最小、patience 25 | | 同じ向きと値。best は `best_generator.pt` | MATCH | `trainer.py:527-542,737-760` |
| resume | epoch 単位、scheduler は表の再構築、RNG は 42 に戻る | | step 単位、RNG・scheduler・patience を保存 | DIFF（上位互換） | `trainer.py:591-620` |
| 指標 PESQ | wb、48→16 kHz `sinc_fast` | | wb、`torchaudio` 既定。`na` あり | PARTIAL | §7.2 |
| 指標 STOI | 48→10 kHz 1 段 | | 48→16 kHz→10 kHz の 2 段 | PARTIAL | §7.2 |
| 指標 SI-SDR | speechmetrics 型、平均除去なし | | 平均除去あり | DIFF | `evaluation_utils.py:599-619`、`sisdr.py` |
| 指標 CSIG/CBAK/COVL | 報告あり | | 未実装 | DIFF | `sepm.py:490-510` |
| test | 学習後に best で `test` split | | 無し | DIFF | `train.py:301-320` |
| 特徴フレーム数 | `ceil(T/hop)`＝300 | | 末尾に 960 を足して 302 | DIFF | `transforms.rs:156-189`、`deepfilternet.py:311` |
| 精度 | fp32、TF32 は PyTorch 既定 | | 非決定モードで matmul の TF32 を有効にする | DIFF | `rng.py:39-40` |

## 9. 差異一覧

影響度: Critical（プロトコルの再現にならない）/ Major（学習結果の分布が変わり得る）/ Minor / Unknown。実装状態: 未実装 / 実装済み公式一致未検証 / 実装済み検証済み（範囲明記）。優先度: P0 / P1 / P2。修正は提案のみで、今回は実装していない。

| ID | 項目 | 影響 | 実装状態 | 優先度 | 対象ファイル | 修正案（提案） |
|---|---|---|---|---|---|---|
| DFN-D01 | 学習データ分布（DNS4＋PTDB/VCTK×10 と VoiceBank 固定ペア） | Critical | 未実装（素材の一部は無い） | P0 | DB スキーマ（新規）、`src/training/data/` | 代替素材（DNS5 雑音・RIR、LibriSpeech/VCTK）での「公式形式のプロトコル」を別名で用意し、公式データの再現とは書き分ける |
| DFN-D02 | オンライン混合（SNR 7 値、gain、2〜5 本の雑音、合成雑音 5%、各種 augmentation、clip ガード） | Critical | 未実装 | P0 | 新規 loader、`src/training/data/dataset.py` | §10 のプロトコル層を作り、公式専用の実装に分ける |
| DFN-D03 | 残響（`p_reverb 0.1`、RIR 再サンプル・減衰・DRR 0.3 のターゲット） | Major | 未実装 | P0 | 同上 | RIR を役割として DB に登録し、`RandReverbSim` 相当を専用実装 |
| DFN-D04 | 発話 RMS 正規化（audio-db 既定で有効、公式に無い） | Major | 実装済み（公式と別仕様） | P0 | `configs/training/deepfilternet3.yaml`（`data.rms_normalize: false` の指定）、`src/training/data/dataset.py` | 固定ペアで続けるなら、少なくとも `rms_normalize: false` と gain の導入を検討 |
| DFN-D05 | 3 秒窓の作り方（連結 vs 零埋め） | Major | 未実装 | P0 | `src/training/data/dataset.py:367-380` | 短い発話を零埋めせず、別発話の連結または除外にする。VoiceBank の 3 秒未満の割合は未測定 [UNKNOWN] |
| DFN-D06 | 検証（3 秒窓のオンライン混合 batch 平均 vs 発話全長の非加重平均） | Major | 実装済み（公式と別仕様） | P1 | `src/training/core/trainer.py`、`deepfilternet.py:246-267` | 公式型の検証セット（固定 seed の 3 秒窓）を別に持つ |
| DFN-D07 | 端数 batch（ceil vs floor）と `max_steps` | Minor | 実装済み（公式と別仕様） | P2 | `trainer.py:256-257`、`deepfilternet3.yaml:27` | 端数を残す設定を持てば `max_steps` は 22,831。今は 22,711 と明記済み |
| DFN-D08 | shuffle と resume の乱数（main RNG の持ち越し） | Minor | 実装済み（別仕様） | P2 | `src/training/data/sampler.py` | 分布同等であれば足りる。順序の完全一致は求めない |
| DFN-D09 | 学習後の test と checkpoint 運用（直近 3＋best 5 の保持） | Minor | 未実装 | P2 | `trainer.py` | 評価は §7.2 の評価基盤で別に行う |
| DFN-D10 | 特徴のフレーム数（300 と 302）と 960 サンプルの零パディング | Minor | 実装済み公式一致未検証 | P2 | `deepfilternet.py:309-315` | 3 秒窓での影響を測る。損失の正規化長が変わる |
| DFN-D11 | 評価の差（再サンプル法、STOI の 2 段、SI-SDR の定義、CSIG/CBAK/COVL 未実装、遅延補正の違い、論文値の再現未実施） | Major | 未実装（一部は別仕様） | P1 | `src/evaluation/metrics/*`、評価設定 | 公式型の評価（clean の STFT 往復、`sinc_fast`、composite）を別名で追加 |
| DFN-D12 | 非有限値の扱い（公式は最大 50 回 skip、audio-db は停止） | Minor | 実装済み（別仕様） | P2 | `deepfilternet.py:411-413` | 方針を文書化 |
| DFN-D13 | TF32（非決定モードで matmul の TF32 を有効にする） | Minor | 実装済み（別仕様） | P2 | `src/training/core/rng.py:39-40` | 公式型の実行では無効にする |
| DFN-D14 | 初期重みの乱数列（seed 43 と 42 で別、構築順も別） | Minor | N/A（一致させる意味がない） | P2 | | 分布が同じであれば足りる |
| DFN-D15 | 固定 commit と学習時コードの差（§2.3） | Unknown | N/A | P1 | | 学習時の commit 特定を試みる（§12 U02） |
| DFN-D16 | valid/test にも augmentation と −100 dB が入る点の扱い | Major | 未実装 | P1 | 検証セット | D06 と合わせて設計 |
| DFN-D17 | 損失・optimizer・scheduler・batch schedule の数値一致 | Minor | 実装済み検証済み（既存記録: CPU の合成入力 20 step で公式/独立の損失・勾配・重みが一致、H100 の 1 step で勾配相対 L2 約 1e-7）。今回は再実行していない | P2 | `deepfilternet.py` | 長時間学習後の同等性は未判定 |

## 10. DB 中心の新データ基盤に必要な情報

公式は「DFN3 の学習データの具体」を公開していない。以下は、公式の挙動をオンライン混合として再現する場合に、DB・レシピ・実装が持つべき情報である。

### 10.1 DB に登録する音声素材

- 発話（speech）: パス、コーパス名、話者、言語、sr、チャンネル数、フレーム数、`max_freq`（HDF5 の属性相当）、ファイルのハッシュ、`sampling_factor`（オーバーサンプルの倍率）。
- 雑音（noise）: 同じ属性と、種別（実録・合成）。5% の合成雑音は素材ではなく生成器の設定である。
- RIR: パス、sr、チャンネル、実測／模擬、ピークの位置。
- 評価ペア: clean と noisy の対応、SNR、話者、文。
- コーデックの違い（pcm / flac / vorbis）は、元ファイルを使うか HDF5 を使うかでサンプル値が変わる。記録しておく。

### 10.2 役割

speech（ターゲットの元）、noise、rir、eval_noisy、eval_clean。VoiceBank の収録ペアは「固定ペア」の役割で、オンライン混合の素材ではない。

### 10.3 必要な split と対応

train / valid / test のそれぞれに、speech・noise・RIR の別々のリストが要る（公式は split ごとの HDF5 集合）。評価ペアは別に持つ。公式は話者排他を保証しないため（`split_hdf5.py`）、話者の重なりを記録する。

### 10.4 固定で保存する生成条件

- プロトコルのパラメータ: `snrs`、`gains`、`max_len_s`、`p_reverb`、各 augmentation の確率と範囲、雑音本数の範囲、合成雑音の確率と傾きの範囲、DRR・rt60・遅延、クリップガード。環境変数による上書きの値も含める。
- ファイルリストとその順序、`sampling_factor`、ハッシュ。
- 乱数の方式: 基準 seed と、sample ごとの seed の式。
- valid/test を実現値として固定するか、seed から毎回作るかの区別。

### 10.5 学習時に標本化する条件（確率分布のパラメータ）

SNR（7 値一様）、gain（3 値一様）、雑音本数（2〜5）、雑音 key（一様）、雑音ごとの gain、雑音の開始位置、speech の切り出し位置、連結に使う別発話、augmentation の適用有無と値、残響の適用有無（speech・noise で独立）、RIR の選択、RIR の再サンプル率、減衰の rt60、epoch の発話順。

### 10.6 dataloader 固有の処理

sample ごとの seed、worker 出力の順序再構成、`collate` の長さ揃え、端数 batch、batch size schedule に連動した lr 表の作り直し、STFT・ERB・unit-norm 特徴（`DFStream` に相当する処理は audio-db に既にある）。

### 10.7 共有できる処理

素材の索引とメタデータ、ファイルリストとハッシュの管理、プロトコルの保存、sample ごとの seed を作る関数、二乗和比による SNR 混合の核（`mix_snr` の `rms_v1` と `mix_f` は、イプシロンの扱いを除き同じ定義 [導出]）、読み込み・リサンプル・切り出しの部品、sample ごとの「実際に引いた値」の記録。

### 10.8 公式の挙動を保つのに専用実装が要る処理

- speech の連結による埋めと、その切り出し。
- 2〜5 本の雑音の加算と、雑音ごとの gain、短い雑音の倍化連結。
- 合成雑音（色付き雑音の生成）。
- `RandReverbSim` の全体（RIR の再サンプル・減衰・トリム・L2 正規化、speech の 0.3 乾き＋0.7 後期減衰ターゲット、RMS の合わせ直し、noise への独立適用）。
- speech の augmentation（DC 除去、LFilt、Resample）。
- `snr ≤ −100` の雑音のみの分岐と、その縮小。
- 3 信号まとめてのクリップガード（`1/(max+1e-10)`）。
- 局所 SNR の目標（`noisy − clean` から作る）。
- 設計判断が要る点: 公式の `libdfdata` を実行時に呼ぶと、独立実装が公式コードに実行時依存しない方針（`AGENTS.md`）に反する可能性がある。Python などの独立実装にする場合、Rust の `Xoshiro256++` と同じ乱数列は再現されない。分布同等で足りるか、親エージェントの判断が要る。

### 10.9 再現のために保存する状態

基準 seed と main RNG の初期 seed（42）、epoch（`epoch_seed`）、sample ごとの seed の式、train の key 列の順序（`generate_keys` の shuffle 結果）、ファイルリストのハッシュと `dataset.cfg`（key キャッシュの書き戻しを含む）、各 epoch の実効 batch size、lr・wd の表と現在位置、環境変数の上書き値、公式 commit とツールの版（`torch`、`torchaudio`、`pystoi`、`pesq`）、optimizer の状態、patience の状態、best の記録。sample ごとの内容は `(基準 seed, epoch_seed, sample_idx, key 列, 環境変数)` で決まるため、epoch と sample_idx を保存すれば再生成できる [導出]。

### 10.10 既存レシピ JSON v1 で表せないこと

判定: **公式のオンライン混合には不十分**。固定ケース 1 件の記録には使える。

v1 が持つもの [確認:コード/既存 docs: `src/evaluation/audio/render.py:93-134`、`mixing.py:66-147`、`docs/db/RECIPE.md`]: `sources`（clean / noise / rir / noisy）、`operations`（`additive_noise`、`rir`、`peak_normalize`）、`output`（sr、reference、`max_seconds`）。

表せないもの:

1. 確率分布（SNR の 7 値一様、gain の 3 値一様、雑音本数など）。レシピは実現値 1 つ。
2. 複数雑音の加算（`additive_noise` は 1 本）と、雑音ごとの gain。
3. 雑音のみのサンプル（`snr ≤ −100`、clean が零）。v1 は clean が無音だと混合せず返す。
4. speech の連結による埋めと、ランダム切り出し。v1 は `max_seconds` の先頭切り詰めのみ。
5. speech の gain（clean と mixture への ±6 dB）。
6. クリップガード（clean・noise・mixture の 3 つの最大で縮小、閾値 1.0）。v1 の `peak_normalize` は `limit` と観測信号の最大値のみ（既定は 0.99 前後）。
7. 残響の詳細: speech と noise への独立適用、RIR の再サンプル・減衰・−80 dB トリム・L2 正規化、`0.3 乾き＋0.7 後期減衰` のターゲット、RMS の合わせ直し。v1 の `rir` は単純な因果畳み込み（`trim_to_input`）で、clean と observed に同じカーネルを掛け、参照は `dry_clean` か `reverberant_clean` の二択。
8. 合成雑音（色付き雑音）。
9. speech・noise の augmentation（DC 除去、LFilt、Resample、Clipping）。
10. 再サンプルの方法（`kaiser_best`、`sinc_fast`）と、HDF5 のコーデック。v1 は `torchaudio` の既定のみ。
11. 複数チャンネルの扱い（公式は speech は先頭チャンネル、noise はランダムに除去・複製）。v1 は平均。
12. sample ごとの RNG（`epoch_seed + sample_idx`）と、epoch・順序の状態。v1 の `seed` は保存するだけで再計算しない。
13. `sampling_factor` と key 列の循環。
14. valid/test の「毎回同じ seed の再生成」という意味。
15. 局所 SNR の目標のような、損失側が使う副産物。
16. ERB・unit-norm 特徴（`DFStream` に相当）。

整理の提案: 「プロトコル」（分布とパラメータ、ファイルリスト、seed の式）と「ケース」（1 件の実現値、レシピ）を別の層にする。公式の valid を固定ケースへ落とすなら、上の 1〜9 を表せるレシピ v2 が要る。学習用のオンライン生成は、プロトコルから都度作り、実現値の記録だけを残す案がある。

## 11. 論文と公式コードの矛盾、既存ドキュメントの不整合・誤り

### 11.1 論文と公式コード・設定

| ID | 内容 | 確度 |
|---|---|---|
| C01 | DFN3 論文は学習条件を書かない。DFN2 論文の「100 epoch、batch 8→96、λ_spec=1e3・λ_MR=5e2」と、DFN3 出荷 ini の「120 epoch、batch 16→64（最大 64）、スペクトル損失 0・MR 500＋複素 500・局所 SNR 1e-3」は一致しない。DFN3 の学習設定を DFN2 論文から読んではならない | [確認:論文/ファイル] |
| C02 | DFN2 論文の式 (6) は二乗ノルムの比の形（`‖…‖₂ / ‖…‖₂`）。コードの `MultiResSpecLoss` は `mse_loss` | [確認:論文/コード] `loss.py:128,133` |
| C03 | DFN2 論文は「窓 5〜40 ms」。コードの FFT 256〜2048（48 kHz で約 5.3〜42.7 ms）と近いが一致の記述ではない | [導出] |
| C04 | 論文の DF 次数・先読みは「N=5、先読み 2」。出荷 ini は `df_order 5`、`df_lookahead 2`、`conv_lookahead 2`、`pad_mode output`。コードの既定は `df_lookahead 0`、`pad_mode input` で、出荷値が実効 | [確認:コード/ファイル] |
| C05 | README は `DeepFilterNet/pretrained_models/DeepFilterNet/config.ini` へのリンクを書くが、このファイルは `b1f8b04`（2022-05-26）以降の履歴に無い | [確認:ファイル] |
| C06 | `prepare_data.py:78` のコメントは「100 ms 以上」だが、式は `sr/100`（10 ms） | [確認:コード] |
| C07 | `combine_noises` のコメントは "Average noises" だが、実際は雑音を加算し、チャンネル数で割るだけ（`dataset.rs:2019-2021`） | [確認:コード] |
| C08 | `p_sample_full_speech(1.0)` は呼ばれるが使われない（`p_fill_speech` は未使用） | [確認:コード] |
| C09 | 出荷 ini の `num_prefetch_batches 8` と Python の `prefetch_loader = batch_size × num_workers` は別の量 | [確認:コード] `torch_dataloader.py:101,134` |
| C10 | 論文 Table 1 の checkpoint が出荷モデルと同じかは不明（投稿 2023-05-14、出荷更新 2023-05-23） | [UNKNOWN] |

### 11.2 audio-db の既存ドキュメント・設定の不整合

| ID | 内容 | 確度 |
|---|---|---|
| E01 | `.agents/DECISIONS.md:46` と `docs/training/OFFICIAL_TRAINING_AUDIT.md:22` が `configs/training/deepfilternet3_schedule_seed17.yaml` を、同 `:21` が `tf_gridnet_chunk_seed17.yaml`、`:19` が `semambapp_seed17.yaml` を挙げる。`configs/training/` には `deepfilternet3.yaml`（seed 42、`checkpoint_dir` は `voicebank-seed42`）など 6 本しかなく、`*_seed17.yaml` は存在しない。`VERIFICATION.md` の記述は未確認 | [確認:ファイル] |
| E02 | `README.md:109-118`、`docs/evaluation/PIPELINE.md:22,266-269`、`.agents/DEVELOPMENT.md:35,46,53` が `configs/evaluation/{smoke,example,e2e}.yaml` を挙げるが、`configs/evaluation/` は存在しない（`configs/` は `db`・`training` のみ） | [確認:ファイル] |
| E03 | `AGENTS.md` が `data/db/resource.duckdb` と `data/db/backups/resource-legacy-20261031.duckdb` の存在を前提とし、`configs/training/deepfilternet3.yaml:59` が `data/db/voicebank.duckdb` を指すが、この作業ツリーの `data/db/` は空（Git 管理外のため別環境にある可能性は未確認） | [確認:ファイル] |
| E04 | `docs/se/VALIDATION.md:5,19,48,70`、`docs/se/IMPLEMENTATION_STATUS.md:5,28`、`docs/db/TRAINING_SETS.md:17` が旧名 `.venvs/lab`、`.venvs/ref`、`.venvs/semamba` を使う。現在の環境は `asr`、`deepfilternet`、`official`、`se`、`semambapp` のみ。旧名は `.agents/DEVELOPMENT.md:17` が旧環境と明記しているが、これらの docs 側は更新されていない | [確認:ファイル] |
| E05 | `OFFICIAL_TRAINING_AUDIT.md` の DFN3 節（`:72` 以降）と総括（`:22`）は「検証集約とオンライン混合は別」とするが、発話 RMS 正規化（公式 DFN3 に無い。audio-db の既定は有効）を差異として挙げていない。`:66` の共通表は TF-GridNet だけ `rms_normalize: false` で、他モデルは発話 RMS と書く | [確認:ファイル/コード] |
| E06 | `.agents/DECISIONS.md:46` は「10,700 発話の 120 epoch は更新 22,711 回」と書き、「研究用の検証」の差は明記する。一方、公式の `dataloader_len` は ceil で 22,831 回であり、端数 batch を捨てる差（DFN-D07）は書かれていない | [確認:コード/導出] |
| E07 | DFN3 の「論文」を optimizer・schedule の出典のように扱う記述があるが、DFN3 論文は学習条件を書かない（C01）。出典は出荷 `config.ini` と `train.py` | [確認:論文] |
| E08 | `configs/training/deepfilternet3.yaml` の `scheduler.steps_per_epoch: 668` は初期表の値で、実際の表は epoch ごとに作り直される（読み手が固定値と誤解しうる） | [確認:コード] |

## 12. UNKNOWN 一覧

| ID | 内容 | 調べたこと | 閉じられない理由 | 追加調査案 |
|---|---|---|---|---|
| U01 | 学習の `dataset.cfg`、ファイルリスト、各 split の件数、`sampling_factor` | README、`scripts/`、`assets/`、論文 | 公開されていない | 著者への問い合わせ。GitHub Issues/Discussions の検索（今回は未実施） |
| U02 | 出荷モデルを学習した commit | `git log`、タグ、モデル更新の日付 | 学習時の commit は記録されていない。2023-03〜05 の範囲までしか絞れない | 同範囲の commit ごとの差を整理し、影響が出る項目（§2.3）だけ個別に判断 |
| U03 | 出荷モデルの学習で `min_nb_erb` の不具合（1 固定）が効いていたか | `e71c8ba` の差分 | ini の値（2）が学習時に使われたか分からない | ERB 幅を checkpoint の形状から照合できないか検討 |
| U04 | seed 43 の使われ方 | `train.py`、`dataset.rs` | 実行ログが無い | なし（再現上は分布同等で足りる） |
| U05 | DNS4 のどの部分（言語・サブセット・VCTK/PTDB の範囲）か | 論文 1 文、DFN2 論文 | 具体的な列挙が無い | DNS Challenge 4 のデータ構成と照合 |
| U06 | RIR の集合（実測・模擬の数） | DFN2 論文の数 | DFN3 では書かれていない | 同上 |
| U07 | 評価の test 話者 p232、p257 が学習側の VCTK（×10）から除かれていたか | DFN2 論文は「VoiceBank は話者排他」と書く | DFN3 では書かれていない | 著者への問い合わせ |
| U08 | Table 1 の checkpoint、post-filter、PESQ のモード | 論文 v1 | 記載なし | Interspeech 版の確認（今回は arXiv v1 のみ） |
| U09 | 出荷モデルの検証損失の曲線・最終値 | 公開されていない | | 無し |
| U10 | 環境変数 `DF_*` が学習時に設定されていたか | | ログが無い | なし |
| U11 | VoiceBank 28 話者の発話のうち 3 秒未満の割合（零埋めの影響） | 未測定（DB 書き込みを伴わない集計で測れる） | 今回は実行していない | `corpora.audio_files` の読み取りで長さの分布を集計（読み取りのみ） |
| U12 | RMS 正規化が DFN3 の学習に与える影響 | 未測定 | 学習を実行していない | 同条件の短い比較実験（GPU、承認が要る場合あり） |
| U13 | DNS5 の `noise_fullband` が DNS4 の雑音と同じ素材か、sr は何か | ディレクトリと件数のみ | 中身は未確認 | 読み取りのみで sr を集計 |
| U14 | 出荷 checkpoint を audio-db の評価基盤にかけたときの PESQ/STOI（論文 3.17/0.944 との差） | 記録を見つけられない | 今回は実行していない | VoiceBank test 824 件で評価（GPU は不要。CPU で可） |
| U15 | audio-db の損失・optimizer の数値一致（今回の読み取りは式と定数の照合のみ） | 既存記録を引用 | 今回は実行していない | 既存の 4 プロセス比較の再実行 |
| U16 | 公式 `Stft`・`Istft`（`loss.py:50-92`）と audio-db の `_vorbis_istft` の端の扱い | 式の読み取り | 数値比較は未実施 | 既存の比較テストの再実行 |
| U17 | valid の key の並び（HDF5 の key 順）と端数 batch の中身 | コード読みのみ | HDF5 を開いていない | 該当データがあれば確認 |
| U18 | DFN3 の `deepfilternet3.py` の +63 行が学習時の forward に影響するか | 差分の存在のみ | 学習時のコードは特定できない（U02） | strict load の成否は既存記録 |

## 13. 参照した一次資料

### 13.1 公式コード（`OFF/` @ `d375b2d8309e0935d165700c91da9de862a99c31`、行番号は `rg -n` / `cat -n` で確認）

- `DeepFilterNet/df/train.py`: `:43`、`:92-93`、`:135-168`、`:176-246`、`:247-320`、`:324-457`、`:474-553`
- `DeepFilterNet/df/{lr,checkpoint,config,loss,modules,model,utils,deepfilternet3,enhance,io,sepm,evaluation_utils}.py`: 上記の行番号は本文に記載
- `DeepFilterNet/df/scripts/{prepare_data,split_hdf5,trim_silence_hdf5,test_voicebank_demand}.py`
- `libDF/src/{dataloader,dataset,augmentations,transforms,util}.rs`
- `pyDF-data/src/lib.rs`、`pyDF-data/libdfdata/torch_dataloader.py`
- `scripts/download_process_dns4.sh`、`assets/dataset.cfg`、`README.md:177-273`、`DeepFilterNet/pyproject.toml`
- `models/DeepFilterNet3.zip` と `DeepFilterNet3/config.ini`（`data/checkpoints/deepfilternet3/DeepFilterNet3/config.ini` と一致）

### 13.2 論文

- DFN3: H. Schröter ほか、"DeepFilterNet: Perceptually Motivated Real-Time Speech Enhancement"、Interspeech 2023、arXiv 2305.08227（v1 のみ）。
- DFN2: arXiv 2205.05474。DFN1: arXiv 2110.05588（今回は DFN1 の本文は精読していない）。

### 13.3 audio-db 側（現行コード、行番号は `rg -n` で確認）

- `configs/training/deepfilternet3.yaml`
- `src/training/se/deepfilternet.py`、`optim.py`、`config.py`、`run.py`
- `src/training/core/{trainer,batch,batch_schedule,rng}.py`
- `src/training/data/{dataset,sampler}.py`
- `src/evaluation/audio/{render,mixing}.py`、`src/evaluation/metrics/{pesq,stoi,sisdr,compute}.py`
- `src/se/models/deepfilternet3/{inference,model}.py`
- `requirements/official-assets.toml`
- `docs/db/{CORPORA,DATASETS,RECIPE}.md`、`docs/training/{OFFICIAL_TRAINING_AUDIT,VERIFICATION}.md`、`docs/se/{MODELS,VALIDATION}.md`、`.agents/{DATA,DECISIONS,DEVELOPMENT}.md`

### 13.4 履歴調査（`/tmp/audit-dfn/dfn`、別 clone）

`git log` / `git diff --ignore-cr-at-eol v0.5.0 d375b2d`。`dataset.rs` の約 4,876 行の差は改行コードの違いなので、`--ignore-cr-at-eol` で比較した。
