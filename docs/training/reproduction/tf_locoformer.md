# TF-Locoformer 公式再現プロトコル監査

P5（2026-11-06）で 4 rank の chunk 計画と `iterator_stop` は実装した。下の「4 rank DDP を再現できない」という行は、この監査時点の記録であり、現行の計画は [P5_VERIFICATION](P5_VERIFICATION.md) を見る。P7 で DNS のオンデマンド条件と `valid.loss.ave_5best` を実装した。このファイルの本文は監査当時の記録である。現行の Dataset、段階名、検証数値は [P7_VERIFICATION](P7_VERIFICATION.md) を見る。P7.1 の受理・リサンプル・`ave_1best` は [P7_1_VERIFICATION](P7_1_VERIFICATION.md) を見る。`cv_synthetic_small` は不明のままである。

調査・文書化のみ。コード・設定・DB・データは変更していない。実装案は「提案」として書く。

確度ラベル: 【確認:コード】【確認:論文】【確認:設定】【推測】【UNKNOWN】。本監査で行った読み取り専用の実測（ファイル数、秒数、波形統計）には【確認:実測】を使う。これは上の 5 種への注記用の追加ラベルである。

根拠の書式は `[リポジトリまたは URL @ commit/tag : ファイル : 行 : 関数/設定キー]`。略称は 13 章で定義する。

- `TFL` = merlresearch/tf-locoformer @ v1.1.0（7a615460d347ff7334a13dbb831d16280da72cdc）
- `ESP90` = espnet/espnet @ 90eed8e53498e7af682bc6ff39d9067ae440d6a4（TF-Locoformer が動作確認した版）
- `ESP0d` = espnet/espnet @ 0d136c5456d403cb2c0a4c13e76e8136e32e9b7a（audio-db のピン）
- `DNS` = microsoft/DNS-Challenge branch `interspeech2020/master` @ 70f19285c36cca4df2338f9248775ddc50980c6b
- `ADB` = audio-db @ 3bee9ad（branch feature/reproduction）
- `PAPER` = arXiv:2408.03440v1

---

## 0. 要約

### 0.1 監査結論

**分類: 事前生成（pre-generated）・固定ペア。オンライン混合ではない。**

- 公式の DNS レシピは、DNS-Challenge の `noisyspeech_synthesizer_multiprocessing.py` で noisy / clean / noise の 3 本の wav を事前に生成する。ESPnet はそれを `wav.scp`（speech_mix）と `spk1.scp`（speech_ref1）として読むだけである。学習中の混合・SNR 再サンプリング・RIR・雑音の再選択・音量変更は無い。【確認:コード】【確認:設定】
  - `[ESP90 : egs2/dns_ins20/enh1/local/dns_create_mixture.sh : 61-70]`
  - `[TFL : egs2/dns_ins20/enh1/exp/enh_train_enh_tflocoformer_raw/config.yaml : 104-117, 159-164, 173-178 : train_data_path_and_name_and_type, speech_volume_normalize, rir_scp, noise_scp, dynamic_mixing]`
- 学習中にランダムなのは、元発話の順序、チャンクのキャッシュ内シャッフル、チャンクの開始オフセットだけである。DNS の 30 秒発話では、このオフセットは常に 0 になる（5.3 節）。
- 学習データは 3000 h（360,000 本 × 30 秒、うち学習 2700 h / 検証 300 h）の再生成物である。公式リポジトリは生成済みデータも `cv_synthetic_small` の一覧も配布していない。したがって再現は「公式スクリプトで再生成して登録する」形になり、ビット一致ではなく統計的同等にしかならない（生成が multiprocessing で非決定的なため）。【確認:コード】【推測】

**audio-db の現状: DNS の学習データ基盤は無く、TF-Locoformer の学習設定は VoiceBank 代用である。**

- `configs/training/tf_locoformer.yaml` は VoiceBank-DEMAND 用の研究設定で、公式 DNS の再現ではない（同ファイル 1-16 行が自認している）。`data/db/` は空で、`voicebank.duckdb` も `resource.duckdb` も無い。DNS2020 は `corpora_catalog` に生の素材（`se-source`）として載るだけで、学習用・評価用の名前付きセットは無い。【確認:設定】【確認:実測】
- ネットワーク・損失 `mr_l1_tfd`・AdamW・勾配クリップ・Xavier 初期化・`WarmupReduceLROnPlateau` の計算は、コード読解で公式と一致を確認した。ただし数値一致の検証は「standalone separator に対する縮小モデルの 1 step / 5 step」と「公式スケジューラクラスとの比較」までである（8 章）。

### 0.2 主な差異 TOP

| 順位 | ID | 内容 | 影響度 |
|---|---|---|---|
| 1 | TFL-D01 | 学習・検証データが DNS ではなく VoiceBank。DNS の学習セット・評価セットが DB に無い | Critical |
| 2 | TFL-D02 | 公式の混合生成（複数 clean/noise の連結、clean のランダム切り出し、RMS −25 dBFS 正規化、最終レベル乱択、16 bit 量子化）は recipe JSON v1 で表現できない | Critical |
| 3 | TFL-D03 | `data.rms_normalize: true`（発話 RMS=1）は公式の `speech_volume_normalize: null` と逆。損失の絶対スケールが変わる | Major |
| 4 | TFL-D04 | 1 epoch の定義が違う。公式は rank あたり 1000 発話 × 4 rank = 4000 発話、約 14,000 更新。audio-db は 10,700 発話 | Major |
| 5 | TFL-D05 | 公式は 4 rank DDP（rank あたり batch 1、シャード `[rank::4]`、cache 1024//4=256）。audio-db は 1 プロセス専用 | Major |
| 6 | TFL-D06 | 評価用重みは公式が `valid.loss.ave_5best.pth`（5-best 平均）。audio-db は平均を作らない | Major |
| 7 | TFL-D07 | DNS 合成テスト（no_reverb / with_reverb 各 150 本）が名前付き評価セットに無い。SDR/SAR/SIR・出力ピーク正規化も無い | Major |
| 8 | TFL-D08 | 既存 docs に誤りと、存在しないファイルへの参照が多い（11 章） | Major |

### 0.3 論文と公式コードの矛盾（要点）

- 論文の「DNS は 21,000 step を 1 epoch とする」に対し、公式設定から計算できるのは約 14,000 更新である（5.4 節）。
- 論文の Macaron の 1/2 係数は公式コードに無い（公式コード自身が認めている）。
- 論文の「3 epoch 改善しなければ lr 半減」に対し、実装は 4 回連続の悪化で下げる。
- 論文の SI-SNR 23.3 に対し、公式 RESULTS.md は 23.23。PESQ-WB 3.72 は RESULTS.md に無い。
- 公式 `best_model_criterion` の `valid/si_snr/max` は stats に `si_snr` キーが無いため無効で、実効は `valid/loss/min` だけである。

詳細は 11 章。

---

## 1. 調査メタデータ

| 項目 | 内容 |
|---|---|
| 調査日 | 2026-11-06 |
| audio-db | branch `feature/reproduction`、HEAD `3bee9addc0a1ff3d10d4e47a2a0eafe34f4bc110`（2026-11-06 01:26 +0900） |
| 調査対象（公式） | TF-Locoformer v1.1.0（7a615460…、2025-08-08）。v1.0.0（1d2be38、2024-07-31）、v1.0.1（b76d38b、2025-05-16）と差分比較 |
| ESPnet | 90eed8e53498e7af682bc6ff39d9067ae440d6a4（2024-05-27）。audio-db ピン 0d136c5456d403cb2c0a4c13e76e8136e32e9b7a（2026-10-30）と差分比較 |
| DNS-Challenge | `interspeech2020/master` 70f19285…（2020-07-22）。ローカル `data/corpora/DNS2020` は同 commit の shallow clone で、スクリプト 6 本が upstream と一致（`diff` 確認、【確認:実測】） |
| 論文 | arXiv:2408.03440v1 の HTML 版テキスト抽出（`/tmp/audit-tf_locoformer/p.html.txt`）。本文確認は HTML 版のみ。PDF 版との照合と、v2 以降の有無の確認はしていない |
| 調査方法 | 公式・ESPnet・DNS の /tmp 別クローンのソース読解（`rg -n` / `cat -n` で行番号を確認）、ローカルの DNS2020 データの読み取り専用統計（`wave` モジュール、numpy）、audio-db の現行コード読解 |
| 作業領域 | `/tmp/audit-tf_locoformer/{tf-locoformer,espnet,dns-challenge,rope}`。`src/se/official/` と `build/` では git 操作をしていない |
| 調査していないこと | 学習・推論・評価の実行（GPU 不使用）、生成データの再生成、DuckDB の読み書き、公式チェックポイントの読み込み（`.pth` は 133 バイトの LFS ポインタで、実体はローカルに無い）、ESPnet の実行時動作の実測、公式学習ログ（train.log）の入手 |
| 作業ツリーの観測 | `data/db/`・`data/staging/` は空。`data/checkpoints/tf-locoformer/` は無い（既存 docs と `scripts/setup/doctor.py:206` はここに DNS 重みがあると書く）。`configs/` は `db/` と `training/`（6 ファイル）のみ |

variant の選定は 2 章を参照。

---

## 2. A. 公式資料

| 項目 | 内容 | 確度 / 根拠 |
|---|---|---|
| 論文タイトル | TF-Locoformer: Transformer with Local Modeling by Convolution for Speech Separation and Enhancement（Saijo, Wichern, Germain, Pan, Le Roux。IWAENC 2024） | 【確認:論文】`[PAPER : 表題]`、`[TFL : espnet2/enh/separator/tflocoformer_separator.py : 23-29 : docstring]` |
| arXiv ID と版 | **2408.03440**、v1 のみ確認。audio-db の `docs/se/MODELS.md:21` は `2409.16674` と書いており、誤り | 【確認:論文】。不整合は 11 章 |
| 公式 GitHub | https://github.com/merlresearch/tf-locoformer | 【確認:コード】 |
| 対象 tag / commit | `v1.1.0` = 7a615460d347ff7334a13dbb831d16280da72cdc（2025-08-08）。audio-db の `requirements/official-assets.toml` も同じ pin | 【確認:設定】 |
| タグ間の差 | v1.0.0 → v1.1.0: `pos_enc`（rope / nope）引数の追加、`input.shape[1]` 検査のバグ修正、Macaron の 1/2 係数が無いというコメントの追加、`standalone/` と `tests/` の追加。DNS の設定・recipe・チェックポイントのファイルは変化なし | 【確認:コード】`git diff v1.0.0 v1.1.0 --stat`（/tmp クローン） |
| 論文と公式コードの時期 | 論文 2024-08（arXiv）。コード v1.0.0 は 2024-07-31、事前学習済み DNS 重みの RESULTS は 2024-06-25（ESPnet 90eed8e で学習・評価）。v1.1.0 は 2025-08 の追加リリースで、学習条件は変えていない | 【確認:コード】`[TFL : egs2/dns_ins20/enh1/exp/enh_train_enh_tflocoformer_raw/RESULTS.md : 10-15]` |
| 前提 ESPnet 版 | **90eed8e**（2024-05-27）。`git checkout 90eed8e5…` を README が指示。torch 2.1.0、CUDA 11.8、`rotary-embedding-torch==0.6.1` | 【確認:設定】`[TFL : README.md : 98, 104, 110]`、`[TFL : requirements.txt]` |
| ESPnet への組み込み | `copy_files_to_espnet.sh` が separator を `espnet2/enh/separator/` にコピーし、`enh.patch` で `espnet2/tasks/enh.py` に `tflocoformer=TFLocoformerSeparator` を追加する。**ESPnet 本流には TF-Locoformer が無い**。0d136c5 の `espnet2/enh/separator/` にも `tflocoformer_separator.py` は無い（audio-db の `src/se/official/espnet` と `build/clean-20261104/sources/espnet` の両方で `ls` 確認） | 【確認:コード】`[TFL : copy_files_to_espnet.sh : 5-12]`、`[TFL : espnet2/tasks/enh.patch]` |
| 学習コード | 公式リポジトリは ESPnet-SE のパイプライン（`egs2/TEMPLATE/enh1/enh.sh`、`espnet2/bin/enh_train.py`）を使い、独自の学習ループを持たない | 【確認:コード】`[TFL : README.md : 88-90, 139-159]` |
| 設定ファイル（DNS） | `egs2/dns_ins20/enh1/conf/tuning/train_enh_tflocoformer.yaml`、学習時に保存された `exp/enh_train_enh_tflocoformer_raw/config.yaml` | 【確認:設定】 |
| データローダー | ESPnet `espnet2/iterators/chunk_iter_factory.py`、`sequence_iter_factory.py`、`espnet2/train/dataset.py`、`espnet2/train/preprocessor.py`（`EnhPreprocessor`）。本リポジトリは何も上書きしない | 【確認:コード】 |
| 評価スクリプト | ESPnet `espnet2/bin/enh_inference.py`、`enh_scoring.py`、`egs2/TEMPLATE/enh1/enh.sh` stage 7-8 | 【確認:コード】 |
| 事前学習済みチェックポイント | `egs2/dns_ins20/enh1/exp/enh_train_enh_tflocoformer_raw/valid.loss.ave_5best.pth`（git-LFS）。audio-db の記録: sha256 `6ffa518f2b73bb…`（`requirements/official-assets.toml`）、59,972,116 バイト。ローカルの checkout 内は 133 バイトのポインタ | 【確認:設定】。実体の読み込みは今回していない |
| 事前学習済み DNS モデルの配布先 | 上記 LFS ファイルのみ。公開用の別配布（Zenodo 等）は確認していない | 【UNKNOWN】 |

### 2.1 variant の選定

リポジトリには 4 レシピがある: `wsj0_2mix`、`librimix`、`whamr`（以上は話者分離）、`dns_ins20`（単一話者強調）。目的は単一チャンネル・単一話者 SE の再現なので、**`dns_ins20`、Medium モデル（B=6, C=384, K=4, H=4, G=4, 15.0 M パラメータ）の `train_enh_tflocoformer.yaml`** を選ぶ。

- 論文 Table 1 に Small / Medium / Large（B=4/6/9、C=256/384/384）がある。DNS の Table 4 は Medium だけである。【確認:論文】`[PAPER : Table 1, Table 4]`
- DNS の設定は `num_spk: 1`（`[TFL : …/train_enh_tflocoformer.yaml : 15]`）。話者分離用の設定とは `ffn` 次元以外も別物である。
- `train_enh_tflocoformer-nope.yaml`（RoPE 無し、v1.1.0 で追加）は論文の主結果ではない。選ばない。

### 2.2 候補差異と研究上の判断が要る点

次の点は決めずに候補を残す。

1. **ESPnet の版**: 公式が学習・評価した 90eed8e と、audio-db が比較用に固定した 0d136c5 がある。本監査で、DNS の学習に効く範囲（チャンク iterator、`mr_l1_tfd`、`WarmupReduceLROnPlateau`、`average_nbest_models`、`initialize`、`Stft`、`ESPnetEnhancementModel` の `normalize_variance` 経路、`egs2/dns_ins20`）の実効動作が同じことを確認した（6.4 節）。差は AMP/DDP 通信フック/再開時の早期停止再判定/`discard_short_samples` 引数の追加などで、DNS 設定の実効動作には出ない。【確認:コード】
2. **`rotary-embedding-torch`**: 公式は 0.6.1、audio-db は 0.8.3（`.agents/DEVELOPMENT.md:62`）。0.6.1 と 0.8.3 の差分は、型検査の削除、`autocast('cuda')` 化、キャッシュの buffer 化、fp32 では結果の変わらない型変換だった。回転式は同一。fp32 で数値差が無いと推測するが、公式重みでの両版比較は今回していない。【確認:コード】【推測】
3. **検証データの大きさ**: 論文は 300 h、保存済み学習 config は `cv_synthetic_small`（大きさ不明）。README は「300 h の検証は非常に遅いので縮小を勧める」と書く（`[TFL : README.md : 218]`）。【確認:設定】

---

## 3. B. 学習データセット

### 3.1 DNS-Interspeech 2020 の素材

| 項目 | 内容 | 確度 / 根拠 |
|---|---|---|
| データセット | Interspeech 2020 Deep Noise Suppression Challenge のデータ。スクリプトは `DNS` の `interspeech2020/master` | 【確認:論文】`[PAPER : §3.1]`。【確認:コード】 |
| 配布元 | https://github.com/microsoft/DNS-Challenge （スクリプトとテストセット。音声は同リポジトリの download スクリプトで取得する） | 【確認:コード】 |
| clean speech | ローカル `data/corpora/DNS2020/datasets/clean`: **65,348 ファイル、全て 31.0 秒・16 kHz**（300 件の無作為標本で min=max=31.0 秒）。推定 約 563 h。ファイル名 `book_*_chp_*_reader_<ID>_<n>.wav`、reader ID は **1,948 種** | 【確認:実測】 |
| noise | `datasets/noise`: **65,302 ファイル**、長さ 1〜10.0 秒（2,000 件の標本平均 9.95 秒）、推定 約 180 h。2,000 件の標本のサンプリングレートは 16 kHz が 1,996 件、48 kHz が 3 件、96 kHz が 1 件 | 【確認:実測】 |
| RIR | 学習データには**使わない**。`audiolib.add_reverb` は定義されているが、どこからも呼ばれない。DNS の学習用 RIR は配布されていない | 【確認:コード】`[DNS : audiolib.py : 84-90 : add_reverb]`、`[DNS : noisyspeech_synthesizer_multiprocessing.py : 全体]` |
| clean の性質 | 学習用 clean は DNS 配布のまま（残響除去・VAD 加工なし）。残響を含み得るかは配布元の性質で、ここでは判断しない | 【UNKNOWN】 |
| 形式 | 16 kHz、モノラル、wav。生成物は `soundfile.write` の既定で **PCM_16** | 【確認:コード】`[DNS : audiolib.py : 80 : audiowrite]` |

### 3.2 分割と件数

| 項目 | 内容 | 確度 / 根拠 |
|---|---|---|
| 総量 | 論文: 3000 h = 学習 2700 h + 検証 300 h。README: 同じ。生成本数は `int(total_hours*3600/audio_length)` = 3000×3600/30 = **360,000 本** | 【確認:論文】`[PAPER : §3.1]`、【確認:設定】`[TFL : README.md : 209]`、【確認:コード】`[DNS : noisyspeech_synthesizer_multiprocessing.py : 271]` |
| 生成設定の変更 | `dns_create_mixture.sh` の既定は `total_hours=100`、`snr_lower=0`、`snr_upper=40`。公式は README の `sed` で `total_hours=3000`、`snr_lower=-5`、`snr_upper=15` に書き換える | 【確認:コード】`[ESP90 : egs2/dns_ins20/enh1/local/dns_create_mixture.sh : 18-20]`、【確認:設定】`[TFL : README.md : 213-215]` |
| 学習 / 検証の分け方 | 生成した noisy の一覧を `sort -u`、uttid を `reader_<ID>_fileid_<N>` に `sed` で作り、`sort -k1,1` し、**先頭 9/10 を tr_synthetic、残りを cv_synthetic** にする。すなわち 324,000 / 36,000 本（【推測】: 件数は計算値で、生成物は無い） | 【確認:コード】`[ESP90 : egs2/dns_ins20/enh1/local/dns_data_prep.sh : 59-72]` |
| 分割の性質 | ファイル名の `sed` は greedy で、単一 clean 由来なので uttid の reader ID はその混合の clean の話者である。uttid の辞書順の連続 block を cv にするため、**clean の話者は tr / cv でほぼ分離する（境界の 1 話者だけ分割され得る）**。一方、**noise は分割しない**。clean の各ファイルは 360,000/65,348 ≒ 5.5 回再利用される（カウンタの巡回）が、同じ話者は同じ側に入る | 【確認:コード】【確認:実測】`sed` の動作を代表ファイル名で試験。再利用倍率は【推測】（低活動の棄却分の消費は除く） |
| 学習時の検証セット | 保存された学習 config は `dump/raw/cv_synthetic_small` を使用（`cv_synthetic` ではない）。**`cv_synthetic_small` を作るスクリプトも件数も公式リポジトリに無い** | 【確認:設定】`[TFL : …/exp/…/config.yaml : 111-116]`。内容は【UNKNOWN】 |
| テスト | `tt_synthetic_no_reverb`、`tt_synthetic_with_reverb`。DNS Challenge の `datasets/test_set/synthetic/{no_reverb,with_reverb}`。各 **150 本・10.0 秒・16 kHz・モノラル・16 bit**（ローカルで全数確認、合計 0.417 h ずつ）。clean は Graz 大学の clean 音声（20 話者・4,270 文）、雑音は AudioSet / freesound で学習セットに含まれない。with_reverb は Microsoft 内部録音の RIR（RT60 300〜1300 ms）で残響を付与 | 【確認:実測】【確認:設定】`[DNS : datasets/test_set/README.md（ローカル同内容）]`。論文は「non-blind anechoic test set」を使用（`[PAPER : §3.1]`）= no_reverb |
| with_reverb の参照 | 残響付きの "clean"（reverberant clean）。根拠: no_reverb と with_reverb の `clean_fileid_0.wav` / `clean_fileid_101.wav` は RMS が 6 桁まで一致する（0.078271 / 0.017779）のに、ラグを許した正規化相互相関の最大が 0.30 / 0.48 で、波形が異なる（同じ話者の別音源か、残響付加とレベル再調整のいずれかで、どちらかは判別できない）。また with_reverb の noisy と clean の SI-SNR 平均は 9.19 dB（no_reverb は 9.23 dB）で、clean が dry なら残響分だけ大きく下がるはず | 【推測】（【確認:実測】の統計からの推論） |
| 教師波形の定義 | 学習: 生成時の `clean_snr`（clean を RMS −25 dBFS に揃え、noisy と同じ係数 `scalarnoisy` を掛けた波形。残響なし）。保存名 `clean_fileid_N.wav` | 【確認:コード】`[DNS : audiolib.py : 119-142 : snr_mixer]` |
| clean / noisy の対応 | uttid の末尾の `fileid_N` を共通キーに、`noise_fileid_N.wav`、`clean_fileid_N.wav` へ `sed` で置換する（`dns_data_prep.sh` 86-92）。テストは `fileid_N` を共通キーにする（108-112） | 【確認:コード】 |
| 論文と公式コードの実使用データの差 | 論文は 3000 h（2700+300）、保存 config の検証は `cv_synthetic_small`（縮小）。論文の SNR は「−5〜15」と読めるが、実装は整数一様で最大 14 | 【確認:論文】【確認:コード】。11 章 |

### 3.3 重複・リークの条件

- tr / cv 間の clean 話者: ほぼ分離（境界話者を除く）。【確認:コード】
- tr / cv 間の noise ファイル: **共有される**（雑音を分割しない）。【確認:コード】
- テストの clean / noise は学習セットと別ソース（Graz 大学の clean、学習に無い雑音）。【確認:設定】
- DNS の train clean と test clean の話者重複: 確認していない。【UNKNOWN】

---

## 4. C. データ生成・前処理

### 4.1 分類

事前生成・固定ペア。生成は 1 回で、結果の wav が学習の入力になる。ESPnet の `EnhPreprocessor` は `speech_volume_normalize: null`、`rir_scp: null`、`noise_scp: null`、`dynamic_mixing: false` のため、何も加えない（【確認:設定】`[TFL : …/config.yaml : 159-164, 173-178]`）。

学習時に毎回再サンプリングされるものは**無い**。再サンプリングされるのは生成時で、**1 本ごと**（`main_gen` の 1 回の呼び出しごと）。

### 4.2 処理順序（生成時。`main_gen` と `build_audio`）

`[DNS : noisyspeech_synthesizer_multiprocessing.py : 142-216 : main_gen]`、`[DNS : audiolib.py : 111-152 : snr_mixer]`。

1. 設定を読む。`num_files = int(total_hours*3600/audio_length)`（L271）。clean の一覧は `glob`、noise の一覧も `glob`。両方を `random.shuffle`（Python の `random`、`random.seed(3)`、L30, L298, L302）。
2. 共有カウンタ 2 本（clean 用、noise 用）を `multiprocessing.Value` で作り、`Pool(PROCESSES=cpu_count())` に渡して `main_gen(params, filenum)` を `starmap` で 360,000 回呼ぶ（L305-311）。
3. clean を作る（`gen_audio(True, …)` → `build_audio`）:
   1. カウンタを 1 進めて `idx = counter % N` のファイルを読む（L71-75）。一覧の巡回である。
   2. サンプリングレートが 16 kHz でなければ `librosa.resample`（L76-77）。【確認:コード】。librosa の版は `requirements.txt` が無指定のため【UNKNOWN】。
   3. 読んだ音声が残り長より長く、**学習セット（`is_test_set=False`）**なら、`np.random.randint(0, len-remaining)` で開始位置を選び、残り長ちょうどに切り出す（L81-83）。DNS の clean は 31 秒、残り長は 30 秒（480,000 サンプル）なので、**開始位置は [0, 16000) の一様乱数で、30 秒に切る**。
   4. `is_clipped`（絶対値が 0.99 超）なら棄却し、`tries_left` を減らす（最大 `MAXTRIES=50`、L86-89）。
   5. 結合し、残りがあれば 0.2 秒の無音を挟む（L97-100）。
4. clean の活動率を `activitydetector` で測り、`> clean_activity_threshold=0.6` でなければ最初から作り直す（L127-139）。`activitydetector` は clean を −25 dBFS に揃えたあと、50 ms 窓のエネルギー確率（シグモイド `a=-1, b=0.2`）を平滑化（`alpha_att=0.8`, `alpha_rel=0.05`）し、しきい値 0.13 を超えた窓の割合を返す（`[DNS : audiolib.py : 224-258]`）。
5. noise を clean と同じ長さまで作る（`gen_audio(False, …, len(clean))`）。noise は 1〜10 秒なので、**複数ファイルを 0.2 秒の無音を挟んで連結し、最後の 1 本を残り長に合わせてランダムに切る**（手順 3 と同じ `build_audio`。`noise_activity_threshold=0.0` で活動率検査は実質行わない、L130）。
6. SNR を決める。`randomize_snr=True` なら `np.random.randint(snr_lower, snr_upper)`（**上限は含まれない整数一様**）。公式設定では **−5, −4, …, 14 dB の 20 値**（L168-172）。【確認:コード】
7. `snr_mixer`:
   1. 長さを揃える（短い方にゼロ詰め、L114-117）。実際には両方 480,000 サンプル。
   2. clean を**ピーク正規化**（`/ (max|clean|+EPS)`）し、**RMS −25 dBFS** に揃える（L120-122）。noise も同様（L124-126）。
   3. `noisescalar = rmsclean / 10**(snr/20) / rmsnoise` で noise を縮尺（L129-130）。SNR は**全サンプルの平均二乗**（無音区間を含む）で定義される。
   4. `noisy = clean + noise_scaled`（L133）。
   5. noisy の最終 RMS を `np.random.randint(target_level_lower, target_level_upper)`（上限を含まない。**−35〜−16 dBFS の 20 値**）にし、clean と noise にも同じ係数を掛ける（L137-142）。
   6. noisy が 0.99 を超えたら、noisy / clean / noise を同じ係数で縮める（L145-150）。
8. `main_gen` に戻り、clean・noise・noisy のいずれかが 0.99 を超えたら全体を作り直す（L184-187）。
9. `soundfile.write` で noisy / clean / noise の 3 本を **PCM_16 の wav** として書く（L208-213、`audiolib.py:80`）。ファイル名は `<clean名>_<noise名>_snr<SNR>_fileid_<N>.wav`、`clean_fileid_<N>.wav`、`noise_fileid_<N>.wav`。clean と noise の名前の連結は 100 文字で切られる（L192, L194, `MAXFILELEN=100`）。
10. ESPnet 側のデータ準備（`dns_data_prep.sh`）で一覧化・分割する（3.2 節）。
11. ESPnet `enh.sh` の stage 3-4 で `dump/raw` にコピーし、**`utt2num_samples` が `0.1 s 超 31 s 未満`（`min_wav_duration=0.1`、`max_wav_duration=31`）の発話だけ**を残す（`awk '$2 > min && $2 < max'`）。30 秒の混合は残る。

### 4.3 項目別の確認

| 項目 | 内容 | 確度 / 根拠 |
|---|---|---|
| 雑音選択 | 一覧（shuffle 済み）の巡回。共有カウンタ `idx = counter % N`。複数プロセスが同じカウンタを使うので、どの本がどの雑音を取るかは**スケジューリングに依存し、再実行で再現しない** | 【確認:コード】`[DNS : …multiprocessing.py : 56-75, 305-311]` |
| 乱数 | `np.random.seed(2)`（L29）と `random.seed(3)`（L30）はモジュール import 時の 1 回だけ。`Pool` を fork する時点の状態が全 worker に複製されるため、worker 間で同じ乱数列が出る可能性がある（SNR・レベル・切り出し位置が重複する）。実測はしていない | 【確認:コード】。重複は【推測】 |
| SNR | 整数一様 −5…14 dB（`randint` 上限除外）。論文本文の記述は「SNR 範囲」を明示しない。README は `snr_upper=15` と書く | 【確認:コード】【確認:設定】 |
| ゲイン | 最終 RMS −35…−16 dBFS（整数一様）。音量正規化を ESPnet 側では行わない | 【確認:コード】【確認:設定】 |
| 残響 / RIR | 学習データには無い。テスト `with_reverb` は Microsoft 内部 RIR | 【確認:コード】【確認:設定】 |
| リサンプリング | 学習データは全て 16 kHz。noise の標本で約 0.2 %（48 / 96 kHz、2,000 件中 4 件）だけ `librosa.resample`。librosa の版・`res_type` は【UNKNOWN】。librosa ≥ 0.10 は位置引数の `resample(y, sr1, sr2)` を受け付けないため、古い版を使ったはずである（【推測】） | 【確認:コード】 |
| モノラル変換 | `audioread` が多チャンネルを平均（`audiolib.py:50-52`）。DNS はモノラルなので実際は何もしない | 【確認:コード】 |
| 正規化 | 生成時のみ（ピーク→RMS −25 dBFS→最終レベル）。ESPnet 側は `speech_volume_normalize: null`。モデル内で入力の標準偏差を割る `normalize_variance`（6.4 節）が別にある | 【確認:コード】【確認:設定】 |
| clipping | 0.99 超は棄却、または全信号を縮尺（上の 4.2-7.6, 8） | 【確認:コード】 |
| ランダムクロップ | 生成時: clean（31→30 秒）と noise の切り出し。学習時: チャンク切り出し（5.3 節） | 【確認:コード】 |
| padding | 無し。短い発話は破棄する（`lg < L` を満たすチャンク長が無ければ discard） | 【確認:コード】`[ESP90 : espnet2/iterators/chunk_iter_factory.py : 170-177]` |
| 音声長制限 | 学習: `max_wav_duration=31`（31 秒未満）。生成物は全て 30 秒の想定 | 【確認:コード】`[ESP90 : egs2/TEMPLATE/enh1/enh.sh : 491-499]`。30 秒は【推測】（生成物が無い） |
| その他の augmentation | 無い（速度摂動 `speed_perturb_factors` は未指定。`dynamic_mixing: false`） | 【確認:設定】 |
| 量子化 | 3 本とも 16 bit に独立に量子化される。noisy = clean + noise は量子化後には成立しない | 【確認:コード】 |

### 4.4 学習時（ESPnet 側）の処理順序

1. `dump/raw/<set>/wav.scp`（speech_mix）と `spk1.scp`（speech_ref1）を `ESPnetDataset` が読む。`sound` 型、float32（`train_dtype: float32`）。
2. `EnhPreprocessor`（`preprocessor: enh`）は no-op（上記 4.1 の設定で何も適用しない）。
3. `ChunkIterFactory` が 1 発話ずつ（`SequenceIterFactory` を batch 1 で）取り出し、チャンクに分割（5.3 節）。
4. `ESPnetEnhancementModel.forward`: `speech_mix` を標準偏差で割る（不偏 `torch.std`）→ STFT → separator → iSTFT → 標準偏差を掛け戻す → 損失は**正規化していない `speech_ref1`** と比較する。

---

## 5. D. Dataset/DataLoader/Sampler

### 5.1 構成と処理の流れ

- `build_chunk_iter_factory`（`iterator_type: chunk`）。`[ESP90 : espnet2/tasks/abs_task.py : 1818-1881]`
  - 発話の一覧は `UnsortedBatchSampler(batch_size=1, key_file=shape_files[0])`。**一覧の順序は `exp/enh_stats_16k/train/speech_mix_shape` のファイル順**（L1842）。そのファイルは `collect_stats` の集約結果で、順序の実体は【UNKNOWN】（再現するなら uttid ソート順の可能性が高いが、確認できない）。
  - 分散時は `batches = batches[rank::world_size]`（L1866）、`batch_size`（global 4）は rank 数で割る（4//4 = 1）、`num_cache_chunks = args.num_cache_chunks // world_size`（1024 // 4 = **256**、L1860）。
  - `num_samples_per_epoch = num_iters_per_epoch`（L1879）。**チャンク iterator では `num_iters_per_epoch` は更新回数ではなく、rank あたりの元発話数**である。
- `SequenceIterFactory.build_iter`（`[ESP90 : espnet2/iterators/sequence_iter_factory.py : 75-140]`）が epoch ごとの発話列を作る。
  - 学習セットが `num_iters_per_epoch`（1000）より大きいとき（L82-108）: `real_epoch, offset = divmod(1000*epoch, N)`。`RawSampler.generate` は並べ替えず（L31-32）、`np.random.RandomState(real_epoch + seed).shuffle` で 1 周期ごとに並べ替える。epoch は周期をまたぐ窓を順に取る。
  - 検証（`num_iters_per_epoch` 無し、L135-138）: `shuffle=False`。全発話を 1 回ずつ。
- `ChunkIterFactory.build_iter`（`[ESP90 : espnet2/iterators/chunk_iter_factory.py : 121-267]`）
  - `state = np.random.RandomState(epoch + seed)`（L130）。
  - 発話ごとに `W = int(state.choice(chunk_lengths, 1))`（L186）、`S = int(W * shift_ratio)`、`N = (L - W)//S + 1`（L191-193）、**`Z = state.randint(0, (L-W) % S + 1)`（学習）/ 0（検証）**（L194-197）。チャンクは `v[Z + i*S : Z + i*S + W]`（L212-214）。端の余りは捨てる。
  - `len(cache) > num_cache_chunks` で `_generate_mini_batches`（L220-226）: 学習なら cache 全体を `state.shuffle`（L251-255）し、batch 単位で取り出す。epoch の最後に残りを全て取り出す（L231-242）。`bs=1` なので端数チャンクは捨てられない。
- `worker` の乱数: DataLoader worker は epoch+seed 由来で rank に依らない。DNS レシピでは前処理に乱数が無いので影響しない。【確認:コード】`[ESP90 : espnet2/tasks/abs_task.py : 1250 : set_all_random_seed(args.seed)]`、seed=0（`[TFL : …/config.yaml : 15]`）。
- DDP: 各 rank が自分のシャードで epoch を回し、いずれかの rank が尽きたら全 rank が止まる（`iterator_stop`、`[ESP90 : espnet2/train/trainer.py : 554-564, 787-788]`）。
- 勾配蓄積は `accum_grad: 1`（`[TFL : …/config.yaml : 55]`）。勾配は rank 間で平均される。`loss = Σ(loss*weight) / Σweight * world_size / accum_grad` を DDP の勾配平均と組み合わせるので、実効損失は **4 個のチャンク（各 rank 1 個）の平均**である（`[ESP90 : espnet2/train/trainer.py : 655-666]`）。【確認:コード】

### 5.2 DNS 公式の数値

| 量 | 値 | 計算・根拠 |
|---|---|---|
| 学習混合数 | 324,000 | 360,000 × 9/10【推測】（生成物が無い）。`train_num=$(($num*9/10))`、`dns_data_prep.sh:68` |
| rank あたりのシャード | 81,000 | 324,000 / 4【推測】 |
| 全発話長 | 480,000 サンプル（30 秒） | 4.2-3-3 のコード動作【推測】 |
| チャンク | W=64,000（4 秒）、S=32,000、N=14 | `(480000-64000)//32000+1 = 14` |
| オフセット Z | **常に 0** | `(480000-64000) % 32000 = 0` なので `randint(0, 1) = 0` |
| 1 epoch の元発話数 | rank あたり 1000、全 rank 4000 | `num_iters_per_epoch: 1000`、4 rank |
| 1 epoch のチャンク数 | rank あたり 14,000、全 rank 56,000 | 1000 × 14 |
| 1 epoch の optimizer 更新数 | **14,000**（global batch 4 チャンク、蓄積なし） | rank あたり batch 1、14,000 batch / epoch |
| 150 epoch の更新数 | 最大 2,100,000 | 14,000 × 150 |
| 実際に学習した epoch 数 | **約 108**（目視） | 保存画像 `images/loss.png`、`si_snr_loss.png` の曲線が約 108 epoch で終わる【確認:設定】（画像を直接確認）。valid loss は終盤も微減しており、停止理由が patience 10 の早期停止か、ジョブの打ち切りかは判別できない【UNKNOWN】 |
| 実際の更新数 | 約 1.5 M | 14,000 × 約 108【推測】（曲線の epoch 数から） |
| 学習データの走査回数 | 150 epoch で 600,000 発話 / 324,000 = 1.85 周、108 epoch で 1.33 周 | 4000 × epoch / 324,000【推測】 |

### 5.3 epoch 定義の表（単位を分ける）

| 単位 | 公式 DNS（4 GPU） | audio-db 現行 `tf_locoformer.yaml`（VoiceBank、1 プロセス） |
|---|---|---|
| 元発話数 / epoch | rank あたり 1000（全体 4000） | 10,700（`num_samples_per_epoch`、train 全件）`[ADB : configs/training/tf_locoformer.yaml : 32]` |
| 発話の選び方 | 周期ごとに `RandomState(real_epoch+seed)` で並べ替え、窓を順に取る | `select_utterances` が同じ式を 1 プロセスで再現 `[ADB : src/training/data/chunk_plan.py : 59-80]` |
| チャンク長 / shift | 64,000 / 32,000 | 32,000 / 16,000（`chunk_iterator.chunk_length` / `chunk_shift_ratio`） |
| チャンク数 / 発話 | 14（30 秒） | 発話ごとに `(L-32000)//16000+1`。2 秒以下の発話は捨てる（10,700 件中 9,321 件が残るという docs の記録、DB が無く再確認不可）【UNKNOWN】 |
| チャンクのオフセット | `randint(0, (L-W)%S+1)`。30 秒では常に 0 | 同じ式 `[ADB : src/training/data/chunk_plan.py : 51]`。VoiceBank の長さでは 0 以外が出る |
| cache | 1024 // 4 = 256 チャンク / rank | 1024 チャンク（1 本の流れ）`[ADB : configs/training/tf_locoformer.yaml : 35]` |
| batch（forward） | rank あたり 1、global 4 | 4（`micro_batch_size_per_gpu: 4`）、leftover 4 未満は捨てる `[ADB : src/training/data/chunk_plan.py : 41-44]` |
| optimizer 更新数 / epoch | 14,000【推測】 | 総チャンク数 / 4（切り捨て）。実長の DB が無く【UNKNOWN】 |
| 検証の 1 回 | `cv_synthetic_small` の全発話を 1 回、64,000 / 0 offset、rank あたり batch 1 | `voicebank_28_valid`（872 件）を全て、chunk 32,000、batch 4、rank 0 だけ `[ADB : src/training/core/trainer.py : 443-475]` |
| 検証の頻度 | 毎 epoch 末 | 毎 epoch 末 `[ADB : src/training/core/trainer.py : 383-385]` |
| 最大 | 150 epoch | `max_epochs: 150`、`max_steps: 2000000`（安全上限）`[ADB : configs/training/tf_locoformer.yaml : 22, 37]` |
| 論文の記述 | 「21,000 step を 1 epoch」 | — |

論文の 21,000 と計算値 14,000 の差は 21,000 / 14 = 1,500 発話に相当する（`num_iters_per_epoch=1500` なら 21,000）。公式 config は 1000 と保存されている。論文時の設定が別だった可能性があるが、確認できない。【推測】【UNKNOWN】

---

## 6. E. モデル・学習条件

### 6.1 アーキテクチャ

| 項目 | 値 | 根拠 |
|---|---|---|
| separator | `tflocoformer`、`num_spk=1`、`n_layers=6`、`emb_dim=128`、`norm_type=rmsgroupnorm`、`num_groups=4`、`tf_order=ft`、`n_heads=4`、`attention_dim=128`（既定、yaml 未指定）、`pos_enc=rope`（既定）、`ffn_type=[swiglu_conv1d, swiglu_conv1d]`、`ffn_hidden_dim=[384, 384]`、`conv1d_kernel=4`、`conv1d_shift=1`、`dropout=0.0`、`eps=1e-5`、`flash_attention=false` | 【確認:設定】`[TFL : …/train_enh_tflocoformer.yaml : 60-83]`、【確認:コード】`[TFL : espnet2/enh/separator/tflocoformer_separator.py : 66-89]` |
| STFT | `n_fft=256`、`hop_length=128`（16 ms / 8 ms）、Hann、`center=True`、onesided。encoder / decoder は `stft` | 【確認:設定】`[TFL : …/yaml : 51-59]`、論文 §3.1 |
| 入力 | 混合の不偏標準偏差で割り、出力に掛け戻す（`normalize_variance: true`） | 【確認:設定】`[TFL : …/yaml : 23-24]`、【確認:コード】`[ESP90 : espnet2/enh/espnet_model.py : 279-285, 298-301]` |
| パラメータ数 | 15.0 M | 【確認:論文】Table 1 |
| Macaron | コードは `x + FFN(norm(x))` で**1/2 係数が無い**。公式コードのコメントが「論文の実験は 1/2 を使ったが、実装では誤って外した。性能に影響しなかったので事前学習済みモデルとの一貫性のためそのままにする」と明記 | 【確認:コード】`[TFL : espnet2/enh/separator/tflocoformer_separator.py : 371-381]` |
| ConvSwiGLU | `Conv1d(dim, 2*inner, k=4)` → 前半 × SiLU(後半) → `ConvTranspose1d` | 【確認:コード】`[TFL : …separator.py : 491-533]` |
| 位置符号化 | RoPE（`RotaryEmbedding(attention_dim//n_heads)` を freq 用・time 用に各 1 個、全ブロックで共有） | 【確認:コード】`[TFL : …separator.py : 106-108]` |

### 6.2 設定値と実効値

| 項目 | 設定値 | 実効値 | 根拠 |
|---|---|---|---|
| init | `xavier_uniform` | 次元が 2 以上の全パラメータ（Conv2d / Conv1d / ConvTranspose / Linear の重み）を `xavier_uniform_`。名前に `.bias` を含む 1 次元を 0。LayerNorm / GroupNorm / Embedding を `reset_parameters()`（`self.conv[1]` の GroupNorm は weight=1, bias=0）。`RMSGroupNorm.gamma`（1 次元・名前に `.bias` 無し）は 1 のまま。RoPE の `freqs`（1 次元、`requires_grad=False`）は無変更。`espnet_initialization_fn` を持つモジュールは呼ぶ | 【確認:コード】`[ESP90 : espnet2/torch_utils/initialize.py : 70-90]`、`[ESP90 : espnet2/tasks/enh.py : 553]`。0d136c5 の同関数も同じ手続き（モジュールのリセット節まで同一） |
| optimizer | `adamw`、lr 1e-3、eps 1e-8、weight_decay 1e-2 | **betas=(0.9, 0.999)（PyTorch 既定）、amsgrad=False**。`exclude_weight_decay=False` なので、**norm の gamma と bias を含む全パラメータに weight decay** がかかる。`requires_grad=False` の RoPE `freqs` は更新されない | 【確認:設定】`[TFL : …/config.yaml : 125-129, 130-135]`、【確認:コード】`[ESP90 : espnet2/tasks/abs_task.py : 967-1000 : build_optimizers]`、PyTorch 既定 |
| scheduler | `warmupreducelronplateau`: `warmup_steps=4000`、`mode=min`、`factor=0.5`、`patience=3` | 既定で `threshold=1e-4`（rel）、`cooldown=0`、`min_lr=0`、`eps=1e-8`。**lr は step_num ≤ 4000 で `base*step/4000`（線形）**。docstring の `step**-0.5` 式は実装と違う。`optimizer.step` の**後**に `scheduler.step()`（batch）を呼ぶため、**最初の更新は構築時の lr=1e-3 のまま**。plateau は `val_scheduler_criterion=(valid, loss)` を epoch 末に呼ぶ（warmup 中も `best` 等を更新）。`num_bad_epochs > patience` で下げるので、**4 回連続で悪化して初めて半減**する | 【確認:コード】`[ESP90 : espnet2/schedulers/warmup_reducelronplateau.py : 38-54, 105-114, 122-142]`、`[ESP90 : espnet2/train/trainer.py : 350-357, 754-755]` |
| 勾配クリップ | `grad_clip: 5.0`（`grad_clip_type: 2.0`）。保存 config に明記 | 勾配ノルムが非有限なら**その更新と scheduler step を飛ばす**（学習は止まらない）。損失は `sum` 縮約で値が大きい（epoch 1 の train loss ≒ 1845、終盤 ≒ 850）ため、クリップが働く頻度は高いと推測。`clip` 統計は記録されるが値は【UNKNOWN】 | 【確認:設定】`[TFL : …/config.yaml : 52-53]`、【確認:コード】`[ESP90 : espnet2/train/trainer.py : 699-711, 735-739]`、【推測】 |
| 損失 | `mr_l1_tfd`（weight 1.0）+ `si_snr`（weight 0.0）、`fixed_order` | `MultiResL1SpecLoss(window_sz=[256,512,768,1024], hop=window/2, time_domain_weight=0.5, reduction=sum, eps=1e-8, normalize_variance=False)`。推定を最適スケール `scale = <est,ref>/(<est,est>+eps)` で拡縮してから、時間領域 L1 の**和**と 4 解像度の振幅 L1 の**和**の平均（0.5 : 0.5）を取る。scale には detach が無い。STFT は矩形窓（`window=None`）、`center=True`。`si_snr` は weight 0 だが stats に `si_snr_loss`（負の SI-SDR）を出す。学習損失は 1 発話あたりの値で、バッチ平均 | 【確認:設定】`[TFL : …/yaml : 85-102]`、【確認:コード】`[ESP90 : espnet2/enh/loss/criterions/time_domain.py : 416-430, 447-498]`、`[ESP90 : espnet2/enh/loss/wrappers/fixed_order.py : 28-45]` |
| batch | `batch_size: 4`、`batch_type: folded`（チャンク iterator では使われない）、`valid_batch_size: null` | 学習 global 4 = rank あたり 1。検証も同じ（`valid_batch_size` は `batch_size` を継ぐ）。README は「各 GPU で batch 1」。yaml のコメントは「batch size 4 on 4 Quadro RTX 6000」で、global 4 と読める | 【確認:設定】`[TFL : …/yaml : 9-10]`、`[TFL : README.md : 152]`、【確認:コード】`[ESP90 : …/abs_task.py : 1860-1866]` |
| GPU 数 | README: 4 GPU。保存 config: `ngpu: 1`、`dist_world_size: 4`、`multiprocessing_distributed: true` | 4 プロセス × 1 GPU の DDP | 【確認:設定】`[TFL : …/config.yaml : 14, 20, 26]` |
| accum_grad | 1 | 1 | 【確認:設定】`[TFL : …/config.yaml : 55]` |
| max_epoch / patience | 150 / 10 | 早期停止は `epoch - best_epoch > 10`（`valid.loss` の最小）。学習曲線は約 108 epoch で終わる（停止理由は UNKNOWN）。max_step の上限は無い | 【確認:設定】`[TFL : …/config.yaml : 34-36]`、【確認:コード】`[ESP90 : espnet2/train/reporter.py : 396-418]`、`[ESP90 : espnet2/train/trainer.py : 499]` |
| keep_nbest_models / best | 5 / `best_model_criterion=[[valid,si_snr,max],[valid,loss,min]]` | **`valid.si_snr` は stats に無い**（stats のキーは `si_snr_loss`）ので `reporter.has` が偽で、`si_snr` の best は作られない。実効の基準は `valid.loss` の最小のみ。5-best の平均 `valid.loss.ave_5best.pth`（float は算術平均）が評価用重み | 【確認:設定】`[TFL : …/config.yaml : 43-50]`、【確認:コード】`[ESP90 : espnet2/enh/loss/criterions/time_domain.py : 243]`、`[ESP90 : espnet2/train/trainer.py : 419-431, 461-466]`、`[ESP90 : espnet2/main_funcs/average_nbest_models.py : 43-48, 57-105]`。最終ファイルが `valid.loss.ave_5best.pth` なのは保存 exp ディレクトリで確認【確認:設定】 |
| 検証間隔 | 毎 epoch | 毎 epoch | `[ESP90 : espnet2/train/trainer.py : 317-360]` |
| checkpoint 保存 | 毎 epoch に `<n>epoch.pth`（n-best の和集合と最新以外は削除） | 同左 | 【確認:コード】`[ESP90 : …/trainer.py : 455-470]` |
| 再開 | `resume: true`（保存 config）。`checkpoint.pth` から model / optimizer / scheduler / reporter / scaler を復元 | 同左 | 【確認:設定】`[TFL : …/config.yaml : 57]`。復元内容は【確認:コード】だが全項目を行単位では追っていない |
| seed / 決定性 | seed 0、`cudnn_deterministic: true`、`cudnn_benchmark: false`。モデル初期化直前に `set_all_random_seed(seed)` | 同左。**rank 間の seed は同じ**（`epoch + seed`） | 【確認:設定】`[TFL : …/config.yaml : 15, 29-31]`、【確認:コード】`[ESP90 : …/abs_task.py : 1250]` |
| 精度 | `use_amp: false`、`train_dtype: float32` | float32 | 【確認:設定】`[TFL : …/config.yaml : 58-59]` |
| dropout | 0.0 | 0.0 | 【確認:設定】 |
| Discriminator | 無い | — | N/A |

### 6.3 学習時間・計算量

README 記載の構成は 4 GPU（Quadro RTX 6000、24 GiB）、バッチ 1/GPU。1 epoch ≒ 14,000 更新で、150 epoch の上限は約 210 万更新。実際の所要時間は公式資料に無く、【UNKNOWN】。audio-db 側で同じ規模を走らせるには 1 プロセスの制約（8.1 節）を外す必要がある。TSUBAME で 30 分以上のジョブになるため、投入前に明示的な指示か承認が要る（AGENTS.md）。

### 6.4 ESPnet 90eed8e と 0d136c5 の実効差（DNS 経路）

`git diff 90eed8e 0d136c5` を /tmp クローンで取り、DNS の学習・評価に効くファイルの差分を読んだ（【確認:コード】）。

| ファイル | 差分（行数は追加/削除） | DNS 設定での実効への影響 |
|---|---|---|
| `egs2/dns_ins20/` | 差分なし | なし |
| `egs2/TEMPLATE/enh1/enh.sh` | `inference_args` に `--output_format wav`、`utt2fs`・`download_model`・scoring の `--gpu` を追加（+46/−4） | DNS の経路に影響なし |
| `espnet2/iterators/sequence_iter_factory.py`、`samplers/unsorted_batch_sampler.py`、`schedulers/warmup_reducelronplateau.py`、`torch_utils/set_all_random_seed.py` | 差分なし | なし |
| `espnet2/iterators/chunk_iter_factory.py`（+57/−19） | `discard_short_samples`（既定 True）、`chunk_max_abs_length`（既定 None）の追加。`W = int(state.choice(chunk_lengths, size=1).item())` へ変更（コメントは「旧来の乱数の意味を維持」） | 既定値では乱数消費・分割が同じ |
| `espnet2/tasks/abs_task.py`（+538/−84） | `category_chunk` iterator と関連引数の追加。`build_chunk_iter_factory` は `batches[rank::world_size]`、`batch_size // world_size`、`num_cache_chunks // world_size`、`num_samples_per_epoch=num_iters_per_epoch` のまま（両版の関数本体を比較） | シャード・cache・epoch の意味は同じ |
| `espnet2/enh/loss/criterions/time_domain.py`（+30/−14） | `ci_sdr` / `fast_bss_eval` の import を遅延化、`@torch.amp.autocast("cuda")` 化、`MultiResL1SpecLoss.normalize_variance` の式を `sqrt(var+eps)` に変更 | `normalize_variance=False`（既定）の `mr_l1_tfd` には影響なし。`get_magnitude` は両版とも `abs()` |
| `espnet2/enh/espnet_model.py`（+63/−15） | `always_forward_in_48k`（既定 False）、BSRNN/TFGridNetV3 の STFT 調整、`zero_weight` の損失で `no_grad` | DNS では分岐に入らない。`normalize_variance` の経路は変更なし |
| `espnet2/layers/stft.py`（+14/−69） | librosa による代替実装を削除、`torch.stft(input.float())` | CUDA / 通常の CPU 経路は同じ呼び出し |
| `espnet2/train/trainer.py`（+61/−24） | `torch.amp` への移行、`gradient_as_bucket_view` と DDP 通信フック（既定は無効）、`safe_torch_load`、**再開時にすでに早期停止条件を満たしていれば学習ループを飛ばす**判定の追加 | 通常の学習では同じ。再開時の挙動だけ違う |
| `espnet2/train/reporter.py`（+39/−11） | `state_dict` の保存形式を weights_only 互換に変更（numpy スカラー・timedelta の変換） | 数値に影響なし。90eed8e のチェックポイントを 0d136c5 で読む互換性は未確認 |
| `espnet2/main_funcs/average_nbest_models.py`（+19/−5） | `avg = states` を `avg = states.copy()` に変更（別名参照の修正）、`safe_torch_load`、deepspeed 分岐 | 有効な criterion が 1 つ（`valid.loss`）なら平均値は同じ。複数 criterion では 90eed8e の別名参照が結果を変え得る。DNS では `si_snr` が無効なので影響しない【推測】 |
| `espnet2/torch_utils/initialize.py`（+65/−98） | `chainer` 初期化の削除、`normal` の追加、コードの並べ替え | `xavier_uniform` の手続き（重み → bias → Embedding/LayerNorm/GroupNorm のリセット → `espnet_initialization_fn`）は同じ |
| `espnet2/train/preprocessor.py`（+975/−142） | 大幅な追加 | DNS は `preprocessor_conf: {}` で `EnhPreprocessor` が no-op。差分の全行は追っていない |

---

## 7. F. Validation/Test/評価

### 7.1 検証（学習中）

| 項目 | 内容 | 確度 / 根拠 |
|---|---|---|
| データ | `cv_synthetic_small`（内容・件数【UNKNOWN】） | 【確認:設定】`[TFL : …/config.yaml : 111-116]` |
| 前処理 | 学習と同じ（`EnhPreprocessor` は no-op）。チャンク分割: 64,000 / shift 0.5 / **シャッフル無し・オフセット 0**、1 発話を全チャンク化 | 【確認:コード】`[ESP90 : …/chunk_iter_factory.py : 194-197]` |
| バッチ | rank あたり 1、全 rank 並列。検証損失は重み付き平均で全 rank に集約 | 【確認:コード】`[ESP90 : …/abs_task.py : 1860-1866]`、`recursive_average` |
| 指標 | `loss`（= `mr_l1_tfd` の 1 チャンクあたり値）。`si_snr_loss`（負の SI-SDR）も stats に出る。画像では valid loss が約 865、`si_snr_loss` が約 −23.7 | 【確認:設定】`images/loss.png`、`images/si_snr_loss.png` の目視 |
| 選択 | `valid.loss` 最小 5 epoch の重み平均 | 6.2 節 |

### 7.2 テストと評価（公式）

| 項目 | 内容 | 確度 / 根拠 |
|---|---|---|
| テストセット | `tt_synthetic_no_reverb`、`tt_synthetic_with_reverb`（各 150 本・10 秒） | 【確認:コード】`[ESP90 : egs2/dns_ins20/enh1/local/dns_data_prep.sh : 98-120]`、【確認:実測】 |
| 評価重み | `valid.loss.ave_5best.pth`（README の推論コマンド） | 【確認:設定】`[TFL : README.md : 156]` |
| 推論 | **発話全体を一度に**（`segment_size` / `hop_size` は未指定で分割しない）。出力は `--normalize_output_wav true`（既定）で**ピーク 0.9 に正規化** | 【確認:コード】`[ESP90 : espnet2/bin/enh_inference.py : 104-107, 182, 362-364]`、`[ESP90 : egs2/TEMPLATE/enh1/enh.sh : 81]` |
| 採点 | `scoring_protocol="STOI SDR SAR SIR SI_SNR"`（既定）。STOI は `pystoi.stoi(ref, est, fs)` ×100、SDR/SAR/SIR は `mir_eval.separation.bss_eval_sources`、SI_SNR は `-SISNRLoss()`（zero-mean）。ESTOI は常に計算。**PESQ は `--use_pesq` を渡したときだけ**で、公式 RESULTS.md には PESQ 列が無い | 【確認:コード】`[ESP90 : espnet2/bin/enh_scoring.py : 199-250]`、`[ESP90 : egs2/TEMPLATE/enh1/enh.sh : 86]` |
| 時間整合・長さ補正 | 補正なし。ref と inf が同じ長さの前提（`enh_scoring` は長さが違えば error） | 【確認:コード】。長さ検査の行は未確認 |
| 集約 | 発話ごとの値の算術平均（`show_enh_score.sh`） | 【確認:コード】`[ESP90 : …]`。スクリプトは未読 |
| チャンク評価か | 全体評価（10 秒を一度に推論・採点） | 【確認:コード】 |

### 7.3 公式の報告値

| データ | STOI | SDR/SAR | SI-SNR | 根拠 |
|---|---|---|---|---|
| no_reverb | 98.79 | SDR 23.35 / SAR 23.35 / SIR 0.00 | 23.23 | 【確認:設定】`[TFL : …/exp/…/RESULTS.md : 24]` |
| with_reverb | 83.19 | SDR 13.17 / SAR 13.17 | 10.97 | 【確認:設定】同 25 行 |
| 論文 Table 4（Medium、15.0 M、no_reverb） | 98.8 | — | 23.3 | 【確認:論文】`[PAPER : Table 4]` |
| 論文 Table 4 PESQ-WB | 3.72（Noisy 1.58） | — | Noisy: SI-SNR 9.1、STOI 91.5 | 【確認:論文】 |

- 論文の PESQ-WB 3.72 は RESULTS.md に無く、どのコマンドで計算したかは【UNKNOWN】。
- 本監査でローカルのテストデータから noisy と clean の SI-SNR（zero-mean、150 本平均）を numpy で計算すると、no_reverb は **9.23 dB**、with_reverb は 9.19 dB だった。論文 Noisy の 9.1 とは 0.13 dB 離れる。実装差（fast_bss_eval か否か、clamp 等）の可能性があるが、確定できない。【確認:実測】【UNKNOWN】
- WER 評価（固定 ASR）は原論文の SE 評価とは別実験で、混同しない。

### 7.4 audio-db の評価実装との比較

| 項目 | 論文 / 公式 | `src/evaluation` | 判定の要点 |
|---|---|---|---|
| PESQ | 公式 RESULTS は PESQ 無し。論文は PESQ-WB。ESPnet は `pesq(fs, ref, inf, mode)`（wb/nb は fs で決まる） | `pesq` パッケージ。16 kHz は wb、8 kHz は nb。それ以外は 16 kHz にリサンプル。**長さ境界の不連続検査で `na` にする独自ガード**（しきい値 0.1）`[ADB : src/evaluation/metrics/pesq.py : 28-110]` | 同じ ITU 実装系。ガードは公式に無い |
| STOI | `pystoi.stoi` ×100 | `pystoi.stoi`、0〜1。16 kHz 以外はリサンプル `[ADB : src/evaluation/metrics/stoi.py : 28-83]` | 値の尺度が 100 倍違う |
| ESTOI | 常時計算 | `metrics.estoi: true` のときだけ（既定オフ） | 条件違い |
| SI-SNR/SI-SDR | `-SISNRLoss()`（fast_bss_eval、zero-mean） | zero-mean の SI-SDR を float64 で自前計算。零エネルギーは `na` `[ADB : src/evaluation/metrics/sisdr.py : 11-55]` | 定義は同じ。数値比較は未実施 |
| SDR/SAR/SIR | `bss_eval_sources` | **未実装** | 欠落 |
| 出力正規化 | ピーク 0.9 | 無し（推論の出力そのもの） | 条件違い（STOI・SI-SNR はほぼ不変、PESQ は不明） |
| テストセット | DNS 合成 no/with_reverb 各 150 | **名前付きセットが無い**（`data/db` が空） | 欠落 |
| 長さ・整合 | 補正なし | 長さが違えば短い方に切る（`trimmed_samples` を記録） | 公式は error、audio-db は切り詰め |
| 集約 | 発話平均 | 条件別に有限値の平均（status と reason を保持）。`na` を 0 にしない | 方針は audio-db が厳しい |

---

## 8. 現行 audio-db との項目別比較表

audio-db の主張は現行コードで確認した（既存 docs の「GO」は転記していない）。判定: MATCH（確認した範囲で公式と一致）/ PARTIAL / DIFF / UNKNOWN / N/A。設定の一致と数値の一致は区別する。

「audio-db 現状」は `configs/training/tf_locoformer.yaml`（VoiceBank 代用）と現行コードを指す。

| 項目 | 公式設定 | 公式実効動作 | audio-db現状 | 判定 | 根拠 |
|---|---|---|---|---|---|
| Dataset | DNS 2020 合成 3000 h（tr 2700 h / cv 300 h → 保存 config は `cv_synthetic_small`） | 事前生成 360,000 本の 90/10 分割 | VoiceBank `voicebank_28_train`（10,700）/ `_valid`（872）。DB 無し | DIFF | `[ADB : configs/training/tf_locoformer.yaml : 62-68]` / `[TFL : …/config.yaml : 104-116]` |
| 混合方式 | 事前生成・固定ペア | wav を読むだけ。学習中の混合なし | 録音ペアを固定レシピ（case）で登録する設計。DNS 生成は未実装 | PARTIAL（方式は同系、生成は未実装） | `[ADB : src/training/data/dataset.py : 181-223]` |
| 雑音選択 | shuffle 済み一覧の巡回、複数雑音の連結 | 生成時のみ。非決定的 | recipe は `sources.noise` 1 本 + `noise_start`。連結は不可 | DIFF | `[ADB : src/data/recipe.py : 24, 89-93, 139-164]` |
| SNR | 整数 −5…14 dB（`randint` 上限除外）、RMS（無音含む） | 生成時に 1 本ごと | recipe は case ごとに固定 `snr_db`（float）。定義 `rms_v1` は全サンプル平均二乗で同系。分布は保持しない | PARTIAL | `[ADB : src/evaluation/audio/mixing.py : 3-15, 51-66]` |
| ゲイン | 最終 RMS −35…−16 dBFS（整数）、clip 時は全体縮尺 | 生成時のみ | `peak_normalize` のみ。RMS 目標や乱数レベルは無い。学習時は `rms_normalize: true` で RMS=1 | DIFF | `[ADB : src/data/recipe.py : 190-194]`、`[ADB : src/training/data/dataset.py : 359-364]` |
| 残響 / RIR | 学習なし。テスト with_reverb は残響付き参照 | — | recipe は `rir` 演算を持つが DNS で使わない。録音ペアの参照を `reverberant_clean` にできない | N/A（学習）/ DIFF（with_reverb 参照） | `[ADB : src/evaluation/audio/render.py : 158-174]` |
| リサンプリング | 学習データは 16 kHz。noise の約 0.2 %（標本 4/2,000 件）のみ librosa（版 UNKNOWN） | 生成時のみ | 読み込み時 `torchaudio.functional.resample`。DNS の clean は 16 kHz なので無関係 | N/A（clean）/ PARTIAL（noise） | `[ADB : src/training/data/dataset.py : 331-347]` |
| 正規化 | データ側は生成時のレベル調整のみ。`speech_volume_normalize: null`。モデル内 `normalize_variance` | 学習時の追加正規化なし | **データ側 RMS=1 を強制**（`rms_normalize: true`）。モデル内の標準偏差正規化は同じ（不偏 std、1e-6 未満は拒否） | データ側 DIFF / モデル内 MATCH | `[ADB : configs/training/tf_locoformer.yaml : 68]`、`[ADB : src/se/models/tf_locoformer/model.py : 55-64]` |
| Crop | チャンク 64,000 / shift 0.5、30 秒では offset 常に 0 | `Z=randint(0,(L-W)%S+1)` | チャンク 32,000 / shift 0.5。式は同じ | PARTIAL | `[ADB : src/training/data/chunk_plan.py : 28, 49-52]` |
| Padding | 無し。短発話は破棄 | `lg < L` | 同じ（`length <= chunk_length` を破棄）。非チャンク経路の `_segment` はゼロ詰めだが使わない | MATCH（コード読解） | `[ADB : src/training/data/chunk_plan.py : 46-47]` |
| 音声長制限 | 0.1 s 超 31 s 未満 | `enh.sh` の `awk` | 無し | DIFF（DNS 生成物が 30 秒なので実害は小さいと推測） | `[ESP90 : egs2/TEMPLATE/enh1/enh.sh : 491-499]` |
| Shuffle / Sampler | `SequenceIterFactory`（周期ごとの `RandomState(real_epoch+seed)`）+ cache 内 shuffle | rank シャード `[rank::4]`、cache 256 | `select_utterances`、`chunk_batches` が式を再現。1 プロセスで cache 1024 | PARTIAL | `[ADB : src/training/data/chunk_plan.py : 19-80]` |
| Epoch 定義 | rank あたり 1000 元発話（全体 4000）、約 14,000 更新 | 5.2 節 | 10,700 元発話、更新数は実長次第 | DIFF | 5.3 節の表 |
| Batch | global 4（rank あたり 1）、蓄積 1 | 実効損失は 4 チャンクの平均 | global 4、micro 4、蓄積 1 を強制 | PARTIAL（global は同じ、構成が違う） | `[ADB : src/training/core/trainer.py : 355-356]` |
| DDP / accumulation | 4 rank DDP、`accum_grad 1` | 勾配は rank 平均 | 1 プロセスのみ。`world_size != 1` はエラー | DIFF | `[ADB : src/training/core/trainer.py : 348-349]` |
| 損失 | `mr_l1_tfd`（時間 0.5 + 4 解像度、sum、最適スケール、si_snr 重み 0） | 6.2 節 | `multi_resolution_l1`: 式・窓・矩形窓・`center` 相当・`sum` 縮約が同じ | MATCH（コード読解）。**ESPnet の損失クラスとの数値比較は TF-Locoformer では未実施** | `[ADB : src/training/se/spectral.py : 20-91]`、`[ADB : src/training/se/tf_locoformer.py : 110]` |
| Optimizer | AdamW、lr 1e-3、eps 1e-8、wd 1e-2、betas 既定 | 全パラメータに wd | AdamW lr 1e-3、betas (0.9, 0.999)、wd 0.01。eps は PyTorch 既定 1e-8 | MATCH | `[ADB : src/training/se/optim.py : 20-32]` |
| Scheduler | warmup 4000（線形）+ plateau（factor 0.5, patience 3） | 最初の更新は lr=1e-3。4 回連続悪化で半減 | 同じ式。公式クラスとの比較テストあり（公式 90eed8e と checkout のファイルが同一）。**1 epoch の更新数が違うので warmup と plateau の相互作用は公式と別になる** | MATCH（スケジューラ単体）/ 運用は PARTIAL | `[ADB : src/training/se/optim.py : 35-134]`、`tests/training/test_tf_locoformer_scheduler.py` |
| 勾配クリップ | 5.0、L2 | 非有限は更新と scheduler step を飛ばす | 5.0、L2。**非有限勾配は飛ばさず（損失の非有限は `TrainingError` で停止）** | PARTIAL | `[ADB : src/training/se/optim.py : 191-194]`、`[ADB : src/training/se/tf_locoformer.py : 111, 116-119]` |
| 初期化 | `xavier_uniform`（実効手続きは 6.2 節） | 同左 | 同じ手続きを自前関数で実装。**公式 separator に同じ関数を当てた比較**であり、ESPnet の `initialize` 自体との比較ではない。seed は 42（公式 0） | MATCH（手続き）/ 数値同一性は別 | `[ADB : src/training/se/initialization.py : 18-32]` |
| 最大 epoch / step | 150 epoch。step の上限なし | 保存された学習曲線は約 108 epoch で終わる（停止理由は UNKNOWN） | `max_epochs: 150`、`max_steps: 2000000`（安全上限） | PARTIAL | `[ADB : configs/training/tf_locoformer.yaml : 22, 37]` |
| Validation（データ） | `cv_synthetic_small` | DNS 合成 | `voicebank_28_valid` | DIFF | 同上 |
| Validation（前処理） | 学習と同じ（no-op）、offset 0 | — | `rms_normalize` あり、offset 0（`shuffle=False`） | PARTIAL | `[ADB : src/training/core/trainer.py : 427-441]` |
| Validation（バッチ） | rank あたり 1 | 重み付き平均で集約 | batch 4、leftover は捨てる、**バッチ値の単純平均**（全バッチが同じ大きさなので結果は同じ） | PARTIAL | `[ADB : src/training/core/trainer.py : 854-888]` |
| Best / early stop | valid loss 最小、`epoch - best > 10`。実効 `si_snr` 基準は無効 | `ave_5best` を評価に使う | `valid.loss.best.pt`、`valid.si_snr.best.pt`（SI-SDR を**最大化**する研究上の選択）、5-best の epoch ファイルを保持。**平均は作らない**。早期停止の式は同じ | PARTIAL | `[ADB : src/training/core/trainer.py : 477-523]` |
| Resume | `resume: true`（ESPnet の `checkpoint.pth`） | model / optimizer / scheduler / reporter | `step_*.pt`（task state、RNG、epoch、`micro_batches_consumed`、`chunk_history`、データ識別子）。ESPnet と相互再開は不可 | PARTIAL | `[ADB : src/training/core/trainer.py : 591-620]` |
| 評価指標 PESQ | 論文は PESQ-WB。RESULTS には無い | 7.2 節 | `pesq`（wb/nb）、境界ガード付き | PARTIAL | 7.4 節 |
| 評価指標 STOI | `pystoi` ×100 | — | `pystoi`、0〜1 | PARTIAL | 同上 |
| 評価指標 SI-SDR | zero-mean SI-SNR | fast_bss_eval | zero-mean を自前計算 | PARTIAL（数値比較未実施） | 同上 |
| 評価指標 SDR/SAR/SIR | `bss_eval_sources` | — | 未実装 | DIFF | — |
| Test | DNS 合成 no / with reverb、各 150 | 全体推論 | 名前付きセット無し | DIFF | `[ADB : src/data/corpora_catalog.py : 26]` |
| 評価重み | `ave_5best` | 5 epoch の重み平均 | 単一重み（`generator.pt` / best） | DIFF | 同上 |
| 推論 | 全体一括、出力ピーク 0.9 | — | 全体一括（チャンク分割なし）、出力正規化なし | PARTIAL | `[ADB : src/se/models/tf_locoformer/inference.py]` |
| seed / 決定性 | seed 0、`cudnn_deterministic: true` | 全 rank 同じ seed | seed 42、`deterministic: false` | DIFF（比較条件ではない） | `[ADB : configs/training/tf_locoformer.yaml : 18-19]` |
| 精度 | float32、AMP 無し | — | float32（AMP 無し） | MATCH | `rg autocast src/training` が空 |

---

## 9. 差異一覧

影響度: Critical = プロトコル・学習結果に直接影響 / Major = 性能・比較条件に影響し得る / Minor = 限定的 / Unknown。実装状態は「未実装」「実装済みだが公式一致は未検証」「実装済みで検証済み（範囲を明記）」。優先度案: P0 = 公式プロトコル再現に不可欠 / P1 = 再現性・検証可能性に重要 / P2 = 性能・効率・運用。

| ID | 項目 | 影響度 | 実装状態 | 優先度案 | 対象ファイル / モジュール |
|---|---|---|---|---|---|
| TFL-D01 | 学習・検証データが DNS でない。DNS 学習セット・評価セットが DB に無い | Critical | audio-db に未実装 | P0 | `src/data/corpora/`、`configs/db/`、新規の学習セット登録、`configs/training/`（DNS 用 yaml を新設する提案） |
| TFL-D02 | 公式の事前生成（連結・clean 切り出し・RMS −25 dBFS・最終レベル乱択・16 bit）を再現する生成手段が無い | Critical | audio-db に未実装 | P0 | 生成スクリプト（`data/corpora/DNS2020` の公式スクリプトを read-only で呼ぶ提案）、`data/staging/` への出力、登録コード |
| TFL-D03 | `rms_normalize: true`（RMS=1）が公式の `speech_volume_normalize: null` と逆。損失の絶対スケールとクリップの効き方が変わる | Major | 実装済み（切替は `data.rms_normalize`）だが TF-Locoformer の公式モードは未設定・未検証 | P0 | `configs/training/tf_locoformer.yaml`、`src/training/data/dataset.py` |
| TFL-D04 | epoch 定義: 公式 rank あたり 1000 元発話、約 14,000 更新。audio-db は全件 | Major | 実装済み（元発話数の式は再現）だが公式一致は未検証 | P0 | `configs/training/`、`src/training/data/chunk_plan.py` |
| TFL-D05 | 4 rank DDP（rank あたり batch 1、シャード `[rank::4]`、cache 256）を再現できない | Major | audio-db に未実装（`world_size != 1` は拒否） | P1 | `src/training/core/trainer.py`（`_run_chunk_epochs`）、`src/training/data/chunk_plan.py`（rank 分割の提案） |
| TFL-D06 | 評価重みの 5-best 平均（`ave_5best`）が無い | Major | audio-db に未実装 | P1 | `src/training/core/trainer.py::_retain_chunk_models`、チェックポイント平均の新規ユーティリティ（提案） |
| TFL-D07 | DNS 合成テスト（no/with reverb 各 150）が評価セットに無い。with_reverb の参照は reverberant（recorded recipe の参照名が `dry_clean` 固定） | Major | audio-db に未実装 | P1 | `src/data/recipe.py::_check_recorded`、`src/evaluation/audio/render.py::_recorded`、評価セット定義 |
| TFL-D08 | `cv_synthetic_small` の構成が不明。検証データの再現ができない | Major | 公式側が UNKNOWN | P1 | 12 章。追加調査案あり |
| TFL-D09 | SDR/SAR/SIR、ESTOI 常時、出力ピーク 0.9 正規化が無い | Minor | audio-db に未実装 | P2 | `src/evaluation/metrics/`、`src/se/` の書き出し |
| TFL-D10 | 非有限勾配は公式がスキップ、audio-db は停止（損失）または素通し（勾配ノルム） | Minor | 実装済みだが公式一致は未検証 | P2 | `src/training/se/tf_locoformer.py`、`src/training/se/optim.py::clip_gradients` |
| TFL-D11 | チャンク cache 長（公式 256/rank、audio-db 1024）と rank 構成の違いによる局所シャッフルの差 | Minor | 実装済みだが公式一致は未検証（1 プロセスでは再現不可） | P2 | `src/training/data/chunk_plan.py` |
| TFL-D12 | seed / `cudnn_deterministic` の違い（公式は seed 0、決定的） | Minor | 設定の違い。公式一致は意図しない | P2 | `configs/training/` |
| TFL-D13 | `rotary-embedding-torch` の版差（公式 0.6.1、audio-db 0.8.3）。fp32 で数値差が無いと推測 | Unknown | 公式重みでの両版比較は未実施 | P2 | `.venvs/*`、`.agents/DEVELOPMENT.md:62` |
| TFL-D14 | 公式のゼロ分散入力は NaN になる。audio-db は拒否（学習）/ 零波形（推論） | Minor | 実装済みで意図的 | P2 | `src/se/models/tf_locoformer/model.py` |
| TFL-D15 | 既存 docs の誤りと、存在しないファイルへの参照 | Major | 文書の修正が未実施 | P1 | `docs/training/TF_LOCOFORMER.md`、`docs/se/MODELS.md`、`.agents/DECISIONS.md`、`tests/training/test_tf_locoformer_protocol.py` |
| TFL-D16 | 損失・初期化の公式一致が「ESPnet 本体のクラス」との比較になっていない（standalone separator 側 + audio-db 自前関数との比較） | Major | 実装済みだが ESPnet クラスとの数値一致は未検証 | P1 | `tests/training/`（比較テストの追加提案）、`src/se/official/espnet` を import する検証スクリプト |

---

## 10. DB中心の新データ基盤に必要な情報(1〜10)

方針（依頼文）: 元音声はファイルシステム、索引・分割・生成条件・プロトコルは DuckDB（新学習用 DB 候補 `data/db/reproduction.duckdb`、評価用 DB と責務分離）、公式が固定ペアなら維持する。今回は DB スキーマ・ローダーを実装しない。

### 10.1 DB に登録すべき音声素材

| 素材 | 内容 | 備考 |
|---|---|---|
| DNS clean | 65,348 ファイル、31 秒、16 kHz、モノラル。reader ID（ファイル名から取れる）を属性にする | ローカルに実在【確認:実測】。`corpora.audio_files` への登録状況は DB が無く確認不可 |
| DNS noise | 65,302 ファイル、1〜10 秒。16 kHz が大半 | 同上 |
| 生成物（公式スクリプトの出力） | `noisy_<…>_fileid_N.wav`、`clean_fileid_N.wav`、`noise_fileid_N.wav` の 3 本 × 本数 | 3000 h で約 1 TB。ESPnet のコメントが 100 h = 33 GB と書く（`[ESP90 : egs2/dns_ins20/enh1/local/dns_create_mixture.sh : 68]`）ので 30 倍の 990 GB 前後（【推測】）。参考に 100 h 版（33 GB）から始める案がある |
| 生成ログ | `source_files.csv`、`clipped_files.csv`、`low_activity_files.csv`（`utils.write_log_file`） | 生成条件の証拠として保存 `[DNS : …multiprocessing.py : 321-323]` |
| テスト | 合成 no_reverb / with_reverb の clean / noisy 各 150 本（10 秒） | ローカルに実在【確認:実測】 |
| 入手できないもの | 公式が生成した 3000 h の実物、`cv_synthetic_small` の一覧、`exp/enh_stats_16k` の shape ファイル、with_reverb の RIR 本体、公式学習ログ | 12 章 |

### 10.2 clean / noise / RIR の役割

- clean: 31 秒の読み上げ音声。生成時に 30 秒へランダムに切り出す。教師は切り出した clean を RMS −25 dBFS にしたのち最終レベル係数を掛けたもの。
- noise: 1〜10 秒の雑音クリップ。連結して 30 秒にする。
- RIR: 学習では使わない。テスト with_reverb は Microsoft 内部の RIR（本体は配布なし）。with_reverb の clean は残響付きの参照。

### 10.3 分割と対応関係

- 生成物は 3 本の wav が `fileid_N` で結び付く（clean_fileid_N、noise_fileid_N、noisy 名の末尾）。
- 学習 / 検証: uttid（`reader_<ID>_fileid_<N>`）の辞書順で先頭 9/10 を学習、残りを検証（3.2 節）。DB では「分割ルール」と「分割結果の一覧（case_id の集合のハッシュ）」の両方を保存する。
- `cv_synthetic_small` は部分集合（作り方が UNKNOWN）。
- テスト: `fileid_N` で noisy と clean が対応する。

### 10.4 固定して保存すべき生成条件

公式の生成は 1 回で終わる。DB には次を「生成プロトコルの定数」として保存する。

- `sampling_rate=16000`、`audio_length=30`、`silence_length=0.2`、`total_hours`（3000 または縮小）、`clean_activity_threshold=0.6`、`noise_activity_threshold=0.0`、`snr_lower=-5`、`snr_upper=15`（上限除外）、`target_level_lower=-35`、`target_level_upper=-15`（上限除外）、`is_test_set=False`、クリップしきい値 0.99、`MAXTRIES=50`、`MAXFILELEN=100`、`activitydetector` の係数（窓 50 ms、`energy_thresh=0.13`、`a=-1`、`b=0.2`、`alpha_att=0.8`、`alpha_rel=0.05`）。
- 乱数: `np.random.seed(2)`、`random.seed(3)`、`PROCESSES`（`cpu_count`）。ただし multiprocessing のため**同じ値でも再現しない**。再現性は生成物のハッシュと生成ログで担保する。
- 実行環境: numpy / librosa / soundfile / Python の版（公式は無指定で【UNKNOWN】）。
- 1 本ごとの実現値: 使った clean ファイル、noise ファイル群、各切り出し位置、SNR、最終レベル、`fileid`。生成ログが一部（ファイル名の一覧）しか持たないため、**SNR と最終レベルは noisy のファイル名（`_snr<N>_`）と、波形から逆算する**ことになる。

### 10.5 学習時にサンプリングすべき生成条件（確率分布のパラメータ）

**公式 DNS 学習では無い。** 混合条件はすべて生成時に確定している。学習時に乱数を使うのは次の 3 つだけで、いずれも `(seed, epoch, rank シャード, ファイル一覧の順序)` から決まる。

1. 発話の順序（周期ごとの `RandomState(real_epoch + seed)`）。
2. チャンクキャッシュ内の並べ替え（`RandomState(epoch + seed)`）。
3. チャンクの開始オフセット Z（30 秒発話では常に 0）。

オンライン混合を行う別プロトコル（他モデルや DNS 以外の拡張）の分布パラメータは、本モデルの公式再現には不要である。

### 10.6 データローダー固有の処理

- ESPnet `ChunkIterFactory` の分割・cache・シャッフル（5.1 節）。
- `RawSampler` + `real_epoch` / `offset` の窓。
- rank シャード `[rank::world_size]`、`batch_size // world_size`、`num_cache_chunks // world_size`、「どれかの rank が尽きたら全体停止」。
- audio-db 独自: 長さのキャッシュ（`corpora.audio_files` から）、`rms_normalize`、読み込み時のリサンプル、`case_id` とレシピ SHA-256 の照合。

### 10.7 共通化できる処理

- チャンク計画（`src/training/data/chunk_plan.py`）。TF-GridNet の公式モードと同じ ESPnet のチャンク iterator を再現している。
- `mr_l1_tfd`（TF-GridNet と共通）、AdamW、勾配クリップ、`initialize`（xavier_uniform）。
- 評価指標（PESQ / STOI / SI-SDR）と `na` の扱い。
- ファイルの SHA-256 検証、`case_id`、DB 登録。
- 生成物の登録（recorded recipe）。

### 10.8 公式動作維持に専用実装が必要な処理

- DNS 合成スクリプトの全工程（連結・活動率検査・RMS 正規化・最終レベル・クリップ縮尺・16 bit）。公式スクリプトを read-only で呼ぶ案と、同じ式を Python で再実装する案がある（研究判断）。
- 4 rank を 1 プロセスで模擬する chunk 計画（rank ごとのシャード、cache 256、batch 1 を交互に並べた global batch 4）。
- `ave_5best`（5 epoch の算術平均、整数型は加算のみ）。
- `WarmupReduceLROnPlateau` の細部（最初の更新は lr 1e-3、plateau を warmup 中も更新）。すでに audio-db に実装済みで、公式クラスと比較済み。
- DNS 評価: `normalize_output_wav`（ピーク 0.9）、`bss_eval_sources`、with_reverb の参照定義。

### 10.9 再現性確保のために保存すべき状態

- 生成側: プロトコル ID と全定数（10.4）、公式スクリプトの commit（70f1928）、実行環境の版、生成物のファイル一覧とハッシュ、生成ログ 3 本。
- 学習側: seed、`world_size`、`rank`、epoch、**サンプラー位置**（`real_epoch`、`offset`。`epoch` と式から導けるが、ファイル一覧の順序に依存するため一覧のハッシュも保存）、**chunk 計画のハッシュ**（epoch ごとの発話順 + 各発話の長さ）、ファイル一覧のハッシュ（`case_ids_sha256`）、**長さ一覧のハッシュ**、RNG 状態、optimizer / scheduler の状態（`step_num`、`best`、`num_bad_epochs`、`cooldown_counter`、`last_epoch`）、epoch ごとの valid loss 履歴（n-best と早期停止の再判定に必要）、n-best の epoch ファイル、設定のハッシュ。
- audio-db の現行 checkpoint が持つもの: `global_step`、`epoch`、`micro_batches_consumed`、task state（optimizer、scheduler）、`rng`、`config`、`batch_plan`、`dataset.identity()`（`case_ids_sha256`、サンプルレート、segment、seed、`rms_normalize`）、`chunk_history`（`[ADB : src/training/core/trainer.py : 591-612]`）。**不足**: 長さ一覧と chunk 計画のハッシュ、epoch ごとの発話順のハッシュ、生成プロトコル ID、スケジューラの `best` 等を含む全状態の検証（`state_dict` に含まれるが、再開比較は縮小モデルのみ）。

### 10.10 既存 recipe JSON v1 で表現できない情報

「固定 1 ケースのレシピ」と「オンライン生成の確率プロトコル」は別物である。TF-Locoformer の公式は前者（事前生成の固定ペア）だけを必要とし、後者は要らない。

#### 固定 1 ケースとしての表現可能性

公式の事前生成ペアは、**operations を空にして `sources.clean` と `sources.noisy` を登録する recorded recipe**（参照 `dry_clean`）で表現できる。`case_id` はレシピの SHA-256 なので、生成物が固定されていれば成立する。360,000 本でも構造上は問題ない。

ただし次の理由で、**公式の生成過程そのものは表現できない**。

| 公式の処理 | recipe v1 | 不足の内容 |
|---|---|---|
| SNR の分布（整数一様 −5…14） | `snr_db` は 1 値 | 分布はケースのレシピには入れず、プロトコル側に置く。実現値は `snr_db` に保存できる |
| 雑音選択の確率・巡回 | `sources.noise` は 1 ファイル | 選択規則はプロトコル側の情報 |
| 複数雑音の連結（0.2 秒の無音を挟む） | 不可（`additive_noise.source` は `noise` 固定で、noise は 1 本） | **連結・間の無音の演算が無い**。短い雑音は `align_noise` が同じ雑音を繰り返す（別動作） |
| 複数 clean の連結 | 不可 | DNS は 1 ファイルで足りるが、一般の連結は無い |
| clean のランダム切り出し | `output.max_seconds` は先頭からの切り詰めのみ | **clean の開始位置（切り出し）** を表す項目が無い |
| レベル（RMS −25 dBFS、最終 −35…−16 dBFS） | `peak_normalize` の `limit` のみ | **RMS / dBFS / LUFS / ゲイン演算が無い** |
| clipping（0.99 超の全体縮尺 / 棄却） | `peak_normalize`（0.99 限界の全体縮尺は近い）。棄却と再生成は無い | 棄却は生成プロトコル側 |
| 活動率による clean の棄却 | 無し | プロトコル側（生成時のみ） |
| 16 bit 量子化 | 無し（レンダリングは float） | **量子化演算が無い**。生成物を recorded として登録すれば不要 |
| リサンプル方式 | `torchaudio.functional.resample` 固定でレシピに記録なし | 方式・版をレシピかプロトコルに持たせる必要。公式は librosa（版 UNKNOWN） |
| RIR の適用確率・選択 | `rir` は入れれば必ず適用 | 確率・選択規則はプロトコル側（DNS 学習では不要） |
| with_reverb の参照 | recorded recipe は参照が `dry_clean` 固定（`_check_recorded`）、描画も `reverberant_clean` を拒否 | **録音ペアの参照を残響付きと宣言できない** |
| `seed` | `additive_noise.seed` は `noise_start` が保存されていれば使われない | 実現値は保存済みなので問題ない |

#### オンライン生成の確率プロトコルとしての表現可能性

recipe v1 は 1 ケースの固定記述であり、確率分布・乱数ストリーム・エポックごとの再サンプリングを持たない。**公式 DNS はこれを要しない**が、別プロトコル（他モデル）が必要とするなら、次を別のテーブルで持つ必要がある。

- 分布: SNR、ゲイン、雑音選択（一様 / 巡回）、RIR 適用確率、雑音の個数、clean の切り出し。
- 再サンプリングの単位（毎サンプル / epoch / バッチ）。
- 乱数ストリームの定義（seed、epoch、サンプル ID から決まる純関数）。

#### 提案（実装しない）

1. 公式の DNS 再現は、公式スクリプトの出力を `data/staging/` に置いたうえで、recorded recipe（clean + noisy）として登録する。生成プロトコルは別テーブルに保存し、ケースと紐づける。
2. 生成の過程を DB 内で再実行できるようにしたい場合は、`concat`、`crop_clean`、`rms_normalize`、`gain_db`、`quantize` などの演算を recipe v2 として追加する。ただし、公式の生成が非決定的なので、再実行しても公式の生成物とは一致しない。
3. `reference` に「録音ペアの reverberant clean」を許す変更（`_check_recorded` と描画）。

---

## 11. 論文と公式コードの矛盾、既存ドキュメント(audio-db内)の不整合・誤り

### 11.1 論文と公式コード・設定の矛盾

| # | 論文 | 公式コード・設定 | 再現上の影響 |
|---|---|---|---|
| P-1 | DNS は 21,000 step を 1 epoch とする `[PAPER : §3.1 脚注]` | 設定から計算できる 1 epoch は約 14,000 更新（5.2 節）。21,000 は `num_iters_per_epoch=1500` に相当 | 学習量・warmup の位置・plateau 判定の頻度が変わる。論文時の設定が別だった可能性【推測】 |
| P-2 | 「3 epoch 改善しなければ lr を 0.5 倍」 | `num_bad_epochs > patience`（=3）なので 4 回連続悪化で半減。早期停止も `> 10`、すなわち 11 epoch 後 | 実装を正とする |
| P-3 | Macaron の FFN に 1/2 係数 | コードに 1/2 が無い（公式コードのコメントが認める） | 公式重みはこの実装で学習。1/2 を入れると重みと合わない |
| P-4 | SI-SNR 23.3、STOI 98.8、PESQ-WB 3.72（no_reverb） | RESULTS.md: SI_SNR 23.23、STOI 98.79、PESQ 列なし。23.23 を丸めると 23.2 であり、論文の 23.3 と合わない（SDR は 23.35） | 論文値と RESULTS.md の測定条件が同じか不明 |
| P-5 | Noisy の SI-SNR 9.1 | ローカルのテストデータ 150 本の noisy–clean SI-SNR 平均は 9.23 | 0.13 dB の差の原因は不明 |
| P-6 | 検証 300 h | 保存 config は `cv_synthetic_small`。README が縮小を推奨 | 検証データの再現不可 |
| P-7 | SNR の範囲は本文に明示なし（README: −5〜15） | `randint(-5, 15)` は上限除外で最大 14 の整数 | 生成の再現では上限 14 を使う |
| P-8 | 5 つの最良 checkpoint の平均を評価に使う | 一致。ただし `best_model_criterion` の `si_snr` は無効 | 実効は valid loss のみ |
| P-9 | batch 4、入力 4 秒 | 一致（global 4 = rank あたり 1、64,000 サンプル）。yaml のコメントは「4 GPU で batch size 4」と読める | 混同しやすい |
| P-10 | — | ESPnet `WarmupReduceLROnPlateau` の docstring は逆平方根の式だが、実装は線形 | docstring に従うと誤る |
| P-11 | — | ESPnet 標準の `dns_ins20/enh1/run.sh` は BLSTM 設定、`ngpu 2`、`--use_noise_ref true`、`--inference_model valid.loss.best.pth`。TF-Locoformer の保存 config の学習データに `noise_ref1` が無く、実際の学習コマンドは run.sh と違う | 実コマンドは README の `./run.sh --stage 6 … --ngpu 4` と保存 config から推定するしかない【UNKNOWN】 |
| P-12 | — | ESPnet の `dns_create_mixture.sh` の既定は 100 h・SNR 0〜40。論文の 3000 h・−5〜15 は README の `sed` による変更 | 再現には `sed` の手順が必要 |

### 11.2 audio-db の既存 docs・設定・テストの不整合

| # | 場所 | 内容 | 根拠 |
|---|---|---|---|
| D-1 | `docs/se/MODELS.md:21` | 論文の arXiv ID が `2409.16674`。正しくは `2408.03440` | 【確認:論文】 |
| D-2 | `docs/training/TF_LOCOFORMER.md:49`、`tests/training/test_tf_locoformer_protocol.py:62-63` | 「スモークは `tf_locoformer.yaml`、研究設定は `tf_locoformer_voicebank.yaml`」と書く。**`tf_locoformer_voicebank.yaml` は無く**、`tf_locoformer.yaml` が研究設定（VoiceBank、`max_steps 2000000`）。同ファイルの 2 行目は「`tf_locoformer.yaml` のスモークではない」と自己言及している | `ls configs/training`、`[ADB : configs/training/tf_locoformer.yaml : 1-2]` |
| D-3 | `tests/training/test_tf_locoformer_protocol.py:62-63` | `configs/training/tf_locoformer_voicebank.yaml` と、2 step のスモーク `tf_locoformer.yaml` を読む。現ツリーではファイルが無く、また `tf_locoformer.yaml` は 2 step ではないため、`test_research_yaml_is_separate_from_the_smoke_file` は失敗すると考えられる（未実行） | 【推測】（ファイルの有無は確認済み） |
| D-4 | `docs/training/TF_LOCOFORMER.md`（学習アルゴリズムの表） | 「バッチ: 4 GPU で各 forward batch 4」。公式は global 4（rank あたり 1）。「epoch: `num_iters_per_epoch: 1000`」は rank あたりで、全体では 4000 発話 | 【確認:コード】`[ESP90 : …/abs_task.py : 1860-1879]`、`[TFL : README.md : 152]` |
| D-5 | `docs/training/OFFICIAL_TRAINING_AUDIT.md`（11 行） | `*_seed17.yaml` 5 本と `trial100/` が「本格学習の設定」として挙がるが、`configs/training/` に無い。TF-Locoformer の記述は同書に無い | `ls configs/training` |
| D-6 | `docs/training/VERIFICATION.md` | TF-Locoformer の記録が無い（TF-GridNet などの節のみ） | `rg -c locoformer` が 0 |
| D-7 | `docs/training/TF_LOCOFORMER.md`、`scripts/setup/doctor.py:206` | `data/checkpoints/tf-locoformer/dns/valid.loss.ave_5best.pth` に重みがあると書くが、`data/checkpoints/tf-locoformer/` は現ツリーに無い | `ls data/checkpoints` |
| D-8 | `AGENTS.md`、`.agents/DATA.md` | `data/db/resource.duckdb` があるとの前提だが、現ツリーの `data/db/` は空。`data/staging/` も空 | `ls data/db data/staging` |
| D-9 | `docs/training/TF_LOCOFORMER.md`（検証の記述） | 損失・初期化・学習 step の「公式との一致」は、standalone separator に audio-db の損失・初期化関数を当てた比較で、**ESPnet の `MultiResL1SpecLoss` や `initialize` との数値比較ではない**（TF-GridNet は `docs/training/VERIFICATION.md` の公式エントリ表で `MultiResL1SpecLoss` と比較している）。本監査のコード読解では式が一致するが、数値一致は未検証 | `tests/training/test_tf_locoformer_protocol.py:87`（init）、`:127`（5 step） |
| D-10 | `.agents/DEVELOPMENT.md:62` | `rotary-embedding-torch==0.8.3`。公式は 0.6.1 | `[TFL : requirements.txt]` |
| D-11 | `docs/training/TF_LOCOFORMER.md`（推論の表） | 公式との波形一致は同じ `rotary-embedding-torch` 版で両側を動かした結果で、0.6.1 との一致ではない | 同上 |

---

## 12. UNKNOWN一覧

| # | 内容 | 調べた資料 | 確定できなかった理由 | 追加調査案 |
|---|---|---|---|---|
| U-1 | `cv_synthetic_small` の構成（件数、選び方） | 保存 config、README、`copy_files_to_espnet.sh`、`dns_data_prep.sh`、ESPnet `egs2/dns_ins20`（sparse）で `synthetic_small` を検索 | 作成スクリプトが公式リポジトリに無い。README は縮小を勧めるだけ | MERL への問い合わせ。もしくは 36,000 件の `cv_synthetic` から適当な部分集合を作り、その旨を明記 |
| U-2 | 公式が生成した学習データの実物と、生成時の環境（librosa 版、CPU 数） | DNS の `requirements.txt`、スクリプト | 版が無指定。生成物は非公開。multiprocessing が非決定的 | 生成物の統計（SNR・レベル・長さ）で再現確認。環境の版を固定して記録 |
| U-3 | 生成物の長さが全て 480,000 サンプルか | コードの読解（clean 31 秒 → 30 秒に切る） | 生成物が無い | 小規模再生成（100 h）で長さを確認 |
| U-4 | 学習データの一覧順（`speech_mix_shape` のファイル順） | `abs_task.py`、`enh.sh` stage 5 | collect_stats の集約順は実行物依存 | 同じ手順で生成・集計して確認 |
| U-5 | 実際に学習した epoch 数、更新数、停止理由（早期停止か打ち切りか） | `images/loss.png`、`si_snr_loss.png` の目視（約 108 epoch で終わる） | `train.log` と `RESULTS` に epoch 数・停止理由が無い。valid loss は終盤も微減 | 公式ログの入手。画像の点数を数える |
| U-6 | 論文 21,000 step/epoch の設定 | 論文脚注、config | 公式 config は 1000。差の原因が不明 | 論文著者への確認。`num_iters_per_epoch=1500` を仮説とする |
| U-7 | 論文の PESQ-WB 3.72 の算出コマンド | RESULTS.md、`enh.sh`、`enh_scoring.py` | RESULTS に PESQ 列が無い | `--use_pesq true` を付けた評価の再実行（GPU 不要）。公式重みの実体が必要 |
| U-8 | 論文の Noisy SI-SNR 9.1 と実測 9.23 の差 | ローカルデータでの numpy 計算 | 測定実装（fast_bss_eval の clamp 等）の差を確認していない | `fast_bss_eval` で同じ計算を行い比較 |
| U-9 | 学習の実コマンド | README、`run.sh`（BLSTM 用）、保存 config | run.sh と保存 config が合わない（P-11） | 保存 config と `enh.sh` の引数から復元 |
| U-10 | `clip`（勾配クリップ）が働く頻度、実際の学習時間 | 保存ファイルに `train.log` が無い | 公式ログなし | 再学習での計測 |
| U-11 | DNS の clean 話者とテスト clean 話者の重複 | DNS README（local） | 話者対応表が無い | DNS Challenge の文書を確認 |
| U-12 | with_reverb の clean が残響付きか（推測の確認） | ローカルデータの統計 | 生成条件の文書が無い | DNS Challenge 論文（Reddy et al., Interspeech 2020）の確認 |
| U-13 | 公式重み（`.pth` の実体）と audio-db の独立実装との数値一致 | audio-db の docs の記録のみ | 実体が無い。今回は読み込まない | 実体を入手して `scripts/se/verify_tf_locoformer.py` を実行（別作業） |
| U-14 | `rotary-embedding-torch` 0.6.1 と 0.8.3 の fp32 数値差 | ソース差分 | 実行していない | 小さい入力で両版を比較 |
| U-15 | audio-db 側の 1 epoch の更新数 | `chunk_plan.py` の式 | `voicebank.duckdb` が無く長さが取れない | DB 再構築後に `waveform_lengths()` で計算 |
| U-16 | 論文の v2 以降の有無 | arXiv の HTML 版 | 版一覧を確認していない | arXiv の abs ページで版を確認 |

---

## 13. 参照した一次資料

| 略称 | 内容 | URL / パス | commit / tag |
|---|---|---|---|
| `PAPER` | TF-Locoformer 論文（HTML 版テキストを `/tmp/audit-tf_locoformer/p.html.txt` に保存） | https://arxiv.org/abs/2408.03440 | v1 |
| `TFL` | merlresearch/tf-locoformer | https://github.com/merlresearch/tf-locoformer （ローカル pinned: `src/se/official/tf-locoformer`、別クローン: `/tmp/audit-tf_locoformer/tf-locoformer`） | v1.1.0 = 7a615460d347ff7334a13dbb831d16280da72cdc。v1.0.0 = 1d2be38、v1.0.1 = b76d38b |
| `ESP90` | espnet/espnet（TF-Locoformer の動作確認版） | https://github.com/espnet/espnet （別クローン: `/tmp/audit-tf_locoformer/espnet`、sparse） | 90eed8e53498e7af682bc6ff39d9067ae440d6a4 |
| `ESP0d` | espnet/espnet（audio-db のピン） | 同上（`src/se/official/espnet`、`build/clean-20261104/sources/espnet`） | 0d136c5456d403cb2c0a4c13e76e8136e32e9b7a |
| `DNS` | microsoft/DNS-Challenge `interspeech2020/master` | https://github.com/microsoft/DNS-Challenge （ローカル `data/corpora/DNS2020`、別クローン `/tmp/audit-tf_locoformer/dns-challenge`） | 70f19285c36cca4df2338f9248775ddc50980c6b |
| `rope` | lucidrains/rotary-embedding-torch | https://github.com/lucidrains/rotary-embedding-torch （`/tmp/audit-tf_locoformer/rope`） | 0.6.1、0.8.3 |
| `ADB` | audio-db 本リポジトリ | `/gs/bs/tga-shinoda/seki/exp/audio-db` | 3bee9addc0a1ff3d10d4e47a2a0eafe34f4bc110 |

### 13.1 主に参照したファイル

- `TFL`: `README.md`、`requirements.txt`、`copy_files_to_espnet.sh`、`espnet2/tasks/enh.patch`、`espnet2/enh/separator/tflocoformer_separator.py`、`egs2/dns_ins20/enh1/conf/tuning/train_enh_tflocoformer.yaml`、`egs2/dns_ins20/enh1/exp/enh_train_enh_tflocoformer_raw/{config.yaml,RESULTS.md,images/loss.png,images/si_snr_loss.png}`
- `ESP90`: `egs2/dns_ins20/enh1/{run.sh,README.md,local/data.sh,local/dns_create_mixture.sh,local/dns_data_prep.sh}`、`egs2/TEMPLATE/enh1/enh.sh`、`espnet2/tasks/abs_task.py`、`espnet2/tasks/enh.py`、`espnet2/iterators/{chunk_iter_factory,sequence_iter_factory}.py`、`espnet2/train/{trainer,reporter,preprocessor}.py`、`espnet2/main_funcs/average_nbest_models.py`、`espnet2/schedulers/warmup_reducelronplateau.py`、`espnet2/enh/espnet_model.py`、`espnet2/enh/loss/{criterions/time_domain.py,wrappers/fixed_order.py}`、`espnet2/layers/stft.py`、`espnet2/torch_utils/initialize.py`、`espnet2/bin/{enh_inference,enh_scoring}.py`
- `DNS`: `noisyspeech_synthesizer_multiprocessing.py`、`audiolib.py`、`noisyspeech_synthesizer.cfg`、`requirements.txt`、`datasets/test_set/README.md`
- `ADB`: `configs/training/tf_locoformer.yaml`、`src/training/se/{tf_locoformer,spectral,optim,initialization,config}.py`、`src/training/data/{dataset,chunk_plan,sampler}.py`、`src/training/core/{trainer,rng}.py`、`src/se/models/tf_locoformer/{model,separator,inference}.py`、`src/data/recipe.py`、`src/evaluation/audio/{render,mixing}.py`、`src/evaluation/metrics/{pesq,stoi,sisdr,compute}.py`、`tests/training/test_tf_locoformer*.py`、`docs/training/{TF_LOCOFORMER,OFFICIAL_TRAINING_AUDIT,OTHER_MODELS,VERIFICATION}.md`、`docs/se/MODELS.md`、`docs/db/RECIPE.md`、`.agents/{DECISIONS,DEVELOPMENT}.md`

### 13.2 本監査で行った読み取り専用の実測

- DNS2020 clean / noise の件数と標本の長さ・サンプリングレート（`wave` モジュール、`os.listdir`）。
- DNS 合成テスト 150 本 × 4 組（no/with reverb × clean/noisy）の長さ・サンプリングレート・ビット幅、noisy と clean の SI-SNR 平均、`clean_fileid_0.wav` の no/with_reverb 間の相互相関。
- ローカル `DNS2020` のスクリプト 6 本と upstream の `diff`。
- `dns_data_prep.sh` の uttid `sed` を代表ファイル名で試験。
- 書き込みは `/tmp/audit-tf_locoformer/` と本ファイルだけ。`src/se/official/`、`build/`、`data/` には書いていない。
