# MP-SENet 公式再現プロトコル監査

調査のみ。コード・設定・DB・既存 docs は変更していない。修正案は「提案」として書く。

## 0. 要約

### 0.1 監査結論

- **主 variant は journal 版（arXiv:2308.08926）に対応する公式 `main` HEAD（`89932cf`）の VoiceBank+DEMAND（VB-DEMAND）レシピ**。Generator 2.263M（実測）は journal 版の 2.26M と一致し、Interspeech 版の 2.05M（Conformer）とは別物である。Interspeech 版は公式 repo の旧 commit（`f37e211`、2024-04-05 以前）にだけ残る。【確認:コード】【確認:論文】
- **データプロトコル分類は「固定ペア」**（VB-DEMAND の配布済み clean/noisy を使い、学習時の混合・SNR・雑音選択・RIR は無い）。公式 Dataset が行う処理は、(1) wav 読み込み、(2) noisy の RMS による clean/noisy 共通の正規化、(3) 2 秒（32000 サンプル）のランダム crop または末尾 0 padding だけ。【確認:コード】
- **DNS 版は公式コードにも学習データ生成コードにも無い**。`best_ckpt/g_best_dns` と論文の記述（DNS 公式スクリプトで 3,000 時間、SNR −5〜15 dB、10 秒に分割）があるだけで、学習設定は UNKNOWN。【UNKNOWN】
- **公式の Generator は `batch_first` の既定値（False）を知らずに書かれた**。時間・周波数 Transformer が、時間軸・周波数軸ではなく「バッチ×周波数」「バッチ×時間」の軸に沿って動く。公式 README が自認している。出力はバッチの構成に依存することを、公式モデルと audio-db のモデルの両方で実測した（micro-batch 4 と 2 で、サンプル 0 の出力が最大 0.031 ずれる）。audio-db のモデルは同じ挙動を複製しており、docs にその記述は無い。【確認:コード】【確認:実測】
- **audio-db の `configs/training/mp_senet.yaml` は公式プロトコルではなく「研究共通条件」**（10,700 発話の学習、872 発話の話者分離 validation、seed 42）。公式は 11,572 発話すべてで学習し、validation と best 選択に test.txt（824 発話）を使う。既存 docs の「GO」の多くは、この差を無視した行ごとの判定である。

### 0.2 主な差異 TOP（詳細は §9）

| ID | 内容 | 影響度 |
|---|---|---|
| MPS-D01 | 学習セットが公式 11,572 発話ではなく 10,700 発話（p286・p287 を validation へ移した） | Critical |
| MPS-D02 | 公式は validation と best 選択に test.txt（824）。audio-db は p286/p287（872）で、test は学習・validation 禁止 | Critical |
| MPS-D03 | 学習量が論文間・論文とコードで不一致（Interspeech 100 epoch、journal v1 120 epoch、journal v2 500k step、公式既定 400 epoch）。audio-db は 400 epoch（1,070,000 step） | Major |
| MPS-D04 | 評価が公式と別手順（48 kHz 原音で評価、SE 内で往復リサンプル、CSIG/CBAK/COVL/SSNR が無い、STOI 実装が別） | Major |
| MPS-D05 | 48 kHz→16 kHz の変換器が違う（公式は事前変換済み 16 kHz、変換器は不明。audio-db は `torchaudio.functional.resample` をロード時に実行） | Unknown |
| MPS-D06 | `batch_first` 由来のバッチ依存（複製されているが未文書化、検証テストなし） | Major |
| MPS-D07 | 単 GPU 公式はエポック間で順序が固定（1 回だけ shuffle）。audio-db は毎エポック再 shuffle | Minor |
| MPS-D08 | resume の挙動が違う（公式は ExponentialLR が再開のたびに 1 回余分に減衰し、`best_pesq` が 0 に戻る。audio-db は完全復元） | Minor |

### 0.3 分類まとめ

| 項目 | 分類 | 根拠 |
|---|---|---|
| VB-DEMAND 学習 | 固定ペア（事前生成済み。配布元が作成） | `O: dataset.py:60-92` |
| DNS 学習 | 事前生成と読めるが、公式コードに無く UNKNOWN | `P2v2 §IV-A1` |
| オンライン混合 | 無し | `O: dataset.py` に混合処理が無い |

## 1. 調査メタデータ

| 項目 | 内容 |
|---|---|
| 調査日 | 2026-11-06 |
| audio-db | ブランチ `feature/reproduction`、HEAD `3bee9ad`（作業ツリーは clean） |
| 調査対象（公式） | `https://github.com/yxlu-0102/MP-SENet` `main` HEAD `89932cfe90d1dacb8e170e4a331d762462c21792`（2025-05-19）。`requirements/official-assets.toml` の pin と一致 |
| 履歴調査用 | `/tmp/audit-mp_senet/repo`（別 clone）。タグは無い |
| 論文 | arXiv:2305.13686v1（Interspeech 2023、改版なし）、arXiv:2308.08926 v1（2023-08-17）・v2（2024-04-01）。v3 は存在しない（HTML が返る）。journal 最終版（Neural Networks 189, 107562, 2025）の本文は未入手 |
| 調査方法 | 公式ソースを行単位で読み、`rg -n`・`awk` で行番号を確認。論文は PDF を `pypdf` でテキスト化して確認。audio-db は実コードを読んだ。CPU 限定の実測を 3 件行った（§5.5、§5.3、§6.1） |
| 実測（CPU、`/tmp`） | (a) 公式 `MPNet` と audio-db `build_model('mp_senet')` で、同じ重みの `dense_channel=16` モデルに batch 4・batch 2・batch 1 を与え、サンプル 0 の出力を比較。(b) PyTorch 2.6.0 の `ExponentialLR(last_epoch=k)` の再開時 lr を測定。(c) G/D のパラメータ数を数えた |
| 調査していないこと | GPU を使う学習・評価、公式学習の再実行、DNS 学習データの再生成、公式 checkpoint でのスコア再現、journal 最終版（出版版）の本文、`tests/` の内容、`VERIFICATION.md` の全文（MP-SENet に関する箇所を `rg` で確認しただけ）、`src/training/core/{checkpoint,distributed}.py`・`src/training/data/chunk_plan.py`・`src/evaluation/audio/mixing.py`・`wer.py`・`src/data/recipe.py` の詳細（MP-SENet の固定ペア経路では使わない） |

### 引用記法

- `O` = `yxlu-0102/MP-SENet @ 89932cf`（ローカル pin `src/se/official/mp-senet/` と /tmp クローンで同一）。`O1` = 同 repo `@ f37e211`（2024-04-05、Interspeech 版時代の最終 `train.py`）。
- `P1` = arXiv:2305.13686v1。`P2v1` / `P2v2` = arXiv:2308.08926 v1 / v2。
- `A` = audio-db `@ 3bee9ad`。パスは repo ルートからの相対。
- 形式は `[資料 : ファイル : 行 : 関数/キー]`。

## 2. A. 公式資料

### 2.1 資料表

| 項目 | 内容 | 確度 |
|---|---|---|
| 論文 (a) Interspeech 版 | "MP-SENet: A Speech Enhancement Model with Parallel Denoising of Magnitude and Phase Spectra"、arXiv:2305.13686v1（2023-05-23、改版なし）。VB-DEMAND のみ。PESQ 3.50、2.05M | 【確認:論文】`[P1 §3.1, Table 1]` |
| 論文 (b) journal 版 | "Explicit Estimation of Magnitude and Phase Spectra in Parallel for High-Quality Speech Enhancement"、arXiv:2308.08926。v1（2023-08-17）は VB-DEMAND、dereverb、BWE。v2（2024-04-01）で DNS Challenge 2020 を追加。Neural Networks 189 (2025) 107562 | 【確認:論文】`[P2v1 §IV, P2v2 §IV-A1]`、出版版は【UNKNOWN】 |
| 公式 repo | `https://github.com/yxlu-0102/MP-SENet`。ライセンス MIT | 【確認:コード】 |
| 調査対象 commit | `89932cf`（2025-05-19、`main` HEAD）。タグ無し | 【確認:コード】 |
| 学習コード | `train.py`、`dataset.py`、`models/{model,transformer,discriminator}.py`、`config.json`、`env.py`、`utils.py` | 【確認:コード】 |
| 評価コード | `cal_metrics/{cal_metrics_vb.py, compute_metrics.py, cal_metrics_dns.py}`、`inference.py` | 【確認:コード】 |
| データファイルリスト | `VoiceBank+DEMAND/training.txt`（11,572 行、28 話者）、`test.txt`（824 行）。wav は同梱せず、`wavs_clean` と `wavs_noisy` へ置く | 【確認:コード】 |
| 配布 checkpoint | `best_ckpt/g_best_vb`（9,142,950 B、sha256 `aedfb1aa…63f4f9`）、`best_ckpt/g_best_dns`（9,138,054 B、sha256 `97a77ba6…7206`）、`best_ckpt/config.json`（sha256 `339cb607…58f92`、ルートの `config.json` と差分なし） | 【確認:設定】`[A: requirements/official-assets.toml:70-100]`。3 ファイルとも sha256 を実測で再計算して一致 |
| 配布 16 kHz データ | README が Google Drive の 16 kHz 版を案内。変換器は不明 | `[O: README.md:16]`【UNKNOWN】 |
| DNS 学習コード | 無い。`git grep -i dns 89932cf` は README の画像だけに一致し、コードに一致しない | 【確認:コード】 |

### 2.2 公式 repo の履歴と論文の対応

| commit | 日付 | 内容 | 対応する論文 |
|---|---|---|---|
| `37fd01f`〜`31d0177` | 2023-05-18 | README のみ | — |
| `debf49a`〜`4562cde` | 2023-08-16 | 学習・推論コード、`config.json`、`g_best` を追加。`f37e211` 時点の README は "This source code is only for the MP-SENet accepted by Interspeech 2023" と書く | P1（Interspeech 版） |
| `f37e211` | 2024-04-05 | `train.py` 更新。`training_epochs` 既定 200、best 保存なし、損失に consistency 項なし、D の入力は `mag_g` | P1 |
| `a84fb8b` 他 | 2024-09-08 | README を journal 版へ更新（"PESQ 3.60 on VB-DEMAND, 3.62 on DNS"） | P2v2 |
| `17564b3`、`3045134`、`23fca59` ほか | 2024-09-08 | `models/model.py`（TS-Transformer、GRU FFN）、`dataset.py`、`train.py` を journal 版へ置換。consistency 損失、`mag_g_hat`、best checkpoint、`training_epochs` 既定 400 を追加 | P2 |
| `955d2fa` | 2024-09-08 | `best_ckpt/g_best_vb`、`g_best_dns` を追加。HEAD に旧 `g_best` は無い | P2 |
| `1563b38` | 2025-03-21 | README に `batch_first` の注意を追加し、引用に Neural Networks を記載 | — |
| `89932cf` | 2025-05-19 | README の更新（引用の書き換え） | — |

`[O: git log (/tmp clone) -- train.py dataset.py best_ckpt README.md]`【確認:コード】

### 2.3 variant 選定

| 候補 | 内容 | 判断 |
|---|---|---|
| **V1: journal 版 + VB-DEMAND レシピ（採用）** | HEAD の `train.py`、`dataset.py`、`models/model.py`。Generator 2.263M。損失に consistency 項。`g_best_vb` が対応する | 今回の目的「単一話者 SE の再現」に合う。audio-db の `src/se/models/mp_senet` と `src/training/se/mp_senet.py` が既にこの版。選定理由は、(1) 公式 HEAD と配布 checkpoint が一致、(2) audio-db の実装がこの版 |
| V2: Interspeech 版 | `f37e211` 以前。Conformer、`ExponentialLR`、200 epoch 既定、consistency 損失なし。論文は 100 epoch・30 epoch ごとに lr 半減（コードと矛盾。§11） | audio-db に実装なし。必要なら別実装が要る。研究上の判断が要る |
| V3: journal 版 + DNS レシピ | `g_best_dns`。学習コード・データ生成は非公開 | 再現不能（UNKNOWN）。評価だけは `g_best_dns` で可能 |
| V4: Long/Plus（dereverb、BWE） | journal 版の他タスク。公式コードは denoising だけ | 対象外（`dense_channel`、損失は同じ。タスク別の学習コードは無い） |

研究上の判断が要る点: V1 を主とし、V2 を「再現優先度の低い別系統」とするか。決めずに候補を残す。

## 3. B. 学習データセット

### 3.1 VB-DEMAND（V1）

| 項目 | 内容 | 根拠 |
|---|---|---|
| 名前・版 | VoiceBank+DEMAND（Valentini et al.）。Edinburgh DataShare handle 10283/1942 | `[O: README.md:16]`【確認:コード】 |
| clean の出典 | Voice Bank corpus | `[P1 §3.1]`【確認:論文】 |
| noise の出典 | DEMAND の 8 種と人工 2 種（学習）。未見 5 種（test） | `[P1 §3.1, P2v2 §IV-A1]`【確認:論文】 |
| SNR | 学習 0/5/10/15 dB、test 2.5/7.5/12.5/17.5 dB | 同上 |
| RIR | 無し | — |
| 分割（公式リスト） | training.txt 11,572 発話（28 話者）、test.txt 824 発話（2 話者） | `[O: VoiceBank+DEMAND/training.txt, test.txt]`（`wc -l`）【確認:コード】 |
| validation | **test.txt を validation として使う**（`--input_validation_file` 既定） | `[O: train.py:293]` `[O: train.py:75, 88-95]`【確認:コード】 |
| 話者の内訳 | 学習 28 話者（p226〜p287）。test は p232（393 発話）と p257（431 発話） | `[O: training.txt, test.txt]`（ID の接頭辞を集計）【確認:コード】 |
| 件数の矛盾 | P1 は test を「872 発話、2 話者」と書く。P2 と公式リストは 824 | `[P1 §3.1]` 対 `[P2v1 §IV-A1, O: test.txt]` |
| サンプリング | 原データ 48 kHz。公式は 16 kHz に変換して使う | `[O: README.md:16; config.json: sampling_rate]` |
| チャンネル・形式 | モノラル、PCM 16 bit（audio-db 手元の 48 kHz 版で確認） | `[A: data/corpora/VoiceBank-DEMAND/clean_trainset_28spk_wav/p226_001.wav]`（`wave` モジュール、読み取りのみ） |
| 対応関係 | 同名の clean と noisy。長さが違えば短い方へ揃える | `[O: dataset.py:65-66]` |
| 教師波形 | 配布された clean（時間整合済みとみなす。noisy は clean に加算混合） | 公式コードは整合を検証しない |
| 手元の実データ | `data/corpora/VoiceBank-DEMAND/{clean,noisy}_trainset_28spk_wav`（11,572 件）、`{clean,noisy}_testset_wav`（824 件）、48 kHz | `ls`、`wave`【確認:実測】 |

### 3.2 DNS Challenge 2020（V3）

| 項目 | 内容 | 確度 |
|---|---|---|
| 論文の記述 | 500 時間の clean（2150 話者）、180 時間超の noise。clean を 10 秒に分割。「公式スクリプトに従い」SNR −5〜15 dB で 3,000 時間の noisy を生成。test は reverb なしの non-blind 4,270 クリップ（SNR 0〜25 dB） | 【確認:論文】`[P2v2 §IV-A1]` |
| 使った生成スクリプト・設定 | 不明。`microsoft/DNS-Challenge` の `interspeech2020/master`（2020-07-23）と `master`（2024-03-29）で既定値が違う（下表）。論文の値（−5〜15 dB、3,000 h、10 秒）はどちらの既定にも一致しない | 【UNKNOWN】 |
| RIR | 論文は reverb なしと書く。スクリプトに RIR 分岐がある版もあるが、使用有無は不明 | 【UNKNOWN】 |
| 件数・分割 | 3,000 時間のうち validation をどう取ったか不明 | 【UNKNOWN】 |
| 手元 | `data/corpora/DNS2020/` に公式 repo の synthesizer スクリプトとデータがある。スクリプトは `interspeech2020/master` と cfg・単一プロセス版が一致 | 【確認:実測】`diff` |
| test 件数 | 論文の 4,270 が DNS 2020 non-blind 配布物の件数と合うか未確認 | 【UNKNOWN】 |

DNS 公式スクリプトの既定値（参考。MP-SENet が使ったとは限らない）:

| 設定 | `interspeech2020/master` (`70f19285`) | `master` (`591184a9`) | 論文（P2v2） |
|---|---|---|---|
| `audio_length` | 30 s | 30 s | clean を 10 秒に分割 |
| `total_hours` | 100 | 500 | 3,000 |
| `snr_lower` / `snr_upper` | 0 / 40 | −5 / 20 | −5 / 15 |
| SNR の選び方 | `np.random.randint(snr_lower, snr_upper)`（整数 dB、上限は含まない） | 同じ | 不明 |
| ミキサー | `snr_mixer`（全体 RMS） | `segmental_snr_mixer` | 不明 |
| `target_level` | −35〜−15 dBFS（`randint`） | 同じ | 不明 |

`[microsoft/DNS-Challenge @ 70f19285, 591184a9 : noisyspeech_synthesizer.cfg; noisyspeech_synthesizer_singleprocess.py : main_gen; audiolib.py:111-152 : snr_mixer]`【確認:コード（DNS-Challenge）】。取得物は `/tmp/audit-mp_senet/dns/`。

## 4. C. データ生成・前処理

### 4.1 VB-DEMAND（公式学習）の処理順序

学習時（`split=True`）:

1. ファイルリスト `training.txt` の各行を `|` で分割し、先頭（発話 ID）だけを使う。`|` 以降のパスは使わない。`[O: dataset.py:32-39 : get_dataset_filelist]`【確認:コード】
2. `Dataset.__init__` が `random.seed(1234)` を呼び、`shuffle=True`（単 GPU のとき）なら ID リストを 1 回だけ `random.shuffle` する。以後、リストは固定。`[O: dataset.py:46-48; train.py:78]`【確認:コード】
3. `__getitem__` が `librosa.load(<wavs_clean>/<id>.wav, sr=16000)` と `librosa.load(<wavs_noisy>/<id>.wav, sr=16000)` を読む。ファイルが 16 kHz なら変換は起きない。librosa 0.7.2（`requirements.txt`）の既定は float32、モノラル。`[O: dataset.py:63-64; requirements.txt]`【確認:コード】
4. 長さを `min(len(clean), len(noisy))` に揃える。`[O: dataset.py:65-66]`
5. **noisy の RMS で正規化**する。`norm_factor = sqrt(len(noisy) / sum(noisy²))`。clean と noisy の両方へ同じ係数を掛ける。crop より前、全発話長に対して行う。`[O: dataset.py:75-78]`【確認:コード】
6. 長さが `segment_size`（32000）以上なら、`random.randint(0, len - 32000)`（両端を含む）で開始位置を決めて切り出す。短ければ末尾に 0 を足して 32000 にする。clean と noisy は同じ位置。`[O: dataset.py:82-90]`【確認:コード】
7. DataLoader が `batch_size=4`、`drop_last=True` でバッチにし、`train.py` が `mag_pha_stft`（Hann 窓 400、hop 100、n_fft 400、reflect、圧縮 0.3）で clean と noisy の STFT を計算する。`[O: train.py:82-86, 121-122; dataset.py:7-19]`
8. Generator は noisy の振幅・位相から振幅マスク、位相、複素スペクトルを返す。`mag_pha_istft` で波形へ戻し、再び STFT して `mag_g_hat` を得る。`[O: train.py:124-127]`

validation と test（`split=False`）では 6 を行わず、全発話長のまま batch 1 で処理する。`[O: train.py:88-95]`

### 4.2 処理ごとの実効動作

| 処理 | 公式の実効動作 | 毎回再サンプルされるもの | 根拠 |
|---|---|---|---|
| 混合 | 無し（固定ペア） | — | `O: dataset.py` |
| 雑音選択・SNR・ゲイン・RIR | 無し | — | 同上 |
| リサンプリング | 公式コードでは `librosa.load(sr=16000)`。入力が 16 kHz なら no-op。48 kHz→16 kHz は README が事前変換を指示（変換器は不明） | — | `O: dataset.py:63-64; README.md:16` |
| モノラル化 | `librosa.load` の既定（mono=True） | — | `O: dataset.py:63-64` |
| 正規化 | noisy の RMS を 1 に。clean へも同係数。ピーク正規化・クリッピング処理は無い | 毎ロード（決定的） | `O: dataset.py:75-78` |
| crop | 一様乱数の開始位置。PyTorch DataLoader が worker ごとに `random.seed(base_seed + worker_id)` を設定する（`base_seed` は `iter(train_loader)` ごとに torch の既定 RNG から得る）。`torch.manual_seed(1234)` を `main` が呼ぶので、worker 数と PyTorch 版が同じなら再現可能 | 毎サンプル・毎エポック | `O: dataset.py:85-86; train.py:312`、`torch/utils/data/_utils/worker.py:258-260`（PyTorch 2.6.0） |
| padding | 末尾 0 埋め（constant）。mask は無く、padding 部分も損失に入る | 決定的 | `O: dataset.py:88-91` |
| 長さ制限 | 学習は 32000 サンプルに固定。validation・test は無制限 | — | `O: train.py:88-95` |
| cache | `n_cache_reuse=0`。毎回ディスクから読む | — | `O: train.py:77-78` |

【推測】worker の乱数: PyTorch 2.6.0 のソースで確認した。公式の想定環境（PyTorch 1.8.1）で同じ式かは未確認。

### 4.3 DNS 版

処理順序は UNKNOWN。コードに無い。論文は「10 秒に分割 → 公式スクリプトで混合 → 2 秒に切り出して学習（P2v2 §IV-B）」としか書かない。2 秒 crop は VB-DEMAND と同じ `segment_size=32000` で、`config.json` は DNS 版でも同じ（`best_ckpt/config.json` を `diff` して差なし）。

## 5. D. Dataset / DataLoader / Sampler

### 5.1 公式の構成

| 項目 | 実効動作 | 根拠 |
|---|---|---|
| Dataset | `Dataset(training_indexes, clean_dir, noisy_dir, 32000, 16000, split=True, shuffle=…, n_cache_reuse=0)` | `O: train.py:77-78` |
| shuffle（単 GPU） | `Dataset` が初期化時に 1 回だけ shuffle（seed 1234）。`DataLoader(shuffle=False, sampler=None)`。**エポック間で順序は同じ** | `O: train.py:78, 82-83; dataset.py:46-48` |
| shuffle（複数 GPU） | `Dataset` は shuffle しない。`DistributedSampler(trainset)`（seed 0、`shuffle=True` 既定）、毎エポック `set_epoch(epoch)` | `O: train.py:78, 80, 109-110` |
| DDP の分割 | `batch_size = int(batch_size / num_gpus)`（切り捨て）。2 GPU なら rank あたり 2 | `O: train.py:316` |
| 初期化 | `mp.spawn` の子プロセスは `torch.manual_seed` を再設定しない。モデルの初期重みは再現性がない | `O: train.py:312-322, 29`【確認:コード】、子プロセスの乱数状態は【推測】 |
| drop_last | `True`。11,572 発話・batch 4 で余りなし | `O: train.py:86` |
| workers | `num_workers=4`、`pin_memory=True`。validation は `num_workers=1`、batch 1 | `O: config.json; train.py:82-95` |
| 可変長 | 学習は crop/pad で固定長。validation は batch 1 で可変長のまま | `O: dataset.py:82-90` |
| 重複サンプリング・サンプル数制限 | 無し | — |
| chunk iterator | 無し | — |
| 勾配蓄積 | 無し。1 バッチ = 1 更新（G と D 各 1 回） | `O: train.py:133-184` |

### 5.2 epoch 定義表

| 単位 | 公式（VB、batch 4） | audio-db 現行 yaml |
|---|---|---|
| 元発話数 / エポック | 11,572（`training.txt` 全行） | 10,700（`voicebank_28_train`） |
| 切り出しセグメント数 / エポック | 11,572（1 発話 1 crop） | 10,700 |
| バッチ数 / エポック | 2,893 | 2,675 |
| optimizer 更新 / エポック（G と D それぞれ） | 2,893 | 2,675 |
| scheduler step | エポック終了時に 1 回（G と D） | エポック終了時に 1 回 |
| 最大エポック | 400（CLI 既定） | 400（`max_steps` 1,070,000 = 400×2,675） |
| 総 step | 1,157,200 | 1,070,000 |
| validation 間隔 | `steps % 5000 == 0`（約 1.73 エポックごと） | `validation_interval` 2,675（毎エポック末） |
| checkpoint 間隔 | 5,000 step | 2,675 step |
| lr（エポック 400 の開始時） | 5e-4 × 0.99⁴⁰⁰ ≈ 8.98e-6 | 同じ |

論文の学習量:

| 資料 | 学習量 | 換算 |
|---|---|---|
| P1（Interspeech） | 100 epoch、lr は 30 epoch ごとに半減 | 289,300 step |
| P2v1（journal v1） | 単一話者実験は 120 epoch、lr は 0.99/epoch | 347,160 step |
| P2v2（journal v2） | 500k step、lr は 0.99/epoch | 約 172.8 epoch |
| 公式コード HEAD | `--training_epochs` 既定 400 | 1,157,200 step |
| 公式コード `f37e211` | 既定 200 | 578,600 step |

`[P1 §3.1; P2v1 §IV-B; P2v2 §IV-B; O: train.py:296; O1: train.py:278]`。どれが配布 checkpoint の学習量かは【UNKNOWN】。

### 5.3 公式 resume の挙動（`O: train.py:45-73, 104`）

- `scan_checkpoint` が `g_????????`・`do_????????`（8 桁の step 番号）に一致するファイルの辞書順で最後を選ぶ。`g_best` は一致しない。`steps = do['steps'] + 1`、`last_epoch = do['epoch']`。`[O: utils.py:86-91]`【確認:コード】
- `for epoch in range(max(0, last_epoch), …)` なので、保存時のエポックを **最初からやり直す**（保存時点のバッチ位置は復元しない）。
- `ExponentialLR(last_epoch=last_epoch)` を生成すると、コンストラクタ内で `step()` が 1 回走り、**lr がさらに ×0.99 される**。PyTorch 2.6.0 の CPU 実測で比 0.99 を確認。PyTorch 1.8.1 でも同じ式のはずだが未確認。【確認:実測 (2.6.0)】【推測 (1.8.1)】
- `best_pesq = 0` に戻る（`O: train.py:102`）。再開後の最初の validation が、悪くても `g_best` を上書きする。
- scheduler の状態、RNG、`DistributedSampler` の位置は checkpoint に無い。

### 5.4 audio-db の構成

- `RecipeDataset`（`src/training/data/dataset.py`）が `render_recipe` で wav を読み、`_match_rate`（`torchaudio.functional.resample`）→ `_normalize`（noisy の RMS）→ `_segment`（crop/pad）の順で処理する。`[A: src/training/data/dataset.py:181-222, 331-380]`【確認:コード】
- 順序は `epoch_indices(size, seed, epoch, world_size, rank)` が `torch.randperm`（`mix_seed("order", seed, epoch)`）で毎エポック作る。crop 位置は `mix_seed("crop", seed, epoch, case_id) % (length - segment + 1)`。`[A: src/training/data/sampler.py:16-46]`【確認:コード】
- バッチは `micro_batch_size_per_gpu × accumulation × world_size = global_batch_size`。端数は捨てる。`[A: src/training/core/trainer.py:244-343]`
- 公式との一致: crop の分布（`[0, T−32000]` 一様、両端含む）、`>=` で crop、短い発話は末尾 0 padding、noisy の RMS で正規化してから crop は一致する。順序（単 GPU の固定順 vs 毎エポック再 shuffle）と乱数列は一致しない。

### 5.5 `batch_first` の挙動（公式のバグ。実測）

- `TransformerBlock` は `MultiheadAttention(d_model, n_heads, dropout)` と `GRU(d_model, 2*d_model, 1, bidirectional)` を `batch_first` 指定なしで作る（既定 False）。`[O: models/transformer.py:9, 31]`【確認:コード】
- `TSTransformerBlock.forward` は `(b*f, t, c)` を渡す。そのため「time_transformer」は系列軸 `b*f`、バッチ軸 `t` で動き、「freq_transformer」は `(b*t, f, c)` を系列軸 `b*t` で動く。**注意と GRU が他のバッチ要素と周波数（または時間）を混ぜる**。`[O: models/model.py:124-131]`【確認:コード】
- 公式 README は小さなバグを認め、再学習するなら `batch_first=True` を強く推奨する（"does not affect the overall performance"）。`[O: README.md:8-10]`【確認:コード】
- 実測（CPU、`dense_channel=16`、同じ重み）: batch 4 と batch 2 でサンプル 0 の最大差 0.0306、batch 4 と batch 1 で 0.0537、他の 3 要素を並べ替えるだけで 0.0234。公式モデルと audio-db の `build_model('mp_senet')` で同じ値。【確認:実測】
- 帰結: (1) 学習の結果はバッチの構成（同じバッチに誰が入るか、バッチサイズ、DDP の rank あたりサイズ）に依存する。(2) 推論は batch 1 なので、学習時と文脈が異なる。【確認:コード】(1) の定性的な部分は実測。(2) は読解。
- audio-db は `batch_first` を指定せず、挙動を複製している（`src/se/models/mp_senet/transformer.py:9, 31`）。`batch_first` は docs のどこにも出てこない（`rg` で確認）。
- 再現上の判断: V1 で公式挙動を保つなら現状の複製でよい。README の推奨に従う場合は重みの互換が無くなり、配布 checkpoint を読めない。決めずに両方を候補とする。

## 6. E. モデル・学習条件

### 6.1 設定値と実効値

| 項目 | 設定値 | 実効値 | 根拠 |
|---|---|---|---|
| Generator | `dense_channel=64`、TS-Transformer 4 ブロック | **4 ブロック固定**。`config.json` の `num_tsconformers` は `MPNet(h)` で使われず、`num_tsblocks=4` の既定値になる | `O: config.json; models/model.py:135-143; train.py:32` |
| パラメータ数 | — | G 2.263M、D 0.182M（実測） | 実測 |
| STFT | n_fft 400、win 400、hop 100、圧縮 0.3、`beta=2.0` | Hann（`torch.hann_window`）、reflect、`center=True`、`sqrt(power+1e-9)`、`atan2(imag+1e-10, real+1e-5)` | `O: dataset.py:7-19` |
| segment | 32000 | 学習のみ。validation は全長 | `O: config.json; dataset.py:82-90` |
| 損失 | mag 0.9、phase 0.3、complex 0.1（×2）、consistency 0.1（×2）、metric 0.05、time 0.2 | `loss_mag*0.9 + loss_pha*0.3 + loss_com*0.1 + loss_stft*0.1 + loss_metric*0.05 + loss_time*0.2`、`loss_com` と `loss_stft` は `mse*2` | `O: train.py:166-181` |
| 位相損失 | IP + GD + IAF（anti-wrapping） | 3 項の単純和 | `O: models/model.py:164-174` |
| D の更新順 | D → G | D は `(clean_mag, clean_mag)`→1、`(clean_mag, mag_g_hat.detach())`→(PESQ−1)/3.5。G は `(clean_mag, mag_g_hat)`→1。D のパラメータは G の backward 中も凍結しない | `O: train.py:132-146, 178-179` |
| PESQ 目標 | wb、16 kHz | `pesq(sr, clean, noisy, 'wb')`。失敗は −1。1 件でも −1 があればバッチ全体が `None` で D の fake 項は `0`（int） | `O: models/discriminator.py:13-28; train.py:138-144` |
| Optimizer | AdamW、lr 5e-4、betas (0.8, 0.99) | `weight_decay` は引数を省略したので PyTorch 既定の **0.01**、eps 1e-8 | `O: train.py:65-66` |
| Scheduler | `lr_decay=0.99` | `ExponentialLR(gamma=0.99)`、エポック終了時に G と D を 1 回ずつ step。warmup 無し | `O: train.py:72-73, 277-278` |
| 勾配クリップ | — | 無し | `O: train.py`（`clip` が無いことを `rg` で確認） |
| 初期化 | — | PyTorch 各層の既定。明示初期化なし。単 GPU のみ `torch.manual_seed(1234)` で再現可能 | `O: train.py:312-314` |
| mixed precision | — | 無し（FP32） | `O: train.py` |
| cudnn | — | `benchmark=True` | `O: train.py:22` |
| 最大 epoch | `--training_epochs` 既定 400 | 400（`f37e211` では 200） | `O: train.py:296; O1: train.py:278` |
| batch / GPU | `batch_size: 4`、`num_gpus: 0` | `torch.cuda.device_count()` で上書き。`batch_size = int(4 / num_gpus)` | `O: config.json; train.py:315-316` |
| 検証間隔 | 5,000 step | 同じ。rank 0 だけが実行 | `O: train.py:300, 222-273` |
| checkpoint 間隔 | 5,000 step | `g_{steps:08d}`（G のみ）と `do_{steps:08d}`（D、optimizer、steps、epoch）。scheduler の状態は無い | `O: train.py:200-209` |
| best | `best_checkpoint_start_epoch=40` | `epoch >= 40`（0 始まり）かつ validation PESQ 平均が `best_pesq` を上回るとき、`g_best` に G のみ保存。初期値 `best_pesq=0` | `O: train.py:266-271` |
| early stopping | — | 無し | — |

### 6.2 audio-db の設定

`configs/training/mp_senet.yaml`（HEAD、直近 commit `b95bbd6`）:

```6:16:configs/training/mp_senet.yaml
seed: 42
deterministic: false
global_batch_size: 4
micro_batch_size_per_gpu: 4
max_steps: 1070000
segment_size: 32000
loss: official
checkpoint_dir: ./data/results/training/voicebank-seed42/mp_senet
checkpoint_interval: 2675
validation_interval: 2675
best_checkpoint_start_epoch: 40
```

- optimizer、scheduler、モデルの値は公式と同じ（AdamW 5e-4、betas 0.8/0.99、weight decay 0.01 を明示、ExponentialLR 0.99、`dense_channel` 64）。`[A: configs/training/mp_senet.yaml:18-37; src/training/se/optim.py:25-29, 156]`
- `data.database: data/db/voicebank.duckdb`、`train_set: voicebank_28_train`、`validation_set: voicebank_28_valid`（L39-44）。**`data/db/` には `.gitkeep` しか無く、`voicebank.duckdb` は作業ツリーに存在しない**（`ls`）。この yaml は現状では実行できない。
- `deterministic: false` のため `cudnn.benchmark=False`（公式 True）、`use_deterministic_algorithms(False)`。`[A: src/training/core/rng.py:29-48]`
- 勾配クリップ: `grad_clip` は config のキーとして存在するが、`clip_gradients` を呼ぶのは `tf_gridnet.py`、`mossformergan.py`、`tf_locoformer.py`、`semambapp.py` で、`mp_senet.py` は呼ばない（`rg` で確認）。公式（クリップなし）と一致。

### 6.3 audio-db の損失と更新順（コードで確認）

- D の fake 目標は `metric_targets`（`pesq(..., 'wb')`、失敗は −1、1 件でも負なら `None`、`(score−1)/3.5`）。`None` のとき fake 項を足さない。公式と一致。`[A: src/training/se/losses.py:76-125]`
- G の損失の重みと式は公式と一致（`generator_losses`）。1 つの Generator グラフを micro-batch ごとに作り、D を更新してから G を backward する。D のパラメータは G の backward 中だけ凍結する（公式は凍結しない）。D の勾配は次の `zero_grad` で消えるので、更新結果は変わらないはず。CPU 1 更新の比較は `docs/training/MP_SENET.md` が「差なし」と記録している（本監査では再検証していない）。【確認:コード】【確認:docs】
- 蓄積すると、バッチの構成が変わり（§5.5）、公式の batch 4 とは別の計算になる。

## 7. F. Validation / Test / 評価

### 7.1 公式の validation

| 項目 | 実効動作 | 根拠 |
|---|---|---|
| データ | `test.txt`（824 発話、**テストセットそのもの**）。`split=False`、batch 1、全長 | `O: train.py:88-95, 293` |
| 前処理 | 学習と同じ RMS 正規化（noisy 基準）。crop・padding なし | `O: dataset.py:75-80` |
| 指標 | PESQ（`pesq(sr, clean, esti)`、既定 wb）の平均。失敗は −1 のまま平均に入る。他に mag/phase/complex/consistency の MSE（TensorBoard のみ） | `O: models/model.py:177-195; train.py:247-264` |
| 長さ | clean は全長。生成波形は `istft` の既定で hop の倍数に切れる（最大 99 サンプル短い）。**長さの違う 2 本を PESQ に渡す**。損失はスペクトル領域なので影響しない | `O: dataset.py:27; train.py:242-245` |
| 並列 | `joblib.Parallel(n_jobs=30)` | `O: models/model.py:179` |
| 実行 | rank 0 のみ | `O: train.py:222-273` |
| best 選択 | `epoch >= 40` で PESQ 最大の G を `g_best` へ。**選択に使う集合がテスト集合**であり、報告値も同じ集合のため、選択バイアスを含む可能性がある | `O: train.py:266-271` |

【確認:コード】。報告値が `g_best` で出されたかは論文に書かれておらず【UNKNOWN】。

### 7.2 論文・公式スクリプト・audio-db の評価比較

| 項目 | 論文 | 公式スクリプト | `src/evaluation` |
|---|---|---|---|
| データ | VB-DEMAND test（P1 は 872、P2 は 824 と記載）。DNS 2020 reverb なし | `inference.py` は `--input_noisy_wavs_dir` を `os.listdir` で全件。既定 `VoiceBank+DEMAND/testset_noisy`（README の `wavs_noisy` と名前が違う） | `voicebank_demand_test`（`configs/db/lists/voicebank_demand_test.txt`、824）。48 kHz の原音のまま |
| 入力 SR | 16 kHz に変換 | `librosa.load(sr=16000)` | 評価は原音の 48 kHz。SE は内部で 16 kHz に落とし、48 kHz へ戻す |
| 推論 | — | RMS 正規化 → STFT → モデル → iSTFT → `/ norm_factor` → `sf.write(..., 'PCM_16')` | `_MPEnhancer`: `AF.resample` → RMS 正規化 → STFT → モデル → `istft(length=...)` → `/ norm` → `AF.resample` で戻す → 長さ補正。float のまま |
| 出力長 | — | hop の倍数に切れる（入力より最大 99 サンプル短い） | 入力長に 0 で補う |
| 量子化 | — | PCM 16 bit | なし |
| PESQ | WB-PESQ（P2）。P1 は "PESQ" | `compute_metrics.py:64` `pesq(sr, ref, deg, 'wb')`（16 kHz）。短い方に切り、長さが違うときだけ `+ np.spacing(1)` | `score_pesq`: 16 kHz のとき `wb`、それ以外は 16 kHz へ再サンプルして `wb`。最短長に切る。境界長の 2 点評価で差が 0.1 超なら `na`（`pesq_utterance_split_discontinuity`） |
| STOI | STOI（0〜1） | `compute_metrics.py:321-374` の自前実装（CMGAN 由来）。DNS 用は `pysepm.stoi`（`cal_metrics_dns.py:48`） | `pystoi.stoi`（`extended=False`） |
| CSIG / CBAK / COVL | 報告 | `compute_metrics.py:67-75`（PESQ・LLR・WSS・segSNR から、[1,5] に clip） | **実装なし** |
| SSNR | P1 のみ報告 | `compute_metrics.py:59-61`（`segSNR`）。`cal_metrics_vb.py` が 5 番目に出力 | **実装なし** |
| SI-SDR | DNS のみ報告（dB） | `cal_metrics_dns.py:10-28`（DC 除去あり、`20 log10(‖t‖/‖n‖)`）。`apd` は未使用 | `score_si_sdr`: 平均除去、float64、`10 log10(Ps/Pn)`。定義は同じ |
| NB-PESQ | DNS のみ報告 | `cal_metrics_dns.py:47` | 16 kHz では `wb` のみ。`nb` は 8 kHz のときだけ |
| ViSQOL・PD | P2 Table III の ablation | スクリプト無し | 無し |
| 集約 | テスト全体の平均 | 単純平均。`cal_metrics_vb.py` は `indexes = sorted(os.listdir(clean_wav_dir))` を回し、各指標の合計を `num` で割る | `na` / `unavailable` / `error` を除いた平均（count を記録） |
| 失敗の扱い | — | PESQ 例外は学習時 −1（検証の平均に入る）。評価スクリプトは例外を捕まえない | 0 として扱わない |
| 評価 checkpoint | — | README は `g_best_vb` | — |

`[O: cal_metrics/cal_metrics_vb.py:10-25; compute_metrics.py:28-79, 321-345; cal_metrics_dns.py:10-65; inference.py:40-59, 66; A: src/se/api.py:59-88; src/evaluation/metrics/{pesq,stoi,sisdr}.py; src/evaluation/pipeline.py:326-371; src/data/corpora/mpsenet_filelist.py]`【確認:コード】

### 7.3 報告値

| 資料 | データ | PESQ | CSIG | CBAK | COVL | SSNR | STOI | 他 | パラメータ |
|---|---|---|---|---|---|---|---|---|---|
| P1 Table 1 | VB-DEMAND | 3.50 | 4.73 | 3.95 | 4.22 | 10.64 | 0.96 | — | 2.05M |
| P2v1・v2 Table I | VB-DEMAND | 3.60（WB） | 4.81 | 3.99 | 4.34 | — | 0.96 | — | 2.26M |
| P2v2 Table II | DNS 2020 reverb なし | 3.62（WB）、NB 3.92 | — | — | — | — | 98.16 % | SI-SDR 21.03 dB | 2.26M |
| P2 Table III（ablation） | VB-DEMAND | 3.60 | — | — | — | — | — | PD 7.28°、SI-SDR 19.47 dB、ViSQOL 4.34 | 2.26M |
| Noisy | VB-DEMAND | 1.97 | 3.35 | 2.44 | 2.63 | 1.68 | 0.91 | — | — |

【確認:論文】`[P1 Table 1; P2v1 Table I; P2v2 Table I, II, III]`。どの checkpoint・どの seed の値かは論文に無く【UNKNOWN】。audio-db での数値再現は本監査の範囲外であり、していない。

## 8. 現行 audio-db との項目別比較表

判定: MATCH / PARTIAL / DIFF / UNKNOWN / N/A。audio-db 側の主張はコードを読んで確認した。

| 項目 | 公式設定 | 公式実効動作 | audio-db現状 | 判定 | 根拠 |
|---|---|---|---|---|---|
| Dataset | VB-DEMAND、`training.txt` 11,572 | 全件で学習、`test.txt` 824 を validation と test に使う | `voicebank_28_train` 10,700（p286・p287 を除く）、`voicebank_28_valid` 872。DB ファイル未作成 | DIFF | `O: train.py:75, 293`; `A: src/data/voicebank_sets.py:1-8, 44-50` |
| 混合方式 | 固定ペア | 固定ペア | 固定ペア（`operations: []`、`reference: dry_clean`） | MATCH | `O: dataset.py:60-92`; `A: voicebank_sets.py` |
| 雑音選択 | N/A | N/A | N/A | N/A | — |
| SNR | N/A（配布値 0/5/10/15 dB） | N/A | N/A | N/A | `P1 §3.1` |
| ゲイン | N/A | N/A（noisy RMS 正規化のみ） | N/A | N/A | — |
| 残響/RIR | N/A | N/A | N/A | N/A | — |
| リサンプリング | 16 kHz の配布物を使用（README） | `librosa.load(sr=16000)`。16 kHz 入力なら no-op | 48 kHz 原音をロードごとに `torchaudio.functional.resample`（既定の sinc Hann）で 16 kHz へ。長さ検査あり | UNKNOWN（公式の変換器が不明。差は Minor〜Major の可能性） | `O: README.md:16; dataset.py:63-64`; `A: dataset.py:331-347` |
| 正規化 | noisy の RMS（1.0）。clean へ同係数 | crop の前、全長 | 同じ。resample 後に計算（float32） | MATCH | `O: dataset.py:75-78`; `A: dataset.py:215-216, 359-364` |
| Crop | `segment_size` 32000、一様乱数、`>=` | `random.randint(0, T−32000)`、worker ごとの乱数 | 一様、`SHA-256(seed, epoch, case_id) % (T−32000+1)`、`>=` | PARTIAL（分布は同じ、乱数列は別） | `O: dataset.py:82-87`; `A: sampler.py:34-46` |
| Padding | 末尾 0 | 末尾 0（mask なし） | 末尾 0 | MATCH | `O: dataset.py:88-91`; `A: dataset.py:379-380` |
| 音声長制限 | 学習 32000、validation 全長 | 同じ | 同じ（validation は hop 倍数へ切る） | PARTIAL | `O: train.py:88-95`; `A: mp_senet.py:147-166` |
| Shuffle/Sampler | 単 GPU は初期化時 1 回 shuffle（seed 1234）で固定順、複数 GPU は `DistributedSampler` | 単 GPU では毎エポック同じ順 | 毎エポック `randperm(seed, epoch)`、端数は捨てる | DIFF（Minor） | `O: train.py:78-83, 109-110; dataset.py:46-48`; `A: sampler.py:16-31` |
| Epoch定義 | データ 1 周（2,893 step） | 同じ | データ 1 周（2,675 step）。母数が違う | PARTIAL | §5.2 |
| Batch | `batch_size=4`、GPU 数で切り捨て除算 | `drop_last=True` | `global_batch_size=4`、micro 4、厳密に保つ | MATCH（単 GPU）/ PARTIAL（複数 GPU の割り方） | `O: train.py:316`; `A: config 8-9; trainer.py:254-257` |
| DDP/accumulation | DDP、蓄積なし | rank ごとに batch 2（2 GPU）。`batch_first` により計算がバッチ構成に依存 | DDP と蓄積を実装。蓄積は公式にない。`batch_first` 依存も複製 | PARTIAL | §5.5 の実測 |
| 損失 | §6.1 | 同左 | `generator_losses` が同じ式と係数 | MATCH（コード確認） | `O: train.py:166-181`; `A: src/training/se/losses.py` |
| D 目標・更新順 | D→G、PESQ 目標、失敗時 fake 項 0 | 同左 | 同じ。G の backward 中は D を凍結（公式は凍結しない） | MATCH（更新結果は同等のはず。未検証） | `O: train.py:132-184`; `A: mp_senet.py:234-315` |
| Optimizer | AdamW 5e-4、betas (0.8, 0.99) | weight decay 0.01、eps 1e-8 | 同じ（weight decay を明示） | MATCH | `O: train.py:65-66`; `A: optim.py:25-29` |
| Scheduler | `ExponentialLR` 0.99 | エポック終了時 | 同じ。蓄積では進まない | MATCH | `O: train.py:72, 277-278`; `A: optim.py:156; trainer.py:311-312` |
| 勾配クリップ | 無し | 無し | 無し | MATCH | `O: train.py`; `A: mp_senet.py` |
| 初期化 | PyTorch 既定。seed 1234（単 GPU のみ有効） | 同左 | PyTorch 既定（`seed_model_init(seed)`）。seed 42 | PARTIAL | `O: train.py:312-314`; `A: rng.py; yaml:6` |
| 最大 epoch/step | 400 epoch（既定）。論文は 100/120 epoch または 500k step | 1,157,200 step | 1,070,000 step（400 epoch） | PARTIAL（コード既定に従う。論文と矛盾） | §5.2 |
| Validation データ | `test.txt` 824 | テスト集合 | `voicebank_28_valid` 872（p286・p287）。`voicebank_demand_test` は学習・validation に使えない | DIFF | `O: train.py:88-95, 293`; `A: voicebank_sets.py:3-4`、`docs/training/MP_SENET.md` |
| Validation 前処理 | RMS 正規化、全長 | 同左 | 同じ。ただし clean と noisy を hop 倍数へ切る | PARTIAL | `A: mp_senet.py:159-166` |
| Validation バッチ | batch 1 | 同左 | 1 発話ずつ（`collate([index])`） | MATCH | `O: train.py:91-95`; `A: trainer.py:631-633` |
| Validation 間隔 | 5,000 step | step 倍数 | 2,675 step（毎エポック） | DIFF（Minor） | `O: train.py:223`; `A: yaml:15` |
| Validation PESQ 集約 | `pesq` 平均。失敗は −1 のまま平均に入る | 長さ不一致のまま | `na`・`error` を除いた平均（count 付き）。長さ境界ガードで `na` になり得る。clean と estimate を最短長へ切る | DIFF（Minor） | `O: models/model.py:177-195`; `A: trainer.py:880-888; pesq.py` |
| Best 選択 | `epoch >= 40`、PESQ 最大、`g_best` | `best_pesq=0` から | `epoch >= 40`（0 始まり）、PESQ 最大、`best_generator.pt`。`na` なら更新しない | MATCH（規則）。集合が違う | `O: train.py:266-271`; `A: trainer.py:737-786` |
| Early stopping | 無し | 無し | 無し | MATCH | — |
| Resume | `g_*`・`do_*` から。エポックを頭から。lr が再開ごとに ×0.99、`best_pesq` が 0 に戻る | RNG なし | model・optimizer・scheduler・step・消費済み micro-batch・RNG を復元 | DIFF（audio-db が厳密） | `O: train.py:45-73, 102, 104`; `A: trainer.py:168-195` |
| Generator の `batch_first` | 既定 False のまま | バッチ依存 | 複製。docs に記述なし | MATCH（挙動）/ 未文書化 | §5.5 |
| `num_tsconformers` | 4 | 使われない（既定 4） | `build_model` が `num_tsblocks` へ渡す。値は 4 | MATCH（値）、意味は PARTIAL | `O: models/model.py:135; train.py:32`; `A: src/se/api.py:13, 35` |
| 評価 PESQ | wb、16 kHz、全長 | `compute_metrics.py:64` | wb、評価は 48 kHz 原音を 16 kHz へ再サンプル。SE は内部で往復変換 | PARTIAL | `A: pipeline.py:326-371; api.py:59-88` |
| 評価 STOI | 自前実装（CMGAN 由来） | `compute_metrics.py:321-374` | `pystoi`（`extended=False`） | UNKNOWN（数値の同等性は未確認） | `A: stoi.py:38-52` |
| 評価 SI-SDR | DNS のみ。DC 除去あり | `cal_metrics_dns.py:10-28` | 同じ定義 | MATCH（定義）。VB では報告されない | `A: sisdr.py:23-27` |
| 評価 CSIG/CBAK/COVL/SSNR | 報告 | `compute_metrics.py` | 未実装 | DIFF | `A: src/evaluation/metrics/`（実装なし） |
| 出力の量子化・長さ | PCM 16 bit、hop 倍数に短縮 | 同左 | float、入力長へ 0 補完 | DIFF（Minor） | `O: inference.py:54-59`; `A: api.py:86-88` |
| Test | `test.txt` 824。公式スクリプトは出力ディレクトリ全件 | 同左 | `voicebank_demand_test`（824、リストは MIT） | MATCH（件数・リスト） | `A: configs/db/init.yaml:44-53` |
| 乱数 seed | 1234（`config.json`）。`random.seed(1234)` を Dataset 初期化で | 同左 | 42、`deterministic: false` | DIFF（Minor） | `O: config.json; dataset.py:46`; `A: yaml:6-7` |
| 学習データ生成の DB 化 | N/A | — | レシピ JSON + `eval.cases` の名前付きセット | N/A | `docs/db/RECIPE.md` |

## 9. 差異一覧

影響度: Critical = プロトコル・学習結果に直接影響、Major = 性能・比較条件に影響し得る、Minor = 影響限定的、Unknown。実装状態: 未実装 / 実装済み（公式一致は未検証）/ 実装済み・検証済み（範囲明記）。優先度: P0 = 公式プロトコル再現に不可欠、P1 = 再現性・検証可能性に重要、P2 = その他。

| ID | 項目 | 差異 | 影響度 | 実装状態 | 優先度 | 対象ファイル |
|---|---|---|---|---|---|---|
| MPS-D01 | 学習セット | 公式は `training.txt` 全 11,572 発話。audio-db は p286・p287 を除いた 10,700 | Critical | 未実装（公式リスト全件のセットが無い。`voicebank.duckdb` も未作成） | P0 | `src/data/voicebank_sets.py`、`configs/training/mp_senet.yaml`、`configs/db/init.yaml` |
| MPS-D02 | validation・best 選択 | 公式は `test.txt`（824）。audio-db は p286/p287（872）。`voicebank_demand_test` は `src/training/data/dataset.py:39` の拒否リストにあり、`docs/training/MP_SENET.md` も学習・検証への使用を禁じるため、公式方式をそのまま表現できない | Critical | 未実装。**研究上の判断が要る**（公式方式は test の選択バイアスを含む可能性がある） | P0 | `src/training/data/dataset.py`（`_set_rejection`）、`src/training/core/trainer.py`、yaml |
| MPS-D03 | 学習量 | 論文 100 epoch / 120 epoch / 500k step、コード 400 epoch。audio-db は 400 epoch（1,070,000 step）。公式 11,572 発話なら 400 epoch は 1,157,200 step | Major | 実装済み（公式コード既定に一致するが、論文と矛盾） | P0（決定が必要） | `configs/training/mp_senet.yaml` |
| MPS-D04 | 評価手順 | 48 kHz 原音で評価、SE 内で往復リサンプル、CSIG/CBAK/COVL/SSNR 未実装、STOI が `pystoi`、NB-PESQ なし | Major | 未実装 | P0（論文値と比べるなら必須） | `src/evaluation/metrics/`、`src/evaluation/pipeline.py`、`configs/db/init.yaml` |
| MPS-D05 | リサンプラ | 公式の 16 kHz 版の変換器が不明。audio-db は `torchaudio` のロード時変換 | Unknown | 実装済み・公式一致は未検証 | P1 | `src/training/data/dataset.py:331-347` |
| MPS-D06 | `batch_first` | 公式のバグを複製。バッチ構成依存。docs・テストなし。蓄積・micro-batch・world size で計算が変わる | Major | 実装済み（公式一致は §5.5 の実測のみ） | P1 | `src/se/models/mp_senet/transformer.py`、`docs/training/MP_SENET.md`、`tests/` |
| MPS-D07 | 順序 | 単 GPU 公式は固定順。audio-db は毎エポック再 shuffle | Minor | 未実装（公式順モード） | P2 | `src/training/data/sampler.py` |
| MPS-D08 | resume | 公式は lr が再開ごとに ×0.99、`best_pesq` が 0 に戻る、エポックを頭から。audio-db は厳密復元 | Minor | 実装済み（audio-db 側が上位互換。公式の挙動は再現しない） | P2 | `src/training/core/trainer.py` |
| MPS-D09 | Validation 長・PESQ | 公式は clean 全長と hop 倍数に短縮した生成波形を渡す。audio-db は両方を hop 倍数に切る。失敗の扱い（−1 平均 vs `na` 除外）と長さ境界ガードも違う | Minor | 実装済み・公式一致は未検証 | P1 | `src/training/se/mp_senet.py:147-218`、`src/evaluation/metrics/pesq.py` |
| MPS-D10 | Validation 間隔 | 5,000 step 対 2,675 step。best 候補の集合が違う | Minor | 実装済み | P2 | yaml |
| MPS-D11 | seed・初期化 | 公式 1234、audio-db 42。公式は複数 GPU で初期化が再現しない | Minor | 実装済み | P2 | yaml |
| MPS-D12 | 出力の量子化・長さ | 公式は PCM 16 bit・hop 倍数に短縮。audio-db は float・0 補完 | Minor | 未実装 | P2 | `src/se/api.py:59-88`、評価 |
| MPS-D13 | Interspeech 版 | Conformer 版、200 epoch 既定、consistency 損失なし。論文は 100 epoch・lr 半減。audio-db に実装なし | Major（V2 を再現する場合） | 未実装 | P2（V1 が主なら） | `src/se/models/`、`src/training/se/` |
| MPS-D14 | DNS 版 | 学習コード・データ生成・設定が非公開 | Unknown | 未実装（再現不能） | P2 | — |
| MPS-D15 | `num_tsconformers` | 公式は設定を無視（既定 4）。audio-db は設定を使う | Minor | 実装済み | P2 | `src/se/api.py:13, 35` |
| MPS-D16 | データ生成の DB 化 | VB は固定ペアでレシピ JSON で表現可能。ただしリサンプラ・RMS 正規化・crop は DB 外 | Minor | 実装済み | P1 | `docs/db/RECIPE.md` |
| MPS-D17 | 作業ツリーの不整合 | docs が参照する seed17/trial100 yaml、`configs/evaluation/`、`.venvs/lab`、`voicebank.duckdb` が無い | Minor | 未実装（docs 側の修正） | P1 | `docs/training/*.md`、`docs/db/*.md` |
| MPS-D18 | validation の「公式」記述 | `OFFICIAL_TRAINING_AUDIT.md` が validation 長を「GO」とするが、公式は clean を切らない | Minor | 未実装（docs 側の修正） | P1 | `docs/training/OFFICIAL_TRAINING_AUDIT.md` |

## 10. DB 中心の新データ基盤に必要な情報

### 10.1 項目別

| # | 項目 | MP-SENet（V1: VB-DEMAND）の内容 |
|---|---|---|
| 1 | DB に登録すべき音声素材 | VB-DEMAND の clean と noisy（学習 28 話者 11,572 組、test 824 組）。原音は 48 kHz。素材の `sha256` と原音 SR・長さ・チャンネル数。オンライン混合は無いので noise・RIR は不要 |
| 2 | 素材の役割 | `clean` = 教師（`dry_clean`）、`noisy` = 入力（観測）。配布済みの混合 |
| 3 | 分割・対応関係 | `training.txt`（11,572）、`test.txt`（824）の公式リスト。clean と noisy は同名ペア。公式は `training` を学習に、`test` を validation と test の両方に使う。audio-db は話者分離の別分割（p286/p287）も持つ。同じケースを複数の名前付きセットに入れる必要がある。リスト内の `|` 以降のパスは参照情報 |
| 4 | 固定して保存すべき生成条件 | 固定ペアなので `operations: []` で足りる。ただし (a) 16 kHz 変換の方式とパラメータ、(b) 変換後のサンプル数、(c) RMS 正規化の有無を保存しないと再現しない |
| 5 | 学習時にサンプリングすべき条件 | crop の開始位置（一様、両端含む）、エポックごとの順序。分布は固定ペアに対して 1 つだけ（`segment_size` と一様 crop）。混合の確率分布は無い |
| 6 | データローダー固有の処理 | noisy の RMS 正規化（crop の前、全長）、32000 サンプルの crop と末尾 0 padding、`drop_last`、単 GPU は固定順（公式）、複数 GPU は `DistributedSampler`（seed 0、`set_epoch`）。validation は全長・batch 1 |
| 7 | 共通化できる処理 | ファイルリストからのペア読み込み、リサンプル、モノラル化、noisy RMS 正規化（フラグで切替）、crop/pad、固定長バッチ化、seed とエポックからの決定的な順序・crop |
| 8 | 公式動作の維持に専用実装が要る処理 | 単 GPU の固定順（1 回 shuffle）、`DistributedSampler` の再現、`batch_first` 依存のバッチ構成の保持、validation = test の選択方式、PESQ 失敗を −1 のまま平均する公式の集約、PCM 16 書き出しと hop 倍数への短縮、公式評価スクリプト（CSIG/CBAK/COVL/SSNR、自前 STOI） |
| 9 | 再現性のために保存すべき状態 | seed、エポック、DataLoader の消費位置（バッチ位置）、ファイルリストの SHA-256、リサンプラ名と版、`librosa`・`torch`・`pesq` の版、worker 数、GPU 数（バッチ構成を決める）、`best_pesq`、scheduler の状態（公式は保存しない）、torch RNG |
| 10 | 既存レシピ JSON v1 で表現できない情報 | §10.2 |

### 10.2 既存レシピ JSON v1 の表現力

VB-DEMAND の固定ペアは、`sources.clean` と `sources.noisy`、`operations: []`、`output.reference: dry_clean` で表現できる（現行の `voicebank_sets.py` がそうしている）。これで足りない点:

| 不足 | 内容 |
|---|---|
| ロード時の SR 変換 | レシピはファイルの SR のまま。変換は学習 Dataset が `torchaudio` で行い、`case_id` に入らない。リサンプラの種類・パラメータを固定する場所が無い |
| 正規化・crop・padding | `rms_normalize`、`segment_size` は学習設定にあり、レシピにもケースにも無い |
| データセット（集合）レベルの順序・サンプリング規則 | 名前付きセットは `case_id` の集合のみ。順序、シャッフル方式、DDP の割り方、エポック定義は持たない |
| 同じケースが複数の役割（train / valid / test）に現れる場合 | 学習用と評価用のセットを分ける規則が AGENTS にあり、公式の「test = validation」は表現も実行もできない |
| 公式との整合を示す属性 | ファイルリストの出典・行・SHA-256、公式 commit、`|` 以降のパス |
| 評価プロトコル | 16 kHz 評価、PCM 16、出力長、指標セット（CSIG/CBAK/COVL/SSNR）。これは `eval` 側の設定で、レシピの問題ではない |

DNS 版（再現する場合）で、固定 1 ケースのレシピでは表現できない点（公式のオンライン混合ではなく**事前生成**だが、生成プロトコルは別途要る）:

| 不足 | 内容 |
|---|---|
| SNR 分布 | `randint(snr_lower, snr_upper)` の整数一様（上限を含まない）。レシピは固定 `snr_db` のみ |
| 雑音選択 | `np.random.randint` による一様選択。複数雑音の重ね、30 秒分の連結（無音 0.2 秒） |
| `target_level`・clipping | 雑音混合後の RMS 目標（−35〜−15 dBFS の一様）と、clip 時の全体縮小。レシピの `peak_normalize` は条件が違う |
| 活動検出 | `clean_activity_threshold=0.6` による発話の棄却 |
| RIR 適用確率 | スクリプトに分岐があるが、使用有無は UNKNOWN |
| 生成 RNG | `np.random.seed(2)`。順序依存。固定 1 ケースへ分解するには、全件を事前に生成して `noise_start` や `seed` を各ケースへ書き出す必要がある |
| 10 秒分割 | `max_seconds` は先頭だけを使う。分割して複数ケースにする規則が無い |

以上は DNS スクリプトの既定に基づく。MP-SENet が実際にどの設定を使ったかは UNKNOWN で、レシピ拡張の仕様は決められない。

## 11. 論文と公式コードの矛盾、既存ドキュメントの不整合・誤り

### 11.1 論文 ↔ 公式コード（片方を採用せず両方を記録）

| # | 論文 | コード | 再現上の影響 |
|---|---|---|---|
| 1 | P1: 100 epoch、lr は 30 epoch ごとに半減 | `O1: train.py:71-72, 278`: `ExponentialLR(0.99)`、`training_epochs` 既定 200 | Interspeech 版の学習量・スケジュールが不明。V1 では関係が薄い |
| 2 | P2v1: 単一話者は 120 epoch。P2v2: 500k step | `O: train.py:296`: 既定 400 epoch。11,572 発話・batch 4 で 1,157,200 step | 配布 checkpoint の学習量が不明（§5.2） |
| 3 | P1: test は 872 発話（2 話者） | `O: test.txt`: 824 行。P2 も 824 | P1 の誤記の可能性が高い。【推測】 |
| 4 | P2: TS-Transformer は時間軸・周波数軸に沿って動く | `O: models/model.py:124-131; transformer.py:9, 31`: `batch_first=False` のため別軸 | 学習・推論ともバッチ依存（§5.5）。README が認める |
| 5 | P2v2: DNS は「公式スクリプトに従う」3,000 時間、−5〜15 dB | 公式 repo に生成コードなし。DNS-Challenge の既定値は論文と一致しない | DNS 学習は再現不能 |
| 6 | P2 Table I: STOI は 0〜1 | `compute_metrics.py` の自前 STOI。DNS は `pysepm.stoi`（%で出力、`cal_metrics_dns.py:64`） | 2 系統の STOI 実装 |
| 7 | P2: 「学習時 2 秒に切る」 | `segment_size 32000`、padding は 0 | 一致 |
| 8 | P2: β1=0.8、β2=0.99、weight decay 0.01、lr 5e-4、0.99/epoch | `O: config.json; train.py:65-73` | 一致 |
| 9 | README: "small bug … does not affect the overall performance" | 実測ではバッチ構成で出力が変わる | 「性能に影響しない」は未検証 |
| 10 | 論文は validation の選び方を書かない | `O: train.py:293`: test.txt を validation に使う | 選択バイアスの可能性 |

### 11.2 公式コード内の不整合

- `inference.py` の既定入力 `VoiceBank+DEMAND/testset_noisy` は README の `wavs_noisy` と名前が違う（`inference.py:66`）。
- `config.json` の `num_tsconformers` は使われない（`train.py:32` は `MPNet(h)`）。
- `training.txt` の `|` 以降のパス（`wav_clean/…`）は使われない。ディレクトリ名も README の `wavs_clean` と違う。
- `train.py` に `loss_disc_g = 0`（int）の分岐があり、D の fake 項を省略するバッチが生じ得る。バッチ内に 1 件でも PESQ 失敗があると全体が省略される。

### 11.3 audio-db 内 docs の不整合・誤り

| # | 箇所 | 内容 |
|---|---|---|
| 1 | `docs/training/OFFICIAL_TRAINING_AUDIT.md`（冒頭段落の本格学習設定） | `mp_senet_seed17.yaml`、`trial100/` を参照するが、作業ツリーに無い（`b09b7f3`、2026-11-06 で削除。git 履歴にだけある） |
| 2 | 同（MP-SENet 表の `epoch` 行） | 「データ 1 周」を GO とするが、公式は 11,572 発話で 2,893 step、audio-db は 10,700 発話で 2,675 step。母数の差を書いていない |
| 3 | 同（`validation 長` 行） | 「公式は hop 倍数に揃える」と書くが、公式は clean を切らず、生成波形が hop 倍数に短くなるだけ（§7.1） |
| 4 | 同（`best` 行） | 「同じ」を GO とするが、validation の集合が違う（公式は test、audio-db は p286/p287）。選択規則だけが一致 |
| 5 | 同（`再開` 行） | 「公式 checkpoint の範囲で戻す」と書くが、公式は lr が ×0.99 される・`best_pesq` が 0 に戻る・エポックを頭からやり直す（§5.3） |
| 6 | 同（冒頭） | 「共通の VB-DEMAND（train 10,700 / valid 872 / test 824）」と書く。公式プロトコルとの差は書いてあるが、validation = test の公式方式との差は書いていない |
| 7 | `docs/training/MP_SENET.md`（「学習データ」節） | `configs/training/mp_senet.yaml` が `data/db/training.duckdb` の `libri_demand_smoke_*` を指すと書くが、現在の yaml は `data/db/voicebank.duckdb` の `voicebank_28_*`（`b09b7f3`、`b95bbd6`）。「実行」節の `CHANGE_ME_TRAIN` の説明（L89）も、現在の yaml に `CHANGE_ME` が無いため古い |
| 8 | 同（「実行」節、「追加の Python パッケージ」） | `.venvs/lab/bin/python` を使うと書くが、`.venvs/lab` は作業ツリーに無い（実在は `.venvs/{asr,deepfilternet,official,se,semambapp}`） |
| 9 | 同（「未実装」節）と yaml | 「VoiceBank-DEMAND 全件での学習」を未実装とするが、yaml のコメントは「Research full run」と書き、出力先は `data/results/training/voicebank-seed42/mp_senet`。そのディレクトリは存在しない（`ls`）。`data/results/training/` にあるのは試験・受け入れ試験（`final-accept-*`、`mp-official-*` など）で、全件学習の記録は見当たらない（ディレクトリ名のみ確認） |
| 10 | `docs/se/VALIDATION.md`（L15） | 「元リポジトリ commit `d91b88c2…`」と書くが、`requirements/official-assets.toml` と `docs/se/MODELS.md` の pin は `89932cfe…` |
| 11 | `docs/se/*`、`docs/training/MP_SENET.md` | `batch_first` の挙動と、バッチ依存の計算になることを書いていない |
| 12 | `docs/training/MP_SENET.md`（損失の節） | 「公式は PESQ が失敗したバッチは生成側の項を足さない」は正しい。ただし公式の validation PESQ では −1 が平均に入ることには触れていない |
## 12. UNKNOWN 一覧

| # | 何が | 調べた資料 | 確定できない理由 | 追加調査案 |
|---|---|---|---|---|
| 1 | 配布 checkpoint（`g_best_vb`、`g_best_dns`）の学習量・seed・GPU 数・batch | 公式 repo の `train.py`、`config.json`、README、論文 P1/P2v1/P2v2 | 学習ログも条件の記述も無い。論文が 100/120 epoch・500k step、コードが 400 epoch で食い違う | 著者への問い合わせ。checkpoint を読み込み、保存された他のキー（無ければ不可）を確認 |
| 2 | 公式の 16 kHz 版 VB-DEMAND の変換器 | README、`dataset.py` | README は「16 kHz に再サンプルせよ」とだけ書く。Google Drive の配布物は未取得 | 配布物を取得して audio-db の変換結果とサンプル差を測る（ネットワークとデータ取得が要る） |
| 3 | DNS 版の生成スクリプト・設定・件数 | 公式 repo（コード・履歴）、論文 P2v2、DNS-Challenge の 2 branch | 公式 repo にコードが無く、論文の設定は DNS の既定のどちらとも一致しない | 著者への問い合わせ。DNS 2020 の `noisyspeech_synthesizer.cfg` の複数組み合わせを試す |
| 4 | DNS test の 4,270 件が公式配布の件数と合うか | 論文のみ | 配布 test の件数を確認していない | `data/corpora/DNS2020/datasets/test_set` の件数を数える（読み取りのみ） |
| 5 | 報告値の checkpoint 選択（`g_best` か最終か） | 論文、`train.py` | 書かれていない | 著者への問い合わせ |
| 6 | journal 出版版（Neural Networks 189）の設定・数値 | arXiv v1/v2 のみ | 出版版の本文を入手していない | 出版版の取得 |
| 7 | 公式の想定 PyTorch 1.8.1 での `ExponentialLR(last_epoch)` の挙動、worker の乱数式 | 公式 `requirements.txt`、PyTorch 2.6.0 の実測とソース | PyTorch 1.8.1 を入れていない | 旧版の実装を確認（CPU のみで可能） |
| 8 | 公式学習を実際に走らせたときの GPU 数（単 GPU の固定順か、`DistributedSampler` か） | README の例は `CUDA_VISIBLE_DEVICES=0,1` | 論文が書かない | 著者への問い合わせ |
| 9 | `pystoi` と公式の自前 STOI の数値差 | `compute_metrics.py:321-374`、`pystoi 0.4.1` | 同じ入力で比較していない | 824 件を CPU で両方計算して比較 |
| 10 | 長さ不一致の PESQ（clean 全長 vs 生成波形が hop 倍数に短縮）と audio-db の両方切り方の差 | `pesq 0.0.4` の仕様、`src/evaluation/metrics/pesq.py` | 実データで比較していない | 824 件で 2 方式の PESQ を比較（CPU のみで可能） |
| 11 | audio-db の学習が公式と同等の PESQ/学習曲線になるか | — | 本格学習は未実施で、監査範囲外 | 別承認の学習ジョブ |
| 12 | 公式の `batch_first` 複製が audio-db の DDP 構成（2 GPU など）で公式と一致するか | `src/training/core/trainer.py` | GPU を使えない | CPU 2 プロセスで確認するテストを追加 |
| 13 | 未読の audio-db ファイル | `chunk_plan.py`、`checkpoint.py`、`distributed.py`、`mixing.py`、`wer.py`、`recipe.py` | 固定ペアの MP-SENet 経路で使われないと判断し、詳細を読んでいない | 必要になったときに読む |

## 13. 参照した一次資料

### 13.1 論文

| 略称 | 資料 | URL |
|---|---|---|
| P1 | MP-SENet: A Speech Enhancement Model with Parallel Denoising of Magnitude and Phase Spectra（Interspeech 2023）、arXiv:2305.13686v1（2023-05-23） | https://arxiv.org/abs/2305.13686 |
| P2v1 | Explicit Estimation of Magnitude and Phase Spectra in Parallel for High-Quality Speech Enhancement、arXiv:2308.08926v1（2023-08-17） | https://arxiv.org/abs/2308.08926v1 |
| P2v2 | 同 v2（2024-04-01） | https://arxiv.org/abs/2308.08926v2 |

PDF とテキストは `/tmp/audit-mp_senet/p_*.pdf`、`p_*.txt`。

### 13.2 公式ソース

| 略称 | 資料 |
|---|---|
| O | https://github.com/yxlu-0102/MP-SENet @ `89932cfe90d1dacb8e170e4a331d762462c21792`。ローカル pin は `src/se/official/mp-senet/`、別 clone は `/tmp/audit-mp_senet/repo` |
| O1 | 同 @ `f37e211`（2024-04-05、Interspeech 版時代の `train.py`、`datasets/dataset.py`、`models/conformer.py`、`models/generator.py`） |
| — | https://github.com/microsoft/DNS-Challenge `interspeech2020/master` @ `70f19285c36cca4df2338f9248775ddc50980c6b`（2020-07-23）、`master` @ `591184a9fcb2cbdec02520fed81a32bbbf9d73ff`（2024-03-29）。取得物は `/tmp/audit-mp_senet/dns/` |

### 13.3 参照した公式ファイルと行

- `train.py`: 20（cudnn）、29（`cuda.manual_seed`）、32-33（モデル生成）、45-73（resume、optimizer、scheduler）、75-95（Dataset、DataLoader）、102（`best_pesq`）、104（epoch ループ）、109-110（`set_epoch`）、121-146（D 更新）、164-184（G 損失）、200-209（checkpoint）、222-273（validation）、266-271（best）、277-278（scheduler step）、287-301（引数）、312-324（`main`）
- `dataset.py`: 7-29（STFT/iSTFT）、32-39（ファイルリスト）、42-95（`Dataset`）
- `models/model.py`: 117-143（TS-Transformer）、164-195（位相損失、PESQ）。`models/transformer.py`: 9、31。`models/discriminator.py`: 13-28、40-67
- `inference.py`: 40-59、66。`config.json`、`best_ckpt/*`、`VoiceBank+DEMAND/{training,test}.txt`、`requirements.txt`、`README.md`（8-10、16、29-36）
- `cal_metrics/cal_metrics_vb.py`: 10-25。`compute_metrics.py`: 28-79、321-374。`cal_metrics_dns.py`: 10-65

### 13.4 audio-db の参照ファイル

`AGENTS.md`、`.agents/{PROJECT,DATA,DEVELOPMENT,DECISIONS}.md`、`configs/training/mp_senet.yaml`、`configs/db/init.yaml`、`requirements/official-assets.toml`、`docs/training/{MP_SENET,OFFICIAL_TRAINING_AUDIT,OTHER_MODELS,VERIFICATION}.md`、`docs/se/{MODELS,VALIDATION,IMPLEMENTATION_STATUS}.md`、`docs/db/{RECIPE,CORPORA,DATASETS,TRAINING_SETS}.md`、`src/training/data/{dataset,sampler}.py`、`src/training/core/{trainer,rng}.py`、`src/training/se/{mp_senet,losses,optim,initialization}.py`、`src/se/api.py`、`src/se/models/mp_senet/`、`src/evaluation/{pipeline.py,metrics/{pesq,stoi,sisdr}.py}`、`src/data/voicebank_sets.py`、`src/data/corpora/mpsenet_filelist.py`

### 13.5 実測（CPU、/tmp、audio-db の PyTorch 2.6.0）

- `/tmp/audit-mp_senet/t_batch.py`: 公式と audio-db の `MPNet`（`dense_channel=16`）でバッチ依存を測定
- `ExponentialLR(last_epoch=3)` の再開時 lr（比 0.99）
- G 2.263M、D 0.182M
- `diff best_ckpt/config.json config.json`: 差なし。sha256: `g_best_vb`、`g_best_dns` が `official-assets.toml` と一致
