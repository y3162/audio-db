# 再現対象の確定仕様（TARGETS）

作成日: 2026-11-06。ブランチ `feature/reproduction`。調査と文書化だけの成果物で、コード・設定・DB・データ・checkpoint・結果は変更していない。

この文書は、6 モデルの「何を再現するか」を、公式 GitHub の実効動作を最優先にして定めた**再現対象の正本**である。DB やデータパイプラインの設計・実装は含まない（設計は [DB 設計](../../db/REPRODUCTION_DB_DESIGN.md)、[Pipeline 設計](DATA_PIPELINE_DESIGN.md)、[実装計画](IMPLEMENTATION_PLAN.md)）。先行する監査（[OVERVIEW](OVERVIEW.md)、[MP-SENet](mp_senet.md)、[DeepFilterNet3](deepfilternet3.md)、[MossFormerGAN](mossformergan.md)、[SEMamba++](semambapp.md)、[TF-GridNet](tf_gridnet.md)、[TF-Locoformer](tf_locoformer.md)）の詳細は、ここに写さずリンクで参照する。

> **2026-11-06 追記（採用の記録）**: §6 の D-01〜D-09 を採用した。各項の末尾に「採用結果」を足し、§2.1 の状態を更新した。**根拠・選択肢・UNKNOWN は採用前の記録のまま残している。** 採用した決定は [`.agents/DECISIONS.md`](../../../.agents/DECISIONS.md) の「採用決定済み・未実装」にあり、**どれも未実装**である。PROVISIONAL・BLOCKED は、素材・公式資料が揃うまで上げない。D-10 は決定でなく前提で、変わらない。

---

## 1. 目的と基本方針

### 1.1 目的

後続の DB・データパイプライン設計と学習実装が、**迷わず参照できる再現対象**を固定する。具体的には、6 モデルについて、公式の学習データ、データ生成方式、前処理、学習条件、検証・best 選択、評価を、一次資料から確認した実効動作で書き、確定できない点を確定できないと書く。

### 1.2 優先順位

1. 公式 GitHub が実際に実行する学習・データ・評価の実装。
2. 公式の config・recipe・script。
3. 公開 checkpoint と、それに付随する config。
4. 原論文。

論文と公式コードが食い違うときは、公式の実効動作を再現対象とし、論文の記述は参考として §4 に残す。

### 1.3 原則

| 記号 | 原則 |
|---|---|
| P-1 | 目標は**プロトコルの再現**である。浮動小数点・乱数列・マルチプロセスのスケジューリングまで含めた bit 単位の一致は目標にしない（判断は [D-09](#d-09-再現の同等性基準)）。 |
| P-2 | 公式の**実効値**を採る。YAML に書いた値ではなく、コードが実際に使う値（例: `AdamW` の既定 weight decay 0.01、ESPnet の `grad_clip` 既定 5.0）を採る。 |
| P-3 | 公式の欠陥・癖は、再現対象の一部として記録する。再現するか直すかは [D-02](#d-02-公式の欠陥癖の扱い) で決める。 |
| P-4 | 「公式が使った」ことと「audio-db の規則で許される」ことを分ける。検証集合へのテスト使用は後者に反する（[D-01](#d-01-検証集合にテスト集合を使うか)）。 |
| P-5 | 事実・推測・UNKNOWN を分ける。確認できない先行監査の記述は採用せず、「監査引用」と明記する。 |

### 1.4 状態の定義

状態は**仕様の確定度**であり、実装の進み具合ではない。

| 状態 | 意味 |
|---|---|
| CONFIRMED | 再現対象と、その実効条件が一次資料で確定している。 |
| PROVISIONAL | 再現対象は決まったが、一部の条件が未確定、または確定には前提（素材・決定）が要る。実装の入口にはなるが、未確定箇所は仮置きと明示する。 |
| BLOCKED | 公式資料・素材が不足し、現時点で再現条件を決められない。 |
| PENDING_DECISION | 公式資料では決まらず、研究上の判断（ユーザー判断）が要る。 |

状態は 1 モデルに 1 つ書く。同じモデルの中に別状態の副次項目があるときは §2.2 に分けて書く。

### 1.5 根拠の表記

- `[repo@SHA : path : 行 : 関数/キー]` の形式で書く。略号は §1.6。
- **【R】**: 今回の作業で、一次資料（公式 repo の checkout、同梱 zip、HF の設定、論文テキスト）を直接読んで確認した。
- **【A】**: 先行監査に書かれており、今回は一次資料を読み直していない。重要度の高い【A】は、状態を PROVISIONAL に下げる理由として §3 に明記した。
- 一次資料の checkout は `/tmp/targets-verify/`（リポジトリ外）に置いた。公式 repo の HEAD は、ESPnet を除きピン留めと一致する（確認日 2026-11-06）。

### 1.6 略号と固定するバージョン

| 略号 | 対象 | 固定する版 | 備考 |
|---|---|---|---|
| MPS | `yxlu-0102/MP-SENet` | `89932cfe90d1dacb8e170e4a331d762462c21792`（2025-05-19、upstream HEAD） | `best_ckpt/` の追加は 2024-09-08 の `955d2fa`。HEAD の `train.py` はそれ以後変わらない【R】 |
| CVS | `modelscope/ClearerVoice-Studio` | `6b3774dc79c46ae8bed2a4fa5f706f0ac8c75c61`（2025-08-14、upstream HEAD） | `train/speech_enhancement`（`T/`）は `v0.1.1` 以後差分なし【R】 |
| DFN | `Rikorose/DeepFilterNet` | **学習コードは `v0.5.0`（`a76dbcb540e177ba94dd560809fd6af496b18913`、2023-05-26）を第一候補**。既存のピン `d375b2d`（HEAD、2024-09-25）は参照用に残す | 出荷 `DeepFilterNet3.zip` の最終変更は `82b0c7a`（2023-05-23）。根拠は §3.1 DFN |
| SMB | `Eddieyjlee/SEMambapp`（旧 `infected4098/SEMambapp`） | `cd369c89a8106072c077b468417b765b227afd74`（2026-10-26、HEAD） | ライセンスファイル無し |
| TFL | `merlresearch/tf-locoformer` | `7a615460d347ff7334a13dbb831d16280da72cdc`（`v1.1.0`、2025-08-08） | 学習は ESPnet に注入して行う |
| ESP90 | `espnet/espnet` | `90eed8e53498e7af682bc6ff39d9067ae440d6a4` | TF-Locoformer の公式が学習・評価に使った版【A】 |
| ESP | `espnet/espnet` | `0d136c5456d403cb2c0a4c13e76e8136e32e9b7a`（既存ピン。upstream はこれより先へ進んでいる） | TF-GridNet の採用版（根拠は §3.5） |

---

## 2. 6 モデルの再現対象一覧

### 2.1 一覧

| モデル | 公式リポジトリ | 固定 commit / tag | variant | 学習レシピ | データ方式 | 状態 |
|---|---|---|---|---|---|---|
| MP-SENet | https://github.com/yxlu-0102/MP-SENet | `89932cf` | journal 版コードの HEAD ＋ VoiceBank+DEMAND（`best_ckpt/g_best_vb` に対応） | `train.py`、`config.json`（既定 400 epoch） | 固定ペア | **PROVISIONAL**（D-01、D-03 採用済み。実効条件は確定。残る UNKNOWN は 16 kHz 版の公式の変換器と、実学習の epoch・GPU 数。変換器は派生素材として明示し、差を記録する）|
| MossFormerGAN | https://github.com/modelscope/ClearerVoice-Studio | `6b3774d` | `MossFormerGAN_SE_16K`。**公開学習コード `T/` ＋ VoiceBank+DEMAND 28 話者**（配布 checkpoint の学習ではない） | `T/train.py`、`T/solver.py`、`T/config/train/MossFormerGAN_SE_16K.yaml` | 固定ペア | **PROVISIONAL**（D-01、D-04 採用済み。実効条件は確定。残る UNKNOWN は 16 kHz 版の公式の変換器）|
| DeepFilterNet3 | https://github.com/Rikorose/DeepFilterNet | 学習コード `v0.5.0`（`a76dbcb`）＋ 出荷 `config.ini`（`82b0c7a`） | 出荷 `DeepFilterNet3.zip`（48 kHz、非 low-latency） | `DeepFilterNet/df/train.py`、出荷 `config.ini`、Rust `libDF`/`pyDF-data` | オンライン混合 | **PROVISIONAL**（学習データの実体と `dataset.cfg` が UNKNOWN。D-06、D-09 採用済み。公式素材の Dataset は BLOCKED として別管理し、代替素材は別プロトコル準拠）|
| SEMamba++ | https://github.com/Eddieyjlee/SEMambapp | `cd369c8` | **修正後の公開実装＋ HEAD の `config.yaml`**（公開重み `yongjoonlee/semambapp` に対応）。論文の重み（バグあり）ではない | `train.py`（引数は `main()` に直書き）、`config.yaml` | オンライン混合（GSR 劣化） | **PROVISIONAL**（出荷状態では学習が起動しない。素材とマニフェストが無い。D-05 採用済み。最小補修は許すが未実装。代替素材は別プロトコル準拠）|
| TF-GridNet | https://github.com/espnet/espnet | ESPnet `0d136c5`（recipe は `466e8a89d4d7`、学習環境は `239f5166a`） | **ESPnet 公開 SE recipe `egs2/ms_snsd/enh1`**（v1 `tfgridnet`、2.55 M パラメータ）。**論文の実験の再現ではない** | `egs2/ms_snsd/enh1/run.sh`、`conf/tuning/train_enh_tfgridnet.yaml` | 事前生成（MS-SNSD 合成） | **PROVISIONAL**（生成は未シードで同一物を再現できない。D-07 採用済み。「ESPnet 公開 SE レシピの再現」で、seed 固定は差として記録）|
| TF-Locoformer | https://github.com/merlresearch/tf-locoformer | TFL `7a61546`（`v1.1.0`）＋ ESP90 | `dns_ins20`、Medium（B=6、15.0 M）、`train_enh_tflocoformer.yaml` | `egs2/dns_ins20/enh1` に注入して `run.sh` | 事前生成（DNS 2020 合成 3000 h） | **PROVISIONAL**（`cv_synthetic_small` が UNKNOWN のまま。3000 h 生成の承認が要る。D-08 採用済み。段階生成）|

### 2.2 副次ターゲットの状態

| モデル | 副次ターゲット | 状態 | 理由 |
|---|---|---|---|
| MP-SENet | `g_best_dns`（DNS 版）の学習 | **BLOCKED** | 学習コードもデータ生成コードも無い【R】。評価だけは公開重みで可能 |
| MP-SENet | Interspeech 2023 版コード（旧 commit）の再現 | 対象外 | journal 版 HEAD が配布重みに対応する |
| MossFormerGAN | 配布 checkpoint（`alibabasglab/MossFormerGAN_SE_16K`）の学習 | **BLOCKED** | 社内データを含み、生成コードが無い。D の lr と LR scheduler が公開コードと食い違う【R】 |
| MossFormerGAN | README の PESQ 3.47 | 目標にしない | 大規模データ学習の checkpoint の値である【R】 |
| DeepFilterNet3 | 公式データ（DNS4 ＋ PTDB/VCTK ×10）での学習 | **BLOCKED**（素材とリストが不足） | `dataset.cfg`・件数・RIR 集合が非公開【A】。DNS4・PTDB は手元に無い |
| DeepFilterNet3 | 「プロトコルの再現」（代替素材で同じ混合分布を作る） | 代替素材で実装済み。公式データセットの再現は BLOCKED | 分布は `DFN3_ONLINE_V1`。登録は `dfn3.online.substitute@1`。ビット一致は主張しない。検証は [P8_VERIFICATION](P8_VERIFICATION.md) |
| SEMamba++ | 論文の重み（バグあり）の再現 | **BLOCKED** | バグ入りコードの版が特定できない |
| SEMamba++ | 論文の学習データ（VCTK 0.92 ＋ GSR 素材） | PROVISIONAL | VCTK は 0.80 のみ、Arni の RIR が無い【A】 |
| TF-GridNet | 論文（arXiv:2211.12433）の各タスク | 対象外 | 本書は「ESPnet 公開 recipe の再現」と明記する |
| TF-Locoformer | 論文 Table 4（DNS）の数値 | 参考 | プロトコルの再現が目標で、数値一致は目標にしない |

---

## 3. モデル別の再現仕様

各表の「根拠」は §1.5 の記法による。先行監査の該当節は各モデルの冒頭にリンクした。

### 3.1 DFN: DeepFilterNet3

詳細: [deepfilternet3.md](deepfilternet3.md)。

**コードの版の決定【R】**

- 出荷 `models/DeepFilterNet3.zip` は 2023-05-23 の `82b0c7a` で入り、`v0.5.0`（2023-05-26）と HEAD `d375b2d`（2024-09-25）で zip の内容は同一（SHA-256 `49c52edc…22284d2` が `requirements/official-assets.toml` と一致）。
- 学習に効くコード（`df/train.py`、`df/lr.py`、`df/loss.py` の損失本体、`libDF` のデータ生成）は、モデル作成時点と `v0.5.0` で同一。HEAD は 2023-06〜07 以降の変更を含む。
- 差分は 3 点: (a) `DATALOADER_GAINS` の既定 `[-6, 0, 6]` は `v0.5.0` の `dataset.rs` L638 に既に在る（先行監査の UNKNOWN は解消）。(b) `LocalSnrTarget` の範囲は `v0.5.0` で ±5、HEAD で ±1。(c) `FftDataset` の `min_nb_erb_freqs` は `v0.5.0` で 1 固定、HEAD で出荷 ini の 2 が効く。
- **採用: 学習コードは `v0.5.0`**。出荷モデルと同じ時期のコードだからである。HEAD（既存ピン）は「現行コード」の参照に留める。(b)(c) の影響は、`v0.5.0` と HEAD の両方の動作を §5 の「許容できる逸脱」として記録する。

| 項目 | 確定内容 | 根拠 | 確度 |
|---|---|---|---|
| エントリ | `DeepFilterNet/df/train.py`（`main`）、設定は出荷 `config.ini`（SHA-256 `415eb925…c4290`） | 同 zip | 【R】 |
| データ方式 | **オンライン混合**。Rust `libdfdata` が発話ごとに seed 付き RNG（`seed_from_u64(seed + sample_idx)`）で混合を生成する。HDF5 は素材のみで、混合済みは保存しない | `[DFN@v0.5.0 : libDF/src/dataset.rs : 混合生成]`、先行監査 §4 | 【R】【A】 |
| 学習データ（公式） | DNS4 全体 ＋ PTDB-TUG ＋ VCTK（×10 の重み）。論文 `dfn3.txt:115`。HDF5 への変換は `scripts/download_process_dns4.sh`。件数・`dataset.cfg`・RIR 集合は非公開 | 論文、`download_process_dns4.sh:22-75` | 【A】（zip 内の ini に素材の記述は無い【R】） |
| train / valid / test | DNS4 の `dev_testset` を含む blob を VALID、他を TRAIN とする。TEST と `dataset.cfg` は作らない。評価は VoiceBank+DEMAND test の固定ペア | `download_process_dns4.sh:46-50`、`test_voicebank_demand.py:26-27` | 【A】 |
| SNR | `[-100, -5, 0, 5, 10, 20, 40]` から一様に 1 値（−100 は雑音のみ）。valid/test にも同じ | 出荷 ini `dataloader_snrs` | 【R】 |
| gain | `[-6, 0, 6]` から一様（ini に記述無し、既定） | `[DFN@v0.5.0 : libDF/src/dataset.rs : L638]` | 【R】 |
| 雑音 | 発話ごとに 2〜5 本（`uniform(2,6)` を整数化）を加算。train のみ 5 % で合成雑音。noise augmentation あり。clipping ガードあり | 先行監査 §4 | 【A】 |
| 残響 | `p_reverb = 0.1`。speech と noise に独立に適用。RIR の集合は不明 | 出荷 ini | 【R】（RIR 集合は UNKNOWN） |
| 他の歪み | `p_bandwidth_ext = p_clipping = p_air_absorption = p_zeroing = p_interfer_sp = 0`（実質無効） | 出荷 ini | 【R】 |
| 窓 | `max_sample_len_s = 3.0`（144,000 サンプル、48 kHz）。短い素材は**別発話の連結**で埋める（零パディングではない）。valid/test にも同じ窓 | 出荷 ini、`dataset.rs` | 【R】【A】 |
| 正規化 | 発話 RMS 正規化は無い | コード | 【A】 |
| STFT・DF | `sr 48000`、`fft 960`、`hop 480`、`nb_erb 32`、`nb_df 96`、`df_order 5`、`df_lookahead 2`、`lsnr -15…35`、`min_nb_erb_freqs 2` | 出荷 ini | 【R】 |
| batch | `batch_size = 64`（train の実効は epoch ごとの `batch_size_scheduling`: 0/16、2/24、5/32、10/64、20/128、40/9999 を `min(.., 64)`）。valid は 64 | `df/train.py` L175-182 | 【R】 |
| epoch | `max_epochs = 120`。epoch の長さは speech の key 数 × `sampling_factor` / batch（端数 batch は残す） | 出荷 ini、`dataset.rs` | 【R】【A】 |
| optimizer | `AdamW(lr 1e-3, betas (0.9, 0.999), amsgrad=True)`、`weight_decay 1e-12`。`weight_decay_end 0.01` のとき wd を 1e-12 → 0.01 に cosine で増やす | 出荷 ini、`df/train.py` | 【R】【A】 |
| scheduler | step ごと。warmup 3 epoch（`lr_warmup 1e-4` → 1e-3）の後に cosine（`lr_min 1e-6`）。batch 変更ごとに表を作り直す。lr は小数 epoch の関数 | `df/lr.py`、`df/train.py` | 【A】 |
| 勾配クリップ | `clip_grad_norm_(1.0, error_if_nonfinite=True)`。ini に無く、コードに直書き | `df/train.py` | 【A】 |
| 損失 | `MultiResSpecLoss`（FFT 256/512/1024/2048、`gamma 0.3`、振幅 500、複素 500）＋ `LocalSnrLoss`（1e-3）。他は 0。**損失本体は MSE**（論文の norm ratio ではない） | 出荷 ini、`df/loss.py` | 【R】【A】 |
| 乱数 | `[train] seed = 43`。main の RNG は 42 固定（`lib.rs`）で設定の seed と無関係。worker の RNG は sample ごとに再設定する | 出荷 ini、`df/train.py` L92-93 | 【R】【A】 |
| validation | 3 秒窓のオンライン混合を **seed 固定**（`sample_idx` のみ）で毎回同じに作り、batch 平均を取る | `dataloader.rs`、`pyDF-data/src/lib.rs` | 【A】 |
| best / early stopping | 検証損失が最小の epoch を `.best` として保存。出荷ファイル名は `model_120.ckpt.best` で、最終 epoch が best | 出荷 zip | 【A】 |
| 評価 | VoiceBank+DEMAND test（824）を `test_voicebank_demand.py` で評価（PESQ、CSIG、CBAK、COVL、STOI）。論文 Table 1: PESQ 3.17 | 論文 | 【A】 |
| 出荷 checkpoint との対応 | 出荷 ini は `v0.5.0` の学習設定そのものとして読める。学習データ・乱数・停止の履歴は不明 | | 【R】 |
| UNKNOWN | `dataset.cfg`（ファイル一覧・`sampling_factor`）、素材の件数、RIR 集合、DNS4 の取得時点、実際の GPU 数 | | |
| 採用理由 | 出荷モデルの学習時期のコードと、出荷 ini の実効値が、プロトコルの最も近い定義だから | | |

### 3.2 MFG: MossFormerGAN

詳細: [mossformergan.md](mossformergan.md)。

**目標の決定**: 公開学習コード `T/` を第一目標にする。配布 checkpoint の学習レシピは再現できず（§2.2）、README の PESQ 3.47 は目標にしない【R】。

| 項目 | 確定内容 | 根拠 | 確度 |
|---|---|---|---|
| エントリ | `T/train.py`（`main`）、`T/solver.py`（`Solver.train`）、`T/losses/loss.py`。設定 `T/config/train/MossFormerGAN_SE_16K.yaml`。起動は `T/train.sh`（`torch.distributed.launch`、`n_gpu=2`） | `[CVS@6b3774d : T/train.sh]` | 【R】 |
| データ方式 | **固定ペア**。noisy/clean の 2 列 scp を読む。オンライン混合は無い | `[CVS@6b3774d : T/dataloader/misc.py : L15-37]` | 【R】【A】 |
| train | `tr_demand_28_spks_16k.scp`（11,572 行）。VoiceBank+DEMAND の 28 話者 | YAML L14 | 【R】 |
| validation | `cv_demand_testset_16k.scp`（**824 行 = 公式 test**）。**検証にテスト集合を使う**（D-01） | YAML L15、`T/README.md` L62 | 【R】 |
| test | `tt_list` は YAML に無く、学習中に走らない | `[T/solver.py : L204]` | 【R】 |
| 前処理 | 素材は事前に 16 kHz へ変換済み（変換法は UNKNOWN）。noisy と clean を**独立に** `audio_norm`（RMS −25 dBFS を 2 段）で正規化し、相対ゲインが保たれない。次に長さ 32,000 の窓（2 秒）。短ければ**先頭に貼り付けて末尾を 0 埋め**、そうでなければ一様なランダム crop。solver が noisy の発話ごとに `c = sqrt(T/(Σx²+1e-6))` を掛け、損失内で clean にも同じ `c` を掛ける | `[T/dataloader/dataloader.py : L34-43, L65-76]`、`[T/solver.py : L266-268]`、`[T/losses/loss.py : L29-30]` | 【A】 |
| STFT | hamming（`periodic=False`）、`win=fft=400`、hop 100、`center=True`、power 圧縮 0.3 | `[T/solver.py : L269-271]` | 【A】 |
| batch・蓄積 | `batch_size 1`、`accu_grad 1`（YAML）、`effec_batch_size 2`（GPU あたり）。GPU あたり micro-batch 1 × 蓄積 2 で 1 更新。**global batch = 2 × GPU 数**（`train.sh` の 2 GPU なら 4） | YAML L28-32、`[T/solver.py : L278-292]` | 【A】 |
| DDP / sampler | 独自 `DistributedSampler`。rank r の担当は index が r mod world のものに**固定**され、epoch ごとに変わるのは rank 内の並びだけ。sampler の seed は常に 0（`args.seed` と無関係）。検証側の並びは毎回同じ | `[T/dataloader/dataloader.py : L192-242]` | 【A】 |
| optimizer | G: `AdamW(lr 5e-4)`、D: `AdamW(lr 1e-3)`。weight decay は YAML の 1e-5 を**使わず**、`AdamW` の既定 **0.01**。betas 既定 | `[T/train.py : L56-57]` | 【R】 |
| scheduler | 無い。plateau は solver の独自規則 | `[T/solver.py : L215-252]` | 【R】 |
| 勾配クリップ | `clip_grad_norm_(.., 10)` を G と D に、**micro-batch ごと** | `[T/solver.py : L283, L286]` | 【A】 |
| 損失 | `0.1·loss_ri + 0.9·loss_mag + 0.2·time_loss + 0.05·gen_loss_GAN`。**`time_loss` は `power_uncompress(labels_real, pred_imag)`（clean の実部と予測の虚部）を iSTFT して L1**（公式の欠陥） | `[T/losses/loss.py : L26-77, L49]` | 【R】 |
| D の学習 | `MSE(D(label,label),1) + MSE(D(label,pred.detach()), pesq_norm)`、`pesq_norm=(wb PESQ−1)/3.5`。PESQ が 1 件でも失敗すればバッチを飛ばす | `[T/losses/loss.py : L62-75]` | 【A】 |
| 更新順 | G の backward → G クリップ → D の backward → D クリップ → 蓄積数に達したら G と D を step | `[T/solver.py : L278-292]` | 【A】 |
| epoch | `max_epoch 120`、ループは 0〜120 の 121 回。1 epoch は rank あたり N/world 個の micro-batch | `[T/solver.py : L185-187]` | 【A】 |
| early stopping / plateau | 検証損失が `best_val_loss` 以上のとき、**悪化した値で `best_val_loss` を上書き**し、カウント +1。カウント 5 で G の lr を 0.5 倍（best checkpoint を読み戻し、epoch/step も巻き戻る）。**10 で停止**。改善時はカウント 0。カウンタは checkpoint に保存されず、再起動で 0 に戻る | `[T/solver.py : L65-66, L215-252]` | 【R】 |
| best 選択 | 「直前の epoch より検証損失が小さい」ときに `last_best_checkpoint` を更新（真の best-so-far ではない）。halving 時は best ファイルが「best の重み＋半減した lr の optimizer」で上書きされる | `[T/solver.py : L164-182, L229-252]` | 【R】 |
| validation | 2 秒ランダム crop（検証にも同じ窓）。G の**全損失**（GAN 項と PESQ を含む）の平均。G は eval、**D は train モードのまま** | `[T/dataloader/dataloader.py : L129]`、`[T/solver.py : L197, L275, L326-329]` | 【R】【A】 |
| seed | YAML `seed: 777`、`cudnn.deterministic=True` | | 【A】 |
| 評価 | 学習側に評価スクリプトは無い。README の評価は `speechscore`（16 kHz VB test） | | 【A】 |
| 出荷 checkpoint との関係 | `last_best_checkpoint.pt`: epoch 7 / step 1,169,381、optimizer の lr 0.0005（D も 0.0005）、`initial_lr` あり = 公開コードに無い LR scheduler が在った。**公開コードの産物ではない** | `requirements/official-assets.toml`、先行監査 §2.2 | 【A】 |
| `train.sh` の罠 | 既定 `network=MossFormer2_SE_48K`、`init_checkpoint_path=…/last_best_checkpoint.pt`。ファイルがあれば **finetune として始まる**（G の lr 1e-4）。scratch には `init_checkpoint_path='None'`（文字列）を渡す | `[T/train.sh]`、`[T/solver.py : L65-66, L119-149]` | 【R】 |
| UNKNOWN | 素材の 16 kHz 変換の方法、`speechscore` 以外の評価、配布 checkpoint の実データ | | |
| 採用理由 | 公開コードは「実際に実行できる唯一のプロトコル」であり、優先順位 1 に当たるから | | |

### 3.3 MPS: MP-SENet

詳細: [mp_senet.md](mp_senet.md)。

| 項目 | 確定内容 | 根拠 | 確度 |
|---|---|---|---|
| エントリ | `train.py`（`main` → `train`）、`config.json`。README の学習コマンドは `CUDA_VISIBLE_DEVICES=0,1 python train.py --config config.json` | `[MPS@89932cf : README.md : L20]` | 【R】 |
| variant | journal 版（`MPNet`、Generator 2.263 M）＋ VoiceBank+DEMAND。`best_ckpt/g_best_vb` が HEAD の `train.py` に対応する（`g_best_*` は `955d2fa` 以後不変）。`num_tsconformers` は `MPNet(h)` で使われず 4 ブロック固定 | | 【R】 |
| config 値 | batch 4（GPU 数で割る）、segment 32,000、16 kHz、n_fft 400、hop 100、win 400、dense_channel 64、compress_factor 0.3、lr 5e-4、adam_b1/b2 0.8/0.99、lr_decay 0.99、seed 1234、num_workers 4 | `[MPS@89932cf : config.json]` | 【R】 |
| データ方式 | **固定ペア** | | 【R】 |
| train | `training.txt`（11,572）。VoiceBank+DEMAND の 28 話者 | `[dataset.py : L32-39]` | 【R】 |
| validation | **`test.txt`（824）= 公式 test**（`--input_validation_file` の既定）。D-01 | `[train.py : L293]` | 【R】 |
| test / best | best 選択も同じ `test.txt`。`epoch >= best_checkpoint_start_epoch(40)`（0 始まり）かつ validation PESQ 平均が `best_pesq`（初期値 0）を超えたとき `g_best`（G のみ）を保存 | `[train.py : L102, L266-271]` | 【R】 |
| 前処理 | `librosa.load(sr=16000)` で 16 kHz に変換。clean/noisy の短い方に長さを揃える。**noisy の RMS で正規化**（`norm_factor = sqrt(len/Σnoisy²)` を両方へ）。crop は `randint(0, L−32000)`（両端含む）、短ければ末尾 0 埋め | `[dataset.py : L63-66, L75-78, L82-90]` | 【R】 |
| DDP / sampler | `h.batch_size = int(batch_size / device_count)`（global は 4 のまま）。**2 GPU 以上**: `DistributedSampler`（seed 0、`set_epoch`）、`Dataset` の shuffle 無し、`drop_last`。**1 GPU**: `Dataset` が seed 1234 で**1 回だけ** shuffle し、`DataLoader(shuffle=False)` なので**全 epoch で順序が同じ** | `[train.py : L75-95, L312-316]`、`[dataset.py : L46-48]` | 【R】 |
| optimizer | `AdamW(lr 5e-4, betas [0.8, 0.99])`、weight decay は**未指定 = 既定 0.01**（論文 v2 の 0.01 と一致） | `[train.py : L65-66]`、論文 v2 L743-751 | 【R】 |
| scheduler | `ExponentialLR(gamma 0.99)`、**epoch ごと**に step | `[train.py : L72-73, L277-278]` | 【R】 |
| 損失 | `0.9·mag + 0.3·phase + 0.1·complex(MSE×2) + 0.1·consistency(MSE×2) + 0.05·metric + 0.2·time` | `[train.py : L166-181]` | 【A】 |
| D の更新 | D → G の順。D は `(clean,clean)→1`、`(clean, G出力.detach())→(PESQ−1)/3.5`。PESQ が 1 件でも失敗すれば fake 項は 0 | `[train.py : L132-146]` | 【A】 |
| epoch / step | `--training_epochs` の既定 **400**（1 epoch = 2,893 step、計 1,157,200 step）。論文: 100 / 120 epoch、または 500k step | `[train.py : L296]`、論文 | 【R】 |
| validation 頻度 | `steps % 5000 == 0`（約 1.73 epoch ごと）。全長・batch 1。指標は PESQ（wb）平均。PESQ 失敗は −1 のまま平均に入る | `[train.py : L222-273]` | 【R】【A】 |
| early stopping | 無い | | 【R】 |
| `batch_first` の欠陥 | `MultiheadAttention`/`GRU` が `batch_first=False`。時間・周波数の軸が崩れ、**出力がバッチの構成で変わる**。公式 README が自認し、再学習なら `True` を推奨。実測: 同じ重みで batch 4 と 2 でサンプル 0 が最大 0.031 ずれる | `[README.md : L8-10]`、`[models/transformer.py : L9, L31]`、先行監査 §5.5 | 【R】（README・コード）【A】（実測値） |
| 公式の再開 | lr を余分に減衰し、`best_pesq` を 0 に戻す | `[train.py : L102]`、`[utils.py]` | 【R】【A】 |
| 評価 | `inference.py` ＋ `compute_metrics`（PESQ、CSIG、CBAK、COVL、SSNR、STOI）。報告値は VB test 824 | README、論文 | 【A】 |
| 出荷 checkpoint との関係 | `g_best_vb` は HEAD と同じ journal 版コードの best。論文の epoch（100/120）と 400 の関係、学習時の GPU 数、報告値が `g_best` かは UNKNOWN | | 【R】（UNKNOWN の確認） |
| UNKNOWN | 実学習の epoch 数と GPU 数、`g_best_dns` の学習、論文の test が 872 か 824 か | | |
| 採用理由 | HEAD が `g_best_vb` と対応し、優先順位 1 の実効動作が確定できるから | | |

### 3.4 SMB: SEMamba++

詳細: [semambapp.md](semambapp.md)。

**版と config の決定【R】**

- `config.yaml` は 2026-06-05 に、切り出し長 24,000 → 48,000、SNR −10…20 → −5…20、劣化確率 0.7 → 0.5 へ変更された。README はその後「公開重みは `config.yaml` に従う」旨を書く。`config_orig.yaml`（2026-06-06）は変更前の値の写しである。
- **第一候補は HEAD の `config.yaml`（修正後の公開実装）**。論文の値は §4 に参考として残す。
- 出荷状態では学習を起動できない（修正は「補修」に当たる。§5）。

| 項目 | 確定内容 | 根拠 | 確度 |
|---|---|---|---|
| エントリ | `train.py`（`main()` に引数を直書き: `config: 'semambapp_cfg.yaml'`、`training_epochs 100`、`checkpoint_interval 5000`、`validation_interval 5000`、データ JSON のパス）。**既定の config ファイル名が存在しない** | `[SMB@cd369c8 : train.py : L441-487]` | 【R】 |
| データ方式 | **オンライン混合**。`GSRDataset` がマニフェスト 3 つ（speech / noise / rir）から毎回抽選する | `[dataset.py]` | 【R】【A】 |
| 学習素材（公式の記述） | 論文: VCTK **0.92**（p280、p315 を除く）、16 kHz、「VCTK-GSR train/test」。noise: WHAM!/DNS 2020、RIR: Arni/DNS5。学習 44 時間（Table 3） | 論文 §4.1 | 【A】 |
| マニフェスト | `data/*.json` は**空**。件数・話者・長さは UNKNOWN | リポジトリ | 【R】 |
| 劣化 | noise 1.0、reverb 0.5、bandwidth 0.5、clipping 0.5（独立）。適用順は reverb → noise → clipping → bandwidth で固定。SNR は {−5, 0, 5, 10, 15, 20} から一様（有音フレームの電力比 = VAD 基準。reverb 後の信号に対して） | `config.yaml`、`utils.py` | 【R】【A】 |
| 窓 | `segment_size 48,000`（3 秒、16 kHz） | `config.yaml` | 【R】 |
| batch | `batch_size = batch_size // n_gpus`（global は config 値）。公式は global 8 を 1 forward（40 GB では OOM） | `[train.py : L474]` | 【R】【A】 |
| DDP | `n_gpus > 1` で `mp.spawn`（`localhost`、ポートは `50000 + randint(0,100)`）。train loader は `shuffle=False` ＋ `DistributedSampler` | `[train.py : L101-103, L478-487]` | 【R】 |
| optimizer | G・D とも `AdamW(lr, betas=[adam_b1, adam_b2])`、weight decay **未指定 = 0.01** | `[train.py : L163-164]` | 【R】 |
| scheduler | `ExponentialLR(lr_decay)`、epoch ごと | `[train.py : L45-53, L435-436]` | 【R】 |
| 損失 | 振幅 0.9、位相 0.3、複素 0.1、整合 0.1、mel 0.1、敵対 1、特徴整合 1（先行監査 §6）。振幅損失は**MSE**（論文は L1） | | 【A】 |
| epoch | `training_epochs 100`。step 数は素材の件数に依存し UNKNOWN | `[train.py : L441-487]` | 【R】 |
| validation | 5000 step ごと（`validation_interval`）。検証ループ内で PESQ・UTMOS・MRSTFT。**best 選択は無い**（checkpoint は 5000 step ごと） | `[train.py : L331, L371]` | 【R】 |
| early stopping | 無い | | 【R】 |
| 評価 | 公式に評価スクリプトは無い。論文は SCOREQ・UTMOS・DNSMOS・PESQ・LSD・LPS で VCTK-GSR test、URGENT 2025、DNS 2020、AATC 2025 を評価 | | 【A】 |
| 出荷物との対応 | 公開重み（`semambapp.pth`、2026-08-20）は、チャンネル FFN のバグ修正後に「同じ設定で」再学習した重み。**再学習のデータは README に無い**。論文の重みは第 3 解像度の Channel FFN が使われないバグ入り | README、先行監査 §1.2 | 【R】 |
| 補修が要る点 | (1) 既定 config 名が存在しない（`config.yaml` に向ける）、(2) `dataset.py` が存在しない `utils.util` を import、(3) `data/*.json` が空、(4) `infer_folder.py` が `infer.py` に無い関数を import | `[train.py : L441-487]`、`[dataset.py : L11]` | 【R】 |
| UNKNOWN | 学習データの実体と件数、VCTK-GSR の分割規則、乱数の扱い、検証データ（`val_clean/degraded.json`）の作り方 | | |
| 採用理由 | 公開重みと対応する最新の実装を、優先順位 1・2・3 の順に採るから | | |

### 3.5 TFG: TF-GridNet（ESPnet 公開 SE recipe の再現）

詳細: [tf_gridnet.md](tf_gridnet.md)。

**宣言**: これは **ESPnet の公開 SE recipe（`egs2/ms_snsd/enh1`）の再現**であり、TF-GridNet 論文（arXiv:2211.12433）の実験の再現ではない。モデルは v1 `tfgridnet` の小型構成（2,552,790 パラメータ、362 テンソル）で、論文の大型構成ではない【R】。

**ESPnet の版の決定【R】**

- HF `espnet/ms_snsd_tfgridnet` の `config.yaml` の `version` は `'202511'`、`dist_world_size` は 2。ESPnet の `version.txt` は 2025-11 から 2026-04（`v.202604`）まで `202511` だった。
- recipe 本体は `466e8a89d4d7`（2026-02-18）で完成し、recipe の README が学習環境として記す ESPnet は `239f5166a`（2026-02-05）。
- `239f5166a` / `466e8a89d` から既存ピン `0d136c5` への差分で、TF-GridNet の学習経路（`tfgridnet_separator.py`、損失 `criterions/*`、`espnet_model.py`、`chunk_iter_factory.py`、`initialize.py`、`reducelronplateau.py`、`trainer.py`、`reporter.py`）に**動作の差は無い**: 差は torch の旧版分岐の削除、エラーメッセージ、AMP の API、`W = int(state.choice(..., size=1).item())`（同じ乱数消費）、再開時の early stopping 再判定だけ。
- **採用: ESP `0d136c5`**（既存ピンを維持。状態 CONFIRMED）。

| 項目 | 確定内容 | 根拠 | 確度 |
|---|---|---|---|
| エントリ | `egs2/ms_snsd/enh1/run.sh`（`--ngpu 2 --ref_num 1 --use_noise_ref true --max_wav_duration 31 --inference_model valid.loss.best.pth`）、`conf/tuning/train_enh_tfgridnet.yaml`（`conf/train.yaml` はそれへのリンク） | `[ESP@0d136c5 : egs2/ms_snsd/enh1]` | 【R】 |
| データ | MS-SNSD（Microsoft。MIT）から生成。`total_hours_train=30`、`total_hours_test=1` が README の既定。**事前生成**、生成後に `tr_ms_snsd` / `cv_ms_snsd`（train 側の 20 %）/ `tt_ms_snsd` | recipe README | 【R】【A】 |
| 生成条件 | 名目 SNR [0, 10, 20, 30, 40] を全ペアで。**実 SNR は名目の半分**（`snr_mixer` の仕様: [0, 5, 10, 15, 20] dB）。各ファイルを −25 dBFS RMS に正規化して混合。残響なし。**乱数は未シード**で、生成物は再現できない | `ms_snsd_noisyspeech_synthesizer.py`、`audiolib` | 【A】 |
| 前処理 | `speech_volume_normalize: null`。モデル内部（分離器）が入力の標準偏差で正規化 | HF config、`tfgridnet_separator.py` | 【R】【A】 |
| チャンク | `iterator_type: chunk`、`chunk_length 48000`（3 秒）、shift 0.5、`Z` は乱数、`L ≤ W` の発話は捨てる | yaml、`chunk_iter_factory.py` | 【R】 |
| batch・DDP | `batch_type folded`、`batch_size 4`（global）= rank あたり 2。**2 rank の DDP** | yaml、HF config | 【R】 |
| epoch の定義 | `num_iters_per_epoch 5000` は **rank あたり 5,000 個の元発話**（2 rank で 10,000）。更新数は発話の長さで変わる。`max_epoch 35` | yaml、`abs_task.py` | 【R】【A】 |
| optimizer | `Adam(lr 1e-3, eps 1e-8)`、`weight_decay 0` | HF config | 【R】 |
| scheduler | `ReduceLROnPlateau(mode min, factor 0.7, patience 1)`、`val_scheduler_criterion: valid loss` | HF config | 【R】 |
| 勾配クリップ | YAML に無く、`--grad_clip` の既定 **5.0**。非有限勾配はその更新を飛ばす | `trainer.py` | 【A】 |
| 初期化 | `xavier_uniform`。**独自 LayerNorm の gamma/beta まで乱数化する** | `initialize.py` | 【A】 |
| 損失 | `mr_l1_tfd`（窓 256/512/768/1024、`time_domain_weight 0.5`、`reduction sum`、`normalize_variance` 無し）×1.0 ＋ `si_snr` ×**0.0**。学習に効くのは `mr_l1_tfd` のみ | yaml | 【R】 |
| early stopping | `patience 5`（`valid loss` 最小基準）、`max_epoch 35` | HF config | 【R】 |
| best 選択 | `keep_nbest_models 5`、`best_model_criterion: [valid si_snr max], [valid loss min]`。**`valid si_snr` は無効**（stats のキーは `si_snr_loss`）で、実効は `valid.loss` のみ。終了時に `valid.loss.ave_5best.pth` も作る。**推論に使う `valid.loss.best.pth`** | `[ESP@0d136c5 : espnet2/enh/loss/criterions/time_domain.py : L255 `si_snr_loss`]`、HF config | 【R】 |
| validation | `cv_ms_snsd` をチャンク化（3 秒、shift 1.5 秒、shuffle 無し）、rank あたり batch 2、損失のみ。PESQ/STOI は計算しない | | 【A】 |
| 評価 | `enh_scoring.py`（SI-SNR、SDR、STOI。PESQ は既定で無効）。recipe README: `tt_ms_snsd` STOI 96.01 / SDR 17.84 / SI_SNR 17.52 | recipe README | 【A】 |
| 出荷 checkpoint との関係 | HF の config が学習時の実効設定。学習に使った**生成データは未シードで再現不能** | | 【R】 |
| UNKNOWN | 生成データの実体、実学習の epoch 数、MS-SNSD の版 | | |
| 採用理由 | recipe は 2026-02-18 以後不変で、HF の config と一致し、実行可能な優先順位 1・2 の資料だから | | |

### 3.6 TFL: TF-Locoformer

詳細: [tf_locoformer.md](tf_locoformer.md)。

| 項目 | 確定内容 | 根拠 | 確度 |
|---|---|---|---|
| エントリ | `egs2/dns_ins20/enh1` に TFL の `espnet2/enh/separator/tflocoformer_separator.py` を**注入**して `run.sh`（`--ngpu 2` の既定に対し、保存 config は `ngpu 1`、`dist_world_size 4`、`multiprocessing_distributed true`） | README、保存 `config.yaml` | 【R】 |
| variant | Medium（B=6、C=384、K=4、H=4、G=4、15.0 M）、`train_enh_tflocoformer.yaml`（RoPE あり）。`-nope` は採らない | README、保存 config | 【R】 |
| データ方式 | **事前生成（固定ペア）**。生成は 1 回で、学習中に再サンプリングするものは無い | 保存 config（`dynamic_mixing false`、`rir_scp null`、`noise_scp null`） | 【R】 |
| 素材 | DNS-Challenge Interspeech 2020（`interspeech2020/master`）。手元 `data/corpora/DNS2020`: clean 65,348 本（31.0 秒、16 kHz、約 563 h）、noise 65,302 本（1〜10 秒、約 180 h）。RIR は学習に使わない | 先行監査 §3.1 | 【A】 |
| 生成条件 | `dns_create_mixture.sh` を README の `sed` で `total_hours=3000`、`snr_lower=-5`、`snr_upper=15` に書き換える。30 秒の混合 360,000 本。SNR は `np.random.randint(-5, 15)` = **整数 −5…14**（上限除外）。最終レベルは `randint(-35, -16)` dBFS。16 bit。noise は 0.2 秒の無音を挟んで連結 | `[ESP90 : egs2/dns_ins20/enh1/local/dns_create_mixture.sh : L18-20]`、README、`noisyspeech_synthesizer_multiprocessing.py` | 【R】（スクリプトの既定値と書き換え）【A】（生成の詳細） |
| 生成の再現性 | 共有カウンタと `Pool` の fork により、**どの混合がどの雑音を取るかは再実行で再現しない**。乱数は import 時に 1 回だけ seed | | 【A】 |
| train / valid | uttid（`reader_<ID>_fileid_<N>`）で `sort`、**先頭 9/10 を `tr_synthetic`、残りを `cv_synthetic`**（324,000 / 36,000）。clean の話者はほぼ分離、noise は共有 | `[ESP90 : egs2/dns_ins20/enh1/local/dns_data_prep.sh : L59-72]` | 【R】 |
| 学習時の valid | 保存 config は **`cv_synthetic_small`**。作成スクリプトも件数も公式に無い（UNKNOWN）。README は「300 h の検証は遅いので縮小を勧める」 | 保存 config L111-116、README L218 | 【R】 |
| test | `tt_synthetic_no_reverb`（論文の non-blind anechoic に当たる）/ `with_reverb`。各 150 本・10 秒 | | 【A】 |
| チャンク | `chunk_length 64000`（**4 秒**）、shift 0.5、offset は 30 秒の入力で常に 0（14 チャンク）。1 発話を全チャンク化 | 保存 config、`chunk_iter_factory.py` | 【R】 |
| batch・DDP | `batch_size 4`（global）= **4 rank × 1**。シャード `[rank::4]`。`accum_grad 1` | 保存 config | 【R】 |
| epoch の定義 | `num_iters_per_epoch 1000` は rank あたり 1000 発話（4 rank で 4000）、約 14,000 更新 / epoch。論文は 1 epoch = 21,000 step | 保存 config | 【R】【A】 |
| optimizer | `AdamW(lr 1e-3, weight_decay 1e-2)`、betas 既定 | 保存 config | 【R】【A】 |
| scheduler | `warmupreducelronplateau`（`warmup_steps 4000`、`mode min`、`factor 0.5`、`patience 3`） | 保存 config | 【R】 |
| 勾配クリップ | `grad_clip 5.0` | 保存 config | 【R】 |
| 初期化 | `xavier_uniform` | 保存 config | 【R】 |
| 損失 | `mr_l1_tfd`（窓 256/512/768/1024、`time_domain_weight 0.5`、`normalize_variance true`）。SI-SNR は重み 0 | 保存 config | 【R】 |
| epoch / early stopping | `max_epoch 150`、`patience 10`。学習曲線は約 108 epoch で終わり、停止理由は UNKNOWN | 保存 config、保存画像 | 【R】【A】 |
| best 選択 | `keep_nbest_models 5`、`best_model_criterion` は `[valid si_snr max], [valid loss min]`（`si_snr` は TF-GridNet と同じ理由で無効）。**評価・公開重みは `valid.loss.ave_5best.pth`**（終了時に上位 5 epoch の平均） | `[ESP90 : espnet2/train/trainer.py : average_nbest_models 呼び出し]`、README | 【R】 |
| 評価 | `enh_scoring.py` と DNS の test。論文 Table 4 は `dns_ins20`。公開重み: `exp/enh_train_enh_tflocoformer_raw/` と `RESULTS.md` | | 【R】【A】 |
| ESPnet の版 | 公式は ESP90。DNS の学習に効く範囲で ESP `0d136c5` と実効動作が同じ（先行監査 §6.4）。`rotary-embedding-torch` は公式 0.6.1、audio-db 0.8.3（差は fp32 で無いと推測、未検証） | 先行監査 | 【A】 |
| UNKNOWN | `cv_synthetic_small`、実学習の所要時間・停止理由、素材の版（librosa）、DNS 学習 clean と test clean の話者重複 | | |
| 採用理由 | 保存 config と生成スクリプトが優先順位 1・2 の資料として揃い、生成・学習・評価が一貫して読めるから | | |

---

## 4. 原論文との差異

論文の記述は参考であり、再現対象は §3 の公式実効動作である。

| モデル | 項目 | 論文 | 公式の実効動作（採用） | 影響 |
|---|---|---|---|---|
| MP-SENet | 学習長 | Interspeech 版 100 epoch（lr は 30 epoch ごとに半減）、journal v1 120 epoch、v2 500k step | `--training_epochs 400`（1,157,200 step）、`ExponentialLR 0.99` | 学習長が 3〜4 倍違う。論文の値で学習すると別プロトコル |
| MP-SENet | test 集合 | 872 発話（Interspeech 版の記述） | `test.txt` は 824 行 | 母数が違い、PESQ の値が変わる |
| MP-SENet | weight decay | v2 は 0.01 | 実効 0.01（AdamW 既定） | 一致 |
| MP-SENet | `batch_first` | 記述なし | `False` のバグ。README が自認 | バッチの構成が出力に効く |
| MossFormerGAN | 論文 | **専用の論文は無い**。ツールキット論文（arXiv:2506.19398）が 8000 h・SNR 0〜15 dB・clean:残響 7:3 を記述 | 公開コードは VB の固定ペア 11,572 | 配布 checkpoint と公開コードは別のレシピ |
| MossFormerGAN | VB test の SNR | 論文 Table: −7.5〜7.5 dB | ローカル VB は 2.5〜17.5 dB | 論文の見出しがデータと合わない |
| DFN3 | 学習条件 | Interspeech 2023 の 4 ページ論文で、条件はほぼ記述なし | 出荷 ini が実効条件。損失は MSE、DFN2 論文の値とは別 | 論文値で再現しない |
| DFN3 | 学習データ | DNS4 ＋ PTDB/VCTK ×10 | 件数・リストは非公開 | 素材の再現は不可 |
| SEMamba++ | 切り出し長 | 24,000（1.5 秒） | 48,000（3 秒） | 窓が 2 倍 |
| SEMamba++ | SNR | −10〜20 dB | HEAD は −5〜20 dB（6 値）。`config_orig.yaml` は −10〜20 | 範囲が違う |
| SEMamba++ | 劣化確率 | 記述なし | HEAD 0.5、orig 0.7 | 論文値は不明 |
| SEMamba++ | magnitude 損失 | L1 | MSE | 損失の形が違う |
| SEMamba++ | 重み | バグ入り（第 3 解像度の Channel FFN 不使用） | 公開重みは修正後の再学習 | 論文の数値は公開重みと別 |
| SEMamba++ | 学習データ量 | 44 時間（Table 3）。旧 README は VCTK ＋ LibriTTS 約 500 h | 公開重みのデータは不明 | UNKNOWN |
| TF-GridNet | 実験 | SMS-WSJ、WHAMR! 等。窓 32 ms・hop 8 ms・平方根 Hann、B=4 | `ms_snsd`: 窓 32 ms・hop 16 ms・周期 Hann、n_layers 4、emb_dim 32、2.55 M | 論文と別モデル・別データ |
| TF-GridNet | 学習 | 4 秒 1 区間 / 混合、clip 1.0、3 epoch で lr 半減 | 3 秒チャンク、clip 5.0（既定）、ReduceLROnPlateau | 別プロトコル |
| TF-Locoformer | epoch | 1 epoch = 21,000 step、3 epoch 改善無しで lr 半減 | 約 14,000 更新、`patience 3`（4 回連続の悪化で半減） | step 数が違う |
| TF-Locoformer | 構造 | Macaron の 1/2 係数 | 1/2 係数は無い | 構造が違う |
| TF-Locoformer | SNR | 「−5〜15 dB」と読める | 整数 −5…14 | 上限が 1 dB 低い |
| TF-Locoformer | 検証 | 300 h | `cv_synthetic_small`（縮小） | 検証の規模が違う |

---

## 5. 公式プロトコルへの準拠範囲

「必須」はこれを外すと別プロトコルになる条件、「許容できる逸脱」は理由と影響を記録した上で認める条件、「未確定」は一次資料で決められない条件、「再現不能」は資料か素材が無い条件である。

### 5.1 MP-SENet

| 区分 | 内容 |
|---|---|
| 必須 | 固定ペア VB-DEMAND 28 話者 11,572 / noisy RMS 正規化 / crop 32,000（一様、両端含む、末尾 0 埋め）/ global batch 4 / `AdamW(5e-4, (0.8, 0.99), wd 0.01)` / `ExponentialLR 0.99` を epoch ごと / 損失の重みと D→G の更新順 / PESQ 目標 / `epoch >= 40` で PESQ 最大の G を保存 / `batch_first=False` の構造 |
| 許容できる逸脱 | 乱数列（seed 1234 でも bit 一致は不要）。データ読み込みの並列化。`librosa.load` の resample 実装差（素材が 16 kHz なら無関係）。影響は小さい |
| 許容できる逸脱（要記録） | 1 GPU の固定順を毎 epoch 再 shuffle にすること（影響は小さいが、公式の順序固定からは外れる） |
| 未確定 | 実学習の epoch 数と GPU 数 / `g_best` が論文の報告値か |
| 再現不能 | `g_best_dns` の学習 |

### 5.2 MossFormerGAN

| 区分 | 内容 |
|---|---|
| 必須 | 固定ペア 11,572 / noisy・clean の独立 `audio_norm` / 2 秒 crop（短ければ先頭貼り付け・末尾 0 埋め）/ `c` の掛け方 / G `AdamW(5e-4, wd 0.01)`、D `AdamW(1e-3, wd 0.01)` / 損失 `0.1/0.9/0.2/0.05` と `time_loss` の実部＋虚部 / 勾配クリップ 10 を micro-batch ごと / 更新順 / plateau 規則（悪化値の上書き、5 で半減、10 で停止、best の上書き）/ 検証は 2 秒 crop の全損失 / D は train モード / scratch で開始（D-04） |
| 許容できる逸脱 | worker の乱数列。PESQ 計算の並列化（`joblib`）。影響は小さい |
| 未確定 | 素材の 16 kHz 変換法 / 実学習の GPU 数 |
| 再現不能 | 配布 checkpoint の学習 / README の PESQ 3.47 |

### 5.3 DeepFilterNet3

| 区分 | 内容 |
|---|---|
| 必須 | オンライン混合の分布（SNR 7 値 [−100…40]、gain [−6, 0, 6]、雑音 2〜5 本、合成雑音 5 %、`p_reverb 0.1`、3 秒窓を連結で埋める）/ batch schedule（16→24→32→64）/ `AdamW(1e-3, wd 1e-12→0.01, amsgrad)` と step ごとの cosine / 損失（MSE 形）/ クリップ 1.0 / 検証は seed 固定の 3 秒窓 / 120 epoch、最終 best |
| 許容できる逸脱 | RNG の bit 一致（Rust `Xoshiro` の再現は不要。D-09）/ `LocalSnrTarget` の範囲（`v0.5.0` ±5、HEAD ±1）/ ERB の `min_nb_erb_freqs`（`v0.5.0` 1、HEAD 2）。**どちらを使ったかを記録する**。素材の代替（D-06）|
| 未確定 | `dataset.cfg`、件数、RIR の集合、実学習の GPU 数 |
| 再現不能 | 公式の学習データ（DNS4 ＋ PTDB の実体）の同一性 |

### 5.4 SEMamba++

| 区分 | 内容 |
|---|---|
| 必須 | GSR 劣化の分布（noise 1.0、reverb/bandwidth/clipping 0.5、順序 reverb→noise→clipping→bandwidth、SNR {−5…20}）/ 窓 48,000 / 損失 / AdamW wd 0.01 / `ExponentialLR` / 修正後のモデル構造 |
| 許容できる逸脱（補修） | 起動に必要な修復（config 名、import、マニフェストの作成）。**アルゴリズムを変えない範囲に限る**。補修の内容を全て記録する |
| 許容できる逸脱 | 学習素材の代替（D-05）。RNG の bit 一致 |
| 未確定 | 学習データの件数・話者、VCTK-GSR の分割、検証データの作り方、再学習の実データ |
| 再現不能 | 論文の重み（バグ入り）/ 公式の学習マニフェスト |

2026-11-07 追記: オンライン劣化のプロトコルは代替素材で実装した（`semambapp.gsr.substitute@1`、`semambapp.gsr.smoke@1`）。公式素材の Dataset `semambapp.gsr.official@1` は blocked。公式の学習結果は未達。検証は [P9_VERIFICATION](P9_VERIFICATION.md)。上の必須・未確定の表は採用時の記録のまま。

### 5.5 TF-GridNet（ESPnet recipe）

| 区分 | 内容 |
|---|---|
| 必須 | v1 `tfgridnet`（n_layers 4、emb_dim 32、2.55 M）/ チャンク 48,000・shift 0.5・`L ≤ W` を捨てる / 元発話を単位とする epoch（rank あたり 5,000）/ 2 rank 相当の global batch 4 / `Adam(1e-3)` / `ReduceLROnPlateau(0.7, 1)` / クリップ 5.0 / `mr_l1_tfd` / `xavier_uniform`（独自 LayerNorm の乱数化を含む）/ `patience 5`、`max_epoch 35` / `valid.loss.best` で推論 |
| 許容できる逸脱 | 生成の seed 固定（公式は未シードなので、固定することは「公式の統計的な再現」への許容）。MS-SNSD の実 SNR が名目の半分になる癖は**再現する**（D-02）。DDP の代わりに 1 プロセスで 2 rank 分のシャードを模す場合は、その旨を記録する |
| 未確定 | 公式モデルの生成データの実体、実学習の epoch 数 |
| 再現不能 | 公式の生成データの同一物 |

### 5.6 TF-Locoformer

| 区分 | 内容 |
|---|---|
| 必須 | DNS 2020 合成 3000 h（整数 SNR −5…14、レベル −35…−16 dBFS、30 秒、16 bit）/ 9:1 の分割（clean の話者がほぼ分離）/ チャンク 64,000（4 秒）・shift 0.5・offset 0 / rank あたり 1000 発話の epoch / `AdamW(1e-3, wd 1e-2)` / `warmupreducelronplateau` / クリップ 5.0 / `mr_l1_tfd`（`normalize_variance true`）/ `xavier_uniform` / `patience 10`、`max_epoch 150` / **`valid.loss.ave_5best`** |
| 許容できる逸脱 | 生成の seed 固定と並列化の順序。ESPnet の版（ESP90 → `0d136c5`、実効動作は同じ）。`rotary-embedding-torch` の版（0.6.1 → 0.8.3、要検証）。4 rank DDP を別の並列度で模すこと（global batch 4 を保つ） |
| 未確定 | `cv_synthetic_small` の大きさと作り方 / 停止理由 / 所要時間 / DNS 学習 clean と test clean の話者重複 |
| 再現不能 | 公式が生成した 3000 h の同一物（乱数・スケジューリング依存） |

---

## 6. 実装前の未決定事項

各項目に、理由、選択肢と長所・短所、推奨と根拠を書いた。**D-01〜D-09 は 2026-11-06 に採用した。各項の末尾の「採用結果」が決定で、本文（理由・選択肢・推奨・根拠）は採用前の記録である。** D-10 は決定でなく前提。

### D-01 検証集合にテスト集合を使うか

対象: MP-SENet、MossFormerGAN。

**理由**: 両方の公式コードが、公式 test（824 発話）を validation と best 選択に使う【R】。audio-db は、テスト集合を学習・検証・モデル選択に使うことを禁じている（既存規則で、今回も変えない）。公式に忠実な再現と、リークを避ける規則が衝突する。

| 案 | 内容 | 長所 | 短所 |
|---|---|---|---|
| A: 公式忠実（Official-faithful） | validation と best 選択に公式 test 824 を使う | 公式プロトコルと完全に同じ。公式の報告値と比較できる | test を選択に使うため、**選択バイアスを含む**。audio-db のテスト使用禁止に例外が要る。結果を汎化性能として扱えない |
| B: リーク回避（Leakage-safe） | 28 話者の train を分け、validation 集合（例: p286/p287 の 872 件）で選ぶ。test は評価のみ | 既存規則と整合。評価が汚れない | 公式とは別プロトコル。学習集合が 10,700 に減り、公式の母数（11,572）と違う |
| A+B 併走 | 同じコードで検証集合をパラメータ化し、A と B の両方を走らせる | プロトコル忠実度と公平比較を両方得る。差から選択バイアスの大きさも測れる | 計算資源が 2 倍。A の実行に例外承認が要る |

**推奨**: **A+B 併走**。A を「公式再現 profile」に限定し、`reproduction` 用の DB・実行にだけ許し（既存の評価用 DB や規則は変えない）、結果に「テストを選択に使った」旗を付けて、汎化性能として報告しない。B を比較用の主 profile にする。

**根拠**: GitHub 優先の原則（P-2）に従えば A が公式だが、A だけでは audio-db の中核規則に反する。両方持てば、規則を変えずに公式との距離を測れる。**どれを採るかはユーザーの判断で、今回は決めていない。**

**採用結果（2026-11-06）**: **A+B 併走を採用**。A（公式忠実）は再現用 DB・プロトコルの中だけで、test を validation・選択に使う。`selection_uses_test = true` を記録し、独立した汎化性能として報告しない。B（leakage-safe）は独立な分割。評価用 DB と「test を学習・検証に使わない」既存規則は変えない。未実装。

### D-02 公式の欠陥・癖の扱い

対象: 全モデル（MP-SENet の `batch_first`、MossFormerGAN の `time_loss`・plateau の上書き・独立 `audio_norm`、TF-GridNet/TF-Locoformer の無効な `valid si_snr`、MS-SNSD の SNR 半減、SEMamba++ の起動不能、など）。

| 案 | 長所 | 短所 |
|---|---|---|
| 再現する（公式の実効動作） | 優先順位 1 に忠実。差異の原因を切り分けやすい | 欠陥が学習を損なう場合は、成果が公式の限界に縛られる |
| 直す | 良いモデルを作れる | 別プロトコルになり、公式との比較が曖昧になる |

**推奨**: **再現する**。欠陥は §3 に列挙済みで、「公式の欠陥」として識別できるようにし、直した版は別 profile に分ける。**ただし SEMamba++ の起動不能は補修が必須**で、アルゴリズムに触れない範囲に限り、全て記録する。

**根拠**: P-2。補修と欠陥修正を混ぜると、後から公式との差を説明できなくなる。

**採用結果（2026-11-06）**: **公式の実効動作を欠陥も含めて再現する**。差は `official_effective` / `minimal_launch_repair` / `algorithm_change` / `research_improvement` に分け、直した版は別 Protocol にする。SEMamba++ の補修は許すが**今は実装しない**。未実装。

### D-03 MP-SENet の学習長と GPU 構成

**理由**: 公式コードの既定は 400 epoch（1,157,200 step）だが、論文は 100/120 epoch または 500k step【R】。実学習の epoch 数は UNKNOWN。README の例は 2 GPU で、`DistributedSampler` の経路（global batch 4 のまま）を使う。1 GPU では順序が固定される。

| 案 | 内容 | 長所 | 短所 |
|---|---|---|---|
| 400 epoch | コード既定 | GitHub 優先に忠実 | 先行見積もりで 1 seed 約 190 時間と大きい |
| 120 epoch | journal v1 | 計算が小さい | 公式コードの既定と食い違う |
| 段階（まず 120 でスモーク、承認後に 400） | | 途中で品質を確認できる | 2 段の承認が要る |

**推奨**: **400 epoch を再現対象とし、実行は段階化**する。GPU 構成は README の 2 GPU（global batch 4）を基準にし、1 プロセスで模す場合は逸脱として記録する。

**根拠**: GitHub 優先。400 epoch の実行は 30 分を超えるので、投入前に明示の承認が必要（AGENTS.md）。

**採用結果（2026-11-06）**: **400 epoch、README の 2 GPU・global batch 4 を再現対象**とし、段階的に実行する。実行の承認は含まない。1 プロセスで模す場合は逸脱として記録する。未実装。

### D-04 MossFormerGAN の開始点

**理由**: `train.sh` の既定は、配布 checkpoint があれば finetune として始まる【R】。公開 `train.py` の scratch 学習（`init_checkpoint_path='None'`）を採るか。

**推奨**: **scratch**。公式 YAML の `init_learning_rate` と整合し、「公開コードのプロトコル」の定義に合う。finetune は別の実験とする。

**根拠**: finetune の既定は `MossFormer2_SE_48K`（別ネットワーク）向けで、MossFormerGAN の学習プロトコルではない。

**採用結果（2026-11-06）**: **scratch**。結果に開始条件を明記する（公式では `init_checkpoint_path='None'` という文字列。`train.sh` の既定は finetune になる罠）。未実装。

### D-05 SEMamba++ の設定版・素材・補修範囲

| 項目 | 選択肢 | 推奨 |
|---|---|---|
| 設定版 | HEAD の `config.yaml`（公開重みに対応）／ `config_orig.yaml`（論文寄り） | **`config.yaml`**。論文値は参考（§4） |
| 素材 | (i) 手元の VCTK 0.80・WHAM48・DNS2020 noise・DNS5 RIR で代替／(ii) VCTK 0.92 と Arni RIR を入手して公式に近づける | **(i) を先に**（「素材代替・プロトコル忠実」と明記）。(ii) は入手可否をユーザーに確認 |
| 補修 | 起動に必要な最小限（config 名、import、マニフェスト）／それ以上 | **最小限のみ**。算法の変更は含めない |

**根拠**: VCTK は 0.80 のみで、p315 が無く p280 の扱いも違う可能性がある【A】。マニフェストの件数は UNKNOWN のため、素材を決めるまで総 step も決まらない。

**採用結果（2026-11-06）**: 公式 GitHub の固定 commit と HEAD の `config.yaml` を再現対象とする。代替素材は許すが、**公式素材の再現とは分け**、「公式プロトコル準拠・代替素材」と呼ぶ（「公式データセットの再現」と呼ばない）。最小補修は**許すが今は実装しない**。素材（VCTK 0.92、Arni RIR）は入手するまで `blocked` / `unknown`。未実装。

### D-06 DFN3 の素材と `dataset.cfg`

| 案 | 内容 | 長所 | 短所 |
|---|---|---|---|
| DNS4 ＋ PTDB（＋ VCTK ×10）を入手 | 公式の手順（`download_process_dns4.sh`）で HDF5 化し、`dataset.cfg` は論文の記述から自分で定義 | 素材が最も公式に近い | 規模が大きく、`dataset.cfg` の件数は推定になる |
| 手元の素材で代替 | VoiceBank・LibriTTS・DNS2020/DNS5 の noise・RIR などで混合分布を再現 | すぐ始められる | 分布は公式コードに従うが、素材は別物 |

**推奨**: **まず代替素材で、混合の分布（SNR、gain、雑音本数、残響、窓）が公式コードと一致することを統計で確認**し、DNS4 の入手はユーザーの判断に委ねる。

**根拠**: 分布はコードから確定できる【R】が、素材の実体は非公開。DNS5 には clean 音声が無い【A】。

**採用結果（2026-11-06）**: **公式素材を優先**し、`dataset.cfg`・DNS4・PTDB・RIR 集合が無い間は UNKNOWN / BLOCKED とする。代替素材は別プロトコル準拠として分け、「出荷モデルの学習データの再現」と呼ばない。代替素材の混合は [P8_VERIFICATION](P8_VERIFICATION.md)。公式素材の `dfn3.online.official@1` は blocked のまま。

### D-07 TF-GridNet の範囲宣言と MS-SNSD の生成

**理由**: 再現対象は ESPnet recipe であって論文ではない。生成は未シードで同一物を再現できない。

**推奨**: **範囲宣言（§3.5）を採用**し、MS-SNSD（手元に在る）から、公式の生成式（名目 SNR [0, 10, 20, 30, 40]、実効は半分、−25 dBFS）で生成して**シードを固定して保存**する。30 h / 1 h の規模、cv は train の 20 %、は README に従う。

**根拠**: 統計的に同等な生成が得られ、再実行できる。seed 固定は公式からの逸脱として記録する。

**採用結果（2026-11-06）**: 範囲宣言を採用し、「**ESPnet 公開 SE レシピの再現**」と呼ぶ（論文の実験の再現ではない）。seed・生成条件・出力マニフェストを記録する。追加する決定性は文書化した差。未実装。

### D-08 TF-Locoformer の 3000 h 生成と `cv_synthetic_small`

**理由**: 公式は 3000 h（30 秒 × 360,000 本、noisy・clean・noise の 3 本、16 bit）。1 本あたり 345.6 GB（3000 h × 16 kHz × 2 B）なので、3 本で約 **1.04 TB**【導出】。生成の所要時間は UNKNOWN。`cv_synthetic_small` は公式に無い。

| 案 | 内容 | 長所 | 短所 |
|---|---|---|---|
| 3000 h（公式規模） | 公式どおり | 公式に最も近い | 約 1 TB、長時間、承認が要る |
| 縮小（例: 300 h） | 規模を落とす | 軽い | 公式プロトコルとは別。論文と数値が比べにくい |
| 段階（縮小でパイプラインを確認 → 3000 h） | | 失敗の被害が小さい | 2 段の承認 |

`cv_synthetic_small` は、`cv_synthetic`（36,000 本）の**固定部分集合**として自分で定義し、その規則（例: 1/10）を記録する。**大きさはユーザーの判断**。

**推奨**: **段階**。3000 h の生成と学習（約 14,000 更新 / epoch × 最大 150 epoch）は、30 分を超えるジョブなので、投入前にユーザーの明示の承認が要る（AGENTS.md）。優先度は `-p -5`。

**採用結果（2026-11-06）**: **段階実施**（小規模検証 → 公式の 3000 h）。今は生成しない。保存・I/O・時間・再開・失敗時の設計は [DB 設計 §9](../../db/REPRODUCTION_DB_DESIGN.md) にある。30 分以上のジョブは別の承認が要る。**`cv_synthetic_small` は未解決のまま**。未実装。

### D-09 再現の同等性基準

**理由**: オンライン混合（DFN3、SEMamba++）、事前生成（TF-GridNet、TF-Locoformer）では、公式の乱数・スケジューリングまで再現するのは非現実的である。

| 案 | 長所 | 短所 |
|---|---|---|
| 統計的同等（分布・規則の一致を検定） | 現実的。公式の乱数に依存しない | 個々のサンプルの一致は保証しない |
| bit 一致（公式の RNG を再現） | 厳密 | DFN3 は Rust の `Xoshiro` に依存し、TF-Locoformer の生成はスケジューリング依存で、不可能に近い |

**推奨**: **統計的同等**。各サンプルの seed と抽選結果を記録して再実行できるようにする。公式のデータローダー（例: `libdfdata`）は、**テスト時の参照（オラクル）にだけ**使い、実行時の依存にしない（AGENTS.md の公式コード非依存の原則）。

**採用結果（2026-11-06）**: **プロトコルと統計的同等性**（ビット一致ではない）。ただし次は**統計的同等性に任せず厳密に確認する**: 固定の clean / noisy ペア、train / valid / test の分割、サンプリングレート、教師波形の定義、損失、optimizer・scheduler の更新規則、checkpoint 選択の規則。独立実装は公式コードに実行時依存しない。未実装。

### D-10 実施フェーズの承認条件（決定ではなく前提）

30 分以上の TSUBAME ジョブは、そのジョブを含むユーザーの明示の指示か承認が要る。優先度は `-p -5`（指定できないキューはその制約に従う）。GPU は共有資源のため、使用前に `nvidia-smi` で空きを確認する。**今回は何も投入していない。**

---

## 7. 次フェーズへの引き継ぎ（DB・データパイプラインの入力条件）

スキーマや API は固定しない。**設計が満たすべき入力条件**だけを書く（2026-11-06: 設計は [DB 設計](../../db/REPRODUCTION_DB_DESIGN.md)、[Pipeline 設計](DATA_PIPELINE_DESIGN.md) に作成済み。実装は未着手）。

### 7.1 共通の条件

| 条件 | 内容 |
|---|---|
| 方式の 3 分類 | 固定ペア（MP-SENet、MossFormerGAN）、事前生成（TF-GridNet、TF-Locoformer）、オンライン混合（DFN3、SEMamba++）を別の表現で持てること |
| レシピ JSON v1 の限界 | v1 は 1 件の固定ケース（雑音 1 本・SNR 1 値・RIR 1 本）の実現値しか書けず、**確率分布・複数雑音・連結・VAD 基準 SNR・確率的選択を表せない**（[RECIPE.md](../../db/RECIPE.md)、先行監査）。オンライン混合は、レシピとは別の層（プロトコル）で持つ必要がある |
| ケース層とプロトコル層の分離 | 「1 回の抽選結果（case）」と「抽選分布（protocol）」を分けて保存する |
| 評価用 DB との分離 | 学習用の再現 DB は、評価用の `resource.duckdb` と別に置く案を推奨（例: `data/db/reproduction.duckdb`）。**検証にテストを使う profile（D-01 の A）を評価用 DB へ混ぜない** |
| 現状 | 2026-11-06 の追補: `data/db/resource.duckdb` が存在する（`corpora.audio_files` 1,375,481 行、`eval` スキーマ無し。読み取り専用で確認、変更していない）。`data/staging/` は空。既存の DB を作り直さない（AGENTS.md）。上の「確認」時点（`data/db/` が空）とは状態が違う |
| 再生成の保存先 | 事前生成データは DuckDB へ書く前に `data/staging/` へ保存する（AGENTS.md） |

### 7.2 モデル別の入力条件

| モデル | データ層に必要な情報 |
|---|---|
| MP-SENet | 28 話者の固定ペア。train 11,572 / test 824 のリスト（`training.txt`、`test.txt`）。validation 集合は D-01 次第（A: test、B: 分割）。1 GPU 固定順と DDP の `DistributedSampler` を区別できること。noisy RMS 正規化と crop 規則 |
| MossFormerGAN | 固定ペア 11,572 / 824（16 kHz 変換済み）。validation の 2 秒ランダム crop を**再現可能にする**（検証の乱数 seed）。D-01 次第 |
| DFN3 | オンライン混合の**プロトコル**: `snrs`、`gains`、窓長、`p_reverb` と RIR の再サンプル・減衰・DRR・rt60、雑音本数の範囲、合成雑音の確率と傾き、クリップ、`sampling_factor`。素材の役割（speech / noise / rir）、sample ごとの seed の式。valid は seed 固定 |
| SEMamba++ | 劣化の確率・順序・SNR 集合・VAD 基準、RIR のチャンネル平均と遅延補正の有無、雑音の wrap 配置。素材の版（VCTK 0.80 / 0.92）と話者除外（p280、p315）。マニフェストの内容ハッシュ |
| TF-GridNet | MS-SNSD の素材（clean / noise）、生成の seed、train 30 h・test 1 h、cv = train の 20 %。2 rank の epoch の定義（元発話 5,000 / rank）を扱えること |
| TF-Locoformer | DNS 2020 の素材 65,348 / 65,302、生成の seed と各混合の抽選結果（clean・noise・SNR・レベル・切り出し位置）、9:1 の分割規則、`cv_synthetic_small` の規則。4 rank の epoch（1000 発話 / rank）と 64,000 チャンクの計画 |

### 7.3 学習実装側に残る未決定（設計ではなく実装時に決めること）

| 項目 | 内容 |
|---|---|
| 仮想 rank | 2 / 4 rank を 1 プロセスで模すときの、シャード・cache・epoch の扱い |
| 検証の crop seed | MossFormerGAN と DFN3 の検証を再現可能にする seed の持ち方 |
| 補修の記録 | SEMamba++ の補修を、公式との差分として保存すること |
| 評価 | PESQ は論文が NB、audio-db が wb など、指標の帯域が違う。直接比較しない |

---

## 8. 既存ドキュメントの訂正事項（記録のみ。今回は修正していない）

先行監査 [OVERVIEW §11](OVERVIEW.md) の内容に、今回の確認で追加・解消したものを足す。

| 区分 | 内容 |
|---|---|
| 解消 | DFN3 の `gains` の UNKNOWN（先行監査の `deepfilternet3.md` 表）は、`v0.5.0` の `dataset.rs` L638 で既定 `[-6, 0, 6]` が既に在ることから**解消**した |
| 修正が要る | [OVERVIEW](OVERVIEW.md) §2 の表に「TF-Locoformer: 2 秒チャンク（audio-db は 32,000。論文系は 64,000）」とあるが、**64,000 サンプルは 16 kHz で 4 秒**（audio-db の 32,000 が 2 秒）。同じ OVERVIEW 内の別節と `tf_locoformer.md` は「4 秒」と正しく書いている |
| 修正が要る | `docs/se/MODELS.md` の TF-Locoformer の arXiv ID `2409.16674` は誤り。正しくは `2408.03440`（先行監査） |
| 修正が要る | `.agents/DECISIONS.md` #26 などの TF-GridNet の best 基準は、`valid si_snr max` が有効と読める。ピン留め ESPnet では stats のキーが `si_snr_loss` のため**無効**で、実効は `valid.loss` のみ【R】。#31（TF-Locoformer）は正しく、矛盾している |
| 修正が要る | `OFFICIAL_TRAINING_AUDIT.md` は、TF-GridNet を単一プロセス想定と書くが、公式は 2 rank の DDP【R】。MP-SENet の「GO」の根拠に、母数（11,572 と 10,700）、best 選択集合、公式の再開動作の違いが書かれていない |
| 修正が要る | `docs/se/VALIDATION.md` の MP-SENet の commit（`d91b88c2…`）は、ピン `89932cf` と合わない |
| 存在しない参照 | `configs/training/*_seed17.yaml`、`configs/training/trial100/`、`configs/evaluation/`、`.venvs/lab`、`data/db/*.duckdb`（`resource.duckdb` を含む）。作業ツリーに無い。**2026-11-06 の追補**: `data/db/resource.duckdb` はその後に存在を確認した（`eval` スキーマは無い）。`.venvs/semamba`（`DEVELOPMENT.md` が DuckDB 1.4.3 と書く）も無く、現在は `.venvs/semambapp`（DuckDB 2.0.0）。`.venvs/deepfilternet` は DuckDB を持たない |
| URL の不統一 | SEMamba++ の URL が、`requirements/official-assets.toml`（旧名）と `src/se/models/semambapp/NOTICE`（新名）で混在 |

---

## 9. この調査で確認した範囲と限界

- **確認したこと**: 6 つの公式 repo の checkout を読み、ピン留め commit が upstream の HEAD と一致することを確認した（ESPnet を除く）。重要な実効値（optimizer の既定、チャンクの式、plateau の規則、best 基準の無効なキー、DFN3 の版の差、SEMamba++ の config の履歴と起動不能箇所、ESPnet の版の差）を、行番号付きで読んだ。
- **確認していないこと**: GPU 学習・評価の実行、公式コードの実行、数値の一致検証、DuckDB の読み書き、データ生成。【A】の項目は先行監査の記述のままで、今回は一次資料を読み直していない（§1.5）。
- **変更したファイル**: この `TARGETS.md` と、利用者の依頼で書いた `OUTPUT.md`（最終報告）だけ（作成時点）。2026-11-06 の採用の記録では、`TARGETS.md`、`.agents/DECISIONS.md`、新規の設計書 3 本、`docs/INDEX.md`、`OUTPUT.md` を更新した。`src/se/official/`、既存の監査文書、設定、DB、checkpoint、結果は変更していない。
- **`.agents/` の更新**: 作成時点では不要と判断した。D-01〜D-09 の採用に伴い、2026-11-06 に `DECISIONS.md` の「採用決定済み・未実装」へ移した。`AGENTS.md`、`DATA.md`、`PROJECT.md`、`DEVELOPMENT.md` は、実装した時点で更新する（[実装計画 §4](IMPLEMENTATION_PLAN.md)）。
