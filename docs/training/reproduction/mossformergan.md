# MossFormerGAN 公式再現プロトコル監査

この文書の監査本文は調査記録である。P2 で、公開学習コードの VoiceBank 固定ペアを `reproduction.duckdb` の `mossformergan.vb28.official@1` と `mossformergan.vb28.safe@1` として登録した。配布 checkpoint の学習データ再現ではない。16 kHz 化は soxr HQ で、公式 scp の事前変換器とは別である。学習 Pipeline への接続はまだ無い。

表記:

- `CVS@6b3774d` は `modelscope/ClearerVoice-Studio` の commit `6b3774dc79c46ae8bed2a4fa5f706f0ac8c75c61`。audio-db が `src/se/official/clearervoice-studio` に固定したもので、upstream `main` の HEAD と同じである。
- `T/` は `train/speech_enhancement/` の略。
- `adb@3bee9ad` は audio-db（ブランチ `feature/reproduction`）の作業ツリー。
- 根拠の書式は `[repo@commit : ファイル : 行 : 関数/キー]`。論文は `[論文 arXiv:XXXX vN §節/表]`。
- 確度ラベルは 【確認:コード】【確認:論文】【確認:設定】【推測】【UNKNOWN】。

## 0. 要約(監査結論、主な差異TOP、分類: online/pre-generated/fixed pair)

### 0.1 結論

1. **MossFormerGAN を主題とする論文は見つからなかった。**
   - 公式 README は `MossFormerGAN_SE_16K` を「MossFormer2 backbone + GAN」と書き、論文として MossFormer2（arXiv:2312.11825、時間領域の話者分離）を挙げる。しかし MossFormer2 の本文に SE も GAN も出てこない（"MossFormerGAN" の出現数 0）。
   - 実装は CMGAN（arXiv:2203.15149）と TF-GridNet（arXiv:2209.03952）に着想を得たと README に書かれている。
   - このモデルの学習データ・損失・最適化を文章で述べているのは、ツールキット論文 ClearerVoice-Studio（arXiv:2506.19398 v1、Interspeech 2025）§2.2.3、§3.1.1、§3.1.2 だけである。
   - 論文を推測で割り当てない。【確認:コード】【確認:論文】
2. **公開された学習コード（`T/`）のプロトコルは「固定ペア（事前生成・オフライン）」である。**
   - 学習は noisy/target の 2 列 scp を読む（`tr_demand_28_spks_16k.scp` 11,572 行、`cv_demand_testset_16k.scp` 824 行）。
   - オンライン混合、SNR のサンプリング、RIR の適用、雑音の選択はコードに無い。
   - データの中身は VoiceBank+DEMAND の 28 話者 train と、公式 test。いずれも作者の NAS 上の事前に 16 kHz へ変換したファイルである。【確認:コード】
3. **配布 checkpoint（`alibabasglab/MossFormerGAN_SE_16K`）は、この公開 recipe の産物ではない。**
   - モデルカードとツールキット論文は「公開データ＋社内データの大規模学習」と書く。
   - 論文の記述は、clean 850 h、noise 300 h、合成 RIR 10 万本、clean:残響 = 7:3、SNR 0〜15 dB の混合、8000 h train / 50 h dev である（§3.1.1）。生成がオンラインかオフラインかは書かれていない。「8000 時間の学習セット」という言い方からオフライン生成と推測する。【推測】
   - checkpoint を読むと、`epoch 7 / step 1,169,381` である。discriminator の optimizer の lr は 0.0005 で、公開コードの `init_lr*2 = 0.001` と食い違う。optimizer の param_group に `initial_lr` が残っていて、公開コードに無い LR scheduler が存在したことを示す。
   - したがって VoiceBank のみから学習して、README の PESQ 3.47 を再現できる根拠はない。数値一致は今回の最優先ではない。【確認:コード】【推測】
4. **分類: 公開コード = 固定ペア（pre-generated）。配布 checkpoint の実データ生成 = UNKNOWN（論文の文章からはオフライン混合だったと推測）。** オンライン混合を再現する対象は無い。
5. **audio-db の学習実装は、損失・更新順序・optimizer の実効値・plateau 規則を公開コードに合わせている。**
   - コード読みで確認した。数値一致の再実行は今回していない。
   - 一方で、プロトコル面（検証、データ分割、正規化、best 選択、評価）に差がある。

### 0.2 主な差異 TOP（詳細は §9）

| 順位 | ID | 内容 | 影響度 |
|---|---|---|---|
| 1 | MFG-D02 | 公式の検証は「2 秒ランダム crop ＋ Generator の全損失（GAN 項と PESQ 込み）」で、plateau を決める。audio-db は全長 1 発話の時間 L1 だけで plateau を決める。 | Critical |
| 2 | MFG-D03 | 公式の `last_best_checkpoint`（配布物と同名）に当たる成果物を audio-db は作らない（`best_generator.pt` は生成されない）。 | Major |
| 3 | MFG-D12 | 評価条件が違う。公式 README は 16 kHz の VB test を SpeechScore で評価する。audio-db は 48 kHz の VB test を 48→16→48→16 kHz と往復して、別実装の指標で評価する。 | Major |
| 4 | MFG-D05 | 公式は noisy と clean を独立に `audio_norm`（−25 dBFS 二段）で正規化する。audio-db は相対ゲインを保つ。実測で clean/noisy のゲイン比がずれる（平均 +0.38 dB、最大 +3.14 dB）。 | Major |
| 5 | MFG-D01 | 公式は 28 話者 11,572 を train、公式 test 824 を validation に使う。audio-db は 10,700 / 872（p286, p287）に分け、test を validation に使えない。 | Major |
| 6 | MFG-D14 | 配布 checkpoint の学習レシピ（大規模データ、D の lr、scheduler）が公開コードと食い違う。VB のみの学習では PESQ 3.47 に届く保証が無い。 | Unknown |
| 7 | MFG-D08 | DDP の分割・実効バッチ・検証との整列が、公式と異なる。 | Major |

### 0.3 この監査で確認していないこと

- GPU での学習・評価・公式コードの実行（禁止されている）。
- audio-db と公式の数値一致（既存 docs の「CPU 4 step 一致」などは未再検証）。
- DDP 2 GPU での公式 solver の動作。

## 1. 調査メタデータ

| 項目 | 内容 |
|---|---|
| 調査日 | 2026-11-06 |
| audio-db | ブランチ `feature/reproduction`、HEAD `3bee9ad`。作業ツリーは clean（`git status --short` が空）。 |
| 公式 repo | `https://github.com/modelscope/ClearerVoice-Studio`。pin は `6b3774dc79c46ae8bed2a4fa5f706f0ac8c75c61`（2025-08-14 "Update README.md"）。`git describe` は `v0.1.2-35-g6b3774d`。 |
| 別クローン | `/tmp/audit-mossformergan/cvs`（上記 repo を `git clone`）。pin は upstream `main` の HEAD と同じ（先行 0 commit）。タグは `v0.1.1`（2025-04-30）と `v0.1.2`（2025-07-11）のみ。 |
| 版差 | `git diff v0.1.1 6b3774d -- train/speech_enhancement` は**空**。学習コードは v0.1.1、v0.1.2、pin で同一。 |
| 学習コードの履歴 | 追加は 2024-11-27 の `f0d52aa`（"update train"）。以降の変更は `89dc5c7`、`4b214ba`（lr 表示）、`7266724`（2024-12-04 "fixed audio mag."）、`9076de3`（2024-12-30 "fix audioread for se"）、`83ba731`（2025-03-27 "fix model check"）。MossFormerGAN の損失・モデル・YAML・`train.py` は `f0d52aa` 以後変わっていない。 |
| HF モデル | `alibabasglab/MossFormerGAN_SE_16K`、repo commit `dcb1b20e1e44d435090072ab9b223c792ea1f624`（lastModified 2024-11-25）。 |
| 一次資料の取得 | HF のモデルカード・API、arXiv API と HTML（`/tmp/audit-mossformergan/`）。 |
| 実測（読み取りのみ） | VoiceBank の train/test の長さと、`audio_norm` による相対ゲインの変化（§3.4）。`.venvs/se` の Python で checkpoint（G は `data/checkpoints/mossformergan/last_best_checkpoint.pt`、D は HF から `/tmp` へ取得）の中身を読んだ。 |
| 調べていないこと | GPU を使う学習・評価、公式コードの実行（`src/se/official/` へ `__pycache__` が出るため）、audio-db のテスト実行、DuckDB の読み書き（`data/db/` が空）。 |

作業ツリーの実在確認（`ls`）:

- `configs/training/` にあるのは `deepfilternet3.yaml`、`mossformergan.yaml`、`mp_senet.yaml`、`semambapp.yaml`、`tf_gridnet.yaml`、`tf_locoformer.yaml` だけ。
- `configs/training/mossformergan_seed17.yaml`、`configs/training/trial100/`、`configs/evaluation/` は**無い**。
- `data/db/` は `.gitkeep` のみ、`data/staging/` は空。`voicebank.duckdb`、`resource.duckdb` が無い。
- `data/checkpoints/mossformergan/last_best_checkpoint.pt` は在り、sha256 が `requirements/official-assets.toml` の `04bf00db575389e1ea8b1fcca48d9efcf073e50ad170e6dd5230df3bac98be27` と一致する。

## 2. A. 公式資料(表。variant選定とその理由、候補差異)

### 2.1 公式資料の表

| 項目 | 内容 | 根拠・確度 |
|---|---|---|
| モデル | `MossFormerGAN_SE_16K`（16 kHz、単一チャネル SE） | 【確認:コード】`[CVS@6b3774d : T/networks.py : L15-18 : network_wrapper]` |
| 原論文 | **専用論文なし**。README は MossFormer2（arXiv:2312.11825、ICASSP 2024）を引くが、本文は話者分離である。 | `[CVS@6b3774d : T/README.md : L10, L21-23]`、`[論文 arXiv:2312.11825 v2]`（"MossFormerGAN" の出現 0）【確認:論文】 |
| 関連論文 | CMGAN（arXiv:2203.15149）。TF-GridNet（arXiv:2209.03952）。README が着想元として挙げる。 | `[CVS@6b3774d : T/README.md : L23]` |
| ツールキット論文 | ClearerVoice-Studio（arXiv:2506.19398 v1、2025-06-24、Interspeech 2025）が唯一の記述。 | `[論文 arXiv:2506.19398 v1 §2.2.3, §3.1.1, §3.1.2, 表1]` |
| 学習コード | `train/speech_enhancement/{train.py, solver.py, losses/loss.py, dataloader/, utils/misc.py, models/mossformer_gan/}` | 【確認:コード】 |
| 設定 | `T/config/train/MossFormerGAN_SE_16K.yaml`。推論設定は `T/config/inference/` と `clearvoice/clearvoice/config/inference/` の 2 系統。 | 【確認:設定】 |
| データローダー | `T/dataloader/dataloader.py`、`T/dataloader/misc.py` | 【確認:コード】 |
| 評価スクリプト | 学習側には評価スクリプトが無い。検証損失だけを計算する。ベンチマークの指標は `speechscore/`（README が明記）。 | `[CVS@6b3774d : clearvoice/README.md : L124]` |
| 配布 checkpoint | HF `alibabasglab/MossFormerGAN_SE_16K`。`last_best_checkpoint.pt` は 38,845,593 B、sha256 `04bf00db…be27`（`x-linked-etag` と一致）。同じ repo に `last_best_checkpoint.disc.pt`（97,588,777 B、sha256 `4a5ebead6e364896c1cd972526edef1dd48937a7b98e702e348d11b2000632f1`）と、1 行の指示ファイル `last_best_checkpoint` がある。 | 【確認:コード】HF API・`x-linked-etag`。audio-db は G のみを `requirements/official-assets.toml` へ登録している。 |
| ライセンス | Apache-2.0（HF カード、repo） | 【確認:設定】 |
| 公開時期の対応 | HF の重みは 2024-11-14 に作られ 2024-11-25 に最終更新。学習コードの公開は 2024-11-27（`f0d52aa`）で、重みの**後**である。 | 【確認:コード】HF API、`git log`。 |

### 2.2 配布 checkpoint の中身（読み取りのみ）

- G: キー `['model','optimizer','epoch','step']`。`model` は 1065 キー、3,128,639 要素（buffer を含む）。`epoch 7`、`step 1,169,381`。
  - optimizer の param_group は lr 0.0005、betas (0.9, 0.999)、eps 1e-8、**weight_decay 0.01**、amsgrad False、**`initial_lr: 0.0005` あり**。state は 1053 エントリ。
- D（`last_best_checkpoint.disc.pt`）: キー `['discriminator','optimizer_disc','epoch','step']`。`discriminator` は 117 キー、8,125,019 要素（buffer を含む）。`epoch 7`、`step 1169381`。
  - `optimizer_disc` の param_group は lr **0.0005**、weight_decay 0.01、`initial_lr: 0.0005` あり。state は 93 エントリ。
- 公開 `train.py` は `lr=args.init_learning_rate*2`（0.001）で D の optimizer を作る。`[CVS@6b3774d : T/train.py : L57 : main]` 【確認:コード】
  - 配布 D の lr は G と同じ 0.0005 なので、**配布 checkpoint は公開 `train.py` とは別の学習コード（社内版）で作られた可能性が高い**。【推測】
  - `initial_lr` は PyTorch の LR scheduler を optimizer に付けたときに入るキーである。公開コードに scheduler は無い。【確認:コード】【推測】
- `step / (epoch+1) ≈ 146,173`。epoch の巻き戻し（plateau 時の再読み込み）が無く、step が micro-batch の単位だとすれば、1 rank・1 epoch あたり約 14.6 万 micro-batch である。【推測】
- 配布 G の key に `module.` 接頭辞は無い。

### 2.3 variant の選定と理由

- `MossFormerGAN_SE_16K` のみが「単一チャネル・単一話者の SE の再現」に当てはまる。同 repo の `FRCRN_SE_16K`（Adam、SI-SNR 系損失）や `MossFormer2_SE_48K`（Adam、PSM）は別モデルで、今回の対象ではない。
- 学習レシピの候補が 2 つある。研究判断として、**決めずに**差異を記す。

| 候補 | データ | 出典 | 再現可能性 |
|---|---|---|---|
| (a) 公開 `train.py` ＋ VoiceBank-DEMAND 28 話者の固定ペア（`T/data/*.scp`） | VB-DEMAND 28 spk 11,572 / test 824 | コード | 可能。ただし配布 checkpoint の数値とは無関係。 |
| (b) 配布 checkpoint の学習（ツールキット論文 §3.1.1） | 8000 h の混合（DNS4、VCTK、社内 TTS、WHAM!48k、合成 RIR 等） | 論文 | 社内データを含み、**公開コードに生成処理が無い**。再現不可。 |

audio-db の現行構成は (a) に近い（VB のみ）。比較する公式側は (a) のコードと、(b) の論文記述を、混同せずに分けて扱う。

## 3. B. 学習データセット

### 3.1 公開コードの学習データ（候補 (a)）

| 項目 | 内容 | 確度 |
|---|---|---|
| データセット | VoiceBank+DEMAND（Valentini et al.、Edinburgh DataShare 10283/2826）の 28 話者 train と公式 test。README は VoiceBank-DEMAND（リンクは `https://datashare.ed.ac.uk/handle/10283/2826`）を推奨する。 | 【確認:コード】`[CVS@6b3774d : T/README.md : L62]` |
| train リスト | `data/tr_demand_28_spks_16k.scp`、**11,572 行**。 | 【確認:設定】`wc -l` |
| validation リスト | `data/cv_demand_testset_16k.scp`、**824 行**。VB の公式 **test** 話者（p232、p257）。README は「testing data list」と呼ぶ。つまり**検証にテストセットを使う**。 | `[CVS@6b3774d : T/config/train/MossFormerGAN_SE_16K.yaml : L14-15]`、`[T/README.md : L62]` 【確認:設定】 |
| test リスト | `tt_list` は YAML に無く、solver は `args.tt_list is not None` のときだけテストを走らせる（既定は走らない）。 | `[CVS@6b3774d : T/solver.py : L204 : Solver.train]` 【確認:コード】 |
| パス | `/mnt/nas/mit_sg/shengkui.zhao/DEMAND_SE_data/data/audio_16k/{28spks,testset}/{noisy,target}/*.wav`。作者の NAS で、**事前に 16 kHz へ変換済み**。変換に使ったツール・方法は不明。 | 【確認:設定】【UNKNOWN】 |
| 形式 | 1 行 2 列（noisy パス、clean パス）。3 列目に duration を書けるが使わない。 | `[CVS@6b3774d : T/dataloader/misc.py : L15-37 : read_and_config_file]` 【確認:コード】 |
| clean/noisy の対応 | 同名ファイルの対応。教師波形 = 対応する clean 波形。 | 【確認:設定】 |
| noise | DEMAND 8 種＋人工 2 種（babble、ssn）。ファイル単位の SNR は付属ログ `log_trainset_28spk.txt` にある（0/5/10/15 dB）。 | 【確認:データ】ローカル VB の `log_trainset_28spk.txt` |
| RIR | 無し。 | 【確認:データ】 |
| サンプリングレート | 16 kHz（scp 側）。元の VB は 48 kHz。 | 【確認:設定】 |
| チャネル | 1。コードは多チャネルなら先頭チャネルだけを使う。 | `[CVS@6b3774d : T/dataloader/dataloader.py : L30-31 : audioread]` |

### 3.2 論文のデータ記述（候補 (b)）

`[論文 arXiv:2506.19398 v1 §3.1.1]` 【確認:論文】

- clean 850 h: DNS-Challenge 4 の speech（DNSMOS で選別）、VCTK（110 話者×各 400 発話）、社内 TTS（中国語 200 話者×500 発話、英語 2 話者×50,000 発話）。
- noise 300 h: DNS4 noise、WHAM! 48 kHz、社内収録。
- RIR: rir-generator による合成 100,000 本。
- 混合: clean と残響音声を 7:3、SNR を 0〜15 dB でランダム。
- 規模: train 8000 h / dev 50 h。10 % を 16 kHz に、5 % を 8 kHz に落としてから 48 kHz へ戻す。
- 16 kHz の広帯域用には、フルバンドを resample したものに LibriTTS 262 h を加える。

この文章が配布 checkpoint の学習に**そのまま**当てはまるとは、論文に書かれていない。【UNKNOWN】

### 3.3 論文と公式コードの実使用データの差

- 公開コードの学習データ（VB のみ、約 9.4 h）と、論文の 8000 h は規模も混合方法も別である。
- 論文表 2 の見出しは、VoiceBank+DEMAND test を「824 clips from two speakers with multiple SNR levels (−7.5, −2.5, 2.5, 7.5 dB)」と書く。ローカルの VB の `log_testset.txt` では SNR は 2.5/7.5/12.5/17.5 dB（824 件）である。論文の SNR 見出しはデータと食い違う。【確認:論文】【確認:データ】
- 論文表 2 は 48 kHz の VB の表で、MossFormerGAN は載っていない（載るのは `Resemble_Enhance`、`DeepFilterNet`、`MossFormer2_SE_48K`）。MossFormerGAN の VB 16 kHz の値（PESQ 3.47）は README の表にだけある。

### 3.4 ローカル VB の実測（読み取り）

| 項目 | 28 話者 train | test |
|---|---|---|
| ファイル数 | 11,572 | 824 |
| 総時間 | 9.39 h | 0.58 h |
| 平均 / 中央値 / 最小 / 最大（秒） | 2.92 / 2.68 / 1.09 / 15.11 | 平均 2.51 |
| 2 秒未満 | 1,497 件（12.94 %） | 260 件（31.55 %） |
| SNR（付属ログ） | 0/5/10/15 dB | 2.5/7.5/12.5/17.5 dB |
| 雑音の種類（付属ログ） | 10 種 | 5 種（bus、cafe、living、office、psquare） |

- 2 秒未満の発話は、公式の 2 秒 crop では末尾 0 埋めになる（§4）。validation（test）では 31.55 % が 0 埋めになる。
- train は p286、p287 を除くと 10,700 件、除いた 2 話者が 872 件である。audio-db の分割がこれに当たる（`src/data/voicebank_sets.py`）。

## 4. C. データ生成・前処理(処理順序を番号付きで)

### 4.1 分類

- **固定ペア（事前生成）。** 公開コードにオンライン混合は無い。雑音選択、SNR、ゲイン、RIR、重ね合わせは学習中に一度も選ばれない。【確認:コード】
- 毎回サンプリングされるのは、**ランダム crop の開始位置**と、DataLoader の順序だけである。

### 4.2 公開コードの 1 サンプルあたりの処理順序（train・validation 共通）

1. `read_and_config_file` が scp の 1 行から (noisy, clean) の 2 パスを読む。`[CVS@6b3774d : T/dataloader/misc.py : L15-37]`
2. `AudioDataset.__getitem__` が `Wave_Processor.process(..., segment_length=sampling_rate*max_length=32000, ...)` を呼ぶ。`[CVS@6b3774d : T/dataloader/dataloader.py : L129, L135-137]`
3. noisy を `audioread` で読む。`sf.read`（float64、全チャネル）→ `audio_norm` → fs≠16000 なら `librosa.resample` → 先頭チャネル。`[CVS@6b3774d : T/dataloader/dataloader.py : L25-32 : audioread]`
4. `audio_norm`（noisy 用）: RMS を −25 dBFS に合わせ（`10**(-25/20)/(rms+EPS)`）、続けて「平均パワーより大きいサンプルだけの RMS」をもう一度 −25 dBFS に合わせる。`[CVS@6b3774d : T/dataloader/dataloader.py : L34-43 : audio_norm]`
5. clean にも**独立に**手順 3〜4 を適用する。noisy と clean は別々のゲインで正規化される。**相対ゲインが保たれない。** `[T/dataloader/dataloader.py : L65-66 : Wave_Processor.process]` 【確認:コード】
6. 長さの扱い。`len_wav = wave_labels.shape[0]`（clean の長さ）。
   - `wave_inputs.shape[0] < 32000` なら、長さ 32000 のゼロ配列の**先頭**へ貼り付ける（末尾 0 埋め）。
   - それ以外は `st_idx = random.randint(0, len_wav - 32000)` で同じ区間を noisy と clean から切り出す。`[T/dataloader/dataloader.py : L67-76]`
7. `collate_fn_2x_wavs` が `torch.FloatTensor` へまとめる。長さは全件 32000 に揃うので、可変長処理は不要である。`[T/dataloader/dataloader.py : L159-163]`
8. solver 側: `c = sqrt(T / (sum(x²) + EPS))` を noisy の発話ごとに計算し、noisy に掛ける（`EPS = 1e-6`）。`[CVS@6b3774d : T/solver.py : L266-268 : _run_one_epoch_mossformergan_se_16k]`
9. 損失関数の中で clean にも同じ `c` を掛ける。`[CVS@6b3774d : T/losses/loss.py : L29-30 : loss_mossformergan_se_16k]`
10. STFT: hamming（`periodic=False`）、`win_len=fft_len=400`、hop 100、`center=True`。power 圧縮 0.3。`[T/solver.py : L269-271]`、`[T/utils/misc.py : L150-196]`
11. クリップや peak 正規化は無い。他の augmentation も無い。【確認:コード】

### 4.3 実効値と注意

| 項目 | 内容 | 確度 |
|---|---|---|
| 雑音選択・SNR・ゲイン・RIR | 該当なし（固定ペア）。VB 付属ログの SNR は 0/5/10/15 dB（train）。 | N/A |
| リサンプリング | scp が 16 kHz なので、通常は `librosa.resample` を通らない。通る場合の `res_type` は `librosa==0.10.2.post1` の既定（`soxr_hq`）。事前変換の方法は不明。 | 【確認:コード】【UNKNOWN】 |
| 正規化の副作用 | `audio_norm` による clean/noisy のゲイン比のずれ。VB 28 話者から無作為 200 ペアで測定: `20·log10(g_clean/g_noisy)` は平均 +0.38 dB、標準偏差 0.66、最小 −0.90、最大 +3.14、5 %点 −0.34、95 %点 +1.70。 | 【確認:コード】実測（`/tmp/audit-mossformergan/gainstat.py`） |
| 推論側 | ClearerVoice の推論パッケージ（`clearvoice/`）は MossFormerGAN で `use_norm=False`（`audio_norm` 無し）。`T/inference.py` は `audioread`（`audio_norm`）を通し、`output * scalar` で戻す。 | `[CVS@6b3774d : clearvoice/clearvoice/dataloader/dataloader.py : L236-240]`、`[T/inference.py : L36-39]` |
| 短い発話 | 2 秒未満の発話は末尾 0 埋めのまま、`c` の計算に 0 の区間が入る。 | 【確認:コード】 |

### 4.4 オンライン混合の場合に毎回再サンプリングされるもの

該当なし（公開コードにオンライン混合は無い）。配布 checkpoint の混合（論文 §3.1.1）の再サンプリング単位は【UNKNOWN】。

## 5. D. Dataset/DataLoader/Sampler(epoch定義表を含む)

### 5.1 公式の実効動作

| 項目 | 内容 | 根拠・確度 |
|---|---|---|
| Dataset | `AudioDataset`（train/val/test で同じクラス）。 | `[T/dataloader/dataloader.py : L104-141]` |
| segment 長 | `sampling_rate * max_length = 32000`。**train だけでなく val にも同じ長さが適用される。検証も 2 秒のランダム crop である。** | `[T/dataloader/dataloader.py : L129]` 【確認:コード】 |
| shuffle の実装 | DDP なし: `DataLoader(shuffle=True)`（標準の RandomSampler、train・val とも）。DDP あり: 独自 `DistributedSampler`。 | `[T/dataloader/dataloader.py : L225-242 : get_dataloader]` |
| 独自 `DistributedSampler` | `ind = randperm(int(N/world), g) * world`、`indices = Σ_i (ind+i)`、rank r は `indices[r*n:(r+1)*n]` を取る。つまり rank r の担当は**インデックスが r mod world のサンプルで固定**（話者・順序の偏りが固定）され、epoch ごとに変わるのは rank 内の並びだけである。 | `[T/dataloader/dataloader.py : L192-211]` 【確認:コード】 |
| sampler の seed | `DistributedSampler(datasets, num_replicas, rank)` と呼び、`seed` 引数を渡さない（既定 0）。**`args.seed` と無関係**。順序は `seed(0)+epoch` で決まる。 | `[T/dataloader/dataloader.py : L225-228]` |
| validation の sampler | 検証にも同じ sampler を作る（`shuffle=True`）。`set_epoch` は train の sampler だけに呼ぶので、検証の並びは毎回同じ（epoch 0）。 | `[T/solver.py : L188]` |
| 端数 | `num_samples = ceil(N/world)`、足りない分は先頭から繰り返して補う（重複あり）。 | `[T/dataloader/dataloader.py : L187-188, L205]` |
| `drop_last` | 指定なし（False）。batch 1 なので影響なし。 | `[T/dataloader/dataloader.py : L237-242]` |
| 可変長 | 全件 32000 に揃えるので、なし。 | §4.2 |
| chunk iterator | 無い。 | 【確認:コード】 |
| worker の乱数 | `random.randint` は DataLoader worker の Python `random` を使う。PyTorch は worker ごとに `random.seed(base_seed + worker_id)` を設定し、`base_seed` は main の torch 乱数から DataLoader が epoch の最初に引く。`torch.manual_seed(args.seed)` を呼ぶので理屈の上では決定的だが、main の CPU 乱数の消費順（モデル初期化、dropout 等）に依存する。 | `[T/train.py : L15-20]`。PyTorch の仕様【推測】 |
| seed | YAML の `seed: 777`（`train.py` の argparse 既定は 20 で、YAML が上書きする）。`random`、`np.random`、`torch.manual_seed` を設定し、`cudnn.deterministic=True`、`benchmark=False`。 | `[T/train.py : L15-20, L86]`、`[T/config/train/MossFormerGAN_SE_16K.yaml : L25]` |
| DDP | `args.distributed = WORLD_SIZE > 1`。`train.sh` は `n_gpu=2`、`torch.distributed.launch`。 | `[T/train.py : L132-136]`、`[T/train.sh]` |
| accumulation とバッチ | `batch_size: 1`、`accu_grad: 1`、`effec_batch_size: 2`（"per GPU"）。1 GPU あたり micro-batch 1 × 蓄積 2 で 1 回更新する。グローバルバッチ = 2 × GPU 数（`train.sh` 既定の 2 GPU なら 4）。 | `[T/config/train/MossFormerGAN_SE_16K.yaml : L28-32]`、`[T/solver.py : L278-292]` |
| 蓄積の打ち切り | epoch の最初に `accu_count = 0` と `zero_grad` を呼ぶので、epoch の終わりに余った蓄積分は捨てる。 | `[T/solver.py : L259-261]` |
| PESQ 失敗時 | その micro-batch の更新を飛ばす（`accu_count` を進めない）が、`self.step += 1` は進む。 | `[T/solver.py : L279, L303]`、`[T/losses/loss.py : L73-75]` |

### 5.2 epoch の定義表

N = 11,572（公式 train）、audio-db は 10,700。公式は 1 GPU と 2 GPU（`train.sh` 既定）、audio-db は `configs/training/mossformergan.yaml`（1 プロセス、global batch 2）。

| 定義 | 公式 1 GPU | 公式 2 GPU（rank あたり） | audio-db（1 プロセス） |
|---|---|---|---|
| 1 epoch の元発話数 | 11,572 | 11,572（rank に 5,786 ずつ） | 10,700 |
| micro-batch 数（DataLoader の長さ） | 11,572 | 5,786 | 10,700 |
| optimizer 更新回数（PESQ の失敗が無ければ） | 5,786 | 2,893 | 5,350 |
| `Solver.step` の増分 | 11,572（micro-batch 単位、失敗分も +1） | 5,786 | 該当なし（`global_step` は更新回数） |
| 学習ループ回数 | `range(start_epoch, max_epoch+1)` = **121**（0〜120） | 同左 | `max_steps 642000 / 5350` = **120** |
| 検証 | 毎 epoch 1 回 | 同左 | 5,350 更新ごと（world=1 のときだけ epoch 末に一致） |
| checkpoint | 毎 epoch、および `checkpoint_save_freq`（`train.sh` は 1000）micro-batch ごと | 同左 | 5,350 更新ごと |

根拠: `[CVS@6b3774d : T/solver.py : L185-187, L248-252, L323-324]`、`[adb@3bee9ad : configs/training/mossformergan.yaml]`、`[adb@3bee9ad : src/training/data/sampler.py : L15-34 : epoch_indices]`、`[adb@3bee9ad : src/training/core/trainer.py : L292-312]`。【確認:コード】

注意:

- 公式の `epoch` 番号は、plateau の再読み込み（§6.3）で巻き戻る。`self.epoch` と `self.step` が best の値に戻る。
- audio-db の validation と plateau 判定は、`world_size > 1` のとき epoch 末と一致しない。1 epoch の更新数が `validation_interval` より少なくなるためである（§6.4）。【確認:コード】

## 6. E. モデル・学習条件(設定値/実効値の表)

### 6.1 モデル

| 項目 | 内容 | 確度 |
|---|---|---|
| Generator | `SyncANet(num_channel=64, num_features=fft_len//2+1=201)`。6 段の SyncANetBlock。block 内は FFConvM、UniDeepFsmn（`lorder 20`）、MossFormer（intra は `group_size=n_freqs`、inter は 256）、`SELayer`、head 数 4 の注意。 | `[CVS@6b3774d : T/models/mossformer_gan/generator.py : L20-27, L262-338]` |
| Discriminator | `Discriminator(ndf=16)`。NISQA の AdaptCNN → Conformer 2 段 → attention pool → LearnableSigmoid。入力は波形でなく、圧縮後の magnitude の組。`mode == 'train'` のときだけ生成される。 | `[T/models/mossformer_gan/generator.py : L24-27]`、`[T/models/mossformer_gan/discriminator.py]` |
| 初期化 | PyTorch の既定＋モジュール固有の xavier（`conv_module.py`、`mossformer.py`）。専用の初期化関数は無い。 | `[T/models/mossformer_gan/conv_module.py : L221-228]` |
| 初期化の乱数順 | `network_wrapper(args)` を 1 回呼ぶごとに `MossFormerGAN_SE_16K(args)` が 2 回作られる（G 用と D 用）。`train.py` は `network_wrapper` を 2 回呼ぶ。合計 4 回の構築のうち、G は最初の構築、D は最後の構築が使われる。 | `[T/networks.py : L17-18]`、`[T/train.py : L29, L32]` 【確認:コード】 |

### 6.2 学習条件（設定値 / 実効値）

| 項目 | 設定値 | 実効値 | 確度・根拠 |
|---|---|---|---|
| optimizer | YAML に名前なし | G: `AdamW(lr=5e-4)`、D: `AdamW(lr=1e-3)` | `[T/train.py : L56-57]` 【確認:コード】 |
| weight_decay | YAML `weight_decay: 1e-5`、argparse 既定 `1e-5` | **0.01**（`AdamW` の既定）。GAN 分岐は `args.weight_decay` を使わない。 | `[T/train.py : L56-57, L122]`。配布 checkpoint の optimizer も 0.01（§2.2）。【確認:コード】 |
| betas / eps / amsgrad | 指定なし | (0.9, 0.999) / 1e-8 / False | PyTorch の既定 |
| lr | `init_learning_rate: 0.0005`、`finetune_learning_rate: 0.0001` | 新規学習: G 5e-4、D 1e-3 | `[T/config/train/MossFormerGAN_SE_16K.yaml : L17-18]` |
| scheduler | 無し | 無し。plateau は solver の独自処理（§6.3）。 | 【確認:コード】 |
| warmup | 無し | 無し | 【確認:コード】 |
| 勾配クリップ | `clip_grad_norm: 10` | G と D それぞれに `clip_grad_norm_(…, 10)`。**micro-batch ごと**（蓄積の途中でも毎回）。 | `[T/solver.py : L283, L286]` |
| 損失 | — | `0.1·loss_ri + 0.9·loss_mag + 0.2·time_loss + 0.05·gen_loss_GAN` | `[T/losses/loss.py : L57]` |
| time_loss | — | `power_uncompress(labels_real, pred_imag)`（**clean の実部＋予測の虚部**）を iSTFT して L1。 | `[T/losses/loss.py : L49-55]` |
| D 損失 | — | `MSE(D(label,label), 1) + MSE(D(label, pred.detach()), pesq_norm)` | `[T/losses/loss.py : L68-71]` |
| D の目標 | — | `pesq_norm = (wb PESQ − 1) / 3.5`。**PESQ の入力は正規化後の clean と、上記の「実部＋虚部」の合成音声**。 | `[T/losses/loss.py : L62-64]`、`[T/utils/misc.py : L133-148]` |
| PESQ | — | `pesq(16000, clean, noisy, 'wb')`、例外は −1 とし、1 件でも −1 があればバッチごと `None`。joblib `Parallel(n_jobs=-1)`。結果を `.to('cuda')` する（CUDA 必須）。 | `[T/utils/misc.py : L133-148]` |
| 更新順（`accu_grad: 1`） | — | G 損失の backward → G クリップ → D 損失の backward → D クリップ → 蓄積数に達したら G と D の optimizer を step。G の backward は GAN 項を通じて D のパラメータにも勾配を流す。 | `[T/solver.py : L278-292]` |
| `accu_grad: 0` の経路 | — | D 損失の `backward()` が**呼ばれない**（D は G 損失の勾配だけで更新される）。YAML は `accu_grad: 1` なので使われない。 | `[T/solver.py : L293-302]`【確認:コード】 |
| 最大 epoch | `max_epoch: 120` | 121 回のループ（0〜120） | `[T/solver.py : L187]` |
| early stopping | — | 検証損失が改善しない回数が 10 以上で停止（§6.3）。 | `[T/solver.py : L215-226]` |
| 検証間隔 | — | 毎 epoch | `[T/solver.py : L196-202]` |
| checkpoint 間隔 | `checkpoint_save_freq`（`train.sh` は 1000） | 毎 epoch ＋ 1000 micro-batch ごと | `[T/solver.py : L248, L323-324]`、`[T/train.sh]` |
| best 選択 | — | 前の epoch の検証損失より小さいとき `last_best_checkpoint` を更新（§6.3） | `[T/solver.py : L249-250]` |
| batch / GPU | `batch_size: 1`、`effec_batch_size: 2`（per GPU） | グローバル 2×GPU 数。`train.sh` は 2 GPU（`gpu_id=4,6`）。 | §5.1 |
| 初期化の読み込み | `init_checkpoint_path` | 文字列 `'None'` 以外なら `_load_pretrained_model`。ファイルが無ければ「新規学習」にフォールバック。Python の `None`（argparse 既定）は文字列 `'None'` と別で、`os.path.isfile(None)` が例外になる。 | `[T/solver.py : L65-66, L119-149]`、`[T/train.py : L95]` |
| `train.sh` の既定 | — | `network=MossFormer2_SE_48K`（MossFormerGAN ではない）、`init_checkpoint_path=../../clearvoice/checkpoints/${network}/last_best_checkpoint.pt`、`train_from_last_checkpoint=0`。配布 checkpoint があれば**finetune**（G の lr を 1e-4 に設定、D は `.disc.pt` から optimizer ごと読む＝lr は checkpoint の 5e-4）として始まる。 | `[T/train.sh]`、`[T/solver.py : L138-144]` |

### 6.3 plateau と best（実効動作）

`[CVS@6b3774d : T/solver.py : L215-252 : Solver.train]` 【確認:コード】

1. 検証損失 `val_loss >= best_val_loss` のとき、`best_val_loss = val_loss`（**悪化した値で上書きする**）、`val_no_impv += 1`。
   - `val_no_impv == 5` のとき `halving = True`。
   - `val_no_impv >= 10` で停止する。
   - そうでなければ `val_no_impv = 0`、`best_val_loss = val_loss`、`find_best_model = True`。
2. `best` は「全期間の最小」ではなく「直前の epoch より小さい」である。この上書きは 83ba731（2025-03-27）で入った。それ以前は真の best-so-far を保持していた。【確認:コード】
3. `halving` のとき `_load_model(mode='last_best_checkpoint')` で G、D、両 optimizer を読み直し、**G の lr だけ**を 0.5 倍する。D の lr は変えない。この `_load_model` は `self.epoch`、`self.step` も best の値へ巻き戻す。
4. 続く `save_checkpoint()` は巻き戻した `epoch`、`step` で `model.ckpt-{epoch}-{step}.pt` を書く。best checkpoint と同じファイル名なので、**best のファイルが「best の重み＋半減した lr の optimizer」で上書きされる**。【確認:コード】`[T/solver.py : L164-182, L229-237, L248-252]`
5. カウンタ `val_no_impv` は半減後もリセットされない。再起動すると `_init` が `best_val_loss = inf`、`val_no_impv = 0` に戻す（checkpoint に保存されない）。`[T/solver.py : L47-52]`
6. 検証は `self.model.eval()` だけを呼ぶ。**D は train モード**のまま（BatchNorm と Dropout が動く）。`[T/solver.py : L197]` 検証でも G 損失は D の出力と PESQ を使う（`state='val'` でも `loss_mossformergan_se_16k` を呼ぶ）。`[T/solver.py : L275]`
7. 検証損失は `loss_gen`（`0.1 RI + 0.9 mag + 0.2 time + 0.05 GAN`）の平均で、PESQ 失敗で `None` になったバッチは 0 として加算される（分母は `i+1`）。`[T/solver.py : L326-329]`

### 6.4 audio-db 側の対応コード（`adb@3bee9ad`）

| 項目 | 内容 | 根拠 |
|---|---|---|
| plateau の関数 | `mossformer_validation_decision`: `loss >= best` で best を更新して stall+1、5 で `halve`、10 以上で `stop`。公式と同じ形。 | `src/training/se/mossformergan.py : L56-73` |
| 半減 | best snapshot（G、D、両 optimizer）を戻し、G の lr だけ 0.5 倍。snapshot は `improve` のたびに取り直す。 | `src/training/se/mossformergan.py : L126-163` |
| optimizer | `build_optimizer`、weight_decay は `OFFICIAL_ADAMW_WEIGHT_DECAY = 0.01` に固定。D の lr は G の 2 倍。 | `src/training/se/mossformergan.py : L53, L95-97` |
| 更新 | 発話ごとに `scale = sqrt(N/(sum(noisy²)+1e-6))`、G backward → G クリップ → D backward → D クリップ → 蓄積数に達したら両 step。 | `src/training/se/mossformergan.py : L203-264` |
| PESQ 失敗 | `TrainingError` で停止する（公式はバッチを飛ばす）。 | `src/training/se/mossformergan.py : L238-242` |
| 検証 | **時間領域 L1 だけ**。1 発話ずつ、crop なし。PESQ、STOI は `na`（"not computed"）。`losses.generator` に時間 L1 を入れて plateau に渡す。 | `src/training/se/mossformergan.py : L266-282`、`src/training/core/trainer.py : L622-632` |
| 検証と plateau の整列 | 検証は `global_step % validation_interval == 0`、plateau は epoch 末（`cursor >= usable_batches`）。両者は 1 プロセスで `steps/epoch = validation_interval` のときだけ一致する。 | `src/training/core/trainer.py : L299-312` |
| best | `_consider_best` は `steps_per_epoch` も `batch_schedule` も無い構成では**検証 PESQ**で判断する。MossFormerGAN の PESQ は `na` なので `best_generator.pt` は書かれない。さらに `epoch < best_checkpoint_start_epoch`（YAML は 40）なら早期に何もしない。 | `src/training/core/trainer.py : L737-757` |
| 乱数・初期化 | `seed_model_init(config["seed"])` の中で G、D の順に構築。ESPnet 流の初期化は使わない。YAML の `seed: 42`。 | `src/training/se/mossformergan.py : L89-92`、`src/training/core/rng.py : L53-75` |
| 初期化 checkpoint | `init_checkpoint` や finetune の経路は `src/training` に無い。D の公開 checkpoint は未登録。 | `rg` による確認、`requirements/official-assets.toml` |
| 実効バッチ | YAML は global 2 ＝ micro 1 × 蓄積 2 × world 1。 | `configs/training/mossformergan.yaml` |

## 7. F. Validation/Test/評価(論文 vs 公式スクリプト vs src/evaluation)

### 7.1 公式の validation / test

- 学習中の validation は §6.3 のとおり。VB の公式 test（824）を使い、2 秒のランダム crop で、G の全損失を測る。
- 学習側に test 評価は無い（`tt_list` 既定なし）。
- 評価対象 checkpoint は `last_best_checkpoint`（「直前の epoch より損失が小さい」最新の epoch）である。【確認:コード】

### 7.2 比較表

| 観点 | 論文（arXiv:2506.19398 v1） | 公式の評価（README / SpeechScore） | audio-db（`src/evaluation/`） |
|---|---|---|---|
| 評価データ | 表 1: DNS-2020 non-reverb 300 クリップ（16 kHz、SNR 0〜25 dB 一様）。表 2: VB-DEMAND 48 kHz（MossFormerGAN は載らない）。 | VB-DEMAND test（"tested on 16kHz"）と DNS-2020 non-reverb（16 kHz）。 | 評価セットは DuckDB の名前付きセット。VB test は 824 ペア（48 kHz のファイル）。 |
| モデル | 配布の統一モデル | "unified models on the two test sets"（VB で学習したモデルではない） | `src/se/models/mossformergan/` で配布 G を推論 |
| PESQ | 表 1: 3.57（wb）、NB_PESQ 3.88 | `pesq(rate, ref, test, 'wb')`、`score_rate=16000`、`pesq==0.0.4` | `pesq==0.0.4` の wb。8k/16k 以外は 16 kHz へ resample。最短長へ切る。長さ不連続のガード（`SPLIT_ABS_TOL=0.1`）で `na` へ変換。 |
| STOI | 表 1: 98.05 | `stoi(ref, test, rate, extended=False)`。pin は `pystoi==0.3.3`。 | `pystoi` 非 extended。venv は 0.4.1。16 kHz へ resample、最短長、フレーム不足は `na`。 |
| SI-SDR | 論文表 1 に無い | bsseval 流。**平均除去なし**。`a = <ref,est>/<ref,ref>`。 | float64、**平均除去あり**、最短長へ切る。 |
| 長さ | — | 長い方へゼロ埋め（`audio_reader`）。複数 rate は `min(rate)` へ resampy。 | 最短長へ切る。 |
| リサンプラー | — | `resampy` | `torchaudio.functional.resample` |
| 評価単位 | — | 発話単位。平均は `compute_mean_results`。 | 発話単位の `score_utterance` ＋ 集計 |
| 報告値 | 表 1 のみ（上記） | VB test: PESQ 3.47、NB_PESQ 3.96、STOI 0.96、SISDR 19.45。DNS-2020: PESQ 3.57、NB_PESQ 3.88、STOI 0.98、SISDR 20.60。noisy VB は PESQ 1.97 / STOI 0.92。 | 現行の評価設定ファイル（`configs/evaluation/`）は作業ツリーに無い。 |

根拠: `[論文 arXiv:2506.19398 v1 表1, 表2]`、`[CVS@6b3774d : clearvoice/README.md : L124-139]`、`[CVS@6b3774d : speechscore/scores/pesq.py : L8-15]`、`[speechscore/scores/stoi.py : L10-15]`、`[speechscore/scores/sisdr.py : L10-31]`、`[speechscore/speechscore.py : L113-155]`、`[CVS@6b3774d : requirements.txt : pesq==0.0.4, pystoi==0.3.3]`、`[adb@3bee9ad : src/evaluation/metrics/{pesq,stoi,sisdr,compute}.py]`。【確認:コード】【確認:論文】

### 7.3 評価条件の差が数値へ与える影響

- README の VB 値は 16 kHz の test ファイルを使う。audio-db は 48 kHz のファイルを 16 kHz モデルへ通す。推論 `enhance()` が 48→16→48 kHz と往復し、指標は 48 kHz の両信号を 16 kHz へ落とす。clean 側は 1 回、enhanced 側は 2 回リサンプルされる。影響の大きさは未測定である。【確認:コード】【UNKNOWN】
- SI-SDR の平均除去の有無、長さの扱い（切る / ゼロ埋め）、`pystoi` の版は、README の値と audio-db の値を直接比べるときの交絡になる。
- 配布 checkpoint は VB で学習したものではない（"unified models"）。VB のみで学習した audio-db のモデルと README の 3.47 を同列に比べてはいけない。
- 論文の表 1（DNS-2020）を再現するには DNS-2020 のテストセットが要る。audio-db にそのセットがあるかは未確認である（`data/db/` が空）。

## 8. 現行 audio-db との項目別比較表

判定: `MATCH`（確認した範囲で一致）、`PARTIAL`、`DIFF`、`UNKNOWN`、`N/A`。「公式設定」は YAML・argparse の値、「公式実効動作」は実際の処理。いずれも `CVS@6b3774d`。audio-db 側は `adb@3bee9ad` のコードを読んで確認した。

| 項目 | 公式設定 | 公式実効動作 | audio-db現状 | 判定 | 根拠 |
|---|---|---|---|---|---|
| Dataset（train） | `tr_demand_28_spks_16k.scp` | VB 28 話者 11,572 ペア（16 kHz に事前変換済み） | `voicebank_28_train`: 10,700 ペア（48 kHz 素材、p286・p287 を除外） | DIFF | `T/config/train/MossFormerGAN_SE_16K.yaml:L14`、`src/data/voicebank_sets.py`、`configs/training/mossformergan.yaml` |
| Dataset（validation） | `cv_demand_testset_16k.scp` | VB 公式 test 824 ペア | `voicebank_28_valid`: 872 ペア（p286・p287）。test の validation 利用は `_set_rejection` が拒否する。 | DIFF | `T/config/…yaml:L15`、`src/training/data/dataset.py:L29-43, L300-327` |
| 混合方式 | — | 固定ペア（事前生成）。オンライン混合なし。 | 固定ペア（空の `operations` のレシピ） | MATCH | §4、`src/data/voicebank_sets.py`、`docs/db/RECIPE.md` |
| 雑音選択 | — | 該当なし | 該当なし | N/A | §4.3 |
| SNR | — | VB 付属ログの値（train 0/5/10/15 dB）。学習時の選択なし。 | 同じファイルを使うため同じ | N/A | `log_trainset_28spk.txt` |
| ゲイン | — | `audio_norm` が noisy と clean に独立に作用する。clean/noisy のゲイン比が平均 +0.38 dB（最大 +3.14 dB）ずれる。 | noisy の全発話 RMS で noisy と clean に**同じ**ゲインを掛ける（相対ゲインを保つ） | DIFF | `T/dataloader/dataloader.py:L34-43, L65-66`、`src/training/data/dataset.py:L359-364` |
| 残響/RIR | — | 該当なし（VB に RIR 無し） | 該当なし | N/A | §3.1 |
| リサンプリング | — | scp は 16 kHz の事前変換済み（方法は不明）。`librosa` は fs≠16k のときだけ。 | 48 kHz 素材を読み込み時に `torchaudio.functional.resample` で 16 kHz へ | UNKNOWN | `T/dataloader/dataloader.py:L28-29`、`src/training/data/dataset.py:L208-209` |
| 正規化 | `audio_norm`（−25 dBFS 二段）→ 発話ごとの `c` | noisy・clean を独立に `audio_norm`。続いて noisy 基準の `c` を両方へ。 | noisy の全発話 RMS=1 ＋ 同じゲインを clean へ。続いて同じ `c`。 | PARTIAL | `T/solver.py:L266-268`、`src/training/se/mossformergan.py:L218-219` |
| Crop | `max_length: 2`（32000） | clean の長さを使い `random.randint(0, L-32000)`。noisy・clean で同じ位置。 | `crop_start`（case_id・seed・epoch から決定的）。範囲は `[0, L-segment]` で同じ。 | PARTIAL | `T/dataloader/dataloader.py:L74`、`src/training/data/sampler.py:L37-46` |
| Padding | 2 秒未満は末尾 0 埋め | 末尾 0 埋め（先頭へ貼り付け） | 末尾 0 埋め | MATCH | `T/dataloader/dataloader.py:L68-72`、`src/training/data/dataset.py:L367 _segment` |
| 音声長制限 | `max_length: 2` | train・val とも 2 秒。推論は 10 秒窓（`clearvoice`）または 1 秒窓（`T/` の推論 YAML）。 | train 32000。validation は全長。 | PARTIAL | `T/dataloader/dataloader.py:L129`、`src/training/se/run.py:L111-119` |
| Shuffle/Sampler | 独自 `DistributedSampler`（seed 0 固定、rank 固定の部分集合）、DDP 無しは RandomSampler | 同左 | `torch.randperm`（`mix_seed("order", seed, epoch)`）、`order[:usable][rank::world]` | PARTIAL | `T/dataloader/dataloader.py:L192-211`、`src/training/data/sampler.py:L15-34` |
| Epoch定義 | `max_epoch: 120` | ループは 0〜120 の 121 回。micro-batch 単位の `step`。 | `max_steps 642000` ＝ 120 epoch（5,350 更新/epoch） | PARTIAL | §5.2 |
| Batch | `batch_size 1`、`effec_batch_size 2`（per GPU） | グローバル 2×GPU 数 | micro 1 × 蓄積 2 × world 1 = 2 | PARTIAL | `T/solver.py:L278-292`、`configs/training/mossformergan.yaml` |
| DDP/accumulation | DDP＋`SyncBatchNorm`、`find_unused_parameters=True`。蓄積に `no_sync` 無し。 | micro-batch ごとにクリップ、蓄積途中の勾配に積む。epoch 末の余りは捨てる。 | 独自の DDP 処理（D の勾配を手動 all-reduce）。蓄積途中のクリップも行う。 | PARTIAL | `T/solver.py:L30-34, L259-261`、`src/training/se/mossformergan.py:L203-264` |
| 損失 | 0.1 RI＋0.9 mag＋0.2 time＋0.05 GAN | 同左。time は clean 実部＋予測虚部。D は MSE(D(l,l),1)＋MSE(D(l,p),PESQ_norm)。 | 同じ式と係数 | MATCH | `T/losses/loss.py:L26-77`、`src/training/se/mossformergan.py:L231-247`。数値一致は未再検証。 |
| Optimizer | YAML に指定なし | AdamW、G 5e-4、D 1e-3、weight_decay 0.01 | AdamW、G 5e-4、D 1e-3、weight_decay 0.01 | MATCH | `T/train.py:L56-57`、`src/training/se/mossformergan.py:L53, L95-97` |
| Scheduler | 無し | 無し（独自 plateau） | `scheduler: none` ＋ 独自 plateau | MATCH | `T/solver.py:L215-237`、`src/training/se/mossformergan.py:L56-73` |
| 勾配クリップ | `clip_grad_norm 10` | G と D それぞれ、micro-batch ごと | 同じ | MATCH | `T/solver.py:L283, L286`、`src/training/se/mossformergan.py:L250-258` |
| 初期化 | seed 777 | PyTorch 既定＋モジュール固有。G は最初、D は 4 回目の構築。 | seed 42。G、D の順に 1 回ずつ構築。 | PARTIAL | `T/train.py:L29-32`、`T/networks.py:L17-18`、`src/training/se/mossformergan.py:L89-92` |
| 最大 epoch/step | `max_epoch 120` | 121 ループ | 642,000 更新（120 epoch） | PARTIAL | `T/solver.py:L187` |
| Validation（データ） | VB test 824 | 同左 | 872（p286、p287） | DIFF | §8 の Dataset |
| Validation（前処理） | `audio_norm`＋2 秒ランダム crop | 同左（val にも crop） | 全長・crop なし | DIFF | `T/dataloader/dataloader.py:L129`、`src/training/se/run.py:L111-119` |
| Validation（バッチ・損失） | `batch_size 1` | G 全損失（GAN、PESQ 込み）。D は train モード。 | 1 発話ずつ、時間 L1 のみ。PESQ、STOI は `na`。 | DIFF | `T/solver.py:L197, L275, L326-329`、`src/training/se/mossformergan.py:L266-282` |
| Best選択/early stopping | `last_best_checkpoint`。5 で G の lr 半減、10 で停止。 | 「直前の epoch より小さい」で best を更新。半減時に best を上書き。 | plateau の規則は同じ。best の成果物は作らない（`na`）。 | PARTIAL | `T/solver.py:L215-252`、`src/training/core/trainer.py:L737-757` |
| Resume | `train_from_last_checkpoint` | model、optimizer、epoch、step を復元。plateau のカウンタは復元しない。 | model、optimizer、plateau、best snapshot、RNG を復元 | DIFF | `T/solver.py:L47-52, L74-112`、`src/training/se/mossformergan.py:L171-200` |
| 評価指標（PESQ） | wb、16 kHz | `pesq 0.0.4`、SpeechScore | `pesq 0.0.4` wb、16 kHz。ただし 48 kHz ファイルから往復。 | PARTIAL | §7.2 |
| 評価指標（STOI） | 非 extended | `pystoi 0.3.3` | `pystoi 0.4.1`、非 extended | PARTIAL | §7.2 |
| 評価指標（SI-SDR） | — | 平均除去なし、bsseval 流 | 平均除去あり | DIFF | `speechscore/scores/sisdr.py:L10-31`、`src/evaluation/metrics/sisdr.py` |
| Test | `tt_list` 既定なし | 学習中は走らない | 評価は `src/evaluation/`（学習とは別） | N/A | `T/solver.py:L204` |

## 9. 差異一覧

影響度: Critical = プロトコルや学習結果に直接影響、Major = 性能や比較条件に影響し得る、Minor = 影響限定的、Unknown。優先度: P0 = 公式プロトコル再現に不可欠、P1 = 再現性・検証可能性に重要、P2 = 性能改善・効率化・運用。

| ID | 項目 | 影響度 | 実装状態 | 優先度案 | 修正が要りそうなファイル |
|---|---|---|---|---|---|
| MFG-D01 | 学習/検証の分割（公式: train 11,572、validation = test 824。audio-db: 10,700 / 872） | Major | 実装済み（別の分割）。公式分割は未実装で、validation に test を使う経路は `_set_rejection` が拒否する。 | P1 | `src/data/voicebank_sets.py`、`src/training/data/dataset.py`、`configs/db/` |
| MFG-D02 | 検証の定義（2 秒ランダム crop・G 全損失・D は train モード / 全長・時間 L1 のみ）。plateau の判定値が変わる。 | Critical | audio-db に未実装 | P0 | `src/training/se/mossformergan.py`、`src/training/core/trainer.py` |
| MFG-D03 | best checkpoint の成果物が無い（PESQ が `na`、`best_checkpoint_start_epoch: 40` は無関係） | Major | audio-db に未実装 | P0 | `src/training/core/trainer.py`、`configs/training/mossformergan.yaml` |
| MFG-D04 | plateau の細部（best の上書き、半減時の epoch/step 巻き戻し、再起動でのカウンタ初期化 / audio-db は保持） | Minor | 実装済みだが公式一致は未検証（`tests/training/test_mossformer_plateau.py` が単体の規則を確認。本監査では再実行せず） | P2 | `src/training/se/mossformergan.py` |
| MFG-D05 | noisy・clean の独立 `audio_norm`（相対ゲインのずれ）。audio-db は相対ゲインを保つ。 | Major | audio-db に未実装 | P1（研究判断。公式再現を優先するなら P0） | `src/training/data/dataset.py`（`_normalize`） |
| MFG-D06 | リサンプリング（公式は事前変換 16 kHz ＋ `librosa`、audio-db は `torchaudio` で読み込み時） | Unknown | 実装済みだが公式一致は未検証 | P1 | `src/training/data/dataset.py`、`src/data/` |
| MFG-D07 | crop の乱数・順序の生成（公式: worker の `random` と torch の乱数の消費順 / audio-db: case_id 決定的） | Minor | 実装済み（意図的な差。範囲と 0 埋めは同じ） | P2 | `src/training/data/sampler.py` |
| MFG-D08 | DDP の分割（公式: rank 固定の部分集合・seed 0 固定 / audio-db: 全体順列の飛び飛び）、実効バッチ（2×GPU 数）、validation と plateau の整列（world>1 で崩れる） | Major | 実装済みだが公式一致は未検証 | P1 | `src/training/data/sampler.py`、`src/training/core/trainer.py`、`src/training/core/batch.py` |
| MFG-D09 | epoch の数え方（121 ループ / 120 epoch、step は micro-batch 単位、epoch 末の余りの蓄積を捨てる） | Minor | 実装済み（意図的な差） | P2 | `configs/training/mossformergan.yaml` |
| MFG-D10 | PESQ 失敗時の扱い（公式: バッチを飛ばして `step` だけ進める / audio-db: `TrainingError` で停止） | Minor | audio-db に未実装 | P1 | `src/training/se/mossformergan.py` |
| MFG-D11 | Hamming の `periodic`（公式の学習コードは `False`、`clearvoice` の推論は `True`）。配布 checkpoint の学習側は UNKNOWN。audio-db は学習 `False`、推論 `True`。 | Unknown | 実装済みだが公式一致は未検証 | P1 | `src/se/models/mossformergan/inference.py`、`src/training/se/mossformergan.py` |
| MFG-D12 | 評価条件（16 kHz test ＋ SpeechScore / 48 kHz test ＋往復リサンプル、SI-SDR の平均除去、長さ処理、`pystoi` の版） | Major | 実装済みだが公式一致は未検証 | P0（README 値と比べる場合） | `src/evaluation/metrics/*`、評価セット定義（16 kHz 版 VB test を追加する） |
| MFG-D13 | 初期化の乱数順（公式は 4 回構築、G は最初・D は最後。seed 777。audio-db は G→D の 1 回ずつ。seed 42） | Minor | 実装済み（意図的な差） | P2 | `src/training/se/mossformergan.py` |
| MFG-D14 | 配布 checkpoint の学習レシピが公開コードと食い違う（D の lr 0.0005、`initial_lr`、大規模データ、epoch 7）。VB のみの学習では README の数値は期待できない。 | Unknown | 該当なし（文書で扱う） | P1（比較の注記） | `docs/training/`、`docs/se/MODELS.md` |
| MFG-D15 | 作業ツリーに DB・staging・`configs/evaluation/`・`*_seed17.yaml`・`trial100/` が無い。YAML の見出しは seed 42、docs は seed 17。 | Minor | audio-db に未実装（再構築が必要） | P1 | `docs/training/*.md`、`configs/training/mossformergan.yaml` の見出し |
| MFG-D16 | 公式の finetune（`init_checkpoint_path`、G の lr 1e-4、D は `.disc.pt` から optimizer ごと）。audio-db に経路が無く、D の公開 checkpoint も未登録。 | Minor | audio-db に未実装 | P2 | `src/training/se/run.py`、`requirements/official-assets.toml` |

「実装済み」と書いた差異でも、audio-db の値が公式の値と同じ数値を出すとは確認していない。

## 10. DB中心の新データ基盤に必要な情報(1〜10)

### 10.1 DB に登録すべき音声素材

- 候補 (a) は固定ペアなので、VB-DEMAND の noisy/clean 対応ペアだけでよい（28 話者 train 11,572、test 824）。
  - 16 kHz の事前変換版を使うか、48 kHz 原本から変換するかを、**素材ごとに記録する**（MFG-D06）。
  - 作者の NAS 上の 16 kHz 版は手元にない。
- 候補 (b)（論文の混合）を再現するなら、次の素材が必要になる。社内データを含むので全部は揃わない。
  - clean: DNS4 speech（DNSMOS 選別済み）、VCTK、LibriTTS（262 h）、社内 TTS。
  - noise: DNS4 noise、WHAM! 48 kHz、社内収録。
  - RIR: rir-generator の 10 万本。

### 10.2 clean / noise / RIR の役割

- 候補 (a): clean = 教師波形、noisy = 入力。noise と RIR は独立の素材としては不要（混合済み）。
- 候補 (b): clean（7 割はそのまま、3 割は RIR で残響付与）に noise を SNR 0〜15 dB で足す。教師は dry clean か残響 clean かが論文で明示されていない。【UNKNOWN】

### 10.3 必要なデータ分割・対応関係

- 公式: train（28 話者全体）と validation（= test）の 2 つ。公式の validation は test と同一である。
- audio-db の現行: train / valid（p286・p287）/ test。公式にそろえる場合は、「公式 train」「公式 validation（= test）」を別セット名で用意し、test と重なる validation を**評価用途に使わない**ことを区別して記録する。
- noisy と clean は同名ファイルの 1 対 1。話者・雑音の重複は、VB の仕様（test の話者と雑音は train と別）に従う。

### 10.4 固定して保存すべき生成条件

- VB の対応ペアのファイル一覧とハッシュ。scp の行順。
- 事前変換の条件（リサンプラー名、版、`res_type`）。
- `audio_norm` の定数（−25 dBFS、EPS 1e-6、二段目の「平均パワー超過サンプルの RMS」）。
- segment 長 32000、`max_length 2`。

### 10.5 学習時にサンプリングすべき生成条件（確率分布）

- 候補 (a): crop の開始位置 `U{0, L−32000}` と、サンプルの順序のみ。
- 候補 (b): 論文に書かれているパラメータ。形（一様かなど）は書かれていない。
  - SNR は「0〜15 dB のランダム」（一様と推測【推測】）。
  - clean と残響の比は 7:3。
  - 帯域制限の複製は 10 % が 16 kHz、5 % が 8 kHz。その後に 48 kHz へ戻す。
  - RIR は 10 万本から選ぶ（選び方は不明）。noise の選び方も不明。
  - 複数雑音の重ね合わせの有無は不明。

### 10.6 データローダー固有の処理

- `audio_norm`（二段の −25 dBFS）を noisy と clean に独立に適用する。
- 2 秒の crop（train、validation とも）と末尾 0 埋め。
- 発話ごとの `c` 正規化は solver 側にあり、損失の中で clean にも `c` を掛ける。
- 独自 `DistributedSampler`（rank 固定の部分集合、seed 0）。
- 検証にも crop と全損失（D は train モード）を使う。

### 10.7 共通化できる処理

- ペア素材の読み込み、チャネル選択、リサンプリング（方式を引数にする）、末尾 0 埋めと crop（乱数の引き方を引数にする）。
- 発話ごとの `c` 正規化（audio-db は `MossFormerGANTask.train_update` 内に既にある）。
- case_id による決定的な順序と crop（audio-db の `sampler.py`）。

### 10.8 公式動作の維持に専用実装が必要な処理

- 独立 `audio_norm`（相対ゲインのずれを含めて再現する）。
- 検証の「ランダム crop ＋ G 全損失（D は train モード）」と、その損失値を plateau に渡す経路。
- `last_best_checkpoint` の意味（直前 epoch との比較）、半減時の best ファイル上書きと epoch/step の巻き戻し。
- 独自 `DistributedSampler` の rank 固定分割。
- PESQ 失敗時にバッチを飛ばす処理。
- 蓄積途中のクリップと、epoch 末に余った蓄積分の破棄。
- 配布 checkpoint を使う finetune の経路（D の `.disc.pt` を含む）。

### 10.9 再現性のために保存すべき状態

- seed（公式は YAML の 777。sampler の seed は 0 固定で別）、epoch、optimizer の更新回数と micro-batch 数の両方。
- plateau の状態: `best_val_loss`、`val_no_impv`、best checkpoint のファイル名、半減した lr。公式は再起動で初期化されるため、「公式どおり」を選ぶか「保持」を選ぶかを記録する。
- sampler の位置（rank ごとの、epoch ごとの順序）。DataLoader worker の乱数（torch の base_seed を含む）。
- ファイルリストのハッシュ（scp ごと）。事前変換の条件。
- 環境: `pesq`、`pystoi`、`librosa`、`torch` の版（公式 `requirements.txt` は `pesq==0.0.4`、`pystoi==0.3.3`、`librosa==0.10.2.post1`、`soundfile==0.12.1`、`torch>=2.0.1`）。
- 実効バッチ（GPU 数、蓄積数）。

### 10.10 既存レシピ JSON v1 で表現できない情報

判断: **v1 は候補 (a) の「固定 1 ケース」だけは表せる。公式の学習プロトコルとしては足りない。**

表せるもの（`docs/db/RECIPE.md`）:

- `sources`（noisy と clean の対応）、`output.sample_rate`、`output.reference: dry_clean`、空の `operations`。
- 実現した 1 ケースの `additive_noise`（SNR、`noise_start`）、`rir`、`peak_normalize`。
- これらにより、候補 (b) の「実現した 1 件」は書ける。

表せないもの:

| 不足 | 候補 | 内容 |
|---|---|---|
| ケース集合の確率分布 | (b) | SNR の分布（0〜15 dB）、clean:残響 = 7:3、RIR の選択確率と選択方法、noise の選択確率、複数雑音の重ね合わせ。v1 は 1 ケースの確定値しか持たない。 |
| ゲイン | (a) | `audio_norm`（−25 dBFS 二段、noisy と clean に独立）。v1 に `gain` や正規化の操作が無い（`peak_normalize` は両者へ同じゲインを掛ける）。 |
| 正規化の対象 | (a) | `c` 正規化は solver 側。レシピの範囲外でよいが、検証の前処理として記録する場所が無い。 |
| リサンプラーの方式 | (a)(b) | `torchaudio` か `librosa` か、`res_type`。v1 は方式を持たない。 |
| 帯域制限 | (b) | 10 %/5 % の 16 kHz/8 kHz への降下と 48 kHz への復元。v1 に操作が無い。 |
| crop・padding の方針 | (a) | `output.max_seconds` は発話の先頭を切るためのもので、train ごとのランダム crop と末尾 0 埋めは表せない。学習時のポリシーとして別に持つ。 |
| 検証の定義 | (a) | 「検証にも crop を適用する」「G 全損失を測る」はデータセットの属性ではなく、プロトコルの属性。 |
| 乱数の状態 | (a)(b) | worker の乱数、sampler の位置、epoch。レシピは再現可能な 1 ケースであって、乱数の状態は持たない。 |
| clipping | (b) | 混合後のクリップ処理が論文に無く、v1 の `peak_normalize` が同じ意味かは不明。 |
| 音量の単位 | (b) | LUFS などの規定は論文に無い。 |
| 分割の意味 | (a) | 公式の validation = test。v1 の `split` は評価用と学習用の区別を含み、この使い方を禁止している。 |

固定 1 ケースのレシピ（v1）と、オンライン生成の確率分布を定めるプロトコルは別物として扱う。公式の候補 (a) は、**ケース（固定ペア）はレシピ、学習時の処理（crop、`audio_norm`、sampler）はプロトコル**に分ける設計が自然である（提案。今回は実装しない）。

## 11. 論文と公式コードの矛盾、既存ドキュメントの不整合・誤り

### 11.1 論文と公式コード・資料の矛盾

1. README は `MossFormerGAN_SE_16K` の論文として MossFormer2 を引くが、その本文は話者分離で、SE や GAN の記述が無い。MossFormerGAN の専用論文は見つからなかった。`[CVS@6b3774d : T/README.md : L10]`、`[論文 arXiv:2312.11825 v2]`
2. ツールキット論文表 2 の VB test の SNR 見出し（−7.5〜7.5 dB）は、ローカル VB の `log_testset.txt`（2.5〜17.5 dB）と違う。`[論文 arXiv:2506.19398 v1 表2]`
3. 配布 checkpoint は D の lr が 5e-4 で `initial_lr` を持つ。公開 `train.py` は D を `init_lr*2 = 1e-3` で作り、scheduler も無い。両者は別の学習コードの産物と考えられる。【推測】
4. 公式 repo の中で Hamming 窓が食い違う。
   - 学習コード（`T/utils/misc.py`）と `T/utils/decode.py` は `periodic=False`。
   - `clearvoice/clearvoice/utils/decode.py` の MossFormerGAN 推論は `periodic=True`（`center=True`）。`clearvoice/` は 2024-11-26 の最初のリリース（`6eb378a`）から `periodic=True` を使う。HF の重み（2024-11-25）に付く推論はこちらである。
   - 配布 G が学習時にどちらの窓を使ったかは確定できない（【UNKNOWN】）。
   - `[CVS@6b3774d : T/utils/misc.py : L171-196]`、`[CVS@6b3774d : clearvoice/clearvoice/utils/decode.py : L294, L308]`
5. 推論の窓長が 2 系統ある。`T/config/inference/MossFormerGAN_SE_16K.yaml` は `decode_window: 1`、`clearvoice/clearvoice/config/inference/MossFormerGAN_SE_16K.yaml` は `decode_window: 10`（どちらも `one_time_decode_length: 10`）。
6. `train.sh` の既定は `network=MossFormer2_SE_48K` で、README の例も同じ。MossFormerGAN の学習には編集が要る。`[T/train.sh]`
7. 公式 solver の best の定義は 83ba731（2025-03-27）で変わった。それ以前は真の best-so-far、以後は「直前の epoch との比較」。同じ公式 repo の版間で plateau 挙動が違う。MossFormerGAN の学習ファイルはその間変更されていない（`f0d52aa` 以降）。

### 11.2 audio-db の既存ドキュメントの不整合・不足

1. `docs/training/OFFICIAL_TRAINING_AUDIT.md`、`VERIFICATION.md`、`EXPERIMENT_PLAN.md` は `configs/training/mossformergan_seed17.yaml` を参照する。`configs/training/trial100/` と `configs/evaluation/` も参照される。**いずれも作業ツリーに無い。** `configs/training/mossformergan.yaml` の見出しは「seed 42」で、「`trial100/mossformergan.yaml` と同じ更新」と書くが、`trial100/` は無い。
2. `OFFICIAL_TRAINING_AUDIT.md` の MossFormerGAN 表（L91-100）は、データの行を「公式ローダのランダム切り出し / case id 決定的 / INTENTIONAL」とだけ書く。次の公式動作が抜けている: 独立 `audio_norm`、検証 = test 824、検証にも 2 秒 crop、検証損失が G の全損失、D は train モード、PESQ 失敗時にバッチを飛ばす。
3. 同表は D の lr を「0.001（G の 2 倍）」とするが、配布 checkpoint の D の lr は 5e-4 である（公開コードと配布物の差）。
4. `docs/se/MODELS.md` は「10 秒を超える発話は公式 decode と同じく 10 秒窓、ストライド 75 %」と書く。これは `clearvoice/` の設定（`decode_window: 10`）に合うが、`T/` の推論 YAML（`decode_window: 1`）とは違う。公式が 2 系統あることが書かれていない。
5. `docs/se/ARCHITECTURE.md`（推論は periodic Hamming）と `docs/training/OTHER_MODELS.md`（学習は `periodic=False`）は、それぞれ正しいが、公式 repo 内の食い違い（§11.1-4）を述べていない。
6. `docs/training/OTHER_MODELS.md` は「CPU の 4 step では公式と独立の損失・勾配・重みが一致した」と書く。本監査では再実行していない。
7. `docs/db/CORPORA.md` は VB の分割（11,572 / 872 / 10,700 / test 824）を述べるが、「公式の validation は test と同一」であることには触れない。
8. `.agents/DATA.md` と docs は DB と staging の存在を前提にするが、この作業ツリーの `data/db/`、`data/staging/` は空である。再構築が必要。

### 11.3 公式コード上の注意（矛盾ではないが再現に関わる）

- `accu_grad: 0` の経路は D 損失を `backward()` しない（§6.2）。YAML は `accu_grad: 1` なので使われない。
- `batch_pesq` は結果を `.to('cuda')` する。CPU 学習では PESQ 項が動かない。
- 配布 `last_best_checkpoint`（指示ファイル）は 1 行（G のファイル名）だけで、D のファイル名を持たない。`T/solver.py` の `_load_model` は 2 行目を D のファイル名として読む（空文字列になる）。この指示ファイルを `checkpoint_dir` に置いて `train_from_last_checkpoint=1` で再開すると、D の読み込みが失敗するはずである（実行していない。【推測】）。`_load_pretrained_model` 経路は `.pt` → `.disc.pt` の置換で D を読む。【確認:コード】`[T/solver.py : L85-87, L106-111, L141-144]`

## 12. UNKNOWN一覧

| # | 何が | 調べた資料 | 確定できなかった理由 | 追加調査案 |
|---|---|---|---|---|
| U1 | 配布 checkpoint の実際のデータ生成（オンライン/オフライン、SNR の分布形、RIR の選択、雑音の選択、複数雑音） | ツールキット論文 §3.1.1、HF カード、公式 repo の全 `train/`、`clearvoice/` | 論文は件数と範囲を書くだけで、生成コードが公開されていない。社内データを含む。 | 著者への問い合わせ。GitHub Issue の検索。 |
| U2 | 配布 checkpoint を作った学習コード（公開版との差） | checkpoint の optimizer 状態（D の lr 5e-4、`initial_lr`、epoch 7、step 1,169,381）、`git log` | 公開コードの追加（2024-11-27）が重みの最終更新（2024-11-25）より後。scheduler の種類は checkpoint から読めない。 | 公開前の commit を持つ fork や社内版の有無を Issue で確認。 |
| U3 | 配布 G の学習時の STFT 窓（`periodic` の別） | `T/utils/misc.py`、`clearvoice/.../decode.py`、`git log -S` | 学習コードと推論が食い違い、配布物から窓は逆算できない。 | 配布 G で 2 通りの窓を比べて推論の出力を評価する（GPU 不要の CPU 実行で可能。今回は未実施）。 |
| U4 | VB の 16 kHz 版を作ったリサンプラー | scp、README | 作者の NAS 上のファイルで、変換の記述が無い。 | 48 kHz 原本から複数の方法で作り、PESQ の差を測る。 |
| U5 | 配布 checkpoint の GPU 数、グローバルバッチ、総 epoch 数 | checkpoint の `epoch`、`step` | 巻き戻しの有無が分からず、`step/epoch` から規模を逆算できない。 | 著者への問い合わせ。 |
| U6 | README の VB 16 kHz の値（PESQ 3.47）を出したときの正確な手順 | README、`speechscore/` | 呼び出しコマンドと、16 kHz 版 test の作り方が書かれていない。noisy の PESQ 1.97 は標準的な VB test の値と合う。 | `speechscore` を CPU で実行し、配布 G の出力と README の値を比べる（今回は未実施）。 |
| U7 | 公式 solver が 2 GPU の DDP で動くか（D の DDP が 1 回の更新で 2 回 backward される） | `T/solver.py : L30-34, L278-292` | GPU を使えず、実行していない。DDP の reducer が二重 backward を許すかは版による。【推測】 | 2 GPU で 1 step の実行を確認（承認が要る）。 |
| U8 | audio-db と公式の数値一致 | 既存 docs の記述、`tests/training/test_mossformer_plateau.py` の存在 | 本監査ではテストと比較を実行していない。 | 公式コードを `/tmp` へ複製して CPU で数 step 比べる。 |
| U9 | `pystoi` 0.3.3 と 0.4.1 の差の大きさ | `requirements.txt`、`.venvs/se` の版 | 同一入力での比較を行っていない。 | 同じ信号を両方で計算する。 |
| U10 | audio-db の DB の中身（`voicebank_28_train` などの実際の件数・ハッシュ） | `data/db/`、`data/staging/` | 両方とも空。 | DB を再構築したあとで確認（`make db-ingest` は明示指示がある場合のみ）。 |
| U11 | 48 kHz→16 kHz→48 kHz の往復が PESQ・STOI に与える影響 | `src/se/models/mossformergan/inference.py`、`src/evaluation/metrics/` | 測定していない。 | 16 kHz 版 VB test のレシピを作り、同じ G で比較する。 |
| U12 | ツールキット論文表 2 の SNR 見出しが VB と違う理由 | 論文 HTML、`log_testset.txt` | 論文の誤記か別条件かが分からない。 | 著者への問い合わせ。 |

## 13. 参照した一次資料

### 論文（arXiv）

- ClearerVoice-Studio: arXiv:2506.19398 v1（Interspeech 2025、2025-06-24）。HTML（CC BY-NC-SA 4.0）を `/tmp/audit-mossformergan/2506.19398.{html,txt}` に取得。§2.2.3、§3.1.1、§3.1.2、表 1・2。
- MossFormer2: arXiv:2312.11825 v2（2024-11-28）。`/tmp/audit-mossformergan/2312.11825.{html,txt}`。
- MossFormer: arXiv:2302.11824（話者分離。本文は未精読で、README が引用していないことだけ確認）。
- CMGAN: arXiv:2203.15149（README の引用先のみ確認）。
- TF-GridNet: arXiv:2209.03952（README の引用先のみ確認）。

### 公式 repo（`CVS@6b3774d`）

- `https://github.com/modelscope/ClearerVoice-Studio`、commit `6b3774dc79c46ae8bed2a4fa5f706f0ac8c75c61`。ローカル pin は `src/se/official/clearervoice-studio`。履歴は `/tmp/audit-mossformergan/cvs` の別クローン。
- `train/speech_enhancement/`:
  - `train.py`、`solver.py`、`networks.py`、`train.sh`、`inference.py`、`README.md`
  - `config/train/MossFormerGAN_SE_16K.yaml`、`config/inference/MossFormerGAN_SE_16K.yaml`
  - `data/tr_demand_28_spks_16k.scp`、`data/cv_demand_testset_16k.scp`
  - `dataloader/dataloader.py`、`dataloader/misc.py`
  - `losses/loss.py`、`utils/misc.py`、`utils/decode.py`
  - `models/mossformer_gan/{generator,discriminator,conv_module,mossformer}.py`
- `clearvoice/clearvoice/`:
  - `config/inference/MossFormerGAN_SE_16K.yaml`、`utils/decode.py`、`utils/misc.py`、`dataloader/dataloader.py`
  - `clearvoice/README.md`（L118-148 のベンチマーク）
- `speechscore/`: `speechscore.py`、`basis.py`、`scores/{pesq,stoi,sisdr}.py`
- `requirements.txt`
- 履歴: `f0d52aa`（2024-11-27）、`7266724`、`9076de3`、`83ba731`（2025-03-27）、`6eb378a`（2024-11-26 `clearvoice` 追加）、`a697da1`（2025-04-30）。タグ `v0.1.1`、`v0.1.2`。

### Hugging Face

- `https://huggingface.co/alibabasglab/MossFormerGAN_SE_16K`、repo commit `dcb1b20e1e44d435090072ab9b223c792ea1f624`。
  - `last_best_checkpoint.pt`: sha256 `04bf00db575389e1ea8b1fcca48d9efcf073e50ad170e6dd5230df3bac98be27`、38,845,593 B。
  - `last_best_checkpoint.disc.pt`: sha256 `4a5ebead6e364896c1cd972526edef1dd48937a7b98e702e348d11b2000632f1`、97,588,777 B（`/tmp/audit-mossformergan/disc.pt` に取得。読み取りのみ）。

### audio-db（`adb@3bee9ad`）

- 規則・docs: `AGENTS.md`、`.agents/{PROJECT,DATA,DEVELOPMENT,DECISIONS}.md`、`docs/training/{OFFICIAL_TRAINING_AUDIT,OTHER_MODELS,VERIFICATION,EXPERIMENT_PLAN}.md`、`docs/se/{MODELS,ARCHITECTURE}.md`、`docs/db/{CORPORA,RECIPE,TRAINING_SETS}.md`、`requirements/official-assets.toml`。
- 設定: `configs/training/mossformergan.yaml`。
- 学習: `src/training/se/{mossformergan,run,config,optim,losses,initialization}.py`、`src/training/core/{trainer,rng}.py`、`src/training/data/{dataset,sampler}.py`。
- データ: `src/data/voicebank_sets.py`、`src/data/corpora/mpsenet_filelist.py`。
- 推論・評価: `src/se/models/mossformergan/inference.py`、`src/evaluation/metrics/{pesq,stoi,sisdr,compute}.py`、`src/evaluation/audio/render.py`。
- テスト（存在のみ確認）: `tests/training/test_mossformer_plateau.py`、`tests/training/test_mossformer_discriminator.py`。

### 実測した補助データ（`/tmp/audit-mossformergan/`）

- `gainstat.py`: VB 28 話者から無作為 200 ペアの `audio_norm` ゲイン比。
- `durstat.py`: VB train・test の発話長の統計。
- ローカル VB（`data/corpora/VoiceBank-DEMAND`）の `log_trainset_28spk.txt`、`log_testset.txt`: SNR と雑音の種類。
