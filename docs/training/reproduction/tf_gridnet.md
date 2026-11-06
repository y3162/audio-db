# TF-GridNet 公式再現プロトコル監査

P5（2026-11-06）で複数 rank の chunk 計画は実装した。下の「DDP は未実装」という行は、この監査時点の記録であり、現行の計画は [P5_VERIFICATION](P5_VERIFICATION.md) を見る。MS-SNSD の学習入口はオンデマンド再構築で、記録は [P6_VERIFICATION](P6_VERIFICATION.md)。全量 WAV は作らない。

監査日 2026-11-06。対象リポジトリ `audio-db`、ブランチ `feature/reproduction`、HEAD `3bee9ad`。この文書は調査と記録だけで、コード・設定・DB は変更していない。修正案は提案として書く。

凡例。確信度ラベルは 【確認:コード】【確認:論文】【確認:設定】【推測】【UNKNOWN】。根拠は `[場所 @ 版 : ファイル : 行 : 関数/キー]` または `[論文 arXiv:XXXX vN §節/表]`。
- `espnet@0d136c5` は `build/clean-20261104/sources/espnet`（読み取り専用）の行番号。履歴・タグは `/tmp` のブロブレス clone（master `66f0ff5af`）で調べた。
- `HF:ms_snsd` は `data/checkpoints/tf-gridnet/ms-snsd/config.yaml`（HF `espnet/ms_snsd_tfgridnet` rev `49144db0`、中身は HF と `cmp` 一致）。
- `HF:urgent24` は HF `wyz/tfgridnet_for_urgent24` rev `85f2296b` の `exp/enh_train_enh_tfgridnet_raw/config.yaml`（作業ツリーには無く、`/tmp/audit-tfgridnet/urg_config.yaml` に取得した）。
- 既存 docs は出発点であり根拠として扱っていない。「GO」の記述は一次資料と現行コードで再確認した。

---

## 0. 要約

1. **「公式 GitHub」は ESPnet**。著者 Zhong-Qiu Wang 個人の TF-GridNet 単独リポジトリは見つからない。論文 (b) が脚注で ESPnet PR #5395 を指す。 [arXiv:2211.12433v2 脚注 / https://github.com/espnet/espnet/pull/5395]【確認:論文】
2. **どの論文実験の ESPnet レシピも、audio-db が評価する単一話者 SE の構成にはなっていない**。論文 (a)(b) の実験は WSJ0-2mix / SMS-WSJ / WHAMR! / WSJ0CAM-DEREVERB / L3DAS22 で、MS-SNSD・DNS・URGENT は含まない。ピン留め ESPnet で TF-GridNet の conf を持つレシピは `wsj0_2mix`、`wsj0_2mix_spatialized`、`wsj_kinect`、`dns_ins20`、`ms_snsd`、`urgent24` の 6 つ。
3. audio-db の `ms-snsd` 変種は **ESPnet `egs2/ms_snsd/enh1`**（TF-GridNet v1、HF `espnet/ms_snsd_tfgridnet`）を指す。このレシピは **2026-02-17 にコミュニティ貢献者が追加した**もので、論文の再現ではない。dns_ins20 の TF-GridNet 設定をほぼ流用している。【確認:コード】
4. `ms_snsd` のデータは **事前生成の固定ペア（オフライン混合）**。オンライン混合ではない。`preprocessor: null` のため `speech_volume_normalize`・`noise_scp`・`rir_scp` は効かない。【確認:コード】【確認:設定】
5. **MS-SNSD 生成スクリプトの `snr_mixer` は、名目 SNR の半分の SNR を作る**（`noisescalar = sqrt(rmsclean / 10^(snr/20) / rmsnoise)`）。名目 [0,10,20,30,40] dB の実 SNR は [0,5,10,15,20] dB。ローカル audiolib で数値確認した。レシピ README と cfg は「0–40 dB」と書いている。【確認:コード】（数値確認済み）
6. **公式の学習は 2 GPU の DDP**（`run.sh --ngpu 2`、HF の config は `dist_world_size: 2`）で、rank あたり batch 2、rank あたり 5,000 発話、num_cache_chunks は rank あたり 512。audio-db は単一プロセス（batch 4、5,000 発話、cache 1024）で、複数プロセスは `TrainingError`。【確認:コード】
7. **公式の `best_model_criterion` にある `valid si_snr max` は、ピン留め ESPnet では一度も働かない**。stats のキーは `si_snr_loss` であり `si_snr` ではない。実際に残るのは `valid.loss.best` と `valid.loss.ave_*` だけで、HF にも `valid.loss.best.pth` しか無い。audio-db は `valid.si_snr.best.pt`（SI-SDR の低い順に最大化）を実装しており、これは公式の挙動ではない。DECISIONS #26 と `OFFICIAL_TRAINING_AUDIT.md` はこの点が誤り。【確認:コード】
8. **公式の `init: xavier_uniform` は、独自 LayerNorm の gamma/beta まで乱数化する**（`p.dim() > 1` に当たる 4 次元パラメータのため）。HF チェックポイントの gamma は平均 0 付近・符号が混在で、1 付近ではない。audio-db の `TFGridNetTask` は初期化を適用しない（既定値 gamma=1, beta=0）。【確認:コード】＋実チェックポイント統計（間接証拠）
9. **有効な grad_clip は 5.0**。YAML に書かれず、`--grad_clip` の既定値が使われる。論文は 1.0。audio-db の ms-snsd 設定は 5.0 で公式実効値と一致。【確認:設定】
10. **VoiceBank-DEMAND での「公式 epoch」は、公式の更新数の約 1/25 しかない**（【推測】）。公式の学習クリップは ≥10 s（30 h÷5 SNR÷約 11.5 s）で 1 発話あたり約 6 チャンクできる。VoiceBank の発話は短く、`chunk_length` 48000（3 s）以下は捨てられる。既存 docs の実測値は 575 更新/epoch（再検証できていない）。
11. **MS-SNSD の clean に Edinburgh 56 speaker（= VoiceBank の音声源）、noise に DEMAND が含まれる**（MS-SNSD README の License 節）。HF チェックポイントを VoiceBank-DEMAND で評価するとき、学習データとの重複の有無は UNKNOWN。
12. 作業ツリーに `data/db/*.duckdb`、`data/staging/*`、`configs/training/tf_gridnet_chunk_seed17.yaml`、`tf_gridnet_fixed5000_seed17.yaml`、`trial100/` は無い。これらを根拠にした既存 docs の記述は再検証していない。
13. 既存の独立実装は forward/loss/更新 1 回の一致を示す記録を持つが（`docs/training/VERIFICATION.md`）、本監査では再実行していない。**「公式プロトコル一致」は未検証**。

---

## 1. 調査メタデータ

| 項目 | 値 |
|---|---|
| 監査日 | 2026-11-06 |
| audio-db | branch `feature/reproduction`、HEAD `3bee9addc0a1ff3d10d4e47a2a0eafe34f4bc110`（2026-11-06 01:26 JST）|
| ESPnet ピン | `0d136c5456d403cb2c0a4c13e76e8136e32e9b7a`（2026-10-30、PR #6703 のマージ）。`requirements/official-assets.toml` L47。直近タグ `v.202611`（2026-10-23）|
| ESPnet 履歴調査 | `/tmp/audit-tfgridnet/espnet`（blobless clone、master `66f0ff5af`）。`src/se/official/`・`build/` には git を使っていない |
| 論文 (a) | arXiv:2209.03952（ICASSP 2023）。v1 2022-09-08、v2 2023-03-15（本監査は v2 の HTML を使用）|
| 論文 (b) | arXiv:2211.12433（TASLP 2023）。v1 2022-11-22、v2 2023-08-04（v2 を使用、v1 と §V-G の訓練条件は同文）|
| 取得した外部資料 | arXiv HTML（`/tmp/audit-tfgridnet/p1.txt`, `p2.txt` 他）、HF `espnet/ms_snsd_tfgridnet`、HF `wyz/tfgridnet_for_urgent24`、MS-SNSD（ローカルの `data/corpora/MS-SNSD`、`fe61c4b` 2023-01-20）|
| 実行したこと | 読み取り（rg/sed/git log）、CPU で `audiolib.snr_mixer` の SNR 数値確認、HF チェックポイントのパラメータ統計、MS-SNSD ファイルの長さ統計（各 300 ファイル標本）|
| 実行していないこと | 学習、評価、GPU 使用、DB の読み書き（`data/db` は空）、ESPnet 公式学習の再実行、公式出力との数値比較 |

TF-GridNet のコード履歴（ESPnet）【確認:コード】:

| 日付 | commit / 出来事 | 備考 |
|---|---|---|
| 2023-01-11 | `63094c063` Zhongqiu Wang「Add local changes to new branch」 | `tfgridnet_separator.py` の初出。リリースタグ `v.202304`（2023-05-01）が最初に含む |
| 2023-01-25 | `dac353c3a` マージ | zqwang7 の `tfgridnet` ブランチ |
| 2023-04-12/14 | `3f73b7b79`, `eca0f6730` Wangyou Zhang | SE の不具合修正、fp16 対応 |
| 2023-08-04 | `869a6ee6c`「speed up TFGridNet code」 | `tfgridnetv2_separator.py` 追加（PR #5395）。マージ `3826362c7` は 2023-10-17、最初のタグ `v.202310` |
| 2023-08-13/15 | `a0702913d`, `3fecf8623` | v1 に 3 行追加 / 修正（WASPAA 再現用）。`wsj0_2mix_spatialized` の conf 追加 |
| 2024-03-20 | `f3a67e6ec` Zhaoheng Ni | `dns_ins20` に TF-GridNet レシピ追加 |
| 2024-06-05〜09 | `5208894ce` ほか Wangyou Zhang | `tfgridnetv3_separator.py`、`urgent24` レシピ（`bdb9ebe1f`）。最初のタグ `v.202409` |
| 2026-02-17 | `3515fe270` Dahee96 | `egs2/ms_snsd/enh1` 追加。最初のタグ `v.202604` |
| 2026-10-12/15 | `1840844c3` ほか ftshijt | `urgent24` を AQA タスクへ移行。ピン留めの `urgent24` レシピは HF 訓練時（espnet `202402`）と別物になっている可能性 |

---

## 2. A. 公式資料

### A-1 論文

| 論文 | 内容 | 本監査での位置づけ |
|---|---|---|
| (a) arXiv:2209.03952 ICASSP 2023「TF-GridNet: Making Time-Frequency Domain Models Great Again for Monaural Speaker Separation」 | WSJ0-2mix のみ、8 kHz。SI-SDRi 23.4 dB（動的混合なし）。コードは ESPnet-SE++ で公開と記載 [arXiv:2209.03952v2 §1]【確認:論文】 | 単一話者 SE ではない |
| (b) arXiv:2211.12433 TASLP 2023「TF-GridNet: Integrating Full- and Sub-band Modeling for Speech Separation」 | 5 タスク（WSJ0-2mix、SMS-WSJ、WHAMR!、WSJ0CAM-DEREVERB、L3DAS22）。脚注で ESPnet PR #5395 を指す [arXiv:2211.12433v2 脚注 2]【確認:論文】 | 単一話者・単一チャンネルに最も近いのは WSJ0CAM-DEREVERB 1ch（Table X）|

audio-db の `docs/se/MODELS.md:29` は論文 (b) を引用している。

### A-2 論文と ESPnet 実装の対応

- 論文 (b) §V-G の訓練条件【確認:論文】: STFT 窓 32 ms・ホップ 8 ms・平方根 Hann（16 kHz で 512 点、8 kHz で 256 点）、各 epoch で各混合から 4 秒を 1 区間サンプル、混合区間の標本分散を 1 に正規化し同じスケールを目的信号に掛ける、Adam、L2 勾配クリップ 1.0、学習率 0.001 から検証損失が 3 epoch 改善しなければ半減、動的混合・データ拡張なし。[arXiv:2211.12433v2 §V-G]
- 論文 (a) §3（実験設定）も同じ（4 秒、Adam、クリップ 1、半減 3 epoch）。[arXiv:2209.03952v2 §3]【確認:論文】
- 論文で epoch 数、early stopping、best 選択、seed、バッチサイズは **記載が見つからない**（Table XIII 脚注 (d) に「A100 40GB で epoch は 20,000 個の 4 秒区間、バッチサイズは GPU メモリをほぼ使い切る値」とだけある）。【UNKNOWN】
- ESPnet の `TFGridNet`（v1）は LN-Unfold（正規化してから unfold）である。[espnet@0d136c5 : espnet2/enh/separator/tfgridnet_separator.py : L274-L281 : GridNetBlock.forward]【確認:コード】論文 (b) は「予備論文 (a) は Unfold-LN で、本論文が LN-Unfold を提案」と書く [arXiv:2211.12433v2 §III-A]ため、v1 は (a) の式と一致せず (b) の式に近い。【確認:論文】【確認:コード】
- v1 と v2 は、入力を `torch.std` で割り出力に掛け戻す正規化を分離器の内部で行う。v3 は行わず、`model_conf.normalize_variance_per_ch` とログ側（損失の `normalize_variance`）に任せる。[espnet@0d136c5 : tfgridnet_separator.py : L141-L142, L163 / tfgridnetv2_separator.py : L146-L147, L168 / tfgridnetv3_separator.py : L39-L43（docstring のみ）]【確認:コード】

### A-3 バリアントと候補レシピ

| 候補 | 分離器 | データ | 論文実験との関係 | HF/Zenodo 重み | 備考 |
|---|---|---|---|---|---|
| ESPnet `wsj0_2mix`（v1: `train_enh_tfgridnet_tf_lr-patience3_patience5.yaml`、v2: `..._I_1_J_1_D_128.yaml`）| v1 / v2 | WSJ0-2mix 8 kHz、事前生成固定ペア | 論文 (a)(b) の WSJ0-2mix に最も近い訓練設定。ただし STFT が n_fft 128 / stride 64（論文は 256 点 / 64 サンプル）で不一致 | Zenodo 7565926（v1）、8433041（v2）[egs2/wsj0_2mix/enh1/README.md L196-L215] | 2 話者分離。単一話者 SE ではない |
| ESPnet `wsj0_2mix_spatialized` / `wsj_kinect` | v1 / v2 | 多チャンネル | 論文 (b) の主実験ではない（WASPAA 2023 など）| — | 範囲外 |
| 論文 (b) WSJ0CAM-DEREVERB | v1 相当 | 自前生成（WSJCAM0 + Pyroomacoustics + REVERB の雑音）| 単一話者 SE に最も近い（Table X: DNN1 SI-SDR 16.6、PESQ(NB) 3.72、eSTOI 0.947）| なし | ESPnet にレシピなし（`rg -il tfgridnet egs2` に出ない）。生成コードの所在は UNKNOWN |
| ESPnet `dns_ins20` | v1 | DNS Interspeech 2020 合成（100 h、SNR 0–40 dB、事前生成）| 論文実験ではない | HF `Zhaoheng/tfgridnet_dns_ins20_epoch33` | `run.sh` の既定 `--enh_config` は BLSTM（`train_enh_blstm_tf.yaml`）で、TF-GridNet は tuning の別ファイル。README は存在しない `train_enh_tfgrid.yaml` を指す |
| ESPnet `ms_snsd` | v1 | MS-SNSD 事前生成（train 30 h、test 1 h）| 論文実験ではない。2026-02 追加 | HF `espnet/ms_snsd_tfgridnet` | audio-db の `ms-snsd` 変種 |
| ESPnet `urgent24` | v3 | URGENT 2024 事前シミュレーション（外部 `urgent2024_challenge`）| 論文実験ではない | HF `wyz/tfgridnet_for_urgent24`（`21epoch.pth`）| audio-db の `urgent24` 変種。`run.sh` の既定設定は BSRNN で TF-GridNet ではない |

**`ms_snsd` の選択は妥当か（候補と差異の報告のみ）**
- 妥当な点: audio-db が現に評価しているチェックポイントの訓練レシピである。単一話者 SE である。16 kHz。データ生成が公開スクリプトで再現可能。
- 妥当でない点: 論文の訓練法（4 秒 1 区間/混合、clip 1.0、halving 3 epoch）とは別物。コミュニティ追加で `ms_snsd` の結果は README に 1 行しかない。HF モデルカードに ESPnet 版・結果・ライセンスが無い。データ生成に未シードの乱数と SNR の係数誤りがある（§4）。
- 「どれを採るか」は研究判断。論文忠実な再現を優先するなら `wsj0_2mix` + 論文 (a)(b) の手順、audio-db の既存評価系を優先するなら `ms_snsd`、という分け方になる。決めない。

---

## 3. B. 学習データセット

### B-1 `ms_snsd`（audio-db の `ms-snsd` 変種の公式学習データ）

| 項目 | 値 | 根拠 |
|---|---|---|
| 入力コーパス | MS-SNSD。`clean_train` 23,075 wav、`noise_train` 128、`clean_test` 1,100、`noise_test` 51（ローカル `ls` と `docs/db/DATASETS.md` の件数が一致）| `data/corpora/MS-SNSD`【確認:コード】|
| 形式 | 16 kHz モノ（各 300 ファイル標本で全件 16000 Hz/1ch）| 標本調査【確認:コード】|
| 長さ（標本平均）| `clean_train` 3.00 s（1.31–9.46）、`clean_test` 11.99 s（10.01–15.74）、`noise_train` 92.6 s、`noise_test` 46.5 s | 300 ファイル標本【確認:コード】|
| clean の出所 | PTDB-TUG（ODbL）、Edinburgh 56 speaker | MS-SNSD `README.md` L59-L61【確認:設定】|
| noise の出所 | Freesound（CC0 のみ）、DEMAND（CC BY-SA 3.0）| 同 L62-L64【確認:設定】|
| 混合の生成 | ESPnet `local/ms_snsd_create_mixture.sh` が `noisyspeech_synthesizer.py`（recipe 版）を train/test で実行 | [espnet@0d136c5 : egs2/ms_snsd/enh1/local/ms_snsd_create_mixture.sh : L20-L28, L93, L105, L108]【確認:コード】|
| 生成量 | train 30 h、test 1 h。ただし `num_samples` は 5 SNR の全コピーを足すので、**一意なクリップは 6 h / 0.2 h** | [同 L20-L21; ms_snsd_noisyspeech_synthesizer.py : L67, L125]【確認:コード】|
| SNR 格子 | `np.linspace(0, 40, 5)` = [0,10,20,30,40]（名目）。全ペアが 5 SNR すべてで作られる | [同 L50, L102-L125]【確認:コード】|
| 実 SNR | **名目の半分**（[0,5,10,15,20] dB）。数値確認済み | §4-5、`[MS-SNSD : audiolib.py : L55-L71 : snr_mixer]`【確認:コード】|
| クリップ長 | 最小 10 s（`audio_length`）。短い発話は 0.2 s の無音を挟み連結 | [synthesizer : L71-L82]【確認:コード】|
| 分割（train / valid）| 生成した train 側の noisy をファイル名順に 80/20。`noisyN_` 単位でまとまるので、同じクリップ N の 5 SNR は同じ側に入る | [ms_snsd_data_prep.sh : L82-L93]【確認:コード】|
| test | 別の `clean_test`/`noise_test` から別途生成 | [同 L96-L97]【確認:コード】|
| 件数（実際）| 乱数が未シードのため **実際の生成件数は UNKNOWN**。推定: train 約 1,800 クリップ ×5 SNR ≒ 9,000 noisy、valid はその約 20%、test は約 60 クリップ ×5 SNR ≒ 300 noisy（【推測】: 平均長 11.5 s と仮定）| §4【推測】|
| 再現性 | `np.random.randint` を seed していない。clean/noise の選択・連結・ランダム再抽選が毎回変わる。HF チェックポイントの実データは再現できない | [synthesizer : L68, L79, L84, L95]【確認:コード】|
| 参照信号 | `spk1.scp` = クリップごとの `clnspN.wav`、`noise1.scp` = SNR ごとの noise。`clnspN.wav` は 5 SNR で同一内容を上書き（`clean_snr` は SNR に依らない）| [synthesizer : L115, L123; audiolib.py : L55-L61]【確認:コード】|
| 漏れ（リーク）| train/valid 間でクリップは分かれるが、clean 発話と noise ファイルは同じプールから抽選するため **発話・雑音の重複が起こり得る** | 【確認:コード】（構造）。重複量は UNKNOWN |

### B-2 他の候補データ（簡略）

- `dns_ins20`: DNS Interspeech 2020 のスクリプトで 100 h を生成、SNR 0–40 dB、9:1 分割（`tr_synthetic` / `cv_synthetic`）、test は DNS 合成 `with_reverb` / `no_reverb` [espnet@0d136c5 : egs2/dns_ins20/enh1/local/dns_create_mixture.sh : L18-L20; dns_data_prep.sh : L67-L72]。【確認:コード】DNS 側の SNR の係数は未調査（UNKNOWN）。
- `urgent24`: 外部 `urgent2024_challenge/prepare_espnet_data.sh` で `data/` を作る。WSJ0/WSJ1/CommonVoice 11.0 などを使う [egs2/urgent24/enh1/local/data.sh : L40-L56, L97-L103]。SNR 分布やシミュレーション手順はこのツリーに無い（UNKNOWN）。
- `wsj0_2mix`: 20,000 / 5,000 / 3,000 混合、8 kHz [論文 (a) §4 / (b) §V-A]。

### B-3 audio-db 側の学習データ（現状）

- `configs/training/tf_gridnet.yaml` は VoiceBank-DEMAND の 28 話者分割（`voicebank_28_train` 10,700 件 / `voicebank_28_valid` 872 件）を使う [configs/training/tf_gridnet.yaml : L68-L70]。件数は既存 docs の値で、`data/db/voicebank.duckdb` が無いため再確認できない。【確認:設定】件数は【UNKNOWN】（再検証不能）
- MS-SNSD の事前生成は `src/reproduction/build/ms_snsd.py` にある。登録済みの学習入口は小規模 Dataset `tf_gridnet.ms_snsd.smoke@1` で、公式規模の 30 h / 1 h とは別 Build である。状態は [P6_VERIFICATION](P6_VERIFICATION.md)。
- **重複リスク**: MS-SNSD の clean に Edinburgh 56 speaker、noise に DEMAND が入っている。VoiceBank-DEMAND は同系の音声と雑音から作られた対だが、同一ファイルが含まれるかは、ファイル名が変わっているため不明。【UNKNOWN】（確認案: 16 kHz へ変換した波形の指紋照合）

---

## 4. C. データ生成・前処理

### C-1 `ms_snsd` の処理順（公式、番号は実行順）

1. `local/data.sh` stage 1 → `ms_snsd_create_mixture.sh`: train と test を別々に生成。`rm -r data/` で既存データを消す。[local/data.sh : L49-L56; ms_snsd_create_mixture.sh : L30]【確認:コード】
2. cfg を書く: `sampling_rate 16000`、`audio_length 10`、`silence_length 0.2`、`total_snrlevels 5`、`snr_lower 0`、`snr_upper 40`。`sampling_rate` は `float(cfg["sampling_rate"])` を読むだけで、**リサンプルには使われない**（`audioread` が返す `fs` が使われる）。[ms_snsd_create_mixture.sh : L58-L71; synthesizer : L32, L69]【確認:コード】
3. clean を `np.random.randint` で選ぶ。`audioread(..., norm=True)` が **各ファイルを -25 dBFS RMS に正規化して読む**。10 s 以下なら次のファイル（`idx_s+1`、末尾なら再抽選）を 0.2 s の無音を挟んで足し、10 s を超えるまで続ける。[synthesizer : L68-L82; audiolib.py : L12-L26]【確認:コード】
4. noise を `randint` で選び、同様に `norm=True` で読む。clean より長ければ先頭から切り、短ければ次のファイルを無音挟みで足す（`noise[0:len(clean)]`、ランダムなオフセットは無い）。[synthesizer : L84-L99]【確認:コード】
5. 5 SNR それぞれで `snr_mixer`: clean と noise を再度 -25 dBFS RMS に揃え、`noisescalar = sqrt(rmsclean / 10^(snr/20) / rmsnoise)` を noise に掛けて足す。**平方根のため実 SNR = 名目 / 2**。クリップ防止のゲイン処理は無い。[audiolib.py : L55-L71]【確認:コード】
6. `audiowrite(..., norm=False)` で `sf.write` する（WAV の既定は PCM_16 と【推測】。float を直接書くときの ±1 超過時の挙動は未確認）。noisy・clean・noise を `NoisySpeech_training` / `CleanSpeech_training` / `Noise_training` へ。[synthesizer : L122-L124; audiolib.py : L37-L52]【確認:コード】/【UNKNOWN】（クリップ挙動）
7. ESPnet データ準備: noisy の `.wav` 一覧をソートし、先頭 80% を `tr_ms_snsd`、残りを `cv_ms_snsd`。ファイル名から `clnspN` と noise 名を取り `spk1.scp`・`noise1.scp` を作る。test は `tt_ms_snsd`。[ms_snsd_data_prep.sh : L45-L97]【確認:コード】
8. `enh.sh` stage 3/4: `audio_format=flac`（既定）、`fs=16k`、`min_wav_duration 0.1`、`max_wav_duration 31`（`run.sh` が 31 に上書き）で `dump/raw` を作る。16 kHz 16-bit の無損失エンコードで、リサンプルは無い（入力が既に 16 kHz）。[egs2/TEMPLATE/enh1/enh.sh : L50-L53; ms_snsd/enh1/run.sh : L8, L28]【確認:コード】
9. 学習時: `preprocessor: null` なので前処理なし。`speech_volume_normalize`・`noise_scp`・`rir_scp`・`noise_db_range`（HF config の `13_15`）は **すべて未使用**。[espnet@0d136c5 : espnet2/tasks/enh.py : L416, L418, L475; HF:ms_snsd : L186-L195, L228]【確認:コード】【確認:設定】
10. チャンク化 → `CommonCollateFn(float_pad_value=0.0)`。波形そのまま（レベル正規化は生成時の -25 dBFS のみ）。[espnet@0d136c5 : espnet2/tasks/enh.py : L408 : build_collate_fn]【確認:コード】
11. モデル内部: v1 分離器が入力を `torch.std` で割り、出力に掛け戻す。`model_conf.normalize_variance` は False。[tfgridnet_separator.py : L141-L142, L163; HF:ms_snsd : L161]【確認:コード】

### C-2 ESPnet の既存動的混合機能（このレシピでは未使用）

`EnhPreprocessor` は、`rir_apply_prob`・`noise_apply_prob`・`noise_db_range`（既定 `3_10`、README 系の YAML 既定は `13_15`）・`speech_volume_normalize`・`data_aug_*` を持つ。`preprocessor: enh` のとき `kwargs.update(args.preprocessor_conf)` で上書きされる。[espnet@0d136c5 : espnet2/train/preprocessor.py : L1073-L1160; tasks/enh.py : L443-L475]【確認:コード】`urgent24` は `preprocessor: enh` だが `rir_scp`/`noise_scp` が null のため、実効はチャンネル処理（`force_single_channel`、`channel_reordering`）のみ。[HF:urgent24 : L172-L184, L225-L226]【確認:設定】

---

## 5. D. Dataset/DataLoader/Sampler

### D-1 公式（`ms_snsd`）

- `iterator_type: chunk`、`valid_iterator_type: null`（学習と同じ chunk）。[HF:ms_snsd : L6-L7]
- 学習: `UnsortedBatchSampler(batch_size=1)` で発話単位のバッチを作り、DDP では `batches[rank::world_size]` に分割、per-rank `batch_size = 4 // 2 = 2`、`num_cache_chunks = 1024 // 2 = 512`。[espnet@0d136c5 : espnet2/tasks/abs_task.py : L2159-L2186]【確認:コード】
- `num_iters_per_epoch` はチャンク iterator では **発話数**（`num_samples_per_epoch`）として渡される。[abs_task.py : L2193-L2196]【確認:コード】
- 発話選択: `SequenceIterFactory.build_iter` が、`num_iters_per_epoch` < N なら 1 周期の shuffle 列を切り出し、≥ N なら周回して連結する。[espnet@0d136c5 : espnet2/iterators/sequence_iter_factory.py : L79-L133]【確認:コード】
- チャンク化: 幅 W は `chunk_lengths` から `state.choice`（1 候補でも乱数を消費）、`S = int(W*0.5)`、`N = (L-W)//S + 1`、`Z = state.randint(0, (L-W)%S + 1)`（学習時のみ）。`L ≤ W` は捨てる。キャッシュが `num_cache_chunks` を超えたら shuffle してバッチ化し、端数は持ち越し、epoch 末の端数バッチ（バッチサイズ未満）は捨てる。[chunk_iter_factory.py : L140, L184-L193, L209-L220, L243-L249, L275-L305]【確認:コード】
- 検証: `build_iter_options` の valid は `num_iters_per_epoch=None`、`batch_size = args.batch_size`（`valid_batch_size` null）、`shuffle=False`（`train=False`）。DDP でも rank 分割される。[abs_task.py : L1710-L1738, L1817-L1820, L2165-L2183, L2197]【確認:コード】

### D-2 epoch の定義の比較

| 系 | 1 epoch の単位 | 値 | 1 epoch の更新数 | scheduler の刻み | 根拠 |
|---|---|---|---|---|---|
| 論文 (a)(b) | 各混合から 4 秒を 1 区間 | WSJ0-2mix で 20,000 区間 | 値は書かれていない（【UNKNOWN】）| 検証損失を epoch ごとに見る | [arXiv:2211.12433v2 §V-G, Table XIII 脚注 (d)]【確認:論文】|
| ESPnet `wsj0_2mix` | 全発話を 1 回（`num_iters_per_epoch` 無指定）。チャンク幅 32000（4 s）、shift 0.5、発話ごとに複数チャンク | 論文の「1 混合 1 区間」と一致しない | 可変 | epoch ごと（val loss）| [egs2/wsj0_2mix/enh1/conf/tuning/train_enh_tfgridnet_tf_lr-patience3_patience5.yaml L5-L7, L25-L29; v2 yaml L1-L29]【確認:設定】|
| ESPnet `ms_snsd`（実効）| **rank あたり** 5,000 発話 → 2 rank で 10,000 発話 | 5,000×2（実際は N/2 の rank 分割を周回）| 可変。1 発話 約 6 チャンク（11.5 s と仮定）→ rank あたり約 15,000 更新（【推測】）| epoch ごと（val loss）| [HF:ms_snsd : L15, L79-L80, L109-L111; abs_task.py : L2173-L2183] |
| ESPnet `urgent24` | 8,000 発話（HF では `dist_world_size: 4`、`ngpu: 1` の記録）| 実効 world size は UNKNOWN | 可変 | epoch ごと（steplr）| [HF:urgent24 : L9, L15, L71-L72]【確認:設定】|
| audio-db `chunk_iterator`（`tf_gridnet.yaml`）| 1 プロセスで 5,000 発話 | 公式 10,000 の半分 | VoiceBank で 約 575（既存 docs の値、再検証不能）| epoch ごと | [configs/training/tf_gridnet.yaml L28-L36; src/training/core/trainer.py L346-L356]【確認:設定】|
| audio-db 固定 5,000 更新 | 論理 epoch = 固定更新数 | 設定ファイルが作業ツリーに無い | 5,000 | — | `configs/training/` に `tf_gridnet_fixed5000_seed17.yaml` が無い【UNKNOWN】|

### D-3 audio-db の Dataset/DataLoader

- `RecipeDataset._load` は、レシピを `render_recipe` で描画し、モノ化 → 16 kHz へ `torchaudio.functional.resample`（長さ検証あり）→ 任意で RMS 正規化 → 学習なら 1 区間を切り出す（チャンクモードは `crop=False` で全長を読んでからスライス）。[src/training/data/dataset.py : L181-L223, L331-L347, L359-L364, L158-L179]【確認:コード】
- `chunk_plan.chunk_batches`/`select_utterances`/`epoch_chunk_batches` が上記の ESPnet iterator を単一プロセスで再現する。`length <= chunk_length` を捨てる、`choice`→`randint` の順、flush は `len(cache) > max(cache_size, batch_size)`。[src/training/data/chunk_plan.py : L19-L56, L59-L80, L83-L110]【確認:コード】
- 公式 iterator との一致試験は、合成長 `[8,13,17,25,10,29]` と `[9]*7` の 1 プロセスのみ。DDP（rank 分割・cache/world）、実データ長、`num_iters_per_epoch` ≥ N の周回での本番規模は含まない。[tests/training/test_chunk_official.py : L33-L47, L57-L65]【確認:コード】

---

## 6. E. モデル・学習条件（設定値 vs 実効値）

| 項目 | 設定値（YAML）| 実効値 | 根拠 |
|---|---|---|---|
| Optimizer | `adam`、`lr 1e-3`、`eps 1e-8`、`weight_decay 0` | betas は PyTorch 既定 (0.9, 0.999)。YAML に無い | [egs2/ms_snsd/enh1/conf/tuning/train_enh_tfgridnet.yaml L1, L10-L13; HF:ms_snsd L144-L148]【確認:設定】|
| Scheduler | `reducelronplateau`、`mode min`、`factor 0.7`、`patience 1` | PyTorch 既定の `threshold 1e-4`（relative）、`cooldown 0`、`min_lr 0`。epoch ごとに valid loss で step | [同 yaml L26-L30; HF:ms_snsd L36-L38, L149-L153; espnet@0d136c5 : espnet2/train/trainer.py L377-L384]【確認:設定】【確認:コード】|
| 勾配クリップ | YAML に無い | **5.0**（`--grad_clip` 既定）、L2 ノルム | [abs_task.py L615-L619; HF:ms_snsd L52-L53]【確認:コード】|
| 勾配が非有限 | — | 更新をスキップ | [trainer.py L731-L742]【確認:コード】|
| accum_grad | 無指定 | 1 | [HF:ms_snsd L55]|
| バッチ | `batch_size 4`（`#2` の注記あり）、`batch_type folded` | DDP 2 rank で rank あたり 2、グローバル 4 チャンク | [yaml L4-L5; abs_task.py L2173-L2176; HF:ms_snsd L15, L80, L99]|
| チャンク | `chunk_length 48000`（3 s）、`chunk_shift_ratio 0.5`（HF）、`num_cache_chunks 1024`（HF）| rank あたり cache 512。`discard_short_samples true` | [yaml L7; HF:ms_snsd L109-L115; abs_task.py L2177]|
| 初期化 | `init: xavier_uniform` | `dim > 1` の全パラメータに `xavier_uniform_`、`.bias` かつ 1 次元はゼロ、`LayerNorm/GroupNorm/Embedding` はモジュール既定に戻す。**独自 `LayerNormalization4D/4DCF` の gamma/beta（4 次元）は xavier で乱数化される**（モジュール既定に戻されない）| [espnet2/torch_utils/initialize.py L36-L64; tfgridnet_separator.py L348-L393; tasks/enh.py L558]【確認:コード】＋ HF チェックポイント: `blocks.0.intra_norm.gamma` 平均 0.012、標準偏差 0.233、最小 -0.50、最大 0.50（`beta` は 0 付近）。1 付近ではない【確認:コード】（実重みの統計は間接証拠）|
| seed | `seed: 0`（HF）| `cudnn_deterministic true`、`cudnn_benchmark false`、`use_tf32 false`、`use_amp false` | [HF:ms_snsd L10, L28-L31, L58-L59]|
| max_epoch / patience | 35 / 5 | early stopping は `epoch - best_epoch > patience`、基準は valid loss min | [yaml L3, L14; HF:ms_snsd L34-L42; reporter.py L389-L412; trainer.py L524-L529]|
| best / nbest | `keep_nbest_models 5`、`best_model_criterion: [valid si_snr max], [valid loss min]` | **`valid si_snr` は無視される**（キーが無い）。実際は `valid.loss` のみ（`has()` で除外）。最後に `valid.loss.ave_5best.pth` を作る。推論は `valid.loss.best.pth` | [trainer.py L446-L448, L487-L491, L538-L542; average_nbest_models.py L46-L49; reporter.py L414-L421; time_domain.py L255 `si_snr_loss`; wrappers/fixed_order.py L41; tasks/enh.py L159]【確認:コード】|
| 再開 | `resume true` | epoch 末の `checkpoint.pth`（model・optimizer・scheduler・reporter）| [HF:ms_snsd L57; trainer.py L141-L168, L221-L223]|
| 損失 | `mr_l1_tfd` 重み 1.0 + `si_snr` 重み 0.0 | 学習損失は `mr_l1_tfd` のみ。窓 [256,512,768,1024]、hop=窓/2、`time_domain_weight 0.5`、`reduction "sum"`（時間・周波数方向の総和）、`normalize_variance` はコメントアウト（False）、STFT は矩形窓（`window=None`）。推定に最適スケーリング係数を掛けてから計算。stats キーは `l1_timedomain+magspec_loss`・`si_snr_loss`・`loss` | [yaml L50-L68; time_domain.py L379-L452, L471-L514 (`reduction` 既定 "sum"、`name` は L452)]【確認:コード】|
| 分離器 | v1 `tfgridnet`、`n_fft 512`、`stride 256`（32 ms/16 ms）、`window hann`（torch の周期 Hann）、`n_layers 4`、`lstm_hidden_units 128`、`attn_n_head 4`、`attn_approx_qk_dim 512`、`emb_dim 32`、`emb_ks 4`、`emb_hs 4`、`prelu`、`eps 1e-5` | `E = ceil(512/257) = 2`。チェックポイントは 362 テンソル・2,552,790 パラメータ（確認済み）。入力の標準偏差による正規化を内部で実施 | [yaml L31-L48; tfgridnet_separator.py L56-L167, L217-L219]【確認:設定】【確認:コード】|
| 論文との差 | 論文: 窓 32 ms・ホップ **8 ms**・**平方根 Hann**、B=4（SMS-WSJ/WHAMR!/DEREVERB）、H=192、D=48（SMS-WSJ/WSJ0CAM）、I=4、J=2（16 kHz）| ESPnet の STFT は周期 Hann で平方根 Hann の選択肢が無い（`window` は `torch.<name>_window`）| [arXiv:2211.12433v2 §V-G, §VI-B, §VI-C; espnet2/layers/stft.py L34-L36, L75-L77]【確認:論文】【確認:コード】|

HF の `urgent24`（v3）の設定値は別で、`n_layers 6`、`lstm_hidden_units 200`、`emb_dim 48`、`emb_hs 1`、`weight_decay 1e-5`、`steplr(2, 0.99)`、`grad_clip 1.0`、`patience 40`、`keep_nbest_models 1`、`batch_size 4`、`chunk_length 200`（4 s、`chunk_default_fs 50`）、`chunk_max_abs_length 100000`、`normalize_variance true`（損失・モデルとも）、`init: null`、`allow_multi_rates true`、`preprocessor: enh` [HF:urgent24 : L29-L47, L68-L96, L131-L140, L172-L185, L225-L226]。ピン留めの `egs2/urgent24/enh1/conf/tuning/train_enh_tfgridnet.yaml` はこれとほぼ同じで、`chunk_max_abs_length` がコメントアウトされている。【確認:設定】

---

## 7. F. Validation/Test/評価

### F-1 検証（公式 `ms_snsd`）

- データは `cv_ms_snsd`（生成した train 側の 20%）。チャンク化（3 s、shift 1.5 s、shuffle なし）、バッチ rank あたり 2、損失のみ。PESQ/STOI は計算しない。【確認:コード】
- best は valid loss min（`si_snr` は無視、§6）。

### F-2 テスト・評価の比較

| 項目 | 論文 (b) | ESPnet 公式スクリプト | `src/evaluation`（audio-db）|
|---|---|---|---|
| 入力 | 各タスクのテストセット（WSJ0CAM-DEREVERB は自作の 3,262 混合）| `tt_ms_snsd`（1 h 生成、未シード）| DuckDB の名前付き評価セット。評価時に Manifest を読まない（AGENTS.md）|
| SI-SDR | SI-SDR/SI-SDRi | `SISNRLoss`（`zero_mean=True`、`clamp_db=None`）を負にして報告 [enh_scoring.py L209-L214; time_domain.py L236-L256, L286-L297] | `score_si_sdr`: 平均除去、clamp なし、f64、無音は `na` [sisdr.py L11-L55] |
| SDR | SDR/SDRi | `mir_eval.bss_eval_sources`（SDR/SAR/SIR）[enh_scoring.py L200-L202] | 実装なし（SDR は評価対象外）|
| STOI | pystoi v0.3.3、eSTOI | `pystoi.stoi`（STOI と ESTOI を 2 本、×100）、サンプリングレートそのまま [enh_scoring.py L205-L208, L269-L270] | `pystoi`、`extended` 指定可。STOI の警告で `na`（`insufficient_active_speech`）[stoi.py L28-L74] |
| PESQ | python-pesq v0.0.2、**狭帯域（MOS-LQO）** | `--use_pesq` 既定 **False**。有効時は 8 kHz で `nb`、16 kHz で `wb`、16 kHz 超は 16 kHz へ変換して `wb`。`pesq` パッケージ版は不明 [enh_scoring.py L110-L118, L224-L268, L371-L375] | `pesq==0.0.4`、16 kHz で `wb`、それ以外は `nb`、境界長ガード（1 binのズレで 0.1 超なら `na`）[pesq.py L45-L100, L124-L142] |
| WER | SMS-WSJ は Kaldi、L3DAS22 は wav2vec2（STOI と合成）| ESPnet の `score_with_asr`（既定 false）| 固定 ASR で主指標 |
| 報告値 | WSJ0-2mix 23.4 dB SI-SDRi。WSJ0CAM-DEREVERB 1ch DNN1: SI-SDR 16.6 dB、PESQ(NB) 3.72、eSTOI 0.947 [(b) Table X] | `ms_snsd` README: `enhanced_cv_ms_snsd` STOI 88.49 / SDR 20.16 / SI_SNR 19.90、`enhanced_tt_ms_snsd` STOI 96.01 / SDR 17.84 / SI_SNR 17.52（PESQ なし）[egs2/ms_snsd/enh1/README.md L120-L123]。環境は espnet2 202511、PyTorch 2.10、git `239f5166a` [同 L109-L113] | 未実施 |

- `cv` の STOI（88.49）が `tt`（96.01）より低い点は、cv の構成（train 由来で同一プールの clean/noise）と tt の違いが原因の可能性があるが、原因は UNKNOWN。
- PESQ は論文が NB、audio-db が 16 kHz で WB のため、**論文の PESQ 値と audio-db の PESQ 値は直接比べられない**。ESPnet の `ms_snsd` は PESQ を報告していない。
- 推論: ESPnet は全長で推論（`segment_size` 無指定）、`normalize_output_wav False`（出力のピーク正規化なし）。`model_conf.normalize_variance(_per_ch)` が True のモデルは推論時にも標準偏差で割る（v1 は分離器内部）。[espnet2/bin/enh_inference.py L109-L114, L252-L262, L281, L394-L400]【確認:コード】

---

## 8. 現行 audio-db との項目別比較表

判定: MATCH / PARTIAL / DIFF / UNKNOWN / N/A。「公式」は `ms_snsd`（v1）。論文 (b) との差は §11。

| 項目 | 公式設定 | 公式実効動作 | audio-db現状 | 判定 | 根拠 |
|---|---|---|---|---|---|
| Dataset | MS-SNSD（train 30 h・test 1 h を生成）| 生成物は未シードで再現不能。実 SNR は名目の半分 | VoiceBank-DEMAND 28 話者（train 10,700 / valid 872 は docs の値）。MS-SNSD の学習セットは無い | DIFF | HF:ms_snsd L116-L135; `configs/training/tf_gridnet.yaml` L68-L70 |
| mixing mode | 事前生成の固定ペア | 同左 | 事前に収録済みの noisy/clean 対（レシピは操作なしの paired）| PARTIAL（方式は同じ種類、データが別）| DECISIONS #26 / `docs/db/RECIPE.md` L38 |
| noise selection | 一様乱数で 1 ファイル（短ければ連結）| 未シード | なし（収録済み noisy）| N/A | synthesizer L84-L99 |
| SNR | 名目 [0,10,20,30,40] を全ペアで | 実 [0,5,10,15,20] dB | 収録値（推定しない）| DIFF | audiolib L68; `docs/db/CORPORA.md` L28 |
| gain | 各ファイルを -25 dBFS RMS に正規化して混合 | 同左 | `rms_normalize: false`（収録レベルのまま）| DIFF | audiolib L12-L26, L55-L71; `tf_gridnet.yaml` L73 |
| reverb / RIR | なし | なし（`rir_scp` null）| なし | MATCH | HF:ms_snsd L187-L188 |
| resampling | なし（16 kHz のまま）| FLAC に無損失で保存 | 48 kHz 系の VoiceBank を 16 kHz へ `torchaudio.functional.resample`（読み込み時）。元レートは未確認 | DIFF（推測を含む）| dataset.py L331-L347; 元レートは UNKNOWN |
| normalization（入力）| `speech_volume_normalize` 無効 | v1 分離器内部で標準偏差正規化 | RMS 無効（設定）、分離器は引用した公式ブロック | MATCH（モデル内部は未検証）| tfgridnet_separator.py L141-L142 |
| crop | チャンク 48000、shift 0.5、`Z` ランダム、`L ≤ W` 捨てる | 同左 | 同じ計画（`chunk_batches`）| PARTIAL（単一プロセス・合成長の試験のみ）| chunk_plan.py L19-L56; test_chunk_official.py |
| padding | なし（チャンクは全長が W 超のみ）| バッチは同長チャンク | `_segment` はパディングするが、チャンクモードは使わない | MATCH | chunk_iter_factory L184-L193 |
| length limit | `max_wav_duration 31`、`min 0.1` | 生成クリップ ≥ 10 s、test は ≤ 約 16 s | なし | UNKNOWN | enh.sh L52-L53; run.sh L28 |
| shuffle / sampler | `RandomState(epoch + seed)`、cache 1024（DDP で 512）| DDP で rank 分割後に各 rank が独立に周回 | 同じ式、1 プロセス・cache 1024 | PARTIAL | abs_task.py L2177-L2183; chunk_plan.py L31 |
| epoch definition | rank あたり 5,000 発話（2 rank で 10,000）| 更新数は可変（約 15,000/rank と推定）| 5,000 発話、更新数 約 575（docs）| DIFF | §5 D-2 |
| batch | グローバル 4（rank あたり 2）| 同左 | 4（単一）| PARTIAL（グローバル値は同じ）| abs_task.py L2173-L2176; trainer.py L346-L349 |
| DDP / accumulation | 2 rank、accum 1 | 勾配平均 | 単一プロセス、accum 1 | DIFF | trainer.py L348-L356 |
| loss | `mr_l1_tfd` ×1.0 + `si_snr` ×0 | `mr_l1_tfd` のみ | `multi_resolution_l1`（窓・hop・時間重み・スケーリング係数・合計方式が同じ）| MATCH（式の対応を目視で確認。数値一致は未再検証）| time_domain.py L471-L514; spectral.py L20-L59 |
| optimizer | Adam 1e-3、wd 0 | betas 既定、eps 1e-8 | Adam 同値（betas 0.9/0.999）| MATCH | optim.py L20-L32; HF:ms_snsd L144-L148 |
| scheduler | ReduceLROnPlateau 0.7 / patience 1 | threshold 1e-4（rel）、epoch ごと | `ReduceLROnPlateau(threshold 1e-4, rel)` | MATCH | optim.py L161-L169, L182-L187 |
| grad clip | YAML 無し | 5.0、非有限なら更新をスキップ | 5.0。非有限勾配でもスキップせず `step()` | PARTIAL | optim.py L191-L194; trainer.py L740-L742 |
| init | `xavier_uniform` | 独自 LayerNorm の gamma/beta まで乱数化 | 適用しない（`seed_model_init` のみ）| DIFF | tf_gridnet.py L42-L47; initialization.py（TF-Locoformer のみが呼ぶ）|
| max epoch / step | 35 epoch | early stop（patience 5）| `max_epochs 35`、`max_steps` は安全上限 | MATCH | tf_gridnet.yaml L19, L34 |
| validation data | `cv_ms_snsd`（train 側 20%）| チャンク、3 s、shuffle なし | VoiceBank valid 872、同じチャンク計画 | PARTIAL | trainer.py L443-L475 |
| validation preprocessing | なし | なし | `rms_normalize: false` | MATCH | tf_gridnet.yaml L73 |
| validation batch | rank あたり 2 | 同左 | 4（単一）、rank 0 のみ | PARTIAL | trainer.py L443-L454 |
| best selection | valid loss min と `si_snr` max | **`si_snr` は無視。loss のみ** | loss と、SI-SDR の低い順の `si_snr` 最大を別に保存 | DIFF | trainer.py L495-L525; trainer.py(ESPnet) L446-L448 |
| early stopping | patience 5、valid loss | `epoch - best > patience` | 同じ式 | MATCH | trainer.py L477-L493; reporter.py L404 |
| resume | epoch 末 | 途中 iterator は保証しない | 途中再開も可（計画を再生成）| DIFF（意図的な追加）| 既存 docs |
| metrics: PESQ | 既定では計算しない（`--use_pesq`）| — | `pesq==0.0.4`、WB（16 kHz）| DIFF | enh_scoring.py L371-L375; pesq.py L45-L100 |
| metrics: STOI | ×100、native rate | — | pystoi、`extended` 任意 | PARTIAL | enh_scoring.py L205-L208; stoi.py |
| metrics: SI-SDR | `SISNRLoss`（zero_mean）| — | 平均除去、f64 | MATCH（式。数値一致は未確認）| enh_scoring.py L209-L214; sisdr.py |
| test | `tt_ms_snsd`（1 h、5 SNR）| 未シード生成 | DuckDB の評価セット（`ms-snsd` 他）。ms-snsd 評価セットの構成は別途 | UNKNOWN | README L120-L123 |

---

## 9. 差異一覧

状態の語: 「audio-dbに未実装」「実装済みだが公式一致は未検証」「実装済みで検証済み(範囲明記)」。

| ID | 項目 | 差異 | 影響 | 状態 | 優先度 | 対象ファイル / モジュール |
|---|---|---|---|---|---|---|
| TFG-D01 | 学習データ | 公式は MS-SNSD 生成物、audio-db は VoiceBank 28 話者。MS-SNSD を使う学習セットと生成器が無い | Critical | audio-dbに未実装 | P0 | `configs/training/tf_gridnet.yaml`、`src/data/training_sets.py`、新規の再現用生成器 |
| TFG-D02 | 1 epoch の規模 | 公式は約 15,000 更新/rank/epoch（推定）、VoiceBank は約 575 更新/epoch（docs）。`chunk_length` 3 s が短い発話を捨てる | Critical | 実装済みだが公式一致は未検証（規模は未実測）| P0 | `src/training/data/chunk_plan.py`、`src/training/core/trainer.py` |
| TFG-D03 | MS-SNSD 生成の SNR | `snr_mixer` が名目の半分の SNR を作る。README/cfg は「0–40 dB」。audio-db のレシピ `snr_db` は真の SNR（`rms_v1`）| Major | audio-dbに未実装 | P0 | `src/evaluation/audio/mixing.py`、レシピスキーマ |
| TFG-D04 | 生成の再現性 | 未シード乱数、80/20 のファイル名順分割で clean/noise 重複の可能性 | Major | audio-dbに未実装 | P0 | 再現用生成器、`data/staging` の保存方針 |
| TFG-D05 | 初期化 | 公式は `xavier_uniform`（独自 LayerNorm の gamma/beta も乱数化）、audio-db は既定初期化 | Major | audio-dbに未実装 | P1 | `src/training/se/tf_gridnet.py`、`src/training/se/initialization.py` |
| TFG-D06 | DDP | 公式は 2 rank（rank あたり batch 2、cache 512、rank 分割）、audio-db は単一プロセス | Major | audio-dbに未実装（複数プロセスは `TrainingError`）| P1 | `src/training/core/trainer.py` L346-L349、`src/training/data/chunk_plan.py` |
| TFG-D07 | best の選択 | 公式で `si_snr` は無視されるのに、audio-db は `valid.si_snr.best.pt` を保存（SI-SDR の低い順）。既存 docs の記述が誤り | Major | 実装済みだが公式一致は未検証（公式の挙動と食い違う）| P1 | `src/training/core/trainer.py` L495-L525、`configs/training/tf_gridnet.yaml` L7-L9 |
| TFG-D08 | 勾配が非有限のとき | 公式は更新スキップ、audio-db は `clip_grad_norm_` の戻り値を見ずに更新 | Minor | audio-dbに未実装 | P2 | `src/training/se/optim.py` L191-L194 |
| TFG-D09 | レベル正規化 | 公式データは各ファイルを -25 dBFS RMS に揃えて混合。audio-db は VoiceBank の収録レベルで `rms_normalize: false`。`mr_l1_tfd` は目的信号のレベルに依存（`sum` 還元）| Major | audio-dbに未実装 | P1 | レシピ操作（RMS 正規化が無い）、`dataset.py` |
| TFG-D10 | サンプリング | 公式は 16 kHz のまま FLAC、audio-db は 48 kHz 系を読み込み時に変換（元レートは未確認）| Major | 実装済みだが公式一致は未検証 | P1 | `src/training/data/dataset.py` L331-L347 |
| TFG-D11 | 検証データ/指標 | 公式 valid は MS-SNSD の cv 分割、audio-db は VoiceBank valid。検証中の PESQ/STOI は `na` | Minor | 実装済みだが公式一致は未検証 | P2 | `src/training/se/tf_gridnet.py` L117-L139 |
| TFG-D12 | 評価指標の実装 | PESQ は論文が NB、audio-db が WB（16 kHz）。SDR は audio-db に無い。STOI は拡張版の有無が違う | Major | 実装済みだが公式一致は未検証 | P1 | `src/evaluation/metrics/pesq.py`、`stoi.py` |
| TFG-D13 | seed / 決定性 | 公式 seed 0・`cudnn_deterministic true`、audio-db は seed 42・`deterministic false` | Minor | 実装済みだが公式とは別の値（一致は未検証）| P2 | `configs/training/tf_gridnet.yaml` L15-L16 |
| TFG-D14 | 論文忠実プロトコル | 4 秒 1 区間/混合、clip 1.0、halving 3 epoch、平方根 Hann、8 ms ホップ。audio-db には対応する設定・データが無い | Major | audio-dbに未実装 | P2（研究判断待ち）| 新規レシピ、`src/se/models/tf_gridnet` の STFT 窓 |
| TFG-D15 | `urgent24` 変種の学習 | チャンク多レート（`chunk_default_fs 50`、`allow_multi_rates`）、`weight_decay 1e-5`、`steplr(2, 0.99)`、`patience 40`、`clip 1.0`、カテゴリ付き。audio-db にはその設定が無い | Major | audio-dbに未実装 | P2 | `configs/training/`、`src/training/data/chunk_plan.py`（単一カテゴリ・単一レートのみ）|
| TFG-D16 | 既存 docs の誤り | §11 参照 | Major | — | P1 | `.agents/DECISIONS.md` #26、`docs/training/OFFICIAL_TRAINING_AUDIT.md`、`docs/training/VERIFICATION.md` |
| TFG-D17 | 評価データの重複 | MS-SNSD の clean/noise に Edinburgh 56 speaker・DEMAND が含まれる。VoiceBank-DEMAND 評価との重複は未確認 | Unknown | — | P1 | 評価セットの文書、重複検査 |

---

## 10. DB中心の新データ基盤に必要な情報

前提: 生音声は FS、索引・分割・生成条件・プロトコルは DuckDB（`data/db/reproduction.duckdb` 案）。公式が事前生成なら固定ペアを保持し、オンラインなら再現する。

**1. 登録すべき音声素材**
- `ms_snsd`: `clean_train` 23,075、`noise_train` 128、`clean_test` 1,100、`noise_test` 51（16 kHz モノ）。各ファイルの SHA-256、フレーム数、元レートを持つ。
- 生成物（`noisyN_SNRdb_S_clnspN.wav`、`clnspN.wav`、`noisyN_SNRdb_S.wav`）を保存する場合は、各 N の **元ファイル列**（clean の使用順と区切り位置、noise の使用順）も持つ。
- `urgent24` は外部 `urgent2024_challenge` の出力（この監査では取得していない）。

**2. clean / noise / RIR の役割**
- clean=目的、noise=加法雑音（SNR ごとに別ファイル）、RIR=なし。`ms_snsd` の `noise1.scp` は損失に使われない（`si_snr` 重み 0、参照のみ）。

**3. 分割と対応関係**
- 公式の分割は生成後のファイル名順 80/20。再現では **ソース発話・雑音ファイル単位の分割**（重複なし）を新しく決めるか、公式どおり名前順を保つかを明示する。
- 1 クリップ N ↔ 5 SNR のノイジー 5 本 ↔ clean 1 本（`clnspN.wav` は 5 本で共有）の対応表。

**4. 固定で保存すべき生成条件**
- クリップごと: clean ファイル列、挿入した無音長（0.2 s）、noise ファイル列、名目 SNR と **実 SNR**、使用した乱数 seed。
- 生成バージョン（`snr_mixer` の係数式、-25 dBFS 正規化、`audioread norm=True`）。

**5. 学習時にサンプリングする条件（分布パラメータ）**
- `ms_snsd` 公式には無い（固定ペア）。論文 (b) も動的混合なしで、4 秒区間のランダム切り出し（エポックごとに 1 区間/混合）だけが確率的。ESPnet のチャンク iterator は `Z`（開始オフセット）と `choice`（幅）、cache の shuffle を持つ。
- 分布パラメータとして持つべきもの: チャンク幅の候補、shift 比、`Z` の範囲、cache サイズ、num_samples_per_epoch。

**6. DataLoader 固有の処理**
- チャンク計画（`state=RandomState(epoch+seed)`、`choice`→`randint`→`shuffle` の乱数消費順）、短い発話の破棄、端数バッチの破棄、DDP の rank 分割と cache 分割、`num_iters_per_epoch` の周回ロジック。

**7. 共有可能な処理**
- 読み込み（16 kHz モノ・長さ検証）、チャンクのスライス、損失・指標の純関数、`seed_model_init`/`capture_rng`。

**8. 専用実装が必要な処理**
- MS-SNSD 生成器（連結・無音挿入・`snr_mixer`・-25 dBFS 正規化・未シード乱数のシード化）。
- ESPnet の DDP チャンク iterator（rank ごとの `batches[rank::world]`、rank ごとの cache・バッチ）。
- ESPnet の初期化（独自 LayerNorm の乱数化を含む）。

**9. 再現性のために保存すべき状態**
- RNG（Python / NumPy / PyTorch CPU / CUDA）、epoch（1 始まり）、sampler の位置（発話カーソル `num_samples*(epoch-1)`）、チャンク計画（または生成式と seed）、ファイル一覧のハッシュ、optimizer/scheduler、`chunk_history`（best/early stop 用）、バッチ位置（途中再開）、DDP 構成（world size、rank ごとの cache）、ESPnet ピン commit、生成バージョン。

**10. 既存レシピ JSON v1 で表現できないもの**

v1 は `sources`（clean / noise / rir が各 1 ファイル）、`operations`（`rir`、`additive_noise`、`peak_normalize`）、`output` を持ち、`case_id` は正規化レシピの SHA-256。1 ケース固定の記述である。

| 公式 / 論文の要素 | v1 | 不足 |
|---|---|---|
| clean を複数発話＋0.2 s 無音で連結（≥10 s）| `sources.clean` は 1 ファイル | 複数ソースの連結と区切り（`concat` 操作またはソース配列）|
| noise の連結・先頭から切る | `noise_start` と「繰り返し」のみ | 複数ファイル列とその順、ランダム再抽選のログ |
| 各ファイルを -25 dBFS RMS に正規化 | `peak_normalize` のみ（ピーク基準）| RMS 基準レベル正規化の操作 |
| SNR（名目の半分になる式）| `additive_noise.snr_db` は真の SNR（平均二乗電力比、`rms_v1`）| 名目 SNR と実 SNR の区別。同じ実 SNR は `snr_db = 名目/2` で表せるが、元の定義を持てない |
| SNR 分布（一様・格子）と SNR ごとの全ペア展開 | 単一値 | 分布、格子、直積展開 |
| noise 選択確率、複数 noise の重畳 | 1 本 | 選択確率・個数の分布 |
| RIR 適用確率、複数 RIR | `rir` は固定 1 本 | 確率、プール |
| LUFS / ゲイン | なし | ゲイン操作 |
| クリッピング | 「ハードクリップは操作にしない」（`docs/db/RECIPE.md` L48）| 公式データの PCM 書き出し時の挙動 |
| リサンプリング法 | `output.sample_rate` のみ（方法は固定）| 方法・フィルタ幅の指定 |
| チャンク/クロップ | レシピ外（`dataset` 側）| 学習計画の記述 |

「1 ケース固定のレシピ」と「オンライン生成の確率プロトコル」の比較: v1 は **決定済みの 1 ケース**（`noise_start`・`seed` を保存）を表すので、事前生成のペアを後から 1 件ずつ登録するのには向く。一方、プロトコル（SNR 分布・noise 選択確率・RIR 確率・複数 noise 重畳・ゲイン・クリップ・リサンプル法）は **プロトコル行（分布パラメータ）と、そこからケースを生成する関数（seed 付き）** に分けて記録する必要がある。ケース JSON はその出力として保持でき、`case_id` のハッシュ方針は変えずに済む。

---

## 11. 論文と公式コードの矛盾、既存ドキュメントの不整合・誤り

### 11-1 論文と公式コードの矛盾

| # | 内容 | 根拠 |
|---|---|---|
| P1 | STFT: 論文は窓 32 ms・ホップ 8 ms・平方根 Hann。ESPnet の `wsj0_2mix` v1/v2 conf は `n_fft 128` / `stride 64`（8 kHz で窓 16 ms、ホップ 8 ms）、窓は周期 Hann。`ms_snsd` は `n_fft 512` / `stride 256`（16 kHz で 32 ms/16 ms）| [論文 (b) §V-G; train_enh_tfgridnet_tf_lr-patience3_patience5.yaml L35-L37; ms_snsd yaml L36-L38]【確認:論文】【確認:設定】|
| P2 | 勾配クリップ: 論文 1.0。ESPnet recipe は YAML に書かず既定 5.0 | [論文 §V-G; abs_task.py L617]|
| P3 | epoch: 論文は 1 混合 1 区間。ESPnet `wsj0_2mix` は全発話から複数チャンク | §5 D-2 |
| P4 | Unfold の順序: 論文 (b) が LN-Unfold を提案（(a) は Unfold-LN）。ESPnet v1 は LN-Unfold | §2 A-2 |
| P5 | `E`: 論文は 8 kHz で 4、16 kHz で 2。ESPnet は `attn_approx_qk_dim 512` から `E = ceil(512/F)` を計算（16 kHz・F=257 で 2、`wsj0_2mix` n_fft 128・F=65 で 8）| [論文 §V-G; tfgridnet_separator.py L217-L219]【確認:論文】【確認:コード】|
| P6 | 分散正規化: 論文は混合の標本分散を 1 にして同じ係数を目的信号に掛ける。ESPnet v1/v2 は分離器内部で入力を割り出力に掛け戻す（損失側の目的信号は正規化しない設定が多い）。v3 は `normalize_variance` を外側で使う | §2 A-2 |
| P7 | PESQ: 論文は NB。ESPnet の `enh_scoring` は 16 kHz で WB | [論文 §V-H; enh_scoring.py L224-L268]|
| P8 | `ms_snsd` README: 「official MS-SNSD protocol」「SNR 0–40 dB」。実 SNR は名目の半分 | audiolib L68 |
| P9 | README は `local/ms_snsd_noisy_speech_synthesizer.py` と書くが、実ファイルは `ms_snsd_noisyspeech_synthesizer.py`。README の config 名 `train_enh_tfgrid.yaml` も存在しない（実ファイルは `train_enh_tfgridnet.yaml`）| [egs2/ms_snsd/enh1/README.md L61, L117; local/ 一覧]|
| P10 | `urgent24` の `run.sh` の既定は BSRNN。TF-GridNet は `--enh_config` を明示した場合 | [egs2/urgent24/enh1/run.sh L21]|

### 11-2 バージョンずれ

- 論文 (a) v2（2023-03）→ ESPnet v1 初出 2023-01 → ESPnet v2 は論文 (b) v2（2023-08-04）と同日付の PR #5395（マージ 2023-10-17）。
- HF `ms_snsd_tfgridnet` は ESPnet `202511` ＋ commit `239f5166a` で学習（README）。ピン留め `0d136c5` は 2026-10-30 で、`ms_snsd` の追加（2026-02）後、`urgent24` は 2026-10 に変更されている。HF 側の `meta.yaml` に ESPnet のバージョンは無い。
- HF `wyz/tfgridnet_for_urgent24` は ESPnet `202402`（`meta.yaml`）で学習。モデルカードの `license` は `cc-by-4.0` だが、`requirements/official-assets.toml` の `license_note` は「個別の記載なし」としている（記述のずれ）。【確認:設定】

### 11-3 既存ドキュメントの不整合・誤り

| 文書 | 記述 | 問題 |
|---|---|---|
| `.agents/DECISIONS.md` #26（L44）| 「best 5（valid loss 最小と valid si_snr 最大）」「記録される si_snr は負の SI-SDR」 | ピン留め ESPnet に `si_snr` キーは無く、`si_snr_loss` である。`valid si_snr` は適用されない。#31（L50）の記述（TF-Locoformer）は正しく、#26 と矛盾する |
| `docs/training/OFFICIAL_TRAINING_AUDIT.md` L63 | best は「`si_snr` の記録値は SI-SDR 損失。max は SI-SDR が低い方を選ぶ」 | 同上。根拠に挙げた `time_domain.py:290` は `SISNRLoss` の `forward` の近く（実際の `name` は L255）|
| 同 L68 | 「公式レシピは batch 4 の単一プロセスを想定」 | `run.sh` は `--ngpu 2`、HF config は `dist_world_size 2`・`multiprocessing_distributed true` |
| 同 L21, L59, L66 | `tf_gridnet_chunk_seed17.yaml`、`tf_gridnet_fixed5000_seed17.yaml` を参照 | 作業ツリーの `configs/training/` に存在しない（作業ツリーに無い）|
| `configs/training/tf_gridnet.yaml` L7-L9 | best は valid loss(min) と valid si_snr(max) | 同上（公式では `si_snr` 側が働かない）|
| `OFFICIAL_TRAINING_AUDIT.md`、`VERIFICATION.md` | 「GO」「公式と一致」 | 一致の根拠は合成入力・1〜4 更新・STFT スタブ等で、公式のデータ規模・DDP・初期化は含まない。初期化の差（TFG-D05）が記録に出てこない |
| `docs/training/VERIFICATION.md` | `data/results/training/*` の実行ディレクトリを参照 | 存在確認していない。`data/results/training` は存在するが中身は未確認 |
| `docs/training/OFFICIAL_TRAINING_AUDIT.md` | 「`trial100/`」参照 | 作業ツリーに無い |
| `requirements/official-assets.toml` | `tf_gridnet_urgent24` の `license_note` | HF のモデルカードは `cc-by-4.0`（§11-2）|

---

## 12. UNKNOWN一覧

| # | 内容 | UNKNOWN の理由 / 確認案 |
|---|---|---|
| U1 | HF `ms_snsd_tfgridnet` の学習データの実件数・時間 | 乱数が未シードで、生成ログが公開されていない。HF カードに記載なし |
| U2 | `valid.loss.best.pth` が何 epoch か、実際に何 epoch 学習したか | HF に学習ログ・`RESULTS.md` が無い（`ms_snsd` は README に結果のみ）|
| U3 | HF のチェックポイントの学習に使った ESPnet 版 | HF `meta.yaml` に版の記載がない。README は espnet2 202511 / commit `239f5166a` だが HF との対応は未確認 |
| U4 | MS-SNSD の clean/noise と VoiceBank-DEMAND（評価・学習）の重複 | ファイル名が異なる。16 kHz へ変換した波形の指紋照合で確認できる |
| U5 | audio-db の VoiceBank 件数・長さ分布・元サンプリングレート | `data/db` が空。docs の値は再確認不能 |
| U6 | VoiceBank 実データでの公式 epoch の更新数 | 上記と同じ。docs の 575 は未検証 |
| U7 | `urgent24` のデータ生成条件（SNR 分布など）| 外部リポジトリ `urgent2024_challenge` が作業ツリーに無い |
| U8 | DNS Interspeech 2020 のスクリプトにおける SNR の係数 | 調べていない。`snr_mixer` と同じ実装か未確認 |
| U9 | `sf.write` が ±1 超過の float を PCM_16 に書くときの挙動 | libsndfile / soundfile の既定を確認していない。0 dB 付近で影響し得る |
| U10 | 論文で epoch 数・early stopping・best 選択・seed・バッチサイズ | 論文に記載が見つからない |
| U11 | WSJ0CAM-DEREVERB の生成コード・データの所在 | 論文は「自前のシミュレーション」と述べるのみ。ESPnet にレシピなし |
| U12 | HF `urgent24` の実効 world size | config は `ngpu 1` と `dist_world_size 4` を同時に記録 |
| U13 | `ms_snsd` の `cv` が `tt` より STOI が低い原因 | README に構成の説明が無い |
| U14 | audio-db の独立実装と公式の数値一致（forward/loss/更新）の再確認 | 本監査で再実行していない。既存 docs の記録のみ |
| U15 | Zenodo / HF のチェックポイントのライセンス（`ms_snsd`）| HF カードに記載なし |

---

## 13. 参照した一次資料

論文（`/tmp/audit-tfgridnet/` に取得）
- [arXiv:2209.03952v2 HTML](https://arxiv.org/html/2209.03952v2)、[arXiv:2209.03952 abs](https://arxiv.org/abs/2209.03952)
- [arXiv:2211.12433v2 HTML](https://arxiv.org/html/2211.12433v2)、[arXiv:2211.12433 abs](https://arxiv.org/abs/2211.12433)
- ESPnet PR [#5395](https://github.com/espnet/espnet/pull/5395)

ESPnet（`espnet@0d136c5`、ローカル `build/clean-20261104/sources/espnet`）
- `espnet2/enh/separator/tfgridnet_separator.py`、`tfgridnetv2_separator.py`、`tfgridnetv3_separator.py`
- `espnet2/tasks/enh.py`、`espnet2/tasks/abs_task.py`
- `espnet2/iterators/chunk_iter_factory.py`、`sequence_iter_factory.py`
- `espnet2/train/trainer.py`、`reporter.py`、`preprocessor.py`、`espnet2/main_funcs/average_nbest_models.py`
- `espnet2/enh/espnet_model.py`、`enh/loss/criterions/time_domain.py`、`enh/loss/wrappers/fixed_order.py`
- `espnet2/torch_utils/initialize.py`、`espnet2/layers/stft.py`、`espnet2/enh/encoder/stft_encoder.py`
- `espnet2/bin/enh_inference.py`、`enh_scoring.py`、`egs2/TEMPLATE/enh1/enh.sh`
- `egs2/ms_snsd/enh1/`（`run.sh`、`README.md`、`conf/tuning/train_enh_tfgridnet.yaml`、`local/*`）
- `egs2/dns_ins20/enh1/`、`egs2/urgent24/enh1/`、`egs2/wsj0_2mix/enh1/`（conf・README・local）
- 履歴: `/tmp/audit-tfgridnet/espnet`（blobless clone、master `66f0ff5af`）の `git log` / `git tag --contains`

モデル重み・設定
- `data/checkpoints/tf-gridnet/ms-snsd/config.yaml`、`valid.loss.best.pth`（362 テンソル、2,552,790 パラメータ）
- HF `espnet/ms_snsd_tfgridnet` rev `49144db068f74542487eb23c0b6fbe63b67bfdd0`
- HF `wyz/tfgridnet_for_urgent24` rev `85f2296bcc1424750db7c49d2bd58deaf893024c`（`config.yaml`、`meta.yaml`、`README.md`）

MS-SNSD
- `data/corpora/MS-SNSD`（`audiolib.py`、`noisyspeech_synthesizer.py`、`.cfg`、`README.md`、コミット `fe61c4ba0d9ac8dd7e23d719cc79f8947e1dc742`）

audio-db（作業ツリー）
- `AGENTS.md`、`.agents/{PROJECT,DATA,DEVELOPMENT,DECISIONS}.md`
- `configs/training/tf_gridnet.yaml`、`requirements/official-assets.toml`
- `src/training/se/{tf_gridnet,optim,spectral,config,initialization}.py`、`src/training/data/{dataset,chunk_plan,sampler}.py`、`src/training/core/{trainer,rng,distributed}.py`
- `src/evaluation/metrics/{pesq,stoi,sisdr}.py`、`src/se/models/tf_gridnet/`
- `tests/training/test_chunk_official.py`
- `docs/db/{RECIPE,CORPORA,DATASETS}.md`、`docs/se/MODELS.md`、`docs/training/{OFFICIAL_TRAINING_AUDIT,VERIFICATION}.md`

この監査で行った数値確認（再現可能）
- `audiolib.snr_mixer` を乱数波形で呼び、名目 SNR [0,10,20,30,40] に対し実 SNR [0, 5, 10, 15, 20] dB を得た。
- HF チェックポイントの `LayerNormalization4D/4DCF` の gamma/beta の統計（上記）。
- MS-SNSD の各ディレクトリ 300 ファイル標本の長さ・形式。
