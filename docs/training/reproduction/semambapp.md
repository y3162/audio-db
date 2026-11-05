# SEMamba++ 公式再現プロトコル監査

調査専用の文書である。コード・YAML・DB・データは変更していない。本文中の「提案」は実装していない。

## 0. 要約

1. **対象の特定。** SEMamba++ は一般音声復元（GSR）の論文 arXiv:2603.11669（Yongjoon Lee, Jung-Woo Choi）で、公式実装は `Eddieyjlee/SEMambapp`（旧名 `infected4098/SEMambapp`、同一アカウント id 112607019）である。調査対象は `cd369c8`（2026-10-26）。リポジトリにライセンスファイルは無い。重みカードは GPL-2.0 表示。【確認:コード】
2. **公式の学習データ生成はオンライン混合である。** 事前生成の固定ペアではない。`GSRDataset.__getitem__` が、発話 1 件ごと・毎エポックごとに、雑音・RIR をファイルから一様無作為に引き、4 種の劣化（noise 1.0、reverb 0.5、bandwidth 0.5、clipping 0.5）を独立に抽選して、全発話長に適用する。その後、clean と劣化音を別々にピーク正規化し、3 秒（48,000 サンプル）を無作為に切り出す。事前生成の固定ペアは検証用（100 件）だけである。【確認:コード】
3. **audio-db の現行学習は別プロトコルである。** VoiceBank-DEMAND の固定 1 ケースレシピを使い、noisy の RMS を 1 にそろえる。データ素材・劣化の種類・SNR 定義・正規化・サンプラ・検証集合・検証指標のすべてが公式と異なる。損失の式・Optimizer・Scheduler・更新順・Discriminator の構成は、コード読解の範囲では公式と一致している。ただし、数値が合っても「同じプロトコルの再現」にはならない。【確認:コード】
4. **新規に見つけた差。** 独立 SEMamba++ の Mamba 混合層は `dt_proj.bias` を PyTorch 既定で初期化する。公式は `mamba_ssm.Mamba` の dt 初期化（softplus の逆関数で 0.001〜0.1）を使う。学習済み重みを strict load する検証では見えない、**ゼロから学習する場合だけの差**である（SMB-D11）。【確認:コード】
5. **論文と公式コードの食い違い。** 切り出し長（論文 24,000＝1.5 秒、コード 48,000）、SNR 範囲（論文 −10〜20 dB、HEAD の `config.yaml` は −5〜20 dB）、劣化確率（`config_orig.yaml` 0.7、HEAD 0.5）、magnitude 損失（論文 L1、コード MSE）。論文は確率を述べていない。公開重みは、論文の重みと異なる、バグ修正後の再学習である。再学習のデータは README に書かれていない。【確認:論文】【確認:コード】
6. **公式リポジトリは、出荷状態では学習を起動できない。** 既定の設定ファイル名が存在しない、`dataset.py` の `utils.util` import が解決しない、`data/*.json` が空（1 バイト）、など。`infer_folder.py` も、`infer.py` に無い `load_model`・`restore_file` を import する。マニフェストの中身（発話数・話者・雑音・RIR の一覧）は公開されていない。1 エポックの更新数 \(\lfloor N/8\rfloor\) の \(N\) は **UNKNOWN** である。【確認:コード】
7. **再現性の面で公式は弱い。** 無シードの一度きりのシャッフル、1 GPU では毎エポックの再シャッフル無し、DDP では rank ごとに独立の並び、`default_rng(None)` による雑音位置。検証による best 選択も早期終了も無く、5,000 step ごとに保存するだけである。再開は、保存したエポックを最初からやり直し、学習率を余分に \(\gamma^2\) 下げる（CPU の小さな再現で確認）。【確認:コード】
8. **新データ基盤への含意。** 「固定 1 ケースレシピ」ではなく「オンライン生成の確率プロトコル」として別に表す必要がある。レシピ JSON v1 は、1 回の抽選結果の一部（RIR・加法雑音・ピーク制限）しか表せない。bandwidth・clipping・VAD 基準の SNR・雑音の wrap 配置・個別ピーク正規化・RIR のチャンネル平均・確率的選択は表せない（§10）。
9. **評価。** 論文の評価（SCOREQ・UTMOS・DNSMOS・PESQ・LSD・LPS）のうち、`src/evaluation` にあるのは PESQ だけである。公式リポジトリに評価スクリプトは無い（検証ループ内の PESQ・UTMOS・MRSTFT だけ）。公式の学習データを使う検証も、論文のテストセット（VCTK-GSR test、URGENT 2025、DNS 2020、AATC 2025）の再現も、この作業ツリーにはデータが無く未実施である。
10. **この作業ツリーの状態。** `data/db/` は `.gitkeep` だけで DuckDB が無く、`data/staging/` も空である。`configs/training/semambapp_seed17.yaml` は存在しない。学習 YAML は seed 42 の `semambapp.yaml` だけである（§11）。

## 1. 調査メタデータ

| 項目 | 内容 |
|---|---|
| 調査日 | 2026-11-06 |
| 対象モデル | SEMamba++（`semambapp`） |
| audio-db | ブランチ `feature/reproduction`、HEAD `3bee9ad` |
| 公式コード | `https://github.com/Eddieyjlee/SEMambapp`（旧 `infected4098/SEMambapp`）、`main`、commit `cd369c89a8106072c077b468417b765b227afd74`（2026-10-26）。以下 **OFF** と書く |
| 公式コードのライセンス | リポジトリにライセンスファイル無し（GitHub API の `license` は null）。HF 重みカードは `gpl-2.0` |
| 論文 | arXiv:2603.11669 v1（2026-03-12）〜v3（2026-08-26）。以下 **PAP** と書く。Interspeech 2026 Long |
| 公開重み | `https://huggingface.co/yongjoonlee/semambapp`（`semambapp.pth`、最終更新 2026-08-20） |
| 公式 checkout の同一性 | `src/se/official/semambapp` の追跡ファイルと、`/tmp` の clone（HEAD `cd369c8`）に差分なし（`diff -rq`） |
| audio-db 側の確認方法 | docs ではなくコードを読んだ。以下 **ADB** と書く。行番号は HEAD `3bee9ad` |
| 実行したこと | 読み取りのみ。`/tmp/audit-semambapp/` に論文テキスト・clone・小さな CPU 再現を置いた。GPU・DB 書込・データ生成はしていない |
| 確信度ラベル | 【確認:コード】【確認:論文】【確認:設定】【推測】【UNKNOWN】 |

### 1.1 公式・非公式の判断根拠

- 論文プロジェクトページ（`sites.google.com/view/semambapp`）のリンクが `github.com/infected4098/SEMambapp` を指す。【確認:論文】
- GitHub API では、このリポジトリは現在 `Eddieyjlee/SEMambapp`、説明は "An official code repository for SEMamba++ (Interspeech 2026, Long track)."、所有者の表示名は Yong Joon Lee（id 112607019）である。`infected4098` のユーザー検索は結果を返さない（改名）。コミットの作者メールの数字 id も 112607019 で一致する。【確認:コード】
- README が arXiv 2603.11669 と HF 重みへのリンクを持ち、HF の所有者は `yongjoonlee` である。【確認:コード】
- 以上から、公式と判断する。改名のため、`requirements/official-assets.toml` は旧 URL、`src/se/models/semambapp/NOTICE` は新 URL を使っている。どちらも同じリポジトリを指すが、表記が不統一である。

### 1.2 バージョン・設定状態

| 状態 | 内容 |
|---|---|
| `config_orig.yaml`（2026-06-06 作成） | SNR 7 値（−10〜20）、reverb 0.7、bandwidth 0.7 |
| `config.yaml`（HEAD、2026-06-05 更新） | SNR 6 値（−5〜20）、reverb 0.5、bandwidth 0.5。両者の差はこの 3 項目だけ（`diff`） |
| 論文の重み | バグのあるコードで学習。第 3 解像度の Channel FFN が使われず、第 2 解像度で使われていた（README） |
| 公開重み | バグ修正後に「同じ設定で」再学習（README、2026-08）。再学習データは README に無い |
| README の旧記述 | 以前の README は "trained on VCTK and LibriTTS (~500 hours)" と書いていた（commit `8e7b6e8` で削除）。論文 Table 3 は学習データ 44 時間 |

audio-db の独立ネットワークは修正後の構造（`channel_ffns` が 3 解像度分）で、公開重み（修正後）に対応する。【確認:コード】`src/se/models/semambapp/network.py`

## 2. A. 公式資料

| 資料 | 内容 | 確信度 |
|---|---|---|
| 論文 PAP v1/v2/v3 | §4.1 Training に、データ・劣化・学習条件の段落がある。v1〜v3 でこの段落は同一。Table 1 の SEMamba++ 行（v1 と v3）も同一値 | 【確認:論文】 |
| 論文の訂正 | v3（2026-08-26）にも、バグ修正・再学習の記述は無い | 【確認:論文】 |
| 公式 README | データはマニフェスト 5 個（`train_speech/noise/rir.json`、`val_clean/degraded.json`）。学習コマンドの記述は無い。推論の使い方と再学習後の比較表がある | 【確認:コード】 |
| `train.py`（OFF L441–487 主関数） | 学習ループは `run()` L68–204、`train()` L206–436 | 【確認:コード】 |
| `dataset.py`（OFF L46–279） | `GSRDataset` が Train / Validation / Test の 3 モードを持つ | 【確認:コード】 |
| `simulate_degradation.py` / `simulate_degradation_utils.py` | 劣化抽選と各劣化。`mix_noise` は URGENT 2025 系の実装を流用したもの | 【確認:コード】（流用元は【推測】） |
| `requirements.txt` | torch 2.10.0+cu126、librosa 0.11.0、numpy 2.2.6、scipy 1.15.3、torchmetrics 1.9.0、auraloss 0.4.0、nnaudio 0.3.4、pesq 0.0.4、soxr 1.1.0。`mamba-ssm==2.3.2.post1` と `causal-conv1d==1.7.0` は README の別コマンド | 【確認:設定】 |
| 公式の評価スクリプト | 無い。`evaluation_metrics.py` は学習中の検証用だけ。`infer.py` / `infer_folder.py` は推論 | 【確認:コード】 |
| 関連実装 | SEMamba（RoyChao19477/SEMamba、arXiv:2405.06573）。損失係数・位相損失は MP-SENet / SEMamba 系に従うと論文が述べる（§3.4） | 【確認:論文】 |

## 3. B. 学習データセット

### 3.1 公式の学習データ（論文）

| 役割 | 論文の記述 | 確信度 |
|---|---|---|
| 音声 | VCTK **version 0.92**。話者 p280 と p315 を「技術的理由」で除外。16 kHz。「VCTK-GSR train/test に分割、[50] に従う」 | 【確認:論文】 |
| 雑音 | WHAM! と DNS Challenge 2020 の雑音。雑音・RIR も train/test に分割 | 【確認:論文】 |
| RIR | Arni（Prawda ら 2022）と DNS5 challenge（Ko ら 2017 の RIR 系） | 【確認:論文】 |
| 学習データ量 | Table 3: SEMamba++ は 44 時間 | 【確認:論文】 |
| 分割規則 [50] | Kuleshov ら（arXiv:1708.00853）は VCTK の「最初の 99 話者で学習、残り 8 話者でテスト」としている。SEMamba++ の VCTK-GSR が同じ分割か、話者順序・除外後の数はどうか | 【推測】【UNKNOWN】 |
| 雑音・RIR の分割規則 | 記述なし | 【UNKNOWN】 |

### 3.2 公式コードのデータ入力

| 役割 | ファイル | 内容 | 確信度 |
|---|---|---|---|
| 学習音声 | `data/train_speech.json` | パスの JSON リスト（OFF dataset.py L90–94）。リポジトリのファイルは空 | 【確認:コード】 |
| 学習雑音 | `data/train_noise.json` | 同上。発話とは独立に引く | 【確認:コード】 |
| 学習 RIR | `data/train_rir.json` | 同上 | 【確認:コード】 |
| 検証 clean / degraded | `data/val_clean.json`、`data/val_degraded.json` | 事前生成の固定ペア。ファイル名末尾の `_数字` でソートし、**先頭 100 件**を検証、残りを Test モードに割り当てる（L97–101）。Test モードは `train.py` から使われない | 【確認:コード】 |
| 件数・話者・長さ・生成条件 | — | 公開されていない | 【UNKNOWN】 |

### 3.3 audio-db のこの作業ツリーにある素材

`data/corpora/` はシンボリックリンク。最上位の一覧だけを確認した（中身は走査していない）。

| 公式が使う素材 | 作業ツリーの状況 | 確信度 |
|---|---|---|
| VCTK 0.92 | `VCTK` は **version 0.80**（`wav48/`、README に Version 0.80）。docs/db/DATASETS.md によれば 108 話者・44,085 発話、p315 の音声が無く、p280 は 410 発話あり `speaker-info.txt` に無い。0.92 とは版・無音区間処理・p280/p315 の状態が異なる可能性がある | 【確認:設定】版違いは確認、影響は【UNKNOWN】 |
| WHAM! 雑音 | `WHAM48/high_res_wham`（48 kHz 高解像度版）。論文の WHAM! がどの版か不明 | 【UNKNOWN】 |
| DNS2020 雑音 | `DNS2020/datasets/{clean,noise,test_set,blind_test_set}` が見える | 【確認:設定】 |
| DNS5 RIR | `DNS5/datasets_fullband/impulse_responses`（`SLR26`、`SLR28` を含む）。OpenSLR26/28 も別にリンクあり | 【確認:設定】 |
| Arni RIR | 見当たらない | 【確認:設定】 |
| VCTK-GSR test、URGENT 2025 val / blind、AATC 2025 | 見当たらない（`URGENT2024` のリンクはあるが 2025 ではない） | 【確認:設定】 |

## 4. C. データ生成・前処理

### 4.1 公式の学習時の処理順（1 サンプルごと、`GSRDataset.__getitem__` の Train モード）

[OFF dataset.py L145–216、simulate_degradation.py L13–111、simulate_degradation_utils.py L58–303]

1. **発話の選択。** `clean_wavs_path[index]`。リストは `__init__` で無シードの `random.shuffle` を一度だけ受ける（L104–109）。【確認:コード】
2. **雑音・RIR の選択。** 雑音と RIR は、発話とは独立に、毎回 `random.choice` で一様・復元抽出する（L148–149）。【確認:コード】
3. **読み込み。** `librosa.load(path, sr=16000)`。モノラル化（多チャンネルは平均）、リサンプルは librosa 0.11 の既定 `soxr_hq`。雑音・発話・RIR は**ファイル全体**を読む（L151–153）。【確認:コード】
4. **形状を (1, T) にそろえる。**
5. **劣化の抽選**（`random_select_and_order`、L13–62）。
   - SNR は `snr` リストから `random.choice`（HEAD は {−5, 0, 5, 10, 15, 20}）。連続一様ではなく、5 dB 刻みの離散一様。
   - bandwidth は `bandwidth_sr` を {4k〜14k Hz} から（折り返し周波数は 2〜7 kHz）、リサンプル種別 4 通り、ローパス種別 5 通り、次数 {2, 4, 6, 8} を、それぞれ一様に選ぶ。
   - clipping は下側分位 U(0, 0.1)、上側分位 U(0.9, 1.0)。
   - 各劣化の採否は独立ベルヌーイ。noise 1.0（常に採用）、reverb 0.5、bandwidth 0.5、clipping 0.5。何も選ばれなかった場合の救済分岐はあるが、noise 1.0 のため到達しない。組合せは 8 通りが等確率（HEAD）。
   - `packet_loss` は設定に確率 0.2 があるが、`degrad_types` に含まれず選ばれない。
   - この関数は L173 と L176 で**2 回**呼ばれ、最初の結果は捨てられる（乱数を余分に消費するだけ）。
   【確認:コード】
6. **適用順は固定。** reverb → noise → clipping → bandwidth（順序表 L52–58）。選ばれた劣化だけを、この順に**発話全体**へ適用する。
   - reverb: `scipy.signal.convolve(full)` の先頭を入力長に切る。RIR は正規化・遅延補正をしない（L179–189）。
   - noise: 発話（reverb 済みならその信号）と雑音の電力を、`detect_non_silence`（1024 サンプル・シフト 512・箱型窓・フレーム電力が全体平均電力の 0.01 倍超）で選んだ区間の平均二乗で測り、`SNR` に合わせて雑音を拡大して加算する（L112–144）。雑音が短ければ、乱数のオフセットから前に置いて wrap で埋める。長ければ乱数のオフセットで切り出す。オフセットは `default_rng(seed=None)` で、**毎回エントロピーから**作る。
   - clipping: 発話の分位点で `np.clip`。
   - bandwidth: ローパス（`filtfilt`）→ `fs_new` へ縮小 → 元の 16 kHz へ戻す（L234–279）。
   【確認:コード】
7. **教師は元の dry clean**（劣化前の全発話長）。RIR を掛けても教師は reverb 無しの原音である。【確認:コード】
8. **ピーク正規化。** clean と劣化音を、それぞれ別々に `x / (max|x| + 1e-9)`（L41–43、L197–198）。切り出しの**前**に行うので、切り出し範囲のピークは 1 にならない。clean と劣化音の相対レベルは保存されない。【確認:コード】
9. **切り出し。** 長さ ≥ 48,000 なら `random.randint` で開始位置を引いて切り出す。短ければ末尾をゼロ埋め（L200–210）。【確認:コード】
10. **STFT（Dataset 内、CPU）。** n_fft 400、hop 100、win 400、圧縮 0.3 の magnitude・phase・complex を clean と劣化音の両方で作って返す（L213–216）。【確認:コード】
11. **DataLoader。** `batch_size=8`（GPU 数で割る）、`drop_last=True`、`num_workers=32`、`pin_memory=True`。【確認:コード】

乱数源: Python の `random`、NumPy の `np.random`（ワーカーごとに PyTorch が基準シードから設定する）、`default_rng(None)`（毎回エントロピー）。したがって、グローバルなシードを固定しても、雑音位置は再現しない。【確認:コード】【推測】（PyTorch のワーカーシード挙動は `torch.utils.data._utils.worker._worker_loop` の `random.seed`・`torch.manual_seed`・`np.random.seed` で確認）

### 4.2 公式の検証時の処理

[OFF dataset.py L220–244、train.py L371–431]

1. `val_clean` / `val_degraded` の先頭 100 件を、ファイル名末尾の数字でソートして対応づける。
2. `librosa.load(sr=16000)`、それぞれピーク正規化。切り出し・パディングなし（全長）、`batch_size=1`。
3. 生成器を `eval()` で動かし、逆 STFT。出力が長ければ clean の長さに、短ければ clean を出力の長さに揃える。
4. 指標: magnitude MSE、位相損失 3 項の和、complex MSE、MRSTFT（auraloss）、PESQ（torchmetrics、`wb`）、UTMOS（`tarepan/SpeechMOS:v1.2.0` の `utmos22_strong`、`torch.hub` で実行時に取得）。
5. 平均して wandb に記録するだけで、checkpoint の選択には使わない。

【確認:コード】。UTMOS を実行時にネットワークから取得する点は、オフライン環境で再現するときの障害である。

### 4.3 audio-db の学習時の処理順（現行 VoiceBank 設定）

[ADB src/training/data/dataset.py L181–222、L331–380、sampler.py L15–46、evaluation/audio/render.py L72–155]

1. DuckDB の名前付き学習セット（`voicebank_28_train`）から、`case_id`・レシピ・参照種別を得る。
2. `case_id` がレシピのハッシュと一致するか確かめる。
3. `render_recipe` で、レシピの `operations` を順に適用する（`rir` → `additive_noise` → `peak_normalize`）。`peak_normalize` は、ピークが上限を超えたときだけ縮める制限である。
4. 出力 `sample_rate` と学習 `sample_rate` が違えば `torchaudio.functional.resample`（既定のフィルタ）。
5. `rms_normalize` が真なら、noisy の二乗平均を 1 にする共通スケールを、noisy と clean に掛ける（`_normalize`、L359–364）。
6. 学習時は `_segment`（L367–380）。`crop_start` は `(case_id, seed, epoch)` から決定的（sampler.py L37–46）。短ければ末尾ゼロ埋め。
7. バッチ化とサンプラ: エポックごとに `randperm`（`mix_seed("order", seed, epoch)`）で並べ替え、`world_size` で割り切れる分だけ使う（sampler.py L15–34）。

【確認:コード】

## 5. D. Dataset/DataLoader/Sampler

### 5.1 公式

| 項目 | 実効動作 | 根拠 |
|---|---|---|
| Dataset | `GSRDataset`、`__len__` は学習音声の件数 | OFF dataset.py L274–279 |
| サンプラ（1 GPU） | `shuffle=False`、サンプラ無し。順序は `__init__` の一度きりの無シード shuffle のまま。**エポックごとに再シャッフルされない**。ワーカーの乱数だけが毎エポック変わる | OFF train.py L99–103、dataset.py L104–109 |
| サンプラ（複数 GPU） | `DistributedSampler(trainset, rank=rank)`。`mp.spawn` で起動した各プロセスが、それぞれ無シードで shuffle する。そのため、**rank ごとに元リストの並びが異なる**。`DistributedSampler` はインデックスで分割するので、rank 間で同じ発話が重複し、別の発話が抜ける。`set_epoch` はエポックごとに呼ばれる | OFF train.py L99、L219 |
| 重複・欠落の一般論 | `DistributedSampler` は `drop_last=False` で、末尾を繰り返して各 rank を同数にする | PyTorch 既定【確認:コード】（実行はしていない） |
| バッチ | グローバル 8。`batch_size // n_gpus`（`n_gpus = torch.cuda.device_count()`）。8 が割り切れない GPU 数ではグローバル値が 8 にならない | OFF train.py L474 |
| `drop_last` | 真 | OFF train.py L103 |
| ワーカー | 学習 32、検証 1。乱数は PyTorch が設定 | OFF train.py L101、L123 |
| 勾配蓄積 | 無し | OFF train.py（該当処理なし） |
| `n_cache_reuse`・`pcs`・`selected_degrads` | 引数だけで未使用 | OFF dataset.py |
| 設定の `dist_cfg`、`world_size: 1` | 読まれない（GPU 数は `device_count`） | OFF config.yaml L7–10 |

### 5.2 1 エポックの定義（公式 N は UNKNOWN）

\(N\) = `train_speech.json` の件数（**UNKNOWN**）。

| 量 | 公式（1 GPU） | 公式（\(k\) GPU の DDP） | audio-db（VoiceBank、現行） | 確信度 |
|---|---|---|---|---|
| 1 エポックの発話数 | \(N\)（各発話を 1 回） | \(k\) 個の rank が各 \(\lceil N/k\rceil\) 件（末尾は重複で補う） | 10,700（`voicebank_28_train`） | 【確認:コード】 |
| 1 エポックの chunk（切り出し）数 | \(N\)（発話ごとに無作為 1 切り出し。chunk 展開なし） | 同左 | 10,700 | 【確認:コード】 |
| micro-batch 数 | — | — | 2,675（10,700/4。端数なし） | 【確認:設定】 |
| バッチ数 | \(\lfloor N/8\rfloor\) | rank あたり \(\lfloor \lceil N/k\rceil/(8/k)\rfloor\) | 蓄積 2 のため 2,674 micro-batch 分（1 つ余る） | 【確認:コード】 |
| Discriminator 更新数 | \(\lfloor N/8\rfloor\)（1 バッチ 1 回） | 同左 | 1,337 | 【確認:コード】 |
| Generator 更新数 | \(\lfloor N/8\rfloor\) | 同左 | 1,337 | 【確認:コード】 |
| `steps` の増分 | 1 バッチごと（D と G の更新 1 組で +1） | 同左 | Optimizer 更新（D→G の組）1 回で +1 | 【確認:コード】 |
| Scheduler の進め方 | エポック末に G・D とも 1 回（`train()` 末、OFF L435–436） | 同左 | エポック末 | 【確認:コード】 |
| 学習率 | エポック \(e\) で \(2\times10^{-4}\times 0.99^{e}\)。\(e=99\) で約 \(7.4\times10^{-5}\) | 同左 | 同左 | 【確認:設定】 |
| 総 step | \(100\times\lfloor N/8\rfloor\) | 同左 | 133,700 | 【確認:設定】 |
| checkpoint 間隔 | 5,000 step | 同左 | 1,337 step（= 1 エポック） | 【確認:設定】 |
| 検証間隔 | 5,000 step | 同左 | 1,337 step（= 毎エポック） | 【確認:設定】 |

\(N\) が約 40,000 だとすれば、1 エポックはおよそ 5,000 step で、5,000 step ごとの検証は約 1 エポックごとに当たる。これは**例示**であり、公式の件数ではない。【推測】

### 5.3 audio-db

| 項目 | 実効動作 | 根拠 |
|---|---|---|
| Dataset | `RecipeDataset`。固定レシピをその都度レンダリングする | ADB dataset.py L181–222 |
| サンプラ | `epoch_indices`。エポックごとに `mix_seed("order", seed, epoch)` で並べ直し、rank に等分して端数は捨てる（重複させない） | ADB sampler.py L15–34 |
| 切り出し | `(case_id, seed, epoch)` で決まる | ADB sampler.py L37–46 |
| バッチ | micro 4 × 蓄積 2（グローバル 8）。蓄積の除数を直した | ADB configs/training/semambapp.yaml L17–18、docs/training/VERIFICATION.md |
| 再開 | エポック途中でも厳密に再開する設計（Optimizer・Scheduler・RNG・位置を保存） | ADB trainer.py、docs/training/VERIFICATION.md |

## 6. E. モデル・学習条件

「設定値」と「実効値」を分ける。

| 項目 | 設定値 | 実効値・注記 | 根拠 | 確信度 |
|---|---|---|---|---|
| 構造 | hid 48、`num_tfmamba` 4、`d_state` 16、`d_conv` 4、`expand` 2、`norm_epsilon` 1e-5、compress 0.3 | 約 2.7 M（論文 Table 2）。`unet_expansion: 1`。`dropout: 0.1` は設定にあるが、モデルコードに Dropout は無く**無効** | OFF config.yaml L62–77、model/*.py | 【確認:コード】 |
| STFT | n_fft 400、hop 100、win 400、16 kHz、圧縮 0.3 | | OFF config.yaml L56–60 | 【確認:設定】 |
| 生成器の損失 | adv_g 1.0、fm_g 1.0、magnitude 0.9、phase 0.3、complex 0.1、mel 0.1、consistency 0.1 | magnitude と complex と consistency は `F.mse_loss`。complex と consistency は係数の前に ×2。`time` 0.05 と `metric` 0.05 は**使われない**（時間 L1 は計算のみで合計に入らない） | OFF train.py L279–303 | 【確認:コード】 |
| 判別器 | MS-SB-CQTD（nnAudio、3 サブ判別器、8 オクターブ、12/24/36 bin）＋ MRD | LSGAN | PAP §3.4、OFF model/discriminator.py | 【確認:論文】【確認:コード】 |
| 更新順 | 1 バッチで D → G | G の更新は同じ生成物を使う。D の勾配は G 更新時にも溜まるが、次の `zero_grad` で消える | OFF train.py L243–309 | 【確認:コード】 |
| Optimizer | AdamW、lr 2e-4、betas (0.8, 0.99)（G と D 同じ） | **weight_decay の指定なし → PyTorch 既定 0.01**、eps 1e-8。全パラメータに一律（`no_weight_decay` の除外なし） | OFF train.py L163–166 | 【確認:コード】 |
| Scheduler | ExponentialLR γ=0.99、エポックごと | G と D の両方 | OFF train.py L45–53、L435–436 | 【確認:コード】 |
| 勾配クリップ | 無し | | OFF train.py | 【確認:コード】 |
| 混合精度 | 使わない（`autocast` なし） | | OFF train.py | 【確認:コード】 |
| 初期化 | 設定なし | `Mamba` 内は `mamba_ssm` の初期化（`dt_proj.bias` は 0.001〜0.1 の逆 softplus、`A_log` は log(1..d_state)、`D` は 1）。`MambaBlock` ごとに `_init_weights(n_layer=1)` を適用（Linear のバイアスを 0 に。`dt_proj.bias` は `_no_reinit`）。他の層は PyTorch 既定 | OFF model/semambapp.py L86–97、mamba_ssm 2.3.2.post1 `mamba_simple.py` L83–101 | 【確認:コード】 |
| シード | `torch.manual_seed(1234)`（L77）。`torch.backends.cudnn.benchmark = True`（L26）。決定的アルゴリズムの指定は無し | Python / NumPy は無シード。実質**再現不能** | OFF train.py L77 | 【確認:コード】 |
| 総エポック | 100（設定・引数既定とも） | 論文も 100 epoch | OFF config.yaml L35、train.py L453、PAP §4.1 | 【確認:コード】【確認:論文】 |
| batch | 8（グローバル） | | PAP §4.1、OFF config.yaml L36 | 【確認:論文】 |
| segment | 設定 48,000（3 秒） | **論文は 24,000（1.5 秒）**。どちらで論文の重みを学習したかは UNKNOWN | OFF config.yaml L41、PAP §4.1 | 【確認:設定】【UNKNOWN】 |
| checkpoint | 5,000 step ごと（`steps != 0`）。生成器（`ln_g_*.pth`）と、判別器＋ Optimizer＋ `steps` ＋ `epoch`（`ln_do_*.pth`） | **best 保存・早期終了なし** | OFF train.py L331–351 | 【確認:コード】 |
| 再開 | 最新の `ln_g_` / `ln_do_` を名前順で選ぶ。重みは `strict=False`。Optimizer も復元。`steps+1` から開始 | 保存したエポック**を最初からやり直す**。Scheduler を 2 組（L173、L189–190）作り、それぞれが `last_epoch` から 1 回ずつ学習率を γ 倍する。CPU の小さな再現（torch 2.6.0）で、復元した学習率に γ² が余分に掛かることを確認した（保存時 1.9020e-4 → 再開後 1.8641e-4）。torch 2.10 での挙動は【推測】 | OFF train.py L55–65、L156–193 | 【確認:コード】 |
| DDP | 2 GPU 以上で `DDP`（判別器 2 つとも）。検証は rank 0 だけ。`find_unused_parameters` は既定 | 検証は DDP ラップ済みの生成器を rank 0 だけで呼ぶ構造。バッファが無ければ問題にならない見込み（未検証） | OFF train.py L99、L178–181、L371 | 【確認:コード】【推測】 |
| 依存 | torch 2.10.0+cu126、mamba-ssm 2.3.2.post1、causal-conv1d 1.7.0 | audio-db の `.venvs/se` は torch 2.6.0+cu124、librosa 1.0.0、numpy 2.5.4 | OFF requirements.txt、ADB 実測 | 【確認:設定】 |

### 6.1 audio-db 側の同じ項目

[ADB configs/training/semambapp.yaml、src/training/se/semambapp.py、optim.py]

| 項目 | 設定値 | 実効値・注記 |
|---|---|---|
| 構造 | 公式幅と同じ（hid 48、4 block、d_state 16、d_conv 4、expand 2） | `scan_backend: fused`（CUDA）。CPU は reference scan |
| 損失 | `loss: official`。係数は公式と同値 | 検証記録は docs/training/VERIFICATION.md（グループ別の損失合算順を公式に合わせて修正済み） |
| Optimizer | AdamW、lr 2e-4、betas (0.8, 0.99)、`weight_decay: 0.01` | 公式の実効値と同じ。D も同じ（ADB optim.py L20–32） |
| Scheduler | ExponentialLR γ=0.99（エポック毎） | 同じ |
| 勾配クリップ | 設定なし | 公式と同じ（無し） |
| 初期化 | `seed_model_init(config["seed"])` のもと、PyTorch 既定。**`dt_proj.bias` は既定の一様分布**（公式は dt 初期化） | **DIFF**（SMB-D11）。A_log・D は同じ式 |
| seed | 42 | docs は 17 と書く箇所がある（§11） |
| 切り出し | 48,000 | 論文 24,000 とは異なる。コードに合わせた |
| バッチ | micro 4 × 蓄積 2 | 公式は 8 の 1 forward。H100 MIG 4g.47gb で OOM のため |
| best checkpoint | 無効（`_consider_best` は `semambapp` で None を返す、ADB trainer.py L738–785、L741–742） | 設定の `best_checkpoint_start_epoch: 40` は効かない |
| 再開 | 厳密再開 | 公式の再開（エポックのやり直し・γ² ）とは異なる。意図的な改善で、公式一致ではない |

## 7. F. Validation/Test/評価

### 7.1 論文・公式スクリプト・`src/evaluation` の比較

| 項目 | 論文 | 公式スクリプト | audio-db（`src/evaluation` と学習内検証） |
|---|---|---|---|
| 学習中の検証集合 | 記述なし | 固定 100 ペア（`val_clean/degraded`、全長、batch 1） | `voicebank_28_valid`（872 ペア、noisy と clean の対） |
| 検証の頻度 | 記述なし | 5,000 step ごと | 毎エポック（1,337 step） |
| 検証の前処理 | — | 個別ピーク正規化、全長 | RMS 正規化（noisy 基準）、全長 |
| 学習内検証の指標 | — | magnitude / 位相 / complex の損失、MRSTFT、PESQ（torchmetrics wb）、UTMOS22 | L1、SI-SDR、PESQ、STOI（`validation_report`、ADB semambapp.py L224–247） |
| 検証の使い道 | — | ログのみ | ログのみ（SEMamba++ は best 無効） |
| 評価データ | VCTK-GSR test、URGENT 2025 val・blind、DNS 2020 test（実録）、AATC 2025 test | **評価スクリプト無し**。`dataset.py` の Test モードは未使用 | VoiceBank-DEMAND test 等。上記のどれも未登録 |
| 指標 | SCOREQ（合成データ向け・no-reference 版）、UTMOS、DNSMOS（SIG/BAK/OVRL）、PESQ、LSD、LPS（URGENT 公式実装）、RTF | 検証の PESQ・UTMOS・MRSTFT のみ | PESQ（`pesq` 0.0.4、wb）、STOI（pystoi）、SI-SDR、WER。UTMOS・DNSMOS・LSD・LPS・SCOREQ は**無い** |
| PESQ のモード・実装 | 記述なし【UNKNOWN】 | torchmetrics 1.9.0 の `wb`（内部で `pesq`） | `pesq` 0.0.4 の `wb` |
| 報告値（SEMamba++） | Table 1: VCTK-GSR test は SCOREQ 3.27、UTMOS 3.55、OVRL 3.14、PESQ 1.77、LSD 1.25、LPS 0.45。URGENT val は 2.67、2.82、3.20、1.51、1.49、0.61。URGENT test は 2.49、2.61、3.13。Table 2 AATC: SIG 3.45、BAK 4.01、OVRL 3.18、PESQ 1.84（2.7 M、RTF 0.021、A6000）。Table 3 DNS 2020 実録: SIG 3.487、BAK 4.020、OVRL 3.206（44 時間） | — | — |
| 再学習後の報告値 | — | README: AATC OVRL 3.20 / PESQ 1.95、URGENT val PESQ 1.60 / LPS 0.65、URGENT blind OVRL 3.07、DNS 2020 OVRL 3.208 | — |

【確認:論文】【確認:コード】

### 7.2 注意点

- 論文の評価ではどの checkpoint（step）を使ったか書かれていない。公式の保存方式（5,000 step ごと、best なし）から、最終 step の可能性が高いが、**UNKNOWN** である。【UNKNOWN】
- 論文の VCTK-GSR test の生成条件（SNR、劣化の確率、乱数、事前生成かオンラインか）は書かれていない。固定ペアとして配布されているかも不明である。【UNKNOWN】
- 公式の検証 PESQ は、ピーク正規化後の出力と clean を比べる。PESQ はレベル整合をするが、正規化の流儀が違えば値は変わりうる。【推測】
- audio-db の PESQ は、欠損を 0 として扱わない（AGENTS.md の制約）。公式の torchmetrics は異常時に例外を出すだけで、扱いが異なる。

## 8. 現行 audio-db との項目別比較表

判定: MATCH / PARTIAL / DIFF / UNKNOWN / N/A。

| 項目 | 公式設定 | 公式実効動作 | audio-db現状 | 判定 | 根拠 |
|---|---|---|---|---|---|
| Dataset | VCTK 0.92（−p280/p315）＋ WHAM!/DNS2020 雑音 ＋ Arni/DNS5 RIR | `GSRDataset` がマニフェスト 3 つから毎回抽選。件数 UNKNOWN | VoiceBank-DEMAND（train 10,700 / valid 872 / test 824）の固定レシピ | DIFF | PAP §4.1、OFF dataset.py L90–109、ADB semambapp.yaml L55–59。SMB-D01, D02 |
| mixing mode | 記述なし（「ランダムに適用」） | オンライン。発話 1 件・エポック 1 回ごとに再抽選 | 事前生成の固定ペア（case_id 固定） | DIFF | OFF dataset.py L145–182。SMB-D01 |
| noise selection | WHAM!/DNS 雑音 | 毎回一様・復元抽出。全長を読み、乱数オフセットで切り出し／wrap | レシピに保存した `noise_start`。短ければ繰り返し | DIFF | OFF dataset.py L148、utils L112–144、ADB mixing.py L150–180。SMB-D06 |
| SNR | −10〜20 dB（論文）／{−5..20}（HEAD）／{−10..20}（orig） | 5 dB 刻みの離散一様。有音フレームの電力比（VAD）。reverb 後の信号に対して | レシピの `snr_db`（VoiceBank 固定）。`rms_v1`（全サンプルの平均二乗） | DIFF | OFF config.yaml L15、utils L58–144、ADB mixing.py L66–147。SMB-D04 |
| gain | 記述なし | ゲイン拡張なし | なし（`peak_normalize` は上限制限） | MATCH（どちらも無し） | OFF dataset.py、ADB render.py L126–132 |
| reverb/RIR | Arni + DNS5。確率は論文に無し | 確率 0.5（orig 0.7）。RIR は毎回一様抽出、モノ平均、遅延補正・正規化なし、全長畳み込みして入力長に切る | VoiceBank に reverb 無し。レシピの `rir` は 1 チャンネル選択・因果畳み込み・入力長に切る | DIFF | OFF utils L179–189、ADB render.py L95–100、L234–260。SMB-D03 |
| 他の劣化 | bandwidth 2–7 kHz、clipping | bandwidth 0.5、clipping 0.5、packet_loss は未使用 | 無し（`bandwidth`・`clipping` の操作が無い） | DIFF | OFF simulate_degradation_utils.py L234–303、ADB docs/db/RECIPE.md。SMB-D03 |
| 教師（target） | 記述なし | 元の dry clean（全長） | `dry_clean` 参照を取れる（`reference` で指定）。現行 VoiceBank は clean | PARTIAL | OFF dataset.py L182、ADB dataset.py L186–193 |
| resampling | 16 kHz | `librosa.load(sr=16000)`（soxr_hq）で全ファイルを読む時にリサンプル | `torchaudio.functional.resample`（既定フィルタ）、レンダリング後 | DIFF | OFF dataset.py L151–153、ADB dataset.py L331–345。SMB-D16 |
| normalization | 記述なし | clean と劣化を別々にピーク正規化（切り出し前）。RMS 正規化なし | noisy の RMS を 1 にする共通スケール（clean にも同じ値）。切り出し前 | DIFF | OFF dataset.py L41、L197–198、ADB dataset.py L359–364。SMB-D05 |
| crop | 24,000（論文）／48,000（設定） | 一様乱数の開始位置、劣化**後**に切り出し | 48,000。`(case_id, seed, epoch)` で決定的、レンダリング後 | PARTIAL | OFF dataset.py L200–205、ADB sampler.py L37–46。SMB-D07 |
| padding | 記述なし | 長さ不足は末尾ゼロ埋め | 末尾ゼロ埋め | MATCH | OFF dataset.py L208–210、ADB dataset.py L378–380 |
| length limit | 記述なし | なし（全長を読む） | レシピの `max_seconds` で全長を制限しうる | PARTIAL | OFF dataset.py L151、ADB RECIPE.md |
| shuffle/sampler | 記述なし | 1 GPU: 一度きりの無シード shuffle、再シャッフルなし。DDP: rank ごとに独立の並び ＋ `DistributedSampler`（末尾は重複補完） | エポックごとに決定的 `randperm`、rank 等分・端数は捨てる | DIFF | OFF train.py L99–103、dataset.py L104–109、ADB sampler.py L15–34。SMB-D08 |
| epoch definition | 100 epoch | 1 エポック＝全発話 1 回 ＝ \(\lfloor N/8\rfloor\) 回の更新 | 1 エポック＝10,700 発話 ＝ 1,337 更新 | PARTIAL（定義は同じ、\(N\) が異なる） | §5.2。SMB-D08 |
| batch | 8 | 1 回の forward で 8 | micro 4 × 蓄積 2（バッチ形状の差を測定済み） | PARTIAL | OFF train.py L474、ADB semambapp.yaml L17–18。SMB-D09 |
| DDP/accumulation | 記述なし | DDP は `batch//n_gpus`、蓄積なし | DDP と蓄積に対応。GPU 実測は縮小実モデル（2 GPU） | PARTIAL | ADB docs/training/TENSORBOARD.md。SMB-D09 |
| loss | adv 1、fm 1、mag 0.9、pha 0.3、com 0.1、mel 0.1、con 0.1（論文は mag を L1 と書く） | mag・com・con は MSE。time・metric は未使用 | 同値（`loss: official`） | MATCH（コードに対して）。論文の mag L1 とは食い違い | OFF train.py L279–303、ADB semambapp.py L249–289、docs/training/VERIFICATION.md。§11 |
| optimizer | AdamW、2e-4、(0.8, 0.99) | weight_decay 0.01（既定）、eps 1e-8 | 同じ（明示的に 0.01） | MATCH | OFF train.py L163–166、ADB optim.py L20–32 |
| scheduler | ExponentialLR 0.99 / epoch | エポック末に G・D | 同じ | MATCH | OFF train.py L435–436 |
| grad clip | 記述なし | 無し | 無し | MATCH | OFF train.py |
| init | 記述なし | `mamba_ssm` の dt 初期化と `_init_weights` | PyTorch 既定。`dt_proj.bias` が異なる | DIFF | OFF model/semambapp.py L86–97、ADB network.py L60–75。SMB-D11 |
| max epoch/step | 100 epoch | 100 × \(\lfloor N/8\rfloor\) step | 100 epoch ＝ 133,700 step | PARTIAL | OFF config.yaml L35、ADB semambapp.yaml L19 |
| validation data | 記述なし | 固定 100 ペア（`val_*.json` の先頭） | `voicebank_28_valid` | DIFF | OFF dataset.py L97–101。SMB-D10 |
| validation preprocess | 記述なし | 個別ピーク正規化、全長、batch 1 | RMS 正規化、全長 | DIFF | OFF dataset.py L220–244、ADB dataset.py L215–216。SMB-D10 |
| validation batch | 記述なし | 1 | 設定による（学習の micro batch とは別） | UNKNOWN | ADB trainer.py L622–672（未精査） |
| best selection / early stop | 記述なし | 無し。5,000 step ごとの保存のみ | 無し（SEMamba++ は best 無効）。間隔 1,337 | MATCH（方針） | OFF train.py L331–351、ADB trainer.py L741–742。SMB-D13 |
| resume | 記述なし | エポックのやり直し、γ² の余分な減衰、`strict=False` | 厳密再開 | DIFF（意図的） | OFF train.py L55–65、L156–193。SMB-D12 |
| eval metrics PESQ | PESQ（モード不明） | torchmetrics `wb` | `pesq` 0.0.4 `wb` | PARTIAL | §7。SMB-D14 |
| eval metrics STOI | 論文に無し | 無し | あり | N/A | §7 |
| eval metrics SI-SDR | 論文に無し | 無し | あり | N/A | §7 |
| eval metrics 他 | SCOREQ、UTMOS、DNSMOS、LSD、LPS | UTMOS・MRSTFT（検証のみ） | 無し | DIFF | §7。SMB-D14 |
| test | VCTK-GSR test、URGENT 2025、DNS 2020、AATC 2025 | スクリプト無し | VoiceBank-DEMAND test。上記は未登録 | DIFF | §7。SMB-D14 |
| 重み・コードの版 | 論文はバグ版、公開は修正後再学習 | — | 修正後の構造 | PARTIAL | README、ADB network.py。SMB-D15 |

## 9. 差異一覧

影響: Critical = プロトコルが別物になる／Major = 結果に影響しうる／Minor = 軽微／Unknown = 影響不明。実装状態: 未実装、実装済み未検証、検証済み(範囲)。

| ID | 項目 | 影響 | 実装状態 | 優先度 | 対象ファイル |
|---|---|---|---|---|---|
| SMB-D01 | 学習データ生成がオンライン混合（毎回再抽選）か、固定 1 ケースレシピか | Critical | 未実装 | P0 | `src/training/data/dataset.py`、`sampler.py`、新規の再現用 DB と生成器 |
| SMB-D02 | 学習素材（VCTK 0.92−p280/p315 ＋ WHAM!/DNS2020 ＋ Arni/DNS5 の分割）が無く、VoiceBank-DEMAND で学習している。手元の VCTK は 0.80、Arni は無い | Critical | 未実装 | P0 | `docs/db/DATASETS.md`、再現用 DB の素材索引、`data/corpora/` |
| SMB-D03 | 劣化の種類（bandwidth、clipping、確率的な reverb）をレシピ v1 が表せない。RIR のチャンネル平均も表せない | Critical | 未実装 | P0 | `src/evaluation/audio/render.py`、`src/data/recipe.py`、`docs/db/RECIPE.md` |
| SMB-D04 | SNR の定義（有音フレーム基準・reverb 後の信号に対する）と分布（5 dB 刻みの離散一様）が違う | Major（Critical になりうる） | 未実装 | P0 | `src/evaluation/audio/mixing.py` |
| SMB-D05 | 正規化: clean と劣化を別々にピーク正規化（切り出し前）。audio-db は noisy の RMS を 1 にする共通スケール | Major | 未実装 | P0 | `src/training/data/dataset.py`（`_normalize`） |
| SMB-D06 | 雑音の配置（wrap と乱数オフセット、全長から切り出し）、劣化の適用順（reverb→noise→clipping→bandwidth）、教師が dry clean であること | Major | 未実装 | P1 | `src/evaluation/audio/mixing.py`、`render.py` |
| SMB-D07 | 切り出し長: 論文 24,000／コード 48,000。audio-db は 48,000。論文の重みの値は UNKNOWN | Unknown | 実装済み未検証 | P1 | `configs/training/semambapp.yaml`（提案のみ。変更しない） |
| SMB-D08 | サンプラ・シャッフル・エポック。公式は 1 GPU で再シャッフルなし、DDP で rank ごとに別の並び、重複補完。audio-db はエポックごとの決定的 shuffle | Major | 実装済み未検証（audio-db 側は検証済み(小規模)） | P1 | `src/training/data/sampler.py` |
| SMB-D09 | バッチ: 公式は 1 forward で 8。audio-db は micro 4 × 蓄積 2。fused カーネルのバッチ形状差を測定済み | Major | 検証済み(長さ 8,000 の fused、20 step の勾配の分布) | P1 | `configs/training/semambapp.yaml`、`src/training/core/` |
| SMB-D10 | 検証集合・前処理・指標・頻度（公式は固定 100 ペア・5,000 step・PESQ/UTMOS/MRSTFT） | Major | 未実装 | P1 | `src/training/se/semambapp.py`（`validation_report`）、`src/training/core/trainer.py` |
| SMB-D11 | 独立 Mamba 層の `dt_proj.bias`（と dt 初期化）が公式と異なる。ゼロからの学習にだけ影響 | Major（大きさは UNKNOWN） | 未実装 | P0 | `src/se/models/semambapp/network.py`（`SelectiveSSM.__init__`）、`src/training/se/semambapp.py` |
| SMB-D12 | 再開の意味。公式は「エポックやり直し・γ² の余分な減衰」。audio-db は厳密再開 | Minor | 検証済み(audio-db 側) | P2 | `src/training/core/trainer.py`（提案のみ） |
| SMB-D13 | checkpoint の方針。公式は 5,000 step ごと・best なし。論文が評価に使った step は UNKNOWN | Minor | 実装済み未検証 | P2 | `configs/training/semambapp.yaml` |
| SMB-D14 | 評価指標（SCOREQ、UTMOS、DNSMOS、LSD、LPS）と、論文のテストセットが `src/evaluation` に無い | Major | 未実装 | P1 | `src/evaluation/metrics/`、`docs/db/` |
| SMB-D15 | 比較対象が曖昧。論文の重みはバグ版、公開重みは修正後の再学習で、学習データが不明 | Unknown | — | P1 | `docs/se/MODELS.md`、`requirements/official-assets.toml` |
| SMB-D16 | リサンプル・チャンネル処理（librosa soxr_hq とモノ平均、torchaudio の既定フィルタとチャンネル選択） | Minor | 未実装 | P2 | `src/training/data/dataset.py`、`render.py` |
| SMB-D17 | 乱数管理。公式は再現不能、audio-db は決定的。新データ基盤では乱数の保存方式が要る | Minor | 未実装 | P1 | 新規の再現用サンプラ |
| SMB-D18 | 環境。公式は torch 2.10・librosa 0.11、audio-db の `.venvs/se` は torch 2.6.0・librosa 1.0.0・nnAudio 0.3.3（DEVELOPMENT.md） | Minor | — | P2 | `requirements/*.lock.txt`、`.agents/DEVELOPMENT.md` |
| SMB-D19 | ドキュメントと設定の不整合（§11） | Unknown | — | P1 | `docs/training/OFFICIAL_TRAINING_AUDIT.md`、`EXPERIMENT_PLAN.md` |

## 10. DB中心の新データ基盤に必要な情報

### 10.0 原則: 2 種類のものを分ける

- **固定 1 ケースレシピ**: 1 回の抽選結果（素材・SNR・雑音位置・RIR・順序）を全て確定した JSON。`case_id` はそのハッシュ。評価と固定ペアの検証に向く。
- **オンライン生成の確率プロトコル**: 素材集合、抽選分布、適用順の規則、正規化の規則、乱数の仕組みの組。学習のたびに実現値が変わる。`case_id` では表せない。プロトコルの正規化 JSON から `protocol_id`（ハッシュ）を作る形が考えられる（提案）。1 回の実現値を `case_id` として記録するのは、監査用の任意の追加情報になる。

SEMamba++ の学習は後者である。検証の 100 ペアだけが前者に当たる。

### 10.1 音声素材

| 役割 | 公式の素材 | 必要な情報 |
|---|---|---|
| 学習音声 | VCTK 0.92（−p280、−p315） | パス、発話 ID、話者 ID、元の標本化周波数（48 kHz）、長さ、内容ハッシュ、版（0.80 と 0.92 の別） |
| 学習雑音 | WHAM!、DNS 2020 | パス、標本化周波数、チャンネル数、長さ、出自（どちらか）、train/test の別 |
| 学習 RIR | Arni、DNS5（SLR26/28 系） | パス、チャンネル数、標本化周波数、長さ、出自、train/test の別 |
| 検証ペア | `val_clean` / `val_degraded` の先頭 100 件 | 出自・生成条件（UNKNOWN）、ペア対応 |
| 評価素材 | VCTK-GSR test、URGENT 2025 val・blind、DNS 2020 test、AATC 2025 test | 入手の可否、生成条件（VCTK-GSR test は UNKNOWN） |

### 10.2 役割

`speech_clean`（train/val/test）、`noise`（train/test）、`rir`（train/test）、`degraded_pair_clean`、`degraded_pair_noisy`、`eval_*`。同一ファイルが複数の役割に入らない制約が要る。【提案】

### 10.3 分割と対応関係

- 音声: VCTK-GSR train/test。話者単位か発話単位か、順序規則は UNKNOWN。Kuleshov の「最初の 99 話者／残り 8 話者」が候補である。【推測】
- 雑音・RIR: 「分割した」という記述だけで、規則は UNKNOWN。
- 固定ペア: ファイル名末尾の数字で clean と degraded を対応づける（公式の規約）。DB には明示的な対応（`pair_id`）として持たせる。

### 10.4 固定の生成条件

検証 100 ペアの劣化条件（種類・SNR・雑音・RIR）は公開されておらず UNKNOWN。固定ペアとして DB に載せるなら、元の音声ファイルを含めて登録する必要がある。

### 10.5 抽選される条件と分布パラメータ

| パラメータ | 公式の分布（HEAD） | 論文 | 確信度 |
|---|---|---|---|
| 発話 | 学習リストの順（1 GPU では固定順） | 記述なし | 【確認:コード】 |
| 雑音ファイル | 一様・復元 | 記述なし | 【確認:コード】 |
| RIR ファイル | 一様・復元 | 記述なし | 【確認:コード】 |
| 採用する劣化 | noise 1.0、reverb 0.5、bandwidth 0.5、clipping 0.5（独立） | 記述なし。orig は 0.7 | 【確認:コード】 |
| SNR | {−5, 0, 5, 10, 15, 20} から一様 | −10〜20 dB | 【確認:コード】【確認:論文】 |
| 雑音オフセット | 一様整数（`default_rng(None)`） | 記述なし | 【確認:コード】 |
| bandwidth | `fs_new` ∈ {4k…14k}（折り返し 2–7 kHz）、リサンプル種別 4、ローパス種別 5、次数 {2,4,6,8}、一様 | 2–7 kHz、種別（Chebyshev 1、Butterworth 等）、次数 2–8 | 【確認:コード】【確認:論文】 |
| clipping | 下側分位 U(0, 0.1)、上側 U(0.9, 1.0) | 記述なし | 【確認:コード】 |
| 切り出し開始位置 | 一様整数 | 記述なし | 【確認:コード】 |
| 適用順 | reverb → noise → clipping → bandwidth | 記述なし | 【確認:コード】 |

### 10.6 loader 固有の処理

- 個別ピーク正規化（clean と劣化を別々、切り出し前）。
- 圧縮 0.3 の magnitude・phase・complex を Dataset 内で作る（MP-SENet 系と同じ形の STFT。共有できるかは別途確認が要る）。
- 切り出しが劣化の**後**で、劣化は全発話長に適用される（このため、1 サンプルのコストが発話長に比例する）。
- `librosa.load` の既定（モノ平均、soxr_hq）。
- 検証ペアのファイル名規約（末尾の数字でのソート、先頭 100 件）。

### 10.7 共有できる処理

- 加法雑音の SNR 混合、RIR 畳み込み、bandwidth 制限、clipping は、他モデル（URGENT 系のプロトコルを使う場合）と共通の部品にできる。【提案】
- 素材索引、分割、固定ペアの管理、評価指標。
- サンプラの状態管理（エポック・位置・乱数）。

### 10.8 専用実装が要る処理

- 有音フレーム基準の SNR（URGENT 系の `mix_noise` 相当）。
- 雑音の wrap 配置と乱数オフセット。
- 個別ピーク正規化と、切り出しが劣化の後であること。
- 確率的な劣化選択と固定順序。
- 公式の 1 GPU 順序（一度きりの shuffle）と DDP の並びを再現するか、決定的な代替にするかの決定。

### 10.9 再現のために保存する状態

| 状態 | 理由 |
|---|---|
| プロトコル ID と素材リストのハッシュ（発話・雑音・RIR それぞれ） | リストが変わると 1 エポックの意味が変わる |
| 乱数のシードと生成方式（エポック・サンプル ID から決定的に作るか、ストリームを保存するか） | 公式は無シードで再現不能。再現するならサンプル単位の決定的な導出が必要 |
| エポック番号、サンプラの位置、DDP の rank 分割 | 厳密再開 |
| chunk 計画（このモデルでは 1 発話 1 切り出し） | エポック定義 |
| 実現された条件のログ（任意） | 後からの監査 |
| ライブラリ版（librosa、scipy、soxr、numpy、torch） | リサンプル・フィルタの数値が版で変わる |

### 10.10 レシピ JSON v1 で表せないもの

| 公式の処理 | v1 の対応 |
|---|---|
| 確率的な劣化選択 | 不可（1 回の結果しか書けない） |
| SNR の離散一様分布 | 不可。1 値は書ける |
| 有音フレーム基準の SNR | 不可（`rms_v1` のみ） |
| reverb 後の信号に対する SNR | 可（`rir` の後に `additive_noise`）。ただし定義が `rms_v1` |
| 雑音の wrap と乱数オフセット | 不可。繰り返し＋保存した開始位置のみ |
| bandwidth 制限、clipping | 不可（操作が無い） |
| RIR のチャンネル平均（`librosa` の `mono=True`） | 不可（チャンネル 1 つを選ぶ） |
| clean と劣化の個別ピーク正規化 | 不可（`peak_normalize` は上限制限） |
| 全発話長に適用してから切り出す | 切り出しはレシピ外（Dataset 側） |
| エポックごとの再抽選 | 不可（`case_id` が固定のため） |

## 11. 論文と公式コードの矛盾、既存ドキュメントの不整合・誤り

### 11.1 論文と公式コード

| 項目 | 論文 | コード | 備考 |
|---|---|---|---|
| 切り出し長 | 24,000（1.5 秒）、PAP §4.1 | 48,000（3 秒）、OFF config.yaml L41 | 重みの学習にどちらを使ったかは UNKNOWN |
| SNR | −10〜20 dB | HEAD は −5〜20（6 値）、`config_orig.yaml` は −10〜20（7 値） | HEAD の変更は 2026-06-05 |
| 劣化確率 | 記述なし | orig 0.7、HEAD 0.5 | 論文の値は不明 |
| magnitude 損失 | L1（§3.4） | MSE（OFF train.py L279） | コード優先で audio-db は MSE |
| 損失の `time` と `metric` | 論文に無い | 設定にあるが未使用 | 一致 |
| 重み | 論文の表の重み | バグ修正後の再学習が公開重み | 論文 v3 に訂正の記述が無い |
| 学習データ | VCTK 44 時間（Table 3） | 旧 README は VCTK ＋ LibriTTS 約 500 時間と記載 | 公開重みの学習データは UNKNOWN |
| 位相損失 | MP-SENet / SEMamba 系に従う | 同様 | 一致 |

### 11.2 公式リポジトリ自体の不具合

- `train.py` の既定の設定ファイル名が `semambapp_cfg.yaml`（L447）で、リポジトリには `config.yaml` しか無い。
- `dataset.py` が `utils.util.load_config` を import するが（L11）、`utils.util` は無い（`utils.py` のみ）。
- `data/*.json` は 1 バイトの空ファイル。
- `infer_folder.py` が `from infer import load_model, restore_file` とするが（L22）、`infer.py` にこの 2 関数の定義は無い（定義は `infer(args)` のみ）。学習には影響しないが、フォルダ推論は動かない。
- `librosa.resample(noise_audio, noise_sr, self.sampling_rate)` は位置引数で、librosa 0.11 では不正だが、`librosa.load(sr=…)` の直後なので到達しない（L157–161）。

以上から、公式の学習コードは**そのままでは動かない**（`dataset.py` の import、設定ファイル名、空のマニフェスト）。audio-db は公式のステップ・損失・モデルを別プロセスで比較した検証（VERIFICATION.md）をしているが、これは公式コードを部分的に直して動かしたものである。【確認:コード】（動かない箇所の列挙は目視。全体を実行して確認したわけではない）

### 11.3 audio-db の既存ドキュメント・設定の不整合

| 箇所 | 内容 |
|---|---|
| `docs/training/OFFICIAL_TRAINING_AUDIT.md` L11、L20、L45、`EXPERIMENT_PLAN.md` L39 | `configs/training/semambapp_seed17.yaml`（ほか `*_seed17.yaml`）を参照するが、この作業ツリーに存在しない。存在するのは `semambapp.yaml`（seed 42）だけ |
| `configs/training/semambapp.yaml` L1、L15 | seed 42 と書く。VERIFICATION.md 等は seed 17 の結果を述べる |
| `configs/training/semambapp.yaml` L25 | `best_checkpoint_start_epoch: 40` があるが、SEMamba++ では best 無効で効かない。L11–12 のコメントと食い違って見える |
| `configs/training/semambapp.yaml` L56 | `data/db/voicebank.duckdb` を指すが、`data/db/` は `.gitkeep` だけ。AGENTS.md の `data/db/resource.duckdb` と退避ファイルも見当たらない（確認のみ。触れていない） |
| `configs/evaluation/`、`trial100/` | docs が参照するが、この作業ツリーに無い |
| `docs/training/OFFICIAL_TRAINING_AUDIT.md` L11 | 「共通の VoiceBank-DEMAND は研究上の共通条件で、各論文のデータセット再現ではない」と明記している。これは正しい。他の箇所で「公式一致」と読める記述（例: 損失が GO）は、プロトコルの一致を意味しない |
| `docs/training/VERIFICATION.md` 各所 | 検証が「縮小幅・16 kHz・8,000 サンプル・VoiceBank」であることを述べているが、公式の学習データ・劣化との差は扱っていない |
| `requirements/official-assets.toml` と `src/se/models/semambapp/NOTICE` | 旧 URL（`infected4098`）と新 URL（`Eddieyjlee`）が混在 |
| `.agents/DEVELOPMENT.md` | `.venvs/se` の nnAudio 0.3.3 に対し、公式の要件は nnaudio 0.3.4 |

## 12. UNKNOWN一覧

| # | 内容 | 確認したこと | 未解決の理由 | 追跡方法 |
|---|---|---|---|---|
| U1 | 学習発話数 \(N\)、話者、長さ | `data/train_speech.json` は空。論文は 44 時間、VCTK 0.92 から 2 話者を除くと記述 | マニフェスト非公開 | 著者への問い合わせ、または VCTK 0.92 の話者分割を [50] から導く |
| U2 | VCTK-GSR の分割規則 | 論文は [50] に従う。[50] は VCTK の「99 話者で学習・8 話者でテスト」 | 話者順序・除外後の数が不明 | 著者への問い合わせ |
| U3 | 雑音・RIR の分割規則と件数 | 論文は「分割した」とだけ | 規則・件数が不明 | 同上 |
| U4 | 検証 100 ペアの出自と生成条件 | コードは先頭 100 件を使うとだけ | ファイルが非公開 | 同上 |
| U5 | 論文の VCTK-GSR test の生成条件 | 論文に記述なし | 非公開 | 同上 |
| U6 | 論文が評価に使った checkpoint の step | 公式は 5,000 step 間隔で保存、best なし | 論文に記述なし | 著者への問い合わせ |
| U7 | 切り出し長（24,000 か 48,000）と、再学習の設定 | README は「同じ設定で再学習」 | どの設定か不明 | 著者への問い合わせ |
| U8 | 公開重みの学習データ | README の現行版に記述なし。旧版は VCTK ＋ LibriTTS 約 500 時間 | 版により異なる | 著者への問い合わせ |
| U9 | 学習に使った GPU 数・総時間・seed | 設定は seed 1234、`world_size: 1` だが未使用 | 記述なし | 同上 |
| U10 | PESQ のモード、LSD の定義、SCOREQ・DNSMOS の版 | 論文は SCOREQ（合成向け・no-reference）、DNSMOS（SIG/BAK/OVRL）、LPS は URGENT 公式実装としか書かない | 評価スクリプト非公開 | 著者への問い合わせ、URGENT の公式実装の確認 |
| U11 | 手元の VCTK 0.80 と論文の 0.92 の差が学習・評価に与える影響 | 版・p280/p315 の状態の違いは確認した | 実データでの検証は未実施 | VCTK 0.92 の入手可否の確認 |
| U12 | WHAM! の版（16 kHz 版か高解像度版か）と DNS 2020 雑音の使用範囲 | 手元は `WHAM48/high_res_wham`、DNS2020 は全体 | 論文に記述なし | 著者への問い合わせ |
| U13 | 独立 SEMamba++ の `dt_proj.bias` 初期化の影響の大きさ | コードの差は確認した | ゼロからの学習での影響は未測定 | 小規模な学習比較（GPU が要る。今回は未実施） |
| U14 | 公式 DDP の挙動（rank 間の重複・欠落、検証の rank 0 のみの forward） | コードを読んだ | 実行していない | 小さな合成データで 2 GPU 実行 |
| U15 | torch 2.10 での再開時の学習率の挙動 | torch 2.6.0（CPU）で γ² を確認 | 2.10 は未確認 | 同じ小再現を 2.10 環境で実行 |
| U16 | `data/db/` の内容（AGENTS.md は DuckDB の存在を述べる） | この作業ツリーでは空 | 別の場所にある可能性 | ユーザーに確認 |

## 13. 参照した一次資料

### 論文

- arXiv:2603.11669 v1 / v2 / v3、"SEMamba++: A General Speech Restoration Framework Leveraging Global, Local, and Periodic Spectral Patterns"（Yongjoon Lee, Jung-Woo Choi）。v1 2026-03-12、v3 2026-08-26。§3.4（目的関数）、§4.1 Training、§4.2 Evaluation、Table 1–3。
- arXiv:1708.00853 v1、Kuleshov ら "Audio super resolution using neural networks"（VCTK の分割）。
- 関連: arXiv:2405.06573（SEMamba）。

### 公式コード（OFF = `Eddieyjlee/SEMambapp` @ `cd369c89a8106072c077b468417b765b227afd74`）

- `train.py`（L45–65、L68–204、L206–436、L441–487）、`dataset.py`（L36–279）、`simulate_degradation.py`（L13–111）、`simulate_degradation_utils.py`（L58–303）。
- `config.yaml`、`config_orig.yaml`、`model/semambapp.py`（L61–97）、`model/loss.py`、`model/discriminator.py`、`evaluation_metrics.py`（L8–39）、`requirements.txt`、`README.md`（L1–125）。
- `git log`（`config.yaml`、`config_orig.yaml`、`train.py`、`dataset.py`、`README.md`）。
- HF: `https://huggingface.co/yongjoonlee/semambapp`（ライセンス・更新日・ファイル一覧）。
- GitHub API: `Eddieyjlee/SEMambapp`（リポジトリ・ユーザー情報）。
- `mamba_ssm` 2.3.2.post1（`modules/mamba_simple.py`、`models/mixer_seq_simple.py` の `_init_weights`）は `.venvs/semambapp` のものを読んだ。

### audio-db（ADB = HEAD `3bee9ad`）

- `AGENTS.md`、`.agents/{PROJECT,DATA,DEVELOPMENT,DECISIONS}.md`。
- `docs/training/{OFFICIAL_TRAINING_AUDIT,OTHER_MODELS,VERIFICATION,EXPERIMENT_PLAN,TENSORBOARD,RESIDUAL_20261104}.md`、`docs/se/MODELS.md`、`docs/db/{RECIPE,TRAINING_SETS,CORPORA,DATASETS}.md`。
- `requirements/official-assets.toml`、`configs/training/semambapp.yaml`。
- `src/training/data/{dataset,sampler}.py`、`src/training/core/{trainer,batch,batch_schedule,rng}.py`、`src/training/se/{semambapp,optim,spectral,initialization}.py`。
- `src/evaluation/audio/{mixing,render}.py`、`src/evaluation/metrics/{pesq,sisdr,stoi}.py`。
- `src/se/models/semambapp/{network.py,inference.py,NOTICE}`。

### 作業用の一時ファイル（`/tmp/audit-semambapp/`）

論文の HTML・PDF とテキスト、公式の clone、Kuleshov の PDF とテキスト。CPU の小さな再現（`ExponentialLR` を 2 つ作った再開の学習率）は、標準入力から実行しただけで、ファイルは残していない。

## 付録 A. 提案（実装していない）

1. 再現用 DB には、素材・役割・分割・プロトコル（確率分布と順序規則）・サンプラ状態を持たせ、レシピ JSON とは別の `protocol_json` を正本とする。
2. 公式の学習経路を再現するモードでは、劣化の抽選をサンプル ID とエポックから決定的に導く（公式の無シード動作は再現しない）。
3. 独立 SEMamba++ に、Mamba の dt 初期化（`dt_proj.bias`）を公式と同じ分布で与える初期化を足す。足す前後の学習比較は GPU が要る。
4. 論文と同じ指標（UTMOS、DNSMOS、LSD、LPS、SCOREQ）は、各実装の版と入力条件を固定して `src/evaluation/metrics/` に足す。
5. 公式の再開の挙動（γ²）は再現せず、audio-db は厳密再開のままとし、差として記録する。
