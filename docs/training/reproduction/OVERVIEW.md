# 6 モデル 公式再現プロトコル監査: 全体比較

調査日 2026-11-06。ブランチ `feature/reproduction`、audio-db HEAD `3bee9ad`。作業ツリーは未コミットの `docs/training/reproduction/` 以外に変更なし。

この文書は 6 本のモデル別監査を統合した全体表である。個々の事実、行番号、公式ソースの引用は各モデル別文書にある。この文書だけで事実を確定せず、必ずモデル別文書の該当節を見る。

2026-11-06 の実装後: 下文が「現行」と呼ぶ学習データは、監査時点の共通 VoiceBank 経路である。再現用の学習入口は別で、MP-SENet と MossFormerGAN は 48 kHz 固定ペアの学習時リサンプル、TF-GridNet は MS-SNSD の条件からの再構築、TF-Locoformer は DNS の条件からの再構築である。生成済み WAV は必須にしない。DeepFilterNet3 と SEMamba++ のオンデマンド実装は未着手である。TF-Locoformer の現行仕様は [P7_VERIFICATION](P7_VERIFICATION.md)。この概要の差分表は監査当時の記録である。

| モデル | 文書 |
|---|---|
| DeepFilterNet3 | [deepfilternet3.md](deepfilternet3.md) |
| MossFormerGAN | [mossformergan.md](mossformergan.md) |
| MP-SENet | [mp_senet.md](mp_senet.md) |
| SEMamba++ | [semambapp.md](semambapp.md) |
| TF-GridNet | [tf_gridnet.md](tf_gridnet.md) |
| TF-Locoformer | [tf_locoformer.md](tf_locoformer.md) |

今回は調査のみで、学習コード、`configs/training/*.yaml`、DB スキーマ、データ、checkpoint は変更していない。

## 0. 総合判定

**監査完了（一部に UNKNOWN が残る）。** 6 モデルすべてについて、公式の学習・データ生成・データローダー・評価を調べ、現行 audio-db との項目別比較と差異一覧（合計 103 件、うち Critical 12 件）を作った。確定できなかった項目は各文書の「UNKNOWN 一覧」にあり、本書 §10 に集約した。

最も重要な結論は次の 4 点。

1. **現行の学習データ経路は、6 モデルのどれの公式データプロトコルとも一致しない。** 全モデル共通で VoiceBank-DEMAND 28 話者の固定ペア（train 10,700 / valid 872）を使い、既定で発話 RMS 正規化をかけている。損失・optimizer・scheduler・更新順は、コードを読んだ範囲で公式とよく合っている。データとプロトコルが共通のボトルネックである。
2. **公式のデータ方式は 3 種類に分かれる。** オンライン混合は DeepFilterNet3 と SEMamba++。事前生成は TF-Locoformer（DNS）と TF-GridNet（MS-SNSD）。固定ペアは MP-SENet と MossFormerGAN の公開コード。現行のレシピ JSON v1 が表せるのは、固定ペアと「事前生成済みファイルの登録」だけである。
3. **「公式」の指すものが曖昧なモデルが 3 つある。** MossFormerGAN は専用論文が見つからない。TF-GridNet は著者の単独リポジトリが無く ESPnet が実装で、論文の実験に単一話者 SE は無い。SEMamba++ は配布重みが論文の重みと別物である。この 3 つは再現対象の決定に研究上の判断が要る（§9）。
4. **既存ドキュメントに、公式仕様として誤っている記述がある。** TF-Locoformer の arXiv ID（誤り）、TF-GridNet と TF-Locoformer の `valid si_snr max`（ピン留め ESPnet では働かない）、存在しない設定ファイルへの参照などである（§11）。

## 1. 調査対象の固定

| モデル | 公式リポジトリ | 調査 commit / tag | 論文 | 配布 checkpoint |
|---|---|---|---|---|
| DeepFilterNet3 | [Rikorose/DeepFilterNet](https://github.com/Rikorose/DeepFilterNet) | `d375b2d`（v0.5.6 の 89 commit 後） | [arXiv:2305.08227](https://arxiv.org/abs/2305.08227) v1（Interspeech 2023、4 ページ） | `models/DeepFilterNet3.zip` と同梱 `config.ini` |
| MossFormerGAN | [modelscope/ClearerVoice-Studio](https://github.com/modelscope/ClearerVoice-Studio) | `6b3774d`（v0.1.2 の 35 commit 後） | **専用論文なし**。ツールキット論文 [arXiv:2506.19398](https://arxiv.org/abs/2506.19398) v1 の記述のみ | HF `alibabasglab/MossFormerGAN_SE_16K` |
| MP-SENet | [yxlu-0102/MP-SENet](https://github.com/yxlu-0102/MP-SENet) | `89932cf`（main HEAD） | journal 版 [arXiv:2308.08926](https://arxiv.org/abs/2308.08926)（主）、Interspeech 版 [arXiv:2305.13686](https://arxiv.org/abs/2305.13686)（旧 commit `f37e211` 以前） | `best_ckpt/g_best_vb`、`g_best_dns`（repo 内） |
| SEMamba++ | [Eddieyjlee/SEMambapp](https://github.com/Eddieyjlee/SEMambapp)（旧名 `infected4098/SEMambapp`） | `cd369c8`（2026-10-26）。ライセンスファイル無し | [arXiv:2603.11669](https://arxiv.org/abs/2603.11669)（v1 2026-03-12、v3 2026-08-26） | HF `yongjoonlee/semambapp`（GPL-2.0 表示、論文の重みとは別） |
| TF-GridNet | [espnet/espnet](https://github.com/espnet/espnet)（PR #5395） | `0d136c5`（audio-db のピン）。履歴は別 clone で調査 | [arXiv:2209.03952](https://arxiv.org/abs/2209.03952)（ICASSP 2023）、[arXiv:2211.12433](https://arxiv.org/abs/2211.12433)（TASLP 2023） | HF `espnet/ms_snsd_tfgridnet`（audio-db が採用） |
| TF-Locoformer | [merlresearch/tf-locoformer](https://github.com/merlresearch/tf-locoformer) | tag `v1.1.0` = `7a61546`。ESPnet は `90eed8e`（公式動作確認版）と `0d136c5`（audio-db）の両方 | [arXiv:2408.03440](https://arxiv.org/abs/2408.03440) v1 | `valid.loss.ave_5best.pth`（Git LFS。手元は 133 バイトのポインタのみ） |

注意:

- **audio-db のピン commit は「比較用に固定した版」であり、公式の学習時の版ではない。** DeepFilterNet3 の出荷モデルは 2023-05-23 のもので、ピンは v0.5.6 の 89 commit 後である。gain、局所 SNR 目標のクランプ、`min_nb_erb`、合成雑音、残響の遅延が学習時と違う（deepfilternet3.md §2.3）。
- 手元の ESPnet は `espnet2` のみの sparse checkout で、`egs2/` は無い。レシピ設定は別 clone で確認した。
- 論文の ID を、この監査で arXiv の記録と照合した。`docs/se/MODELS.md` の TF-Locoformer の ID `2409.16674` は別分野の論文であり誤り。正しくは `2408.03440`。

## 2. 6 モデルの公式プロトコル概要

| 項目 | DeepFilterNet3 | MossFormerGAN | MP-SENet | SEMamba++ | TF-GridNet | TF-Locoformer |
|---|---|---|---|---|---|---|
| 採用 variant | 出荷 DFN3（48 kHz、非 low-latency） | `MossFormerGAN_SE_16K`（16 kHz、公開コードの `T/` 学習） | journal 版 VB-DEMAND レシピ（main HEAD） | GSR 論文の公式実装（`config.yaml`） | ESPnet `egs2/ms_snsd/enh1`（v1）。選定は未決（§9） | `egs2/dns_ins20` の Medium |
| サンプリング | 48 kHz | 16 kHz | 16 kHz | 48 kHz 系（クロップ 48,000 サンプル） | 16 kHz | 16 kHz |
| 学習データ方式 | オンライン混合 | 固定ペア（VB-DEMAND）。配布 ckpt は別レシピ（不明） | 固定ペア（VB-DEMAND） | オンライン混合（GSR） | 事前生成（MS-SNSD） | 事前生成（DNS Challenge の合成スクリプト） |
| 学習区間 | 3 秒窓（連結で埋める） | 2 秒 crop（32,000） | 2 秒 crop（32,000） | 3 秒 crop（48,000） | 3 秒チャンク（48,000、shift 0.5） | 2 秒チャンク（audio-db は 32,000。論文系は 64,000） |
| 学習ループ | 120 epoch、batch schedule 16→64、patience 25 | 121 ループ、micro 1 × 蓄積 2（per GPU）、plateau 5/10 | 400 epoch（コード既定。論文と不一致）、batch 4 | 100 epoch、batch 8、1 forward | 35 epoch、global batch 4（2 rank × 2） | 150 epoch、global batch 4（4 rank × 1） |

## 3. 公式データセット一覧

| モデル | clean 音声 | noise | RIR | 分割 | 件数・時間 | 確度 |
|---|---|---|---|---|---|---|
| DeepFilterNet3 | DNS4 全体 + PTDB / VCTK ×10（論文の記述） | DNS4 等（詳細不明） | 不明（シミュレーション含む） | 不明 | **不明**（`dataset.cfg` 非公開） | 【確認:論文】件数は UNKNOWN |
| MossFormerGAN（公開コード） | VoiceBank 28 話者 11,572 ペア | DEMAND（混合済み） | なし | train 11,572、validation = 公式 test 824 | 作者 NAS 上の事前 16 kHz 版 | 【確認:コード】 |
| MossFormerGAN（配布 ckpt、論文記述） | clean 850 h | 300 h | 合成 RIR 10 万本 | 8000 h train / 50 h dev | 混合は clean:残響 = 7:3、SNR 0〜15 dB | 【確認:論文】生成方式は UNKNOWN |
| MP-SENet（VB） | VoiceBank 11,572（28 話者） | DEMAND（混合済み） | なし | train 11,572、test 824（p232・p257） | 原音 48 kHz。公式は 16 kHz 版、変換器は UNKNOWN | 【確認:コード】 |
| MP-SENet（DNS） | DNS Challenge | 同 | 同 | 不明 | 論文: 3,000 h、SNR −5〜15 dB、10 秒に分割。**公式にコード無し** | 【UNKNOWN】 |
| SEMamba++ | VCTK 0.92（p280・p315 を除く） | WHAM! + DNS2020 | Arni + DNS5 | 規則は不明 | 件数 N は不明 | 【確認:論文】分割・件数は UNKNOWN |
| TF-GridNet（ms_snsd） | MS-SNSD `clean_train` 23,075 | `noise_train` 128 | なし | 80/20（ファイル名順） | 30 h（5 SNR コピーを数えた値。一意なクリップは約 6 h） | 【確認:コード】生成件数は UNKNOWN（未シード乱数） |
| TF-Locoformer（dns） | DNS2020 clean 65,348 本 | DNS2020 noise 65,302 本 | **学習に使わない** | uttid 辞書順の先頭 9/10 を train | 3,000 h（30 秒 × 360,000。train 2,700 h / valid 300 h） | 【確認:コード】`cv_synthetic_small` は UNKNOWN |

データセット名が同じでも別物である例:

- **VoiceBank-DEMAND**: audio-db の train 10,700 / valid 872 は、公式の 11,572 / 824 とは別の分割である。MP-SENet と MossFormerGAN の公式は、test 824 を validation と best 選択にも使う。
- **MS-SNSD**: `snr_mixer` が名目 SNR の半分の SNR を作る（名目 0〜40 dB が実際は 0〜20 dB）。ローカルの `audiolib.py` で数値確認した（tf_gridnet.md §0 項目 5）。
- **DNS**: SEMamba++ が使う DNS2020 の素材と、TF-Locoformer の合成スクリプトの出力は別である。

## 4. オンライン / 事前生成 / 固定ペアの分類

| モデル | 学習 | 学習中の validation | 評価（論文） | 根拠の要点 |
|---|---|---|---|---|
| DeepFilterNet3 | **オンライン混合**（Rust `libdfdata`、発話ごとに seed 付き） | 同じ生成器だが seed が `idx` のみで毎回同じ結果 | 固定ペア（VB-DEMAND test 824） | `libDF/src/dataset.rs`、`dataloader.rs` |
| MossFormerGAN | **固定ペア**（公開コード）。配布 ckpt は UNKNOWN（オフラインと推測） | VB test 824、2 秒ランダム crop | 固定ペア | `T/dataloader/dataloader.py`、scp 2 列 |
| MP-SENet | **固定ペア** | VB test 824 | 固定ペア | `dataset.py` に混合処理が無い |
| SEMamba++ | **オンライン混合**（GSR。発話ごと・毎エポック） | 固定 100 ペア | 論文の複数テストセット | `dataset.py`、`simulate_degradation.py` |
| TF-GridNet | **事前生成**（`preprocessor: null`） | 同じ事前生成の cv 分割 | 事前生成 | `egs2/ms_snsd/enh1/conf/tuning/*.yaml` |
| TF-Locoformer | **事前生成**（`dynamic_mixing: false`、`rir_scp`・`noise_scp` は null） | 事前生成の `cv_synthetic_small` | DNS 合成 150 本 | `dns_create_mixture.sh`、`config.yaml` |

## 5. 学習条件の比較（実効値）

| 項目 | DFN3 | MossFormerGAN | MP-SENet | SEMamba++ | TF-GridNet | TF-Locoformer |
|---|---|---|---|---|---|---|
| optimizer | AdamW、`amsgrad=True` | AdamW（G 5e-4 / D 1e-3） | AdamW 5e-4、betas (0.8, 0.99) | AdamW 2e-4、betas (0.8, 0.99) | Adam 1e-3 | AdamW 1e-3、wd 1e-2 |
| weight decay（実効） | 1e-12 → 0.01（cosine） | **0.01**（YAML の 1e-5 は渡されない） | **0.01**（省略） | **0.01**（省略） | 0 | 1e-2 |
| scheduler | cosine（warmup 3 epoch、下限 1e-6） | なし。独自 plateau（5 回で G の lr を半減、10 回で停止） | ExponentialLR 0.99（epoch 終了時） | ExponentialLR 0.99（epoch ごと） | ReduceLROnPlateau 0.7 / patience 1 | 4,000 step の線形 warmup + ReduceLROnPlateau 0.5 / patience 3 |
| 勾配クリップ | 1.0（固定） | 10（micro-batch ごと） | なし | なし | 5.0（YAML に無く `--grad_clip` 既定値。論文は 1.0） | 5 |
| batch | 16→24→32→64（上限 64） | micro 1 × 蓄積 2（per GPU） | 4（drop_last） | 8（1 forward） | global 4（2 rank × 2） | global 4（4 rank × 1） |
| 損失 | MR-spec 500/500、局所 SNR 1e-3 | 0.1 RI + 0.9 mag + 0.2 time + 0.05 GAN | mag 0.9、phase 0.3、complex 0.1、consistency 0.1、metric 0.05、time 0.2 | adv 1、fm 1、mel 0.1、mag 0.9、pha 0.3、com 0.1、con 0.1 | `mr_l1_tfd` のみ | `mr_l1_tfd` のみ |
| Discriminator | なし | あり（PESQ 目標） | あり（PESQ 目標、D→G） | あり（CQT + MR、D→G） | なし | なし |
| 1 epoch の意味 | 全発話 1 周（端数 batch は残す） | 全発話 1 周（余りの蓄積は捨てる） | 全発話 1 周 | 全発話 1 周 | **rank あたり 5,000 元発話**（更新数は約 15,000 と推定） | **rank あたり 1,000 元発話**（全 4,000、約 14,000 更新） |
| 最大 epoch | 120 | 120（121 ループ） | 400（コード既定）。論文は 100 / 120 / 500k step | 100 | 35 | 150 |
| 早期停止 | valid loss、patience 25 | 10 回（plateau） | なし | なし | valid loss、patience 5 | valid loss、patience 10 |
| best の選択 | valid loss 最小 | `last_best_checkpoint`（配布名） | validation = test.txt の PESQ 最大、40 epoch 以降 | なし（5,000 step ごと保存） | `valid.loss.best` のみ（`si_snr` は働かない） | `valid.loss.ave_5best`（5-best 平均）で評価 |
| 初期化 | 既定 | 既定 | 既定 | Mamba の dt 初期化 | **xavier_uniform**（独自 LayerNorm の gamma/beta も乱数化） | xavier_uniform |
| DDP | なし | あり（rank 固定の分割、seed 0 固定） | 単 GPU（複数 GPU では `DistributedSampler`） | あり（rank ごとに独立の並び） | 2 rank | 4 rank |

設定値と実効値が食い違う主な箇所:

- MossFormerGAN の weight decay、MP-SENet と SEMamba++ の weight decay: YAML に無い、または渡されないため PyTorch 既定の 0.01 になる。
- TF-GridNet の勾配クリップ: YAML に無く、`--grad_clip` の既定値が実効値になる。
- TF-GridNet と TF-Locoformer の `valid si_snr max`: ピン留め ESPnet の stats のキーは `si_snr_loss` で `si_snr` ではなく、基準は一度も働かない。実効は `valid/loss/min` のみ。
- MP-SENet の `num_tsconformers` は設定が無視され既定 4 になる。

## 6. 評価条件の比較

### 6.1 原論文・公式の評価

| モデル | 評価データ | 指標と実装 | 論文・公式の報告値 |
|---|---|---|---|
| DFN3 | VB-DEMAND test 824 | 48 kHz 読込（`sinc_fast`）、PESQ wb（16 kHz）、STOI（10 kHz）、SI-SDR（speechmetrics 型）、CSIG / CBAK / COVL（`composite`） | PESQ 3.17、CSIG 4.34、CBAK 3.61、COVL 3.77、STOI 0.944（論文 Table 1） |
| MossFormerGAN | VB test（16 kHz） | `speechscore/`: PESQ wb（`pesq` 0.0.4）、STOI（`pystoi` 0.3.3）、SI-SDR は平均除去なし | README: PESQ 3.47、STOI 0.96、SI-SDR 19.45。**VB で学習したモデルの値ではない**。論文 Table 1（DNS no_reverb）: PESQ 3.57、STOI 98.05 |
| MP-SENet | VB test 824 | PESQ wb、CSIG / CBAK / COVL、SSNR、自前実装の STOI。DNS 用に NB-PESQ、`pysepm` の STOI、SI-SDR | Interspeech 版: PESQ 3.50、CSIG 4.73、CBAK 3.95、COVL 4.22、STOI 0.96。journal 版: PESQ 3.60、CSIG 4.81、CBAK 3.99、COVL 4.34 |
| SEMamba++ | VCTK-GSR test、URGENT 2025 val・blind、DNS2020 実録、AATC 2025 | SCOREQ、UTMOS、DNSMOS、PESQ、LSD、LPS。**公式リポジトリに評価スクリプト無し** | VCTK-GSR test: PESQ 1.77、LSD 1.25、LPS 0.45 |
| TF-GridNet | ESPnet scoring: STOI / ESTOI（pystoi ×100）、SDR（`mir_eval`）、SI_SNR。PESQ は既定で無効 | `ms_snsd` README: cv STOI 88.49、SDR 20.16、SI_SNR 19.90 / tt STOI 96.01、SDR 17.84、SI_SNR 17.52。論文は PESQ が NB（python-pesq 0.0.2） | 論文: WSJ0-2mix SI-SDRi 23.4 dB |
| TF-Locoformer | DNS 合成 no_reverb / with_reverb 各 150 本（10 秒） | STOI ×100、SDR/SAR/SIR（bss_eval）、SI_SNR。PESQ は `--use_pesq` 時のみ。推論は発話全体、出力はピーク 0.9 に正規化 | RESULTS.md: STOI 98.79、SI-SNR 23.23。論文: SI-SNR 23.3、STOI 98.8、PESQ-WB 3.72（RESULTS.md に無く算出方法は UNKNOWN） |

### 6.2 現行 `src/evaluation/` との比較

現行の評価（`src/evaluation/metrics/`）の実効動作は次のとおり。

- PESQ は `pesq==0.0.4`。16 kHz は wb、8 kHz は nb、それ以外は 16 kHz にリサンプルする。長さ境界の不連続検査（しきい値 0.1）で `na` にする独自ガードがある。
- STOI は `pystoi`、拡張版なし、16 kHz にリサンプルする。
- SI-SDR は float64、平均除去あり。
- 長さの補正は短い方に切るだけで、遅延の補正はしない。
- CSIG / CBAK / COVL、SSNR、SDR / SAR / SIR、DNSMOS、UTMOS、SCOREQ、LSD、LPS は無い。

主な不一致:

| 項目 | 状況 |
|---|---|
| PESQ のモード | TF-GridNet の論文は NB、TF-Locoformer の論文は WB、DFN3・MP-SENet・MossFormerGAN は wb。audio-db は 16 kHz で wb。論文値と**直接比較できない**モデルがある |
| 評価サンプリングレート | audio-db は VB test を 48 kHz で評価し、SE 内で 48→16→48→16 kHz の往復がある（MossFormerGAN、MP-SENet）。公式は 16 kHz 版を使う |
| SI-SDR | MossFormerGAN の公式は平均除去なし。audio-db は平均除去あり |
| 指標の欠落 | MP-SENet、DFN3 の CSIG / CBAK / COVL、SSNR。TF-Locoformer、TF-GridNet の SDR。SEMamba++ の全指標（PESQ 以外） |
| 評価セット | DNS 合成テスト（TF-Locoformer）、16 kHz 版 VB test（MossFormerGAN）が名前付き評価セットに無い。with_reverb の参照は残響付き clean で、recorded recipe の参照名が `dry_clean` 固定のため表せない |

WER による固定 ASR 評価は原論文の SE 評価とは別の実験であり、上の表には含めていない。

## 7. 現行実装との一致状況

判定はモデル別文書 §8 の項目別表による。MATCH は、コードを読んで確認した範囲での一致で、数値が一致することの証拠ではない。

| 領域 | 状況 | 補足 |
|---|---|---|
| 損失・optimizer・scheduler・更新順 | **おおむね MATCH**（MP-SENet、SEMamba++、DFN3、MossFormerGAN、TF-Locoformer、TF-GridNet） | SEMamba++ の `dt_proj.bias` 初期化（SMB-D11）、TF-GridNet の xavier 初期化（TFG-D05）は DIFF。既存記録の「数値一致」は本監査では再実行していない |
| 学習データ | **全モデル DIFF（Critical）** | 6 モデルとも、公式のデータ生成と別の素材・分割を使う |
| データ前処理（正規化・crop・padding） | DIFF が多い | RMS 正規化: audio-db は既定で有効。DFN3 と TF-Locoformer の公式には無い。SEMamba++ と MossFormerGAN は clean / noisy を別々に正規化する |
| 学習ループ（epoch 定義・DDP・蓄積） | PARTIAL〜DIFF | TF-GridNet と TF-Locoformer の「1 epoch = rank あたり N 元発話」、2〜4 rank の DDP は 1 プロセス専用の audio-db では再現不可 |
| validation と best 選択 | DIFF が多い | MossFormerGAN（検証の定義）、MP-SENet（validation = test）、DFN3（3 秒オンライン窓）、TF-GridNet / TF-Locoformer（`si_snr` の扱い、5-best 平均） |
| 評価指標 | PARTIAL〜DIFF | §6.2 |
| 再開 | audio-db が上位互換 | 公式の再開は lr の余分な減衰などの癖がある。再現しない方針なら問題ないが、公式の挙動とは別（MP-SENet、SEMamba++、MossFormerGAN） |

## 8. Critical / Major の未対応事項

### 8.1 Critical（12 件）

| ID | モデル | 内容 | 実装状態 |
|---|---|---|---|
| DFN-D01 | DFN3 | 学習データ分布が別物（DNS4 + PTDB/VCTK×10 と VoiceBank 固定ペア）。素材の一部は手元に無い | 未実装 |
| DFN-D02 | DFN3 | オンライン混合（SNR 7 値、gain、2〜5 本の雑音、合成雑音、clip ガード）が無い | 未実装 |
| MFG-D02 | MossFormerGAN | 検証の定義が別物（公式: 2 秒 crop・G の全損失・D は train モード / audio-db: 全長・時間 L1 のみ）。plateau の判定値が変わる | 未実装 |
| MPS-D01 | MP-SENet | 学習セットが公式の 11,572 ではなく 10,700 | 未実装 |
| MPS-D02 | MP-SENet | validation と best 選択に公式は test.txt（824）。audio-db は p286/p287（872）で test を拒否する。**研究上の判断が要る** | 未実装 |
| SMB-D01 | SEMamba++ | オンライン混合を、固定 1 ケースのレシピで表せない | 未実装 |
| SMB-D02 | SEMamba++ | 学習素材（VCTK 0.92 + 雑音 + RIR）が無く、VoiceBank-DEMAND で学習している | 未実装 |
| SMB-D03 | SEMamba++ | bandwidth・clipping・確率的 reverb をレシピ v1 が表せない | 未実装 |
| TFG-D01 | TF-GridNet | 学習データが MS-SNSD ではなく VoiceBank | 未実装 |
| TFG-D02 | TF-GridNet | 1 epoch の規模（公式は約 15,000 更新 / rank、VoiceBank は約 575）。**推定値** | 公式一致は未検証 |
| TFL-D01 | TF-Locoformer | 学習・検証データが DNS ではなく VoiceBank。DNS のセットが DB に無い | 未実装 |
| TFL-D02 | TF-Locoformer | 公式の事前生成（連結・clean 切り出し・RMS −25 dBFS・最終レベル乱択・16 bit）の生成手段が無い | 未実装 |

SMB-D04（SNR の定義と分布）は文書で「Major（Critical になりうる）」とされている。

### 8.2 Major（主なもの）

| ID | モデル | 内容 |
|---|---|---|
| DFN-D03 / D04 / D05 / D06 / D11 / D16 | DFN3 | 残響、RMS 正規化（audio-db 既定で有効、公式に無い）、3 秒窓の作り方（連結 vs 零埋め）、検証の方式、評価、valid/test への augmentation |
| MFG-D03 / D05 / D08 / D12 / D01 | MossFormerGAN | best 成果物なし、独立 `audio_norm`（clean/noisy のゲイン比が最大 +3.14 dB ずれる）、DDP、評価条件、train/validation の分割 |
| MPS-D03 / D04 / D06 | MP-SENet | 学習量（論文間・論文とコードで不一致）、評価手順、`batch_first` 由来のバッチ依存（公式のバグを複製。docs に記述なし） |
| SMB-D04 / D05 / D06 / D08 / D09 / D10 / D11 / D14 | SEMamba++ | SNR の定義、個別ピーク正規化、劣化の適用順、サンプラ、バッチ構成、検証、Mamba の `dt_proj.bias` 初期化、評価指標 |
| TFG-D03 / D04 / D05 / D06 / D07 / D09 / D10 / D12 / D16 | TF-GridNet | MS-SNSD の SNR 半減、未シード生成と名前順分割、xavier 初期化、2 rank DDP、`si_snr` best、−25 dBFS 正規化、リサンプル、評価指標、既存 docs の誤り |
| TFL-D03 / D04 / D05 / D06 / D07 / D08 / D15 / D16 | TF-Locoformer | RMS 正規化（公式は null）、epoch 定義、4 rank DDP、5-best 平均、DNS 合成テスト、`cv_synthetic_small` 不明、docs の誤り、損失の ESPnet クラスとの数値比較なし |

## 9. 研究上の判断が必要な事項

調査だけでは決められず、決めずに候補と差異を報告する。実装フェーズに進む前に、あなたの判断が要る。

1. **MossFormerGAN の再現対象。** (a) 公開コード `T/` の VoiceBank 固定ペア（再現可能だが PESQ 3.47 は期待できない）、(b) 配布 checkpoint の学習（8000 h、clean:残響 = 7:3、SNR 0〜15 dB。生成方式と学習コードが UNKNOWN で再現不能）。(a) を選ぶなら、配布 checkpoint の数値と比べない。
2. **TF-GridNet の再現対象。** 論文の実験（WSJ0-2mix、WHAMR! など）は単一話者 SE ではない。単一話者 SE の候補は `ms_snsd`（コミュニティ追加、論文の再現ではない）、`dns_ins20`、`urgent24`（v3）、`wsj0_2mix`（論文に最も近いが分離タスク）。選定は研究目的次第である。
3. **MP-SENet の validation の扱い。** 公式は test.txt（824）で validation と best 選択を行う（test 集合上の選択になる）。(a) 公式どおり（audio-db の禁止規則を例外にする）、(b) 話者分離した validation を維持し公式との差異として記録、のどちらにするか。
4. **MP-SENet の学習量。** Interspeech 論文は 100 epoch、journal v1 は 120 epoch、v2 は 500k step、公式コードの既定は 400 epoch。どれを再現するか。
5. **SEMamba++ の重みの扱い。** 論文の重みはバグ版（第 3 解像度の Channel FFN が未使用）で、公開重みは修正後の再学習（データ不明）。論文値の再現ではなく、公開コードのプロトコルの再現として扱うか。
6. **DFN3 の素材。** DNS4 の clean と PTDB が手元に無い。代替素材で「公式形式のプロトコル」を別名で用意するか、公式データの入手を待つか。
7. **オンライン混合の乱数の同等性。** DFN3 の公式は Rust の `Xoshiro256++`、SEMamba++ は無シードの `default_rng`。Python で書くと乱数列は一致しない。「分布が同じ」で足りるか、`libdfdata` を呼ぶかは、独立実装の方針（公式への実行時依存の禁止）と合わせて決める。
8. **TF-Locoformer の学習データ生成。** 3,000 h の事前生成は約 1 TB（clean・noisy・noise の 3 本、16 kHz 16 bit で各約 346 GB という**推計**）で、生成が multiprocessing で非決定的なため、ビット一致は不可能。統計的同等で足りるか。生成と学習は TSUBAME で 30 分以上のジョブになり得るため、投入前に明示的な指示か承認が要る（`AGENTS.md`）。

## 10. 未確定事項（UNKNOWN）の集約

詳細と、なぜ確定できなかったかは各文書の「UNKNOWN 一覧」にある。件数は DFN3 18、MossFormerGAN 12、MP-SENet 12、SEMamba++ 16、TF-GridNet 15、TF-Locoformer 16。

| モデル | 主な UNKNOWN |
|---|---|
| DFN3 | 学習時の `dataset.cfg` と件数、出荷モデルを学習した commit、DNS4 のどの部分か、RIR の集合、Table 1 の checkpoint と PESQ のモード、3 秒未満の発話の割合 |
| MossFormerGAN | 配布 checkpoint の実データ生成と学習コード、Hamming 窓の `periodic`（公式 repo 内で学習と推論が食い違う）、事前 16 kHz 版のリサンプラー、論文表 2 の SNR 見出しとローカル VB の `log_testset.txt` の違い |
| MP-SENet | 配布 checkpoint の学習量と GPU 数、公式 16 kHz 版の変換器、DNS の生成設定と test 件数、報告値の checkpoint 選択、journal 出版版の本文 |
| SEMamba++ | 学習発話数 N、VCTK-GSR の分割規則、雑音・RIR の分割、検証 100 ペアの出自、論文が評価に使った checkpoint の step、論文の重みの学習クロップ（24,000 か 48,000）、公開重みの学習データ |
| TF-GridNet | HF checkpoint の学習データの実件数・epoch 数・学習時の ESPnet 版、MS-SNSD と VoiceBank-DEMAND の重複、`urgent24` のデータ生成条件、論文の epoch 数・early stopping・seed |
| TF-Locoformer | `cv_synthetic_small` の構成、公式が生成した学習データの実物、論文の 21,000 step / epoch の根拠、論文の PESQ-WB 3.72 と Noisy の SI-SNR 9.1 の算出方法、実際の学習コマンドと停止 epoch |

追加調査の案:

- DFN3: `dataset.cfg` と件数の問い合わせ、出荷モデルの学習 commit の推定（リリース日付近の commit 比較）。
- MossFormerGAN: 作者への問い合わせ。ツールキット論文の v2 以降の確認。
- MP-SENet: journal 出版版（Neural Networks）の本文の入手。
- TF-GridNet / TF-Locoformer: ESPnet issue・PR の学習ログの探索、`cv_synthetic_small` の一覧の問い合わせ。

## 11. 既存ドキュメント・作業ツリーの不整合

各文書の §11 から横断的なものを抜粋した。既存ファイルは変更していない（今回の作業範囲外）。

### 11.1 作業ツリーに無いものを参照している記述

| 参照されているもの | 状況 |
|---|---|
| `configs/training/*_seed17.yaml`、`configs/training/trial100/` | 作業ツリーに無い。`b09b7f3`（設定ファイル等の整理）で削除済み。現存するのは seed 42 の 6 本だけ。`.agents/DECISIONS.md`、`OFFICIAL_TRAINING_AUDIT.md` ほかが参照している |
| `configs/evaluation/` | 無い。`README.md` などが参照している |
| `data/db/*.duckdb`（`resource.duckdb`、`voicebank.duckdb`、`training.duckdb`） | `data/db/` が空。AGENTS.md が前提とする DuckDB が無い |
| `data/staging/*`、`data/checkpoints/tf-locoformer/` | 空、または無い |
| `.venvs/lab` | 無い。現在は役割別の `.venvs/se` など |
| `tests/training/test_tf_locoformer_protocol.py` | 存在しない設定を読むため失敗する可能性（**未実行の推測**） |

### 11.2 公式仕様として誤っている、または欠けている記述

| 記述 | 問題 |
|---|---|
| `docs/se/MODELS.md` の TF-Locoformer の arXiv ID | `2409.16674` は誤り（別分野の論文）。正しくは `2408.03440` |
| `.agents/DECISIONS.md` #26、`OFFICIAL_TRAINING_AUDIT.md`（TF-GridNet の best） | `valid si_snr max` が有効と読める。ピン留め ESPnet では stats のキーが `si_snr_loss` のため働かない。#31（TF-Locoformer）は正しく、#26 と矛盾する |
| `OFFICIAL_TRAINING_AUDIT.md`（TF-GridNet の「単一プロセス想定」） | 公式は 2 rank の DDP |
| `OFFICIAL_TRAINING_AUDIT.md`（MP-SENet の epoch / validation 長 / best / resume の GO） | 母数が 11,572 と 10,700 で違う、公式は clean を切らない、best の集合が違う、公式の再開は lr を余分に減衰して `best_pesq` を 0 に戻す。いずれも GO の根拠に書かれていない |
| `OFFICIAL_TRAINING_AUDIT.md`（RMS 正規化） | DFN3 と TF-Locoformer の公式に発話 RMS 正規化は無いが、差異として挙げていない |
| `OFFICIAL_TRAINING_AUDIT.md`（DFN3 の更新数） | 22,711（audio-db）と 22,831（公式、端数 batch を残す）の差が書かれていない |
| `MP_SENET.md` | `libri_demand_smoke_*` と `.venvs/lab` を参照している。現在の YAML は `voicebank_28_*` |
| `docs/se/VALIDATION.md`（MP-SENet） | `d91b88c2…` という別の commit を書いている。ピンは `89932cf` |
| `docs/training/TF_LOCOFORMER.md` の設定の呼び分け | スモークと研究用が逆 |
| `requirements/official-assets.toml` と `src/se/models/semambapp/NOTICE` | SEMamba++ の URL が旧名と新名で混在。HF の `urgent24` カードは `cc-by-4.0` だが「記載なし」とされている |

### 11.3 論文と公式コードの矛盾（要点）

| モデル | 論文 | 公式コード・設定 |
|---|---|---|
| MP-SENet | Interspeech 版: 100 epoch、lr は 30 epoch ごとに半減、test 872 発話 | `ExponentialLR` 0.99、既定 400 epoch（Interspeech 版の旧コードは既定 200 epoch）、test リストは 824 行。README は `batch_first` のバグを「性能に影響しない」と書くが、実測では出力がバッチ構成で変わる |
| SEMamba++ | クロップ 24,000、SNR −10〜20 dB、magnitude 損失 L1 | クロップ 48,000、HEAD の SNR −5〜20 dB、magnitude 損失 MSE |
| TF-Locoformer | 1 epoch = 21,000 step、3 epoch 改善なしで lr 半減、Macaron の 1/2 係数 | 計算上は約 14,000 更新、4 回連続の悪化で半減、1/2 係数は無い |
| DFN3 | DFN2 論文の学習値、損失は norm ratio | 出荷 `config.ini` と値が違う、損失は MSE。`combine_noises` のコメントは「平均」だが実際は加算 |
| TF-GridNet | 4 秒 1 区間 / 混合、clip 1.0、3 epoch で半減 | `ms_snsd`: 3 秒チャンク、clip 5.0（既定値）、ReduceLROnPlateau |

### 11.4 公式リポジトリ自体の不具合（SEMamba++）

出荷状態では学習を起動できない: 既定の設定ファイル名が存在しない、`dataset.py` が存在しない `utils.util` を import する、`data/*.json` が空、`infer_folder.py` が `infer.py` に無い関数を import する。学習の再現には、これらを補う公式外の作業が要る。

## 12. DB 中心の新データ基盤に必要な情報

DB スキーマは確定しない。各モデルの必要情報は各文書の §10 にある。ここでは横断的な結論を示す。

### 12.1 現行のレシピ JSON v1 の判定

レシピ JSON v1（`docs/db/RECIPE.md`）は「固定した 1 ケースの生成条件」を記録する設計である。**公式のオンライン混合を表現するには不足する。** 固定した 1 ケースのレシピと、オンライン生成の確率分布を定義するプロトコルは、別の層として扱う必要がある。

v1 が表せるもの:

- 固定ペア（空の `operations`、`output.reference: dry_clean`）。MP-SENet、MossFormerGAN の公開コードはこれで足りる。
- 事前生成済みファイルの登録（clean と noisy の recorded recipe）。TF-GridNet と TF-Locoformer の学習セットは、生成物を `data/staging/` に保存したあとなら登録できる。
- 1 回の抽選結果のうち、RIR（1 本）、加法雑音（`rms_v1`）、ピーク制限の部分（DFN3、SEMamba++）。

v1 が表せないもの（複数モデルで共通のものを先に挙げる）:

| 不足 | 該当モデル |
|---|---|
| 確率分布（SNR の離散分布・範囲、gain、雑音の選択確率、劣化の適用確率、RIR の適用確率） | DFN3、SEMamba++（と、生成器を作る場合の TF-GridNet / TF-Locoformer） |
| 複数雑音の加算・連結、雑音のみのサンプル、合成雑音 | DFN3、TF-GridNet、TF-Locoformer |
| 複数発話の連結（3 秒窓の埋め、30 秒混合の組み立て）と、clean のランダム切り出し | DFN3、TF-GridNet、TF-Locoformer |
| レベル演算（RMS・dBFS・ピーク）と、clean / noisy を別々に行う正規化 | MossFormerGAN、SEMamba++、TF-GridNet、TF-Locoformer |
| 有音フレーム基準の SNR、名目 SNR と実 SNR の区別（MS-SNSD は半分） | SEMamba++、TF-GridNet |
| bandwidth 制限、clipping、clip ガード | DFN3、SEMamba++ |
| RIR の詳細（減衰・DRR・チャンネル平均）、with_reverb の参照を `reverberant_clean` と宣言すること（recorded recipe の参照名が `dry_clean` 固定） | DFN3、SEMamba++、TF-Locoformer |
| リサンプラの方式と版（soxr_hq、`sinc_fast`、`torchaudio` 既定）。数値がライブラリ版で変わる | 全モデル |
| 16 bit 量子化 | TF-Locoformer、TF-GridNet |
| `output.max_seconds` 以外の学習時 crop / padding 規則 | 全モデル |
| 乱数・epoch・サンプラー位置・DDP の状態 | 全モデル |
| 同じケースを train / valid / test に重複して置く規則（公式は test を validation に兼用） | MP-SENet、MossFormerGAN |
| ファイルリストの出典と SHA-256、素材集合のハッシュ | 全モデル |

### 12.2 DB に必要な情報（横断）

1. **音声素材と役割。** clean / noise / RIR / noisy 固定ペア / 混合済み事前生成物を役割付きで索引する。素材の元サンプリングレート、チャンネル数、長さ、SHA-256。現行 `corpora.audio_files` の列で足りるが、役割（学習素材として使う用途）と、コーパス内のサブセット（VCTK 0.92 の p280・p315 除外など）の表現は別途要る。
2. **分割と対応関係。** 話者単位、発話単位、ファイル名順（uttid 辞書順 9/10）など、公式ごとに分割規則が違う。公式の分割を「ファイルリスト + ハッシュ + 規則名」として保存し、audio-db 独自分割（10,700 / 872）と区別できること。
3. **固定する生成条件。** 事前生成のモデル（TF-GridNet、TF-Locoformer）は、生成パラメータ（`total_hours`、SNR 範囲、−25 dBFS、最終レベルの範囲、30 秒連結）、生成に使ったスクリプトの commit、生成ログ（乱数が非決定的なので、**実際に生成された混合の一覧**）を保存する。
4. **学習時にサンプリングする条件。** DFN3 と SEMamba++ は、素材集合の参照、各分布のパラメータ、適用順、seed の式を持つ「プロトコル」を、ケースとは別に保存する。
5. **保存すべき再現性の状態。** RNG 状態、epoch（ESPnet は 1 始まり）、サンプラー位置、チャンク計画のハッシュ、長さ一覧のハッシュ、DDP の rank 構成、epoch ごとの valid loss 履歴、ライブラリ版。
6. **評価用 DB と学習用 DB の分離。** 候補 `data/db/reproduction.duckdb`。評価データ（VB test など）を validation に使う公式（MP-SENet、MossFormerGAN）は、現行の禁止規則と衝突する。例外の表現が必要。
7. **staging の規則。** 生成物は DuckDB に書く前に `data/staging/` に保存し、登録後も消さない（`AGENTS.md`）。TF-Locoformer の約 1 TB（推計）の生成物もこの規則に従う。

### 12.3 共通化できる処理 / 専用実装が必要な処理

| 区分 | 内容 |
|---|---|
| 共通化できる | 素材索引（`corpora.audio_files`）、ファイルリストとハッシュの管理、seed を作る関数（SHA-256 ベース）、16 kHz モノの読み込みと長さ検証、リサンプル（方式を引数にする）、末尾 0 埋めと crop（乱数の引き方を引数にする）、チャンクのスライス、SNR 混合の核（`rms_v1`）、RIR 畳み込み、`mr_l1_tfd` などの損失、AdamW・勾配クリップ・xavier 初期化、評価指標の純関数、SHA-256 検証と staging 登録、RNG の保存・復元 |
| 専用実装が必要 | **DFN3**: 3 秒窓の連結、複数雑音の加算と合成雑音、`RandReverbSim`、speech の augmentation、SNR −100 dB の分岐、3 信号まとめてのクリップガード、局所 SNR の目標。**SEMamba++**: 有音フレーム基準の SNR、雑音の wrap 配置、劣化の固定順（reverb → noise → clipping → bandwidth）、clean と劣化音の個別ピーク正規化、劣化後の切り出し、Dataset 内の圧縮 0.3 の STFT。**MossFormerGAN**: 独立 `audio_norm`、検証の定義（crop・G 全損失・D は train モード）、plateau の best 上書きと epoch / step の巻き戻し、rank 固定の `DistributedSampler`、PESQ 失敗時にバッチを飛ばす処理。**MP-SENet**: 単 GPU の固定順、`batch_first` 依存のバッチ構成、validation = test の選択、PESQ 失敗を −1 のまま平均する集約、PCM16 書き出しと hop 倍数への短縮。**TF-GridNet**: MS-SNSD 生成器（連結・無音・`snr_mixer`・−25 dBFS・シード化）、ESPnet の DDP チャンク iterator（rank 分割、rank ごとの cache）、xavier 初期化（独自 LayerNorm を含む）。**TF-Locoformer**: DNS 合成スクリプト全工程、4 rank を 1 プロセスで模擬するチャンク計画、`ave_5best`、DNS 評価の出力ピーク 0.9 と bss_eval、with_reverb の参照定義 |

## 13. 実装課題の整理

修正対象のファイルは各モデル別文書の §9 にある。ここでは優先度別に主なものを示す。詳細な新 API や DB スキーマはこの段階で確定しない。情報不足のまま設計を決めない。

### P0: 公式プロトコル再現に不可欠

| 課題 | 対応する差異 ID | 主な対象ファイル・モジュール |
|---|---|---|
| 公式のデータ素材と分割を登録できる学習用 DB（候補 `data/db/reproduction.duckdb`）。素材索引、ファイルリスト（ハッシュ付き）、分割規則、プロトコル定義、実行状態 | 全モデル（DFN-D01、MFG-D01、MPS-D01 / D02、SMB-D02、TFG-D01、TFL-D01） | 新規（`src/data/` 配下）、`src/training/data/dataset.py`、`docs/db/` |
| 「ケース」と「プロトコル」の分離。確率分布とサンプリング規則を持つプロトコル層 | DFN-D02、SMB-D01 / D03 / D04 | 新規、`src/data/recipe.py`、`src/evaluation/audio/render.py`、`docs/db/RECIPE.md` |
| モデル別の公式正規化への切り替え（audio-db 既定の発話 RMS 正規化を前提にしない） | DFN-D04、MFG-D05、SMB-D05、TFG-D09、TFL-D03 | `src/training/data/dataset.py`（`_normalize`）、各 YAML |
| 事前生成データの生成器（公式スクリプトの呼び出しか再実装かは §9 の判断後） | TFG-D03 / D04、TFL-D02 | 新規の生成器、`data/staging/` への出力方針 |
| DFN3・SEMamba++ のオンライン混合ローダー（モデル固有の専用実装） | DFN-D02 / D03 / D05、SMB-D01 / D03 / D04 / D06 | 新規ローダー、`src/training/data/sampler.py` |
| 検証と best 選択のプロトコル（MossFormerGAN の crop・G 全損失・D は train モード、MP-SENet の validation = test、DFN3 の 3 秒窓） | MFG-D02 / D03、MPS-D02、DFN-D06 / D16 | `src/training/se/mossformergan.py`、`src/training/se/mp_senet.py`、`src/training/core/trainer.py` |
| epoch・更新数・学習量の公式定義（TF-GridNet と TF-Locoformer の rank あたり元発話数、DFN3 の 3 秒窓の埋め方、MP-SENet の学習量の決定） | TFG-D02、TFL-D04、DFN-D05、MPS-D03 | `src/training/data/chunk_plan.py`、`src/training/core/trainer.py`、各 YAML |
| 論文値と比べるための評価の公式化（PESQ のモード、16 kHz 評価セット、欠けている指標） | MPS-D04、MFG-D12 | `src/evaluation/metrics/`、評価セット定義 |
| SEMamba++ の Mamba `dt_proj.bias` 初期化 | SMB-D11 | `src/se/models/semambapp/network.py` |

### P1: 再現性・検証可能性に重要

| 課題 | 対応する差異 ID | 主な対象 |
|---|---|---|
| ESPnet の DDP チャンク iterator の rank 分割（TF-GridNet 2 rank、TF-Locoformer 4 rank）。audio-db は `world_size != 1` を拒否する | TFG-D06、TFL-D05 / D11 | `src/training/core/trainer.py`、`src/training/data/chunk_plan.py` |
| `ave_5best`（TF-Locoformer）、TF-GridNet の `si_snr` best の取り扱い（公式で働かない基準を実装しない） | TFL-D06、TFG-D07 | `src/training/core/trainer.py`、各 YAML |
| TF-GridNet の xavier 初期化（独自 LayerNorm 含む） | TFG-D05 | `src/training/se/tf_gridnet.py`、`src/training/se/initialization.py` |
| DNS 合成テスト、16 kHz 版 VB test の評価セット。recorded recipe の参照名の拡張 | TFL-D07、MFG-D12 | `src/data/recipe.py`、`src/evaluation/audio/render.py` |
| リサンプラの方式の記録と公式との一致確認 | MPS-D05、MFG-D06、TFG-D10、SMB-D16 | `src/training/data/dataset.py` |
| 乱数・サンプラー位置・チャンク計画の保存と、公式の DDP 分割との整合 | MFG-D08、SMB-D08 / D17 | `src/training/data/sampler.py`、`src/training/core/rng.py` |
| `batch_first` 由来のバッチ依存の文書化とテスト | MPS-D06 | `src/se/models/mp_senet/transformer.py`、`docs/training/MP_SENET.md` |
| 公式との数値一致の検証（ESPnet のクラス自体との比較を含む） | TFL-D16、DFN-D17、SMB-D09 | `tests/training/` |
| 既存 docs の修正（§11）、作業ツリーに無いファイルへの参照の整理 | MPS-D17 / D18、TFG-D16、TFL-D15、SMB-D19、MFG-D15 | `docs/`、`.agents/DECISIONS.md` |
| 公式 URL・ライセンスの整理（SEMamba++ の旧名・新名、HF カード） | SMB-D15 | `requirements/official-assets.toml`、`src/se/models/semambapp/NOTICE` |

### P2: 性能改善・効率化・運用

| 課題 | 対応する差異 ID |
|---|---|
| 非有限勾配のスキップ、PESQ 失敗時の扱い | TFG-D08、TFL-D10、MFG-D10、DFN-D12 |
| seed・決定性・初期化の乱数列の差 | MPS-D11、TFG-D13、TFL-D12、MFG-D13、DFN-D14 |
| 再開の意味（公式の再開は再現しない方針なら文書化のみ） | MPS-D08、SMB-D12、MFG-D04 |
| Interspeech 版 MP-SENet、DNS 版、`urgent24` 変種、論文忠実プロトコル | MPS-D13 / D14、TFG-D14 / D15 |
| TF32、環境のバージョン差、`rotary-embedding-torch` の版 | DFN-D13、SMB-D18、TFL-D13 |

## 14. 実装フェーズへの推奨作業順序

「公式がオンライン混合ならオンライン混合を再現し、固定ペア・事前生成ならそれを維持する」方針と、データ・時間のコストから、次の順序を推奨する。**§9 の研究上の判断（特に 1〜4、8）を先に決める。**

1. **判断と文書修正（コードなし）。** §9 の項目を決める。§11.2 の誤り（arXiv ID、`si_snr`）を訂正する。
2. **学習用 DB の設計確定（スキーマ決定）。** §12 の情報を基に、ケースとプロトコルの分離、公式分割の登録、状態の保存を設計する。この監査でスキーマは確定していない。
3. **固定ペアのモデルから再現する。** MP-SENet（公式 11,572 / 824、noisy RMS 正規化、2 秒 crop）と MossFormerGAN の公開コード（独立 `audio_norm`、検証の定義）。生成器が要らず、既存の素材（VoiceBank-DEMAND）で閉じる。ここで DB、公式分割、validation の扱い、評価の公式化を検証できる。
4. **評価の公式化。** 手順 3 と並行して、PESQ のモード、16 kHz 評価、欠けている指標を整える。論文値と比較できる状態にする。
5. **事前生成のモデル。** TF-GridNet（MS-SNSD。素材は手元にある）、TF-Locoformer（DNS。約 1 TB の推計）。生成器の実装、`data/staging/` への保存、登録、DDP チャンク iterator。生成と学習は TSUBAME で 30 分以上のジョブになり得るため、投入前に明示的な指示か承認が要る。
6. **オンライン混合のモデル。** DFN3 と SEMamba++。プロトコル層、モデル固有ローダー、乱数同等性の方針（§9 の 7）。DFN3 は素材（DNS4、PTDB）の入手が前提。SEMamba++ は VCTK 0.92 と Arni の入手と、公式リポジトリの不具合（§11.4）への対応が要る。
7. **検証と DDP。** 公式との数値一致、DDP の再現（TF-GridNet、TF-Locoformer）、再現性（状態の保存と再開）の検証。

## 15. この監査の限界

- 公式コードの学習・評価、audio-db のテスト・学習・評価は実行していない。GPU は使っていない。DB には触れていない（`data/db/` が空）。
- 既存記録の数値一致（損失・勾配・重み）は再実行していない。MATCH は「コードを読んだ範囲での一致」であり、数値が一致することの証拠ではない。
- 各モデル別文書の記述は、調査を担当した 6 本のサブエージェントの報告である。統合時に私が独立に確認したのは次の点に限る。
  - 論文 ID（`2408.03440` / `2409.16674` / `2603.11669` / `2305.08227`）の arXiv 上のタイトル。
  - MP-SENet の `dataset.py`（RMS 正規化、crop、shuffle）、`train.py`（AdamW、ExponentialLR、`drop_last`、`best_checkpoint_start_epoch`）。
  - ESPnet の `time_domain.py` の `si_snr_loss` というキー名。
  - 各文書の見出し構成、§0 の要約、§9 の差異表。
  - 公式 checkout（`src/se/official/*`）に変更が無いこと。
- 公式側の行番号の一部は、文書の執筆後に再確認していない（DFN3 文書に明記）。
- 論文は主に arXiv の版を読んだ。出版版（Interspeech、Neural Networks）、PDF との照合、後続の版の有無は未確認のものがある。
- 生成済みの公式学習データ（DNS 3,000 h、`cv_synthetic_small`、MS-SNSD の生成物、DFN3 の `dataset.cfg`）は入手できておらず、件数・時間数の一部は計算値または UNKNOWN である。
- 公式 checkpoint は、MossFormerGAN の G を除き（手元の sha256 と一致を確認）、実体を読み込んでいない（TF-Locoformer は LFS ポインタのみ）。
