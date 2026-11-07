# P10-C Preflight 監査

監査日: 2026-11-07。対象は `feature/reproduction`。修正と本格学習は開始していない。

## 監査対象

P1 から P10-B.2 までの学習再現基盤。本格開始の判断対象は MP-SENet、MossFormerGAN、TF-GridNet、TF-Locoformer である。DeepFilterNet3 と SEMamba++ は公式素材が blocked である既知制約として扱い、4 モデルの開始条件には入れない。

## 監査方法

1. 固定した公式 checkout の実行コード
2. 現行 `src/` の実行コード
3. 実在する学習 YAML
4. CPU 上の小さな数値比較と、読み取り専用の snapshot 計測
5. 相互レビュー

過去の P10 文書は仮説の索引であり、PASS の根拠にはしていない。GPU の本学習と NCCL は動かしていない。`src/`、学習 YAML、テスト期待値、DuckDB、snapshot、checkpoint は変更していない。

## サブエージェント

Phase 1 は 7 担当を並列に独立調査した。Phase 2 は次の交差で、相手の結論を前提にせずコードを読み直した。

| 交差 | 報告 |
|---|---|
| A と D | `REVIEW_A_D.md` |
| B と D | `REVIEW_B_D.md` |
| C と E | `REVIEW_C_E.md` |
| E と F | `REVIEW_E_F.md` |
| G と全報告 | `REVIEW_G_ALL.md` |

独立報告は `AGENT_A_OFFICIAL_MP_MOSS.md` から `AGENT_G_RED_TEAM.md` である。

## 参照した公式 commit

実測 HEAD。

| 対象 | commit |
|---|---|
| MP-SENet | `89932cfe90d1dacb8e170e4a331d762462c21792` |
| MossFormerGAN（ClearerVoice-Studio） | `6b3774dc79c46ae8bed2a4fa5f706f0ac8c75c61` |
| TF-Locoformer v1.1.0 | `7a615460d347ff7334a13dbb831d16280da72cdc` |
| ESPnet 作業木 | `0d136c5456d403cb2c0a4c13e76e8136e32e9b7a` |
| TF-Locoformer が学習参照とする ESPnet | `90eed8e53498e7af682bc6ff39d9067ae440d6a4` |

`90eed8e` の commit オブジェクトと `WarmupReduceLROnPlateau` の blob はローカルにある。DNS と MS-SNSD のレシピ本文の多くは blob が無く、作業木にも無い。`git grep` は欠けるオブジェクトを GitHub から取ろうとしてタイムアウトした。`cv_synthetic_small` の件数と選択規則は、この checkout からは読めない。

## 監査済み

- MP-SENet と MossFormerGAN の構築、初期化、forward、損失、Discriminator、AdamW、ExponentialLR、検証、checkpoint、前処理、乱数順
- TF-GridNet / TF-Locoformer の chunk 計画、`iterator_stop`、`mr_l1_tfd`、warmup、`valid.loss` の n-best、`valid.si_snr` の統計キー
- 存在する 6 snapshot の論理ハッシュ、分割件数、公式 VoiceBank の valid と test の同一性
- DDP の勾配同期、蓄積、BatchNorm、validation の rank 集約（CPU とコード。NCCL は未実行）
- checkpoint の保存順、状態ファイル、再開拒否
- `configs/training` のスモーク上限と本格 YAML の有無

## 未監査

- 実 GPU の NCCL、2 GPU の MossFormerGAN 勾配 norm、SyncBatchNorm の出力差
- VoiceBank 実波形での PESQ 例外率と、hop 切り詰め後の全 824 件の波形差
- 公式 `Stft`（`torch_complex` が無く import できない）との長波形比較
- `cv_synthetic_small` の実ファイル
- `reproduction.duckdb` と snapshot の同一性（DB ファイルが作業木に無い）
- DeepFilterNet3 オンラインの中身
- DNS validation 126,000 batch の公式クラス再実行
- 4 rank の validation 途中停止の実プロセス実験

## 相互レビュー

一致した主結論は、統合 Issue に残した。裂けた点は次で、合意したことにしていない。

| 論点 | 状態 |
|---|---|
| 公式が残す Discriminator の checksum `7729.488281` | DISPUTED。A-D は、その値が残存重みではないと読んだ。初期値が現行の 1 回構築と違う、という方向は維持 |
| validation 途中停止の参照関数 | DISPUTED。現象（interval 0 なら当該 epoch をやり直す）は一致。E の実測は単一プロセス、F の参照は 4 rank 経路で未実行 |
| `_close_chunk_run` の最後の else が通常終了で `COMPLETED` になるか | 到達性だけ DISPUTED。while の出口からは届かない、という F がコードと一致する |
| 2 GPU で片側だけ PESQ に失敗したときの待ち | DISPUTED。コード上は起こり得る。実測は無い |

降格した主張は `P10_C_PREFLIGHT_ISSUES.md` に書いた。学習開始を止める材料から外した主なものは、`reproduction.duckdb` の不在、5000 対 5001 の 1 更新差、21 batch の公式どおりの破棄、公式再開が scheduler を作り直すこと、Generator に BatchNorm があるという読み、である。

## 総合判定

**NO-GO。** 4 モデルの本格再現を、今の設定のまま開始しない。

開始を止めるのは、検証波形を hop で切る MP-SENet、学習開始時の Discriminator が公式と違う MossFormerGAN、PESQ 失敗で学習が例外終了する MossFormerGAN、2 GPU の Discriminator が SyncBatchNorm でないこと、TF の公式規模 YAML がスモーク上限付きでしか無いこと、TF-Locoformer の 150 epoch 全件検証が学習時間を上回る計画のまま研究上の決定が無いこと、である。

snapshot が残る 4 モデルは、DuckDB が無くても学習プロセス自体は起動できる。起動できることと、公式再現として開始してよいことは分けた。
