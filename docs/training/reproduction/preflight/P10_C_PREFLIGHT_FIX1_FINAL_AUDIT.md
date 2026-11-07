# P10-C Preflight Fix.1 最終監査

日付: 2026-11-08。HEAD は `c0dc3c6672e9275133a7b5a6dce878eee4068c62`。作業木は dirty で、commit はしていない。本格学習は開始していない。

実装は [P10_C_PREFLIGHT_FIX1_IMPLEMENTATION.md](P10_C_PREFLIGHT_FIX1_IMPLEMENTATION.md)。独立検証は [P10_C_PREFLIGHT_FIX1_VERIFICATION.md](P10_C_PREFLIGHT_FIX1_VERIFICATION.md)。検証報告を、そのまま合格にはしていない。

## 総合

NO-GO。4 モデルの本格学習を同時に始められる状態ではない。

始められるのは条件付きで、MP-SENet 公式、MossFormerGAN 公式 1 GPU、TF-GridNet である。MossFormerGAN 公式 2 GPU と、TF-Locoformer の研究用 1,024 件は NO-GO のままである。Preflight の PASS は起動条件の確認であり、公式再現の成功ではない。

## 修正した Issue

| ID | 状態 | 内容 |
|---|---|---|
| P10P-A-007 / P10P-D-004 | MITIGATED | 失敗 micro-batch は backward せず、後続の成功で蓄積窓を埋める。カウンタは分けた。データ epoch の末では、Optimizer 更新がずれても検証し、plateau に損失を渡す。2 GPU は片 rank の失敗で全 rank が backward しない |
| P10P-A-008 / P10P-D-002 | MITIGATED | 824 件は world 2 で 412 と 412。公式 DistributedSampler の epoch 0 と一致した。CUDA だけ SyncBatchNorm。`broadcast_buffers=True`、`find_unused_parameters=True` |
| P10P-D-001 | MITIGATED | 増分を rank 間平均してから clip する。実音声 2 micro-batch の clip 前 norm は約 45 と 56 で、閾値 10 を超えた |
| P10P-D-009 | MITIGATED | その 1 更新では未使用パラメータは 0。他の長さは NOT_TESTED |
| P10P-F-007 | BLOCKED | claim `research_checkpoint_selection` はスキーマに追加した。正本 DB が無く、1,024 件の登録と snapshot は未発行 |
| R1 | 解消 | epoch 末の plateau は、PESQ 失敗のあとでも検証値を受け取る |
| R2 | 解消 | Discriminator の `broadcast_buffers` は True。CUDA の 3 forward のあと backward は戻った |
| R3 | 解消 | CPU の複数 rank は BatchNorm のまま |
| R4 | 解消 | step 後の `zero_grad` は残し、空勾配を失敗にしない |
| R6 | 解消 | 検証件数が world size 未満のときは、参加する全 rank が同じ短い列を見る。824 件の分割は変えていない |

差し戻しは R1 から R4、その後 R6。再検証のあと、さらなる差し戻しは無い。

## 残った Issue

| ID | 状態 | 理由 |
|---|---|---|
| P10P-D-005 | OPEN | 更新後に同じプロセスで検証へ進んだ CUDA illegal memory access は、実 batch の再実行 1 回では再現しなかった。直したことにはしない。SyncBatchNorm の内側で片 rank だけ例外になると、collective はまだ deadlock し得る |
| P10P-F-007 / P10P-C-001 | BLOCKED | `data/db/reproduction.duckdb`、backups、staging の同名ファイルは無い。空の DB は作っていない |
| P10P-A-011 | OPEN | 検証 crop の乱数源は変えていない |
| P10P-B-003 | OPEN | 1,024 件は `cv_synthetic_small` ではない |
| 公式 2 GPU の数値一致 | NOT_TESTED | フラグと分割を揃えたあとも、公式の 2 GPU 学習と損失・勾配を比べていない |

2 GPU の全 rank スキップと、増分だけを平均する clip は残す。公式の片 rank スキップと、蓄積済み勾配の再平均は採らない。

## モデル別開始判定

| モデル | 判定 | 根拠 |
|---|---|---|
| MP-SENet 公式 | CONDITIONAL GO | Preflight は PASS。snapshot は train 11,572、valid 824。このセッションで公式 magnitude は再測定していない |
| MP-SENet leakage-safe | NO-GO | snapshot が無く Preflight は FAIL |
| MossFormerGAN 公式 1 GPU | CONDITIONAL GO | 初期化の G1 と D4 は維持。epoch 末 plateau と PESQ の蓄積窓は確認した。`global_step` は公式 `self.step` と別。crop の乱数源は OPEN |
| MossFormerGAN 公式 2 GPU | NO-GO | Preflight は PASS。DDP フラグと 824 件の rank 分割は公式側に寄せた。公式 2 GPU 学習との数値一致は NOT_TESTED。全 rank スキップと増分平均は残る |
| MossFormerGAN leakage-safe | NO-GO | snapshot が無く Preflight は FAIL |
| TF-GridNet | CONDITIONAL GO | Preflight は PASS。件数は 7,345 / 1,835 / 305。固定 `mix_seed` は既知の差 |
| TF-Locoformer 研究用 1,024 | NO-GO | 定義、1,024 件、SHA-256 `73e8d44a3c2196af874f8cddb7a1cc9a6b1745a220d6f1b88fda7a9a502f7ad4` は親 valid から再計算して一致した。登録と snapshot は BLOCKED。Preflight は FAIL。`leakage_safe` でも `cv_synthetic_small` でもない |
| TF-Locoformer 36,000 を毎 epoch | NO-GO | 親 snapshot の valid 36,000 件は残っている。選択方式としては始めない |

## 公式との差異

一致させたもの:

- 検証 824 件の epoch 0 の rank 分割。末尾破棄 0、前方の繰り返し 0、rank あたり 412
- Discriminator の `broadcast_buffers=True` と `find_unused_parameters=True`
- SyncBatchNorm は CUDA かつ world size 2 以上だけ
- 1 GPU の PESQ 失敗は step を進め、成功だけで蓄積窓を埋める
- データ epoch の末で検証し、plateau を更新する
- 失敗 batch は分母に残り、損失 0 としては記録しない
- 初期化は G1 と D4。checksum `7729.488281` は正解にしていない

残した差:

- `global_step` は Optimizer 更新回数。公式の `self.step` は `official_step`
- 2 GPU の PESQ 失敗は、その micro-batch を全 rank で backward しない
- clip は増分を平均したあと。公式の蓄積済み再平均はコピーしない
- 検証件数が world size 未満のときは、公式 sampler を呼ばず同じ短い列を見る。1 件のスモークは 824 件分割ではない
- 研究用 1,024 件は未登録

## P10-C への残存条件

- 公式 2 GPU 学習との損失・勾配の比較が無いあいだ、MossFormerGAN の 2 GPU 本格 YAML は始めない。
- `reproduction.duckdb` がこの作業木で開くまで、研究用 1,024 件の本格 YAML は始めない。親の 36,000 件は毎 epoch 見ない。
- leakage-safe の snapshot が無い MP-SENet と MossFormerGAN は始めない。
- 修正後の `tests/reproduction` と `tests/training` の全件は再実行していない。開始前の回帰として、その全件を別途走らせる。
- R5 の illegal memory access は 1 回の再実行で再現しなかった。再現した場合は、その行を特定するまで 2 GPU を GO にしない。
