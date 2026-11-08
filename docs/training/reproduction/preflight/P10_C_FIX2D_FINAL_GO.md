# P10-C Fix.2-D 暫定 GO

日付: 2026-11-08。ホスト `tensor`。HEAD `b2a3549a12c983b5b702d38727dba6c5a1c62913`。作業木はテストの収集と spawn の import、TensorBoard の期待文だけが dirty である。

総合判定は **GO**。本格 400 / 121 / 35 / 150 epoch は始めていない。第三者検証は始めていない。

Preflight の PASS と、未学習または 1 更新後の損失は、公式再現の成功でも収束後の性能でもない。

## モデル別

| モデル | 判定 | 根拠 | 残るもの |
|---|---|---|---|
| MP-SENet | GO | 本格 YAML の Preflight PASS。snapshot `265f256b…`。2 GPU で 3 更新、別プロセス再開、824 件の全長 validation と PESQ。rank 間 checksum は一致 | Scheduler は保存する。16 kHz 変換器は soxr HQ のまま未確定。`deterministic: false` なので再開重みのビット一致は主張しない |
| MossFormerGAN | GO | Preflight PASS。snapshot `ba0c77c9…`。SyncBatchNorm 8。蓄積 2。`official_step` 6 と `global_step` 3。824 件は 412+412 で交差 0。再開の case id は一致 | 公式 2 GPU は 2 回目の Discriminator backward を同期しない。audio-db は増分を平均するので、公式重みとは一致しない。この差は残す。CUDA illegal memory access は今回も未再現 |
| TF-GridNet | GO | Preflight PASS。snapshot `a69a477a…`。cache 1,024。2 GPU の 3 更新と再開。validation は計画の保持集合と一致。損失は有限 | 固定 `mix_seed`。公式 WAV とのビット一致も統計的同等も未測定。整列で落ちる 6 発話は `iterator_stop` の末尾 |
| TF-Locoformer | GO | Preflight PASS。研究用 snapshot `e4495fa6…`。4 GPU、cache 1,024。1,024 件を完走。`ave_5best` は実 state dict の fixture で差 0 | 研究用 1,024 件であり、`cv_synthetic_small` でも `leakage_safe` でもない。DB は親 snapshot からの再構成。元ファイルとの同一性は未検証 |

## 必須条件

| # | 条件 | 結果 |
|---|---|---|
| 1 | 本格 YAML の Preflight | 4 件 PASS |
| 2 | Dataset と snapshot の identity | 指定ハッシュと一致。`load_snapshot` でも一致 |
| 3 | 公式幅の構築 | GPU 上で学習更新が終わった |
| 4 | 本格 world size の DDP | 2, 2, 2, 4 |
| 5 | 短い実データ学習 | 各モデル 3 更新。validation 試験は 1 更新 |
| 6 | 実損失で Optimizer 更新 | 有限な損失。MossFormer の失敗 micro-batch は 0 |
| 7 | Validation | MP 824、Moss 824、GridNet は計画どおり、Locoformer 1,024 |
| 8 | Checkpoint | `step_*.pt` と、chunk モデルの epoch ファイル |
| 9 | 別プロセス再開 | 4 モデルとも step 1 から step 3。case id は一致 |
| 10 | Data plan と case id | 再開の case id が連続実行と一致。Locoformer の実行 case 集合は 1,024 |
| 11 | RNG | world 2 と 4 の checkpoint に `rng_by_rank` がある。欠けると再開できない経路を通った。ビット一致は主張しない |
| 12 | Rank 間の状態 | 更新後のパラメータ総和の rank 間差は 0 |
| 13 | Checkpoint 保持 | fixture で latest 3、best、n-best 5、平均、中断時の前ファイル |
| 14 | 回帰 | 最終スイートは 411 PASS、11 FAIL、14 SKIP。FAIL は DeepFilterNet3 の glibc、`resource.duckdb` の `eval`、欠落した P8 だけ。4 モデルの経路は PASS |
| 15 | 開始を止めるコード BUG | 残っていない |

TF-Locoformer の追加:

| 条件 | 結果 |
|---|---|
| 1,024 件の完走 | 各 rank 256 件、14 chunk、合計 1,024。交差 0。欠落 0。重み 14,336。628.9 秒 |
| `valid.loss` の集約 | 有限。9,057.88。未学習に 1 更新を足した値であり、収束後の性能ではない |
| n-best 5 | fixture。選ばれた 5 ファイルの平均と生成ファイルの最大絶対差 0 |
| `ave_5best` の読み込み | `strict=True` のあと差 0。GPU の 1 回の validation では件数が足りないので `ave_1best` だけができる |
| 来歴 | `reconstructed_from_verified_snapshot`。元 DB との同一性は未検証。親 36,000 件は変えていない |

## 公式との差

| 区分 | 内容 |
|---|---|
| `OFFICIAL_MATCHED` | 本格 world size、batch、optimizer、loss の設定。MossFormer の 824 の rank 分割。TF-GridNet の保持 chunk 計画。TF-Locoformer の warmup 式 `lr * step / 4000`。n-best は浮動小数の平均と整数の和 |
| `INTENTIONAL_DIFFERENCE` | MP-SENet は scheduler を保存する。16 kHz は soxr HQ。MossFormer は Discriminator 勾配を同期する。PESQ 失敗は全 rank で落とす。`global_step` は optimizer 更新。TF-GridNet は固定 `mix_seed`。TF-Locoformer は研究用 1,024 件。平均ファイルの拡張子は `.pt`。公式の共有乱数カウンタとはビット一致しない |
| `NOT_TESTED` | 本格 epoch。公式 WAV とのビット一致と統計的同等。Leakage-safe の登録。全 asset SHA の再計算。CUDA 再開のビット一致。`cv_synthetic_small` |
| `UNKNOWN` | 公式の 16 kHz 変換器。`cv_synthetic_small` の一覧。元の `reproduction.duckdb` との同一性 |
| `BLOCKED` | DeepFilterNet3 の公式素材と、このホストで読めない `libdf_rfft.so`。P8 DB の欠落。4 モデルの開始条件ではない |

これらの差は、今回の 4 モデルの学習開始を止めるコード欠陥ではない。公式 2 GPU との数値一致を開始条件にはしない。

## 残る issue

- Leakage-safe の VoiceBank Dataset は未登録。今回の GO には入れていない。
- 再構成 DB に VoiceBank、MS-SNSD、P7、過去 run は無い。snapshot から学習する。
- MossFormer の公式 2 GPU との重み差は意図した仕様のまま。
- `deterministic: false`。
- DeepFilterNet3 の glibc 不一致と P8 データの欠落は未解消。4 モデルの回帰失敗としては数えない。

## 第三者検証

今回は起動していない。再実行に必要なものは [GPU 検証](P10_C_FIX2D_GPU_VERIFICATION.md) と [検証マトリクス](P10_C_FIX2D_TEST_MATRIX.md)。重点は、研究用 1,024 件の case ログ、MossFormer の勾配同期を公式重み一致と取り違えないこと、再構成 DB を元 DB と呼ばないこと、未学習の validation 損失を性能と呼ばないことである。
