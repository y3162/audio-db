# P10-C Fix.2-D 検証マトリクス

日付: 2026-11-08。HEAD `b2a3549a12c983b5b702d38727dba6c5a1c62913`。作業木には、pytest 収集を止めないための準備スクリプトのガードと、TensorBoard プレビュー失敗文の期待値の更新がある。

## 環境

| 項目 | 値 |
|---|---|
| ホスト | `tensor` |
| Python | 3.12.13 |
| PyTorch | 2.6.0+cu124 |
| CUDA | 12.4 |
| DuckDB | 2.0.0 |
| 仮想環境 | `.venvs/se` |

GPU 試験の物理番号は 1, 2, 6, 7。詳細は [GPU 検証](P10_C_FIX2D_GPU_VERIFICATION.md)。

## マトリクス

| 対象 | 環境 | 方法 | 期待 | 結果 | 未検証 |
|---|---|---|---|---|---|
| 本格 YAML 4 件 | CPU | `python -m src.training.se.cli --preflight` | PASS。指定の dataset と snapshot。スモーク cap なし | 4 件とも終了コード 0、PASS | 公式再現の成功ではない |
| Snapshot 同一性 | CPU | `load_snapshot` | 指定ハッシュと一致 | 4 モデルと親 DNS が一致 | 全 asset SHA の再計算 |
| 研究用 1,024 | CPU | case id 集合 | 親 valid の部分集合。train / test は親と一致 | 一致。重複 0 | — |
| pytest `tests/training` `tests/reproduction` | CPU | `.venvs/se` | コード由来の FAIL が残らない | 最終実行の件数は [回帰](P10_C_FIX2D_REGRESSION.md) | DeepFilterNet3 の glibc、欠落した P8 DB |
| Checkpoint 原子性、latest、n-best | CPU | 既存テストと `nbest_evidence.py` | 中断で前ファイルが残る。5 件平均が手計算と一致 | 平均の最大絶対差 0。strict load の差 0 | 150 epoch の実ファイル |
| Resume 設定拒否 | CPU | 既存 `resume_config_mismatch` テスト | `logging` 以外を拒否 | スイートに含まれ、失敗一覧に無い | — |
| `rng_by_rank` | GPU world 2 と 4 | checkpoint を読む | rank 数と現在 device | MP / Moss は 2、Locoformer は 4 | world 1 の再確認は既存テスト。CUDA ビット一致は主張しない |
| MP-SENet 2 GPU | 物理 1, 2 | 3 更新、別プロセス再開、824 validation | 公式損失、全長、PESQ、checkpoint、rank 一致 | 実施。PESQ count 824 | 400 epoch |
| MossFormerGAN 2 GPU | 物理 6, 7 | 3 更新、再開、824 の rank 分割 | SyncBatchNorm、蓄積 2、`official_step` と `global_step` | 412+412、`official_step` 6、`global_step` 3 | 121 epoch。CUDA illegal access は未再現 |
| TF-GridNet 2 GPU | 物理 6, 7 | 3 更新、再開、epoch 末 validation | 固定 seed の公式損失、cache 1,024 | 実行集合は計画の保持集合と一致 | 公式 WAV との一致。35 epoch |
| TF-Locoformer 4 GPU | 物理 1, 2, 6, 7 | 3 更新、再開、1,024 全件 | cache 1,024、warmup、欠落 0 | rank あたり 3,584 chunk、和 1,024、交差 0 | 150 epoch。`ave_5best` は fixture |
| TensorBoard | GPU rank 0 | MP-SENet の events ファイル | rank 0 だけが書く | 1 ファイル。他 rank の events は無い | 音声プレビューは無効のまま |
| Leakage-safe | — | snapshot と DB を一覧 | 今回の GO 条件にしない | 未登録 | 登録と学習 |

## 本格 YAML の Preflight

保存先は `data/results/training/reproduction/fix2d/preflight/<model>/`。JSON と Markdown がある。

| モデル | 終了 | dataset | snapshot | config SHA-256 | world | global / micro / accum |
|---|---|---|---|---|---:|---|
| MP-SENet | 0 PASS | `mp_senet.vb28.ondemand.official@1` | `265f256b…` | `c4986eb51eb5f143405bf49e87fcbe220f86e3462fced46bb3207c4779f36913` | 2 | 4 / 2 / 1 |
| MossFormerGAN | 0 PASS | `mossformergan.vb28.ondemand.official@1` | `ba0c77c9…` | `ba416d6b3ab87e33eee31d3f021b1559969740eb087649b1c7fb47b9136b7fac` | 2 | 4 / 1 / 2 |
| TF-GridNet | 0 PASS | `tf_gridnet.ms_snsd.espnet_recipe@1` | `a69a477a…` | `8a25d3fd446a0dda9c7186c8c96c354df502f4a78109e597d6f2d43a7ae38d74` | 2 | 4 / 2 / 1 |
| TF-Locoformer | 0 PASS | `tf_locoformer.dns.accept_recipe.research_valid_fixed@1` | `e4495fa6…` | `2002547bfb6118d3d1cb85caac87e66030cf9536f77a20b5895368c29544beb0` | 4 | 4 / 1 / 1 |

件数は MP / Moss が train 11,572、valid 824、test 824。TF-GridNet が 7,345 / 1,835 / 305。TF-Locoformer が 324,000 / 1,024 / 150。ディスク検査は PASS。空きは約 1.89 TB。スモーク cap は失敗リストに出ていない。Preflight の PASS は公式再現の成功ではない。
