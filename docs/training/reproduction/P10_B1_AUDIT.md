# P10-B.1 監査

対象は MP-SENet、MossFormerGAN、TF-GridNet、TF-Locoformer。HEAD は `b9774e9f397b73607f0df58814f671a69546d70d`（`feature/reproduction`）。本格再現学習は開始していない。検証の数値は [P10_B1_VERIFICATION](P10_B1_VERIFICATION.md)、開始計画は [P10_C_EXECUTION_PLAN](P10_C_EXECUTION_PLAN.md)。

前のエージェントは commit せずに作業を残して離れた。このセッションはその差分を採否し、足りない検証を足した。

## 修正前

P10-B は 4 モデルで Optimizer 更新 3 回、部分 Validation、Checkpoint、別プロセス再開を確認した。再開の手順は、`max_steps` 到達でできた `complete.json` を `complete.phase-train.json` へ改名するものだった。公式完了と検証用の途中停止が同じファイルになっていた。

残っていた回帰は、P10-B の `tests/training` 230 件中 11 失敗、12 SKIP である。報告されていた原因は、TensorBoard DDP の一括失敗、子プロセスへの `sys.path` 汚染、`configs/training/mp_senet.yaml` の `deterministic`、`configs/training/trial100/` の欠落、`libdf`、`mamba_ssm`。

公式との設定差として、放棄された作業ツリーは MP-SENet の検証・Checkpoint 間隔を 2,893 に変えていた。公開 CLI は 5,000 更新ごとである。

## 公式実装との比較

| 項目 | 公式 | audio-db |
|---|---|---|
| MP-SENet 間隔 | 公開 `train.py` は 5,000 更新ごとに検証と Checkpoint。ExponentialLR は epoch 末の 1 回。best PESQ は epoch 40 以降 | `configs/training/reproduction/mp_senet_vb28_official.yaml` を 5,000 と epoch 40 に戻した。1 epoch は 11,572 / 4 = 2,893 更新。400 epoch は `max_steps` 1,157,200。最初にゲートを満たす保存点は step 120,000（epoch index 41） |
| MP-SENet 再開 | 公式は Scheduler 状態を保存せず、再開で ExponentialLR が余分に減衰し得る | audio-db は Scheduler 状態を保存する。この差は修正しない。CPU で、再開が余分に減衰しないことを確認した |
| MossFormerGAN | `max_epoch: 120` だがループは `range(start_epoch, max_epoch+1)` で 121 epoch。2,893 × 121 = 350,053。検証は生成器損失。PESQ 失敗は 0 を足して分母に残す。5 回非改善で G の LR だけ半減、10 回で停止。D の LR は G の 2 倍。AdamW の weight decay は 0.01 | `max_steps: 350053`。plateau は CPU の合成履歴で、悪化・同値・5 回目の半減・10 回目の停止・再開後の stall 保持を確認した。GPU の 12 更新では半減まで進んでいない |
| TF-GridNet | 2 rank、global batch 4、Adam 1e-3、clip 5、`mr_l1_tfd`、ReduceLROnPlateau factor 0.7 patience 1、`num_samples_per_epoch` 5,000、chunk 48,000、35 epoch、選択は `valid.loss` | スモーク YAML はこれを保持する。`max_steps: 2000000` は公式の更新総数ではない。このホストにスナップショットが無く、1 epoch の更新数は未計測 |
| TF-Locoformer | 4 rank、global batch 4、chunk 64,000、shift 0.5、rank あたり 1,000 発話、150 epoch、warmup 4,000、factor 0.5、patience 3。`valid.loss.ave_5best` は損失上位 5 epoch の平均 | このホストの受理スナップショットで、epoch 1–3 は 14,000 更新。150 × 14,000 = 2,100,000。検証は rank あたり 126,000 chunk。`cv_synthetic_small` ではない。`ave_5best` は合成 Checkpoint の CPU テストで、float は平均、整数 Tensor は和。拡張子は公式の `.pth` ではなく `.pt` |

## 修正内容

放棄された作業が入れ、このセッションで残したもの:

- `verification_max_updates`。`max_steps` 以上は設定読み込みが拒否する。到達時は `interrupted.json` と `status.json` の `INTERRUPTED`。`complete.json` は書かない。同じディレクトリで上限を上げれば再開できる。同じ上限の再開は拒否する。
- 公式の `max_steps` 到達、epoch 上限、early stop、plateau 停止は `COMPLETED`。例外は `FAILED` で、`complete.json` は書かない。
- 期限と `segment_steps` は `CHECKPOINTED`。その管理実行が chunk の `max_steps` に達した場合だけ `TrainingError` になり、例外処理が `FAILED` を書く。
- `strict_resume` は `verification_max_updates` と logging を実験条件から外す。
- 子プロセスはリポジトリルートだけを `sys.path` に置く。
- `configs/training/trial100/` の 5 本は、削除 commit `b09b7f3` の直前とバイト一致で戻した。ダミーではない。
- MossFormerGAN の同値 plateau、TF-Locoformer の `ave_5best` 数値、MP-SENet の `deterministic: false` 期待値。

このセッションで足したもの:

- MP-SENet 公式 YAML の間隔を 5,000 に戻した。テストは「5,000 は 2,893 の倍数ではない」ことを見る。
- Trainer が epoch 末の Scheduler を呼び、再開で LR が余分に減衰しない CPU テスト。
- `AUDIO_DB_STEP_SYNC=1` のとき、更新と検証の前後で CUDA を同期する。JSONL の学習行は `update_seconds` と `load_wait_seconds`。検証行は `seconds`。Checkpoint 公開の後に `checkpoint_io` 行を書く。
- TensorBoard の公式幅 2 rank テストの待ち時間を 300 秒から 1,500 秒にした。SEMamba++ は 293 秒、MossFormerGAN は 669 秒で、どちらも step 1 のあと計算が続いていた。
- `tests/training/test_df_rfft_kernel.py` の `tearDown` が `DF_RFFT_LIBRARY` を無条件に消していた。一括実行では、その後の DeepFilterNet3 が共有 `.so` を読んで落ちる。終わったら開始時の値へ戻す。
- TensorBoard の再開は `purge_step` で古い event の wall time を書き換える。中断境界のテストは JSONL の順序を見て、スカラーは step で並べて比較する。

共有の `src/se/models/deepfilternet3/libdf_rfft.so` は置き換えていない。このホストの glibc 2.31 では `GLIBC_2.33` が読めない。`make df-rfft` の出力 `build/df-rfft/release/libdf_rfft.so` を `DF_RFFT_LIBRARY` で読むと数値テストは通る。

## モデル別の残差

| モデル | 残る差 |
|---|---|
| MP-SENet | 公式は再開時に Scheduler を保存しない。audio-db は保存する。16 kHz 変換器の公式実装は不明のまま。このホストに `mp_senet.vb28.ondemand.official@1` のスナップショットは無い |
| MossFormerGAN | GPU では plateau の半減と 10 回停止まで進んでいない。824 発話の全件検証はしていない。検証 32 件は 5.43 秒 |
| TF-GridNet | スナップショットがこのホストに無い。1 epoch の更新数、同期した step 時間、新しい中断経路の GPU 再実行は未計測。P10-B の H100 スモークは、`complete.json` を改名する経路である |
| TF-Locoformer | 全件検証は 126,000 chunk/rank のうち 1 件と 32 件だけ。`ave_5best` の実 Checkpoint は 5 epoch 未満なので、数値確認は合成 Checkpoint である。`cv_synthetic_small` は不明のまま |

## テスト失敗の分類

P10-B の 11 失敗と、このセッションの一括実行（作業ツリーを変えながら走った 238 件、失敗 4、エラー 6、SKIP 12）の分類。

| 現象 | 分類 | 扱い |
|---|---|---|
| MP-SENet 間隔を 2,893 と期待するテスト | テストの期待値が古い | 公式 5,000 に合わせたテストへ差し替えた。単独で成功 |
| `deterministic: false` | テストの期待値が古い | 現行 YAML に合わせた。成功 |
| `trial100` 欠落 | 設定の欠落 | 削除前の実ファイルを復元。ダミーは置いていない |
| DDP 子プロセスの `sys.path` | テスト間の状態汚染 | 子プロセスの path を固定した |
| SEMamba++ と MossFormerGAN の 2 rank が 300 秒で打ち切られる | テストの待ち時間が短い | 1,500 秒にした。293 秒と 669 秒で成功。デッドロックではない |
| `libdf_rfft.so` の `GLIBC_2.33` | 実行環境。加えてテストが環境変数を消す状態汚染 | 共有バイナリは維持。`tearDown` は `DF_RFFT_LIBRARY` を戻す。指定が無いホストでは共有 `.so` の ERROR が残る。DeepFilterNet3 は本学習対象外 |
| `libdf` 未導入の SKIP | 実行環境 | SKIP のまま。本学習 4 モデルの経路ではない |
| `torch_complex` 欠如による ESPnet loss の SKIP | 実行環境 | 1 件。自前の SI-SDR 符号テストは成功している |

`mamba_ssm` は、この環境の SEMamba++ 2 rank テストが 293 秒で成功したので、import 不能ではない。

一括実行の最終件数は [P10_B1_VERIFICATION](P10_B1_VERIFICATION.md) に書く。

## 本格学習への影響

中断は `interrupted.json` で再開できる。`complete.json` の改名は要らない。公式予算まで走ると `complete.json` が再開を拒む。

MP-SENet と TF-GridNet のスナップショットは、このホストの `data/db/snapshots/` に無い。`reproduction.duckdb` も無い。新規生成はしていない。この 2 モデルの P10-C は、スナップショットがあるマシンで、新しい中断経路をその公式幅でも一度確認してから始める。

MossFormerGAN と TF-Locoformer は、このホストの公式幅で中断・再開・連続ランの計画一致まで確認した。全件検証と公式 epoch 数までは走っていない。
