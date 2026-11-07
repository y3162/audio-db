# P10-C Fix.2-B Checkpoint 再開

比較の前に許容を固定した。離散状態は完全一致を要求する。対象は case id 列、plan hash、`global_step`、`official_step`、`accu_count`、学習率、BatchNorm の `num_batches_tracked`、RNG（Python、NumPy、Torch CPU、CUDA）である。

浮動小数は次のとおりである。

| 対象 | atol |
|---|---|
| パラメータ | `1e-3` |
| Adam の moment | `1e-3` |
| BatchNorm の浮動 buffer | `1e-4` |
| 蓄積中の勾配 | `1e-4` |

中断なしの 2 実行の差が atol を超える項目は、再開との差がその連続差の 2 倍以内なら、連続実行と同じばらつきとする。それを超える項目は `restore_mismatch` とする。最終損失が近いことだけでは再開一致にしない。

設定は中断なし、中断、再開で同じ YAML である。`verification_max_updates` は外していない。`deterministic: false`。本格 YAML との違いは、更新上限、checkpoint 間隔、出力先だけである。

## 連続実行

同じ seed、Dataset、world size 2 で、上限 3 更新を 2 回走らせた。どちらも `INTERRUPTED`、checkpoint は `step_00000003.pt`。`official_step` は 6、失敗は 0、`accu_count` は 0。plan hash は `13244439cac19d7548884593287bbeb3a013747fa7ba50aef7a6a54169ed107a`。RNG は一致した。

パラメータだけの最大絶対差は、Generator `9.94e-4`、Discriminator `1.48e-3` だった。BatchNorm `running_mean` の最大は `6.61e-4`。`num_batches_tracked` は一致した。

## 中断

同じ YAML で起動し、最初の完成 checkpoint のあとで両 worker に `SIGTERM` を送った。終了コードは 0。その時点の状態は `CHECKPOINTED`、checkpoint は `step_00000001.pt` だった。再開はこのディレクトリの最新の完成 checkpoint から行った。再開後の `status.json` は step 3 の `INTERRUPTED` で上書きされる。中断直後の記録は実行結果 JSON の方である。

中断までの case id は、中断なし実行の接頭辞と一致した。

## 別プロセス再開

再開は新しい `torchrun` である。同じ YAML、`strict_resume` 相当の照合、`--resume` に相当する checkpoint 読み込み。上限は 3 のままなので、step 1 から step 3 まで進んで `INTERRUPTED` になった。

| 項目 | 中断なし step 3 と再開 step 3 |
|---|---|
| `global_step` | 3 と 3 |
| `official_step` | 6 と 6 |
| `accu_count` | 0 と 0 |
| plan hash | 一致 |
| data state | 一致 |
| case id 列 | 3 更新とも一致 |
| 学習率 | `5e-4` と `1e-3` |
| RNG | Python、NumPy、Torch、CUDA とも一致 |
| `num_batches_tracked` | 一致 |
| `validation_pending` | どちらも無し |
| Plateau | 最良値は無限大、stalls 0。この実行は epoch 末まで行っていない |

パラメータだけの最大絶対差は、Generator `1.50e-3`、Discriminator `2.45e-3` だった。Discriminator は連続差 `1.48e-3` の 2 倍以内である。Generator の連続差は `9.94e-4` で atol 以下だったため、規則は限界を `1e-3` のままにする。再開差 `1.50e-3` はその限界を超えるので、規則上は `restore_mismatch` である。

state dict 全体の Discriminator 最大差は `cnn.bn6.running_var` の `1.45e-2` だった。連続実行の BatchNorm 差は `6.61e-4` で、2 倍規則を超える。Adam の Discriminator moment も `4.48e-3` で atol を超えた。

step ごとのパラメータ差は、再開の境界である step 1 で Generator `9.86e-4`、Discriminator `1.50e-4` だった。step 2 では Generator `1.08e-3`、Discriminator `1.45e-3`。広がったのはその後である。欠落した tensor は無い。

判定は、離散状態の復元は成立し、浮動小数は事前規則の `restore_mismatch` が残る、である。ビット一致の再開とは呼ばない。別の case 列や、更新回数のずれでは説明できない。`deterministic: false` の cuDNN で、プロセスをまたいだあとの BatchNorm 統計が連続 2 回より開いた、と記録する。

## 未完了の蓄積窓

成功した micro-batch だけが窓を埋める。checkpoint がそれを持つと主張するので、保存と復元を比べた。

失敗の置き方は、`official_step` が 1 の micro-batch を rank 0 だけ失敗させる。この番号は checkpoint の counter から復元されるので、再開プロセスが列を先頭からやり直しても同じ micro-batch を指す。

上限 2 更新。中断なし 2 回と、step 1 の完成 checkpoint への `SIGKILL`、別プロセス再開。`SIGKILL` の終了コードは 1。`step_00000001.pt` は残った。

| checkpoint | `global_step` | `accu_count` | `official_step` | Generator 勾配 | Discriminator 勾配 |
|---|---|---|---|---|---|
| 中断なし step 1 | 1 | 1 | 4 | 1053 | 93 |
| SIGKILL 時点 step 1 | 1 | 1 | 4 | 1053 | 93 |
| 中断なし step 2 | 2 | 1 | 6 | 1053 | 93 |
| 再開 step 2 | 2 | 1 | 6 | 1053 | 93 |

step 1 の勾配の最大絶対差は `3.98e-4` で、勾配 atol `1e-4` より大きい。空の窓を復元した状態ではない。再開後の case id 列は中断なしと一致し、RNG も一致した。step 2 の勾配差は `4.65e-3` まで開いた。規則上は `restore_mismatch`。保持しているのは未完了の窓であり、中断なしの同じ step とビット一致ではない。

epoch 先頭ではこの窓を捨てる。残余勾配の実行で、次 epoch の `start_epoch` は蓄積 1、勾配ありから、蓄積 0、勾配なしに戻した。

## validation_pending

`max_updates_per_epoch: 2`、`verification_max_updates: 2`、`validation_max_batches: 80`。epoch 末の Validation 前に pending 付き `step_00000002.pt` ができてから、両 worker を `SIGKILL` した。殺した時点のファイルは `validation_pending` を持ち、`global_step` は 2 だった。

別プロセスの再開は、Optimizer を進めず `global_step` 2 のまま Validation を 1 回確定し、`INTERRUPTED` で止まった。metrics の `validation` で `validation_committed` が真なのは 1 行である。再開後の checkpoint に pending は無い。Plateau の最良値は 0.3866442861035466、stalls は 0、学習率は `5e-4` のままである。二重の確定は観測していない。

この 80 batch は smoke の接頭辞である。824 件の分割の証拠は安定性の文書の 412 と 412 である。

殺した時点では、epoch 末の train 行が metrics にまだ無い。Validation の前に pending checkpoint を書き、train 行はその後に足す。`SIGKILL` がその間に入ると、checkpoint の `global_step` は 2 で、jsonl の train 行は 1 のままになる。再開は checkpoint の step を信頼し、その更新をやり直していない。

## 残った差異

- Generator パラメータは、事前規則の atol を `1.50e-3` 対 `1e-3` で超えた。連続差は `9.94e-4`。
- Discriminator の `running_var` は `1.45e-2` で、連続差 `6.61e-4` の 2 倍を超えた。
- 蓄積勾配は step 1 で `3.98e-4`、再開後の step 2 で `4.65e-3`。
- ファイルの SHA-256 は一致させない。`deterministic: false` の重みが入るためである。
- 公式 2 GPU の割れた Discriminator 更新とは比べていない。Fix.2-A の意図した差のままである。
