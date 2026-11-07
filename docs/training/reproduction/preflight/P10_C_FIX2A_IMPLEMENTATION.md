# P10-C Fix.2-A 実装

MossFormerGAN の学習更新を、固定した ClearerVoice-Studio `6b3774dc79c46ae8bed2a4fa5f706f0ac8c75c61` の実効的な backward、蓄積、clip、Optimizer 更新に合わせた。本格学習は始めていない。第三者検証はしていない。

開始時の `HEAD` は `696da4406462c57d94d796d57e1b2fede6a8b8be`、ブランチは `feature/reproduction`、`git status --short` は空だった。公式チェックアウトの commit は `6b3774dc79c46ae8bed2a4fa5f706f0ac8c75c61` で、公式ソースは変更していない。比較に使った PyTorch は `2.6.0+cu124`。`optimizer.zero_grad()` の既定は `set_to_none=True` である。

## 修正前の問題

Fix.1 は、公式が `no_sync` を使わず、蓄積済み勾配まで DDP が平均し直す、と書いていた。その再平均をコピーしない、という実装説明になっていた。今回、その説明を実行で見直した。

同期が実際に起きる backward では、全 rank で同じ蓄積勾配をもう一度平均しても、新しい増分だけを平均して足す結果と一致した。差が出るのはそこではない。

公式の Discriminator は 1 micro-batch で 3 回 forward し、そのあと Generator 損失と Discriminator 損失を別々に backward する。PyTorch 2.6 の DDP（`find_unused_parameters=True`、`broadcast_buffers=True`）では、1 回目の backward は Generator と GAN 項の Discriminator 勾配を rank 間平均する。2 回目の backward は Discriminator 損失の勾配を all-reduce しない。rank ごとに局所勾配が残り、AdamW の 1 段で重みが割れる。

1 GPU にはこの collective が無い。1 GPU の損失、clip、Optimizer 更新は公式ループと同じ順序だった。

## 公式コードとの比較

公式 `solver.py` の `_run_one_epoch_mossformergan_se_16k` は、epoch の最初に `accu_count = 0` と両 Optimizer の `zero_grad()` を行う。各 micro-batch は次の順である。

1. 正規化と STFT のあと Generator を 1 回 forward する。
2. `loss_mossformergan_se_16k` が Generator 損失を計算する。この中で Discriminator を 1 回 forward する。
3. 損失が NaN、または PESQ が無いときは `None, None` を返し、backward しない。`self.step` は増える。`accu_count` は増えない。
4. PESQ があるときは Discriminator をさらに 2 回 forward し、Discriminator 損失を作る。
5. `accu_grad` が有効なとき、損失を `effec_batch_size / batch_size` で割ってから backward する。公開 YAML ではこの比は 2 である。
6. Generator の backward の直後に `clip_grad_norm_(model, 10)` する。
7. Discriminator の backward の直後に `clip_grad_norm_(discriminator, 10)` する。
8. `accu_count` が 2 になったとき、Generator を `step` して `zero_grad` し、続けて Discriminator を `step` して `zero_grad` する。
9. 失敗した micro-batch も含め、`self.step` を 1 増やす。

DDP は `SyncBatchNorm`、`find_unused_parameters=True`、`broadcast_buffers` の既定 True で、`no_sync` は無い。Optimizer は AdamW、Generator の学習率 `5e-4`、Discriminator はその 2 倍、`weight_decay` はコンストラクタ既定の `0.01` である。

## 実装方針

1 GPU の順序は公式のままにする。損失を蓄積回数で割り、backward し、そのモジュールの走っている勾配和を clip し、窓が埋まったら両 Optimizer を更新する。

2 GPU 以上では forward を `no_sync` にし、新しい増分だけを SUM して world size で割ってから走っている和へ足し、その和を clip する。これは、同期が起きている backward については公式 DDP と同じ値になる。2 回目の Discriminator backward については公式と違う。公式はその勾配を all-reduce せず rank が割れる。割れは安全な DDP と両立しないのでコピーしない。`INTENTIONAL_DIFFERENCE` として残す。

片 rank だけの PESQ スキップもコピーしない。いずれかの rank が落とす micro-batch は全 rank が backward しない。これも `INTENTIONAL_DIFFERENCE` である。

実行時の勾配計算は変えていない。変えたのは、再平均を欠陥だと断定していた説明である。測定は `P10_C_FIX2A_NUMERICAL_COMPARISON.md` に書いた。

## 変更ファイル

- `src/training/se/mossformergan.py`
- `tests/training/test_mossformer_update_rule.py`
- `tests/training/test_mossformer_pesq_skip.py`
- `.agents/DECISIONS.md`
- `docs/training/reproduction/preflight/P10_C_PREFLIGHT_FIX_ISSUES.md`

MP-SENet、TF-GridNet、TF-Locoformer、Dataset、DB、Renderer、Preflight 設定、Checkpoint 保持は変更していない。公式ソースは変更していない。

## 修正内容

`train_update` と `_backward_average_and_clip` の説明を、測定した DDP の挙動に合わせた。増分平均、micro-batch ごとの clip、PESQ 失敗で窓を埋め直す処理、`official_step` と Optimizer 更新の分離は、Fix.1 の実行時の形のままである。

`PyTorch 2.6` の `zero_grad()` は既定で `set_to_none=True` である。audio-db の `set_to_none=True` は、この版の公式呼び出しと同じである。

## 回帰テスト

次を CPU で実行した。

```text
.venvs/se/bin/python -m unittest \
  tests.training.test_mossformer_update_rule \
  tests.training.test_mossformer_pesq_skip.MossFormerPesqSkipTests.test_failure_sequences_keep_the_official_window \
  tests.training.test_mossformer_pesq_skip.MossFormerPesqSkipTests.test_epoch_start_drops_a_partial_window
```

4 件は PASS だった。続けて `test_mossformer_pesq_skip`、`test_mossformer_sampler`、`test_mossformer_epoch_plateau` を実行し、16 件が PASS だった。`test_pesq_skip_does_not_increment_or_validate` もこのセッションで PASS だった。

`test_mossformer_update_rule` は world size 2 の gloo で、1 forward につき 1 backward の蓄積が公式型 DDP と増分平均で重み差 0 であることを確認する。clip 閾値 0.5 で、micro-batch ごとに clip した重みと、最後に 1 回だけ clip した重みは 5e-4 より大きく離れる。clip を実質無効にするとその差は 0 である。3 forward と 2 backward では Generator の rank 差は 0、公式型 Discriminator の rank 差は 1e-3 より大きく、増分平均の rank 差は 0 である。

PESQ の列は、蓄積回数 2 で次を確認した。

| 列 | 成功数 | 失敗数 | Optimizer 更新 | `official_step` | 窓 |
|---|---|---|---|---|---|
| 成功、成功 | 2 | 0 | 1 | 2 | 0 |
| 失敗、成功、成功 | 2 | 1 | 1 | 3 | 0 |
| 成功、失敗、成功 | 2 | 1 | 1 | 3 | 0 |
| 成功、成功、失敗 | 2 | 1 | 1 | 3 | 0 |
| 失敗、失敗 | 0 | 2 | 0 | 2 | 0 |

失敗だけの列は重みが変わらず、勾配は `None` のままである。成功 1 件で止めた窓は `_accu_count == 1` で勾配が残る。`start_epoch(resume_mid_epoch=True)` はその窓を残し、`False` は勾配を消して窓を 0 に戻す。

データカーソルはタスクの中には無い。Trainer は micro-batch を読むときにカーソルを進め、`update_applied` が 0 のときは `global_step` を増やさずに次へ進む。戻さない。この分岐は既存の `test_pesq_skip_does_not_increment_or_validate` が担当している。

## 未解決事項

- 公式 `train.py` を VoiceBank の全 epoch で実行していない。1 更新のループ転写と比較である。
- 2 GPU の Discriminator 重みは、公式の割れた更新と一致させない。
- CUDA 上の 1 更新は、公式ループを 2 回繰り返しても重みが最大約 4.4e-4 動く。ビット一致は主張しない。
- CUDA illegal memory access の長時間追跡、Checkpoint の本格再開、全モデル回帰は Fix.2-B 以降である。
- 今回の実行で新しい CUDA illegal memory access は出ていない。
