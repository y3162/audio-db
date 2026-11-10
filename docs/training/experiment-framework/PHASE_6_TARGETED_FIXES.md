# Phase 6: 研究実験前の残課題 4 件

判定: **GO。** これは設計書の Phase 5（LoRA、難例 Sampler、複数 Expert、Distillation、勾配を通す ASR）ではない。新しい学習アルゴリズムも足していない。損失、Optimizer の更新順、PESQ skip、勾配蓄積、plateau、validation 間隔、scheduler は変えていない。本格候補 YAML は起動していない。公式データ全量、論文スコア、実 ASR の WER は測っていない。

2026-11-10。対象テストは CPU で、Python は `.venvs/se` である。SEMamba++ の import と bandwidth は `.venvs/semambapp` で、CUDA を隠して実行した。fused selective scan の短い forward だけ、物理 GPU 1 の NVIDIA A100-PCIE-40GB で実行した。GPU 0、3、4、5 は使っていない。

## 1. 変更ファイル

- `src/experiments/mossformergan.py`
- `src/experiments/mp_senet.py`
- `src/experiments/tf_gridnet.py`
- `src/experiments/tf_locoformer.py`
- `src/experiments/deepfilternet3.py`
- `src/experiments/semambapp.py`
- `src/experiments/trainers.py`
- `src/evaluation/provenance.py`
- `scripts/experiments/phase5_launch.py`
- `requirements/semambapp.in`
- `requirements/semambapp.lock.txt`
- `configs/experiments/production/` の checkpoint 間隔
- `configs/experiments/production/mossformergan_se_16k_leakage_safe.yaml`（新規。未起動）
- `configs/experiments/readiness/mossformergan_se_16k_leakage_safe.yaml`（新規。CPU の 2 logical step）
- `tests/experiments/test_mossformergan.py`
- `tests/experiments/test_phase6_interrupt.py`
- `tests/experiments/test_phase6_launch.py`
- `tests/experiments/test_phase6_mossformergan_safe.py`
- `tests/experiments/test_phase6_resampy.py`
- `tests/evaluation/test_provenance.py`
- `tests/evaluation/test_phase6_wer_path.py`

checkpoint schema は `audio-db-experiment-v1` のままである。`stop_after_logical_steps` は config hash から外したままなので、上限付きの中断と、上限を外した再開は同じ algorithm config になる。

## 2. MossFormerGAN Leakage-safe

`MossFormerGANTrainer` は `official_faithful` と `leakage_safe` を受ける。未知の claim は構築時に `BuildError` である。profile と `selection_uses_test` の検査は `run()` の `_check_claim` で行う。

- algorithm が `fixed_pair.mossformergan_se_16k` 以外なら拒否する。
- `official_faithful` は profile が `None` または `official_faithful` で、`selection_uses_test` が false だと拒否する。
- `leakage_safe` は profile が `None` または `leakage_safe` で、`selection_uses_test` が true だと拒否する。
- profile が `None` の合成データは、これまでの単体テストと同じく両方の claim を通す。実 snapshot は `require_profile` と `_check_claim` の両方で見る。

公式 YAML `configs/experiments/production/mossformergan_se_16k_official_2gpu.yaml` は `official_faithful` のままである。snapshot は `mossformergan.vb28.ondemand.official@1`、train 11,572、world size 2 の global batch 4 で 2,893 updates/pass、121 pass、350,053 logical step、seed 777、accumulation 2、micro-batch 1 である。world size 1 は global batch 2 になり、この公式条件ではない。

leakage-safe の本格候補は `configs/experiments/production/mossformergan_se_16k_leakage_safe.yaml` である。snapshot は `mossformergan.vb28.ondemand.safe@1`、dataset id `5748381b83c289f58485062ab62f099f48899ec98456d85a8e487bf09cb0d8b6`、profile `leakage_safe`、claim label `leakage_safe_comparison`、`selection_uses_test` は false、train 10,700 / valid 872 / test 824、窓 32,000 sample である。world size 2 の global batch 4 では 2,675 updates/pass、121 pass、323,675 logical step になる。出力先は `data/results/experiments/phase6-production/mossformergan-se-16k-leakage-safe` で、公式出力とは別である。このファイルは起動していない。2 GPU の NCCL 短時間実行は **NOT_RUN** である。

CPU の readiness は world size 1、`deterministic: true`、`distribute: none`、`max_logical_steps: 2`、`stop_after_logical_steps: 1`、`checkpoint_interval: 2` である。これは production の global batch 4 ではない。検証は一時ディレクトリへ出力先を書き換え、committed な readiness ディレクトリは起動していない。

確認したこと:

- dataset identity は profile `leakage_safe`、名前 `mossformergan.vb28.ondemand.safe`、claim label `leakage_safe_comparison`、`selection_uses_test` false。
- `valid.cases` と `test.cases` の case id は交わりが空である。
- `validation_epoch(0)` の最初の batch の sample id は valid にあり、test には無い。872 件の validation forward と plateau は回していない。
- 1 logical step で `INTERRUPTED`、`complete.json` は無い、`step_00000001.pt` がある。
- provenance は上の identity を記録する。checkpoint の trainer state は `experiment_claim=leakage_safe`、`logical_step=1`。
- 別プロセスの resume は `COMPLETED`、`complete.json` がある。metrics の logical step は `[1, 2]`。`step_00000002.pt` の G と D の optimizer update はどちらも 2、data の `update_index` は 0 より大きい。

最初の readiness は `checkpoint_interval: 100` だった。完了状態と metrics `[1, 2]` までは到達し、`step_00000002.pt` は interval に当たらず書かれなかった。間隔を 2 に直して再実行し、246.392 秒で 2 件とも OK である。step 1 は停止境界、step 2 は interval が書く。

## 3. 公式経路との数値一致

同じ初期パラメータ、同じ音声テンソル、同じ PESQ stub、各 claim の前に戻した `torch` RNG で、`official_faithful` と `leakage_safe` の 1 更新を比べた。loss、勾配、更新後パラメータ、G と D の AdamW state の最大差は 0 である。公式 snapshot の音声と safe snapshot の音声の loss は比べていない。損失、G/D の順、PESQ skip、蓄積、plateau の式は変えていない。

## 4. Checkpoint 周期

`checkpoint_interval` の単位は logical step（optimizer 更新）で、`logical_step % interval == 0` のときに `step` を書く。epoch 末と validation は従来どおり `validation_pending` または別の `step` を書く。間隔を変えても `validation_interval` と scheduler のタイミングは変えていない。

Phase 5 の 1 更新あたりの秒数（checkpoint 書き込みを除く）から、最大の空きが約 15 分を超える production YAML だけを、約 7〜8 分になる間隔へ短くした。新しい長時間計測はしていない。

| YAML | 変更前 | 変更後 | Phase 5 の見積もり |
|---|---:|---:|---|
| TF-GridNet MS-SNSD | 1,000,000 | 100 | 4.0 秒/更新。約 6.7 分に書き込み約 2.7 秒 |
| TF-Locoformer DNS | 1,000,000 | 60 | 7.4 秒/更新。約 7.4 分に書き込み約 6.8 秒 |
| MP-SENet 公式 GAN | 5,000 | 210 | 2.1 秒/更新。約 7.4 分。`validation_interval` は 5,000 のまま |
| MP-SENet leakage-safe | 2,675 | 210 | 同じ 2.1 秒の見積もり。`validation_interval` は 2,675 のまま |
| MossFormerGAN 公式 2 GPU | 2,893 | 90 | 5.3 秒/更新。約 8.0 分に書き込み約 3.6 秒 |
| MossFormerGAN leakage-safe | 新規 | 90 | 公式と同じ見積もり。未起動 |
| SEMamba++ micro 1 / micro 4 / batch 8 | 1,000 | 45 | micro 4 は約 10.2 秒/更新。約 7.7 分に書き込み約 1.7 秒。`validation_interval` は 0 のまま |
| DeepFilterNet3 smoke | 1 | 1 | 変えていない。公式 22,711 step の schedule へは外挿しない |

SEMamba++ batch 8 の wall 間隔は **NOT_MEASURED** である。Phase 4-H の約 45.56 GiB OOM を、micro 4 の見積もりで成功扱いにしていない。45 は micro 4 の見積もりを batch 8 の YAML に写しただけで、batch 8 を学習してはいない。

## 5. SIGTERM / Timeout

停止フラグは `src/experiments/signals.py` の `ranks_should_stop` である。Trainer は GAN の optimizer 対の外でだけ見る。loop 先頭で停止し、`logical_step > 0` なら label `step` を書く。MossFormerGAN は `_accu_count > 0` でも書く。`_in_optimizer_pair` または `_in_official_step` の途中では拒否する。validation 境界の直後に既に `validation_pending` を書いた停止では、それを `step` で上書きしない。

`StandardTrainer` の短い検証は CPU、interval 3、`stop_after_logical_steps` 1、最大 3 である。1 step で `INTERRUPTED`、`complete.json` は無く、`step_00000001.pt` があり、`step_00000002.pt` は無い。上限を外した別プロセス resume は 3 step まで進み、連続実行と model、data cursor、optimizer update、rank 別 RNG が一致した。metrics の logical step は `[1, 2, 3]` で重複は無い。完了後は `COMPLETED` と `complete.json` がある。

checkpoint 書き込みは一時ファイル、fsync、`os.replace` である。replace 失敗の既存テストは、直前の完成ファイルを残す。書き込み途中の SIGKILL を実プロセスで切る試験は **NOT_TESTED** である。launcher は完成 checkpoint を消さない。

`scripts/experiments/phase5_launch.py` の猶予は、未指定なら `min(120, timeout/4)` 秒である。1200 秒なら 120 秒、8 秒なら 2 秒。猶予 0 と、猶予が timeout 以上の指定は拒否する。SIGTERM は `timeout - grace`、またはそれより早い `--signal-after` で送り、SIGKILL は hard timeout だけである。report は `grace_seconds`、`killed`、`checkpoints_at_kill`、`signaled`、`timed_out` を持つ。この launcher で長時間ジョブは起動していない。

## 6. SEMamba++ 環境

`requirements/se.in` は既に `resampy==0.4.3` を pin する。`requirements/semambapp.in` は `-r se.in` のまま、その pin を受けることをコメントした。`requirements/semambapp.lock.txt` には `resampy==0.4.3` を手で挿入した。依存の numpy、numba、scipy は lock に既にあった。`uv pip compile` による lock 全体の再解決はしていない。他の pin は動かしていない。

このホストの `.venvs/semambapp` には Phase 5 で入れた wheel がある。クリーンな環境の作り直しは **NOT_TESTED** である。

`.venvs/semambapp` で `resampy` 0.4.3 を import し、`bandwidth_limitation` を `kaiser_best`、`kaiser_fast`、`scipy`、`polyphase` で実行した。入力は 800 sample、440 Hz、16,000 Hz から 4,000 Hz、Butterworth 次数 4 である。出力は長さが同じで有限であり、入力とは一致しない。CUDA は隠しているので、この試験は fused kernel を実行しない。

fused の確認は、同じ環境で `SelectiveSSM(d_model=16, d_state=8, d_conv=4, expand=2, scan_backend="fused")` の 1 forward である。入力は `(1, 8, 16)`、勾配は切っている。物理 GPU 1 で `scan_configured=fused`、`scan_executed=fused`、出力は有限で shape `(1, 8, 16)` だった。これは SEMamba++ 全体の学習でも、公式幅の forward でもない。mamba-ssm との pin 衝突は、この import と forward では起きていない。

## 7. WER Pipeline

使ったのは既存の `src/evaluation` である。新しい評価基盤は作っていない。経路は Clean、Noisy、SE、Enhanced、Mock ASR、corpus WER である。ASR の重みは更新しない。Mock ASR の出力は音声を聞いていない固定文で、実 ASR の品質ではない。

実モデルは 1 つだけ通した。`dense_channel` 16、`num_tsconformers` 1 の MP-SENet を `load_evaluation_enhancer` から `src.se.load_enhancer` で CPU に載せる。`model.training` は false、PESQ と STOI は off、2 発話とも `num_ok` で、enhanced の corpus WER は 2/5 である。checkpoint fingerprint のキーは単一ファイルの `checkpoint` で、長さ 64 である。学習 checkpoint の `generator` キーとは別の記録である。

残る 5 モデルは評価 adapter が loader へ `name`、`variant`、`device` を渡すことまで確認した。対応は TF-GridNet `ms-snsd`、DeepFilterNet3 `deepfilternet3`、MossFormerGAN `se_16k`、SEMamba++ `semambapp`、TF-Locoformer `dns` である。これらの forward と実 WER は **NOT_TESTED** である。Parakeet と Whisper の評価は **NOT_TESTED** である。

SE が例外を出すと status は `error`、message は `enhancer failed` を含み、`num_error` 1、`num_ok` 0、group は空である。WER 0 にはしていない。

## 8. Corpus WER

正規化は `wer-norm-v1`、集計は発話 WER の平均ではなく `(ΣS+ΣD+ΣI)/ΣN` である。

| 発話 | 参照 | N | clean | noisy | enhanced |
|---|---|---:|---|---|---|
| synthetic-0000 | alpha beta gamma | 3 | 一致 | alpha beta（D=1） | alpha beta delta（S=1） |
| synthetic-0001 | one two | 2 | 一致 | one two two（I=1） | two（D=1） |

合計は clean 0/5、noisy は S=0 D=1 I=1 で 2/5、enhanced は S=1 D=1 I=0 で 2/5 である。発話 WER の平均 5/12 とは一致しないことをテストが確認した。clean、noisy、enhanced はすべて記録される。別プロセスの再集計は編集回数と WER が一致する。`enhance_seconds` は再実行で変わるので、group 全体のバイト一致は要求していない。resume は既存 2 レコードを再利用し、`results.jsonl` は 2 行、発話 id は重複しない。

## 9. Provenance と独立性

評価素材が train と selection から離れていても、`selection_uses_test` が true なら independent は false、`confirmed_independent` は false、`split_independent` は false、comparison は `TESTED`、理由は test split を選択に使っているという文である。hash の重なりがあるときは、これまでどおり重なりの理由を先に返す。

評価素材が無い official-faithful は independent が `UNKNOWN`、comparison は `NOT_TESTED` のままである。materials が空の provenance も `UNKNOWN` のままで、independent へ上げない。`is_confirmed_independent` が true になるのは boolean の true だけである。公式再現の test 選択は、独立な最終評価ではない。

## 10. 実行したテスト

すべて `CUDA_VISIBLE_DEVICES=""` の `.venvs/se` である。resampy の子プロセスだけ `.venvs/semambapp` を使う。

| 範囲 | 結果 |
|---|---|
| `tests.experiments.test_phase6_mossformergan_safe` | 2 件、246.392 秒、OK |
| Phase 6 の interrupt、MossFormerGAN claim 3 件、launcher grace、resampy 2 件、WER path 6 件、test 選択の独立性 | 14 件、37.636 秒、OK |
| `tests.evaluation.test_wer`、`tests.evaluation.test_provenance`、checkpoint の中断再開と atomic replace、Phase 4-J の停止境界 | 26 件、33.564 秒、OK |
| 物理 GPU 1 の fused `SelectiveSSM` forward | `scan_executed=fused`、出力は有限 |

`tests/experiments`、`tests/training`、`tests/reproduction` の全件は、この Phase 6 では再実行していない。Phase 5 の記録を、今回の PASS として扱わない。

## 11. 実行していない長時間テスト

- production YAML の本格学習と、数十 step を超える GPU 学習
- MossFormerGAN leakage-safe の 2 GPU / NCCL
- SEMamba++ の再学習、batch 8 の学習、batch 8 の wall 計測
- `.venvs/semambapp` のクリーン再構築
- 書き込み途中の SIGKILL
- 872 件の validation と plateau
- LibriSpeech、SNR sweep、Parakeet、Whisper
- `dfn3.online.official@1` と `semambapp.gsr.official@1` の解除

## 12. 未解決事項

重大な未解決の数値差、checkpoint 欠落、leakage、WER 集計の誤りは、上の対象テストには残っていない。次は未計測であり、失敗ではない。

- SEMamba++ のクリーン環境再構築は **NOT_TESTED**。
- batch 8 の checkpoint 間隔の wall は **NOT_MEASURED**。
- leakage-safe の 2 GPU 短時間学習は **NOT_RUN**。world size 1 は global batch が違う。
- 5 モデルの評価 forward と、実 ASR の WER は **NOT_TESTED**。
- 書き込み途中の SIGKILL は **NOT_TESTED**。replace 失敗時に前の完成ファイルが残ることは既存テストで確認した。
- 設計書の LoRA、難例 Sampler、複数 Expert、Distillation、勾配を通す ASR は未実装のままである。

## 13. 最終判定

**GO。** leakage-safe を公式経路と分けて構築でき、同じ初期値と同じテンソルの 1 更新は一致し、safe snapshot の validation は test を使わず、1 step の中断と別プロセスの 2 step 再開が通った。checkpoint 間隔は Phase 5 の見積もりで時間制限付きの空きを約 7〜8 分へ寄せ、SIGTERM は更新境界で checkpoint を残す。`resampy==0.4.3` は lock にあり、bandwidth の 4 方式と fused scan の短い forward は動いた。Mock ASR の corpus WER と、test 選択を独立評価にしない判定は通った。本格学習は開始していない。
