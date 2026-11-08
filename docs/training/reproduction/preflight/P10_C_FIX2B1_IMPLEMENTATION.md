# P10-C Fix.2-B.1 実装

MossFormerGAN の更新規則は変えていない。変えたのは、複数プロセスの checkpoint が各 rank の学習 RNG を残すことだけである。本格の 121 epoch は始めていない。

## 修正内容

`capture_rank_rng` は、そのプロセスの Python、NumPy、Torch CPU と、`torch.cuda.current_device()` の生成器だけを取る。`get_rng_state_all()` は使わない。

保存のとき、全 rank が `all_gather_object` に入り、`rng_by_rank` を書く。rank 0 だけがファイルを書く。保存するかどうかは rank 0 の判断を `broadcast` し、全 rank が同じ collective に入る。world size 1 にはこの collective は無い。

従来の `rng` も rank 0 が引き続き書く。これはそのプロセスの全可視 device であり、複数プロセスの再開には使わない。

再開は次のとおりである。

- world size 1 は従来の `rng` を `restore_rng` で戻す。
- world size が 2 以上で `rng_by_rank` の長さが world size と違うとき、または中身が欠けるときは `TrainingError` にする。rank 0 の device 一覧を他 rank へコピーしない。
- 戻す CUDA device の番号が、今の `current_device` と違うときも拒否する。
- 復元前に RNG tensor を CPU へ移す。

保存の round-trip では、`rng_by_rank` の Torch CPU と CUDA byte tensor が一致しなければ完成 checkpoint にしない。

BatchNorm、Optimizer、蓄積勾配の保存形式は変えていない。診断では、それらは rank 間で一致し、読み込み直後も保存値と一致した。

## 変更ファイル

- `src/training/core/rng.py`
- `src/training/core/checkpoint.py`
- `src/training/core/trainer.py`
- `tests/training/test_rank_rng.py`
- `tests/training/p10c_fix2b1_probe.py`
- `tests/training/p10c_fix2b1_gpu.py`
- `tests/training/p10c_fix2b_gpu.py`
- `.agents/DECISIONS.md`
- `.agents/DATA.md`
- `.agents/DEVELOPMENT.md`

`p10c_fix2b_gpu.py` に足したのは、比較用の `rng_by_rank` と、最初の Generator forward のハッシュと、rank 間の Adam moment 差である。更新規則は変えていない。Adam の step が CPU にあるときは、比較の broadcast の前だけ CUDA へ移す。

公式ソース、Dataset、Snapshot、DB、既存 checkpoint、本格 YAML は変えていない。

## Checkpoint 形式への影響

形式名は `audio-db-training-v1` のままである。足したキーは `rng_by_rank` だけである。

- world size 1 の古い checkpoint は `rng` だけで再開できる。
- world size 1 の新しい checkpoint も、再開に使うのは `rng` である。
- world size が 2 以上の古い checkpoint に `rng_by_rank` が無いときは再開しない。黙って rank 0 の CUDA RNG を他 rank へは書かない。
- 保存はこれまでどおり一時ファイルへ書いてから rename する。検証に失敗した一時ファイルは完成 checkpoint にならない。
- Dataset identity と resume の設定照合は変えていない。

`rng` の CUDA 一覧は、再開後の未使用 device が新しいプロセスの初期状態のままなので、連続実行の `rng` とバイト一致しないことがある。学習が使うのは `rng_by_rank` の現在 device である。

## 他モデルへの影響

保存と再開は共通 Trainer である。更新式、損失、Sampler は変えていない。world size 1 の再開経路は従来の `rng` のままである。2 GPU 以上で古い checkpoint を再開するモデルは、`rng_by_rank` が無い限り拒否される。これは MossFormerGAN だけの分岐ではない。

保存判断の broadcast は、複数プロセスの更新 1 回につき 1 回である。RNG の all_gather は、実際に checkpoint を書くときだけである。勾配の平均には入らない。

## 回帰テスト

`.venvs/se`、CPU。

| テスト | 結果 |
|---|---|
| `tests.training.test_rank_rng` | 5 PASS |
| `tests.training.test_rng_streams` | PASS（上の一括 49 件に含む） |
| `tests.training.test_checkpoint_atomic` | PASS |
| `tests.training.test_checkpoint_retention` | PASS |
| `tests.training.test_resume_identity` | PASS |
| `tests.training.test_mock_trainer` | PASS |
| `tests.training.test_mossformer_update_rule` | PASS |
| `tests.training.test_mossformer_pesq_skip` | PASS |
| `tests.training.test_mossformer_syncbn_collective` | PASS |
| `tests.training.test_mossformer_sampler` | PASS |
| `tests.training.test_mossformer_epoch_plateau` | PASS |
| `tests.training.test_validation_interrupt` | PASS |
| `tests.training.test_p5_chunk.DistributedChunkTests` | PASS |

一括 49 件は OK だった。`test_rank0_only_syncbn_forward_does_not_finish` の gloo timeout は、片 rank だけが collective に入らないことを見る既存の失敗経路である。FAIL には数えていない。

他モデルの全件回帰はしていない。
