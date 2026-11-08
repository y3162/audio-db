# Phase 2-A checkpoint

CPU / FP32、world size 1 の保存と再開だけを、この文書の測定範囲にした。DDP は [PHASE_2B_DDP.md](PHASE_2B_DDP.md)、CUDA と AMP は [PHASE_2C_CUDA_AMP.md](PHASE_2C_CUDA_AMP.md)。この文書の数値はそれらの成功ではない。

## 復元順

Phase 0 は、resume の module 重みを `prepare_model` の前に載せる。Phase 1 の `build_experiment(config)` は resume パスを受け取らず、Prepare と Optimizer 生成まで終える。この公開 API を維持するため、Phase 2-A の CPU 経路は次の順にする。

```text
build_experiment
  module を構築する
  Runtime.prepare_model（CPU。パラメータオブジェクトは置換しない）
  Optimizer をそのパラメータで構築する
  Trainer を構築する
Experiment.run(resume=...)
  module.state_dict をその場へ載せる
  Optimizer、Scheduler、Trainer、Data、runtime を載せる
  rng_by_rank[0] を戻す
  Trainer を復元した logical step から続ける
```

CPU の `.to("cpu")` はパラメータの同一性を保つ。`load_state_dict` は同じ Parameter を更新するので、既に構築済みの Optimizer はその Parameter を参照したままになる。復元に失敗した実行は Trainer を呼ばない。

後続の戦略はこの順を使わない。

| 戦略 | 復元順 |
|---|---|
| CPU / FP32（Phase 2-A） | Prepare と Optimizer 生成の後、同じ Parameter へ載せる |
| CPU / Gloo DDP（Phase 2-B） | Prepare で wrap し、Optimizer を作ったあと、unwrapped の `state_module` へ全 rank が同じファイルを載せる。DDP は Parameter を置き換えない。記録は [PHASE_2B_DDP.md](PHASE_2B_DDP.md) |
| CUDA / FP32 と CUDA / AMP FP16（Phase 2-C） | 単一 GPU の DDP も Parameter を置き換えないので、同じ順で unwrapped の `state_module` へ載せる。記録は [PHASE_2C_CUDA_AMP.md](PHASE_2C_CUDA_AMP.md) |
| FSDP | prepare がパラメータを置き換えた後、FSDP の state API で載せる。Optimizer はその置換後のパラメータで作る。version 1 の単一ファイルは shard ではないので、FSDP へ resume しない |

## 形式

ファイルは `torch.save` の 1 辞書で、`format` は `audio-db-experiment-v1`、`schema_version` は 1 である。一時ファイルへ書き、読み戻して `format`、`logical_step`、`state_ids`、`config_sha256` を確認し、`fsync` したあと `os.replace` する。名前が `.` で始まるファイルは復元しない。`audio-db-training-v1` は拒否する。

保存対象は module、optimizer、scheduler、trainer、data、runtime と、長さ 1 の `rng_by_rank` である。loss と別名は保存しない。`config_sha256` は `output` と trainer の `stop_after_logical_steps` を除く。別ディレクトリへの resume は、ハッシュではなく保存した出力パスで拒否する。`stop_after_logical_steps` へ到達した実行は `INTERRUPTED` であり、`complete.json` を書かない。`max_logical_steps` まで終わった実行は `COMPLETED` と `complete.json` を書き、そのディレクトリの resume を拒否する。

`initialize_from` は `label: init` かつ `initialize.allow` があるファイルだけを受ける。許可した module の重みだけを載せ、Optimizer、logical step、データ位置、RNG は新しい実験のままである。

## 測定

別プロセスで、Dropout を含む CPU 実験を 4 logical step 連続実行したものと、2 step で `INTERRUPTED` にしたあと再開して 4 まで進めたものを比較した。重み、バッファ、Optimizer、Scheduler、Trainer、Data、`rng_by_rank` は一致した。最大絶対誤差は 0。

| 項目 | 値 |
|---|---|
| logical step | 4 |
| Optimizer の step | 4 |
| Scheduler の `last_epoch` | 4 |
| data の `update_index` | 4 |
| 損失 | 1.8860440254211426、1.136971354484558、2.87058162689209、0.992588222026825 |
| sample id | `ex-0` から `ex-3`、`ex-4` から `ex-7`、`ex-8` から `ex-11`、`ex-12` から `ex-15` |

再開前に足した logical step 3 の余分な `metrics.jsonl` 行は破棄された。先に生成した index 2 の batch は保存位置に含まれず、再開後の次の sample id は `ex-8,ex-9,ex-10,ex-11` だった。

試験は `tests/experiments/test_checkpoint.py`。コマンドと件数は完了報告に書く。
