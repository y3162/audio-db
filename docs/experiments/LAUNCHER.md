# Launcher の仕様

`python -m src.experiments.launcher` は、1 つの実験を起動し、プロセスを監視し、終わり方を記録する。単一 GPU と DDP、研究室の GPU サーバー、ローカル、TSUBAME 4.0 の `qsub` / `qrsh` で同じコマンドを使う。

Launcher がしないこと: ジョブの投入、GPU の選択、再試行、再投入。GPU は呼び出し側が `CUDA_VISIBLE_DEVICES` で決める。失敗した実験は、利用者が次の起動で `--resume` を付けて続ける。

使い方は [学習・実験](RUN.md)、checkpoint の保証は [Checkpoint](CHECKPOINT.md)。

## コマンド

```bash
.venvs/se/bin/python -m src.experiments.launcher --config CONFIG \
  [--resume CHECKPOINT_OR_DIRECTORY] [--nproc N] [--stop-after SECONDS] [--grace SECONDS]
```

| 引数 | 意味 |
|---|---|
| `--nproc N` | 学習プロセスの数。1（既定）は `python -m src.experiments.cli` を 1 つ起動する。2 以上は `torchrun --standalone --max-restarts 0 --nproc-per-node N` で起動する |
| `--resume` | 続ける checkpoint、または出力ディレクトリ（その `latest.pt`） |
| `--stop-after` | 起動からこの秒数で安全な停止を要求する。ジョブの時間制限より前に止めるために使う |
| `--grace` | 停止の要求から終了までに待つ秒数（既定 300）。過ぎたら全プロセスを kill する |
| `--gpu-memory-interval` | `nvidia-smi` でデバイス全体の使用メモリを取る間隔（秒）。既定は 10。環境変数 `AUDIO_DB_GPU_MEMORY_INTERVAL` でも指定でき、コマンドの値が優先する。0 で記録しない |

`scripts/experiments/job.sh CONFIG [NPROC]` は、同じ Launcher を `exec` する薄いシェルである。環境変数 `AUDIO_DB_RESUME`、`AUDIO_DB_STOP_AFTER`、`AUDIO_DB_GRACE`、`AUDIO_DB_PYTHON` を引数に写す。`AUDIO_DB_GPU_MEMORY_INTERVAL` はそのまま Launcher に渡る。`make experiment` も Launcher を呼ぶ。

## 終了状態

学習プロセスは結果を報告するだけで、判定は Launcher が全プロセスの終了後に書く。

| `status.json` | 終了コード | 条件 |
|---|---:|---|
| `COMPLETED` | 0 | 全プロセスが終了コード 0 で終わり、Trainer が完了を報告した。このときだけ `complete.json` を書く |
| `INTERRUPTED` | 3 | 全プロセスが終了コード 0 で終わり、Trainer が安全な停止を報告した。`--resume` で続けられる |
| `FAILED` | 1 | それ以外。例外、CUDA OOM、NCCL のエラー、rank の kill、猶予切れ、結果の報告なしの終了 |
| （変更なし） | 1 | 起動を拒否した。別の Launcher または学習がそのディレクトリを使っている、設定が読めない、など |
| | 2 | 引数の誤り |

- 一部の rank だけが学習を続けることはない。torchrun は 1 つの rank が異常終了すると残りを終了させる。それでも残った rank は、Launcher がプロセスグループごと kill する。Launcher 自身が死んだときは、子に SIGTERM が届く（Linux の `PR_SET_PDEATHSIG`）。
- `--max-restarts 0` なので、torchrun は自動で再起動しない。
- 失敗した実験の `latest.pt` は、最後に検証して置き換えた checkpoint のままである。次の起動は `--resume` でそこから続く。
- `status.json` を書き換えるのは、この起動の学習が始まったあとだけである。拒否された起動は、以前の実行の記録を変えない。

## 停止

SIGTERM、SIGINT、SIGHUP、`--stop-after` は、Trainer に次の安全な点での停止を求める。Trainer はそこで `stop` の保存点を差し出し、Selector の決定に従って保存してから終了コード 0 で終わる。

- 起動中に届いた要求は、学習が始まってから rank へ渡す。まだ受け付けていない rank を kill しないためである。
- DDP では、Launcher が rank へ直接 SIGTERM を送る。torchrun に送ると、torchrun が rank を即座に終了させるためである。
- 2 回目のシグナル、または `--grace` 秒の経過で、全プロセスを kill する。これは `FAILED` である。

安全な停止（`INTERRUPTED`）と異常終了（`FAILED`）は、`status.json` と終了コードで区別できる。

## 記録

出力ディレクトリに次を書く。

| ファイル | 内容 |
|---|---|
| `launch.json` | `audio-db-launch-v1`。起動 ID、ホスト、pid、プロセス開始時刻、コマンド、world size、開始と終了の時刻、15 秒ごとの heartbeat、終了コード、理由 |
| `status.json` | 判定。`FAILED` には `reason`、`exit_code`、各 rank の `failures` が入る |
| `failure.rank<N>.json` | rank N が投げた例外。`kind` は `cuda_oom`、`nccl`、`interrupt`、`exception` |
| `.launch.lock` | Launcher の排他。同じディレクトリを 2 つの Launcher が監視しない |
| `gpu_memory.jsonl` | デバイス全体の GPU メモリの時系列。1 行が 1 つの JSON |
| `gpu_memory.json` | その時系列のサンプル数と、GPU ごとの最大 `memory_used_mib` |

シグナルで kill された rank は、自分の記録を残せない。その場合の手掛かりは `status.json` の `reason`（`terminated by SIGKILL` など）とジョブのログである。

## GPU メモリ

Launcher は、この起動の `status.json` が `RUNNING` になってから、`nvidia-smi` でデバイス全体の使用メモリを取る。学習プロセスの中では取らない。取得のスレッドが止まっても、学習の終了判定は変わらない。

| 項目 | 内容 |
|---|---|
| `memory_used_mib` | `nvidia-smi` の `memory.used`。その GPU 全体の使用量（MiB）で、他のプロセスの分を含む。`torch.cuda.memory_allocated` ではない |
| `memory_total_mib` | `nvidia-smi` の `memory.total`（MiB） |
| 対象 | `CUDA_VISIBLE_DEVICES` に書いた index または UUID。未設定のときはホスト上の全 GPU（`scope` は `host`）。空文字のときは CPU 実行として記録しない |
| 時刻 | 各サンプルの `recorded_at`（UTC）。同じ行の `devices` がその時刻の値 |
| 間隔 | `--gpu-memory-interval` または `AUDIO_DB_GPU_MEMORY_INTERVAL`。既定 10 秒。0 で記録しない |
| 最大 | `gpu_memory.json` の `peaks`。`scripts/experiments/show_status.py` もこれを出す |

`gpu_memory.jsonl` の先頭行は `kind: header` で、指標の意味と `launch_id` が入る。続く行は `kind: sample` である。サンプルのたびに行を追記して fsync し、`gpu_memory.json` を置き換える。Launcher が kill されても、書き終わった行は残る。`nvidia-smi` が無い、失敗した、対象の GPU が無い、というときは `error` を記録して学習は続ける。

`python -m src.experiments.cli` を直接起動した実行には、この時系列は付かない。PyTorch の allocated / reserved のピークと処理時間は、`AUDIO_DB_RESOURCE_PROFILE=1` の `resource_profile.json` である。これは CUDA を同期するため既定では書かない。GPU メモリの時系列とは別の記録である。

## `RUNNING` が生きているか

プロセスが kill されると `status.json` に `RUNNING` が残る。これを実行中と見なさないために、記録にはホスト、pid、プロセス開始時刻を入れている。

```bash
.venvs/se/bin/python scripts/experiments/show_status.py OUTPUT_DIRECTORY
```

| `liveness` | 意味 |
|---|---|
| `alive` | 記録の pid と開始時刻のプロセスがこのホストにある。別ホストの起動は、heartbeat が 120 秒以内 |
| `stale` | `RUNNING` だが、そのプロセスは無い、または heartbeat が古い。実行は記録を残さずに終わっている |
| `finished` | `RUNNING` 以外 |
| `absent` | `status.json` が無い、または読めない |
| `unknown` | Launcher なしで別ホストから起動されていて、ここからは確認できない |

`stale` のディレクトリは、Launcher で `--resume` すれば続けられる。排他は `flock` なので、プロセスが死ねば解放される。

## 検出できないこと

- NCCL の通信が止まり、どの rank も終了しない状態は、Launcher からは実行中に見える。上限は `--stop-after` と `--grace`、またはスケジューラの時間制限で決まる。Launcher は学習の進み具合を監視しない。
- 複数ノードの DDP は扱わない（`--standalone`）。
- 電源断やストレージ障害のあとの整合性は保証しない。
- TSUBAME の実ジョブで、`qdel` と `h_rt` 切れが送るシグナルに対する動作は未確認である。確認した範囲は [テスト](../TESTING.md)。
