# Phase 2-B CPU / Gloo DDP

CPU / Gloo、world size 2 の DDP 学習、rank 間の状態一致、単一ファイル checkpoint、別プロセス群からの再開を実装した。この文書の測定に CUDA、NCCL、AMP は含まれない。それらは [PHASE_2C_CUDA_AMP.md](PHASE_2C_CUDA_AMP.md)。FSDP と sharded checkpoint は未実装である。

## 起動

この文書の測定は `device: cpu` と `precision: fp32` である。world size 1 は process group を作らない。world size 2 以上は `torchrun` が置く `RANK`、`LOCAL_RANK`、`WORLD_SIZE` で Gloo を初期化する。コンストラクタの `world_size` を書いたときは、その値と環境変数が一致しなければならない。欠けている、または食い違うときは `PrepareError` で、単一プロセスには落とさない。`timeout_seconds` は process group の timeout で、既定は 120 秒である。CUDA と AMP の起動は [PHASE_2C_CUDA_AMP.md](PHASE_2C_CUDA_AMP.md)。

world size 1 の `distribute: ddp` は包まない。world size 2 以上で `trainable: true` かつ `distribute` が `ddp` でない module は `PrepareError` になる。`distribute: fsdp` は拒否する。`sync_batchnorm: true` は CUDA のときだけ `SyncBatchNorm` にするので、この CPU runtime は BatchNorm を変換しない。

## Lifecycle と復元

```text
ProcessRuntime を作る（world size 2 なら Gloo）
  rank 0 が output/.lock を非ブロッキングで取る
  他 rank はその成否を gather する
build
  module を構築する
  prepare_model が DDP で包む
  Optimizer は wrap 後の Parameter を受け取る
Experiment.run(resume=...)
  全 rank が同じファイルを読む
  unwrapped の state_module へ state_dict を載せる
  Optimizer、Scheduler、Trainer、Data、runtime を載せる
  rng_by_rank[rank] を戻す
```

CPU の DDP は Parameter オブジェクトを置き換えない。テストは、wrapper の parameter、`state_module` の parameter、Optimizer の parameter が同一オブジェクトであることを確認した。`state_dict` のキーに `module.` は付かない。全 rank が同じ checkpoint をその場へ載せるので、wrap 後の復元でも次の forward、backward、Optimizer 更新は連続実行と一致する。FSDP はこの前提を使えない。

## Directory lock と出力

rank 0 だけが `directory/.lock` の排他 lock を取る。他 rank は同じ排他 lock を取らない。rank 0 の失敗は gather で全 rank に渡り、そのディレクトリの `status.json` は書かない。第二の起動は `experiment is already running` で終わる。

`status.json`、`complete.json`、`metrics.jsonl`、checkpoint の完成ファイルは rank 0 が書く。world size 2 では各 rank が自分の `samples-rank{rank}.jsonl` だけを書く。再開時、その rank の logical step より大きい行を落とす。`metrics.jsonl` の切り捨ては rank 0 だけが行う。

## Data

`TensorPairData.batch_size` は 1 rank の 1 micro-batch である。global micro-batch は `batch_size * world_size`。update `u` の rank `r` は、添字 `u * global_batch_size + r * batch_size` から `batch_size` 件を取る。更新回数は `examples // global_batch_size` で、全 rank で同じになる。余りは捨てる。`plan_hash` はテンソル、`batch_size`、`world_size` から作り、rank は入れない。state にも rank は入れない。Sampler はもう一度掛けない。

## 勾配

`StandardTrainer` は最後の micro-batch だけ `sync=True` にする。`no_sync` はその micro-batch の forward と backward を同じスコープで包む。損失は蓄積回数で割ってから backward する。DDP は同期する backward で、それまでに rank 内へ溜めた勾配を rank 間平均する。この平均と、損失を蓄積回数で割る操作は別である。

`runtime.step` は全 rank が `agree(True, policy="identical")` に入ってから `optimizer.step` する。flag が rank 間で違う `agree` は `RuntimeFailure` になる。

## Checkpoint

形式は `audio-db-experiment-v1`、`schema_version` 1 のままである。保存の順は次である。

1. 全 rank が checkpoint 境界にいる。
2. parameter、buffer、optimizer、scheduler、trainer、data の dtype、shape、値を集めて比べる。
3. 各 rank の RNG を集め、blob にその rank 番号を付ける。
4. 一致したときだけ rank 0 が一時ファイルへ書き、読み戻し、`os.replace` する。
5. 例外は gather し、失敗なら完成ファイルを出さない。
6. 成功後に barrier する。

`rng_by_rank[i]["rank"]` が i でなければ `CheckpointError` になる。長さが world size と違うファイル、world size が違うファイルも拒否する。rank 0 の RNG を rank 1 の枠へコピーしたファイルは拒否する。

## 測定

コマンドは `CUDA_VISIBLE_DEVICES="" .venvs/se/bin/python -m unittest discover -s tests/experiments -v`。GPU は使っていない。2026-11-08 に 43 件が 106.467 秒で OK。代表 CPU 回帰 `tests.training.test_rng_streams`、`test_batch`、`test_accumulation`、`test_protocol` は 16 件が 26.249 秒で OK。

### 勾配同期

seed 17、`torch.nn.Linear(4, 4)`、SGD（lr 0.1、momentum 0、weight decay 0）、MSE、`TensorPairData`（examples 16、batch 2、world size 2、accumulation 1）、2 logical step。

DDP の checkpoint と、rank ごとの勾配を足して 2 で割った単一プロセス参照の最大絶対誤差は 0。両 rank は正常終了し、optimizer update count は 2 で一致した。`metrics.jsonl` は 2 行で、rank 0 だけが書いた。`samples-rank0.jsonl` と `samples-rank1.jsonl` の sample id は交わりが空である。

### 等価な global batch

global micro-batch 4 を二つ比べた。どちらも参照の最大絶対誤差は 0。

| 実行 | batch_size | accumulation | world size | 参照との最大絶対誤差 |
|---|---:|---:|---:|---:|
| wide | 2 | 1 | 2 | 0 |
| narrow | 1 | 2 | 2 | 0 |

二つの実行同士の最大絶対誤差は `7.450580596923828e-09` である。同じ 4 サンプルを、rank 内の mean と micro-batch の加算順が違う経路で足している。この差を許容誤差にして成功にはしていない。成功条件は、各実行が自分の加算順の参照と最大絶対誤差 0 で一致することである。

### 中断再開

seed 17、`DropoutLinear`（dropout 0.5、buffer `marker`）、AdamW lr 0.001、StepLR、examples 32、batch 4、world size 2、accumulation 1。

連続 4 logical step と、2 step で `INTERRUPTED` にしたあと新しい 2 プロセスから resume して 4 step にした結果を比べた。

| 項目 | 最大絶対誤差または一致 |
|---|---|
| module parameter と buffer | 0 |
| optimizer state | 0 |
| scheduler state | 0 |
| trainer state | 0 |
| data state | 0 |
| logical step | どちらも 4 |
| optimizer update count | どちらも 4 |
| `rng_by_rank` | 一致 |
| rank 0 と rank 1 の sample id 列 | 一致 |
| rank 0 と rank 1 の損失列 | 一致 |

再開前に logical step 3 の行を `metrics.jsonl` と両 rank の sample ファイルへ足した。再開後のファイルにその行は残っていない。

world size 1 からの resume は `CheckpointError`。`rng_by_rank` の長さを 1 にしたファイルも `CheckpointError`。rank 0 の RNG を rank 1 へコピーしたファイルは `belongs to rank 0` で拒否した。

### 異常系

rank 1 の weight だけを保存前に +1 した実行と、buffer `marker` だけを +1 した実行は、どちらも `ranks disagree` で終了した。`step_00000002.pt` はできず、`step_00000001.pt` の SHA-256 は保存前に rank 0 が記録した値と一致した。

skip 判断を rank 間でずらした `agree` は、30 秒の subprocess timeout より前に `ranks disagree` で終了した。既に lock を持ったディレクトリへの第二の world size 2 起動は `already running` で終わり、`status.json` を書かなかった。

## 残るもの

CUDA / NCCL と AMP FP16 は Phase 2-C で実装した。残るのは FSDP、sharded checkpoint、AMP BF16、GAN の学習処理、SE モデル、公式 Dataset、動的 Expert である。process group の初期化中に rank が死んだ場合は、設定した timeout で終わる。それより強いハング回避は検証していない。
