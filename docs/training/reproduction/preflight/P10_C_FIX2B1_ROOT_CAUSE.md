# P10-C Fix.2-B.1 根本原因

Fix.2-B の 2 GPU 再開で残った浮動小数差を、同じ YAML・seed・Dataset・world size 2 のまま切り分けた。本格学習は始めていない。`deterministic: false` だけを理由にはしていない。

開始時の `HEAD` は `ff5a5abfe0d2fbe1920c0304cd285ba853cbc8ce`、ブランチは `feature/reproduction`。作業木は修正前は clean だった。PyTorch は `2.6.0+cu124`、CUDA は 12.4、cuDNN は 90100。診断 GPU は物理 0 と 6（`CUDA_VISIBLE_DEVICES=0,6`）。`CUDA_LAUNCH_BLOCKING=1`。

## 原因候補

1. rank 0 だけが checkpoint を書き、`torch.cuda.get_rng_state_all()` を全 rank が復元している。
2. Discriminator の SyncBatchNorm buffer が rank ごとに違い、rank 0 の buffer を全員へ戻すのが誤りである。
3. Optimizer の moment または蓄積勾配の保存・復元が欠ける。
4. DDP の `broadcast_buffers=True` が再開後の最初の forward で buffer を壊す。
5. `deterministic: false` の cuDNN が、再開に固有でない実行差を出す。

## 検証方法

比較の前に許容を固定した。離散状態は完全一致。パラメータ atol `1e-3`、Adam moment atol `1e-3`、BatchNorm 浮動 buffer atol `1e-4`、蓄積勾配 atol `1e-4`。復元直後の tensor は atol 0。

まず修正前の実装で、Dropout だけの 2 GPU と、公式幅 MossFormerGAN の 1 更新を走らせ、rank ごとの CUDA RNG をハッシュした。保存直前、読み込み直後、DDP wrap 後を分けた。BatchNorm 8 個の `running_mean`、`running_var`、`num_batches_tracked`、`weight`、`bias` も同じ時点で比べた。

## 比較した状態

Dropout だけの実験では、両 rank の現在 device の CUDA RNG は一致した。rank 0 の `get_rng_state_all()[0]` はその値と一致し、`[1]` は seed 直後のまま一致しなかった。

公式幅の 1 更新でも同じだった。

| 時点 | rank 0 現在 device | rank 1 現在 device | rank 間の BN |
|---|---|---|---|
| DDP wrap 直後 | seed 直後 | seed 直後 | 一致 |
| 第 1 Generator forward 直後 | 消費後 | 同じ消費後 | 一致 |
| Optimizer 直後 | 消費後 | 同じ消費後 | 一致 |
| checkpoint の `rng.cuda` | `[消費後, seed直後]` | 保存されない | rank 0 の buffer |

Python、NumPy、Torch CPU は両 rank で一致した。Optimizer 更新後のパラメータと BN buffer も両 rank で一致した。`num_batches_tracked` も一致した。

別プロセスが同じ checkpoint を読むと、rank 0 の現在 device は消費後の状態に戻った。rank 1 の現在 device は checkpoint の slot 1、つまり seed 直後に戻った。DDP wrap はこの RNG を変えなかった。読み込んだ重みと BN buffer は、保存直前の rank 0 と一致した。

## 最初に差が生じた箇所

差は checkpoint の読み込み直後、rank 1 の CUDA RNG に出る。学習中の両 rank の現在 device は同じ列を消費している。保存されるのは rank 0 のプロセスが見ている device 一覧である。rank 0 は `cuda:0` だけを消費し、`cuda:1` は seed のまま残る。`set_rng_state_all` は rank 1 の学習 device である `cuda:1` に、その未使用 slot を書く。

Generator の Dropout は現在 device の生成器を使う。再開後の rank 1 は、連続実行なら既に消費済みの Dropout 列を先頭から引き直す。活性化が変わり、SyncBatchNorm の集計と勾配が変わる。Fix.2-B で `cnn.bn6.running_var` が `1.45e-2` まで開いたのは、この後段である。

## 確定した原因

複数プロセスの checkpoint が、rank 0 の `get_rng_state_all()` を全 rank に復元していた。未使用 device の slot が、他 rank の現在 device を seed 直後へ戻す。

## 棄却した仮説

- BN buffer の保存不備。保存直前は 8 個とも rank 間で一致し、読み込み直後も保存値と一致した。`broadcast_buffers=True` のあとでも、測定した forward では rank 間の buffer は割れなかった。rank 0 の buffer を全員へ戻してよい。
- Optimizer と蓄積勾配の欠落。読み込み直後の Generator、Discriminator、Adam moment、勾配は保存値と最大絶対差 0 だった。空の窓も、後の `accu_count=1` の再開も、別の欠落では説明できない。
- 再開プロセスだけが別の case 列や step を進める。`global_step`、`official_step`、plan hash、case id は一致する。
- Python / NumPy / Torch CPU を rank 0 から rank 1 へ写すこと。この実行ではそれらは rank 間で一致していた。壊れていたのは CUDA の device 一覧である。

## 未解決事項

`torch.use_deterministic_algorithms(True)` は、最初の backward で `adaptive_max_pool2d_backward_cuda` に決定的実装が無いとして停止した。呼び出しは Discriminator の NISQA CNN、`F.adaptive_max_pool2d`（出力 `101x7`、`50x7`、`25x5`、`12x5`、`6x3`）である。`warn_only=True` にはしていない。公式の pooling は変えていない。本格 YAML の `deterministic: false` も変えていない。

この停止のため、deterministic true の連続実行と再開の数値表は無い。`deterministic: false` でも、同じ checkpoint から再開した最初の Generator forward の SHA-256 は両 rank で連続実行と一致した。残る重み差は、その一致した forward のあとで cuDNN が積む実行差である。

CUDA illegal memory access は、今回の既知系列でも再現しなかった。原因は不明のままである。
