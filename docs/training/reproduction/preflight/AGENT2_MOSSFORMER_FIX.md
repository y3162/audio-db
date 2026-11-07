# Agent 2: MossFormerGAN の初期化、PESQ スキップ、SyncBN

対象は `src/training/se/mossformergan.py` だけである。Trainer、設定、公式ソースは変えていない。学習ジョブは起動していない。測定は CPU、`CUDA_VISIBLE_DEVICES` を空にした `.venvs/se/bin/python` である。GPU の数値は **NOT_TESTED**。

## 公式が残す重み

`train.py` の `main` は seed のあと `network_wrapper` を 2 回呼ぶ。

```29:33:src/se/official/clearervoice-studio/train/speech_enhancement/train.py
    model = network_wrapper(args).se_network
    model = model.to(device)
    if args.network=='MossFormerGAN_SE_16K':
        discriminator = network_wrapper(args).discriminator
        discriminator = discriminator.to(device)
```

ラッパー 1 回は `MossFormerGAN_SE_16K` を 2 個作る。

```15:18:src/se/official/clearervoice-studio/train/speech_enhancement/networks.py
        elif args.network == 'MossFormerGAN_SE_16K':
            from models.mossformer_gan.generator import MossFormerGAN_SE_16K
            self.se_network = MossFormerGAN_SE_16K(args).model
            self.discriminator = MossFormerGAN_SE_16K(args).discriminator
```

`mode == 'train'` の各モジュールは SyncANet のあとに Discriminator を作る。

```21:27:src/se/official/clearervoice-studio/train/speech_enhancement/models/mossformer_gan/generator.py
    def __init__(self, args):
        super(MossFormerGAN_SE_16K, self).__init__()
        self.model = SyncANet(num_channel=64, num_features=args.fft_len//2+1)
        if args.mode == 'train':
            self.discriminator = Discriminator(ndf=16)
        else:
            self.discriminator = None
```

消費順は G1 D1 G2 D2 G3 D3 G4 D4 である。学習に残るのは 1 個目の `.se_network`（G1）と、2 個目のラッパーの `.discriminator`（D4）である。G1 を device へ移すのは 2 個目のラッパーの前（`train.py` 30 行）である。

audio-db は同じ seed ブロックの中で、`build_model("mossformergan", ...)` と `Discriminator(ndf=16)` をこの順に呼ぶ。捨てる 6 個もコンストラクタを最後まで実行してから参照を落とす。checksum `7729.488281` は期待値にしていない。

## 測った checksum

seed 777、`fft_len=400`、`mode='train'`、`torch.manual_seed`。公式 import の前後で CPU RNG は変わらなかった（`official_import_changed_rng: false`）。パラメータの float64 総和は次のとおり。

| 重み | float64 総和 | 公式との state_dict 最大絶対差 |
|---|---|---|
| 残した Generator（G1） | 114801.78831011125 | 0 |
| 残した Discriminator（D4） | 5798.916322645511 | 0 |
| seed をやり直した 1 回目だけの Discriminator | 5459.690748355152 | 残した D とは `conv1` が不一致 |

合否は総和ではなく、残した G と D の state_dict 全テンソルの一致である。1 回目だけの Discriminator は残した D と違う。`7729.488281` はこの測定の目標にしていない。

## 学習の PESQ / 非有限

公式の損失は NaN か PESQ 欠測で `None, None` を返す。

```58:75:src/se/official/clearervoice-studio/train/speech_enhancement/losses/loss.py
    if torch.isnan(loss):
        print('train loss is nan, skip this batch!')
        return None, None
    ...
    if pesq_score is not None:
        ...
    else:
        discrim_loss_metric = torch.tensor([0.])
        print('train pesq score is None, skip this batch!')
        return None, None
```

`Solver` は `loss_gen is not None` の中だけ backward と `optimizer.step` をする（`solver.py` 277–292 行、`accu_grad` 分岐）。失敗してもループは続く。`accu_count` は成功したバッチだけ増えるので、公式の 1 GPU は後続の成功バッチと蓄積を続ける。

audio-db の `train_update` は固定のマイクロバッチ群を 1 回受け取る。どれか 1 ランクでもスキップなら、その群全体で backward、clip、両方の `optimizer.step` をしない。同じバッチはやり直さない。これは DDP でランクの枝を割らないための差であり、公式の「後続バッチへ蓄積を伸ばす」動作ではない。

スキップ判定は backward の前に行う。各ランクの局所フラグを `all_reduce` の MAX でまとめる。1 つでも真なら全ランクが同じ枝に入る。forward は `no_sync` の中で行うので、backward しないランクが DDP の allreduce を待たせない。判定より前に行った Discriminator forward の回数は全ランクで同じである。成功した群だけ、そのあと fake / real の forward と backward を行う。

戻り値は `dict[str, float]` のままである。

- `update_applied`: 両方の optimizer が step したとき 1.0、群をスキップしたとき 0.0
- スキップ時の `generator` と `discriminator` は 0.0（有限）。`Trainer._require_finite` はここを見て例外にしない
- **Trainer は `update_applied == 0` のとき `global_step` を増やさない。** この担当は Trainer を編集していない。接続は `Trainer._accept_optimizer_update` が行い、偽のときは `continue` して `global_step += 1` に入らない。スキップは optimizer 更新として数えず、再開でその更新を二重実行しない。

`self.last_skip` は直近のスキップ辞書、`self.skip_log` はその追記リストである。中身は `reason`、`case_ids`（バッチにあれば）、`micro_batch_index`、`count`（その群のマイクロバッチ数）、`events` である。`reason` は `pesq`、`non-finite generator loss`、`non-finite discriminator loss`、他ランクだけが落としたときの `peer` である。ログは `mossformer skip reason=... case_ids=... micro_batch_index=... count=...` である。

非有限の Discriminator 損失も群ごとスキップする。公式の `torch.isnan` は Generator 損失だけを見ている。片ランクだけが例外で抜けると、もう片方が collective で待つので、ここは全ランクで揃える。

成功した群の clip 順は変えていない（P10P-D-001 はそのまま）。最後のマイクロバッチは backward のあと、clip の前に勾配を SUM して `world_size` で割る。それより前のマイクロバッチは、ランク間平均の前に局所勾配を clip する。Generator の平均は、forward を `no_sync` にしたため DDP の backward 内ではなく手動の `all_reduce` である。clip と平均の前後は以前と同じである。

## 検証の分母

検証の足し算は `state=='train'` の外にある。

```326:329:src/se/official/clearervoice-studio/train/speech_enhancement/solver.py
            if loss_gen is not None:
                total_loss += loss_gen.data.cpu()

        return total_loss / (i+1)
```

`loss_gen is None` のとき公式は加算しない。分母は常に `(i+1)` なので、そのバッチは平均に残る。`+= 0` とは書かれていない。audio-db の `validation_report` は、この商と Trainer の平均（status `ok` の値だけを、件数で割る）が合うように、PESQ 欠測のバッチを値 0・status `ok` で返す。欠測を status `na` にすると分母から落ちるので、ここは変えていない。PESQ 欠測は `TrainingError` にしない。

検証の非有限 Generator 損失は、従来どおり `_finite` が `TrainingError` にする。公式は NaN でも `None` を返して分母に残す。この差は今回直していない。

検証で `eval()` になるのは Generator だけである。`solver.py` 197 行は `self.model.eval()` だけを呼ぶ。`loss.py` に `discriminator.train()` は無い。Discriminator は構築時の train のままなので、現行の `eval_mode` / `validation_report` も Discriminator を `train()` のままにしている。

## SyncBN

公式は分散時に Generator と Discriminator の両方を `SyncBatchNorm.convert_sync_batchnorm` してから DDP する。`find_unused_parameters=True`。`broadcast_buffers` は渡していない（既定 True）。

```29:34:src/se/official/clearervoice-studio/train/speech_enhancement/solver.py
        if self.args.distributed:
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = DDP(self.model, device_ids=[self.args.local_rank],find_unused_parameters=True)
            if self.discriminator is not None:
                self.discriminator = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.discriminator)
                self.discriminator = DDP(self.discriminator, device_ids=[self.args.local_rank],find_unused_parameters=True)
```

audio-db は次のとおり。

- `world_size == 1`: BatchNorm のまま。変換しない。`validation_requires_all_ranks` は False
- `world_size > 1`: Discriminator だけ `convert_sync_batchnorm`。Generator には生きた BatchNorm が無いので変換しない。`validation_requires_all_ranks` は True
- CUDA では、そのあと Discriminator を `broadcast_buffers=False`、`find_unused_parameters=False` で DDP する。`broadcast_buffers=False` は、3 回の Discriminator forward が 1 つのグラフを共有するためである。True にすると backward 用に保存した BatchNorm 状態を forward ごとに壊し得るので、公式の既定 True との差として残した。この CUDA ラップは実行していない
- CPU では `DistributedDataParallel` が `SyncBatchNorm layers only work with GPU modules` を送出する（torch 2.6 の `distributed.py` `_passing_sync_batchnorm_handle`）。変換した Discriminator は DDP で包まず残す。プロセスグループがあるときは、バッファの broadcast と勾配の手動平均を行う。`SyncBatchNorm.forward` 自体は CPU テンソルに対し、collective の前に `SyncBatchNorm expected input tensor to be on GPU` を送出する。`aten::batch_norm_stats` に CPU カーネルは無い

`participating_validation_forward(batch)` は `validation_report` と同じである。メソッド内に rank 0 だけの collective は無い。SyncBN の collective は Discriminator の forward の中にある。`validation_requires_all_ranks` が True のとき、rank 0 だけが検証 forward するのは安全ではない。GPU では未参加のランク待ちで止まり、CPU では forward が上記の例外になる。Trainer の `_validate` と `_validate_training_data` は、このフラグが真のとき全 rank が `validation_report` を呼び、batch 数が違うときは forward の前に拒否する。その後、物理 GPU 1 と 2 で実 Discriminator の SyncBatchNorm forward は両 rank が終わり、`bn1.running_mean[0]` は一致した。Trainer の検証ループそのものの GPU 実行はしていない。

チェックポイントは、world size 1 の BatchNorm state_dict が別の world size 1 タスクへ `strict=True` で載る。SyncBN に変換した Discriminator の state_dict は BatchNorm と同じ 117 キーで、`running_mean` と `running_var` が 8 個ずつある。1 GPU（BatchNorm）の Discriminator state は、SyncBN Discriminator へ `strict=True` で載った。キー不一致では拒まれない。これは別名への黙った変換ではなく、同じ名前と形のコピーである。1 GPU チェックポイントを 2 GPU へ再開して数値を合わせた測定はしていない。GPU は NOT_TESTED。

running stats は、CPU gloo・world 2 で、公式 gloo 分岐が行う `all_gather` と同じ通信を行う試験用 forward を両ランクが実行すると更新され、両ランクの値は一致した。rank 0 だけがその forward に入ると、8 秒で `Timed out waiting 8000ms for send operation to complete` になった。この forward は `nn.SyncBatchNorm.forward` そのものではない。本物の `forward` は CPU では collective の前に例外になる。

## テスト

`CUDA_VISIBLE_DEVICES=""`、`.venvs/se/bin/python -m unittest`。

| テスト | 結果 |
|---|---|
| `test_mossformer_official_init` 残した G/D が公式と一致 | OK（最大絶対差 0） |
| PESQ `None` で例外にせず、`update_applied==0`、optimizer 不変、skip ログ | OK |
| 非有限 Generator 損失でスキップ | OK |
| 後続マイクロバッチの PESQ 失敗で群全体を step しない | OK |
| gloo world 2、rank 1 だけ PESQ 失敗、および両ランク失敗。両方スキップ、ハングなし | OK（約 350 秒の 4 件再実行に含む） |
| world 1 は BatchNorm のまま再ロードできる。SyncBN に `running_mean` がある | OK |
| 両ランクが統計の all_gather に参加し、running mean が一致 | OK |
| rank 0 だけがその all_gather に入ると 8000 ms でタイムアウト | OK |
| `test_mossformer_validation`、`test_mossformer_plateau`、`test_mossformer_discriminator` | OK |

最初の一括実行では、CPU で SyncBN を DDP に包んだ 3 件が `SyncBatchNorm layers only work with GPU modules` で失敗した。包むのを CPU では止めたあとの再実行で、その 3 件は OK である。検証・plateau・Discriminator の既存テストはその一括実行で OK であり、その後の変更は `world_size>1` の `distribute` だけである。

GPU 上の SyncBN 数値、公式 2 GPU との一致、CUDA での DDP ラップは **NOT_TESTED**。
