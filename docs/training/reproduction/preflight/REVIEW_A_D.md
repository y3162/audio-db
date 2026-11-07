# P10-C Preflight Phase 2: Agent A × Agent D

対象は Agent A と Agent D の交差（MossFormerGAN の GAN、DDP、Validation 状態）である。読んだ preflight は `AGENT_A_OFFICIAL_MP_MOSS.md` と `AGENT_D_DDP_VALIDATION.md` だけである。HIGH と BLOCKER の引用は、公式 checkout `6b3774dc79c46ae8bed2a4fa5f706f0ac8c75c61` と現行 `src/` の該当行を読み直し、足りない箇所だけ CPU で確認した。

CPU は `.venvs/se` の torch 2.6.0+cu124。`CUDA_VISIBLE_DEVICES=""` で起動し、`torch.cuda.is_available()` は False、`device_count` は 0。GPU は使っていない。`src/`、YAML、テスト、DB、Snapshot、checkpoint は変更していない。

## 矛盾

### 1. 勾配 clip は「同じ」ではない

A の一致リストは、公式も現行も各 micro-batch の backward の直後に clip 10 する、と書いている。D-001 は、2 GPU・accumulation 2 では現行が平均の前に局所勾配を切り、公式は micro ごとの DDP backward（平均）のあとに切る、と書いている。

ソースの並びはどちらも `backward` の次の行が `clip_grad_norm_` である。公式 `solver.py` 282–286 行は `no_sync` が無く、DDP の平均は `backward()` が返る前に終わる。現行 `mossformergan.py` 229–257 行は、非最終 micro の Generator を `accumulation_scope(..., last=False)` で `no_sync` にし、その局所勾配を切る。最終 micro の allreduce はその後である。Discriminator は毎 micro を `last=False` にし、最終 micro の backward のあと、手動 allreduce のあとに切る。

gloo、world 2、CPU、Linear でこの並びを比較した。

- clip が掛からない上限では、公式と現行の最終勾配の最大絶対差は `1.2e-7`。
- 両 micro の clip 前 norm が 100 と 58、上限 0.05 のとき、最終 norm はどちらも 0.05、cosine は 1.000、最大絶対差は `3.4e-6`。最後の micro が最終 clip を支配する。
- 2 個目の micro だけ小さいとき、cosine は 0.998、最大絶対差は `2.4e-3`（norm 0.05）。
- 平均が終わったあとに 1 回だけ切る案は、公式との最大絶対差 `2.7e-2` で、公式にも現行にも一致しない。

A の「一致」は、clip が恒等のときと world 1 に限る。2 GPU で途中の micro に clip が効くと、平均との前後が違う。D の順序の指摘がコードと一致する。D が書いた純計算の差 0.020 は、このレビューの「最終 clip まで含めた勾配」とは別の量で、そのスクリプトは再実行していない。

判定: DISPUTED。A の一致リストからは外す。D-001 は条件付きの差として残す。BLOCKER には上げない。

### 2. Generator に BatchNorm は無い

A-008 は、Generator 側の `BatchNorm1d` が `conv_module.py` 182 行だと書いている。D は、Generator は LayerNorm 系で running stats が無いと書いている。

182 行は `select_norm` の else である。`select_norm('bn')` を呼ぶのは同じファイルの `Conformer_ConvModule` だけで、このクラスを構築している箇所は MossFormerGAN の公式ツリーにも現行ツリーにも無い。現行 `SyncANet(num_channel=64, num_features=201)` を CPU で構築すると、`_BatchNorm` は 0、`running_*` buffer も 0、buffer 総数も 0。内訳は `InstanceNorm2d` 16、`LayerNorm` 60、`LayerNormalization4D` 12、`LayerNormalization4DCF` 78。`InstanceNorm2d(affine=True)` の既定は `track_running_stats=False` で、state_dict は weight と bias だけである。

`SyncBatchNorm.convert_sync_batchnorm` は `BatchNorm` だけを `SyncBatchNorm` に置き換える。同じ torch で、`InstanceNorm2d` と `LayerNorm` は変換されず、`BatchNorm2d` だけが `SyncBatchNorm` になった。公式 `solver.py` 30 行は Generator にも `convert_sync_batchnorm` を呼ぶが、BatchNorm が 0 ならこの呼び出しは Generator を変えない。

生きている BatchNorm は Discriminator 側にある。現行 `Discriminator(ndf=16)` は `BatchNorm2d` が 6、`BatchNorm1d(768)` が 2、running buffer が 24。`BatchNorm1d` は `conformer.py` 185 行で、`ConformerBlock` が 2 個ある。公式 `conformer.py` 167 行も同じである。A が引用した公式 `discriminator.py` 125–165 行は `BatchNorm2d` の 6 個だけで、Conformer の `BatchNorm1d` は入っていない。D も AdaptCNN の 6 個だけを数えている。

判定: DISPUTED。Generator に running stats が無い、という D がコードと一致する。A-008 の Generator 引用は、構築されない分岐である。Discriminator を SyncBatchNorm にする差は、この訂正のあとも残る。

### 3. 公式が残す Discriminator は 2 回目ではない

A-006 は、公式が 1 回目の Generator と 2 回目の Discriminator を残し、seed 777 の 2 回目の checksum が `7729.488281` だと書いている。実験の手順は `MossFormerGAN_SE_16K` を 2 回構築する、である。D は初期化順を扱っていない。

公式の呼び出しは次のとおり。

- `networks.py` 17–18 行は、`network_wrapper` 1 回の中で `MossFormerGAN_SE_16K` を 2 回作る。1 回目の `.model` と、2 回目の `.discriminator` を持つ。
- `generator.py` 23–27 行は、`mode == 'train'` のとき SyncANet のあとに Discriminator を作る。YAML と `train.py` の既定は `mode='train'`。
- `train.py` 29–33 行は、`network_wrapper` を 2 回呼ぶ。残すのは 1 個目の `.se_network` と、2 個目の `.discriminator`。

公式 `networks.py` を、構築だけ数える stub で 2 回呼ぶと、消費順は G1 D1 G2 D2 G3 D3 G4 D4 で、残るのは G1 と D4 だった。1 個目の wrapper が持つ D2 は、`train.py` が `.se_network` だけを残すので捨てる。A の 2 回構築が測った「2 回目の Discriminator」は、この D2 に対応する。

現行 `mossformergan.py` 97–99 行は、同じ seed で Generator と Discriminator を 1 回ずつ作る。これは G1 と D1 である。G1 が公式の残す Generator であることと、A が測った checksum の一致は、このレビューでは再測定していない。D1 は公式が学習に使う D4 ではない。A の「2 回目を残す」という対応では D2 になり、`train.py` の初期値には届かない。

判定: DISPUTED。初期 Discriminator が公式と違う、という A-006 の方向は維持する。回数と checksum `7729.488281` は、公式が残す重みの識別としては棄却する。

### 4. PESQ 失敗で学習が止まる形

A-007 と D-004 は、現行の学習が `TrainingError` で止まり、公式は `None, None` を返してそのバッチの backward をしない、と書いている。検証はどちらも 0 を分母に残し、そこでは止まらない、と書いている。

`train_update` 235–240 行は `metric_targets` が `None` のとき必ず raise する。`metric_targets` 113–114 行は、1 本でもスコアが負なら `None` を返す。公式 `batch_pesq` 145–146 行も、`-1` が 1 つでもあれば `None`。公式 `loss.py` 72–75 行はそこで `None, None` を返し、`solver.py` 279 行の `if loss_gen is not None` の外では backward しない。`fit` 267–274 行は例外を FAILED にしてから再送出する。学習ステップが止まる、という結論は両者で一致し、このレビューもその分岐を維持する。

割れるのは 2 GPU の終わり方である。D-004 と D-005 は、失敗した rank だけが allreduce に来ないと、生存側が待つと書いている。A-007 は run 全体が失敗する、と書いている。待ちを起こす実行はどちらもしていない。このレビューもしていない。片 rank だけの PESQ 失敗が、FAILED で終わるのか collective で止まるのかは未確認である。

判定: 学習が例外に入ることは一致。2 GPU の回収不能な待ちは DISPUTED のまま（D は SUSPECTED、A は失敗と書き、実測は無い）。

## 一致した結論

- 検証中の Discriminator は `train()` のままである。公式は `solver.py` 197 行で `self.model.eval()` だけ。現行 `eval_mode` 123–127 行と `validation_report` 277–278 行も Generator だけ `eval()` にする。`BatchNorm1d` を `train()` と `no_grad` で forward すると `running_mean` が動き、`num_batches_tracked` が 0 から 1 になった。`eval()` では変化 0。A の公式 Discriminator での確認と、D の AdaptCNN での確認は、この規則と矛盾しない。
- 学習の PESQ 失敗は現行が例外、公式がスキップ。検証の generator 損失は、PESQ が無いバッチを 0 かつ status `ok` として平均に残す。公式の検証平均は `total_loss / (i+1)` で、`None` のバッチは分子に足さない。batch 1 では同じ形である。
- 公式の 2 GPU は Generator と Discriminator を `SyncBatchNorm.convert_sync_batchnorm` してから DDP し、`find_unused_parameters=True`。`broadcast_buffers` は渡していない。現行は `convert_sync_batchnorm` を呼ばない。Discriminator は `broadcast_buffers=False` で、micro-batch の前に rank 0 の buffer を 1 回 `broadcast` する（220–224 行）。1 プロセスでは `wrap_module` が DDP しない。公式も `args.distributed` が偽のときは SyncBatchNorm にしない。
- 検証 forward は `_validate_training_data` 1262 行以降、rank 0 だけである。checkpoint は rank 0 の `_save` が `task.state_dict()` を書く。Discriminator の state_dict には running buffer が入り、Generator の state_dict には running buffer が無い。検証の直後に保存する step では、ファイルの Discriminator buffer は rank 0 が検証で更新した値になる。halve の step はこの限りではない（見落とし）。
- train モードの BatchNorm は batch stats で正規化する。running stats を配っても、その forward の出力は running stats を読まない。学習の出力を公式からずらすのは、Discriminator の batch stats を GPU 間で平均しないことと、検証を rank 0 が全件見ることである。D-002 のこの切り分けは、A の「次 step で running stats を配る」と同時に成り立つ。配られるのは buffer の値で、次の train forward の正規化式そのものではない。
- world 1 の clip は、DDP が無いので両者とも局所勾配を切る。A の一致リストの clip は、この場合には成り立つ。
- `find_unused_parameters` は公式が True、現行の既定が False。入力 `(1, 2, 6, 201)` の Generator backward では、1053 個のパラメータのうち `grad is None` は 0。厳密に 0 の勾配テンソルが 1 個あった。Discriminator を magnitude `(1, 1, 201, 40)` で 1 回 backward すると、93 個のうち `grad is None` は 0。この入力では、未使用パラメータによる DDP の停止は起きない。全入力長での被覆ではない。

## DISPUTED

| 項目 | A | D | このレビュー |
|---|---|---|---|
| 2 GPU の clip と平均の前後 | 一致 | 非最終は平均前に切る | D。恒等 clip と world 1 だけ A の記述が成り立つ |
| Generator の BatchNorm | `conv_module.py` 182 行を Generator の BN とする | LayerNorm で running stats は無い | D。182 行は未使用。構築結果は BN 0 |
| 公式が残す Discriminator | 2 回目、checksum `7729.488281` | 記述なし | 2 回目ではない。`train.py` が残すのは D4。checksum は再計算していない |
| PESQ 失敗時の 2 GPU の終わり方 | run が失敗する | 片 rank だと collective 待ちがあり得る | 例外に入ることは一致。待ちは未実行のまま |

## 根拠が弱い Issue

- D-001 の実害。順序の差はコードと gloo で確認した。公開 2 GPU（global 4、micro 1、accumulation 2、`grad_clip` 10）で、途中 micro の勾配 norm が 10 を超えるかは、A も D も測っていない。このレビューの Linear では、最後の micro が大きいと最終勾配の cosine が 1 まで戻った。実ネットワークで重みがどれだけずれるかは未知のままである。
- D-009。フラグの差は確定している。上記の 1 入力では未使用は出なかった。D の「開始直後に止まり得る」は、この入力に対しては弱くなる。別の時間長でグラフから外れるパラメータが無いことまでは見ていない。
- D-005。SIGTERM と片 rank 例外の待ちは、コード上は起こり得る形で、意図的には再現していない。BLOCKER の根拠に使うには実測が足りない。
- A-006 の数値。差があること自体は呼び出し回数から言える。`7729.488281` と差 `601.67` は、残る D4 の checksum ではない。
- A のセクション 3。plateau が検証と同じ step で進むかは、1 pass の更新数が `validation_interval`（1 GPU は 5786、2 GPU は 2893）と一致するときだけ、と A が書いている。件数は未計測のままである。`logical_steps_per_epoch` は再現 YAML に無く、`step_schedulers` は `epoch_done` のときだけ走る。検証と checkpoint は `global_step % interval == 0` で走る。件数が違うと、検証と plateau は別の step になる。

## 見落とし

- Discriminator の `BatchNorm1d` が 2 個、forward に入っている。SyncBN の対象は AdaptCNN の 6 個だけではない。running buffer は 24 個で、checkpoint の `state_dict` に残る。
- 波形の micro-batch 1 を、BN の母集団と同一視している。時間 321 フレームの magnitude を `segment_specs` に入れると `(77, 1, 201, 15)` になる。2 秒、hop 100、center=True のフレーム数は 321 なので、AdaptCNN の batch 次元はこの長さでは 77 である。Conformer の `BatchNorm1d` は窓の並びを長さ軸に持つ。公式の SyncBN は、その統計を rank 間で合わせる。現行は rank 内の統計だけを使う。比は約 2 倍であり、母集団の絶対値が 1 と 2 という D-002 の書き方は過大に悪い。torch 2.6 は、チャネルあたりの値が 1 個の train モード BatchNorm を `ValueError` にする。2 秒のこの形はその例外には入らない。
- 検証の非有限損失。`validation_report` 281 行は、PESQ の有無を見る前に `_finite` で raise する。公式は NaN のとき `None` を返し、分子に足さず分母 `(i+1)` に残して次のバッチへ進む。PESQ 失敗を 0 として残す点は公式の検証と揃う。NaN の検証は現行だけがその場で止まる。A-007 の「検証側は公式と揃っている」は PESQ 失敗に限り、非有限には及ばない。
- halve のあとの checkpoint。保存は検証のあと、`epoch_done` なら `step_schedulers` のあとである。halve は best snapshot を読み戻してからファイルを書く。その step のファイルは、直前の検証で更新した buffer ではなく、読み戻した snapshot である。improve と hold の step では、D の「検証で更新された rank 0 の buffer がファイルに残る」が成り立つ。
- 公式の PESQ スキップは `accu_count` を増やさない。`self.step` は増える。スキップは optimizer step の枠を消費せず、後続の成功バッチと組み合わさる。現行は例外なのでこのずれは表面化しない。A は `self.step` の加算を書いており、蓄積カウントが動かないことは書いていない。
- 公式 `save_checkpoint` 164–183 行に rank の条件が無い。全 rank が同じパスへ書く。SyncBN で buffer が揃っていれば中身は同じ系統になる。現行は rank 0 だけが書く。
- 公式の 2 GPU で PESQ 失敗が rank ごとに違うと、公式側も backward の有無が rank で割れ、allreduce が揃わない。A-007 の「公式は学習を続ける」はプロセス単体の分岐である。D-004 の対応方針は、skip を allreduce してから全 rank を同じ枝に入れる、と書いている。公式をそのまま写すだけでは 2 GPU の継続にならない。

## P10-C への影響

| Issue | 判定 | 理由 |
|---|---|---|
| A-006 初期 Discriminator | BLOCKER を維持 | 現行は D1。公式の学習が使うのは D4。GAN の初手から重みが違う。1 GPU でも 2 GPU でも出る。対応を「2 回目を残す」にすると D2 になり、公式の初期値にはならない |
| A-007 / D-004 PESQ で学習が止まる | BLOCKER を維持 | `train_update` は PESQ 失敗で必ず `TrainingError`。公式はそのバッチを飛ばす。検証は 0 を残して止まらない。無音が 1 バッチでも学習 run は終了する。2 GPU の待ちは未実測なので、この BLOCKER の追加理由にはしない |
| A-008 / D-002 SyncBatchNorm と検証の buffer | 2 GPU は BLOCKER を維持。1 GPU は降格 | 2 GPU の学習をずらすのは、Discriminator の 8 個の BatchNorm が局所 batch stats を使うことである。Generator に BatchNorm は無く、公式の Generator への `convert_sync_batchnorm` は変換対象が無い。running buffer は checkpoint の Discriminator に残るが、train モードの正規化はそれを読まない。1 GPU は公式も現行も SyncBatchNorm にしない |
| D-001 clip が平均の前 | BLOCKER にしない | 2 GPU の順序差は確認した。clip が効かない step は最終勾配が一致する。効く step の実 MossFormer 勾配 norm は未計測で、toy では最終 clip が差を小さくすることがある |
| D-009 `find_unused_parameters` | BLOCKER にしない | 試した 1 入力では `grad is None` が 0。フラグの差だけが残る |
| D-005 停止時の待ち | BLOCKER にしない | 分岐は読んだ。待ちは起こしていない |

2 GPU の `mossformergan_vb28_official_2gpu.yaml` を公式 `train.sh`（`n_gpu=2`）と比べるなら、残る BLOCKER は初期 Discriminator、PESQ 失敗での学習停止、Discriminator の SyncBatchNorm である。1 GPU の `mossformergan_vb28_official.yaml` では SyncBatchNorm の BLOCKER は降り、初期 Discriminator と PESQ 停止は残る。
