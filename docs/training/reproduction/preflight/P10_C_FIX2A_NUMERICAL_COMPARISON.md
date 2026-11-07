# P10-C Fix.2-A 数値比較

比較の前に許容誤差を決めた。結果を見てから広げていない。

| 項目 | 値 |
|---|---|
| dtype | float32 |
| 実モデルの装置 | CUDA。1 GPU は物理 GPU 0。2 GPU は物理 GPU 6 と 7、NCCL |
| 最小例 | CPU gloo、および同じ式の NCCL |
| cuDNN | `deterministic = True`、`benchmark = False` |
| 初期重み | 最大絶対誤差 0 |
| 損失スカラー | atol `1e-5`、rtol `1e-5` |
| clip 後の勾配 | atol `1e-4`、rtol `1e-4` |
| 1 更新後の重み | atol `1e-5`、rtol `1e-5` |

重みの atol `1e-5` は、公式ループを同じ初期重みで 2 回走らせた差より小さい。その自己差を下回る一致は、この CUDA では主張しない。勾配の atol `1e-4` は維持する。

## 比較条件

公式は `src/se/official/clearervoice-studio` の commit `6b3774dc` を読んだ。ソースは変更していない。公開 YAML は `batch_size: 1`、`effec_batch_size: 2`、`accu_grad: 1`、`clip_grad_norm: 10`、AdamW、Generator の学習率 `5e-4`、Discriminator はその 2 倍である。

実モデルの 1 GPU 比較は、公式 `network_wrapper` が残す Generator と Discriminator に、公式 `loss_mossformergan_se_16k` をかけた。audio-db は同じ重みを読み、`MossFormerGANTask.train_update` を蓄積回数 2 で 1 回呼んだ。入力は 32,000 サンプルの合成正弦で 2 micro-batch である。PESQ は両方とも `pesq` の wideband を `(score - 1) / 3.5` にしたもので、失敗は無かった。

2 GPU の実モデルは audio-db のモジュールを DDP で包んだ。公式型は `no_sync` を使わず、3 forward のあと 2 backward し、毎回 clip してから 1 回更新した。audio-db 型は `train_update` である。rank ごとに周波数の違う 32,000 サンプルを 2 件ずつ使った。初期重みは rank 0 から broadcast した。

公式 `train.py` をデータローダ付きで複数 epoch 実行してはいない。ここでの「公式」は、solver の更新順を同じモジュール上で実行した参照である。公式リポジトリそのものの長時間学習との一致とは呼ばない。

## 公式参照

1 micro-batch の実効順は次である。

1. Generator forward
2. Discriminator forward（GAN 項）
3. PESQ
4. 成功時だけ Discriminator をさらに 2 回 forward
5. Generator 損失を蓄積回数で割って backward
6. DDP がこの backward に同期を付けているなら、ここで rank 間平均が終わる
7. Generator パラメータを clip
8. Discriminator 損失を蓄積回数で割って backward
9. Discriminator パラメータを clip
10. 成功数が蓄積回数に達したら Generator、Discriminator の順に `step` と `zero_grad`

失敗時は 5 以降を行わない。`self.step` に相当する `official_step` は失敗でも増える。epoch の開始時に残った勾配は捨てる。窓の途中では Optimizer を呼ばない。

## 最小 Tensor の比較

単一パラメータ、world size 2、gloo、蓄積 1 と 2、`gradient_as_bucket_view` の True と False、clip なし、clip 10、clip 0.5 を比較した。1 backward につき 1 forward のとき、公式型 DDP と増分平均の clip 後勾配と AdamW 後の重みの最大絶対差は 0 だった。明示的に蓄積勾配全体を all-reduce する参照も 0 だった。

同じ条件で、最後の backward まで同期せず最後に 1 回だけ clip すると、clip 0.5 と蓄積 2 では 1 回目の backward 直後の勾配が公式型と最大 22.5 違った。単一パラメータは clip 後にどちらも絶対値 0.5 へ潰れるので、重み差は 0 のまま残ることがある。この形では重みだけでは clip 位置の差を検出できない。

パラメータが平行でない CPU モデルでは、clip 無しの重み差は 0、clip 0.5 の micro-batch ごとの clip と最後の 1 回の clip の重み差は `0.0010000020265579224` だった。clip 前の norm は、毎回 clip する側が 9.865501403808594 と 19.472532272338867、最後に 1 回の側が 1 回目 9.865501403808594 のまま、2 回目の clip 前が 26.335575103759766 だった。clip 後はどちらも約 0.5 だが、方向が違うので重みが分かれる。

world size 2 の gloo テストは、この非平行な入力で公式型と増分平均の重み差が 0、最後に 1 回だけの clip との差が 5e-4 より大きいことを固定した。clip を `1e9` にすると最後の 1 回との差も 0 だった。

3 forward と 2 backward の最小 Discriminator は、gloo でも NCCL でも同じ割れ方だった。NCCL の rank 差は次である。

| 量 | 最大絶対差 |
|---|---|
| Generator の clip 後勾配 | 0 |
| 1 回目の backward 直後の Discriminator 勾配 | 0 |
| 2 回目の backward 直後の Discriminator 勾配 | 52.44927215576172 |
| Discriminator の 1 更新後の重み | 0.0020000040531158447 |

0.002 は Discriminator 学習率 `1e-3` の 2 倍で、1 段目の Adam が逆符号に動いた大きさである。増分平均はこの Discriminator 勾配の rank 差を 0 にする。公式型 rank 0 の勾配とは 10 より大きく違う。差は 2 回目の backward から出る。1 回目までは rank 差が 0 である。world size 1 では all-reduce が無いので、この割れは起きない。world size 3 以上は測っていない。蓄積回数を 2 にしても、割れるのは同期されない 2 回目の backward であり、同期された backward の再平均そのものではない。

## 実モデルの比較

初期重みは、別々に構築した公式モジュールと audio-db の最大絶対差が Generator も Discriminator も 0 だった。最悪キーは無い。この一致は `test_mossformer_official_init` の既存期待とも同じ種類で、このセッションで計算し直した。

勾配が無い Generator パラメータは 12 個で、両側とも `blocks.{0..5}.{intra,inter}_mossformer.rotary_pos_emb.freqs` だけだった。Discriminator に欠測は無かった。

同じプロセスで公式重みを audio-db にコピーした 1 GPU の 1 更新は次だった。公式損失は 1 件目が Generator `1.1001978`、Discriminator `0.37416187`、2 件目が `0.57196355` と `0.49374026` である。audio-db が返した平均は Generator `0.8360806703567505`、Discriminator `0.4339510649442673` で、表示した公式スカラーの平均と一致する。clip 前の norm は Generator が 72.115616 と 26.714157、Discriminator が 1.0548322 と 1.7482044 である。Generator は閾値 10 を超える。

clip 後の勾配の最大絶対差は、1 件目の Generator が `5.69317489862442e-06`、Discriminator が `2.7939677238464355e-09`、2 件目の Generator が `5.9604644775390625e-06`、Discriminator が `2.7008354663848877e-08` だった。絶対誤差は事前の `1e-4` 未満である。要素ごとの相対誤差の最大は Generator `0.006559934001415968` で、事前の rtol `1e-4` は超える。最悪キーは `dense_encoder.conv_1.0.weight` で、絶対差は小さい要素に出ている。

そのあとの重みの最大絶対差は Generator `0.0009692218154668808`、Discriminator `6.973743438720703e-05` だった。事前の重み atol `1e-5` は超える。同じ公式更新を初期重みから 2 回繰り返すと、Generator の最大絶対差は `0.0004379786550998688`、Discriminator は `7.694587111473083e-05` だった。audio-db の 2 回繰り返しは Generator `0.000456823967397213`、Discriminator `0.00010448694229125977` だった。公式と audio-db の差は Generator `0.0009037591516971588`、Discriminator `0.00018421560525894165` で、公式自身の繰り返しと同じ桁である。これはアルゴリズム差ではなく、CUDA の非決定性と、小さい勾配要素を Adam の 1 段が符号方向へ増幅したものである。

別プロセスで初期重みをコピーせずに公式損失を計算すると、Generator 損失は `2.264711380004883` と `0.49533045291900635`、clip 前 norm は 326.47088623046875 と 16.930036544799805 だった。同じプロセスの audio-db 平均は `1.3800209164619446` と `0.41243574023246765` で、そのプロセスの公式平均と一致した。プロセス間で公式損失の水準が違った理由は、この 2 本の実行だけでは切っていない。プロセス内の公式と audio-db の対応は両方で取れている。

## Gradient accumulation の差

同期される backward では、蓄積済み勾配の再平均と増分平均の差は 0 だった。差は backward の回数や world size ではなく、その backward が DDP の all-reduce を実際に行うかに依存する。公式の 2 回目の Discriminator backward は行わない。

audio-db はそこも増分平均する。実モデル 2 GPU の 1 更新後、公式型の Generator rank 差は 0、Discriminator rank 差は `0.0020000338554382324`（`cnn.bn1.weight`）だった。audio-db は Generator も Discriminator も rank 差 0 だった。rank 0 での方式差は Generator `0.0006423667073249817`、Discriminator `0.0020000338554382324`（`cnn.bn6.weight`）である。Discriminator の方式差は、公式型の rank 割れと同じ `0.002` である。

この 2 GPU 実行の clip 前 norm は、両 rank の Generator が 26.28291130065918 と 56.12959671020508 で一致した。Discriminator は rank 0 が 1.4648336172103882 と 1.783791184425354、rank 1 が 1.1697158813476562 と 1.6959120035171509 で、公式型では rank が揃っていない。Generator はどちらも 10 を超えた。

## DDP 同期の差

| 処理 | 公式型 DDP | audio-db |
|---|---|---|
| Generator backward | 平均される。rank 差 0 | 増分を平均する。rank 差 0 |
| GAN 項の Discriminator 勾配 | 1 回目の backward で平均される | 同じ |
| Discriminator 損失の勾配 | 2 回目の backward で平均されない | 増分を平均する |
| 片 rank の PESQ 失敗 | その rank だけ backward しないと collective が割れる | 全 rank が backward しない |

2 番目の差が `INTENTIONAL_DIFFERENCE` である。安全な DDP を優先し、割れた重みは採用しない。

## Clip 前後の差

公式も audio-db も、成功した micro-batch の backward ごとに、その時点の走っている和を clip する。最後に 1 回だけ clip する一般的な蓄積とは、1 回目の norm が 10 を超えると重みが分かれる。実モデルの Generator は、測った 1 GPU と 2 GPU のどちらでも 1 回目から 10 を超えた。Discriminator の clip 前 norm は約 1.1 から 1.8 で、この入力では 10 未満だった。学習全体で何回超えるかは測っていない。

## Optimizer 更新後の差

1 GPU の公式モジュールと audio-db の重み差は、上の CUDA 自己差の範囲である。勾配の絶対差は `1e-4` 未満だった。

2 GPU の 1 更新後、audio-db の両 rank は一致する。公式型の Discriminator は最大 `0.0020000338554382324` 割れる。Generator は公式型でも audio-db でも rank 差 0 である。

## 意図的に残した差異

1. Discriminator 損失の勾配を rank 間平均する。公式の 2 回目の backward は平均しない。
2. 2 GPU で PESQ や非有限損失が片 rank だけのとき、全 rank がその micro-batch を backward しない。公式は rank ごとに落とす。
3. `global_step` は Optimizer 更新だけを数える。`official_step` は失敗を含む micro-batch を数える。公式の `self.step` は後者である。

## NOT_TESTED

- 公式 `train.py` の複数 epoch と、VoiceBank ローダを通した公式プロセスとの重み比較
- world size 3 以上
- clip が 10 を超える頻度の 1 epoch 計測
- 2 GPU の実モデルで、PESQ 失敗を挟んだ勾配テンソルの公式型との比較。カウンタと全 rank スキップは CPU テストで確認した
- Checkpoint からの本格再開と、CUDA illegal memory access の再現追跡
- プロセスをまたいだ公式初期重みが、損失 2.26 の実行と損失 1.10 の実行で同じだったかの切り分け
