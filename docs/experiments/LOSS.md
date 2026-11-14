# 損失

学習が backward するスカラーは、YAML の Loss Term と Objective から作る。Trainer の更新順は変えない。変えてはいけない条件は [.agents/ARCHITECTURE.md](../../.agents/ARCHITECTURE.md)。

## 3 つの役割

| 役割 | すること | しないこと |
|---|---|---|
| Loss Term | context から 1 つのスカラーを返す。項の内部パラメータは自分の `kwargs` から取る | backward、Optimizer、checkpoint、Trainer の式 |
| Objective | 注入した項を、その並びの左から `weight * term` で足す。total と項ごとの値を返す | backward、Optimizer の step |
| Trainer | forward、backward、蓄積、Optimizer の順、GAN の更新順、DDP、AMP、再開 | 係数の直書き、項の追加のための式の編集 |

項も Objective も `role: loss` である。既存の `class`、`kwargs`、`inject`、`ref` で組み立てる。新しいローダ、評価用 DSL、`eval` は足さない。

重み `1.0` は乗算しない。その項のテンソルが total にそのまま入る。`0.0` は有効な係数で、項は実行され、その項の勾配は 0 になる。有限な負の重みは、項が禁じていない限り許す。非有限、項名と一致しない重み、重複した項名、欠ける context キー、スカラでない戻り値は構築時または forward 時に拒否する。実行中に YAML を書き換えない。

`config_sha256` は損失コンポーネントの class、role、kwargs、inject を含む。係数を変えた設定は、変える前の checkpoint から再開できない。損失モジュールの内部状態は checkpoint に入れない。重みを step で変えるスケジュールは未実装である。

## 係数を変える

Objective の `kwargs.weights` の数値を変える。項の並びは `inject.terms` の順が加算順である。重みの辞書のキー順は加算順ではない。

TF-GridNet の公式項は次の形である。

```yaml
  loss:
    role: loss
    class: src.experiments.losses.MultiResolutionL1Loss
    kwargs:
      name: mr_l1
      windows: [256, 512, 768, 1024]
      time_domain_weight: 0.5
      normalize_variance: false
  objective:
    role: loss
    class: src.experiments.losses.WeightedObjective
    kwargs:
      weights:
        mr_l1: 1.0
    inject:
      terms:
        - {ref: loss}
```

Trainer は `objective: {ref: objective}` を受け取る。`MultiResolutionL1Loss` を直接渡す経路は、重み 1 の Objective で包む。

## 項を足す

1. Python の `nn.Module` を足す。`name` を持ち、`forward(context)` が 1 つのスカラを返す。
2. 実験 YAML に `role: loss` のコンポーネントを足す。
3. Objective の `inject.terms` に `{ref: ...}` を足し、`weights` にその `name` の係数を書く。
4. Trainer の class と更新順は編集しない。

研究用の例は `src.experiments.losses.WaveformL1` である。context の `estimate` と `target` の平均絶対誤差を返す。TF-GridNet の Trainer は、YAML がこの項と `0.25` を足したとき、total と勾配を変える。`0.0` のときは total も勾配も公式の Objective と同じである。

MossFormerGAN の time 項は、この波形 L1 ではない。clean の real と予測 imag から iSTFT した重なりの平均絶対誤差で、教師は `clean * noisy energy scale` である。Trainer が 1 回だけ計算し、`ContextTerm` がそのスカラーを読む。

## 公式と研究

| claim | 損失 |
|---|---|
| `official_faithful`、`leakage_safe` | MP-SENet と MossFormerGAN は、公開した項、順、係数、内部パラメータと違う Objective を構築時に拒否する。`check_finite: true` も拒否する |
| `research` | 項の追加、削除、重みの変更ができる。公式再現としては記録しない |
| SEMamba++、DeepFilterNet3、TF-GridNet、TF-Locoformer | claim による損失の施錠は無い。出荷 YAML の係数をテストが公開値と比べる。SEMamba++ の `official_reproduction` と DeepFilterNet3 の公式データセットは、損失とは別に拒否されたままである |

`experiment_claim` を公式にしただけでは、任意の損失は通らない。MP-SENet と MossFormerGAN は Objective の型、項の class、kwargs、重み、順を見る。

公式が拒否する例は、項の欠落、順の入れ替え、係数の変更、位相 mode の変更、complex の内部 scale の変更、別の Objective class、`check_finite: true` である。

## モデルごとのグラフ

公開係数は出荷 YAML と `src/experiments/training/official_objectives.py` にある。参照ステップの係数は `reference.py` に残り、学習経路はそれを読まない。

| モデル | Objective | 公開係数 | 検証 |
|---|---|---|---|
| MP-SENet | 信号と metric の 2 つ、および Discriminator | 信号は magnitude 0.9、phase 0.3（`diff`）、complex 0.1（内部 scale 2）、consistency 0.1（内部 scale 2）、time 0.2。metric は 0.05。Discriminator は real を 1 へ、PESQ があるときだけ fake をその目標へ | 参照タスクの式 |
| MossFormerGAN | Generator と Discriminator | RI 0.1、magnitude 0.9、time 0.2、GAN 0.05。Discriminator は real を 1 へ、fake を PESQ へ | 参照タスクの式 |
| SEMamba++ | Generator と Discriminator | adversarial 1、matching 1（各族の feature matching は scale 2、reduction sum、その和へ重み）、mel 0.1、magnitude 0.9、phase 0.3（`matrix`）、complex 0.1（内部 scale 2）、consistency 0.1（内部 scale 2）。波形 L1 は和に入らない。mel の `0 * L1` は項の中に残る | 参照タスクの波形 L1 |
| DeepFilterNet3 | spectral と local SNR | spectral の Objective 重みは 1。項の中は factor 500、complex factor 500、gamma 0.3、FFT 256/512/1024/2048、floor `1e-12`。local SNR の重みは `1e-3`。窓は 20 ms、energy floor は `1e-12`。FFT、hop、sample rate、SNR の上下限はモデルの context | 注入した Objective |
| TF-GridNet、TF-Locoformer | `mr_l1` を 1 つ | windows 256/512/768/1024、time domain 0.5、分散正規化なし、Objective 重み 1 | 注入した Objective |

MP-SENet の CUDA 重なり経路（world size 1、蓄積 1）は、信号、Discriminator、metric の順に backward する。通常経路は Discriminator の更新のあと、信号と metric の和を 1 回 backward する。PESQ の非同期と、Discriminator が Generator より先であることは変えない。

SEMamba++ の蓄積 1 は Generator グラフを再利用する。蓄積 2 以上は Discriminator のあと Generator を再 forward する。CQT 族を足してから MRD 族を足す。

DeepFilterNet3 の分解能ループの中の加算は spectral 項の中に残す。8 個の独立した項には分けない。

## 数値の確認

```bash
CUDA_VISIBLE_DEVICES="" .venvs/se/bin/python -m tests.suite_guard \
  .venvs/se/bin/python -m unittest tests.experiments.test_loss_objectives -v
```

公式 Trainer の 1 step、蓄積、Gloo、再開は、各 `tests/experiments/test_<model>.py` を同じ許容差で見る。GPU と NCCL は `make test` の GPU レーンである。CPU の成功は GPU の成功ではない。`tests/numerics/reference/` の公式 checkout との比較は、`.venvs/official` と `src/se/official/` があるときだけ実行でき、無いときの skip は一致ではない。
