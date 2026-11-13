# Checkpoint の仕様

学習 checkpoint は Latest と Archive の 2 種類である。どちらも `audio-db-experiment-v1` の完全な checkpoint で、どちらからでも再開できる。いつ何を残すかは Selector が決める。ファイル名、形式、置換、検証は共通基盤（`src/experiments/execution/checkpoint.py`）が持つ。6 モデルの Trainer はこの仕組みを共有し、保存の方針を持たない。

起動と再開の手順は [学習・実験](RUN.md)、プロセスの監視と終了状態は [Launcher](LAUNCHER.md)。

## 出力ディレクトリ

`output.directory` の中身は次のとおり。

| パス | 内容 | 書き手 |
|---|---|---|
| `latest.pt` | Latest。常に 1 つ | 共通基盤 |
| `latest.data_provenance.json` | Latest の来歴の sidecar | 共通基盤 |
| `archive/<name>.pt` と sidecar | Archive。名前と保持数は Selector が決める | 共通基盤 |
| `metrics.jsonl` | logical step ごとの記録 | Trainer |
| `status.json`、`complete.json` | 実行状態と完了の印 | 学習プロセス、または Launcher |
| `launch.json`、`failure.rank<N>.json` | Launcher の記録 | Launcher、各 rank |
| `gpu_memory.jsonl`、`gpu_memory.json` | デバイス全体の GPU メモリの時系列と最大値。意味は [Launcher](LAUNCHER.md) | Launcher |
| `data_provenance.json` | Dataset、snapshot、使用集合の内容ハッシュ。rank 0 が一度だけ書く | 共通基盤 |
| `initialization.json`、`initialization.data_provenance.json` | 初期化元の checkpoint と、その来歴の写し。`initialize` を使った実験だけ | 共通基盤 |
| `best_generator.pt`、`valid.loss.best.pt`、`valid.loss.ave_<n>best.pt` など | 公式手順が定める推論用の重み。再開には使えない | 各公式 Trainer |

推論用の重みは公式再現の成果物であり、checkpoint ではない。Selector はこれらを扱わない。再開できる Best や Top-K が要るときは、Selector で Archive として残す。

## 保存点

Trainer は、状態が整合している点をすべて `checkpoint.offer(event, logical_step=..., metrics=...)` で差し出す。保存するかどうかは決めない。

| event | 意味 |
|---|---|
| `step` | epoch の内側で logical step が 1 つ終わった |
| `partial` | 蓄積の途中の安全な点。logical step は終わっていない（MossFormerGAN） |
| `boundary_pending` | epoch 末または検証の境界に入った。境界の処理はまだ |
| `boundary` | 境界の処理が終わった。検証をしたときは `metrics` に有限な値が入る |
| `stop` | 安全な停止の要求を受けた。状態は直前の保存点と同じ |
| `final` | 学習が完了した。状態は直前の保存点と同じ |

GAN の Optimizer 対の途中は保存点ではない。欠損や `na` の指標は `metrics` に入らない。0 にはしない。

## Selector

Selector は Python のクラスで、保存点ごとに `Decision(latest, archive, delete)` を返す。

```python
from src.experiments.selectors import Decision, SavePoint, Selector

class EveryEpoch(Selector):
    def decide(self, point: SavePoint) -> Decision:
        if point.event == "boundary":
            return Decision(latest=True, archive=(f"epoch_{point.logical_step:08d}.pt",))
        return Decision(latest=point.event in ("stop", "final") and not point.latest_is_current)
```

- `latest`: この点で Latest を置き換える。Selector が決めるのは時期だけである。
- `archive`: `archive/` に書く名前。`*.pt` の単純なファイル名だけを受け付ける。
- `delete`: `archive/` から消す名前。Latest は名前で指せないので、Selector は Latest を消せない。
- `state_dict()` / `load_state_dict()`: 保持表などの状態。その決定で書く checkpoint に入り、再開で戻る。

Selector は、引数、復元した状態、差し出された点だけから決まる関数でなければならない。全 rank が同じ Selector を作り、共通基盤が決定と状態を rank 間で比べる。食い違えば何も書かずに失敗する。

同梱の Selector は 3 つである。

| クラス（`src.experiments.selectors.`） | 残すもの | 引数 |
|---|---|---|
| `StandardSelector` | Latest だけ。`latest_every` step ごと、各境界、安全な停止、完了で置き換える | `latest_every`（既定 1000）、`partial`（既定 false。true なら蓄積途中の点でも置き換える） |
| `PeriodicArchiveSelector` | Latest の規則は同じ。加えて `archive_every` step ごとの Archive。`latest_every` を省くと 1 のまま | `archive_every`、`keep_last`（省略は全部残す）、`name`（既定 `step_{step:08d}.pt`）、`latest_every`（既定 1） |
| `TopKSelector` | Latest の規則は同じ。加えて境界の指標が良い順に `k` 個の Archive。`latest_every` を省くと 1 のまま | `metric`、`mode`（`min` / `max`）、`k`、`name`（既定 `best_{step:08d}.pt`）、`latest_every`（既定 1） |

YAML は Selector とその設定だけを書く。Trainer に `checkpoint_interval` は無い。

```yaml
checkpoint:
  selector:
    class: src.experiments.selectors.StandardSelector
    kwargs:
      latest_every: 5000
```

`checkpoint` を省くと `StandardSelector` で、Latest は 1000 logical step ごと、各境界、安全な停止、完了で置き換える。毎 step 保存は `latest_every: 1` と書いたときだけである。`checkpoint` は実験の同一性（`config_sha256`）に入らない。保存の方針を変えても同じ実験として再開できる。ただし、checkpoint に入っている Selector の状態は同じクラスにしか戻せないので、クラスを変えた再開は拒否する。

## 書き込みの保証

1 回の保存は次の順で進む。

1. 全 rank の parameter、buffer、Optimizer、scheduler、Trainer、データ位置、Selector の状態を比べる。不一致なら書かない。
2. rank 0 が `.latest.pt.tmp` に書き、fsync し、読み直して検証する。
3. Archive を `archive/` に置く（一時名から `os.replace`）。
4. Latest を `os.replace` で置き換える。
5. Selector が指定した Archive を消す。

- 手順 4 より前で失敗または kill されたとき、以前の `latest.pt` はそのまま残る。
- `.` で始まる一時ファイルは再開元として拒否する。書きかけのファイルが再開に使われることはない。
- 同じ点で書いた Latest と Archive は同じ内容である。
- 削除は `archive/` の中の名前にしか届かない。
- checkpoint は、その step の `metrics.jsonl` を書いたあとに公開する。
- 電源断やストレージ障害に対する、複数ファイルにまたがる原子性は保証しない。

## 再開（Resume）

再開は同じ実験の厳密な続きである。全状態、rank ごとの RNG、Selector と停止ポリシーの状態を戻す。停止ポリシーの設定と境界の意味は [停止ポリシー](STOPPING.md) を参照。

```bash
.venvs/se/bin/python -m src.experiments.launcher --config CONFIG --resume OUTPUT_DIRECTORY
```

ディレクトリを渡すと `latest.pt` を使う。`archive/` のファイルを直接渡してもよい。次は拒否する。

- 設定の同一性が違う。`config_sha256` は `output`、`checkpoint`、`stop_after_logical_steps` を除いた設定の SHA-256 である。
- 出力ディレクトリが違う。再開元は、その実験のディレクトリ直下か `archive/` にあるファイルだけである。
- world size、Dataset の同一性、データ計画の `plan_hash` が違う。
- `provenance_sha256` を持つ checkpoint の sidecar が欠けている、またはハッシュが違う。
- `complete.json` がある。完了した実験は再開しない。
- `latest.pt` より前の step の Archive から続けようとした。続けると、その先の記録と Latest を黙って捨てることになるためである。古い状態から始めるときは、次節の新しい実験にする。

停止した点より後の `metrics.jsonl` の行は、再開時に巻き戻す。

## 新しい実験の初期化

既存の checkpoint から状態を選んで読み、別の実験を step 0 から始める。元の実験のディレクトリ、checkpoint、記録は読むだけで、変更しない。

```yaml
output:
  directory: data/results/experiments/my-finetune
initialize:
  checkpoint: data/results/experiments/base-run/archive/best_00042000.pt
  states: [generator]
```

- `states` に書けるのは module、Optimizer、scheduler の state id である。データ位置、Trainer の履歴、RNG は元の実験のものなので、選べない。それらを引き継ぐのは再開である。
- Optimizer を選ぶときは、それが更新する module も選ぶ。scheduler を選ぶときは、その Optimizer も選ぶ。足りない組み合わせは拒否する。
- 元の checkpoint に無い state、この実験に無い state は拒否する。
- `initialize` は `config_sha256` に入る。初期状態が違えば別の実験である。出力ディレクトリは元の実験と別にする。
- 初期化元のパス、SHA-256、logical step、選んだ state は `initialization.json` に残る。元の来歴 sidecar は写しを取り、新しい実験の来歴に引き継ぐ。

## 以前の形式との互換

checkpoint の形式は `audio-db-experiment-v1` のままである。`audio-db-training-v1` は読まない。

- Selector 導入前の実験は、checkpoint をディレクトリ直下の `step_*.pt` に持つ。これは再開元として読める。再開後の保存は `latest.pt` と `archive/` に書く。古い `step_*.pt` は消さない。
- 以前の YAML は Trainer に `checkpoint_interval`（MossFormerGAN は `checkpoint_partial` も）を持っていた。新しい YAML ではそれを消し、`StandardSelector` の `latest_every`（と `partial`）に同じ値を書く。この対応が成り立つとき、以前の `config_sha256` を同じ実験として受け付ける。以前の YAML がその引数を省いていた場合も受け付ける。`latest_every` を省いた新しい設定が受け付ける旧間隔は 1000 であり、旧文書の `checkpoint_interval: 1` は `latest_every: 1` を書いたときだけ受け付ける。
- クラスパスを移動した世代（`src.training.se.optim.WarmupReduceLROnPlateau`、`src.experiments.<model>`、`src.experiments.models.*`）のハッシュも、同じ設定として受け付ける。新しい保存は現在のハッシュを書く。
- CLI の `--initialize-from` は無い。初期化は YAML の `initialize` に書く。
- `provenance_sha256` が None の checkpoint は来歴なしのまま再開できる。後から JSON ができても証明済みにはしない。

## 未実装

FSDP、sharded checkpoint、AMP BF16。複数 Optimizer の `amp_fp16` は拒否する。
