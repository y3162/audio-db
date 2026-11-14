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

Trainer は、自分の更新順の中で状態が整合している点を選び `lifecycle.save_if_needed(event, logical_step=..., metrics=..., progress=...)` を呼ぶ。`logical_step` は省略できる。ファイルを書くかどうかは Selector が決める。Lifecycle が `CheckpointManager.offer` を呼び、Selector が断った点は何も書かない。Core は安全地点を推定しない。`start` と `finish` は Trainer の `fit` が呼ぶ。

Trainer は `CheckpointManager` を受け取らない。出力ディレクトリは `lifecycle.output_directory`、設定の同一性は `lifecycle.config_identity` である。研究評価にはこの二つを渡し、Manager も Trainer 自身も渡さない。`lifecycle.restored` は `restore_report.mode` が `resume` のときだけ真である。`initialize` は別の実験なので偽である。直接比較の `DirectSession` は同じ三つの読み取り属性を出す。

`progress` は名前付きの非負整数である。6 つの公式 Trainer と `StandardTrainer` は `logical_step` と `epoch` を出す。出さない Trainer は、カウンタを要らない `final`、`stop`、`boundary` で Latest を残せる。`latest_every` や Archive の間隔は、既定のキー `logical_step` をその点から読む。キーが無いとその点を拒否する。`progress_key` で数える名前を変えられる。`progress_keys` を宣言した Trainer は、その Selector が要するキーを構築後の attach で検査する。宣言が無い Trainer は検査しない。

payload は `progress` を必須にする。同じ形式名でも `progress` が無いファイルは再開を拒否し、`logical_step` だけから progress を作らない。`progress` に `logical_step` があるとき、payload の `logical_step` はその値と一致する。キーが無い保存点では `logical_step` は null である。

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
    checkpoint_class_id = "my_package.policies.EveryEpoch"

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
| `StandardSelector` | Latest だけ。`latest_every` step ごと、各境界、安全な停止、完了で置き換える | `latest_every`（既定 1000）、`partial`（既定 false。true なら蓄積途中の点でも置き換える）、`progress_key`（既定 `logical_step`） |
| `PeriodicArchiveSelector` | Latest の規則は同じ。加えて `archive_every` step ごとの Archive。`latest_every` を省くと 1 のまま | `archive_every`、`keep_last`（省略は全部残す）、`name`（既定 `step_{step:08d}.pt`）、`latest_every`（既定 1）、`progress_key`（既定 `logical_step`） |
| `TopKSelector` | Latest の規則は同じ。加えて境界の指標が良い順に `k` 個の Archive。`latest_every` を省くと 1 のまま | `metric`、`mode`（`min` / `max`）、`k`、`name`（既定 `best_{step:08d}.pt`）、`latest_every`（既定 1）、`progress_key`（既定 `logical_step`） |

`TopKSelector` の `metric` が `loss` や `pesq` のように名前空間を持たないときは、順位付けは境界だけである。`validation.wer` のように `.` を含む名前は、そのキーが付いた step と final も順位付けする。研究用の `research_selection/best.pt` は checkpoint ではなく、Latest や Archive からの再開には使わない。

YAML は Selector とその設定だけを書く。Trainer に `checkpoint_interval` は無い。

```yaml
checkpoint:
  selector:
    class: src.experiments.selectors.StandardSelector
    kwargs:
      latest_every: 5000
```

`checkpoint` を省くと `StandardSelector` で、Latest は 1000 logical step ごと、各境界、安全な停止、完了で置き換える。毎 step 保存は `latest_every: 1` と書いたときだけである。`checkpoint` は実験の同一性（`config_sha256`）に入らない。再開では別途保存した Selector のクラス ID、設定、状態を照合し、クラスまたは設定が違う場合は拒否する。

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

Trainer の検証履歴と Selector・停止ポリシーの状態は checkpoint から復元する。記録された評価状態に整合させるため、停止した logical step より後の `metrics.jsonl` の行は再開時に巻き戻す。

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

## 公開パスとクラス識別

現在の仕様では YAML の `class` に安定して維持する公開 import パスを書く。旧 YAML との互換性は保証しない。Experiment の共通構築、実行、Trainer、損失は `src.experiments` 直下の公開モジュール、データ供給は `src.experiments.data`、公式モデルと Trainer は `src.experiments.official.<model>.model` / `models` / `trainer`、共有 scheduler は `src.experiments.calc.optim`、外部ライブラリはその公開パスを使う。公式モジュールは公開モジュールに直接実装を置いている。推論は `src.se` と `src.asr`、評価は `src.evaluation` の API と CLI を使う。公開モジュール上の monkeypatch は実装モジュールのグローバル参照を書き換えない。

Selector と StopPolicy は具象クラス自身に `checkpoint_class_id` を宣言する。ID は YAML に指定する公開クラスパスと一致させ、実装の移動では変えない。継承による暗黙の ID 再利用、無効な ID、同じプロセス内で異なるクラスによる ID の重複を拒否する。Selector は ID・設定・状態を保存し、StopPolicy は ID と状態を TrainingControl の状態に含める。保存前に rank 間でも照合する。Model、Optimizer、Scheduler、Trainer、DataModule のクラスは設定の公開パスと `config_sha256` で識別し、各 state ID と state 構造を検証する。`output`、`checkpoint` と `stop_after_logical_steps` はハッシュ対象外で、Selector の設定は別途照合する。

形式は `audio-db-experiment-v1` のままである。旧クラスパスや旧 Trainer 引数に基づく設定ハッシュ、Selector 情報のない checkpoint、以前の実装パスを Selector ID として保存した checkpoint は再開できない。停止ポリシー ID のない停止状態も再開できない。これらを自動的に初期化として扱わない。`initialize` は module・Optimizer・scheduler 状態の読み込みであり、再開とは別である。`audio-db-training-v1` は読まない。来歴が欠けた checkpoint の扱いは従来どおり UNKNOWN のままである。

## 未実装

FSDP、sharded checkpoint、AMP BF16。複数 Optimizer の `amp_fp16` は拒否する。
