# Experiment Framework 公開 API

Phase 1 の公開入口、Phase 2-A の CPU checkpoint、Phase 2-B の CPU / Gloo DDP、Phase 2-C の CUDA / NCCL / AMP FP16 は `src/experiments/` にある。`Experiment.run(resume=...)` と `--initialize-from` は CPU / FP32、CUDA / FP32、CUDA / AMP FP16 で動作する。world size 2 は `torchrun` の `RANK`、`LOCAL_RANK`、`WORLD_SIZE` で動き、CPU は Gloo、CUDA は NCCL である。設定と起動環境が食い違うときは単一プロセスや CPU へ落とさない。FSDP と `amp_bf16` は `PrepareError` になる。`audio-db-training-v1` は `CheckpointError` になる。以下は Phase 0 の契約であり、現行 `src/training` の関数ではない。実装との差分は [PHASE_2B_DDP.md](PHASE_2B_DDP.md) と [PHASE_2C_CUDA_AMP.md](PHASE_2C_CUDA_AMP.md)。

例外の基底は `src.experiments.errors.ExperimentError`。呼び出し側が回復できる失敗はすべてこの基底に乗せる。`ImportError`、`TypeError`、`ValueError` をそのまま外へ出さない。

| 例外 | 条件 |
|---|---|
| `ConfigError` | YAML の構文、未知キー、型、`role` の不正 |
| `BuildError` | 未定義参照、循環、引数名の衝突、`parameters_of` の対象が module ではない |
| `PrepareError` | device、process group、DDP / FSDP の準備に失敗した |
| `StateError` | 二重登録、未登録 ID、`resume` と現在グラフの状態 ID 不一致 |
| `CheckpointError` | 形式違い、原子的保存の検証失敗、world size 不一致、破損 |
| `RuntimeFailure` | rank 間で skip が一致しない、AMP の段取り違反。組み込み `RuntimeError` とは別名 |
| `DataError` | snapshot や batch 仕様が供給できない。`src.experiments.errors.DataError`。`setup` の失敗は Builder が `BuildError` で包む |

## CLI

現行の `python -m src.training.se.cli` は残す。新入口は並立する。

```text
python -m src.experiments.cli --config PATH
    [--resume PATH]
    [--initialize-from PATH]
    [--preflight]
    [--preflight-dir PATH]
```

| 引数 | 意味 |
|---|---|
| `--config` | `audio-db-experiment/v1` の YAML。必須 |
| `--resume` | 新形式 checkpoint。logical step、全状態、RNG、データ位置を戻して続ける |
| `--initialize-from` | 指名した重みだけを載せる。`--resume` と同時には指定できない |
| `--preflight` | 構築と検証だけを行い、`run` しない。プロセスグループを作った場合は破棄する |
| `--preflight-dir` | `preflight.json` の出力先。省略時は設定の `output.directory` |

終了コードは、preflight の失敗または `ExperimentError` で 1、完了で 0。学習の成否を `metrics.jsonl` の最終行以外の暗黙の規約にしない。

preflight が確認すること:

- 未知キーが無い
- 参照が解決し、循環が無い
- `entry` の Trainer が存在する
- `parameters_of` の相手が `role: module` である
- `resume` を渡したときは、checkpoint の状態 ID 集合と schema version が現在グラフと一致する
- 重みの読み込みと step 実行はしない

## Python Entry API

```python
def load_experiment(path: str | Path) -> ExperimentConfig:
    """YAML を読み、スキーマを検証した設定を返す。

    インスタンスは作らない。ファイルを書けない。
    未知キー、schema 違い、role 不正は ConfigError。
    """

class Experiment:
    def __init__(self, config: ExperimentConfig) -> None:
        """設定を保持するだけである。構築は build または run が行う。"""

    def build(self) -> Experiment:
        """Builder に構築、準備、Optimizer 生成、状態登録を行わせる。

        二度目の build は StateError。
        戻り値は self。
        """

    def run(
        self,
        *,
        resume: str | Path | None = None,
        initialize_from: str | Path | None = None,
    ) -> RunResult:
        """未構築なら build する。resume または initialize_from のあと Trainer を実行する。

        両方を渡すと ConfigError。
        完了時に Finalize まで進み、RunResult に logical_step、停止理由、出力ディレクトリを入れる。
        例外時も Finalize を試み、プロセスグループを破棄する。破棄に失敗したら元の例外に連鎖させる。
        """

    def get(self, component_id: str) -> object:
        """ComponentRegistry のオブジェクトを返す。

        module は準備後ハンドル。未構築、未知 ID は BuildError。
        状態は変えない。
        """

def build_experiment(config: ExperimentConfig) -> Experiment:
    """Experiment を作り build まで行う。run は呼ばない。"""
```

`RunResult` は凍結データクラスとする。フィールドは `stop_reason: str`、`logical_step: int`、`optimizer_updates: dict[str, int]`、`output_dir: Path`。

## Component Factory と Builder

```python
class ComponentFactory:
    def create(self, spec: ComponentSpec) -> object:
        """spec.class_path を import し、spec.kwargs だけでインスタンス化する。

        inject と parameters_of は見ない。解決済みの引数は Builder が kwargs に含めて渡す。
        クラスが読めない、呼び出しが失敗した場合は BuildError。
        登録表は更新しない。
        """

class ExperimentBuilder:
    def build(self, config: ExperimentConfig, runtime: Runtime) -> BuiltGraph:
        """名前付きコンポーネントを段の順に作る。

        未定義参照、循環、kwargs と inject のキー衝突は BuildError。
        runtime.prepare_model を role: module の各要素に呼ぶ。
        optimizer は準備後の parameters ビューで作る。
        StateRegistry へ登録する。
        学習ステップは実行しない。
        """
```

`BuiltGraph` が持つもの:

- `components: ComponentRegistry`
- `states: StateRegistry`
- `trainer: Trainer`（`config.entry` が指すインスタンス）
- `runtime: Runtime`

Factory と Builder の境界: Factory は 1 オブジェクトの生成だけを知る。Builder は名前、参照、順序、共有、Runtime、StateRegistry を知る。Factory に Registry を渡さない。

## Trainer と Runtime

```python
class StepResult:
    applied: bool
    """全 rank で同じ値。AMP overflow または Trainer が事前に skip を要求したとき False。"""

class Runtime:
    def prepare_model(self, module: torch.nn.Module, policy: PreparePolicy) -> PreparedModel:
        """device へ移し、policy に従って DDP で包む。

        trainable が False のとき requires_grad をすべて False にし、distribute が none なら包まない。
        戻り値の parameters は Optimizer が掴むビュー。
        準備に失敗すると PrepareError。モジュールの置換以外の学習状態は持たない。
        """

    def sync_context(self, modules: list[PreparedModel], *, sync: bool) -> AbstractContextManager[None]:
        """forward の前に入る。sync が False の DDP モジュールは no_sync になる。

        コンテキストを出たあとで同期可否は変わらない。
        """

    def autocast(self) -> AbstractContextManager[None]:
        """設定された precision の autocast。fp32 のときは何もしないコンテキスト。"""

    def backward(self, loss: torch.Tensor, *, retain_graph: bool = False) -> None:
        """Scaler があるときは scale してから backward する。

        同期の選択は行わない。loss が非有限なら RuntimeFailure とし、rank ごとに backward の有無を分けない。
        """

    def clip(
        self,
        parameters: Iterable[torch.nn.Parameter],
        max_norm: float,
        *,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> float:
        """AMP は optimizer を unscale してから clip_grad_norm_ を行う。

        返す値は unscale 後、clip 前の norm である。FP32 は optimizer を無視する。
        同じパラメータ集合に対する二度目の unscale は RuntimeFailure。
        """

    def step(self, optimizer: torch.optim.Optimizer) -> StepResult:
        """optimizer.step を全 rank で同じ適用結果にする。

        GradScaler.step は呼ばない。applied は、optimizer.step を包んだ関数が呼ばれたときだけ True である。
        SGD と Adam は更新しても None を返す。その None を skip とみなさない。
        AMP は found_inf を all-reduce の MAX で揃え、いずれかの rank が溢れたら全 rank が applied False。
        fp32 は agree(True, identical) のあと更新する。
        このメソッドは scaler.update を呼ばない。zero_grad もしない。
        """

    def end_step(self) -> None:
        """論理 step の終わりに Scaler.update を 1 回呼ぶ。fp32 では何もしない。"""

    def average_gradients(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        """明示されたパラメータの grad を rank 間で平均する。

        DDP の自動 all-reduce の代わりに Trainer が呼ぶ。world size 1 では何もしない。
        """

    def agree(self, flag: bool, *, policy: str = "identical") -> bool:
        """bool を全 rank で揃える。

        identical は不一致を RuntimeFailure にする。any は OR、all は AND を返す。
        戻り値は全 rank で同じである。
        """

class PreparedModel:
    component_id: str
    module: torch.nn.Module
    parameters: Iterable[torch.nn.Parameter]
    state_module: torch.nn.Module

class Trainer(Protocol):
    def run(self, runtime: Runtime, checkpoint: CheckpointManager) -> RunResult:
        """学習ループを所有する。Runtime と CheckpointManager 以外の実行環境を自分で初期化しない。"""
```

`PreparePolicy` のフィールドは `trainable: bool`、`distribute: Literal["ddp", "none"]`、`find_unused_parameters: bool`、`broadcast_buffers: bool`、`sync_batchnorm: bool`。`distribute: fsdp` は Phase 1 から 4 では `PrepareError`。

契約:

- Trainer は `optimizer.step()` を呼ばない。`runtime.step` の `applied` が True のときだけ、その Optimizer の update count を 1 増やす。
- Scheduler を進めるのは Trainer である。`applied` が False の更新では、その Optimizer に紐づく per-step Scheduler を進めない。epoch Scheduler は Trainer が epoch 境界で明示的に進める。
- 損失を蓄積回数で割るのは Trainer である。Runtime は割らない。
- 検証ループは Trainer が持つ。検証 forward は `torch.no_grad` と、precision を fp32 に固定した autocast 無効の中で行う。
- checkpoint を書いてよい境界は、`runtime.end_step` の後、かつ次の学習サンプルを消費する前である。検証を挟むときは、検証の前に `validation_pending` を付けた checkpoint を書いてから検証し、検証後にもう一度書く。現行 Trainer と同じ二段である。
- logical step は、Trainer が 1 回のアルゴリズム反復を終えたときに 1 増える。その反復で Optimizer が 0 回でも 2 回でも、logical step の増分は 1 である。スキップして反復自体を無かったことにする場合だけ、Trainer は logical step を進めない。MossFormerGAN の PESQ skip は後者に合わせる。データ cursor は進んでよい。

`StandardTrainer` の動作は次に固定する。Phase 1 でこれを実装する。

1. DataModule から micro-batch を `accumulation` 個取る。
2. 最後以外は `sync_context(..., sync=False)`、最後は `sync=True` の中で forward する。
3. 損失を accumulation で割り、`backward` する。
4. 設定に `grad_clip` があれば `clip` する。
5. 単一 Optimizer を `step` する。
6. `end_step` する。
7. `applied` が True のときだけ update count と、必要なら per-step Scheduler を進める。
8. logical step を 1 進める。`applied` が False でも AMP overflow の反復は消費したデータとして logical step を進める。PESQ のような「更新しないがデータは進む」判定は `StandardTrainer` に入れない。

## Data API

Phase 1 から 2-C の実装はこの節の `TensorPairData` である。音声の Source、Sampler、Protocol、Batch、identity、`plan_hash`、YAML は [DATA_PUBLIC_API.md](DATA_PUBLIC_API.md) を正とする。Phase 3-A は、既存の供給クラスをアダプタで包む案を撤回した。Trainer から見たメソッド名は、この節と音声契約で同じである。

```python
class DataModule(Protocol):
    def identity(self) -> dict[str, Any]:
        """データ集合の同一性。順序乱数の状態は含めない。"""

    def setup(self, runtime: Runtime) -> None:
        """rank と world size を受け取り、ファイルを開く。DuckDB は開かない。"""

    def train_epoch(self, epoch: int, *, start_update: int) -> EpochView:
        """この rank の更新列。start_update より前は返さない。"""

    def validation_epoch(self, epoch: int) -> Iterator[Batch]:
        """検証バッチ。rank の分担は実装が明示する。"""

    def state_dict(self) -> dict[str, Any]:
        """再開に必要な位置。先読み未使用分は含めない。"""

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """identity と world size が違う状態は StateError。"""

    def close(self) -> None:
        """ファイルとワーカーを閉じる。二度呼んでも成功する。"""
```

`EpochView` は `updates: int`、`plan_hash: str`、`batch(update_index: int) -> list[Batch]` を持つ。`batch` は cursor を内部に持たず、同じ index は同じ内容を返す。現行 `EpochStream.render` と同じ契約である。

Sample identity はバッチに `sample_id: str` として入る。Realization identity は `realization_id: str` として入る。固定ペアで crop が epoch に依存するときは、両者は異なる。オンライン混合で波形が `(seed, epoch, index)` の純関数であるとき、Sampler の乱数状態は `state_dict` に入れない。

学習結果で Sampler の分布を変える処理は、Trainer が DataModule の `set_sampling_weights(mapping)` を呼ぶ形にする。Phase 3-F より前の DataModule はこのメソッドを持たない。持つ実装は、変更後の重みを `state_dict` に含め、含めない状態からの `resume` は `StateError` にする。平坦 policy 以外には掛けない。旧文の `update_weights(sample_id, weight)` はこのメソッドに置き換えた。

## StateRegistry と Checkpoint

```python
class StateRegistry:
    def register(self, owner: StateOwner) -> None:
        """state_id で登録する。同じ Python オブジェクトの再登録は無視する。

        別オブジェクトが同じ state_id を主張すると StateError。
        run 開始後の register は StateError。
        """

    def ids(self) -> tuple[str, ...]:
        """登録順ではなく、state_id の辞書順。"""

class StateOwner(Protocol):
    state_id: str
    def state_dict(self) -> dict[str, Any]: ...
    def load_state_dict(self, state: dict[str, Any]) -> None: ...

class CheckpointManager:
    def save(self, directory: Path, *, logical_step: int, label: str) -> Path:
        """登録状態、RNG、manifest を 1 ファイルへ原子的に書く。

        rank 0 だけが書く。全 rank は書き込みの前後で barrier する。
        戻り値は rank 0 では出力パス、他 rank では同じパス文字列を持つ Path。
        """

    def restore(self, path: Path, *, mode: Literal["resume", "initialize"]) -> RestoreReport:
        """resume は全状態と RNG とデータ位置を戻す。

        initialize は manifest の initialize_keys に含まれる state_id の、
        さらに allow リストにあるテンソルだけを書く。
        Optimizer、Scheduler、RNG、logical step、データ位置は初期のまま。
        形式が audio-db-experiment-v1 でなければ CheckpointError。
        """
```

`label` はファイル名の接頭辞に使う。`step` または `init`。検証待ちのファイルは `label="validation_pending"` を manifest に記録する。

詳細な manifest は [STATE_AND_CHECKPOINT.md](STATE_AND_CHECKPOINT.md)。

## YAML の構文と意味

トップレベルは次だけを許す。他のキーは `ConfigError`。

| キー | 必須 | 意味 |
|---|---|---|
| `schema` | はい | 文字列 `audio-db-experiment/v1` |
| `seed` | はい | 0 以上の整数。初期化と、resume しないときの実行 RNG |
| `deterministic` | はい | bool。現行 `set_determinism` と同じ意味を Runtime が適用する |
| `entry` | はい | Trainer の component id |
| `runtime` | はい | Runtime の class、kwargs |
| `components` | はい | component id から spec への対応 |
| `output` | はい | `directory` のみ。checkpoint と `metrics.jsonl` の場所 |

component id は `[A-Za-z_][A-Za-z0-9_]{0,63}`。

spec は次のいずれかである。

別名:

```yaml
generator_view:
  ref: generator
```

実体:

```yaml
generator:
  role: module
  class: torch.nn.Linear
  kwargs:
    in_features: 4
    out_features: 4
  prepare:
    trainable: true
    distribute: none
```

`role` は `module`、`optimizer`、`scheduler`、`trainer`、`data`、`loss`、`metric`、`other` のいずれか。

| フィールド | 意味 |
|---|---|
| `class` | 完全修飾クラス名。Factory が import する |
| `kwargs` | そのクラスのコンストラクタへそのまま渡す辞書。ネストした `ref` は禁止。参照は `inject` に書く |
| `inject` | 引数名から参照への対応。値は `{ref: id}` またはそのリスト |
| `parameters_of` | Optimizer だけ。component id、または `{component, lr, weight_decay}` のリスト |
| `prepare` | module だけ。`PreparePolicy` のフィールド |
| `state` | `save: false` のとき StateRegistry に載せない。省略時は module、optimizer、scheduler、trainer、data を載せる。loss と metric は載せない |

衝突: `kwargs` と `inject` に同じキーがあれば `BuildError`。`parameters_of` は `kwargs` の `params` と同時に書けない。

循環: `inject` の `ref` と `parameters_of` の component を辺として深さ優先で検出する。別名 `ref` も辺である。自己参照は循環である。

生成順の詳細は [COMPONENT_LIFECYCLE.md](COMPONENT_LIFECYCLE.md)。

`runtime` の形:

```yaml
runtime:
  class: src.experiments.runtime.ProcessRuntime
  kwargs:
    device: cpu
    precision: fp32
```

`precision` は `fp32` または `amp_fp16`。`amp_bf16` は Phase 2 のテストが書いてから追加する。未知の値は `ConfigError`。

`device: cuda` でもプロセスグループは `WORLD_SIZE > 1` のときだけ作る。現行 `setup_process` と同じ環境変数 `WORLD_SIZE`、`RANK`、`LOCAL_RANK` を読む。

## 実験例

### 単一モデル、単一 Optimizer

`loss` を Trainer と、ログ用の別名から共有する。インスタンスは 1 つである。

```yaml
schema: audio-db-experiment/v1
seed: 17
deterministic: true
entry: trainer
runtime:
  class: src.experiments.runtime.ProcessRuntime
  kwargs:
    device: cpu
    precision: fp32
output:
  directory: runs/linear-supervised
components:
  model:
    role: module
    class: torch.nn.Linear
    kwargs: {in_features: 8, out_features: 8}
    prepare: {trainable: true, distribute: none}
  loss:
    role: loss
    class: torch.nn.MSELoss
  loss_for_log:
    ref: loss
  data:
    role: data
    class: src.experiments.data.TensorPairData
    kwargs:
      features: 8
      examples: 32
      batch_size: 4
  optimizer:
    role: optimizer
    class: torch.optim.AdamW
    parameters_of: model
    kwargs: {lr: 0.001}
  trainer:
    role: trainer
    class: src.experiments.trainers.StandardTrainer
    kwargs: {max_logical_steps: 4, accumulation: 1}
    inject:
      model: {ref: model}
      loss: {ref: loss}
      data: {ref: data}
      optimizer: {ref: optimizer}
```

### GAN

更新順は `GanTrainer` の Python にある。YAML は二つのモジュールと二つの Optimizer を置くだけである。

```yaml
schema: audio-db-experiment/v1
seed: 17
deterministic: true
entry: trainer
runtime:
  class: src.experiments.runtime.ProcessRuntime
  kwargs: {device: cpu, precision: fp32}
output:
  directory: runs/gan-linear
components:
  generator:
    role: module
    class: torch.nn.Linear
    kwargs: {in_features: 8, out_features: 8}
    prepare: {trainable: true, distribute: none}
  discriminator:
    role: module
    class: torch.nn.Linear
    kwargs: {in_features: 8, out_features: 1}
    prepare: {trainable: true, distribute: none}
  data:
    role: data
    class: src.experiments.data.TensorPairData
    kwargs: {features: 8, examples: 32, batch_size: 4}
  optim_g:
    role: optimizer
    class: torch.optim.AdamW
    parameters_of: {component: generator, lr: 0.0001}
  optim_d:
    role: optimizer
    class: torch.optim.AdamW
    parameters_of: {component: discriminator, lr: 0.0002}
  trainer:
    role: trainer
    class: src.experiments.trainers.GanTrainer
    kwargs: {max_logical_steps: 2}
    inject:
      generator: {ref: generator}
      discriminator: {ref: discriminator}
      data: {ref: data}
      optim_g: {ref: optim_g}
      optim_d: {ref: optim_d}
```

`GanTrainer` は Phase 1 の必須実装には含めない。この YAML は Builder がグラフを組めることの仕様例である。Phase 1 の受け入れは、この YAML に対して `build` が二つの Optimizer を別パラメータへ繋ぐところまでとする。`run` は `StandardTrainer` の例で確認する。

### 固定 ASR を持つ ASR-aware SE

ASR は学習しない。`distribute: none` なので DDP で包まない。Trainer は SE の損失に加え、`inference_mode` の認識結果をログしてよい。認識結果を損失の 0 として逆伝播へ入れない。

```yaml
schema: audio-db-experiment/v1
seed: 17
deterministic: true
entry: trainer
runtime:
  class: src.experiments.runtime.ProcessRuntime
  kwargs: {device: cpu, precision: fp32}
output:
  directory: runs/asr-aware-toy
components:
  enhancer:
    role: module
    class: torch.nn.Linear
    kwargs: {in_features: 16, out_features: 16}
    prepare: {trainable: true, distribute: none}
  asr:
    role: module
    class: src.experiments.adapters.FrozenRecognizer
    kwargs:
      name: whisper
      model_id: large-v3
      device: cpu
      differentiable: false
    prepare: {trainable: false, distribute: none}
  loss:
    role: loss
    class: torch.nn.MSELoss
  data:
    role: data
    class: src.experiments.data.TensorPairData
    kwargs: {features: 16, examples: 8, batch_size: 2}
  optimizer:
    role: optimizer
    class: torch.optim.AdamW
    parameters_of: enhancer
    kwargs: {lr: 0.001}
  trainer:
    role: trainer
    class: src.experiments.trainers.AsrAwareTrainer
    kwargs: {max_logical_steps: 1, asr_every: 1}
    inject:
      enhancer: {ref: enhancer}
      asr: {ref: asr}
      loss: {ref: loss}
      data: {ref: data}
      optimizer: {ref: optimizer}
```

`differentiable: true` は Phase 5 まで `ConfigError`。`FrozenRecognizer` の実チェックポイントを使う試験は、重みが存在する環境だけが行い、無いときは skip する。skip は成功ではない。Phase 1 は `FrozenRecognizer` の代わりに、同じ `prepare` を持つダミーモジュールでグラフ構築を確認してよい。その確認を公式 ASR の接続検証と呼ばない。

同一の `asr` を二つ目の Trainer や評価用 metric から `ref` するとき、インスタンスは 1 つである。認識器をコンポーネントとして二回 `class:` で書くと、重みが二重にロードされる。共有したい場合は `ref` を使う。

### 複数 Expert と Router

Router と Expert の選択アルゴリズムは `MixtureTrainer` に書く。YAML は Expert のリストを `inject` で渡す。

```yaml
schema: audio-db-experiment/v1
seed: 17
deterministic: true
entry: trainer
runtime:
  class: src.experiments.runtime.ProcessRuntime
  kwargs: {device: cpu, precision: fp32}
output:
  directory: runs/mixture-linear
components:
  expert_a:
    role: module
    class: torch.nn.Linear
    kwargs: {in_features: 8, out_features: 8}
    prepare: {trainable: true, distribute: none}
  expert_b:
    role: module
    class: torch.nn.Linear
    kwargs: {in_features: 8, out_features: 8}
    prepare: {trainable: true, distribute: none}
  router:
    role: module
    class: torch.nn.Linear
    kwargs: {in_features: 8, out_features: 2}
    prepare: {trainable: true, distribute: none}
  data:
    role: data
    class: src.experiments.data.TensorPairData
    kwargs: {features: 8, examples: 16, batch_size: 4}
  optim_experts:
    role: optimizer
    class: torch.optim.AdamW
    parameters_of:
      - {component: expert_a, lr: 0.001}
      - {component: expert_b, lr: 0.001}
  optim_router:
    role: optimizer
    class: torch.optim.AdamW
    parameters_of: router
    kwargs: {lr: 0.0005}
  trainer:
    role: trainer
    class: src.experiments.trainers.MixtureTrainer
    kwargs: {max_logical_steps: 2}
    inject:
      experts:
        - {ref: expert_a}
        - {ref: expert_b}
      router: {ref: router}
      data: {ref: data}
      optim_experts: {ref: optim_experts}
      optim_router: {ref: optim_router}
```

Expert を学習中に増やす記法は無い。Phase 5 で `dynamic` 枠を足すまで、このリストが起動時の全 Expert である。

## エラー処理とバリデーション

Builder は最初のエラーで止める。循環と未定義が両方あるときは、未定義を先に報告する。メッセージには component id と、参照しようとした id を含める。

`parameters_of` が指す module の `prepare.trainable` が false のときは `BuildError`。凍結パラメータを Optimizer に渡さない。

同じパラメータオブジェクトが二つの Optimizer の `parameters_of` に含まれるときは `BuildError`。意図して共有したい場合は、Phase 5 まで許可しない。現行の GAN も Generator と Discriminator でパラメータは分かれている。

`entry` が `role: trainer` でなければ `ConfigError`。Trainer が 0 個でも `ConfigError`。

出力ディレクトリが `data/checkpoints`、`data/corpora`、`data/db`、`data/staging` の下にあるときは `CheckpointError`。現行 `assert_output_allowed` と同じ禁止である。
