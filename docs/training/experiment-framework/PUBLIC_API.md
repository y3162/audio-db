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

    def clip_repeated(self, parameters: Iterable[torch.nn.Parameter], max_norm: float) -> float:
        """同じパラメータを micro-batch ごとに clip する。

        ``clip`` の一度きりの記録と AMP unscale は使わない。``amp_fp16`` は RuntimeFailure。
        返す値は clip 前の総ノルムである。MossFormerGAN が使う。
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
        欠けた勾配はゼロとして参加する。
        """

    def average_new_gradients(
        self,
        parameters: Iterable[torch.nn.Parameter],
        baselines: Mapping[int, torch.Tensor | None],
    ) -> None:
        """この backward で増えた勾配だけを rank 間で平均し、baselines の蓄積へ足す。

        勾配が None の Parameter は None のままである。ゼロテンソルは作らない。
        world size 1 では grad を書き換えない。``average_gradients`` の意味は変えない。
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

## MP-SENet の波形アダプタ

Phase 4-A のクラスは `src.experiments.models.mp_senet.MPSENetWaveformAdapter` である。`torch.nn.Module` で、`forward` は `[batch, time]` の波形を受け、同じ形状の波形を返す。中で `src.se.build_model("mp_senet")` の MPNet を 1 組だけ持ち、`mag_pha_stft` と `mag_pha_istft` を呼ぶ。Loss、Optimizer、データ取得は持たない。`enhance()` と `load_enhancer()` は呼ばない。

checkpoint の module state では、Generator のキーが `generator.` で始まる。Optimizer は `parameters_of` でこの module の Parameter を受け取る。別のモジュールへ同じ Parameter を登録しない。

iSTFT の長さが入力と違うときは `RuntimeFailure` で、暗黙の crop や pad はしない。`sampling_rate` は 16000 以外を `BuildError` にする。入力と教師の sample rate は、固定ペアの rule が両方を同じ rate で描く契約であり、Adapter は教師波形を受け取らない。`StandardTrainer` は `model(batch.inputs)` と `loss(prediction, batch.targets)` のままである。MP-SENet 専用の分岐は Trainer に無い。

この L1 は公式の複合損失ではない。設定例は `configs/experiments/mp_senet_l1.yaml`。検証は [Phase 4-A](PHASE_4A_MP_SENET_CONNECTION.md)。

Phase 4-D の公式 GAN は別クラスである。Generator は `src.experiments.models.mp_senet_gan.MPSENetGANGenerator`、Discriminator は `src.experiments.models.mp_senet_gan.MPSENetMetricDiscriminator`、学習ループは `src.experiments.mp_senet.MPSENetGANTrainer`。Loss 関数は現行の `generator_losses` と `discriminator_loss` を Trainer が呼ぶ。`StandardTrainer` と `GanTrainer` には MP-SENet の分岐を足していない。`amp_fp16` はこの Trainer が拒否する。設定例は `configs/experiments/mp_senet_gan.yaml`。検証は [Phase 4-D](PHASE_4D_MP_SENET_GAN.md)。

## MossFormerGAN se_16k

Phase 4-E のクラスは `src.experiments.models.mossformergan.MossFormerGANGenerator`、`src.experiments.models.mossformergan.MossFormerGANDiscriminator`、`src.experiments.mossformergan.MossFormerGANTrainer` である。Factory はクラスパスを呼ぶだけで、モデル名では分岐しない。Generator は `src.se.build_model("mossformergan", variant="se_16k")` の SyncANet を 1 組、Discriminator は既存の NISQA `Discriminator` を 1 組持つ。ネットワークは再実装しない。

初期化は Builder の順に依存しない。Generator は G1 だけを作る。Discriminator は G1 から G4 と D1 から D3 を作って捨て、D4 を残す。捨てたモジュールは登録しない。

学習順は MP-SENet と違う。Generator Loss の backward、増分平均、Generator の clip、Discriminator Loss の backward、増分平均、Discriminator の clip が終わってから、蓄積窓が埋まったときだけ Generator の AdamW、続いて Discriminator の AdamW を `runtime.step` する。PESQ 失敗は backward しない。data cursor と `official_step` は進み、`logical_step` と `accu_count` は進まない。clip は `clip_repeated` で micro-batch ごとである。`runtime.clip` の一度きりは残す。

検証はデータ epoch の末尾である。Generator は `eval()`、Discriminator は `train()`。欠測 Loss は和に足さず、分母には残す。集計は rank 平均の等重み平均である。plateau は `mossformer_validation_decision` で、5 回目の非改善が Generator の学習率だけを半分にし、10 回目で停止する。Scheduler コンポーネントは作らない。

`amp_fp16` は拒否する。設定例は `configs/experiments/mossformergan_se_16k.yaml`。`max_logical_steps` は 1 で、公式規模の学習は起動しない。検証は [Phase 4-E](PHASE_4E_MOSSFORMERGAN_TRAINING.md)。4 モデルの横断検証は [Phase 4-F](PHASE_4F_INTEGRATION.md)。実験 checkpoint は `data_provenance.json` を書かず、`provenance_sha256` は None のままである。None 以外の resume は拒否する。評価側は欠落を UNKNOWN とし、独立とは判定しない。

## TF-GridNet MS-SNSD

Phase 4-B のクラスは `src.experiments.models.tf_gridnet.TFGridNetModel` と `src.experiments.losses.MultiResolutionL1Loss` と `src.experiments.tf_gridnet.TFGridNetTrainer` である。Factory はクラスパスを呼ぶだけで、モデル名では分岐しない。

`TFGridNetModel` は `variant="ms-snsd"` の `src.se.build_model("tf_gridnet")` を 1 組だけ持つ。構築は `seed_model_init` の中で行い、戻る前に `apply_espnet_initialization` の `xavier_uniform` を掛ける。Optimizer はその後に同じ Parameter を受け取る。`urgent24` は `BuildError`。`forward` は `[batch, time]` を受け、同じ形状の有限な波形を返す。長さが `n_fft // 2` 以下、または出力長が違うときは `RuntimeFailure` で、crop も pad もしない。

`MultiResolutionL1Loss.forward(estimate, target)` は既存の `multi_resolution_l1` を、窓 `(256, 512, 768, 1024)`、時間重み 0.5、`normalize_variance=False` で呼ぶ。`normalize_variance=True` は `BuildError`。SI-SDR は足さない。

`TFGridNetTrainer` は `StandardTrainer` の特殊ケースではない。lifecycle は次である。

1. `plan_chunk_epoch` の micro-batch 列を 1 epoch とする。固定の Optimizer 回数ではない。
2. 各 logical step は train、`zero_grad`、蓄積、`runtime.backward`、`runtime.clip`、`runtime.step`、`runtime.end_step` である。`optimizer.step()` は呼ばない。
3. epoch の途中で `max_logical_steps` に達しても、validation、scheduler、n-best は走らない。checkpoint の label は `step` である。
4. epoch の最後の更新のあと、label `validation_pending` の checkpoint を書く。このファイル名は閉じた `step_XXXXXXXX.pt` と同じである。
5. `model.eval()` と `torch.no_grad()` で検証する。学習 cursor と学習 RNG は進めない。forward は `state_module` で、DDP の forward では all-reduce しない。
6. 検証損失は batch 平均に波形数を掛けた和を、全 rank の件数で割る。rank 平均の平均は使わない。
7. そのスカラーで `ReduceLROnPlateau.step` を 1 回だけ呼ぶ。Optimizer 更新のたびには進めない。
8. `valid.loss` の最小だけで n-best を書く。`valid.si_snr` は選択に使わない。平均は `average_state_dicts`（浮動小数は平均、整数は和）である。
9. 同じ logical step のファイルを label `step` で置き換える。検証、scheduler、選択が揃う前の別ファイルは作らない。再開時の二重 step を避けるためである。
10. `validation_pending` からの再開は、ループの前に 5 から 9 を一度だけやり直す。Optimizer 更新はやり直さない。

`group_size` が `accumulation` と違うときは `RuntimeFailure`。early stopping の既定 patience は None で、宣言 YAML は 5。停止理由は `completed` である。設定例は `configs/experiments/tf_gridnet_ms_snsd.yaml`。module の `distribute` は `ddp` で、`global_batch` 4 と `micro_batch` 2 は world size 2 を要求する。world size 1 は `global_batch` 不一致で拒否する。world size 1 は DDP で包まない。この YAML の `max_logical_steps` は 1 で、公式規模の学習は起動しない。検証は [Phase 4-B](PHASE_4B_TF_GRIDNET_TRAINING.md) と [Phase 4-F](PHASE_4F_INTEGRATION.md)。

## TF-Locoformer DNS

Phase 4-C のクラスは `src.experiments.models.tf_locoformer.TFLocoformerModel` と、既存の `MultiResolutionL1Loss` と、`src.experiments.tf_locoformer.TFLocoformerTrainer` である。Factory はクラスパスを呼ぶだけで、モデル名では分岐しない。Scheduler は既存の `src.training.se.optim.WarmupReduceLROnPlateau` をそのまま使う。

`TFLocoformerModel` は `variant="dns"` の `src.se.build_model("tf_locoformer")` を 1 組だけ持つ。構築は `seed_model_init` の中で行い、戻る前に `apply_espnet_initialization` の `xavier_uniform` を掛ける。Optimizer はその後に同じ Parameter を受け取る。`forward` は `[batch, time]` を受け、同じ形状の有限な波形を返す。長さが `n_fft // 2` 以下、無音、分散が極端に小さい入力は `RuntimeFailure` で、モデル本体の拒否を維持する。モデル内部の分散正規化は true のままで、Loss の `normalize_variance` は false のままである。

`TFLocoformerTrainer` は `TFGridNetTrainer` の Scheduler 更新を流用しない。lifecycle は次である。

1. `plan_chunk_epoch` の micro-batch 列を 1 epoch とする。
2. 各 logical step は train、`zero_grad`、蓄積、`runtime.backward`、`runtime.clip`、`runtime.step`、`runtime.end_step` である。`optimizer.step()` は呼ばない。
3. `runtime.step` の `applied` が true のときだけ `WarmupReduceLROnPlateau.step_batch` を呼ぶ。最初の更新はコンストラクタの学習率を使う。`step_batch` は次の更新の学習率を warmup 式で書く。
4. epoch の途中で `max_logical_steps` に達しても、validation、`step_validation`、n-best は走らない。
5. epoch の最後の更新と `step_batch` のあと、label `validation_pending` の checkpoint を書く。
6. `model.eval()` と `torch.no_grad()` で検証する。学習 cursor と学習 RNG は進めない。Loss と SI-SNR は、波形数で重み付けした全 rank の和を件数で割る。
7. 学習ログの `si_sdr` は負の SI-SDR loss である。Validation の `si_snr` は SI-SDR（dB）である。同じ数として比較しない。
8. warmup 中も含め、検証損失で `step_validation` を 1 回呼ぶ。この呼び出しは `step_num` を進めない。warmup 中に plateau が学習率を下げても、次の `step_batch` が warmup 式で上書きする。
9. `valid.loss` は最小、`valid.si_snr` は最大で n-best を分ける。同点は早い epoch を残す。Loss の平均だけが `average_state_dicts` を使う。選択結果には validation profile を書く。宣言の profile は `tf_locoformer.dns.accept_recipe.research_valid_fixed@1` で、`cv_synthetic_small` ではない。
10. 同じ logical step のファイルを label `step` で置き換える。`validation_pending` からの再開は、検証、plateau、選択を一度だけやり直す。Optimizer 更新と `step_batch` はやり直さない。

`export_for_inference` は Generator の state dict である。キーは `separator.` で始まる。推論 loader はその接頭辞を外して `separator` へ載せる。接頭辞の無い辞書も `separator` へ載る。学習 checkpoint とは別である。

設定例は `configs/experiments/tf_locoformer_dns.yaml`。module の `distribute` は `ddp` で、`global_batch` 4 と `micro_batch` 1 は world size 4 を要求する。world size 1 は `global_batch` 不一致で拒否する。world size 1 は DDP で包まない。この YAML の `max_logical_steps` は 1 で、公式規模の DNS 学習は起動しない。検証は [Phase 4-C](PHASE_4C_TF_LOCOFORMER_TRAINING.md) と [Phase 4-F](PHASE_4F_INTEGRATION.md)。

## DeepFilterNet3

Phase 4-G のクラスは `src.experiments.models.deepfilternet3.DeepFilterNet3Model` と、`src.experiments.deepfilternet3.DeepFilterNet3Trainer` である。Factory はクラスパスを呼ぶだけで、モデル名では分岐しない。損失は既存の `_multi_resolution_spectral`、`_local_snr_db`、`_vorbis_istft` を関数として呼ぶ。`DeepFilterNetTask.train_update` は呼ばない。cosine は既存の `deepfilter_cosine_schedules` を呼ぶ。Scheduler コンポーネントは作らない。

`DeepFilterNet3Model` は `src.se.build_model("deepfilternet3")` の `DfNet` を 1 組だけ持つ。`DFStream` は Trainer が持つ。波形ごとに `reset()` してから clean、そのあと noisy を解析する。`enhance()` は呼ばない。`df_fc_a` は state dict に残り、forward では未使用なので、prepare の `find_unused_parameters` は true である。

`DeepFilterNet3Trainer` の lifecycle は次である。

1. 1 epoch は `plan_online_epoch` の更新列である。`steps_per_epoch` が変わると cosine 表を作り直す。
2. index は `epoch * steps_per_epoch + index_in_epoch` で、Optimizer 更新の前に適用する。過去 epoch の更新数の累積ではない。
3. 各 logical step は train、`zero_grad`、蓄積、`runtime.backward`、`runtime.clip`、`runtime.step`、`runtime.end_step` である。
4. epoch の途中では validation、best、early stopping は走らない。
5. epoch の最後は label `validation_pending` を書き、rank 0 が valid 全体を発話ごとの等重み平均で見る。Optimizer は更新しない。
6. best は validation loss の厳密な改善だけである。非改善が patience に達すると止まり、改善するとカウンタを 0 に戻す。
7. 同じ logical step のファイルを label `step` で置き換える。
8. batch schedule があるとき、world size 1 かつ epoch の先頭だけで `Dfn3IndexSampler` の micro-batch を付け替える。world size が 1 でないときは拒否する。
9. resume は、保存された `micro_batch` 以外の identity が一致するときだけ、その幅へ戻してから identity と plan hash を比べる。

`export_for_inference` は `DfNet` の state dict で、`df_fc_a` を含む。設定例は `configs/experiments/deepfilternet3.yaml`。この YAML の `max_logical_steps` は 1 で、`dfn3.online.official@1` は起動しない。検証は [Phase 4-G](PHASE_4G_DEEPFILTERNET3_TRAINING.md)。

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

学習結果で Sampler の分布を変える処理は、Trainer が DataModule の `set_sampling_weights(mapping)` を呼ぶ形にする。`SnapshotDataModule` は Phase 3-F でこのメソッドを持つ。開いている view の変更は `DataError` で、`close()` のあと、まだ計画していない epoch から効く。変更後の重みを、その epoch の `state_dict` に含め、表の有無や中身が違う `resume` は `StateError` にする。checkpoint の表では上書きしない。平坦 policy 以外には掛けない。旧文の `update_weights(sample_id, weight)` はこのメソッドに置き換えた。

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

`GanTrainer` は Phase 1 の必須実装には含めない。この YAML は Builder がグラフを組めることの仕様例である。Phase 1 の受け入れは、この YAML に対して `build` が二つの Optimizer を別パラメータへ繋ぐところまでとする。`run` は `StandardTrainer` の例で確認する。MP-SENet の公式 GAN は `GanTrainer` ではなく `MPSENetGANTrainer` である。

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
