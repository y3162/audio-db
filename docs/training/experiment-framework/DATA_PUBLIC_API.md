# Data System 公開 API

Phase 3-A の契約である。固定ペアの Source、Sampler、Protocol、Batcher、`SnapshotDataModule` は Phase 3-B で `src/experiments/data/` にある。chunk の `ChunkSampler` と `ChunkProtocol` は Phase 3-C で同じ DataModule に載っている。オンラインの `Dfn3IndexSampler`、`Dfn3Protocol`、`SemambaProtocol` は Phase 3-D で同じ DataModule に載っている。world size 2 の正式受け入れは Phase 3-E で記録した。動的 Sampler は未実装である。構成の理由は [DATA_ARCHITECTURE.md](DATA_ARCHITECTURE.md)。Phase 2 まで実装済みの `TensorPairData` はこの契約の合成データ実装として残り、メソッドの意味は変えない。

利用者は `SnapshotDataModule` と、YAML に書く Source / Sampler / Protocol / Batcher だけを触る。純関数のモジュールパスは component 間の API であり、実験 YAML のクラス名には出さない。`_crop` のような先頭が `_` の関数は、Phase 3-B で公開名の委譲を足すまで、新コードから呼ばない。

## 例外

Phase 3-B で `src/experiments/errors.py` に足す。

```python
class DataError(ExperimentError):
    """A snapshot, sampler, or batch specification cannot supply this run."""
```

| 例外 | いつ |
|---|---|
| `DataError` | snapshot が読めない、ready でない、`delivery` が `on_demand` でない、policy がその protocol に無い、world size や batch の組み合わせが protocol の規則に反する、開いている `EpochView` の最中に重みを変える |
| `StateError` | `load_state_dict` の identity、world size、`plan_hash`、重み表の有無が一致しない。`update_index` が範囲外 |
| `BuildError` | `role: data` の `setup` が失敗したとき、Builder が `DataError` を包む。現行の `setup` 失敗と同じ |
| `RuntimeFailure` | Trainer の `accumulation` と DataModule の `group_size` が違う |
| `CheckpointError` | 形式、world size、保存後の再計算で `plan_hash` が違う。現行 CheckpointManager のまま |

素材ファイルの欠損は `DataError`。`ReproductionError` を実験の外まで漏らさない。捕捉して `DataError` にする。

## 共有の値

新しい型は `src.experiments.data.sample` に置く。`src.reproduction.pipeline.types` の dataclass は現行経路が使い続ける。フィールドを足さない。

```python
@dataclass(frozen=True)
class Crop:
    start: int
    length: int
    pad_to: int

@dataclass(frozen=True)
class ChunkSpan:
    start: int
    stop: int

@dataclass(frozen=True)
class SampleRef:
    """音声バイトを持たない。1 rank の計画の 1 要素。"""

    sample_id: str
    realization_id: str
    index: int
    case_id: str | None = None
    crop: Crop | None = None
    chunk: ChunkSpan | None = None
```

`Batch` は現行の `src.experiments.data.Batch` を拡張する。既存の位置引数 4 つは維持する。

```python
@dataclass(frozen=True)
class Batch:
    inputs: torch.Tensor
    targets: torch.Tensor
    sample_id: str
    realization_id: str
    lengths: torch.Tensor | None = None
    sample_ids: tuple[str, ...] = ()
    realization_ids: tuple[str, ...] = ()
    teacher_kind: str | None = None
```

| フィールド | 意味 |
|---|---|
| `inputs` | モデルへの入力。固定ペアと chunk と SEMamba では noisy または degraded。DFN3 では mixture。CPU で作り、Trainer が forward 前に device へ移す |
| `targets` | 教師。固定ペア、chunk、SEMamba は clean。DFN3 は protocol の mixed target。公式損失はパッドをマスクしない |
| `sample_id` | `sample_ids` をカンマで結んだ 1 文字列。ログの現行列を変えない |
| `realization_id` | `realization_ids` をカンマで結んだ 1 文字列 |
| `lengths` | パッド前のサンプル数。形状は `[micro_batch]`、dtype は `torch.long`、device は collation 時は CPU |
| `teacher_kind` | snapshot または protocol の教師種別。合成データの `TensorPairData` は `None` のまま |

`extras` にテンソルを隠さない。損失が読む波形は `inputs` と `targets` だけである。

## 識別子

| 名前 | 定義 | 保存場所 |
|---|---|---|
| Dataset Identity | snapshot の `dataset_id` と `snapshot_hash` の組。name と version はラベルで、identity の一致判定には使わない | `identity()` と checkpoint の `dataset_identity` |
| Dataset Version | 登録行の整数 version。内容ハッシュが同じなら増えない。学習 API は新しい version を採番しない | `identity()` の表示用フィールド |
| Protocol Identity | snapshot の `protocol_id` | `identity()` |
| Sample Identity | 下表 | `SampleRef.sample_id` |
| Realization Identity | 下表。同じ realization は、同じ snapshot のファイルから同じ CPU 波形になる | `SampleRef.realization_id` |
| Experiment Identity | 既存の `config_sha256`。データ計画とは別 | checkpoint manifest |
| Data Plan Hash | 下の `plan_hash`。epoch の全 rank の `SampleRef` 列 | `EpochView.plan_hash` と `state_dict` |

Sample と Realization:

| 方式 | `sample_id` | `realization_id` |
|---|---|---|
| 固定ペア | recipe の `case_id`。両モデルで共有し得る | `"{case_id}:epoch-{epoch}:crop-{start}+{length}"`。ペアのファイルは epoch で変わらず、損失が見る窓が変わる |
| chunk | realization 文書の `case_id` | `"{case_id}:{start}:{stop}"` |
| オンライン、`case_id` が無い train | `"{protocol_id}:{split}:{index}"` | `"{protocol_id}:{snapshot_hash}:{seed}:{split}:{epoch_key}:{index}"`。rank は入れない。train の `epoch_key` は epoch、valid / test は 0 |
| オンライン、valid / test の case | 保存された `case_id` | 上と同じ seed 組。case 文書は SNR の実現値を持たないので、realization は seed 組の方である |

同じ sample から複数の雑音を作るのは、オンラインの `(epoch_key, index)` が realization を分ける。同じ sample の別 crop は、固定ペアの realization 文字列が分ける。chunk の別区間も realization が分ける。sample id はそれらの間で共有してよい。

`plan_hash` は `src.reproduction.ids.digest` に渡す次のオブジェクトである。値は文字列と整数と、それらのリストと辞書だけにする。NumPy のスカラーは入れない。

```text
{
  "epoch": int,
  "world_size": int,
  "protocol_id": str,
  "snapshot_hash": str,
  "seed": int,
  "order_policy": str,
  "order_params": dict,
  "micro_batch": int,
  "group_size": int,
  "weights_sha256": str,   # 重みが無いとき "none"
  "ranks": [ [ref, ...], ... ]
}
```

`ref` は `sample_id`、`realization_id`、`index`、`case_id`、crop の 3 整数または null、chunk の start / stop または null である。`split` と描画用の epoch は `ref` に入れない。`ranks` の長さは world size で、添字が rank である。各プロセスが全 rank の列を計算して同じ digest を得る。`update_index`、worker 数、prefetch、device は入れない。オンラインの dataset、sampling factor、発話キーが epoch で変わるかは、新しいトップレベルキーではなく `order_params` に入れる。トップレベルを足すと固定ペアと chunk の hash が変わる。固定ペアの `order_parameters` は `order_params` だけを返す。

重み表の digest も `ids.digest` である。小数はレシピと同じ 6 桁へ丸まる。Phase 3-F で重みを導入するとき、試験ベクトルはこの正規化のあとで固定する。

## 利用者の API

Trainer と checkpoint が呼ぶ。`TensorPairData` は `validation_epoch`、`group_size`、`lineage` を持たない。持つオブジェクトだけが後続の検査を受ける。

```python
class SnapshotDataModule:
    group_size: int

    def identity(self) -> dict[str, Any]:
        """順序乱数の残状態は含めない。"""

    def setup(self, runtime: Runtime) -> None:
        """snapshot を開き、rank と world size を固定する。DuckDB は開かない。"""

    def train_epoch(self, epoch: int, *, start_update: int) -> EpochView:
        """この rank の micro-batch 列。start_update より前の添字は batch が拒否する。"""

    def validation_epoch(self, epoch: int) -> Iterator[Batch]:
        """setup で受けた rank と world size を使う。プロトコルごとの分担は下表。"""

    def state_dict(self) -> dict[str, Any]: ...

    def load_state_dict(self, state: dict[str, Any]) -> None: ...

    def lineage(self) -> dict[str, Any]:
        """snapshot_provenance_parts の戻り。ファイルは書かない。"""

    def close(self) -> None:
        """スレッドとファイルを閉じる。二度呼んでも成功する。"""
```

`identity()` のキーは次で固定する。

```text
class, dataset_id, dataset_name, dataset_version, snapshot_hash,
protocol_id, algorithm, profile, claim_label, selection_uses_test,
split, seed, order_policy, micro_batch, group_size, window, normalize
```

`class` は `src.experiments.data.SnapshotDataModule`。`window` が無い chunk は、YAML の `chunk_length` を整数で入れる。`normalize` が無い経路は JSON の `null`。

`EpochView` の契約は現行のままである。

```python
class EpochView:
    updates: int
    plan_hash: str
    def batch(self, update_index: int) -> list[Batch]:
        """長さ 1 のリスト。同じ index は同じ CPU テンソル。cursor は進まない。"""
```

`updates` は、この rank の micro-batch の本数である。Optimizer 更新の本数ではない。`StandardTrainer` が `accumulation` 回 `batch` を呼び、その和が 1 logical step になる。`group_size` はその `accumulation` と一致させる。列の長さは `group_size` の倍数にする。足りない末尾は捨てる。例外は DFN3 の world size 1 で、`group_size` は 1 固定とし、最後の短い micro-batch は残す。

`state_dict` のキー:

```text
epoch: int
update_index: int          # 次に batch へ渡す micro-batch 添字
plan_hash: str             # state["epoch"] の計画
world_size: int
identity: dict
weights: null              # Phase 3-F まで null。辞書になったら sample_id から数値
```

`load_state_dict` は、現在の `identity()` と `world_size` と、`state["epoch"]` から再計算した `plan_hash` を比べる。prefetch のバッファは捨ててから比べる。古い先読みの例外は状態へ載せない。`update_index` が負、またはその epoch の `updates` を超えるときは `StateError`。等しいときは、次の `train_epoch` が epoch を繰り上げる現行 Trainer の動作に任せる。

`prefetch=False` は `load_workers` を 0 にする。`prefetch=True` で `load_workers` を省略すると 1 である。`prefetch=True` かつ `load_workers < 1`、または `prefetch=False` かつ `load_workers` が 0 でないときは `ValueError`。先読みは `load_workers` 本の非 daemon スレッドが、次の micro-batch からその本数だけを CPU で描く。キャッシュは `max(1, load_workers + 1)` 件までである。`update_index` は先読みでは進まない。`join` が `worker_join_timeout`（既定 30 秒）を超えたスレッドは `DataError` で、成功として捨てない。worker 数と prefetch の有無は `plan_hash` と `identity()` に入らない。実験 YAML の kwargs なので `config_sha256` には入る。worker 数だけが違う再開は config identity の不一致で拒否する。波形が一致することと、その再開を許すことは別である。

Phase 3-F で足すメソッドは次だけである。3-B から 3-E は定義しない。

```python
def set_sampling_weights(self, weights: Mapping[str, float]) -> None:
    """次の train_epoch から平坦 Sampler に効く。開いている EpochView があれば DataError。"""
```

## component 間の API

YAML の `inject` で渡す。Trainer はこれらのメソッドを直接呼ばない。

```python
class SnapshotSource:
    def open(self) -> None: ...
    def close(self) -> None: ...
    def dataset(self) -> Mapping[str, Any]: ...
    def case_ids(self, slot: str) -> tuple[str, ...]: ...
    def document(self, case_id: str) -> Mapping[str, Any]: ...
    def material_ids(self, slot: str) -> tuple[str, ...]: ...
    def asset_path(self, asset_id: str) -> Path: ...

class IndexSampler:
    def sequence(self, epoch: int, rank: int, world_size: int) -> tuple[SampleRef, ...]:
        """この rank の列。他 rank の列は plan_hash 用に sampler が内部で計算する。"""

    def all_ranks(self, epoch: int, world_size: int) -> tuple[tuple[SampleRef, ...], ...]: ...

class Protocol:
    def render(self, sample: SampleRef) -> tuple[np.ndarray, np.ndarray, int]:
        """noisy, clean, valid_length。CPU の float32。乱数を引いてよいのはオンラインだけ。"""

class MicroBatcher:
    def groups(self, samples: Sequence[SampleRef]) -> tuple[tuple[SampleRef, ...], ...]:
        """各要素が 1 micro-batch。長さは group_size の倍数。"""

    def collate(
        self,
        rendered: Sequence[tuple[np.ndarray, np.ndarray, int, SampleRef]],
    ) -> Batch: ...
```

`SnapshotSource.open` は DataModule.`setup` だけが呼ぶ。Source のコンストラクタはパスを保存するだけで、ファイルを開かない。Builder は `other` に `setup` を呼ばないためである。

`IndexSampler.sequence` が返す `SampleRef` は、Protocol が追加情報無しで `render` できる状態である。固定ペアの crop 座標は、Sampler が `rank_order` のあと Protocol の公開メソッド `locate` で埋める。

```python
class FixedPairProtocol(Protocol):
    def locate(self, case_id: str, length: int, epoch: int) -> Crop:
        """planner の crop 公開関数へ委譲する。式をこのクラスに写さない。"""
```

chunk と DFN3 は `locate` を持たない。座標と添字は、それぞれの Sampler が既存の計画関数から読む。

### 検証の分担

`validation_epoch` は学習の `train_epoch` とは別の計画を作る。rank の再分割はしない。

| protocol | 列 | 波形 |
|---|---|---|
| `fixed_pair.mp_senet` | rank 0 の呼び出しが全件。他 rank は空の iterator | `full_length`。crop 無し。realization の crop 部分は `full` |
| `fixed_pair.mossformergan_se_16k` | `rank_mod_fixed_v1` の sampler epoch 0 | 窓は学習 `epoch` の `locate` |
| `pregenerated.*` | `plan_chunk_epoch` を `shuffle=False`、`num_samples=None` で呼び、この rank の span | 保存済み realization の切り出し |
| `online.dfn3` | `plan_online_epoch` を world size 1 として全件。呼び出し rank が 0 のときだけ返す | 波形の epoch key は 0。並びの seed には引数 `epoch` を入れる。整数の `sampling_factor` では発話キーも epoch 0。非整数では既存の `speech_key_positions` が検証 epoch で発話を入れ替える |
| `online.semambapp_gsr` | 素材順の全件。rank 0 だけが返す | epoch key は 0 |

空の iterator はエラーではない。Trainer がどの rank で検証を呼ぶかは Trainer の責務である。検証損失を rank 間で集める処理は、Phase 3-E では入れていない。公式の検証損失とモデル選択も入れていない。集計は後続 Trainer の責務である。Phase 3-B は固定ペアの 2 行を実装した。Phase 3-C は `pregenerated.*` の行を実装した。Phase 3-D はオンラインの 2 行を実装した。Phase 3-E は同じ分担を CPU / Gloo の 2 プロセスで測った。整数因子の検証波形は epoch が違っても index ごとに最大絶対誤差 0 である。非整数因子の発話キーは epoch で変わる。これは既存関数に合わせた結果で、波形一致の測定は整数因子で行った。固定ペアの `realization_id` は epoch を含むので、波形が同じでも文字列は epoch で変わる。

## 内部だけの API

次は公開しない。試験も、これらの名前に依存しない。

- prefetch スレッドの future
- 描画キャッシュの辞書
- `AssetStore` の非公開属性
- 計画関数が返す現行の `EpochPlan`。新 Sampler は必要な添字と span だけを読む

`plan_fixed_pair` は本番の DataModule からは呼ばない。受け入れ試験の比較対象としてだけ呼ぶ。

## ライフサイクル

1. Factory が Source、Sampler、Protocol、Batcher、DataModule を作る。この時点ではファイルを開かない。
2. Builder が DataModule.`setup(runtime)` を呼ぶ。Source を開き、protocol の `delivery` が `on_demand` でなければ `DataError`。`require_profile` が kwargs にあり profile が違えば `DataError`。blocked dataset は snapshot が無いので、パス解決の失敗として `DataError`。
3. Trainer が `train_epoch` を得る。計画は `SampleRef` の列で、波形はまだ無い。
4. `batch(i)` が描画と collation を CPU で行う。prefetch が次の添字を先に描いても、`update_index` は Trainer が書き込むまで進まない。描画中の例外は次の `materialize` へ伝播する。
5. checkpoint は DataModule の `state_dict` だけを data 状態として書く。
6. `close` は `Experiment.close()` だけが呼ぶ。正常終了、例外、preflight 後、`setup` 失敗、Builder が完成グラフを返す前の失敗、二重 close をこの一箇所で扱う。prefetch スレッドが timeout 以内に終わらないときは `DataError` で、ファイルを閉じたあともそのスレッドを成功扱いにしない。`StandardTrainer` は DataModule を閉じない。構築途中で失敗したときは、Builder が保持する部分レジストリを Experiment が閉じる。

`initialize_from` は data の `update_index` を戻さない。現行の initialize 契約のまま、データ位置は初期値である。

## YAML

`schema` は現行の `audio-db-experiment/v1` のままである。データ用の新しいトップレベルキーは足さない。snapshot のパスは kwargs の相対パスで、コードにホストの絶対パスを書かない。

`group_size` は Trainer の `accumulation` と同じ整数にする。Phase 3-B の Trainer が実行開始時に比べる。

### 固定ペア

```yaml
source:
  role: other
  class: src.experiments.data.SnapshotSource
  kwargs:
    snapshot: mp_senet.vb28.ondemand.safe@1
    snapshot_root: data/db/snapshots
    split: train
sampler:
  role: other
  class: src.experiments.data.OrderSampler
  kwargs:
    seed: 0
  inject:
    source: {ref: source}
    protocol: {ref: protocol}
protocol:
  role: other
  class: src.experiments.data.FixedPairProtocol
  kwargs: {}
  inject:
    source: {ref: source}
batcher:
  role: other
  class: src.experiments.data.MicroBatcher
  kwargs:
    micro_batch: 2
    group_size: 1
data:
  role: data
  class: src.experiments.data.SnapshotDataModule
  kwargs:
    require_profile: leakage_safe
  inject:
    source: {ref: source}
    sampler: {ref: sampler}
    protocol: {ref: protocol}
    batcher: {ref: batcher}
```

`OrderSampler` は policy 名を書かないとき、snapshot の rule を `resolve_order_policy` に渡す。MP-SENet の 2 名リストは、world size 1 で `fixed_single_shuffle_v1`、それ以外で `torch_distributed_sampler_v1` になる。MossFormerGAN は同じ 4 クラスで、rule が `rank_mod_fixed_v1` を返す。正規化の名前は protocol クラスが snapshot の rule から読む。YAML に `noisy_rms_unit` とは書かない。

公式忠実の実行は `require_profile` を付けず、snapshot 名を official 側にする。

### chunk

```yaml
sampler:
  role: other
  class: src.experiments.data.ChunkSampler
  kwargs:
    seed: 0
    num_samples_per_epoch: 32
    chunk_length: 16000
    shift_ratio: 0.5
    num_cache_chunks: 8
    shuffle: true
  inject:
    source: {ref: source}
protocol:
  role: other
  class: src.experiments.data.ChunkProtocol
  kwargs: {}
  inject:
    source: {ref: source}
batcher:
  role: other
  class: src.experiments.data.MicroBatcher
  kwargs:
    micro_batch: 1
    group_size: 1
```

`ChunkSampler` は `plan_chunk_epoch` を、プロセスの world size のまま呼ぶ。shard した列を world size 1 として渡し直さない。`group_size` は 1。現行の chunk 状態が accumulation 1 であるためである。`group_size` が 1 でないときは `DataError`。`num_samples_per_epoch` を省いた検証用の呼び出しは、公開メソッドではなく `validation_epoch` の内部である。

`global_batch` を kwargs に書いたときは、`micro_batch * world_size` と一致しなければならない。省略したときはその積を計画関数へ渡す。`plan_chunk_epoch` の `rank_batches` はすでに micro-batch である。Sampler は iterator stop のあとの batch を平坦な `SampleRef` 列にし、`MicroBatcher` が同じ `micro_batch` で区切り直す。`group_size` が 1 なので、この再区切りは元の batch 境界を保つ。二度目の shuffle と二度目の rank 分割はしない。

MS-SNSD と DNS の違いは `ChunkProtocol` が snapshot の algorithm から描画関数を選ぶことである。YAML のクラスは同じである。`pregenerated.ms_snsd_tfgridnet` は `render_ms_snsd_pair`、`pregenerated.dns_tflocoformer` は `render_dns_pair` を呼ぶ。どちらも既存の `render_realization` と PCM_16 の順を使い、chunk の切り出しはその後である。`OnDemandChunkData` と `DnsOnDemandChunkData` は作らない。式を DataModule に写さない。

### オンライン混合と劣化

```yaml
sampler:
  role: other
  class: src.experiments.data.Dfn3IndexSampler
  kwargs:
    seed: 0
  inject:
    source: {ref: source}
protocol:
  role: other
  class: src.experiments.data.Dfn3Protocol
  kwargs: {}
  inject:
    source: {ref: source}
batcher:
  role: other
  class: src.experiments.data.MicroBatcher
  kwargs:
    micro_batch: 2
    group_size: 1
```

SEMamba は `sampler.class` を `src.experiments.data.OrderSampler`、`protocol.class` を `src.experiments.data.SemambaProtocol` にする。policy は rule の `per_rank_independent_v1` を `resolve_order_policy` が返す。`group_size` は 1 以上で、`micro_batch * world_size * group_size` が protocol の `global_batch` と違うときは `DataError` にしない。現行は警告して指定値を使う。新 API も指定値を使い、`identity` に実際の `micro_batch` と `group_size` を残す。警告のログは rank 0 が 1 行出す。DFN3 だけは `group_size != 1` を `DataError` にする。world size が 1 より大きいとき、`micro_batch * world_size` が計画上の均等分割を崩す指定は、`plan_online_epoch` が拒否する条件をそのまま `DataError` にする。

三つの YAML は、DataModule のクラスと `train_epoch` / `batch` / `state_dict` が同じである。違うのは Sampler と Protocol のクラスだけである。

## Trainer からの取得

`StandardTrainer` の現行ループを維持する。

```text
view = data.train_epoch(epoch, start_update=index)
batches = view.batch(index + offset)   # list の長さは 1
batch.inputs, batch.targets            # forward 前に runtime.device へ
data.update_index = index + accumulation
```

`batch.sample_id` をログに書く現行の行は変えない。研究用の行単位 ID は `batch.sample_ids` を読む側が使う。

デバイス移動と `group_size` の検査と `close` の finally は、Phase 3-B で `StandardTrainer` に足す。Runtime の `autocast` と `step` は触らない。

## 合成データとの関係

`TensorPairData` は snapshot を開かない。`SampleRef` も使わない。`plan_hash` は現行どおりテンソルバイトと `batch_size` と `world_size` から作る。音声用の `plan_hash` オブジェクトを合成データへ強制しない。

両方に共通なのは、`train_epoch`、`batch` が cursor を持たないこと、`update_index` が次の micro-batch であること、先読みを状態に入れないこと、rank 分割を 1 回だけ行うことである。
