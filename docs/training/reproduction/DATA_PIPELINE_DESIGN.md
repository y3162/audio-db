# 再現用 Data Pipeline 設計

状態: **固定ペアの学習 Pipeline、ESPnet chunk の複数 rank、MS-SNSD のオンデマンド混合、VoiceBank の学習時リサンプルは実装済み**（2026-11-06、P3・P5・P6）。学習用の生成済み WAV は必須にしない。波形キャッシュは実装しない。`pipeline/pregenerated.py` は `delivery: on_demand` 以外を拒否し、オンデマンドは `pipeline/ondemand.py` が再構築する。履歴の固定ペアスナップショットは `open_training_data` が読む。新しい学習の `execute` は `on_demand` を必須にする。DeepFilterNet3 のオンライン混合は実装済み。SEMamba++ は未実装。検証は [P3_VERIFICATION](P3_VERIFICATION.md)、[P5_VERIFICATION](P5_VERIFICATION.md)、[P6_VERIFICATION](P6_VERIFICATION.md)、[P8_VERIFICATION](P8_VERIFICATION.md)。DNS の受理レンダラは [P7_1_VERIFICATION](P7_1_VERIFICATION.md)。

## 0. 位置づけと正本の分担

| 内容 | 正本 |
|---|---|
| Dataset / Sampler / Renderer / Batch の API、3 方式の流れ、モデル別の規則、DDP、乱数・再開、性能、YAML とコマンド、評価との関係 | **この文書** |
| DB スキーマ、ID・ハッシュ、Protocol と Case、Official / Research 分離、staging、事前生成の保存・再開、スナップショット | [REPRODUCTION_DB_DESIGN](../../db/REPRODUCTION_DB_DESIGN.md) |
| 実装フェーズ、受け入れテスト、未解決事項、着手前の確認 | [IMPLEMENTATION_PLAN](IMPLEMENTATION_PLAN.md) |
| 6 モデルの公式の実効条件、D-01〜D-09 | [TARGETS](TARGETS.md)、[DECISIONS](../../../.agents/DECISIONS.md) |

記法: 「（確認）」は 2026-11-06 に現行コードを読んで確認した事実、「（設計）」はこの文書が定める内容、「（UNKNOWN）」は一次資料や実機で確認できない事項。

---

## 1. 設計の目標と制約

1. 6 モデルが、**3 つのデータ方式**（固定ペア、事前生成、オンライン混合）を共通の境界で扱う。1 つの巨大な Dataset にしない（§3、§5）。
2. データの**計画（何をどの順で使うか、乱数の抽選）**と**波形の生成（ファイルを読む、混ぜる）**を分ける。計画は `(seed, epoch, world_size, rank, index)` の純関数にする。これで、再開、prefetch の破棄、virtual rank、DDP の整合が単純になる（§9）。
3. 学習は DuckDB を開かない。DB から書き出した不変のスナップショットを読む（DB 設計 §10）。
4. 現行の学習基盤（`Trainer`、`RecipeDataset`、checkpoint）を壊さない。**Legacy 経路は残し**、新 Pipeline は同じ Trainer の入口（§5.1 の `TrainingData`）に差し込む。
5. `src/se/`、`src/asr/` は他の `src/` に依存させない。独立 SE 実装は `src/se/official/` に実行時依存しない。公式のファイルリストなどは、追跡対象の `configs/data/lists/` に複製する（DB 設計 §4.8）。
6. D-09: 次は統計的同等性に任せず、**厳密に**検証する（IMPLEMENTATION_PLAN の受け入れテスト）: 固定ペアの対応、train / valid / test の分割、サンプリングレート、教師波形の定義、損失、optimizer・scheduler の更新規則、checkpoint 選択規則。確率的な混合・サンプリングの結果は、分布の同等性で受け入れる。

---

## 2. 現行実装の事実と限界（確認）

P3 のあと、固定ペアは下表の `RecipeDataset` 一行ではなく `src/reproduction/pipeline` を通る。Legacy の `RecipeDataset` 経路は残している。この表のほかの行は、P3 時点の Legacy と未実装の方式の説明である。

| 項目 | 現状 | 新設計への影響 |
|---|---|---|
| データ読み込み | `RecipeDataset`（`src/training/data/dataset.py`）が `render_recipe` で wav を読み、`_match_rate`（`torchaudio.functional.resample`）→ `_normalize`（noisy の RMS）→ `_segment`（crop/pad）の順に処理する。`load_workers` はスレッドプール | 公式再現では、正規化・crop・リサンプルが Protocol ごとに違う（§7）。固定の 1 経路では足りない |
| 評価用セットの拒否 | `load_recipe_dataset` が `resource.duckdb` と評価用名前付きセットを学習・検証に使うことを拒否する（`EVALUATION_SET_NAMES`） | 公式再現の「test を validation に使う」例外は、この関数を変えずに、新経路で `selection_uses_test` と明示設定の両方があるときだけ許す |
| 素材のハッシュ | `render_recipe` は素材を読むたびに SHA-256 を計算する（`src/evaluation/audio/render.py` L191-199） | 学習では読み取りが 2 倍になる。新経路は準備時とスナップショット作成時に検証し、学習中は既定で行わない（§11） |
| 順序 | `epoch_indices(size, seed, epoch, world_size, rank)`（`src/training/data/sampler.py`）: `mix_seed` の seed でグローバルに shuffle し、`[rank::world]` を取り、端数を捨てる。`crop_start(case_id, seed, epoch, ...)` | 公式の sampler は 4 種類以上あり、どれも違う（§8.2）。`OrderPolicy` に分ける。現行の関数は `global_shuffle_stride_v1` として黄金テストで固定する |
| チャンク | `chunk_plan.py` の `epoch_chunk_batches` が ESPnet の ChunkIterFactory を **単一プロセス**で再現する。`Trainer._run_chunk_epochs` は `world_size != 1` を拒否する | 2〜4 rank の DDP を足す（§8.3）。単一プロセスの計画は `world = 1` の特殊形になる |
| バッチ | `plan_batches`（`src/training/core/batch.py`）が global batch と蓄積を計画する | 新経路でも使えるが、rank 別の計画と蓄積のグループ化は `EpochPlan` が持つ |
| 再開位置 | checkpoint に `micro_batches_consumed`、`dataset.identity()`、`chunk_history` を保存する（`Trainer._save`） | 位置は**更新単位**に移す。互換のため `micro_batches_consumed` も書く（§9.3） |
| 乱数 | `mix_seed`（`src/training/core/rng.py`）は SHA-256 の seed。load の並列は共有 RNG を使わない | そのまま採用し、拡張する（§9.1） |
| prefetch | 1 グループ先読みを別スレッドで行う（`data.prefetch`） | 計画が純関数なので、破棄しても乱数状態は動かない（§9.4） |
| 設定 | `src/training/se/config.py` のフラットな設定。未知のキーを拒否する。`data` は `database`、`train_set`、`validation_set`、`load_workers`、`prefetch`、`rms_normalize` | 新経路のキー（`pipeline`、`dataset` など）を追加し、Legacy のキーとの併用を拒否する（§12.1） |

---

## 3. 責務の分離とモジュール構成

### 3.1 5 つの層

| 層 | 責務 | 入力 → 出力 | 乱数を引くか | DB を開くか |
|---|---|---|---|---|
| **Data Preparation** | 素材の登録、公式リストの取り込み、派生素材の作成、事前生成、DB 登録、検証、スナップショット作成 | 素材・規則 → staging、`reproduction.duckdb`、`data/generated`、スナップショット | 生成時のみ（`gen_index` から導出した seed） | 書き込み（1 プロセス） |
| **Sample Planner** | epoch の計画: 順序、rank への割り当て、チャンク、バッチ、更新グループ、オンライン混合の実現値の抽選 | `DataSnapshot` + `EpochSpec` + `(epoch, world_size)` → `EpochPlan` | **ここだけ**。すべて純関数 | 開かない |
| **Audio Renderer** | 1 サンプルの波形を作る: ファイルを読む、リサンプル、混合、正規化、crop。乱数を引かず、計画の抽選結果だけを使う | `SampleRef` + `AssetStore` → `RenderedSample` | 引かない | 開かない |
| **Batch Assembler** | `RenderedSample` を `Batch` にまとめる（pad、長さ、教師の整理） | サンプル列 → `Batch` | 引かない | 開かない |
| **Trainer** | 順伝播・逆伝播・更新、scheduler、validation の指標、checkpoint・best 選択、rank 間の集約、ログ | `TrainingData` から `UpdateGroup` を受ける | モデル側の乱数（dropout など）のみ | 開かない |

「Dataset」という名の巨大なクラスは作らない。Trainer が見るのは `TrainingData` だけで、その中身が Planner・Renderer・Assembler の組み合わせである（§5）。

### 3.2 モジュール構成（設計）

新しい最上位パッケージ `src/reproduction/` を作る。理由: `src/data/` と `src/evaluation/` は既に相互依存で、これ以上広げない（`PROJECT.md`）。`src/se/`・`src/asr/` から独立させる。

```
src/reproduction/
  errors.py
  ids.py                # ID・ハッシュ（src/data/recipe.py の json_hash を再利用）
  db/
    schema.py           # repro スキーマの作成（reproduction.duckdb）
    registry.py         # staging → DB の登録（1 トランザクション、冪等）
    snapshot.py         # DB → スナップショット（Parquet + JSON）
    audit.py            # 件数・リストのハッシュ・leak・分布・build の検証
  prepare/
    materials.py        # corpora → assets、素材リスト
    lists.py            # 公式ファイルリスト（configs/data/lists/）→ ケース集合
    derive.py           # 派生素材（16 kHz 化など。derivations に記録）
    datasets.py         # configs/data/datasets/*.yaml → Dataset 定義
  build/
    plan.py             # 事前生成の計画（realization_v1 文書の列）
    render.py           # shard 単位の生成（再開可能）
    manifest.py         # shard マニフェスト、verified 記録
    verify.py
    generators/         # ms_snsd.py、dns_synth.py（公式の生成式を専用実装）
  protocols/
    base.py             # Protocol のレジストリ、rule のデータクラス基底、algorithm_version
    fixed_pair.py       # mp_senet、mossformergan
    pregenerated.py     # ファイルを読むだけの Protocol（tf_gridnet、tf_locoformer）
    online_dfn3.py      # DFN3 の実現値の抽選と混合
    online_semamba.py   # SEMamba++ の GSR（reverb → noise → clipping → bandwidth）
  pipeline/
    types.py            # SampleRef、RenderedSample、Batch、UpdateGroup、EpochPlan など
    store.py            # AssetStore（スナップショットから作るメモリ内の索引）
    order.py            # OrderPolicy 群
    chunking.py         # ESPnet の ChunkIterFactory の rank 対応版
    planner.py          # EpochPlanner（kind ごと）
    transform.py        # 正規化・crop・pad・リサンプルの SampleTransform
    renderer.py         # FileRenderer（固定・事前生成）、オンラインは protocols/ の Renderer
    assemble.py         # BatchAssembler
    loader.py           # スレッド（任意でプロセス）で render、prefetch
    state.py            # DataState（checkpoint に保存する位置・hash）
    training_data.py    # TrainingData 実装
```

上の木は設計時の配置である。実装では `prepare/derive.py` は 16 kHz WAV を書かず、呼ぶと拒否する。MS-SNSD の混合は `build/ms_snsd.py` がメモリ上で行い、shard の WAV は書かない。VoiceBank の 16 kHz 化は `prepare/resample.py` が学習時に行う。

依存規則（import テストで固定する）:

| モジュール | import してよい | import しない |
|---|---|---|
| `src/reproduction/protocols/`、`pipeline/` | `numpy`、`soundfile`、`soxr` または `librosa`、`torch`（`pipeline/` の Batch 変換だけ）、`src.data.recipe`（ハッシュ）、`src.reproduction.*` | `src.se`、`src.asr`、`src.evaluation`、`src.training`、`duckdb`（学習経路。`db/` だけが開く）|
| `src/reproduction/db/`、`prepare/`、`build/` | 上記 + `duckdb`、`src.data.*` の読み取り（`corpora` の列定義）| `src.se`、`src.asr`、`src.training`、`src.evaluation`（`eval` の読み取りは `db/audit.py` が `duckdb` で直接 SQL を発行する） |
| `src/training/` | `src.reproduction.pipeline`（`TrainingData` と型）| `src.reproduction.db`、`prepare`、`build`（スナップショットを読むだけ）|
| `src/evaluation/` | — | `src.reproduction.*` 全体 |
| `src/se/`、`src/asr/` | （変更なし）| `src.reproduction.*`、`src.training`、`src.evaluation` |

`pipeline/` が DuckDB を import しないのは、学習環境によって DuckDB の版が違う、または無いため（`.venvs/deepfilternet` は DuckDB を持たない。`DEVELOPMENT.md` は `.venvs/semamba` を 1.4.3 と記録するが、現在の `.venvs/semambapp` は 2.0.0。確認: 各環境の import と `requirements/*.lock.txt`）。スナップショットは `numpy` の `.npz` と JSON だけで読める形式にし（DB 設計 §10.2）、`pyarrow` も要求しない（`pyarrow` は se lock に無い）。

---

## 4. データ表現

### 4.1 型（`pipeline/types.py`）

```python
@dataclass(frozen=True)
class SampleRef:
    """Plannerの出力。音声を含まない。"""
    case_id: str | None        # 固定ペア・事前生成・検証用のオンラインスナップショット
    index: int                 # epoch の全体計画内の通し番号（rank 割り当て前）
    draw: tuple | None         # オンライン混合: (split, epoch, index) → 実現値を導く鍵
    length: int | None         # 学習レートでの全長（バッチ化の長さ。分かるときだけ）
    crop: CropSpec | None      # start, length。Planner が決める（pure）
    chunk: ChunkSpec | None    # チャンクの start, length, 元発話の長さ（事前生成の学習）

@dataclass(frozen=True)
class MicroBatchPlan:  samples: tuple[SampleRef, ...]
@dataclass(frozen=True)
class UpdateGroup:     micro_batches: tuple[MicroBatchPlan, ...]   # 1 回の optimizer.step に対応（蓄積）

@dataclass
class RenderedSample:
    noisy: np.ndarray          # float32 [T] モノラル
    clean: np.ndarray          # float32 [T] 教師波形（teacher_kind の定義どおり）
    sample_rate: int
    valid_length: int          # 実フレーム数（pad 前）
    case_id: str | None
    teacher_kind: str
    extras: dict[str, np.ndarray]   # noise、dry_clean、reverberant_clean（任意）

@dataclass
class Batch:
    noisy: Tensor              # [B, T] float32
    clean: Tensor              # [B, T] float32 教師
    lengths: Tensor            # [B] int64 実フレーム数（pad は 0 値。ESPnet の float_pad_value=0.0 と同じ）
    case_ids: tuple[str | None, ...]
    teacher_kind: str
    extras: dict[str, Tensor]
```

- モノラル `[B, T]` に固定する（6 モデルとも単一チャネルの SE。多チャンネル素材はレンダー時に公式どおりの 1 チャンネルを選び、Protocol が選び方を持つ。DFN3 と MossFormerGAN の `audioread` 後の先頭チャンネルなど）。
- 可変長: 固定窓（MP-SENet、MossFormerGAN、SEMamba++、DFN3 の 3 秒）は全サンプルが同じ長さで、`lengths == T`。チャンク（TF-GridNet、TF-Locoformer）も同じ長さ。可変長のバッチは validation（MP-SENet の全長・batch 1 など）で起きる。pad は右側の 0、`lengths` で実長を示す。損失が `lengths` を使うか（マスクするか）は Task が決める。**pad を有効サンプルとして損失に入れるかは公式の規則どおり**（TF 系は同長のチャンクなので pad は起きない）。
- clean / noisy と、乾いた clean（`dry_clean`）/ 残響つき clean（`reverberant_clean`）の区別: `Batch.clean` は**常に、その Protocol の教師**（`teacher_kind`）である。DFN3 の `official_mixed_target`（乾き 0.3 + 後期減衰 0.7）も `clean` に入る。区別が要る場合（評価時の参照や診断）は `extras["dry_clean"]` と `extras["reverberant_clean"]`。`teacher_kind = none` の Case は学習に使えない。
- 事前生成の参照: `SampleRef.case_id` から `AssetStore` が noisy・clean（任意で noise）のファイルパスと長さを返す。波形は Renderer がファイルから読む。
- オンライン条件: `SampleRef.draw = (split, epoch, index)`。Renderer が `realize(protocol, store, draw) → realization 文書` を純関数で作り、文書から混合する。文書は必要なとき（ログ、診断、`online_snapshot` の保存）にだけ保持する。

### 4.2 `EpochSpec`、`EpochPlan`

```python
@dataclass(frozen=True)
class EpochSpec:           # Protocol の rule と学習設定から作る
    order_policy: str      # §8.2
    order_params: Mapping
    seed: int
    global_batch: int      # 学習設定。公式の値は Protocol の official_global_batch と照合
    accumulation: int      # rank あたりの蓄積数（MossFormerGAN の 2 など）
    epoch_unit: Mapping    # num_samples_per_epoch（ESPnet の num_iters_per_epoch）、leftover 規則
    chunk: ChunkRule | None
    segment: SegmentRule | None

@dataclass(frozen=True)
class EpochPlan:
    epoch: int
    world_size: int
    updates: int                       # 全 rank 共通の更新数（§8.3 の最小値）
    rank_groups: tuple[tuple[UpdateGroup, ...], ...]   # [rank][update]
    plan_hash: str                     # 全 rank の計画のハッシュ
    stats: Mapping                     # 捨てたチャンクバッチ数、元発話数など（ログ用）
```

`plan_hash` は、全 rank の `SampleRef` 列（`case_id`、`draw`、`crop`、`chunk`）の `json_hash`。再開時と DDP の rank 間で一致を確認する。

---

## 5. インターフェース

### 5.1 Trainer との境界（`TrainingData`）

```python
class TrainingData(Protocol):
    def identity(self) -> dict            # 比較用（現行 RecipeDataset.identity() の上位互換）
    def epoch(self, epoch: int, *, rank: int, world_size: int,
              start_update: int = 0) -> EpochStream
    def validation(self, *, rank: int, world_size: int,
                   epoch: int | None = None) -> Iterator[Batch]
    def data_state(self, epoch: int, committed_updates: int) -> dict   # checkpoint 用
    def close(self) -> None

class EpochStream(Protocol):
    updates: int                           # この epoch の（全 rank 共通の）更新数
    plan_hash: str
    def __iter__(self) -> Iterator[tuple[Batch, ...]]   # 1 要素 = 1 更新の micro-batch 列
```

- Trainer は、`epoch()` が返すストリームを回し、1 要素ごとに更新する。インデックス、バッチ分割、seed、チャンクを Trainer は持たない。
- Legacy アダプタ（`src/training/data/legacy.py`）が、現行の `RecipeDataset` + `epoch_indices` / `epoch_chunk_batches` を同じ `TrainingData` に包む。**Legacy 経路の挙動は変えない**（現行のテストが黄金）。
- `Trainer._run_epochs` と `_run_chunk_epochs` は、最終的に 1 つのループに統合できる。統合は Legacy の黄金テストを通してから行う（IMPLEMENTATION_PLAN フェーズ 6 の一部）。

### 5.2 Dataset と Sampler の境界

| 決めること | 担当 | 渡すもの |
|---|---|---|
| どの素材を使うか、素材の属性、教師の定義 | スナップショット（`DataSnapshot`、`AssetStore`） | Planner に読み取り専用で渡す |
| epoch の順序、rank 割り当て、チャンク、バッチ、crop の位置、オンラインの抽選 | **Planner**（`OrderPolicy`、`chunking`、`EpochPlanner`、`realize`）| `EpochPlan` と `SampleRef` |
| 波形の生成、リサンプル、混合、正規化、crop の適用 | **Renderer**（`SampleTransform`、`FileRenderer`、オンライン Renderer） | `RenderedSample` |
| pad、スタック、テンソル化 | **Assembler** | `Batch` |

規則:

1. Renderer は乱数を引かない。Planner が決めた `crop`、`draw` だけを使う。
2. Planner は波形を読まない。長さは `AssetStore` の `frames` と `sample_rate` から求める（`resampled_frame_count` 相当）。
3. どちらも DB と学習の状態（optimizer、モデル）を知らない。

### 5.3 Protocol の登録

`protocols/base.py` のレジストリが、`algorithm` 名から (`RuleModel`, `Planner`, `Renderer`) を返す。`RuleModel` は `rule_json` を検証するデータクラス。未知の `algorithm` / `algorithm_version` は拒否する。出力が変わる変更では `algorithm_version` を上げ、古い版を残す（DB 設計 §6.2）。

---

## 6. 3 方式の扱い

2026-11-06 以降の原則: 音声処理は学習時にオンデマンドで行う。DuckDB は素材、生成規則、Dataset、分割、来歴を持つ。学習は DuckDB を開かず、スナップショットを読む。生成済み WAV は学習の必須データではない。メモリキャッシュとディスクキャッシュは今回は作らない。

固定データ（MP-SENet、MossFormerGAN、TF-GridNet、TF-Locoformer）は、生成条件を先に確定し、同じ条件から同じ音声を再構築する。公式が学習時にランダム化する Crop などは、そのランダム化のまま学習時の処理として残す。公式に seed が無い生成は、アルゴリズムの一致と生成ファイルのビット一致を分け、ビット一致は主張しない。

オンライン混合（DeepFilterNet3、SEMamba++）は、公式の確率分布を Protocol に残し、学習時に抽選する。再開できるよう、seed とサンプル index から再計算する。共有 RNG の残状態には依存しない。DeepFilterNet3 の抽選は実装済み。SEMamba++ は未実装。

P2 の VoiceBank 16 kHz 派生素材は、そのまま固定ペアの入力である。リサンプルを学習時へ移す変更はしておらず、移す場合は P2・P3 の再検証が要る。正規化と Crop は、これまでどおり学習時に Protocol を適用する。

### 6.1 共通の流れ

```
スナップショット → DataSnapshot / AssetStore（メモリ）
 → EpochPlanner.plan(epoch, world) ── EpochPlan（全 rank、純関数）
 → loader: 自 rank の UpdateGroup を先読みして render（スレッド）
 → BatchAssembler → Batch → Trainer
```

### 6.2 固定ペア（MP-SENet、MossFormerGAN）

- Dataset 定義: `train.cases`、`valid.cases`、`test.cases`（`recorded_pair` の Case 集合）。
- Planner: `OrderPolicy`（§8.2）で順序を決め、global batch を rank ごとに分け、蓄積でまとめる。crop の位置は `crop_start`（既存）と同じ式で `(seed, epoch, case_id)` から決める。`drop_last` などの端数規則は Protocol の `epoch_unit.leftover` が持つ。
- Renderer: `FileRenderer`。clean と noisy を読み、レート変換（既に 16 kHz の派生素材なら変換なし）、モノラル化、**正規化（Protocol ごと。§7）**、crop / pad の順で処理する。
- 正規化と crop の**順序**は公式のとおり（MP-SENet は全長で noisy の RMS 正規化してから crop、MossFormerGAN は noisy と clean を独立に `audio_norm` してから crop）。現行の `_normalize` → `_segment` と同順で、MossFormerGAN の `independent_audio_norm_v1` だけが新規。

### 6.3 事前生成（TF-GridNet、TF-Locoformer）

- Dataset 定義: `train.cases`、`valid.cases`、`test.cases`。各 Case は `realization_v1` で、長さは計画の `output.length`。TF-GridNet の新しい Protocol は `rule_json.delivery = on_demand` で、Case は生成 WAV を指さない。素材リスト（`train.speech` など）が元素材の SHA-256 を持つ。
- 旧い `tf_gridnet.ms_snsd.smoke@1` は `delivery` を持たず、生成 WAV を読む。新しい学習入口は `tf_gridnet.ms_snsd.ondemand@1`。
- Planner: ESPnet の `ChunkIterFactory` の再現（§8.3）。長さは文書から取り、ファイルを開かない。
- Renderer: `pipeline/ondemand.py` がバッチの切り出し時に `snr_mixer` を実行し、PCM_16 の読み戻しと同じ値を返す。正規化はしない。同じ文書は同じ波形になる。結果は保持しない。
- TF-Locoformer の DNS 合成も同じ分離である。生成器は `src/reproduction/build/dns2020.py` で、学習時の再構築は `pipeline/dns_ondemand.py` が行う。混合 WAV は書かない。P7 の規則は 16 kHz の区間読みで再試行しない。P7.1 は別規則で、受理した区間を保存し、16 kHz でない雑音だけを描画時にリサンプルする。検証は [P7_VERIFICATION](P7_VERIFICATION.md) と [P7_1_VERIFICATION](P7_1_VERIFICATION.md)。
- 混合 WAV を書く経路は無い。登録は条件と元素材だけを書き、学習時に `render_realization` が PCM_16 を作る。

### 6.4 オンライン混合（DFN3、SEMamba++）

- Dataset 定義: `train.speech`、`train.noise`、`train.rir`（素材リスト）、`valid.cases`（`online_snapshot`）。
- Planner: 順序は `OrderPolicy`。各サンプルの `draw = ("train", epoch, index)` を `SampleRef` に入れる。
- Renderer（専用）: DeepFilterNet3 は `protocols/online_dfn3.py` の `render_mixture`。種は `mix_seed("dfn3-online", protocol_id, snapshot_hash, seed, split, epoch, index)` で、NumPy PCG64 を使う。valid と test は epoch を種に入れない。設計時の `mix_seed("online", dataset_seed, split, epoch, index)` より、protocol と snapshot を種に足した。同じ index は world size に依存しない。SEMamba++ の `online_semamba.py` は未実装。
- 共有する部品: `AssetStore`（素材リストの索引）、音声 I/O（`soundfile`）、`assemble`、`loader`。**共有しない部品**: 混合そのもの（DFN3 の雑音合成と clip ガード、SEMamba++ の VAD 基準の SNR と劣化の順序）。共通の「混合エンジン」は作らない（D-02）。
- 公式との seed の対応: DFN3 の公式は `seed_from_u64(config_seed + epoch_seed + sample_idx)` で発話ごとに RNG を作る。構造は発話ごとの決定性で同じ。乱数の列（Rust の `StdRng`）は再現しない。公式のメインスレッド RNG（seed 42、epoch で再開しない）には依存しない。順序は `mix_seed("dfn3-order", ...)`。
- validation: `valid.cases` は `online_snapshot` の論理 index である。波形は保存しない。epoch が変わっても同じ index は同じ波形になる。学習の world size が 2 以上でも、検証は rank 0 が valid 全体を 1 プロセスで描画する。
- 登録: `data/db/reproduction-p8.duckdb`。`dfn3.online.substitute@1` と `dfn3.online.smoke@1` は代替素材。`dfn3.online.official@1` は blocked。詳細は [P8_VERIFICATION](P8_VERIFICATION.md)。

---

## 7. モデル別の規則

各モデルの公式の実効条件は TARGETS §3 が正本。この表は、Pipeline が `rule_json` で持つ項目だけを示す。`確認` は TARGETS の該当節を参照する。

| 項目 | MP-SENet | MossFormerGAN | DFN3 | SEMamba++ | TF-GridNet | TF-Locoformer |
|---|---|---|---|---|---|---|
| 方式 | 固定ペア | 固定ペア | オンライン | オンライン | 事前生成 | 事前生成 |
| レート | 16 kHz | 16 kHz | 48 kHz | 16 kHz | 16 kHz | 16 kHz |
| 正規化 | `noisy_rms_unit_v1`（全長、noisy から求めた係数を両方へ。crop の前） | `independent_audio_norm_v1`（noisy・clean を独立に −25 dBFS の 2 段。crop の前）。solver が noisy に `c` を掛ける処理は Task | なし（clip ガードは混合内） | `independent_peak_v1`（各ストリームを独立に `x/(max|x|+1e-9)`）| なし（生成時 −25 dBFS）| なし |
| 窓 | 32,000（crop は `randint(0, L−32000)`、短ければ末尾 0 埋め）| 32,000（短ければ先頭を貼って末尾 0 埋め。それ以外は一様な crop）| 144,000（短い素材は別発話を連結）| 48,000 | チャンク 48,000、shift 0.5、`L ≤ W` の発話は捨てる | チャンク 64,000、shift 0.5、offset は 30 秒入力で 0（14 チャンク）|
| 順序 | §8.2 の 1 GPU / 2 GPU 以上で別 | `rank_mod_fixed_v1`（seed 0）| 発話の key 順 × `sampling_factor` | `per_rank_independent_v1` | ESPnet chunk | ESPnet chunk |
| global batch（公式）| 4（rank あたり `4 / world`）| `2 × GPU 数`（rank あたり micro-batch 1 × 蓄積 2）| 64（`batch_size_scheduling` で epoch ごとに変わる）| config 値（公式は global 8）| 4（2 rank × 2）| 4（4 rank × 1）|
| epoch の単位 | 全 train を 1 周（端数は捨てる）| 全 train を 1 周（rank あたり `N/world` micro-batch）| `speech の key 数 × sampling_factor / batch`（端数 batch を残す）| 素材の件数に依存（UNKNOWN）| rank あたり 5,000 発話 | rank あたり 1,000 発話 |
| validation | `valid.cases` = 公式 test 824（official）／ 872（safe）。全長、batch 1。5,000 step ごと | 公式 test 824。**2 秒の random crop**（公式は検証にも同じ窓）。毎 epoch | 3 秒、固定実現値、毎 epoch | 5,000 step ごと。素材は UNKNOWN | `cv` をチャンク化（3 s、shift 1.5 s、shuffle なし）、rank あたり batch 2 | `cv_synthetic_small`（**UNKNOWN**）|
| 教師 | `dry_clean`（正規化後）| `dry_clean`（正規化後）| `official_mixed_target` | `dry_clean`（劣化前。reverb の扱いは `semambapp.md` の規則）| `dry_clean`（`clnspN.wav`）| `dry_clean` |
| checkpoint 選択（Trainer の規則）| `best_pesq`、`epoch >= 40`、検証 PESQ 平均が最大（G のみ）| 「直前より小さい」（真の best ではない）、halving 規則 | `valid loss` 最小 | 選択なし（5000 step ごと）| `valid.loss.best`（`si_snr` は無効）| `valid.loss.ave_5best`（上位 5 epoch の平均）|

- 上の「checkpoint 選択」は Pipeline ではなく Trainer の規則だが、D-09 で厳密に検証するので、ここに並べた。chunk 学習の `Trainer` は、有限な `valid.loss` が `keep_nbest` 件以上あるとき `valid.loss.ave_{n}best.pt` を書く。浮動小数は平均し、整数は累積する。件数が足りないときは書かない。TF-GridNet の `valid.loss.best` と `valid.si_snr.best` は残す。
- DFN3 の epoch ごとの batch 上限（`batch_size_scheduling`）は、現行の `scheduled_batch_size`（`src/training/core/batch_schedule.py`）が解決する（確認）。新 Pipeline の `EpochPlanner` は、epoch ごとに `global_batch` を `EpochSpec` へ渡す。
- 公式の癖の例外処理（D-02）: MP-SENet の `batch_first=False` はモデル側の欠陥で、データ側ではない。データ側の癖は、MP-SENet の 1 GPU 時の固定 shuffle、MossFormerGAN の rank 固定割り当て、SEMamba++ の rank ごとの別 shuffle、MS-SNSD の実 SNR（名目の半分）で、いずれも §8.2 と DB 設計 §9 に入れた。

---

## 8. DDP

### 8.1 方針

- 計画は**全 rank を同じ関数で計算**する（`plan(epoch, world_size)`）。各 rank は自分の `rank_groups[rank]` だけを render する。通信は要らない（`plan_hash` の一致確認だけ、epoch の開始時に 1 回）。
- rank 間の更新数の整合（ESPnet の `iterator_stop` = 全 rank の最小）は、**計画が全 rank の長さを知っている**ので、通信なしで `updates = min(...)` として決まる。
- 実 DDP と、1 プロセスで N rank を再現する virtual rank は、**同じ計画**を使う（§8.5）。

### 8.2 順序の方針（`OrderPolicy`）

| 名前 | 内容 | 使うモデル |
|---|---|---|
| `global_shuffle_stride_v1` | `mix_seed` の seed でグローバルに shuffle、`[rank::world]`、端数を捨てる。現行の `epoch_indices` | leakage-safe 側の既定、DFN3（統計等価）、Legacy |
| `torch_distributed_sampler_v1(seed=0)` | torch の `DistributedSampler` と同じ: `randperm(N, generator(seed + epoch))`、`total = N // world * world` に切り詰め、`[rank:total:world]`。**seed は常に 0** | MP-SENet（2 GPU 以上）|
| `fixed_single_shuffle_v1(seed=1234)` | 開始時に `random.seed(1234); random.shuffle(list)` を 1 回だけ。**全 epoch で同じ順序** | MP-SENet（1 GPU）|
| `rank_mod_fixed_v1(seed=0)` | rank r の担当は index が r mod world のものに固定。epoch ごとに変わるのは rank 内の並びだけ（seed 0 と epoch）| MossFormerGAN |
| `per_rank_independent_v1` | rank r が、開始時に自分専用の順列 `L_r` を seed 付きで 1 回作る。epoch ごとの index の順は torch の `DistributedSampler`（seed 0）で、要素は `L_r[perm[r::world]]`。**rank 間で重複・欠落が起きる**（公式の実効動作）| SEMamba++ |
| `espnet_chunk_v1` | §8.3 | TF-GridNet、TF-Locoformer |

- 公式の「rank ごとに別 shuffle」は未シードなので、再現できるのは**構造**（rank 間で重複・欠落が起きる性質）で、順列そのものではない（D-09）。seed 付きで作るのは audio-db の追加の決定性で、`deviations_json` に書く。
- どの方針が選ばれたかは、Protocol の `rule_json` に持ち、`1 GPU / 2 GPU 以上` の切り替えは `world_size` から決める（MP-SENet）。
- 方針の乱数は、実際の公式の乱数列（torch、Python `random`）をそのまま使える場合は使い（`randperm`、`random.shuffle`）、torch の版が違うと順列が変わることを `meta` に記録する（UNKNOWN: 公式との版差による順列の一致）。

### 8.3 ESPnet ChunkIterFactory の rank 対応（`chunking.py`）

TF-GridNet（2 rank × 2）と TF-Locoformer（4 rank × 1）の計画。現行の `chunk_plan.py` は `world = 1` の場合だけを実装している（確認）。

```
plan_chunk_epoch(snapshot, spec, epoch, world):
    order     = base_utterance_order(snapshot)          # ESPnet のバッチ列に当たる基底順（U-DB-8）
    for rank in range(world):
        shard   = order[rank::world]                    # ESPnet: batches[rank::world]
        picked  = select_utterances(shard, epoch, seed, num_samples_per_epoch)   # rank あたりの発話数
        cache   = num_cache_chunks // world
        batches = chunk_batches(picked, epoch, seed, chunk_length, shift, per_rank_batch, cache)
        rank_batches[rank] = batches
    updates = min(len(b) for b in rank_batches)         # iterator_stop
    rank_groups[rank] = rank_batches[rank][:updates]    # 余りは捨てる。発話の消費位置は進む
```

- `per_rank_batch = global_batch // world`。TF-GridNet は `4 // 2 = 2`、TF-Locoformer は `4 // 4 = 1`。割り切れないときは拒否する。
- 各 rank の `chunk_batches` は同じ `RandomState(epoch + seed)` から**自分の発話列について**独立に始める（ESPnet は rank ごとのプロセスで同じ seed を使う）。rank 間で乱数が共有されないことは、`plan` が rank ごとに新しい `RandomState` を作ることで再現する。
- epoch 境界: `num_samples_per_epoch`（rank あたりの元発話数）が 1 epoch。選択の位置は `num_samples * (epoch − 1)` を `N_rank` で割った余りから進む（現行の `select_utterances` と同じ式を rank の shard に適用）。**捨てたチャンクバッチは epoch をまたいで繰り越さない**（ESPnet と同じ。`stats` に捨てた数を記録）。
- 更新数: TF-Locoformer の 1 epoch は約 14,000 更新（1,000 発話 × 14 チャンク / rank あたり batch 1）。TF-GridNet は発話の長さに依存し、実測で決まる。
- 勾配蓄積: TF-GridNet / TF-Locoformer は `accum_grad 1`。更新グループは 1 micro-batch。MossFormerGAN（蓄積 2）は、rank あたり 2 micro-batch が 1 グループ。
- validation: `valid` も同じ shard 規則 `[rank::world]` で、shuffle なしのチャンクを rank あたり batch で回す。ESPnet の `validate_one_epoch` は学習と同じ `iterator_stop` なので、最短 rank に揃えてから、残したバッチの損失合計と件数を all-reduce し、件数で割る。揃えたあとの平均は、同じバッチを 1 プロセスで重み付き平均した結果と一致する。捨てた尾は 1 プロセスで全件を評価した結果には入らない。
- `global_batch % world_size != 0` は拒否する。ESPnet は `rank < batch_size % world_size` の rank に 1 件多く渡す。公開の TF-GridNet（4/2）と TF-Locoformer（4/4）は割り切れる。
- 再開位置: epoch 開始から**コミット済みの更新数**。各 rank は同じ `update_in_epoch` から始める（§9.3）。

### 8.4 rank 数・割り当ての扱いの一覧

| 項目 | 実 DDP | virtual rank |
|---|---|---|
| rank 数 | `WORLD_SIZE`（torchrun など）| `data.virtual_world_size` |
| 計画 | 全 rank が同じ `plan(epoch, world)` を計算 | 同じ |
| 各 rank の入力 | 自分の `rank_groups[rank]` | 1 プロセスが rank 0…N−1 を順に処理 |
| 1 更新の勾配 | 各 rank の micro-batch の勾配を `all_reduce`（平均） | N 個の rank の micro-batch を順に backward し、1/N を掛けて蓄積。更新は 1 回 |
| validation | 各 rank が shard を処理し、合計を `all_reduce` | 全 shard を 1 プロセスで順に処理し、合計 |

### 8.5 virtual rank と実 DDP の違い（記録して受け入れる）

virtual rank は、計画（割り当て、チャンク、バッチ構成、epoch 境界）を実 DDP と**同一**にする。次は同一にならない。

| 違い | 内容 | 影響の扱い |
|---|---|---|
| 勾配の集約 | 実: rank ごとの平均損失の勾配を平均。virtual: 蓄積で同じ平均。数学的には同じだが、浮動小数の加算順が違う | D-09: ビット一致でなく統計的同等 |
| 損失の正規化 | `reduction sum` や `normalize_variance` は rank のバッチごとに計算される。virtual も micro-batch ごとに計算し、**全体のバッチで一度に計算しない** | 実装規則。テストで固定 |
| バッチ統計を使う層 | BatchNorm などは rank ごとの統計が違う。TF-GridNet・TF-Locoformer は LayerNorm 系で影響しないと想定（**未確認**）。MP-SENet は InstanceNorm 系 | 該当する層があれば virtual rank を同等と扱わない |
| モデル側の乱数 | 実 DDP は rank ごとに乱数の流れが違うか同じか（ESPnet の `set_all_random_seed` の挙動）が **UNKNOWN** | `meta` に記録し、dropout を持つモデルで影響を測る |
| 勾配クリップ | 両方とも集約後に 1 回。micro-batch ごとにクリップする MossFormerGAN（公式の癖）は、公式どおり micro-batch ごと | 規則を Trainer 側で固定 |
| 非有限勾配 | ESPnet は非有限勾配の更新を飛ばす。実 DDP は全 rank の判定が揃う。virtual は 1 プロセスの判定 | 同じ判定規則を使う |
| メモリ・速度 | virtual は N 倍の逐次処理。メモリは小さい | 再現性の検証用で、速度の目的には使わない |

実機の複数 GPU の利用可否・台数は**未確認**（IMPLEMENTATION_PLAN の確認事項）。最初の検証は virtual rank で行い、実 DDP は GPU 確保後に同じ計画で行う。

---

## 9. 乱数・再現性・再開

### 9.1 seed の導出

すべて `mix_seed(*parts)`（SHA-256。現行の `src/training/core/rng.py`）から導く。

| 用途 | 導出 | 備考 |
|---|---|---|
| 順序 | `("order", seed, epoch)`（現行互換）| 方針ごとの専用の seed は §8.2 |
| crop | `("crop", seed, epoch, case_id)`（現行の `crop_start`）| 黄金テストで固定 |
| チャンク | `("chunk", seed, epoch, rank)` から `RandomState` | ESPnet は `RandomState(epoch + seed)`。公式の再現が必要なら公式式を `espnet_chunk_v1` が持つ |
| オンライン混合（DFN3） | `("dfn3-online", protocol_id, snapshot_hash, seed, split, epoch, index)` | valid / test の epoch は 0。`index` はデータセット全体の通し番号。rank に依存しない。SEMamba++ は未実装 |
| 事前生成 | `("gen", build_seed, gen_index)` | DB 設計 §9.2 |
| モデル初期化・学習中の乱数 | 現行の `seed_model_init`、`seed_training_stream` | 変更しない |

データ側の乱数は、すべて**キーから毎回導く**。共有の `Generator` を持たないので、**保存すべき乱数状態が無い**。これが、オンライン混合の「RNG 状態」の再現の答えである。DeepFilterNet3 が記録するのは `protocol_id`、`snapshot_hash`、`seed`、`epoch`、`index` で、乱数器の内部状態は保存しない。

### 9.2 再現性の記録

学習 run が記録する項目と、その場所。

| 項目 | 記録先 |
|---|---|
| Git commit、ライブラリ版、環境 | `metrics.jsonl` の `run` 記録、`data_provenance.json`（既存の `capture_environment`）|
| Protocol の ID・名前・版・`algorithm_version` | `data_provenance.json`、checkpoint の `data_state` |
| Dataset の ID、スナップショットのハッシュ、素材リスト・ケース集合のハッシュ | 同上 |
| 公式リストの出典ファイルの SHA-256 | スナップショットの `dataset.json`（run が参照）|
| seed（データ・モデル・学習）| `data_state`、`run` 記録 |
| epoch、epoch 内の更新数（コミット済み）、epoch の計画の `plan_hash` | checkpoint の `data_state` |
| 計画の状態 | 純関数なので、`(seed, epoch, world_size)` と `plan_hash` で足りる。`chunk_history`（現行）は、`plan_hash` の履歴として残す |
| 消費位置 | `update_in_epoch`（更新単位）。互換のため `micro_batches_consumed` も書く |
| オンライン RNG の状態 | 保存しない（§9.1）。`seed`、`epoch`、`index` から導く |
| world size、rank 数、`ddp_mode`（single / real / virtual）、`virtual_world_size` | `data_state` |
| チャンク計画 | `chunk_spec` と `plan_hash`、`stats`（捨てたバッチ数）|
| checkpoint 選択の対応 | 保存した各 checkpoint の（epoch、update、検証指標の値）の表。`ave_5best` の元になった 5 つの epoch も記録 |
| 選択に使った集合 | `selection_uses_test`、集合の `case_set_id` と素材の SHA-256 の集合のハッシュ |

### 9.3 再開

- 位置の単位は**更新**（`optimizer.step` の回数）。1 更新 = 蓄積した micro-batch 全体。checkpoint は更新の境界でだけ書く（現行の `micro_batches_consumed` は蓄積の倍数。新しい位置は `update_in_epoch = micro_batches_consumed / accumulation` で、互換として両方書く）。
- 再開時: `epoch` と `update_in_epoch` を読み、`plan(epoch, world_size)` を再計算し、`plan_hash` が checkpoint の値と一致することを確認し、`update_in_epoch` 個の更新を飛ばして続ける。**一致しなければ拒否する**（スナップショット、`seed`、`world_size`、`virtual_world_size`、`global_batch`、`accumulation`、`algorithm_version` の違い）。
- `world_size` を変えた再開は、**既定で拒否**する（計画が変わる）。`--allow-world-change` を付けた場合だけ、次の epoch の境界から、新しい計画で続け、`data_state` に履歴を残す。再現用の run では使わない。
- 現行 Legacy の再開（`_check_resume`）は変えない。

### 9.4 prefetch の未消費分

- loader は、自 rank の次の数更新を先読みして render する。**位置は optimizer が更新を終えたときにだけコミット**する（先読みしたが使っていないものは位置に数えない）。
- checkpoint は、コミット済みの `update_in_epoch` だけを保存する。先読み済みの未消費データは**破棄**する。再開時は、同じ位置から再計算・再 render する。
- 先読みは乱数を引かない（§9.1）ので、破棄しても、後続のサンプルの内容は変わらない。これが現行の「1 グループ先読み」を安全にしている性質で、新経路でも保つ。
- 異常終了（OOM、ジョブの打ち切り）で、先読みの結果が一部失われても、同じ結果になる（render の失敗は例外で検知し、**そのサンプルを飛ばさない**。計画どおりに再試行するか、失敗として止める）。

---

## 10. 固定 validation / test

- validation と test の Case は、**ケース集合として DB に固定**し、スナップショットで run に固定する。オンライン混合でも、validation / test は `online_snapshot`（実現値の文書を固定）にする。
- crop の規則は Protocol の `valid_crop` に持つ: `per_epoch_random`（MossFormerGAN の公式: 検証にも random crop。`(seed, epoch, case_id)` から導く）、`fixed`（leakage-safe の既定: `(seed, "valid", case_id)`）、`full_length`（MP-SENet）。
- 公式再現で test を validation に使う場合: Dataset 定義の `selection_uses_test = true` と、学習設定の `data.allow_test_selection: true` の両方が要る。どちらか一方では起動しない。
- test 集合を validation に使わない leakage-safe では、test は学習中に読まない。学習後の独立評価は Research Evaluation（§13）。
- 事前生成の validation（`cv`）は、学習の epoch と独立に、毎回同じチャンク列で評価する（shuffle なし）。

---

## 11. 性能の比較

目的: 混合の分布と処理順序を変えずに、読み込みを速くする。比較した選択肢と判定。

| 選択肢 | 内容 | 長所 | 短所・リスク | 判定 |
|---|---|---|---|---|
| (a) epoch の開始時に DB から一括取得 | 各 epoch の開始時に必要な行を SQL で取る | 実装が簡単 | 学習が DB を開く。DuckDB の版差・不在（`.venvs/deepfilternet` に無い）、ロック、Lustre 上のファイル共有の問題が出る | 不採用。**取得は run の開始時に 1 回、スナップショットから**（実質、全 epoch で 1 回）|
| (b) worker ごとの読み取り専用接続 | 各 worker が DB を read-only で開く | 行の取得が並列 | 接続の起動と版差。書き込みプロセスと衝突する | 不採用 |
| (c) メモリ内の索引 | スナップショットから `AssetStore`（配列）を作る | 取得が配列参照で、DDP の各 rank が持てる | メモリ。TF-Locoformer S4 は素材 108 万行 | **採用**。生成物の素材は、パスを `(build, stream, gen_index)` から**計算**し、文字列を持たない |
| (d) オンデマンドの decode | 使うときに部分読み出し（`soundfile`）| ディスクを節約。チャンクは区間だけ読める | decode の CPU 時間。WAV は軽い（FLAC は重い）| **採用**（既定）。事前生成は PCM_16 の WAV なので軽い |
| (e) リサンプルのキャッシュ | 派生素材を準備時に作る | 学習中のリサンプルが無い | ディスクを使う。レートが違う素材のみ | 固定ペア（VoiceBank の 16 kHz）は**派生素材を準備時に 1 回**作る（DB 設計 §4.3）。オンライン混合の素材は、公式が読み込み時にリサンプルするものを派生素材にする（DFN3 は 48 kHz のままで不要）|

追加の最適化:

- 素材の SHA-256: 準備・スナップショット作成時に検証し、学習中は**既定で再計算しない**（`data.verify_on_read: false`）。小さい validation 集合では `true` を許す。
- 数値は**未測定**。性能は実装フェーズ 5 以降に、`load_workers`、`worker_kind`（スレッド、必要ならプロセス）、prefetch の深さを変えて測り、結果を記録する。測定前にスレッドかプロセスかを決めない。オンライン混合は numpy の処理が GIL を握る時間が長いと想定され、**プロセス並列が必要になる可能性が高い**（想定であり、測定していない）。
- 変えてはいけないもの: 混合の確率分布、処理の順序、各サンプルの実現値。性能のための変更は、同じ `plan_hash` と、同じ（または許容内で同じ）波形を黄金テストで確認する。

---

## 12. 設定とコマンド

### 12.1 設定ファイルの分担

| ファイル | 内容 | 例 |
|---|---|---|
| `configs/training/<model>.yaml`（既存）| モデル、optimizer、scheduler、損失、epoch 数、global batch、蓄積、seed、`data:` セクション | `configs/training/mp_senet.yaml` |
| `configs/data/protocols/<name>.yaml`（新規）| Protocol の規則（`algorithm`、`algorithm_version`、`rule`、`official_ref`、`deviation_class`、`deviations`）| `mp_senet.fixed_pair.yaml` |
| `configs/data/datasets/<name>.yaml`（新規）| Dataset 定義（`protocol`、`profile`、`claim_label`、バインド、素材の出所）| `mp_senet.vb28.official.yaml` |
| `configs/data/builds/<name>.yaml`（新規）| 事前生成（`generator`、`params`、`seed`、入力素材、`shard_size`、段階）| `tf_locoformer.dns.yaml` |
| `configs/data/lists/<model>/`（新規）| 公式リストの複製と `NOTICE.md`（出典、commit、SHA-256、ライセンス）| `mp_senet/training.txt` |

学習設定の `data:` セクション（設計）:

```yaml
data:
  pipeline: reproduction            # legacy（既定）| reproduction
  dataset: mp_senet.vb28.official@1  # name@version
  allow_test_selection: true         # selection_uses_test の Dataset で必須
  virtual_world_size: null           # 1 プロセスで N rank を再現する（N 以上の整数）
  load_workers: 4
  prefetch: true
  verify_on_read: false
```

- `pipeline: legacy` のとき、現行のキー（`database`、`train_set`、`validation_set`、`rms_normalize`）だけを許す。`pipeline: reproduction` のとき、これらのキーを拒否し、`dataset` を必須にする。未知のキーは拒否する（現行の規則）。
- epoch 数、batch、optimizer などは学習設定、**データの意味を決めるもの（正規化、窓、順序、チャンク、epoch の単位、validation の crop）は Protocol**。公式値と学習設定の global batch が違うときは、起動時に警告し、`data_state` に記録する（拒否はしない。leakage-safe の比較で変えるため）。

### 12.2 コマンド（P1 の 5 コマンドと、固定ペアの `reproduction-prepare`、MS-SNSD のオンデマンド登録は Makefile にある）

| コマンド | 引数 | 責務 | 冪等性 | 失敗条件 |
|---|---|---|---|---|
| `make setup PLATFORM=tsubame\|shinoda_lab` | 既存 | 環境（仮想環境、アセット）だけ。**データと DB を作らない**。既存の動作を変えない | 既存どおり | 既存どおり |
| `make reproduction-doctor` | `MODELS=`（任意）| 読み取りのみ。DuckDB の版（環境ごと）、`data/corpora`、`resource.duckdb` の状態（`corpora` / `eval` の有無）、`data/generated` の空き容量、ライブラリの版（`soundfile`、`librosa`、`soxr`、`torch`）、公式リストの有無を表示 | 常に | 致命的な不足があれば非 0（変更はしない）|
| `make reproduction-init-db` | — | `reproduction.duckdb` を作る（`repro` スキーマ）| 同じスキーマなら何もしない | 既存で違うスキーマなら**変更せず拒否**（現行の `db-init` と同じ規則）|
| `make reproduction-prepare` | `STEP=diagnose\|lists\|resolve\|register\|snapshot\|verify\|all`、`WORKERS=` | MP-SENet と MossFormerGAN のオンデマンド VoiceBank 固定ペアを staging に書き、登録し、スナップショットを出す。16 kHz WAV は書かない | 同じコーパス素材と同じ ID は再利用する。内容が違えば上書きせずエラー | 素材が無い場合は `BLOCKED`（終了コード 3）。`derive` は未知のステップとして拒否する |
| `make reproduction-plan-build` | `MS_SNSD_CONFIG=` | MS-SNSD の計画だけを書く。混合 WAV は作らない。既定は `ms_snsd_tfgridnet_ondemand.yaml` | 同じ計画を再度書ける | `delivery: on_demand` が無い。`source_scan: full` が無い上限なし走査 |
| `make reproduction-verify` | `BUILD=` または `DATASET=`、`HASH=1`（任意）| 検査（件数、サイズ、SHA-256、分布、leak）を `audits` に記録 | 常に（追記）| 不整合は `FAIL` を記録して非 0。自動で直さない |
| `make reproduction-register` | `MS_SNSD_CONFIG=` | オンデマンドの条件と元素材を登録し、スナップショットを書く。混合 WAV は作らない。公式規模は `ms_snsd_tfgridnet.yaml` の `source_scan: full` | 同じ内容なら何もしない | 条件検査が PASS でない。`source_scan: full` が無い上限なし走査 |
| `make reproduction-snapshot` | `DATASET=` | スナップショットを書き出す（DB 設計 §10）| 同じ内容なら何もしない。内容が違えばエラー | `declared_status` が `ready` でない。監査が `PASS` でない。素材の大きさが違う |
| `make reproduction-rebuild-db` | `OUT=` | staging から**新しい** DB ファイルを作り、全表のハッシュを比較 | `OUT` が既存ならエラー | ハッシュの不一致 |

- 30 分以上のジョブ（公式規模の素材ハッシュ、P7 以降の生成）は、TSUBAME へ投入する前に、ユーザーの明示の指示または追加承認が要る。優先度は `-p -5`。`h_rt` を 29 分にして回避しない（AGENTS.md）。混合 WAV を書く Make ターゲットは無い。
- どのコマンドも、`data/db/resource.duckdb`、`data/corpora/`、`data/checkpoints/`、`data/results/`、`data/staging/` の既存の内容を削除・上書きしない。`make db-ingest` を呼ばない。
- 学習の起動は、既存の学習 CLI（`configs/training/<model>.yaml` を渡す）を使う。新しいコマンドは足さない。

---

## 13. 評価との関係

| 層 | Official Evaluation | Research Evaluation |
|---|---|---|
| 目的 | 公式のスクリプトの指標（MP-SENet の `compute_metrics`、`speechscore`、ESPnet の `enh_scoring` など）で、公式再現の結果を公式と同じ土俵で見る | **固定した ASR による WER（主）と PESQ / STOI（補助）**、独立した評価用の test（モデル選択に使っていないもの）|
| 入力 | 公式の test リスト（`reproduction.duckdb` の `test.cases`）。評価時に Manifest を読んでよいのは、この層だけで、`src/evaluation/` の本線とは別（将来 `src/evaluation/official/` に置く案）| `resource.duckdb` の**名前付き評価セット**だけ（既存の規則）|
| 結果の置き場 | `data/results/se-official-eval/<run_id>/`（将来）| `data/results/se-evaluation/<run_id>/`（既存）|
| 主張 | 公式再現の指標。`claim_label` と `selection_uses_test` を併記 | 独立性を満たす評価だけを「独立」と書く（DB 設計 §7.4）|

- `resource.duckdb` は変更しない。`src/evaluation/` は `src.reproduction.*` を import しない。P4 以降、評価は checkpoint 横の `data_provenance.json` だけを読み、`independent` と `checkpoint_selection_overlap` を結果に書く。DuckDB も学習パッケージも要らない。
- 学習と評価で**共有するもの**: レシピ JSON v1 とハッシュ（`src/data/recipe.py`）、素材の索引の出所（`corpora.audio_files`）、素材の SHA-256 と `mix_seed` 型の seed 導出の規則、音声ファイルの読み込み。
- **分離するもの**: 学習データの生成（公式の混合、事前生成、オンライン混合）。評価の `rms_v1` 混合とは定義が違うので、評価コードに入れない。公式再現の「test を validation に使う」データ定義は、評価 DB に持ち込まない。
- 公式再現の checkpoint を Research Evaluation で評価するとき、その評価セットの素材が、checkpoint の選択に使った集合と重なるなら、結果に `independent: false` と重なった集合を記録する（IMPLEMENTATION_PLAN フェーズ 4）。

---

## 14. Trainer への変更（最小）

| 変更 | 内容 | 理由 |
|---|---|---|
| `TrainingData` の導入 | `Trainer.fit` が `dataset` の代わりに `TrainingData` を受ける。Legacy は `legacy.py` が包む | 3 方式と DDP を同じ入口で扱う |
| 位置の単位 | `update_in_epoch` を保存。`micro_batches_consumed` は互換のため併記 | §9.3 |
| `data_state` の保存・照合 | `dataset.identity()` を `data_state` に拡張し、再開時に `plan_hash` などを照合 | 再現性 |
| チャンク epoch の DDP | `_run_chunk_epochs` の `world_size != 1` の拒否を、`TrainingData` 経由で `world_size ≥ 1` を許す。DDP のラッパは既存の `distributed.py` | TF 系の 2〜4 rank |
| validation の集約 | 合計と件数の `all_reduce`。virtual rank は 1 プロセスで合計 | §8.3、§8.4 |
| checkpoint 選択の追加 | TF-Locoformer の `valid.loss.ave_5best` は chunk Trainer に実装済み（[P7_VERIFICATION](P7_VERIFICATION.md)）。MossFormerGAN の halving 規則は未実装 | D-09 |
| 変更しないもの | モデル、損失、optimizer、scheduler、AMP、`save_atomic`、metrics ログ | 既に検証済みの部分を壊さない |

---

## 15. 未確定事項

IMPLEMENTATION_PLAN の「未解決事項」にも集約する。

| ID | 内容 |
|---|---|
| O-3 | スナップショットの `.npz` の列構成と文字列の格納の詳細（形式は `numpy` + JSON と決定済み。列の詳細は実装時）|
| O-4 | **確認済み（2026-11-06）。** 基底順は `UnsortedBatchSampler` が読む shape file の順。ファイル自体は事前生成の前には無い。`plan_chunk_epoch` はそのリストを並べ替えず、呼び出し側の順を基底にする。発話選択は `SequenceIterFactory` と同じ `num_samples * (epoch - 1)` からの窓で、`0d136c5` と `90eed8e` で同じ式 |
| O-5 | **確認済み（2026-11-06）。** `0d136c5` と `90eed8e` の `validate_one_epoch` は学習と同じ `iterator_stop`（いずれかの rank が尽きたら残りを捨てる）。実装はその最短長に揃えてから、残したバッチの損失合計と件数を集約する |
| O-6 | **確認済み（2026-11-06）。** 両版とも epoch 開始時に `set_all_random_seed(seed + epoch)` を各プロセスで同じ引数で呼ぶ。rank ごとの seed は足さない。TF-GridNet と TF-Locoformer DNS の dropout は 0 で、この seed は強調の forward を変えない |
| O-7 | 公式の乱数列（torch `randperm`、Python `random.shuffle`）が、公式と audio-db の torch の版の違いで一致するか |
| O-8 | スレッド並列で、オンライン混合の処理が足りるか（プロセス並列の要否）。測定が要る |
| O-9 | 実機の複数 GPU の台数と利用可否 |
