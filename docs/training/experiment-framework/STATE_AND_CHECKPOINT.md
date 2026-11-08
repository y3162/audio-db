# 状態と Checkpoint

Phase 2-A は `audio-db-experiment-v1` を CPU / FP32、world size 1 で保存し、復元する。Phase 2-B は同じ形式のまま、CPU / Gloo の world size 2 で rank 0 が 1 ファイルを書く。shard は作らない。現行形式 `audio-db-training-v1` は `CheckpointError` で拒否する。CPU と CPU/Gloo の復元は Prepare と Optimizer 生成の後である。理由は [PHASE_2A_CHECKPOINT.md](PHASE_2A_CHECKPOINT.md) と [PHASE_2B_DDP.md](PHASE_2B_DDP.md)。以下は Phase 0 の仕様である。

## 再現性の三層

言葉を混ぜない。

| 層 | 意味 | この設計が Phase 2 で保証しようとすること |
|---|---|---|
| State Restore | 保存したテンソルとカウンタが戻る | 登録した state_dict のキーとテンソルが一致する |
| Sequence Reproduction | その後に読むサンプル列が一致する | DataModule の `state_dict` と純関数計画の `plan_hash` が一致する |
| Numerical Reproduction | その後の浮動小数の更新が一致する | 決定的アルゴリズムを有効にした CPU の試験に限る。非決定的 CUDA のビット一致は保証しない |

CUDA で RNG を戻せることと、非決定的カーネルを含む学習がビット一致することは別である。現行 `.agents/DEVELOPMENT.md` の区別を、新経路でも同じ言葉で書く。

公式再現の統計的同等性は、この checkpoint 仕様だけでは成立しない。データ式、損失、更新順が現行実装と一致して初めて比較できる。Phase 4 の移行試験がそれを測る。未測定の条件をこの文書は保証しない。

## StateRegistry の契約

登録単位は `StateOwner` である。

| フィールド | 意味 |
|---|---|
| `state_id` | 実験内で一意な文字列。既定は component id |
| `kind` | `module`、`optimizer`、`scheduler`、`trainer`、`data`、`runtime` のいずれか |
| `owner_key` | オブジェクトの `id()`。重複検出に使う |

規則:

- 同じ `owner_key` を二回登録しても保存エントリは 1 つである。二番目の `state_id` は別名として manifest の `aliases` に残る。
- 違うオブジェクトが同じ `state_id` を渡すと `StateError`。
- `state: {save: false}` のコンポーネントは登録しない。
- loss と metric は既定で登録しない。
- Runtime はプロセスに 1 つ、`state_id` を `runtime` として登録する。中身は Scaler の状態と、分散の world size、backend 名である。学習ループの位置は持たない。
- パラメータ集合のフィンガープリントは、各パラメータの `id()` をソートしたタプルのハッシュである。module と optimizer の両方が同じパラメータを参照してよい。それは所有の重複ではなく、参照である。Optimizer 同士で集合が交わることだけを `BuildError` にする。
- `run` が始まったあとの `register` は失敗する。

状態の意味は所有者が知る。CheckpointManager は `state_dict` の中のテンソルをバイト列として扱い、学習率や epoch の意味を解釈しない。

## 状態の所有者

| 所有者 | 保存するもの | 保存しないもの |
|---|---|---|
| module の `state_module` | そのモジュールの `state_dict` | DDP ラッパーのバッファ複製。unwrap した結果だけ |
| optimizer | `optimizer.state_dict` | パラメータ本体の複製。パラメータは module 側 |
| scheduler | `state_dict`。自作 Scheduler は現行 `WarmupReduceLROnPlateau` と同様に Optimizer オブジェクト自体を除外する | Optimizer への参照 |
| trainer | logical step、Optimizer 別 update count、停止フラグ、検証履歴、アルゴリズム固有のカウンタ | モデル重みの複製 |
| data | epoch、次に読む update index、plan_hash、world size、分布を変えた場合の重み | 先読み済みで未消費の波形 |
| runtime | world size、precision、GradScaler | 他 rank のパラメータ |

Trainer の logical step と、Optimizer ごとの update count は別フィールドである。

```text
logical_step: 4
optimizer_updates:
  optim_g: 4
  optim_d: 3
```

この例は、4 回のアルゴリズム反復のうち Discriminator の `step` が 1 回 `applied=False` だった、という意味である。現行 `global_step` へ単純に写すときは、移行アダプタがモデルごとの対応を明示する。汎用の「どちらか大きい方」は使わない。

## Checkpoint Manifest

ファイルは 1 つで、`torch.save` の辞書である。形式名は `audio-db-experiment-v1`。

```text
format: audio-db-experiment-v1
schema_version: 1
logical_step: int
label: step | validation_pending | init
world_size: int
rank_count_in_rng: int
component_ids: [str]
state_ids: [str]
aliases: {alias_id: canonical_state_id}
states: {state_id: blob}
rng_by_rank: [per-rank blob]
config_sha256: str
code_identity: {git_commit, dirty}
dataset_identity: dict | null
provenance_sha256: str | null
optimizer_updates: {component_id: int}
validation_pending: dict | null
```

`schema_version` は整数 1 から始める。ローダは知らない version を拒否する。未知フィールドが payload にあっても、version が 1 なら拒否する。前方互換の無視はしない。

`config_sha256` は、解決後の ExperimentConfig を正規化 JSON にした SHA-256 である。`resume` は、`output` とログ設定を除き、このハッシュが一致しなければ `CheckpointError`。現行の「`logging` 以外の設定差を拒否する」に合わせ、新形式では除外キーを `output.logging` だけにする。ディレクトリの変更を伴う再開は、同じ実験ディレクトリを前提とし、別ディレクトリへのコピー再開は Phase 2 では拒否する。

`code_identity` は記録する。不一致だけでは再開を拒否しない。拒否すると、コメントだけの修正で再開できなくなる。数値を変えるコード変更は、利用者が `code_identity` を見て判断する。preflight の報告には差分の有無を書く。

`dataset_identity` は DataModule.`identity()` のコピーである。`resume` 時に現在の identity と不一致なら `CheckpointError`。データが無い実験では null であり、現在も null でなければならない。

reproduction 経路の DataModule は、現行と同じ `data_provenance.json` を rank 0 が一度だけ排他作成する。checkpoint の `provenance_sha256` とそのファイルが不一致なら再開しない。来歴の意味は現行 `src/training/provenance.py` に従い、新仕様はハッシュの紐付けだけを継承する。評価の `independent` 判定は現行のまま、この JSON を読む。新形式だからといって判定を変えない。

## Schema Version

| version | 内容 |
|---|---|
| 1 | この文書の単一ファイル、rank 0 書き込み、全 rank RNG を同じファイルに格納 |

version 2 を shard や動的 Expert のために足すときは、version 1 のローダを残す。自動変換は、変換手順と試験が [MIGRATION_PLAN.md](MIGRATION_PLAN.md) に書かれてからだけ行う。

## Save と Restore の順序

保存は rank 0 だけが行う。全 rank が先に Runtime の RNG を `all_gather_object` する。

1. Trainer が `end_step` まで終えていることを、呼び出し側が保証する。CheckpointManager は勾配の有無を検査しない。
2. 全 rank が自分の RNG を gather する。
3. rank 0 が StateRegistry の id を辞書順に `state_dict` する。
4. 同一ディレクトリの一時ファイルへ書く。`fsync` する。
5. 読み戻し、`format`、`logical_step`、`state_ids` が一致することを確認する。現行 `save_atomic` の検証と同じ目的である。
6. `os.replace` で `step_{logical_step:08d}.pt` にする。
7. 全 rank が barrier する。

復元:

1. 全 rank が同じパスを読む。読めないファイルは `CheckpointError`。
2. `format` と `schema_version` を確認する。
3. `world_size` が現在と一致しなければ `CheckpointError`。
4. `state_ids` が現在の Registry と一致しなければ `StateError`。余分も不足も拒否する。
5. module の state を、Prepare の前に `state_module` へ載せる。
6. Prepare する。
7. Optimizer と Scheduler を構築する。
8. それらの state、trainer、data、runtime の Scaler を載せる。
9. 自 rank の RNG を戻す。world size 1 でも `rng_by_rank[0]` を使う。現行の「world size 1 は全 device の `rng`」とは別形式である。新形式は最初から rank 別だけを持つ。
10. DataModule の `plan_hash` を再計算し、保存値と違えば `CheckpointError`。

RNG の 1 rank 分は、現行 `capture_rank_rng` と同じく Python、NumPy、Torch CPU、使用中の CUDA device だけである。`get_rng_state_all` の未使用 device は保存しない。

## Resume と Initialize

| | `resume` | `initialize_from` |
|---|---|---|
| module 重み | すべて | manifest の `initialize.allow` に含まれる state_id とキーだけ |
| optimizer | 戻す | 戻さない |
| scheduler | 戻す | 戻さない |
| trainer の logical step と update count | 戻す | 0 のまま |
| data 位置 | 戻す | epoch 0、update 0 |
| RNG | 戻す | `seed` から新規に初期化する |
| provenance | 既存ファイルと一致を要求する | 新しい run の provenance を書く。元 checkpoint の provenance は `initialized_from` フィールドにハッシュだけ残す |

`initialize.allow` が payload に無いファイルを `initialize_from` に使うと `CheckpointError`。研究の開始重み用に保存するときは、`label: init` で allow リストを明示して書く。学習途中の `step_*.pt` を初期化元に流用する便利機能は Phase 2 に入れない。用途が違うファイルを同じ操作で開かない。

`resume` と `initialize_from` の同時指定は `ConfigError`。

## 分散 Checkpoint

Phase 2 は shard しない。全 rank のパラメータが一致している前提で、rank 0 の `state_dict` だけを module と optimizer に書く。Phase 2-B は保存前に、module の parameter と buffer、optimizer、scheduler、trainer、data の dtype、shape、値を all-gather して比べる。Python の `id()` は使わない。RNG は比べない。不一致なら `CheckpointError` で、完成ファイルは置き換えない。公式 DDP の重み割れをコピーした状態は、この検査で拒否する。それが目的である。

RNG だけは rank ごとに違う。`rng_by_rank` の長さは world size と一致する。

書き込み中にプロセスが死んだ場合、`os.replace` 前の一時ファイルは残ってよい。ローダは接頭辞 `.` の一時ファイルを checkpoint として開かない。完成ファイルは直前のものを残す。複数ファイルにまたがる原子性は、現行と同様、保証しない。正本は 1 つの `.pt` に寄せ、`metrics.jsonl` は追記ログとして、再開時にその logical step より大きい行を切り捨てる。切り捨ては rank 0 だけが行う。

`status.json` の値は現行と同じ五種を使う。`complete.json` があるディレクトリの `resume` は拒否する。`verification_max_updates` に相当する上限は、Trainer の kwargs に `stop_after_logical_steps` として書き、到達時は `INTERRUPTED` とし `complete.json` を書かない。

## RNG と Sampler

実行 RNG（Python、NumPy、Torch、現在 CUDA device）は Runtime が capture する。dropout や Noise がこれを消費する。

データの順序が純関数の Sampler は、RNG を保存しない。保存するのは epoch と次の update index と plan_hash である。現行の fixed pair、legacy `epoch_indices`、`espnet_chunk_v1`、オンライン混合がこれに当たる。アダプタは乱数生成器の残差を追加で保存しない。

`update_weights` を持つ Sampler は、重み表を data state に含める。表が無い古い state からの復元は拒否する。

先読みは位置に含めない。現行と同じく、消費が完了した update の次が `update_index` である。

## Commit 境界

checkpoint を公開してよいのは、次がすべて終わった後である。

- その反復の `runtime.end_step`
- Trainer が logical step を進めるか、進めないことを確定したあと
- rank 0 が `metrics.jsonl` にその反復の行を書いたあと
- TensorBoard を有効にしているときは event の flush のあと
- 次の `train_epoch.batch` を呼ぶ前

検証を行う反復では、現行と同じく二段にする。

1. `validation_pending` を付けた checkpoint を書く。
2. barrier する。
3. 検証する。
4. Scheduler を Trainer の規則で進める。
5. pending を外した checkpoint を同じ logical step で置き換える。

中断からの再開が pending を見たら、Optimizer 更新をやり直さず、検証を最初から行う。検証の途中バッチは保存しない。現行の決定と同じである。

PESQ skip のように Optimizer を進めない反復では、データ位置だけが進んだ checkpoint を書いてよい。logical step は進まない。ファイル名は進めない logical step を使うと上書きになるので、ファイル名は `step_{logical_step:08d}_data_{epoch}_{update}.pt` のようにデータ位置を含める。同じ論理 step の本保存と取り違えない。このファイルも manifest の `logical_step` は進んでいない値である。

## 整合性検証

`resume` が比較するもの:

- format と schema_version
- state_id の集合
- world size
- config_sha256（ログ設定を除く正規化設定）
- dataset identity
- data plan_hash
- provenance_sha256 と横の JSON
- パラメータフィンガープリントが保存時の rank 間一致を記録していること

比較しないもの:

- 壁時計、ホスト名、CUDA の空きメモリ
- `code_identity` の一致。報告だけする
- TensorBoard の event ファイル

復元後、module の `load_state_dict(..., strict=True)` が失敗したら `CheckpointError`。欠落キーを初期値のまま残さない。

## 将来の拡張に必要な前提

動的 Expert:

- 静的 ID と動的 ID を manifest で分ける必要がある。
- 動的 ID の Optimizer state は、その Expert が作られた logical step を持つ必要がある。
- それより前の checkpoint からの resume は、その Expert がまだ無い状態として明示される必要がある。無いことを初期化とみなす暗黙の処理は禁止する。

world size の変更:

- 単一ファイルのパラメータは流用できる可能性がある。
- RNG の長さと Sampler の分割は再利用できない。
- Phase 2 のローダは長さ不一致を拒否する。変換器を足すなら schema version を上げる。

FSDP:

- `state_module.state_dict` が rank ごとに違う。
- rank 0 の単一ファイルという Phase 2 の前提が崩れる。
- その時点で schema version 2 とし、shard の索引を manifest に入れる。version 1 のファイルを FSDP の実行へ resume しない。

## 現行形式との関係

`audio-db-training-v1` の `task` 辞書は、モデルごとのキーが違う。MP-SENet は `optim_g` と `optim_d`、TF-GridNet は `optim_g` だけ、MossFormerGAN は plateau と蓄積勾配を含む。

Phase 4 の移行アダプタは、モデルごとに「どのキーをどの state_id へ写すか」を表で持つ。自動推測はしない。写像が定義されていないキーがある checkpoint は拒否する。

現行ファイルを新ローダが直接開いて学習を続ける機能は、この Phase 0 の契約に含めない。
