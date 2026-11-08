# コンポーネントのライフサイクル

Phase 0 の仕様。実装は未着手。

## 段階

```text
Configure
    ↓
Build
    ↓
Prepare
    ↓
Bind optimizers
    ↓
Register state
    ↓
Restore / Initialize
    ↓
Run
    ↓
Finalize
```

`ExperimentBuilder.build` は Configure 済みの設定を受け、Build から Register state までを行う。Restore と Run と Finalize は `Experiment.run` が行う。preflight は Register state の直後で止まり、Restore も Run もしない。

| 段階 | 行うこと | 状態の変化 |
|---|---|---|
| Configure | YAML の検証。インスタンスは無い | 無し |
| Build | `role` が module、loss、data、metric、other の実体を Factory で作る。別名 `ref` を Registry に繋ぐ | ライブオブジェクトが生まれる |
| Prepare | 各 module を `Runtime.prepare_model` に渡す。ハンドルへ置き換える | module の Python オブジェクトが変わり得る。DDP ではパラメータオブジェクトは変わらない |
| Bind optimizers | optimizer と scheduler を作る。その後 trainer を作る | Optimizer がパラメータを掴む |
| Register state | StateRegistry へ所有者を登録する | 保存対象の集合が固定される |
| Restore / Initialize | checkpoint をその集合へ書く | 重み、Optimizer、RNG、データ位置 |
| Run | Trainer がループする | パラメータとカウンタ |
| Finalize | DataModule.close、TensorBoard を使った場合の close、process group の破棄、lock の解放 | 実行資源が閉じる |

Configure で失敗した実行はディレクトリを作らない。Build 以降で失敗した実行は、rank 0 が `status.json` に `FAILED` を書いてから Finalize する。`status.json` を書く前の失敗ではファイルを残さない。

## 生成順序

依存辺は次だけである。

- `inject` の `ref`
- `parameters_of` の component
- 別名 spec の `ref`

辺の先を先に作る。同じ段で依存が無いものは component id の辞書順である。この順はプロセス間で同じにする。

段の中では、その段の role だけをトポロジカルソートする。optimizer が module に依存していても、module の段が先なので辺は満たされる。trainer が optimizer に依存していても、trainer の段が最後なので満たされる。

段をまたいで逆向きの辺は `BuildError` である。例: module の `inject` が optimizer を指す。損失が Optimizer を必要とする構成は、損失クラスの中で作るか、trainer が両方を受け取って損失オブジェクトに渡す。Builder は後から module を組み立て直さない。

### 単一モデルの具体例

[PUBLIC_API.md](PUBLIC_API.md) の教師あり例は次の順になる。

1. Build: `data`、`loss`、`model`（id 順では data、loss、model）。`loss_for_log` は `loss` と同じオブジェクト。
2. Prepare: `model` を CPU に置き、`distribute: none` なので包まない。ハンドルの `module` と `state_module` は同一。
3. Bind: `optimizer` をハンドルの `parameters` で作る。続けて `trainer` を作る。`trainer` へ渡す model はハンドルである。
4. Register: `data`、`model`、`optimizer`、`trainer`。`loss` は role が loss なので登録しない。別名は登録しない。

### GAN の具体例

1. Build: `data`、`discriminator`、`generator`。
2. Prepare: id 順に `discriminator`、`generator`。
3. Bind: `optim_d`、`optim_g`、最後に `trainer`。
4. `optim_g` が掴むパラメータ集合と `optim_d` の集合は交わらない。Builder がパラメータ id の集合で検査する。

### ASR-aware の具体例

1. Build: `asr`、`data`、`enhancer`、`loss`。
2. Prepare: `asr` は `requires_grad=False`、DDP なし、eval モード。`enhancer` は学習対象。
3. Bind: `optimizer` は `enhancer` だけ。`asr` が trainable false なので `parameters_of: asr` は Build の検証で既に拒否されている。
4. `asr` は state を登録する。`initialize_from` で ASR の重みを上書きするかは、その checkpoint の allow リスト次第である。既定の研究設定では ASR の公開チェックポイントを `FrozenRecognizer` の kwargs で読み、`initialize_from` の対象に ASR を入れない。

### 複数 Expert の具体例

1. Build: `data`、`expert_a`、`expert_b`、`router`。
2. Prepare: その id 順。
3. Bind: `optim_experts` は expert_a と expert_b のパラメータを、YAML の lr 付きグループとして渡す。`optim_router` は router だけ。`trainer` は Expert のリストを inject の順で受け取る。リスト順は辞書順に並べ替えない。

## モデル準備と Optimizer 生成

順序は常に次である。

```text
module を構築する
    ↓
checkpoint のモデル重みを、resume の場合だけ state_module へ載せる
    ↓
Runtime.prepare_model
    ↓
optimizer を parameters ビューで構築する
    ↓
resume の場合、optimizer と scheduler と trainer と data と RNG を載せる
```

DDP は `prepare_model` の中で掛ける。現行 Task は「Optimizer を先に作り、state を載せてから wrap する」。パラメータオブジェクトは wrap の前後で同じなので、現行の数値は保たれる。新契約が Optimizer を wrap の後で作るのは、FSDP がパラメータを置き換える場合に、空の Optimizer を捨てて作り直す手順を避けるためである。DDP だけの Phase 2 でもこの順に固定する。

resume のモデル重みは wrap の前に載せる。現行 `load_state_dict` が DDP 済みモジュールを拒否するのと同じ理由で、DDP の `state_dict` キー接頭辞を新形式の通常経路にしない。wrap のあと rank 0 のパラメータを broadcast する処理は、PyTorch の DDP コンストラクタが行う。したがって resume 時は全 rank が同じ重みを wrap 前に載せる。rank 0 だけが読むファイルを、読み込み後に broadcast する実装でもよい。選ぶのは Runtime である。どちらの場合も、wrap の直前に全 rank の重みが一致していなければならない。不一致は `PrepareError`。

`initialize_from` もモデル重みだけを wrap 前に載せる。Optimizer は初期状態のまま構築する。

Scheduler のコンストラクタが Optimizer を受け取る場合、`inject.optimizer` の ref で渡す。Scheduler を Optimizer より先には作らない。

## 同一インスタンスの共有

`ref` は Registry から同じオブジェクトを返す。`StandardTrainer` が受け取る loss と、別名 `loss_for_log` に対する `get` の結果は `is` が真である。

ハンドル化したあとも、`ref: model` はハンドルを返す。wrap 前の生モジュールを Trainer に渡さない。Trainer が `state_dict` を自分で呼ぶときは `handle.state_module` を使う。

二つの component がそれぞれ `class:` を持っているとき、Builder は二つインスタンスを作る。同じクラスであっても共有しない。共有は `ref` だけが行う。

## DDP と FSDP

Phase 2 の `distribute: ddp` は現行 `wrap_module` の意味に合わせる。

- world size 1 では包まない。
- CUDA の複数プロセスは nccl と `device_ids=[local_rank]`。
- CPU の複数プロセスは gloo。
- `find_unused_parameters` と `broadcast_buffers` は YAML の `prepare` に明示した値を使う。既定は false と true。DeepFilterNet3 の未使用パラメータのように真が必要なモデルは、YAML に真を書く。
- `sync_batchnorm: true` は、CUDA かつ world size 2 以上のときだけ `SyncBatchNorm.convert_sync_batchnorm` を prepare の中で呼ぶ。CPU の複数 rank では BatchNorm のままにする。現行 MossFormerGAN と同じ条件である。

forward の同期は Trainer が `sync_context` で決める。最後の micro-batch だけ `sync=True` にするのが `StandardTrainer` である。MP-SENet 型の遅延 backward は、全 forward を `sync=False` にし、全 backward の後に `average_gradients` を呼ぶ Trainer として書く。

skip の合意は `Runtime.agree` で行う。MossFormerGAN 型は、いずれかの rank が micro-batch を落とすなら全 rank が落とす、という policy `any` を Trainer が選ぶ。policy の選択は Trainer のコードであり、YAML の制御構文にはしない。

FSDP は `distribute: fsdp` を Phase 5 まで拒否する。準備がパラメータを置き換えること、Optimizer をその後に作ること、checkpoint は FSDP の state dict API が必要であること、の三つを Runtime の拡張点として残す。Phase 2 の単一ファイル checkpoint は、全 rank で同じ通常の `state_dict` が取れることを前提にする。FSDP ではこの前提が偽になる。

world size が checkpoint と違う `resume` は `CheckpointError`。Phase 2 では合わせる手段を用意しない。

## リソース解放

Finalize の順:

1. Trainer が保持する先読み Future を cancel し、待つ。
2. DataModule.`close`。
3. rank 0 のログ writer を閉じる。
4. `dist.destroy_process_group`。初期化していなければ何もしない。
5. rank 0 の実験ディレクトリ lock を外す。

lock は現行 `experiment_lock` と同じく、ディレクトリの `.lock` に対する非ブロッキング排他である。rank 0 だけが取る。二重起動は `ExperimentError`。

`deadline` と `segment_steps` は Phase 2 の Runtime には入れない。現行 Trainer の管理停止は現行経路に残す。新経路で同じ停止が必要になった時点で、更新境界のあとだけ checkpoint する契約を Trainer に足す。SIGKILL の途中ファイルは `os.replace` の前なので、直前の完成ファイルが残る。

## 動的コンポーネント

`run` 中の `StateRegistry.register` は `StateError`。Expert を足す実験は、起動前の YAML に全 Expert を書く。

Phase 5 で動的追加を許すときの前提を、今の registry に残す。

- state_id は `component_id` と、動的枠の名前を含む安定な文字列である。
- 動的枠は YAML に `dynamic_slots: []` のように空で宣言した component だけが、後から子 ID を足せる。
- checkpoint manifest は静的 ID と動的 ID を分けて記録する。
- 復元時に、宣言していない動的 ID があれば `CheckpointError`。黙って捨てない。
- 子 Expert の Optimizer を実行中に作る場合も、パラメータビューは Runtime から取る。Trainer が生の `module.parameters()` を、準備前のオブジェクトから集め直さない。

Phase 1 の YAML パーサは `dynamic_slots` キーを未知キーとして拒否する。キーを黙って無視しない。
