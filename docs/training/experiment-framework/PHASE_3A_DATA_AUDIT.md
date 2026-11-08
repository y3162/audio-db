# Phase 3-A 既存 Data System の調査

2026-11-09 に、`src/reproduction/`、`src/training/data/`、`src/experiments/data.py` の現行コードを読んだ調査である。DuckDB は開いていない。スナップショットも素材も変更していない。公式実装との波形一致は測っていない。ここに書く数値条件は、コードと既存の設計書が実装として持っている条件であり、公式リポジトリとの一致検証ではない。

Phase 0 は、公式データ経路を DataModule アダプタで包むことを推奨していた。この調査のあと、その推奨は [DATA_ARCHITECTURE.md](DATA_ARCHITECTURE.md) で撤回する。撤回の理由は末尾にまとめる。過去の記述は削除していない。

## 調査で固定する観察

学習時に残すべき性質は次である。

- 学習プロセスは `reproduction.duckdb` を開かない。読むのは不変スナップショットと素材ファイルである。
- サンプルの並びと切り出しは、保存した乱数生成器の残状態ではなく、引数から再計算する。
- 先読みした未使用波形は checkpoint の位置に入らない。位置は消費済みの更新である。
- オンライン混合の式はモデルごとに別実装で、共通の混合エンジンは無い。
- Worker は乱数を引かない。DataLoader プロセスは無い。

公開面として残さない性質は次である。

- `ReproductionTrainingData`、`OnlineDFN3TrainingData`、`OnlineSemambaTrainingData`、`OnDemandChunkData`、`DnsOnDemandChunkData` は、順序、描画、バッチ、検証、再開を一つの duck type にまとめている。Experiment の `train_epoch` / `state_dict` とはメソッドが違う。
- `SampleRef.chunk` は型が `None` で、chunk 座標を載せられない。chunk は別の `PlannedChunk` と、Trainer 側の `_chunk_state` である。
- 検証の rank 分担はメソッド名は揃っていても、プロトコルごとに違う。
- `dataset_id` と `protocol_id` は name と version をハッシュに入れない。版番号だけの更新はできない。

## 3.1 Dataset 管理

スキーマ名は `repro`、`SCHEMA_VERSION` は `"1"`（`src/reproduction/ids.py`）。表の順は `src/reproduction/db/schema.py` の `TABLE_ORDER` である。`schema_info`、`catalog_snapshots`、`derivations`、`builds`、`assets`、`material_sets`、`material_set_members`、`protocols`、`cases`、`case_sets`、`case_set_items`、`datasets`、`dataset_bindings`、`audits`、`runs`。

分割用の表は無い。split は case set / material set の `split_role` と、binding の slot 名（`train.cases`、`valid.speech` など）である。`pool` の material set は bind できない。

### Dataset DB

| 項目 | 内容 |
|---|---|
| 既存実装 | `initialize_database`、`assert_ready`、`register_staging`、`read_graph`、`rebuild_database`、`Graph`（`src/reproduction/db/`） |
| 責務 | 内容アドレスの行を、staging の検証のあと 1 トランザクションで足す。同じ主キーで内容が違う行は拒否する |
| 公開 API | `register_staging(staging, database, *, resource_db=None) -> dict[str, int]`。失敗は `ReproductionError`。`PreparationBlocked` は素材不足で、CLI の終了コード 3 |
| 強み | 再構築元が staging である。`table_hashes` が一致したときだけ別ファイルの再構築を採用する。学習パッケージは `db` を import しない |
| 問題点 | `assert_ready` は列名と `column_fingerprint` を見て、`CHECK` の文面は見ない。`register_staging` はコミット前に audit を staging へ書く。カタログの行本体は DuckDB に無く parquet に残る |
| 新設計 | 登録の責務はこのパッケージに残す。Experiment の学習ループへ移さない |
| 移行方針 | 再利用。学習 API からは呼ばない |

### 識別子

| 項目 | 内容 |
|---|---|
| 既存実装 | `dataset_id_for`、`protocol_id_for`、`recipe_case_id`（`doc_kind = recipe_v1`）、realization 文書の `digest`（`doc_kind = realization_v1`）、`case_content_hash`、`snapshot_hash` |
| 責務 | 同じ定義が同じ ID になること。`case_id` 自身はハッシュ入力に含めない、というレシピ側の規則を `recipe_v1` が使う |
| 公開API | ID 関数は準備コードが呼ぶ。学習は計算し直さず、スナップショットに書かれた文字列を読む |
| 強み | `selection_uses_test` は保存値と `compute_selection_uses_test` の一致を要求する。blocked / unknown は binding を持てない。ready 以外は `export_snapshot` できない |
| 問題点 | `dataset_id` と `protocol_id` は name、version、binding の `params_json`、`deviations_json` をハッシュに入れない。同じ ID の label 変更は row digest が変わって拒否される。version 列は内容が変わらない限り増えない。`claim_label` はコードが 7 値で、`docs/db/REPRODUCTION_DB_DESIGN.md` の古い列挙は 6 値のままである |
| 新設計 | 学習側の Dataset Identity は、保存済みの `dataset_id` と `snapshot_hash` の組である。ID の式は学習側で再定義しない |
| 移行方針 | 再利用。ID 関数の改修は今回の Data System の範囲外 |

`profile` は `official_faithful` または `leakage_safe`。`declared_status` は `ready`、`blocked`、`unknown`。`claim_label` の実装値は `official_reproduction`、`protocol_conformant_substitute_materials`、`espnet_public_se_recipe_reproduction`、`official_recipe_scaled_down`、`minimal_repair_reproduction`、`leakage_safe_comparison`、`research_checkpoint_selection`。

アルゴリズム名は `ids.ALGORITHMS` に閉じている。`fixed_pair.mp_senet`、`fixed_pair.mossformergan_se_16k`、`online.dfn3`、`online.semambapp_gsr`、`pregenerated.ms_snsd_tfgridnet`、`pregenerated.dns_tflocoformer`。未知の algorithm と未知の rule キーは拒否される。

### Snapshot

| 項目 | 内容 |
|---|---|
| 既存実装 | `export_snapshot`（`src/reproduction/db/snapshot.py`）、`load_snapshot`、`snapshot_hash`、`AssetStore.from_snapshot`（`snapshot_reader.py`、`pipeline/store.py`） |
| 責務 | ready な 1 dataset を、DuckDB 無しで読めるディレクトリにする |
| 公開API | `load_snapshot(directory) -> LoadedSnapshot`。`provenance.format` は `reproduction_snapshot_v1`。部品は `dataset.json`、`assets.npz`、`generated.json`、`cases.npz`、`docs.jsonl`、`case_set_items.npz`、`material_set_members.npz`、`manifest.json`。`snapshot_hash` は部品の digest で、manifest の provenance ブロックはハッシュに入らない |
| 強み | ファイル配置ではなく部品内容が ID になる。同じハッシュの再 export は no-op。違うハッシュは既存ディレクトリを残して拒否する。`np.load(..., allow_pickle=False)` |
| 問題点 | 音声ファイルの SHA-256 は export 時に全件再計算しない。サイズ一致だけを見る。`verify_on_read` が偽のとき、公開後に差し替わった WAV は hot path では検出されない。`cases.npz` は `doc_json` を持たず、オンデマンドの文書は `docs.jsonl` にある。保存済み WAV の generated case は docs を書かない |
| 新設計 | DataSource はこの reader と `AssetStore` の上に乗る。export の実装は学習パッケージへ移さない |
| 移行方針 | 再利用 |

`find_snapshot_dir(root, "name@version")` は子ディレクトリの `dataset.json` を見て、ディレクトリ名だけでは信用しない。

### 準備と登録

`prepare/` はモデル族ごとのグラフ組み立てである。共通の公開基底クラスは無い。`import src.reproduction.prepare` は各モジュールを読み込まない。

| 経路 | 入口 | 書くもの |
|---|---|---|
| VoiceBank オンデマンド固定ペア | `prepare/run.py` の `execute` | staging、`reproduction.duckdb`、snapshot。16 kHz WAV は書かない。描画時に `resample_pcm16` |
| MS-SNSD | `prepare/ms_snsd_run.py` の `run_config` | 計画と、`delivery: on_demand` の登録。混合 WAV は書かない |
| DNS | `prepare/dns_run.py` | 同上。受理計画の shard は `dns_accept_store.write_shard` の JSONL で、スナップショットそのものではない |
| DeepFilterNet3 | `prepare/dfn3_online.py` の `register_p8` | 既定 DB は `data/db/reproduction-p8.duckdb`。公式 dataset は blocked。混合 WAV は無い |
| SEMamba++ | `prepare/semamba_online.py` の `register_p9` | 既定 DB は `data/db/reproduction-p9.duckdb`。公式 dataset は blocked |
| 研究用検証 | `prepare/research_valid_fixed.py` | 親の train / test を維持した子 dataset。公式再現の claim ではない |

学習の `data.pipeline: reproduction` は `src/training/se/config.py` が `database` キーを拒否する。`open_training_data` と `open_chunk_datasets` はスナップショットを読む。`register_training_run` は学習後の CLI で、学習ループは呼ばない。

legacy の `src/training/data/dataset.py` は DuckDB を読み取り専用で開く。`resource.duckdb` は `TrainingError`。この分岐は `pipeline` が reproduction でないときだけである。

## 3.2 学習データ供給

`EpochPlan` を返す関数が、順序と更新への詰めを同時に行う。名前だけの Epoch Planner と Batch Planner は無い。数値上の global / micro / world / accumulation の分解だけが `src/training/core/batch.py` の `plan_batches` である。

### 計画オブジェクト

| 項目 | 内容 |
|---|---|
| 既存実装 | `CropSpec`、`SampleRef`、`MicroBatchPlan`、`UpdateGroup`、`EpochPlan`、`RenderedSample`、`pipeline.types.Batch`（`src/reproduction/pipeline/types.py`） |
| 責務 | 音声バイトを持たない計画と、描画後の配列を分ける |
| 公開API | `SampleRef` は `case_id`、`index`、`draw`、`length`、`crop`、`chunk`。`chunk` は常に `None`。`Batch.as_task_batch()` は `noisy`、`clean`、`lengths`、`case_id`、`teacher_kind` |
| 強み | 計画が純データなので、同じ index を再描画できる |
| 問題点 | `index` の意味が固定ペアとオンラインで違う。chunk は別型 `PlannedChunk(case_id, start, stop)`。Experiment の `Batch` は `inputs`、`targets`、`sample_id`、`realization_id` で、フィールドが対応しない |
| 新設計 | 新しい `SampleRef` は sample 識別子、realization 識別子、crop または chunk 座標を持つ。古い dataclass は現行 Trainer のために残す |
| 移行方針 | 概念は再利用。型は新パッケージに作る。旧型のフィールド追加はしない |

### 順序

| 項目 | 内容 |
|---|---|
| 既存実装 | `rank_order`、`resolve_order_policy`、`order_params`（`pipeline/order.py`）。chunk は `plan_chunk_epoch`。DFN3 の並びは `plan_online_epoch` |
| 責務 | 1 rank、1 epoch の添字列。公式 sampler の差は policy 名で分ける |
| 公開API | `rank_order(policy, size, *, seed, epoch, world_size, rank) -> list[int]`。`espnet_chunk_v1` と `dfn3_online_ceil_v1` はここでは計算しない。前者は例外、後者は未知 policy として拒否される |
| 強み | 五つの平坦 policy が一つの関数である。training seed を使わない公式 sampler（`torch_distributed_sampler_v1` の seed 0、`fixed_single_shuffle_v1` の `random.Random(1234)`）がコードに残っている |
| 問題点 | `resolve_order_policy` が特別扱いするのは MP-SENet の 2 名リストだけである。`DATA_PIPELINE_DESIGN.md` の DFN3 の policy 名は、実装の `dfn3_online_ceil_v1` より古い |
| 新設計 | 平坦 policy は `rank_order` を呼ぶ Sampler。chunk と DFN3 の並びは、それぞれの既存計画関数を呼ぶ専用 Sampler。式は写さない |
| 移行方針 | 再利用 |

policy と使用箇所は次のとおり。

| policy | 分割 | 使う学習 |
|---|---|---|
| `fixed_single_shuffle_v1` | seed 1234 の shuffle を毎 epoch 同じ列で使い、`[rank::world]` | MP-SENet、world size 1 |
| `torch_distributed_sampler_v1` | `torch.randperm`、generator seed は `0 + epoch`、端数を捨てて stride | MP-SENet、world size > 1 |
| `rank_mod_fixed_v1` | `N // world` を shuffle して world 倍に広げ、先頭の繰り返しで `ceil(N/world)` に揃える | MossFormerGAN。検証も sampler epoch 0 |
| `global_shuffle_stride_v1` | `mix_seed("order", seed, epoch)` | 出荷中の reproduction protocol は使っていない。legacy の `epoch_indices` と同じ置換 |
| `per_rank_independent_v1` | rank ごとの `randperm` を、さらに distributed sampler で引く。rank 間の重複と欠落があり得る | SEMamba++ |
| `dfn3_online_ceil_v1` | protocol、snapshot、seed、split、epoch から permutation。world 1 は端数を残す。world > 1 は先頭繰り返しのあと stride | DeepFilterNet3 |
| `espnet_chunk_v1` | `samples[rank::world]` のあと、rank ごとに `RandomState(epoch + seed)` | TF-GridNet、TF-Locoformer |

計画は `(seed, epoch, world_size, rank, index)` だけでは決まらない。case の順序、窓長、batch 形状、protocol id、snapshot hash が入る。chunk は発話列の長さと cache サイズも入る。`per_rank_cache = max(num_cache_chunks // world_size, per_rank_batch)` なので、shard したリストを world size 1 として再計画すると cache が変わる。chunk は「先に shard した列を汎用 Sampler が渡し、Renderer が切る」だけでは現行と一致しない。

`torch.randperm` の版による差は、現行設計が UNKNOWN としている。この調査でも測っていない。

### 描画と Transform

| 項目 | 内容 |
|---|---|
| 既存実装 | `render_sample`、`transform.py` の `trim_to_shorter`、`noisy_rms_unit`、`independent_audio_norm`、`apply_window`。オンラインは `protocols/online_dfn3.py` と `protocols/online_semamba.py`。chunk の混合は `build/ms_snsd.py` と `build/dns2020.py` の `render_realization` |
| 責務 | `SampleRef` または realization 文書から、CPU の波形を作る。乱数をここで新たに引くのはオンライン描画だけである |
| 公開API | `render_sample(store, sample, *, normalize, sample_rate, verify_on_read=False, resample=None) -> RenderedSample`。正規化と窓は引数で、case 行の protocol では選ばない |
| 強み | 固定ペアの crop は描画前に決まっている。`render_sample` は乱数を引かない。オンデマンドの 48 kHz から 16 kHz は `resample_pcm16`（soxr HQ、clip、PCM_16 の往復）で、WAV を書かない |
| 問題点 | `head_place` と `tail_zero` はどちらも先頭に実サンプル、末尾に 0 である。名前は正規化の区別で、pad の幾何は違わない。公式の worker `random.randint` は再現していない（`CROP_NOTE`）。公式の 16 kHz 変換器は不明のままである |
| 新設計 | 固定ペアの正規化と窓は一つの Protocol 呼び出しにまとめる。DFN3 の reverb から混合まで、SEMamba の劣化順は、Transform の列に分解しない |
| 移行方針 | 関数は再利用。呼び出し順を YAML にしない |

`apply_window` の pad は `np.pad(..., (0, pad))`。短い発話も先頭配置である。

### Collation

| 項目 | 内容 |
|---|---|
| 既存実装 | `assemble`（`pipeline/assemble.py`）。chunk は `collate_chunks` |
| 責務 | micro-batch 内の最大長へ末尾 0 で揃え、`lengths` にパッド前の本数を残す |
| 公開API | `assemble(samples, *, dataset_id, snapshot_hash) -> dict`。chunk の戻りは `clean`、`noisy`、`case_id` だけで、`lengths` も `teacher_kind` も無い |
| 強み | 公式損失はパッドをマスクしない、というコメントが型の docstring にある |
| 問題点 | 固定窓では pad が no-op になり、可変長の研究用パッドと公式の「マスクしない」が同じ関数に見える |
| 新設計 | Batcher が micro-batch を組む。パッドは描画後、CPU tensor を作るときだけ。公式経路の損失は `lengths` を見ない |
| 移行方針 | パッドの規則は再利用。戻り型は Experiment の `Batch` に合わせる |

### 現行の供給オブジェクト

Trainer（`src/training/core/trainer.py`）が duck type とみなすのは、`RecipeDataset` ではなく `epoch` と `data_state` が呼べることである。実装は `ReproductionTrainingData`、`OnlineDFN3TrainingData`、`OnlineSemambaTrainingData`、`LegacyTrainingData`。共通の基底クラスは無い。`EpochStream.render(update_index)` は cursor を進めず、同じ index は同じ micro-batch のリストを返す。リストの長さは accumulation である。

chunk はこの duck type ではない。`execute` は `chunk_iterator` があるとき `open_chunk_datasets` を呼び、Trainer の別ループが `_chunk_state` を自分で作る。

| 項目 | 内容 |
|---|---|
| 既存実装 | `open_training_data`、`open_chunk_datasets`、上記クラス |
| 責務 | スナップショットから、現行 Trainer の 1 epoch を供給する |
| 公開API | `epoch(...) -> EpochStream`、`validation(...)`、`data_state(epoch, committed_updates) -> dict`、`identity()`、`check_resume`、`close()`。chunk dataset に `data_state` は無い |
| 強み | `require_resume_match` は保存辞書と再計算辞書の一致を要求する。prefetch の future は状態に入らない |
| 問題点 | Experiment の `train_epoch` が期待する `EpochView.batch -> list[Batch]`（1 呼び出しが 1 micro-batch）とは、accumulation の持ち方が違う。現行の reproduction は 1 update が accumulation 個の micro-batch を返す。`src/experiments` の `StandardTrainer` は accumulation 回 `batch(index + offset)` を呼び、各呼び出しが 1 micro-batch である |
| 新設計 | 新しい DataModule は Experiment 側の契約に合わせる。旧クラスは現行 CLI の実装として残し、新 DataModule の中ではインスタンス化しない |
| 移行方針 | 旧クラスは現行経路が残る間は廃止しない。新 API の実装には使わない |

### Worker、prefetch、検証

`GroupLoader` はスレッドプールである。`workers == 1` のときは呼び出しスレッドで実行する。ワーカーは乱数を引かない。`data.prefetch` の既定は真で、現行 Trainer は次の optimizer step を 1 本のスレッドで描画する。checkpoint には入れない。再開はコミット済み更新から描き直す。

`OnDemandChunkData.load_workers` は保存されるが読まれない。DNS の `open_dns_datasets` は `load_workers` を捨てる。chunk の collate は呼び出しスレッドである。

検証の分担は次のとおり。メソッド名が同じでも意味は違う。

| 経路 | 誰が採点するか | 順序と波形 |
|---|---|---|
| MP-SENet 固定ペア | 呼び出し側。現行 Trainer は rank 0 が既定 | データ順、`crop` は `full_length` |
| MossFormerGAN | 渡された rank | `rank_mod_fixed_v1` の epoch 0。crop は学習 epoch に依存（`per_epoch_random`） |
| DeepFilterNet3 | 呼び出し側が全件 | 並びの seed は epoch を含む。波形 seed の epoch key は 0 |
| SEMamba++ | 呼び出し側が全件 | 素材順。波形の epoch key は 0 |
| chunk | 各 rank、最短 rank で停止 | `shuffle=False`、`num_samples=None`、offset 0 |

`set_batch` を持つのは DFN3 と SEMamba だけである。固定ペアの batch 形状は `open_training_data` のときに固定される。DFN3 は accumulation が 1 で、`micro * world == global`。SEMamba は accumulation を許す。

## 3.3 公式 Protocol

四つの供給方式は混合エンジンを共有しない。共有しているのは `mix_seed`、スナップショットの `AssetStore`、固定ペアとオンラインの `assemble`、`require_resume_match`、スレッドの `GroupLoader` である。`require_on_demand` は `rule_json.delivery == on_demand` 以外の学習を拒否する。歴史的な保存済み WAV の smoke snapshot は、現行の chunk 入口では拒否される。

公式コードの DataLoader クラス、worker の `random`、Rust の `StdRng` は再現対象にしていない。現行コードが文書化している差は、新 API でも差のまま残す。この調査は差分を消していない。

### 固定ペア（MP-SENet、MossFormerGAN）

実装は `plan_fixed_pair` と `render_sample`。両方とも 16 kHz、窓 32,000、教師 `dry_clean`、protocol 上の `global_batch` は 4。SNR と RIR は無い。ペアは収録済みである。

MP-SENet の正規化は短い方へ揃えたあと、noisy の RMS で両者を割る（`noisy_rms_unit`）。MossFormerGAN は各チャネルを float64 で −25 dBFS にし、平均パワーを超える標本でもう一段スケールする（`independent_audio_norm`）。長さが違うペアは拒否する。

crop の開始は `mix_seed("crop", seed, epoch, case_id) % (length - window + 1)`。短い発話は開始 0 で末尾 0。1 epoch の 1 case は 1 開始で、sampler が同じ case を繰り返しても開始は同じである。

`case_id` は 48 kHz の clean / noisy を指す `recipe_v1` で、両モデルが共有する。crop は case 文書に保存しない。

既知の未解決は、公式 16 kHz 変換器が不明であること（オンデマンド protocol の `vb28-ondemand-resample`、`UN-3`）、crop 乱数が公式 worker の `random.randint` ではないこと、`torch.randperm` の版差が UNKNOWN であること。`mossformergan.md` の古い監査行は、現行の `per_epoch_random` 検証 crop の測定記録ではない。公式チェックポイントとの一致とは読まない。

### 事前生成と chunk（TF-GridNet、TF-Locoformer）

混合の identity は `realization_v1` の digest である `case_id`。描画は保存された draws を読む。学習中に SNR を引き直さない。chunk の `(start, stop)` は epoch 計画で、case 文書には書かない。

MS-SNSD の混合は 16 kHz、ファイル全体を −25 dBFS にしてから区間を切り、無音 0.2 秒、目標長 10 秒、名義 SNR は 0, 10, 20, 30, 40 dB である。`snr_mixer` の係数に平方根があり、モジュールの説明は実測 SNR が名義の半分になると書いている。計画乱数は `PCG64(mix_seed("gen", seed, split, clip_index))`。公式スクリプトは `np.random` を seed していない、という差は D-07 として記録済みである。読み込み時に PCM_16 の往復を掛ける。`rule.normalize` は null でなければならない。

DNS の混合は 16 kHz、30 秒、無音 0.2 秒、SNR の整数は下端 −5 以上かつ上端 15 未満、目標レベルは −35 dB 以上かつ −15 dB 未満の範囲を `integers` で引く。係数に外側の平方根は無い。`RETRY_ACCEPT` は 16 kHz でない雑音だけを描画時に `librosa.resample(..., res_type="soxr_hq")` する。公式混合に使われた librosa の版は記録されていない。公式のプール割当と fork した NumPy 列は再生しない。

chunk 計画の現行 YAML は、TF-GridNet が chunk 48,000、`num_samples_per_epoch` 5,000、cache 1,024、global batch 4、world size 2。TF-Locoformer の現行 YAML は chunk 64,000、`num_samples_per_epoch` 1,000。`tf_locoformer.md` 本文の 32,000 は監査当時の記録で、ヘッダが現行 YAML を指している。`global_batch % world_size != 0` は拒否する。ESPnet が端数を低い rank に渡す挙動は採らない。停止は rank 間の batch 数の最小である。

`DATA_PIPELINE_DESIGN.md` の `mix_seed("chunk", ...)` は、実装の `RandomState(epoch + seed)` と違う。実装に合わせる。

### オンライン混合（DeepFilterNet3）

正本の分布オブジェクトは `DFN3_ONLINE_V1`。48 kHz、144,000 サンプル、教師は dry ではなく `official_mixed_target`。SNR は `{-100, -5, 0, 5, 10, 20, 40}`。−100 以下は雑音だけ。gain は `{-6, 0, 6}`。雑音本数は 2 以上 6 未満。残響確率 0.1。リサンプルは `scipy.signal.resample` で、rubato ではない。

波形 seed は `mix_seed("dfn3-online", protocol_id, snapshot_hash, seed, split, epoch_key, sample_index)` を `PCG64` に入れる。train の epoch key は epoch、valid / test は 0。rank も worker も入らない。並びの seed は別で、`dfn3-order` に epoch が入る。公式の main-thread RNG（seed 42 を epoch をまたいで消費する）ではない。rule 内の `official_seed` 43 は、このストリームの seed としては使わない。

train の `case_id` は、`train.cases` が bind されていない限り `None`。valid / test の `case_id` は、index と seed 規則を入れた文書の digest で、SNR や RIR の実現値は入っていない。

子 protocol `dfn3.online.audio-db` の `algorithm_change`（PCG64、scipy resample、epoch 順序、RIR 上限、channel 0、置換回数の上限）は維持する。`dfn3.online.official@1` は blocked のままである。代替素材の ready dataset を公式素材の再現と呼ばない。

### オンライン劣化（SEMamba++）

分布は HEAD `config.yaml` の `SEMAMBA_GSR_V1` で、論文の 24,000 サンプルや −10 dB 設定ではない、と `ids.py` が書いている。16 kHz、窓 48,000。`librosa.load(..., sr=16000, mono=True)` でファイル全体を読む。SNR は `{-5, 0, 5, 10, 15, 20}`。`draw_used` は `draw_once` を 2 回呼び、1 回目を捨てる。適用順は reverb、noise、clipping、bandwidth。教師は dry。その後、clean と degraded を別々に peak 正規化し、crop する。

波形 seed に rank は入らない。同じ index は全 rank で同じ波形になる。公式は rank が重なったファイルで別の雑音を引き得る、という差が `rank_independent_waveform` として記録されている。捨てた 1 回目の draw は、公式ではプロセス乱数を進めて後続サンプルを変える。現行はサンプル内だけである。

並びは `per_rank_independent_v1`。公式の `drop_last=False` の pad は再現しない。world size 1 でも epoch ごとに shuffle する。公式の 1 GPU loader はそうではない、と `POLICY_NOTES` が書いている。

`semambapp.gsr.official@1` は blocked。`.venvs/se` には `resampy` が無く、`kaiser_best` と `kaiser_fast` の検証時だけ `.venvs/asr` の `resampy==0.4.3` を読んだ、という記録が `.agents/DATA.md` にある。この調査ではその検証を再実行していない。

## 3.4 Checkpoint と Experiment の Data API

### 現行 reproduction の状態

固定ペア、DFN3、SEMamba の `data_state` は、消費済み optimizer 更新の本数を `update_in_epoch` に入れる。キーは `pipeline`（`"reproduction"`）、`dataset_id`、`dataset_name`、`snapshot_hash`、`protocol_id`、`algorithm`、`algorithm_version`、`profile`、`selection_uses_test`、`seed`、`epoch`、`update_in_epoch`、`world_size`、`ranks`、`order_policy`、`order_params`、`plan_hash`、`global_batch`、`official_global_batch`、`micro_batch`、`accumulation`、`normalize`、`window`。再開は辞書の完全一致である。prefetch した配列は入らない。

chunk の状態は dataset ではなく `Trainer._chunk_state` が作る。`pipeline` は `espnet_chunk_v1`。`epoch` は 0 始まりの Trainer epoch で、計画関数へ渡すのは `epoch + 1`。`accumulation` は 1。`iterator_stop` は `min_rank_batches`。`virtual_world_size` は chunk のときだけ、プロセス world が 1 なら許される。

`plan_hash` は全 rank の計画を含む。各 rank が同じハッシュを計算できる。RNG の残状態はデータ計画に保存しない。rank 別の学習 RNG は checkpoint の `rng_by_rank` で、データ計画の seed とは別である。

`audio-db-training-v1` と `audio-db-experiment-v1` は相互に開かない。これは Phase 2 で実装済みの契約である。

### Experiment 側

`src/experiments/data.py` の `TensorPairData` が、実装済みの唯一の DataModule である。

- `setup(runtime)` はテンソルを device へ移し、`world_size` と `rank` を受け、`plan_hash` を再計算する。Builder は `role: data` の構築直後に `setup` を呼ぶ。
- `train_epoch(epoch, *, start_update) -> EpochView`。`batch(update_index) -> list[Batch]` は 1 要素のリストで、cursor を持たない。
- `state_dict` は `epoch`、`update_index`、`plan_hash`、`world_size`、`identity`。`load_state_dict` は identity、world size、plan hash の不一致を `StateError` にする。
- `update_index` は次に読む micro-batch の添字である。`StandardTrainer` は logical step ごとにこれを `accumulation` だけ進める。
- `batch_size` は 1 rank の 1 micro-batch。global micro-batch は `batch_size * world_size`。余りは捨てる。二度目の shard を禁止する、とモジュールの docstring が書いている。
- `close` はある。Trainer は呼んでいない。
- 検証メソッドは無い。`sample_id` は micro-batch 内の ID をカンマで結んだ 1 文字列である。

StateRegistry が保存するのは `module`、`optimizer`、`scheduler`、`trainer`、`data` だけである。`other` は保存しない。

## アダプタ推奨を維持しない理由

Phase 0 の [ARCHITECTURE.md](ARCHITECTURE.md) と [MIGRATION_PLAN.md](MIGRATION_PLAN.md) は、DataModule が `open_training_data` と online クラスと `open_chunk_datasets` をそのまま呼ぶ、と書いていた。調査した呼び出し面は、その推奨と合わない。

- 旧クラスは `train_epoch` も `state_dict` も持たない。アダプタはメソッドの翻訳だけでなく、accumulation の単位（optimizer 更新か micro-batch か）も翻訳する。
- chunk の再開状態は dataset の外、現行 Trainer の中にある。`data_state()` を転送するアダプタは chunk を落とす。
- 研究用の Sampler 重み変更は、融合した `epoch()` の内側に入口が無い。アダプタのままでは、Phase 5 で再びクラスを割ることになる。
- 公式の数値はクラスではなく、`rank_order`、`plan_chunk_epoch`、`plan_online_epoch`、`render_sample`、`render_realization`、`render_mixture`、`render_example` にある。クラスを包むことは、これらの関数を守ることと同じではない。

したがって新設計は、これらの関数を呼び、旧クラスは現行 CLI に残す。詳細は [DATA_ARCHITECTURE.md](DATA_ARCHITECTURE.md)。
