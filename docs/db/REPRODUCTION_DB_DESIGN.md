# 再現用 DB の仕様

`reproduction*.duckdb` は、公式再現と leakage-safe 学習のデータ定義を `repro` スキーマで持つ。学習プロセスはこの DB を開かず、`data/db/snapshots/<dataset_id>/` を読む。どのファイルにどの Dataset が入っているか、日付つきの観測は [配置と記録](INVENTORY.md)。

DDL の正本は `src/experiments/data/db/schema.py`。登録は `src/experiments/data/db/registry.py`。ID は `src/experiments/data/ids.py`。`src/data/reproduction/` は無い。

## 0. この文書の位置づけ

| 項目 | 正本 |
|---|---|
| `repro` の表、ID、Protocol と Case、claim、staging、スナップショット | この文書。列と CHECK は `src/experiments/data/db/schema.py` |
| レシピ JSON v1 と `eval.*` | [レシピ](RECIPE.md)、`src/data/recipe.py`、`src/data/db/eval_schema.py` |
| Dataset と Protocol の YAML、公式リスト | `configs/data/` |
| 学習がスナップショットを読む手順 | [学習](../experiments/RUN.md) |
| 評価の WER と名前付きセット | [評価パイプライン](../evaluation/PIPELINE.md) |
| 公式指標での採点 | [公式評価](../evaluation/OFFICIAL_EVALUATION.md)。仕様のみで、`src/evaluation/official/` は無い |
| ファイルの中身と構築記録 | [配置と記録](INVENTORY.md) |

## 1. 責務

持つのは、素材、リスト、分割、固定ペア、混合の規則、事前生成の定義、構成の版である。

| 持たない | 置く場所 |
|---|---|
| 音声波形 | `data/corpora/`、派生素材、生成物のファイル。現行のオンデマンド Dataset は混合 WAV を書かない |
| 学習中の一時波形 | メモリ |
| 評価用の名前付きセット（`eval.*`） | `resource.duckdb` |
| optimizer、RNG、学習位置 | checkpoint |
| 学習ログ | `data/results/` |

原則:

1. DuckDB に音声バイナリを入れない。
2. 行は DuckDB へ書く前に staging へ保存する。登録後も消さない。
3. 行は内容アドレスで識別する。同じ ID で内容が違う登録はエラーにする。UPDATE は、`sha256` が NULL の素材へ値を入れるときだけ（`asset_hashes/`）。
4. 時刻、ホスト名、絶対パスは行の ID に入れない。同じ staging から作った DB の内容を揃えるため。時刻は staging の `meta.json` に置く。
5. 学習の波形はスナップショットから読む。再現用 DB は開かない。発話長の一括取得だけは、渡されたカタログの `corpora.audio_files` を読み取り専用で開く（`src/experiments/calc/dataset.py`）。`load_recipe_dataset` は `resource.duckdb` というファイル名での学習を拒否する。DuckDB の版と、書き込み中のロックから学習の本処理を離すため。
6. 公式再現と leakage-safe は、同じスキーマの中で Dataset の `profile` と claim で区別する。評価用 DB に公式再現の Dataset は入れない。
7. 公式の特殊処理は、汎用の分布 DSL にしない。`algorithm` が実装を指し、`rule_json` はその実装が検証する文書である。受理する algorithm は `ALGORITHMS` に閉じ、未知のキーは拒否する。

## 2. ファイル

| ファイル | 役割 |
|---|---|
| `data/db/resource.duckdb` | 評価の本番。`corpora.audio_files` と `eval.*`。準備は、渡されたときだけ読み取り専用で開く。`load_recipe_dataset` はこのファイル名での学習を拒否する |
| `data/db/voicebank.duckdb` | VoiceBank 28 話者のレシピ学習セット。`eval` スキーマ。再現用スキーマではない |
| `data/db/training.duckdb` | LibriSpeech と DEMAND のスモーク。登録コードの出力先。ファイルの有無は [配置と記録](INVENTORY.md) |
| `data/db/reproduction.duckdb` | `repro` スキーマ。`make reproduction-init-db` と `make reproduction-prepare` の既定の DB。`resource.duckdb` には触れない |
| `data/db/reproduction-vb-ondemand.duckdb` | VoiceBank オンデマンド 4 Dataset が入っているファイル。既定の Make ターゲットはこのパスを指さない。`reproduction.duckdb` には入れない |
| `data/db/reproduction-p8.duckdb` | DeepFilterNet3。`make reproduction-register-p8` |
| `data/db/reproduction-p9.duckdb` | SEMamba++。`make reproduction-register-p9` |
| `data/db/snapshots/<dataset_id>/` | 学習が読む不変のスナップショット |

DB ファイル名の stem を `eval` にしない。DuckDB のカタログ名と `eval.cases` が衝突するため。`reproduction` はこの衝突を起こさない。

素材は `resource.duckdb` を ATTACH して常時参照しない。`make db-ingest` が `corpora.audio_files` を DROP して作り直すので、再現用 DB の参照先が索引のやり直しで変わる。準備のとき、使う行だけを `repro.assets` へ写し、出所を `catalog_snapshots` に残す。再構築は staging だけで閉じる。

`corpora.audio_files` から写す列は、path、corpus、subset、sample rate、frames、channels、話者、発話キーなど、その Dataset が要るものだけである。索引の全行は写さない。

2026-11-06 に `eval` スキーマが無い、と観測した記録と、2026-11-12 のファイルの状態は [配置と記録](INVENTORY.md) にある。監査の規則は変わらない。`eval` が無い、または `--resource-db` を渡さない leakage 監査は `UNKNOWN` であり、`PASS` にしない。

## 3. Protocol と Case

| 概念 | 定義 | 識別子 |
|---|---|---|
| 素材（asset） | 1 つの音声ファイル。コーパス、派生、生成物のいずれか | `asset_id` = `<storage_root>/<rel_path>` |
| 素材リスト | 役割（speech / noise / rir）が付いた順序付きの素材 | `material_set_id`（内容ハッシュ） |
| Protocol | 音声を作る規則。素材集合は含まない | `protocol_id`（規則文書のハッシュ） |
| Case | その文書だけで波形を再生成できる 1 サンプル | `case_id` |
| ケース集合 | 順序付きの Case。train / valid / test | `case_set_id` |
| Dataset | Protocol と、素材リストまたはケース集合をスロットに束ねたもの | `dataset_id` |
| Build | 事前生成の 1 定義。生成件数は ID に含めない | `build_id` |

同じ Protocol を、公式素材の Dataset と代替素材の Dataset が別々に束ねる。規則自体が変わるときは、`parent_protocol_id` を持つ別 Protocol にする。`official_effective` 以外は親が必須である（SQL の CHECK）。`official_effective` も親を持ってよい。VoiceBank の ondemand protocol は、親を持つ `official_effective` である。

### ID

| 対象 | ID の入力 |
|---|---|
| レシピ v1 で表せる Case | `src/data/recipe.py` の `case_id_for`。`resource.duckdb` と同じレシピ文書は同じ ID になる |
| その他の Case | `realization_v1` 文書の正規化 JSON の SHA-256 |
| Protocol | `doc_type`、`algorithm`、`algorithm_version`、`kind`、`rule`、`deviation_class`、`parent`。名前と版の整数は ID に入れず、`UNIQUE(name, version)` の列にする |
| Dataset | `doc_type`、`protocol_id`、スロットから集合 ID への対応、`profile`、`material_origin`、`claim_label`、`selection_uses_test`、`declared_status` |
| ハッシュの正規化 | `src/data/recipe.py` のキーソート、空白なし、浮動小数は小数 6 桁。`ids.py` の `digest` がそれを使う |

`recipe_v1` の `sources.*.path` は、`data/db/` に置いた DB からの相対パスで書く。`reproduction.duckdb` も `data/db/` に置くので、`resource.duckdb` と同じ文書が同じ `case_id` になる。別ディレクトリに置くと ID が変わる。その DB は使わない。

`realization_v1` の外枠は `realization_version`、`algorithm`、`algorithm_version`、`sample_rate`、`sources`、`draws`、`steps`、`output` である。未知のキーは拒否する。検証は `src/experiments/data/ids.py` と、DFN3 / SEMamba++ では `src/experiments/data/protocols/`。`teacher` は `dry_clean`、`reverberant_clean`、`official_mixed_target`、`none`。乱数は Planner が引き、結果を文書に書く。Renderer は乱数を引かない。

オンライン混合の学習サンプルは、`(seed, epoch, index)` からその場で作る。1 epoch 分を DB の行にはしない。validation と test の実現値はケース集合に固定する。

| 方式 | `kind` | 学習が読むもの |
|---|---|---|
| 固定ペア | `fixed_pair` | `recipe_v1` の Case が指す clean と noisy |
| 事前生成 | `pregenerated` | `delivery: on_demand` のときは元素材と `realization_v1` から再構成する。生成 WAV のパスを必須にしない |
| オンライン混合 | `online_mixing` | 学習は素材リストから描画する。valid / test は固定した実現値を描画する |

受理する algorithm は次の 6 つで、版はどれも 1 である。`fixed_pair.mp_senet`、`fixed_pair.mossformergan_se_16k`、`online.dfn3`、`online.semambapp_gsr`、`pregenerated.ms_snsd_tfgridnet`、`pregenerated.dns_tflocoformer`。

## 4. スキーマ

スキーマ名は `repro`。CHECK は同じ行の列だけを見る。表をまたぐ条件は登録関数が検証する。`content_hash` は、その行から ID を除いた本体のハッシュである。

`make reproduction-init-db` は、同じスキーマの既存ファイルには何もしない。違うスキーマや未完了のファイルは変更せず拒否する。準備ができた DB の `schema_info` には `schema_version` = `1`、`ready` = `1`、`column_fingerprint` が入る。指紋がこのコードの列順と違う DB は拒否する。再構成は、これらに加え `database_origin`、`original_database_identity`、`reconstructed_parent_dataset_id`、`reconstructed_parent_snapshot_hash` を書く（`src/experiments/data/prepare/reconstruct_accept.py`）。

### `catalog_snapshots`

`catalog_id`、`source_kind`（`resource_duckdb` または `staging_parquet`）、`corpora_json`、`row_count`（0 以上）、`source_db_sha256`。写した索引の版を残す。パスは ID に入れない。

### `derivations`

`derivation_id`、`kind`（`resample` または `format_convert`）、`spec_json`。公式の変換器が不明なとき、audio-db が選んだ方法をここに書き、公式と同じとは書かない。

### `builds`

`build_id`、`generator`、`generator_version`（1 以上）、`params_json`、`seed`、`inputs_json`、`shard_size`（1 以上）。完了は列にせず、`audits` の `build_verification` と staging の `verified/` に置く。`build_id` に生成件数は含めない。

### `assets`

`asset_id` は `storage_root || '/' || rel_path`。`storage_root` は `corpora`、`generated`、`derived`。`rel_path` に `..`、絶対パス、`.` は拒否する。

`corpora` は `catalog_id` が必須、`derived` は `derivation_id` と `derived_from` が必須、`generated` は `build_id` が必須。`sample_rate`、`frames`、`channels` は 1 以上。

`integrity` の SQL は次のとおり。

| 値 | 条件 |
|---|---|
| `sha256` | `sha256` が NOT NULL |
| `size` | `sha256` が NULL かつ `size_bytes` が NOT NULL |
| `catalog` | `sha256` が NULL |

登録関数は、`integrity = sha256` のとき `size_bytes` も要求する。SQL だけでは `size_bytes` は無くてよい。既に `sha256` がある行へ違う値を書くことは拒否する。NULL から値への更新だけを、staging の `asset_hashes/` 経由で許す。

索引は `(corpus, subset)`、`sha256`、`speaker_id`、`build_id`。

### `material_sets` と `material_set_members`

集合は `material_set_id`、`name`、`version`（`UNIQUE(name, version)`）、`role`（`speech` / `noise` / `rir`）、`split_role`（`train` / `valid` / `test` / `pool`）、`origin`（`official` / `substitute` / `derived_from_official` / `unknown`）、`selection_rule_json`、`provenance_json`、`member_count`、`total_seconds`、`list_sha256`、`integrity_min`。

メンバーは `(material_set_id, position)` が主キー、`(material_set_id, asset_id)` が UNIQUE。`group_key` と `attrs_json` を持つ。役割は素材ではなく、リストの `role` と Dataset のスロットに付く。同じファイルが、ある Protocol では speech、別の Protocol では noise になれる。

### `protocols`

`protocol_id`、`name`、`version`（`UNIQUE(name, version)`）、`kind`（`fixed_pair` / `pregenerated` / `online_mixing`）、`algorithm`、`algorithm_version`、`rule_json`、`official_ref_json`、`deviation_class`、`parent_protocol_id`、`deviations_json`。

`deviation_class` は `official_effective`、`minimal_launch_repair`、`algorithm_change`、`research_improvement`。`official_effective` 以外は `parent_protocol_id` が必須。`kind` は algorithm の種別に一致する。未知の algorithm と版は拒否する。変更は新しい版であり、行の上書きではない。

2026-11-12 に開いた 4 ファイルに、`minimal_launch_repair` の行は無かった。SEMamba++ と DeepFilterNet3 の代替は `algorithm_change` の子 protocol（`semambapp.gsr.audio-db`、`dfn3.online.audio-db`）である。

### `cases`

`case_id`、`doc_kind`（`recipe_v1` / `realization_v1`）、`doc_json`、`case_role`（`recorded_pair` / `generated` / `online_snapshot`）、`protocol_id`、`clean_asset_id`、`noisy_asset_id`、`noise_asset_id`、`teacher_kind`、`sample_rate`、`frames`、`speaker_id`、`utterance_key`、`transcript`、`build_id`、`gen_index`、`content_hash`。

SQL の CHECK:

- `recorded_pair` は clean と noisy が NOT NULL、`teacher_kind` は `dry_clean`、`doc_kind` は `recipe_v1`。
- `generated` は `build_id` と `gen_index` が NOT NULL。
- `online_snapshot` は clean、noisy、noise、build、gen_index がすべて NULL。

`delivery: on_demand` の事前生成 Case は、clean と noisy の素材 ID を空にしてよい。文書はスナップショットの `docs.jsonl` に書く。`teacher_kind = none` の Case は学習に使わない。

索引は `(build_id, gen_index)`、`clean_asset_id`、`speaker_id`。

### `case_sets` と `case_set_items`

集合は `case_set_id`、`name`、`version`、`split_role`（`train` / `valid` / `test`）、`origin`（素材リストと同じ 4 値）、`selection_rule_json`、`provenance_json`、`member_count`、`list_sha256`。項目は `(case_set_id, position)` と、集合内の `case_id` の UNIQUE。同じ `case_id` が複数の集合に入ることは許す。test を validation に兼用できるのは、Dataset の制約を通ったときだけである。

公式リストは `configs/data/lists/` に置き、`provenance_json` に出典ファイルの SHA-256 を残す。`src/se/official/` は再構築元にしない。

### `datasets` と `dataset_bindings`

`dataset_id`、`name`、`version`、`protocol_id`、`profile`（`official_faithful` / `leakage_safe`）、`material_origin`（`official` / `derived_from_official` / `substitute` / `mixed` / `unknown`）、`claim_label`、`declared_status`（`ready` / `blocked` / `unknown`）、`blocking_json`、`selection_uses_test`、`selection_note`、`deviations_json`。

`claim_label` の SQL は次の 7 値である。

`official_reproduction`、`protocol_conformant_substitute_materials`、`espnet_public_se_recipe_reproduction`、`official_recipe_scaled_down`、`minimal_repair_reproduction`、`leakage_safe_comparison`、`research_checkpoint_selection`。

`research_checkpoint_selection` を許さない古い CHECK を持つ DB は、`ensure_dataset_claim_labels` が `datasets` とそれを参照する表をコピーして作り直す。DuckDB は CHECK をその場で外せないためである。行の値と列順は変えない。

SQL の CHECK は、`leakage_safe` なら `selection_uses_test` が false かつ claim が `leakage_safe_comparison`、その claim なら profile が `leakage_safe`、`official_reproduction` なら profile が `official_faithful`。これ以外の組み合わせは登録関数が拒否する（下の 7 章）。

スロットは `train|valid|test` の `.cases` と、`.speech` / `.noise` / `.rir`。`(case_set_id IS NULL) <> (material_set_id IS NULL)`。必須スロットは `ids.py` の `REQUIRED_SLOTS` で、`fixed_pair` と `pregenerated` は train / valid / test の cases、`online_mixing` は `train.speech` と `valid.cases`。オンライン混合は、それに加えて素材スロットと `test.cases` を許す。

`ready` の `blocking_json` は空である。`blocked` と `unknown` はバインドを持たず、`blocking_json` は空でない。スナップショットは作らない。`ready` の `official_reproduction` は、束ねた全素材の `integrity` が `sha256` であることを登録関数が要求する。

### `audits` と `runs`

監査は `audit_id`、`subject_kind`（`build` / `dataset` / `material_set` / `case_set`）、`subject_id`、`audit_kind`（`build_verification`、`count_check`、`list_hash_check`、`leak_check`、`distribution_check`、`hash_verify`）、`status`（`PASS` / `FAIL` / `WARN` / `UNKNOWN`）、`report_json`。確認できなかったことを `PASS` にしない。時刻の列は無い。

`runs` は `run_id`、`dataset_id`、`snapshot_hash`、`config_hash`、`git_commit`、`world_size`、`ddp_mode`（`single` / `real` / `virtual`）、`selection_uses_test`、`data_state_json`。学習中は書かない。学習のあと `register_training_run` が 1 プロセスで書く。`--database` の既定値は無く、本番ファイルを暗黙には開かない。同じ `run_id` で dataset、snapshot、config、git、world size、`ddp_mode`、`selection_uses_test`、claim、データ計画が同じなら、checkpoint の位置とパスだけの変化は更新である。それらが違う再登録は拒否する。`claim_label`、`checkpoint_path`、`checkpoint_sha256` は列が無いので `data_state_json` に足す。再開が読むのは checkpoint 本体の `data_state` である。

挿入順は、catalog、derivations、builds、assets、material sets、protocols、cases、case sets、datasets、bindings、audits、runs。DuckDB は挿入時に外部キーを検査するので、派生素材より先に親の corpora 素材を入れる。

## 5. JSON と型付き列

条件で絞る列、結合する列、CHECK する列は型付きにする。`rule_json`、`doc_json`、選択規則、出典、`deviations_json`、`blocking_json`、監査の報告は JSON である。JSON の中身で SQL の WHERE を組まない。未知のキーは、algorithm ごとの検証で拒否する。分布のパラメータは、その algorithm のフィールドにする。新しい Protocol は、algorithm のレジストリと検証を足す。表の追加を前提にしない。

## 6. ハッシュ

```text
file bytes → assets.sha256
assets → material_sets → material_set_id ─┐
cases → case_sets → case_set_id ──────────┼→ dataset_id → snapshot_hash → run
protocol rule → protocol_id ──────────────┤
builds → build_id ────────────────────────┘
```

意味の版（`algorithm_version`、`realization_version`、`recipe_version`、`generator_version`、名前に付く `version`）は ID に入る。Git commit とライブラリ版は ID に入れず、`meta.json` とスナップショットの provenance に残す。出力が変わる実装変更は版を上げる。

同じ staging から作った DB は、登録順が違っても全表の内容が一致する。事前生成の計画は、Build の定義から同じ `realization_v1` の列が再び得られることを要求する。波形のバイト一致は要求しない。オンライン混合の実現値は `(seed, epoch, index)` の関数である。

## 7. Official と leakage-safe

| ラベル | 登録が要求すること | 表記 |
|---|---|---|
| `official_reproduction` | `official_faithful`。素材の由来は `official` または `derived_from_official`。protocol は `official_effective`。ready のときは全素材が SHA-256。派生なら `deviations_json` に差を書く | 公式再現。派生の差を併記する |
| `protocol_conformant_substitute_materials` | 由来は `substitute` または `mixed` | 公式プロトコル準拠・代替素材。公式データセットの再現と呼ばない |
| `espnet_public_se_recipe_reproduction` | algorithm は MS-SNSD または DNS の事前生成 | ESPnet 公開 SE レシピの再現。論文の実験の再現と呼ばない |
| `official_recipe_scaled_down` | `pregenerated` | 公式レシピの縮小版 |
| `minimal_repair_reproduction` | `deviation_class` が `minimal_launch_repair` | 最小補修を含む再現 |
| `leakage_safe_comparison` | `leakage_safe` かつ `selection_uses_test` が false | leakage-safe 比較 |
| `research_checkpoint_selection` | `official_faithful`、`selection_uses_test` が false、protocol は `research_improvement` | 研究用検証集合。公式再現の claim ではない |

ラベルの付け替えは、新しい版の Dataset としてだけ行う。

`leakage_safe` は登録時に次を検査し、1 つでも FAIL なら登録しない。

1. train / valid / test の `asset_id` が重ならない。
2. 同じ `speaker_id` が train と valid / test にまたがらない。話者が定義できない素材は `WARN`。
3. `sha256` がある素材の内容が重ならない。ハッシュが無いものは `UNKNOWN`。
4. train / validation の素材が `resource.duckdb` の `eval.*` と重ならない。test は最終評価セット自身であってよい。`eval` が無い、または resource DB を渡さないときは `UNKNOWN`。`PASS` にしない。

`official_faithful` では、test と validation の重なりは許す。そのときは `selection_uses_test` が true になり、`selection_note` に理由を書く。train と test の重なりは FAIL にする。

公式再現でモデル選択に使った集合は、研究評価の独立 test として扱わない。重なりの判定は、評価レシピの SHA-256 と checkpoint 横の来歴 JSON で行い、`reproduction.duckdb` は開かない。情報が足りない `independent` は `UNKNOWN` である。公式指標で公式 test を採点する経路は未実装で、仕様は [公式評価](../evaluation/OFFICIAL_EVALUATION.md)。

`selection_uses_test` が true の学習は、Dataset の値と、学習設定の明示の両方があるときだけ許す。既存の評価用セットを学習に使う規則は変えない。

## 8. 登録と再構築

staging は内容アドレスである。同じ ID で内容が違う書き込みはエラー。`write_graph`（`src/experiments/data/db/staging.py`）が書く形:

```text
catalog/<catalog_id>/catalog.json, audio_files.parquet, meta.json
assets/<batch_id>/assets.parquet, meta.json
derivations/<derivation_id>.json
materials/<material_set_id>/set.json, members.parquet
protocols/<protocol_id>.json
cases/<case_set_id>/set.json, items.parquet, cases.parquet
datasets/<dataset_id>/dataset.json, bindings.json
builds/<build_id>/build.json, verified/complete.json
audits/<audit_id>.json
runs/<run_id>.json
asset_hashes/<update_id>.json
INDEX.json
```

`meta.json` は ID に入らない来歴である。音声は staging に入れない。`INDEX.json` が無いディレクトリは `register_staging` が読めない。

登録は `register_staging` が staging を検証してから 1 トランザクションで書く。失敗したバッチは残らない。同じ ID と同じ内容は何もしない。Parquet は `INSERT … SELECT … FROM read_parquet` で入れる。書き込みは、DB のあるディレクトリの `.reproduction.lock` を `flock` で取れたプロセスだけが行う。

コマンド:

```bash
make reproduction-doctor
make reproduction-init-db
make reproduction-prepare
make reproduction-snapshot DATASET=name@version
make reproduction-rebuild-db OUT=data/db/reproduction-rebuilt.duckdb
make reproduction-verify DATASET=name@version HASH=1
make reproduction-register-p8
make reproduction-register-p9
```

`DATASET` が無い snapshot、`OUT` が無い rebuild は失敗する。再構築は `OUT` が既存なら拒否し、新しいファイルへ書く。全表のハッシュが一致したときだけ置き換える。leakage 監査が `UNKNOWN` の Dataset を書き出すときだけ `ALLOW_UNKNOWN_LEAK_CHECK=1` を付ける。このフラグはスナップショットの provenance に残り、ハッシュには入らない。

`reproduction-register-p8` は `register_p8()`、`reproduction-register-p9` は `register_p9()`。既定の出力はそれぞれ `data/staging/reproduction-p8` と `data/db/reproduction-p8.duckdb`、`data/staging/reproduction-p9` と `data/db/reproduction-p9.duckdb` で、ready な Dataset のスナップショットを `data/db/snapshots/` に書く。既存の同じ内容は増やさない。`reproduction.duckdb` と `resource.duckdb` には入れない。

`reproduction-prepare` は VoiceBank の固定ペア（`src/experiments/data/prepare/run.py`）で、段は diagnose、lists、resolve、register、snapshot、verify である。既定の DB は `data/db/reproduction.duckdb`、staging は `data/staging/reproduction` である。ケースは 48 kHz のコーパスを指す。16 kHz への変換は学習が読むときである。オンデマンド 4 Dataset の実ファイルは別パスで、そのパスを既定にするターゲットは無い。

DNS と MS-SNSD の計画と登録は `reproduction-plan-dns`、`reproduction-register-dns`、`reproduction-plan-dns-accept`、`reproduction-register-dns-accept`、`reproduction-plan-build`、`reproduction-register`。設定ファイルは Makefile の `DNS_CONFIG`、`DNS_ACCEPT_CONFIG`、`MS_SNSD_CONFIG`。

`reproduction-rebuild-db` は、`OUT` が既存なら拒否する。`OUT` と同じ名前の `.partial` に作り、参照 DB と全表のハッシュが一致したときだけ `os.replace` する。失敗時は `.partial` を消す。強制終了で `.partial` が残ることがある。評価用 DB の `--backup` と `.incoming-*` / `.build-*` は `src/data/bootstrap.py` の話で、再現用の再構築は使わない。

## 9. 事前生成とオンライン混合

現行の登録経路は `rule_json.delivery = on_demand` である。事前生成の Case は clean / noisy の素材 ID を空にし、文書を `docs.jsonl` に書く。DNS と MS-SNSD の検証記録は `wav_files = 0` である。公式規模を混合 WAV にすると約 1.04 TB になる、という 2026-11-06 の見積りは [配置と記録](INVENTORY.md) にある。現行の登録はその WAV を書かない。MS-SNSD の上限なし走査は `source_scan: full` が無いと拒否する。DNS の受理レシピは、その設定の `source_scan` を検査する。保存済み WAV を読む旧 Dataset は、2026-11-12 の DuckDB とスナップショット一覧には無い。

混合 WAV を shard に書いて再開する手順は、現行の登録コマンドには無い。生成物の相対パスを計算する `generated_rel_path` は残っている。`storage_root = generated` の素材が在るときは、そのパスと一致することをスナップショット作成が検査する。2026-11-12 の作業ツリーに `data/generated/` は無かった。

各 `gen_index` の乱数は `mix_seed("gen", build.seed, gen_index)` から作る。共有カウンタや並列の実行順には依存しない。公式の生成は未シードなので、この決定性は公式との差として `deviations_json` に書く。公式の式の癖（MS-SNSD の名目 SNR と実 SNR、TF-Locoformer の整数 SNR とレベル）は規則に残す。

生成のパラメータ（時間、SNR とレベルの範囲、無音の長さ、seed）の正本は `configs/data/builds/*.yaml` で、式は `src/experiments/data/prepare/ms_snsd.py` と `dns2020.py` にある。`build_id` は件数を含まないので、同じ Build の小さい計画は大きい計画の先頭と同じ文書になる。train と valid の分け方は Build ではなくケース集合の規則で決める。MS-SNSD は、sorted utt id の先頭 8 割を train、残りを valid にする（`sorted_ratio`）。

評価の `rms_v1` 混合と、これらの公式学習データの生成は別の定義である。`src/evaluation/` は再現用の生成器を import しない。

DNS のオンデマンド描画が学習へ返すのは clean と noisy である（`render_dns_pair`）。雑音は混合の入力であり、第三の学習ストリームとしては返さない。チャンク計画は呼び出し側の発話順を保ち、並べ替えない。MS-SNSD は `speech_mix_shape` の順を渡す（`src/experiments/data/render/chunking.py`）。

`cv_synthetic_small` の構成は未確定のままである。検証に使う valid は、sorted uttid の末尾であり、`cv_synthetic_small` ではない。

VoiceBank のオンデマンドでは、共有ケースの `protocol_id` は `mp_senet.fixed_pair.ondemand` である（その protocol が読み込まれているとき。`_case_protocol_id`）。Dataset 行はモデル自身の protocol を指す。学習時の正規化、窓、順序は Dataset の protocol を読む。公式の 16 kHz 事前変換器は不明である。オンデマンドは 48 kHz のファイルを指し、読み出し時に soxr HQ で 16 kHz float32 にし、追加のクリッピングと量子化をしない。配布 PCM16 の読み込みとは別である。MS-SNSD と DNS2020 の公式生成に含まれる PCM16 処理は維持する。旧 VoiceBank PCM16 変換 snapshot と checkpoint の再開互換性は保証しない。履歴の 16 kHz 派生素材の置き場所は `data/derived/VoiceBank-DEMAND/soxr_hq_16000/` と記録されているが、2026-11-12 にはそのディレクトリは無かった。

## 10. スナップショット

書き出しは `src/experiments/data/db/snapshot.py`。読み取りは `src/experiments/data/snapshot.py` で、こちらは DuckDB を import しない。

ディレクトリは `data/db/snapshots/<dataset_id>/`。公開名は `.` で始まらず、`.partial` で終わらない。export は `.{dataset_id}.partial` に書いてから rename する。同じ内容の再 export は何もしない。内容が違う既存ディレクトリはエラーにする。

中身は `dataset.json`、`assets.npz`、`cases.npz`、`case_set_items.npz`、`material_set_members.npz`、`docs.jsonl`、`generated.json`、`manifest.json`。`generated` の素材は `assets.npz` に入れず `generated.json` に書く。`on_demand` の Case と、生成物ではない Case の文書は `docs.jsonl` に書く。ハッシュはファイル形式ではなく、論理的な行の内容から計算する。`manifest.json` の provenance（Git commit、`allow_unknown_leak_check`）はハッシュに入らない。

書き出せるのは `declared_status = ready` で、参照するファイルのサイズが DB と一致し、`count_check` と `list_hash_check` が PASS、データセットに FAIL の監査が無いものだけである。`leakage_safe` は `leak_check` の PASS が要る。`UNKNOWN` は `--allow-unknown-leak-check` のときだけ許し、manifest に残す。事前生成で Build を参照するときは `build_verification` の PASS が要る。`sha256` の全ファイル再計算は、スナップショット作成のたびにしない。

## 11. モデルごとの Dataset

件数、ID、どのファイルに入っているかは [配置と記録](INVENTORY.md)。ここは置き方だけを書く。

| モデル | algorithm | 登録 | 備考 |
|---|---|---|---|
| MP-SENet | `fixed_pair.mp_senet` | `reproduction-prepare` が official と safe を一つのグラフにする。既定の出力先は `reproduction.duckdb` | official は valid = 公式 test で `selection_uses_test = true`。safe は p286 と p287 を valid に外す。2026-11-12 の 4 Dataset は `reproduction-vb-ondemand.duckdb` にある |
| MossFormerGAN | `fixed_pair.mossformergan_se_16k` | 同じ prepare。ケース行は `mp_senet.fixed_pair.ondemand` を共有する | 正規化は MossFormerGAN の Dataset protocol を読む |
| DeepFilterNet3 | `online.dfn3` | `make reproduction-register-p8` | official は blocked。substitute と smoke は代替素材 |
| SEMamba++ | `online.semambapp_gsr` | `make reproduction-register-p9` | official は blocked。substitute と smoke は代替素材。劣化 WAV は作らない |
| TF-GridNet | `pregenerated.ms_snsd_tfgridnet` | `reproduction-plan-build` と `reproduction-register` | 現行は on_demand。公式規模のスナップショットは INVENTORY の `13c60588…` |
| TF-Locoformer | `pregenerated.dns_tflocoformer` | `reproduction-register-dns-accept` と、研究用 valid の再構成 | `reproduction.duckdb` に accept と `research_checkpoint_selection` が入っている。`cv_synthetic_small` ではない |

DeepFilterNet3 の公式素材（DNS4、PTDB、同梱の `dataset.cfg`、公式 RIR、VCTK の factor）と、SEMamba++ の公式素材（VCTK 0.92 の manifest、Arni RIR、固定の validation ペア）は、`data/corpora/` にリンクされておらず、カタログにも無い。コーパス置き場には `VCTK-0.92` と `Arni-RIR` という名前のディレクトリが在るが、中身は未確認である（[配置と記録](INVENTORY.md)）。その間、公式 Dataset は blocked のままである。手元の VCTK が公式の 0.92 と同一かは未確認で、リンク先 README とカタログ注記の差は [コーパス](CORPORA.md) にある。

## 12. 未確定

| ID | 内容 |
|---|---|
| U-DB-2 | `cv_synthetic_small` の大きさと規則。コードはこれを UNKNOWN のままにし、sorted uttid の末尾を valid にしている |
| U-DB-3 | VoiceBank を 16 kHz にした公式の変換器。オンデマンドは soxr HQ を使い、`official_reproduction` の `deviations_json` に変換器が不明であることを書く |
| U-DB-4 | DeepFilterNet3 の公式素材。入手できるまで `dfn3.online.official@1` は blocked |
| U-DB-5 | SEMamba++ の公式素材と、VCTK-GSR の分割。入手できるまで `semambapp.gsr.official@1` は blocked |
