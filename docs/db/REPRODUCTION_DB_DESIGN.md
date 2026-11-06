# 再現用データ基盤の DB 設計

状態: **P1 の DB 基盤は実装済み**（2026-11-06）。`reproduction.duckdb` には VoiceBank 固定ペア 4 Dataset と、MS-SNSD の保存済み WAV Dataset、オンデマンド Dataset がある。学習はスナップショットを読み、生成済み WAV を必須にしない。詳細は [P6_VERIFICATION](../training/reproduction/P6_VERIFICATION.md)。

P1 で設計からずらした点: `rule_json` は `src/reproduction/ids.py` の閉じたアルゴリズム封筒（完全な Protocol dataclass は P3）。生成素材は `assets.npz` に入れず `generated.json` に書く。leak 監査は `--resource-db` を渡したときだけ `resource.duckdb` を開く。監査に時刻は無く、PASS が存在し FAIL が無いことを使用条件にする。`sha256` の NULL から値への更新だけを `asset_hashes/` 経由で許す。`integrity = sha256` は Python 側で `size_bytes` も要求する。`official_reproduction` は `declared_status = blocked` を拒否する（DFN3 の blocked 公式 Dataset は P8 でラベルを決める）。

## 0. この文書の位置づけ

| 項目 | 正本 |
|---|---|
| DB スキーマ、ID とハッシュ、Protocol と Case の区別、Official / Research の分離、staging、事前生成の保存・再開、DuckDB の読み書き規則 | **この文書** |
| Dataset / Sampler / Renderer / Batch の API、DDP、再開、性能、YAML、コマンド | [DATA_PIPELINE_DESIGN](../training/reproduction/DATA_PIPELINE_DESIGN.md) |
| 実装フェーズ、受け入れテスト、未解決事項、着手前の確認 | [IMPLEMENTATION_PLAN](../training/reproduction/IMPLEMENTATION_PLAN.md) |
| 6 モデルの再現対象、D-01〜D-09 の採用結果 | [TARGETS](../training/reproduction/TARGETS.md)、[.agents/DECISIONS.md](../../.agents/DECISIONS.md) |
| レシピ JSON v1、`eval.*` | [RECIPE](RECIPE.md)、`src/data/recipe.py` |

記法: 「（確認）」は 2026-11-06 に現行コードを読んで確認した事実、「（設計）」はこの文書が定める内容、「（UNKNOWN）」は一次資料で決まらない事項。

---

## 1. DB の責務と原則

`reproduction.duckdb` は、**公式再現と leakage-safe 学習のデータ定義**（素材、リスト、分割、固定ペア、事前生成物、混合規則、構成の版）を保存する。持たないものを先に書く。

| 持たない | 置く場所 |
|---|---|
| 音声波形 | ファイルシステム（`data/corpora/`、`data/generated/`、派生キャッシュ） |
| 学習中の一時波形、オンライン混合の中間信号 | メモリ |
| 評価用の名前付きセット（`eval.*`） | `resource.duckdb`（変更しない） |
| 学習の進行状態（optimizer、RNG、位置） | checkpoint |
| 学習ログ・TensorBoard | `data/results/` |

原則:

1. 音声の実体はファイルに残す。DuckDB に音声バイナリを入れない。
2. 登録行は DuckDB へ書く前に `data/staging/reproduction/` に保存する（AGENTS.md の staging 規則。登録後も消さない）。
3. 行は内容アドレス（ハッシュ）で識別し、追記のみ。同じ ID で内容が違う登録はエラーにする（`eval.*` と同じ規則）。UPDATE は §4.4 の 1 箇所だけ。
4. 時刻・ホスト名・絶対パスは行の内容に入れない。同じ staging から作った DB の内容が一致するため。時刻は staging の `meta.json` と実行ディレクトリに置く。
5. **学習は DB を開かない。** 学習が読むのは、DB から書き出した不変のスナップショット（§10。`numpy` と JSON だけで読める）である。書き込みと学習時の読み取りを分け、DuckDB の版差・不在の影響を受けないため。
6. 公式再現（Official-faithful）と leakage-safe は、同じ DB ファイルの中でも、Dataset 定義の `profile` と制約で区別する（§7）。評価用 DB には公式再現のデータを入れない。
7. 公式の特殊処理は汎用スキーマに押し込まない。Protocol の `algorithm` が専用実装を指し、`rule_json` はその実装が検証する型付き文書である（§5）。

---

## 2. 既存 DB との関係

### 2.1 DB ファイルの役割分担

| ファイル | 役割 | この設計での扱い |
|---|---|---|
| `data/db/resource.duckdb` | 評価の本番 DB。`corpora.audio_files`（素材の索引）と、あれば `eval.*`（名前付き評価セット） | **変更しない。** `corpora.audio_files` は準備時に**読み取り専用**で読む。学習は名前で拒否する（既存 `load_recipe_dataset`）。再現用 DB の内容を書き込まない |
| `data/db/training.duckdb`、`data/db/voicebank.duckdb` | 既存の研究用学習セット（レシピ JSON v1、`eval.*` スキーマ） | 既存の `RecipeDataset` 経路で使い続ける。このワークツリーには無い（再構築元は `data/staging/*-current/`）。新 Pipeline からは Legacy アダプタ経由で読める |
| `data/db/reproduction.duckdb`（**新規・設計**） | 公式再現と leakage-safe 学習のデータ定義。スキーマ名 `repro` | 実装フェーズ 1 で作る。今回は作らない |
| `data/db/snapshots/<dataset_id>/`（**新規・設計**） | 学習用の不変スナップショット（`.npz` + JSON）。DB から書き出す派生物 | §10。再生成できるが、実行に使った版は run に記録する |

DB ファイル名の stem を `eval` にしない規則（`DATA.md`）は新ファイルにも適用する。`reproduction` は問題ない。

### 2.2 `corpora.audio_files` の再利用

**採用: 準備時に必要な行だけを `repro.assets` へ写す。ATTACH による常時参照はしない。**

| 案 | 長所 | 短所 | 判定 |
|---|---|---|---|
| A: `resource.duckdb` を ATTACH して `corpora.audio_files` を直接参照 | 二重管理が無い | `make db-ingest` が表を DROP して作り直す（AGENTS.md）。再現用 DB の参照先が実行ごとに変わる。再構築が `resource.duckdb` の状態に依存する。DuckDB の版が環境で違う（`DEVELOPMENT.md` は `.venvs/semamba` を 1.4.3 で 2.0 のファイルを開けないと記録する。現在の `.venvs/se`、`semambapp`、`asr`、`official` は 2.0.0、`.venvs/deepfilternet` は DuckDB を持たない。確認: `requirements/*.lock.txt` と各環境の import） | 不採用 |
| B: 使う行だけ `repro.assets` へ写し、出所を `catalog_snapshots` に記録 | 再構築が staging だけで閉じる。`db-ingest` の影響を受けない。素材の版（長さ・レート）が固定される | 行の複製がある | **採用** |

写す列は `corpora.audio_files` の `path`、`corpus`、`subset`、`sample_rate`、`frames`、`channels`、`speaker_id`、`utterance_id` と、必要なものだけ（§4.4）。全 1,375,481 行（2026-11-06 観測）は写さない。使うのは、リストと素材リストに入った行だけである（例: VoiceBank は 4 サブセットで約 2.5 万行。DNS2020 を丸ごと使っても約 13 万行）。

### 2.3 観測した現状（確認）

2026-11-06 02:48 に更新された `data/db/resource.duckdb`（189,280,256 バイト）を、`read_only=True` で開いて確認した。

- テーブルは `corpora.audio_files` だけで、**`eval` スキーマが無い**。`.agents/DATA.md` の「本番 `resource.duckdb` はレシピスキーマ」とは異なる。誰がいつ更新したかは未確認で、この作業では触れていない。
- 14 コーパスが入っている。設計に関係する件数は次のとおり。

| コーパス / subset | レート | ch | 件数 | 時間 |
|---|---|---|---|---|
| VoiceBank-DEMAND `clean_trainset_28spk_wav` / `noisy_trainset_28spk_wav` | 48 kHz | 1 | 各 11,572 | 各 9.39 h |
| VoiceBank-DEMAND `clean_testset_wav` / `noisy_testset_wav` | 48 kHz | 1 | 各 824 | 各 0.58 h |
| VoiceBank-DEMAND `*_trainset_56spk_wav` | 48 kHz | 1 | 各 23,075 | 各 19.03 h |
| MS-SNSD `clean_train` / `noise_train` / `clean_test` / `noise_test`（ほかに `hitapps` 20）| 16 kHz | 1 | 23,075 / 128 / 1,100 / 51 | 19.03 / 3.29 / 3.64 / 0.66 h |
| DNS2020 `datasets` | 16 kHz（188 件は 44.1 / 48 / 96 kHz） | 1 | 132,150 | 747.5 h |
| DNS5 `datasets_fullband` | 48 kHz | 1 / 2 / 16 / 30 | 124,058 | 193.9 h |
| VCTK `wav48` | 48 kHz | 1 | 44,085（話者 108） | 43.9 h |
| WHAM48 `high_res_wham` | 48 kHz | 2 | 1,578 | 78.5 h |
| OpenSLR26 `sim_rir_16k` / OpenSLR28 `RIRS_NOISES` | 16 kHz（OpenSLR26 には `sim_rir_8k` 60,000 件が 8 kHz で別にある）| 1（RIRS_NOISES の 379 件は 2〜30 ch）| 60,000 / 61,260 | 19.44 / 25.5 h |

- VoiceBank-DEMAND は **48 kHz のみ**で、16 kHz 版は無い。MP-SENet と MossFormerGAN の公式は 16 kHz のリストを使う。公式の配布物は**事前に 16 kHz へ変換済みで、変換器は不明**（UNKNOWN。[mp_senet.md](../training/reproduction/mp_senet.md)、[mossformergan.md](../training/reproduction/mossformergan.md)）。変換は派生素材（§4.3 の `derivations`）として明示し、§11 と DATA_PIPELINE_DESIGN の ResampleSpec で扱う。
- `data/staging/` は空。`data/db/` に `resource.duckdb` 以外は無い。

この状態が、再現用 DB の設計に影響する点: (a) `eval` スキーマに頼らない（再現用 DB は自前のスキーマ）。(b) leakage 監査が評価セットと突き合わせるときは、`eval` が無ければ **UNKNOWN** とし、PASS にしない（§7.4）。

### 2.4 学習と評価で共有するもの、分離するもの

| 区分 | 内容 |
|---|---|
| 共有する | 素材の索引（`corpora.audio_files` を出所とする）、`src/data/recipe.py` の正規化 JSON とハッシュ（`json_hash`、`sha256_file`）、`mix_seed` 型の SHA-256 seed、音声の読み込みと明示的なリサンプル、固定ペアと事前生成物の「レシピ JSON v1 で表せるケース」（同じ `case_id`） |
| 分離する | 公式学習データの生成器（MS-SNSD、DNS の合成、DFN3 と SEMamba++ のオンライン混合）。評価の `rms_v1` 混合とは定義が違う（TF-GridNet は名目 SNR の平方根、DFN3 は gain とクリップ保護、SEMamba++ は有音フレーム基準）。評価コード（`src/evaluation/`）はこれらを import しない。公式再現の test を validation に使うデータ定義。公式評価の指標（`src/evaluation/official/`、将来）|

評価の `eval.cases` に操作ごとの列を足さない規則（AGENTS.md）は変えない。

---

## 3. Protocol と Case の分離

### 3.1 定義

| 概念 | 定義 | 識別子 | 不変か |
|---|---|---|---|
| **素材（asset）** | 1 つの音声ファイル。元のコーパス、派生（リサンプル）、生成物のいずれか | `asset_id`（`<root>/<相対パス>`） | 内容は不変 |
| **素材リスト（material set）** | 役割（speech / noise / rir）が付いた順序付きの素材の列。オンライン混合の抽選母集団、事前生成の入力 | `material_set_id`（内容ハッシュ） | 不変 |
| **Protocol** | **音声を作る規則**: 混合の確率分布、処理順序、正規化、crop、教師の定義、epoch の単位。素材集合を含まない | `protocol_id`（規則文書のハッシュ） | 不変。変更は新しい版 |
| **Case / Realization** | Protocol から実際に作られた**具体的な 1 サンプル**。素材、抽選で決まった全パラメータ、教師の定義を持つ。その文書だけで波形を再生成できる | `case_id`（文書のハッシュ） | 不変 |
| **ケース集合（case set）** | 順序付きの Case の列。公式 split（train / valid / test）、固定ペアのリスト | `case_set_id` | 不変 |
| **Dataset 定義** | Protocol と、素材リスト・ケース集合を**スロット**に束ねたもの。プロファイル、主張ラベル、選択にテストを使うかを持つ | `dataset_id` | 不変 |
| **Build** | 事前生成の 1 回の定義（生成器、パラメータ、seed、入力素材）。出力ファイルは素材として登録する | `build_id` | 定義は不変 |

Protocol は素材を含まない。素材だけが違う Dataset（公式素材と代替素材）は、**同じ Protocol を別の Dataset 定義が束ねる**。これは D-05 / D-06 の「代替素材は別プロトコル準拠実験」を、`material_origin` と `claim_label`（§4.9、§7）で区別する。Protocol 自体が変わる場合（公式の欠陥を直した版など）は、`parent_protocol_id` と `deviation_class` を持つ**別の Protocol** にする（D-02）。

### 3.2 ID の規則

| 対象 | ID |
|---|---|
| 固定ペア・レシピ v1 で表せる Case | **既存のレシピ JSON v1 の `case_id`**（`src/data/recipe.py` の `case_id_for`）。`resource.duckdb` / `voicebank.duckdb` と同じレシピは同じ ID になる（§3.3） |
| その他の Case（事前生成、オンライン混合の実現値） | `realization_v1` 文書（§3.4）の正規化 JSON の SHA-256 |
| Protocol | `{"doc_type": "protocol", "algorithm", "algorithm_version", "kind", "rule", "deviation_class", "parent"}` の正規化 JSON の SHA-256。名前と版は含めない（同じ規則が同じ ID）。名前と版は `UNIQUE(name, version)` の別列 |
| Dataset 定義 | `{"doc_type": "dataset", "protocol_id", "bindings": [スロット → 集合のハッシュ]、"profile", "material_origin", "claim_label", "selection_uses_test", "declared_status"}` |
| 素材リスト、ケース集合、Build | 各表の説明（§4） |

文書にバージョン（`realization_version`、`recipe_version`、`algorithm_version`）を必ず含め、未知の版は拒否する（`recipe.py` と同じ方針）。ハッシュの定義（キーのソート、空白なし、浮動小数は小数 6 桁）は `recipe.py` の `_dump` / `_normalize` を再利用する（確認: `src/data/recipe.py` L252-271）。

### 3.3 既存レシピ JSON v1 との互換

- v1 は「確定した 1 ケース」の生成条件であり、**変更しない**。確率分布や複数雑音を表すために v1 を拡張しない。
- 再現用 DB の `repro.cases.doc_kind` は `recipe_v1` と `realization_v1` の 2 種類。
  - `recipe_v1`: 文書は v1 そのもの、`case_id` は v1 のハッシュ。VoiceBank の固定ペア（`operations: []`、`output.reference: dry_clean`）と、v1 で表せる 1 本雑音の混合がこれに当たる。`render_recipe`（`src/evaluation/audio/render.py`）が波形を作れるが、学習 Pipeline は Renderer の実装を共有する必要はない（§DATA_PIPELINE 6）。
  - `realization_v1`: v1 で表せないケース（複数雑音の連結、RMS 基準のレベル正規化、clip 保護、名目 SNR と実 SNR の区別、残響の詳細、音声の連結など）。
- v1 の `sources.*.path` は DB ファイルからの相対パスで、`..` を含む。**再現用 DB の `recipe_v1` Case もこの規則に従わせると、`reproduction.duckdb` の置き場所で `case_id` が変わる。** そこで次の規則にする: `recipe_v1` Case を再現用 DB に入れるときは、`resource.duckdb` と同じ置き場所（`data/db/`）からの相対パスで書く。`reproduction.duckdb` も `data/db/` に置くので、同じ文書になる。別の場所に置く場合は ID が変わるため、その DB を使わない（実装時にテストで固定する）。
- 事前生成物のうち、`recorded` として v1 で表せるもの（ファイル 2 本を読むだけ）は、生成物の素材を v1 で包んだ別の `recipe_v1` Case としても登録できるが、**生成条件の正本は `realization_v1` 文書**である（§9）。v1 の包みは学習の読み取り用の薄い表現にすぎず、必須ではない。

### 3.4 `realization_v1` 文書

Renderer は文書だけの純関数（ファイルの内容と実装の版は別）。**乱数の抽選は Planner が行い、結果を文書に書く。** Renderer は乱数を引かない（レシピ v1 が seed を再計算しないのと同じ性質）。

```json
{
  "realization_version": 1,
  "algorithm": "dfn3_online",
  "algorithm_version": 1,
  "sample_rate": 48000,
  "sources": {
    "speech": [{"asset": "corpora/…/a.wav", "sha256": "…", "start": 1200, "length": 144000, "gain_db": 0.0}],
    "noise":  [{"asset": "corpora/…/n1.wav", "sha256": "…", "start": 0, "gain_db": -6.0}],
    "rir":    {"asset": "corpora/…/r.wav", "sha256": "…", "channel": 0}
  },
  "draws": {"snr_db": 10, "gain_db": 6, "n_noises": 2, "p_reverb_speech": true, "…": "…"},
  "steps": ["speech_concat", "speech_aug", "noise_combine", "reverb", "mix", "clip_guard"],
  "output": {"length": 144000, "teacher": "official_mixed_target"}
}
```

- `sources`、`draws`、`steps` の語彙は `algorithm` ごとに決まり、`src/data/reproduction/realization/<algorithm>.py` が検証する。未知のキーは拒否する。共通の外枠は `realization_version`、`algorithm`、`algorithm_version`、`sample_rate`、`sources`、`draws`、`output` だけ。
- `teacher` は `dry_clean`、`reverberant_clean`、`official_mixed_target`（DFN3 の 0.3 乾き + 0.7 後期減衰）、`none` のいずれか。他の値は拒否する。
- Case の ID は `algorithm` と `algorithm_version` を含むので、Renderer の意味が変わるときは版を上げる。`protocol_id` は文書に入れない。同じ実現値が複数の Protocol・モデルで共有できる。
- 文書は 1 件あたり約 1 KB（DFN3 で雑音が最大 5 本）。360,000 件で約 0.4 GB の JSON 列。DuckDB で扱える。

### 3.5 3 方式の表現

| 方式 | Protocol の `kind` | 素材 | Case | 学習時に読むもの |
|---|---|---|---|---|
| 固定ペア（MP-SENet、MossFormerGAN） | `fixed_pair` | clean と noisy の素材（`cases` が直接参照） | 1 ペア = 1 `recipe_v1` Case（`case_role: recorded_pair`） | Case が指す clean / noisy の素材 |
| 事前生成（TF-GridNet、TF-Locoformer） | `pregenerated` | 生成物（`storage_root: generated`）を素材として登録。入力素材リスト（speech / noise）は Build の入力 | 生成した 1 混合 = 1 `realization_v1` Case（`case_role: generated`）。`build_id`、`gen_index` を持つ | Case が指す生成物の noisy / clean |
| オンライン混合（DFN3、SEMamba++） | `online_mixing` | 素材リスト（speech / noise / rir）を Dataset 定義のスロットに束ねる | 学習: **保存しない**（Planner が `(seed, epoch, index)` から実現値を作る）。validation / test: 実現値の文書を**固定してケース集合に保存**（`case_role: online_snapshot`、音声は不要） | 学習は素材リストから抽選して Render。validation はケース集合の文書を Render |

オンライン混合の学習サンプルの識別子は `realization_id`（文書のハッシュ）。必要なときにだけ計算し、学習ログに残せる。DB には入れない（1 epoch で数万〜数十万、全体で数十億になる）。

---

## 4. スキーマ案

DuckDB のスキーマ名は `repro`。型は DuckDB の型。CHECK は同じ行の列だけを使う（DuckDB の CHECK は他の表を参照できない）。外部キーは DuckDB の制約により、参照される行を更新・削除できない。そのため行は追記のみで、規則どおり問題ない。表間の不変条件は登録関数（Python）が検証する。

全表に共通: 時刻・ホスト・絶対パスを持たない。`content_hash` は、その行の「ID を除く本体」の `json_hash` である（`eval.cases.content_hash` と同じ）。

### 4.1 `repro.schema_info`

| 列 | 型 | 備考 |
|---|---|---|
| `key` | VARCHAR PK | `schema_version` など |
| `value` | VARCHAR NOT NULL | `schema_version = '1'` |

登録元: スキーマ作成関数。更新: 版を上げるときは移行コマンドを別に作る（追記のみの原則の例外ではない。新しい DB ファイルへ写す）。

### 4.2 `repro.catalog_snapshots`

| 列 | 型 | 備考 |
|---|---|---|
| `catalog_id` | VARCHAR PK | 写した `corpora.audio_files` 行（`path` 順）の `json_hash` |
| `source_kind` | VARCHAR NOT NULL | CHECK IN (`resource_duckdb`, `staging_parquet`) |
| `corpora_json` | JSON NOT NULL | コーパス名ごとの件数 |
| `row_count` | BIGINT NOT NULL | |
| `source_db_sha256` | VARCHAR | 元 DB ファイルの SHA-256。計算した場合だけ |

- 目的: 素材の長さ・レートがどの時点の索引に由来するかを残す。`source_ref`（パス）は ID に入れず、staging の `meta.json` にだけ置く。
- 登録元: 準備コマンドが `resource.duckdb` を読み取り専用で読んだ結果。更新: なし。

### 4.3 `repro.derivations`

| 列 | 型 | 備考 |
|---|---|---|
| `derivation_id` | VARCHAR PK | `{kind, tool, tool_version, params}` の `json_hash` |
| `kind` | VARCHAR NOT NULL | CHECK IN (`resample`, `format_convert`) |
| `spec_json` | JSON NOT NULL | 例: `{"tool": "soxr", "tool_version": "1.1.0", "quality": "HQ", "from": 48000, "to": 16000}` |

- 派生素材（VoiceBank の 16 kHz 版など）の作り方。**公式の変換法が UNKNOWN のとき、選んだ方法をここに明示する。** 公式の方法と同じとは書かない。
- 登録元: 派生素材の作成コマンド。更新: なし。

### 4.4 `repro.assets`

| 列 | 型 | 備考 |
|---|---|---|
| `asset_id` | VARCHAR PK | `<storage_root>/<rel_path>`（例 `corpora/VoiceBank-DEMAND/clean_trainset_28spk_wav/p267_184.wav`） |
| `storage_root` | VARCHAR NOT NULL | CHECK IN (`corpora`, `generated`, `derived`) |
| `rel_path` | VARCHAR NOT NULL | ルートからの相対。`..`・絶対パス・`.` 要素を拒否 |
| `corpus` | VARCHAR | `generated` では NULL |
| `subset` | VARCHAR | |
| `speaker_id` | VARCHAR | 話者単位の分割と leakage 監査に使う |
| `utterance_key` | VARCHAR | 固定ペアの対応づけ（拡張子なしのファイル名など） |
| `sample_rate` | INTEGER NOT NULL | CHECK > 0 |
| `frames` | BIGINT NOT NULL | CHECK > 0 |
| `channels` | INTEGER NOT NULL | CHECK > 0 |
| `sample_format` | VARCHAR | `PCM_16` など。分かるときだけ |
| `size_bytes` | BIGINT | |
| `sha256` | VARCHAR | 64 桁の小文字 16 進。未計算は NULL |
| `integrity` | VARCHAR NOT NULL | CHECK IN (`sha256`, `size`, `catalog`) |
| `catalog_id` | VARCHAR | FK → `catalog_snapshots`。`corpora` で必須 |
| `derivation_id` | VARCHAR | FK → `derivations`。`derived` で必須 |
| `derived_from` | VARCHAR | FK → `assets(asset_id)`。`derived` で必須 |
| `build_id` | VARCHAR | FK → `builds`。`generated` で必須 |
| `extra_json` | JSON NOT NULL | 文字起こしのパスなど、検索に使わない属性 |

- 主キー: `asset_id`。採番 ID は使わない（`corpora.audio_files.path` と同じ方針）。
- 不変条件: `asset_id = storage_root || '/' || rel_path`。`integrity = 'sha256'` ⇔ `sha256` が NOT NULL。同じ `asset_id` で `sample_rate` / `frames` / `sha256` が違う登録はエラー。`storage_root` ごとの必須外部キー（上表）。`size` は `size_bytes` が NOT NULL。
- 索引: `(corpus, subset)`、`(sha256)`（重複内容の検出）、`(speaker_id)`、`(build_id)`。
- 登録元: `corpora` は準備コマンド（`resource.duckdb` の行）。`derived` は派生コマンド。`generated` は Build の shard マニフェスト（§9）。
- 更新方針: 追記のみ。**例外は、`sha256` が NULL の行へ値を入れる `verify --hash` だけ。** 既に値があるときの不一致はエラー（素材が変わったことを意味する）。`integrity` は同時に `sha256` へ上げる。
- 同じ内容で別パスのファイル（`clean_trainset_56spk_wav` と `28spk` の重複など）は別の `asset_id` になる。`sha256` が分かれば leakage 監査で検出できる。

整合性の強さ（`integrity`）の扱い: 公式のファイルリストに入る素材（VoiceBank 約 2.4 万本）は `sha256` を必須にする。DNS2020 の 13 万本（約 747 時間）や雑音プールは、初回は `size`（またはカタログのみ）で登録でき、`verify --hash` で上げる。`material_sets.integrity_min` が最弱の値を持ち、`claim_label = official_reproduction` の Dataset は、全リストが `sha256` であることを要求する（§7）。

### 4.5 `repro.material_sets` と `repro.material_set_members`

`material_sets`:

| 列 | 型 | 備考 |
|---|---|---|
| `material_set_id` | VARCHAR PK | `{doc_type, name, version, role, split_role, rule, members:[(asset_id, sha256 または size)]}` の `json_hash` |
| `name` | VARCHAR NOT NULL | 例 `ms_snsd.clean_train`、`vctk092.speech.train` |
| `version` | INTEGER NOT NULL | UNIQUE(`name`, `version`) |
| `role` | VARCHAR NOT NULL | CHECK IN (`speech`, `noise`, `rir`) |
| `split_role` | VARCHAR NOT NULL | CHECK IN (`train`, `valid`, `test`, `pool`) |
| `origin` | VARCHAR NOT NULL | CHECK IN (`official`, `substitute`, `derived_from_official`, `unknown`) |
| `selection_rule_json` | JSON NOT NULL | 選び方。例 `{"rule": "subset_exclude_speakers", "exclude": ["p280", "p315"]}` |
| `provenance_json` | JSON NOT NULL | 公式リストの出典（repo、commit、ファイルパス、ファイルの SHA-256）、または audio-db が作った規則 |
| `member_count` | BIGINT NOT NULL | |
| `total_seconds` | DOUBLE NOT NULL | |
| `list_sha256` | VARCHAR NOT NULL | メンバー列だけのハッシュ |
| `integrity_min` | VARCHAR NOT NULL | メンバーの `integrity` の最弱値 |

`material_set_members`:

| 列 | 型 | 備考 |
|---|---|---|
| `material_set_id` | VARCHAR | FK |
| `position` | INTEGER | 0 始まり。順序は公式リストの順 |
| `asset_id` | VARCHAR NOT NULL | FK → `assets` |
| `group_key` | VARCHAR | 話者など、分割・leakage の単位 |
| `attrs_json` | JSON | メンバー固有の属性（DFN3 の `sampling_factor` など）。無ければ NULL |

- 主キー: `material_sets.material_set_id`、`material_set_members(material_set_id, position)`。UNIQUE(`material_set_id`, `asset_id`)。
- 不変条件: メンバーの `role` に合う素材だけ（`rir` に複数チャンネルを許すかは Protocol が決める）。`list_sha256` は登録時に再計算して一致。`origin = official` の集合は `provenance_json` に出典と SHA-256 を持つ。`member_count` と実メンバー数が一致。
- 役割の表現（課題 4）: 素材の役割は**素材ではなく、使い方に付く**。同じファイルが、ある Protocol では speech、別の Protocol では noise になれる。役割は素材リストの `role` と、Dataset 定義のスロット名（`train.speech` など）で表す。
- 登録元: 準備コマンド（公式ファイルリスト、コーパスの走査、規則）。更新: なし。

### 4.6 `repro.protocols`

| 列 | 型 | 備考 |
|---|---|---|
| `protocol_id` | VARCHAR PK | §3.2 |
| `name` | VARCHAR NOT NULL | 例 `mp_senet.fixed_pair`、`dfn3.online` |
| `version` | INTEGER NOT NULL | UNIQUE(`name`, `version`) |
| `kind` | VARCHAR NOT NULL | CHECK IN (`fixed_pair`, `pregenerated`, `online_mixing`) |
| `algorithm` | VARCHAR NOT NULL | 実装のレジストリキー。例 `fixed_pair.mp_senet`、`pregenerated.dns_tflocoformer`、`online.dfn3` |
| `algorithm_version` | INTEGER NOT NULL | 出力が変わる実装変更で上げる |
| `rule_json` | JSON NOT NULL | 規則文書。`algorithm` が検証する（§5） |
| `official_ref_json` | JSON NOT NULL | 出典: repo、commit、ファイル、`TARGETS.md` の節。必須 |
| `deviation_class` | VARCHAR NOT NULL | CHECK IN (`official_effective`, `minimal_launch_repair`, `algorithm_change`, `research_improvement`)（D-02） |
| `parent_protocol_id` | VARCHAR | FK → `protocols`。`deviation_class <> official_effective` のとき必須 |
| `deviations_json` | JSON NOT NULL | 公式との既知の差のリスト（`id`、`class`、`text`、`ref`）。D-02、D-07、D-09 |

- 不変条件: `kind` は `algorithm` の種別と一致する。`rule_json` が `algorithm` のスキーマに通る。未知の `algorithm` / `algorithm_version` は拒否する。`deviation_class <> official_effective` では `parent_protocol_id` が NOT NULL。
- 索引: PK、UNIQUE(`name`, `version`)。
- 登録元: `configs/data/protocols/*.yaml`（DATA_PIPELINE_DESIGN 12）を、準備コマンドが正規化して staging に書き、そこから登録する。更新: なし。変更は新しい版。
- **オンライン混合の確率分布**（課題 8）は `rule_json` の型付きパラメータで持つ。汎用の分布 DSL は作らない（§5）。

### 4.7 `repro.cases`

| 列 | 型 | 備考 |
|---|---|---|
| `case_id` | VARCHAR PK | §3.2 |
| `doc_kind` | VARCHAR NOT NULL | CHECK IN (`recipe_v1`, `realization_v1`) |
| `doc_json` | JSON NOT NULL | 正規化済み文書 |
| `case_role` | VARCHAR NOT NULL | CHECK IN (`recorded_pair`, `generated`, `online_snapshot`) |
| `protocol_id` | VARCHAR NOT NULL | FK → `protocols`。作った（または対応する）Protocol |
| `clean_asset_id` | VARCHAR | FK → `assets`。学習が読む教師の素材 |
| `noisy_asset_id` | VARCHAR | FK。学習が読む入力の素材 |
| `noise_asset_id` | VARCHAR | FK。TF-GridNet / TF-Locoformer の雑音参照 |
| `teacher_kind` | VARCHAR NOT NULL | CHECK IN (`dry_clean`, `reverberant_clean`, `official_mixed_target`, `none`) |
| `sample_rate` | INTEGER NOT NULL | 文書の `output.sample_rate` と一致 |
| `frames` | BIGINT | 入力の長さ。レンダー後に決まるときは NULL |
| `speaker_id` | VARCHAR | |
| `utterance_key` | VARCHAR | |
| `transcript` | VARCHAR | 持つ場合だけ。部分音声に全文を対応させない |
| `build_id` | VARCHAR | FK。`case_role = generated` で必須 |
| `gen_index` | BIGINT | Build 内の通し番号。`generated` で必須 |
| `content_hash` | VARCHAR NOT NULL | |

- 不変条件: `case_id` が文書から再計算した値と一致。`recorded_pair` は `clean_asset_id` と `noisy_asset_id` が NOT NULL で、`teacher_kind = dry_clean`、`doc_kind = recipe_v1`。型付き素材列が文書の `sources` と一致。`teacher_kind` が文書の教師定義と一致。`generated` は `build_id` と `gen_index` が NOT NULL で、素材が `storage_root = generated`。`online_snapshot` は音声列が NULL。`teacher_kind = none` の Case は学習に使えない（読み取り側でエラー）。
- 索引: PK、`(build_id, gen_index)`、`(clean_asset_id)`、`(speaker_id)`。
- 登録元: 固定ペアは準備コマンド。`generated` は Build の計画と shard マニフェスト（§9）。`online_snapshot` は準備コマンドが Planner を呼んで文書にした検証・テストの実現値。更新: なし。
- 規模の見積り: TF-Locoformer の公式規模で 360,000 行 + 素材 1,080,000 行。DuckDB の COPY（Parquet 読み込み）で登録する。Python の `executemany` は使わない。

### 4.8 `repro.case_sets` と `repro.case_set_items`

`case_sets`（`eval.sets` と同じ形）:

| 列 | 型 | 備考 |
|---|---|---|
| `case_set_id` | VARCHAR PK | `{doc_type, name, version, split_role, rule, case_ids(順序付き)}` の `json_hash` |
| `name` | VARCHAR NOT NULL | 例 `mp_senet.vb28.train.official` |
| `version` | INTEGER NOT NULL | UNIQUE(`name`, `version`) |
| `split_role` | VARCHAR NOT NULL | CHECK IN (`train`, `valid`, `test`) |
| `origin` | VARCHAR NOT NULL | `material_sets.origin` と同じ値域 |
| `selection_rule_json` | JSON NOT NULL | 例 `{"rule": "official_filelist", "file": "training.txt"}`、`{"rule": "sorted_ratio", "key": "official_uttid", "ratio": [9, 10], "part": "head"}`、`{"rule": "speaker_holdout", "speakers": ["p286", "p287"]}`、`{"rule": "fixed_subset", "of": "<case_set_id>", "k": …}` |
| `provenance_json` | JSON NOT NULL | 公式リストの出典（repo、commit、パス、**ファイルの SHA-256**）。audio-db 独自の分割はそう明記する |
| `member_count` | BIGINT NOT NULL | |
| `list_sha256` | VARCHAR NOT NULL | |

`case_set_items`: `case_set_id`（FK）、`position`（INTEGER）、`case_id`（FK → `cases`）。主キー(`case_set_id`, `position`)、UNIQUE(`case_set_id`, `case_id`)。

- 公式分割の保存（課題 5）: 公式のファイルリスト（`training.txt`、`test.txt`、scp、ESPnet の uttid 規則）を、**出典ファイルの SHA-256 付き**でケース集合にする。リストは追跡対象の `configs/data/lists/<model>/` へ複製して出典・ライセンスを記録する（`src/se/official/` は Git 管理外で将来削除されるので、そこを読まない。既存の `configs/db/lists/` と同じ扱い）。
- 同じ `case_id` が複数の集合に入ることを許す（公式は test を validation に兼用する）。それが許されるのは Dataset 定義の制約（§7）を通ったときだけ。
- 登録元・更新方針: 準備コマンド。更新なし。

### 4.9 `repro.datasets` と `repro.dataset_bindings`

`datasets`:

| 列 | 型 | 備考 |
|---|---|---|
| `dataset_id` | VARCHAR PK | §3.2 |
| `name` | VARCHAR NOT NULL | 例 `mp_senet.vb28.official` |
| `version` | INTEGER NOT NULL | UNIQUE(`name`, `version`) |
| `protocol_id` | VARCHAR NOT NULL | FK |
| `profile` | VARCHAR NOT NULL | CHECK IN (`official_faithful`, `leakage_safe`)（D-01） |
| `material_origin` | VARCHAR NOT NULL | CHECK IN (`official`, `derived_from_official`, `substitute`, `mixed`, `unknown`)。`derived_from_official` は公式素材から audio-db が派生した素材（16 kHz 化など。変換器が公式と同一とは限らない）|
| `claim_label` | VARCHAR NOT NULL | CHECK IN (§7.2 の 6 値) |
| `declared_status` | VARCHAR NOT NULL | CHECK IN (`ready`, `blocked`, `unknown`)。`blocked` は素材が無いなどで、バインドを持たない宣言だけの Dataset（DFN3 の公式素材など）|
| `blocking_json` | JSON NOT NULL | `blocked` / `unknown` の理由と欠けているもの。`ready` では `[]` |
| `selection_uses_test` | BOOLEAN NOT NULL | validation / モデル選択に test 集合を使うか。登録関数が束縛から**計算**して記録する |
| `selection_note` | VARCHAR | 例「公式 `train.py` が `test.txt` を validation と best 選択に使う」 |
| `deviations_json` | JSON NOT NULL | 素材・分割・生成の seed 固定など、公式との差（Protocol の差とは別） |

`dataset_bindings`:

| 列 | 型 | 備考 |
|---|---|---|
| `dataset_id` | VARCHAR | FK |
| `slot` | VARCHAR | CHECK: `{train,valid,test}.cases` または `{train,valid,test}.{speech,noise,rir}` |
| `case_set_id` | VARCHAR | FK。`*.cases` スロットで必須 |
| `material_set_id` | VARCHAR | FK。`*.speech|noise|rir` スロットで必須 |
| `params_json` | JSON | スロットの追加パラメータ（`sampling_factor`、チャンク規則を除く） |

- 主キー: `datasets.dataset_id`、`dataset_bindings(dataset_id, slot)`。CHECK: `(case_set_id IS NULL) <> (material_set_id IS NULL)`。
- 不変条件（登録関数が検証）:
  1. Protocol の `kind` に必要なスロットが揃う（`fixed_pair` / `pregenerated`: `train.cases`、`valid.cases`、`test.cases`。`online_mixing`: `train.speech` など素材スロットと `valid.cases`）。
  2. `profile = leakage_safe` なら `selection_uses_test = false`（SQL の CHECK でも持つ）。かつ train / valid / test の素材・話者・内容 SHA-256 が重ならない（§7.3）。
  3. `selection_uses_test` は、`valid.cases` と `test.cases` の `case_id` に共通部分があるとき、または `valid` が `test` 由来の素材のとき真。登録関数が計算し、列の値と一致しなければ拒否。
  4. `claim_label = official_reproduction` なら、`material_origin` が `official` または `derived_from_official`（後者は、派生の `derivations` を `deviations_json` に項目として列挙し、公式の変換器が UNKNOWN の間はその旨を書く）、Protocol の `deviation_class = official_effective`、全素材リストが `integrity_min = sha256`、`declared_status = ready`。
  5. `declared_status = blocked | unknown` の Dataset はバインドを持たず、学習に使えない（スナップショットを作らない）。
- 登録元: `configs/data/datasets/*.yaml`。更新なし。

### 4.10 `repro.builds`（事前生成の定義）

| 列 | 型 | 備考 |
|---|---|---|
| `build_id` | VARCHAR PK | `{doc_type, generator, generator_version, params, seed, input_material_set_ids}` の `json_hash`。**生成件数は含めない**（§9.6 の段階生成で小さい Build が大きい Build の先頭になる） |
| `generator` | VARCHAR NOT NULL | 例 `ms_snsd_synth`、`dns_synth` |
| `generator_version` | INTEGER NOT NULL | |
| `params_json` | JSON NOT NULL | 公式パラメータ（`audio_length`、SNR 範囲、レベル範囲…）と audio-db の追加（`shard_size`、seed の導出式） |
| `seed` | BIGINT NOT NULL | 基準 seed。各ファイルの seed は `mix_seed("gen", seed, gen_index)` |
| `inputs_json` | JSON NOT NULL | スロット（`speech`、`noise`）→ `material_set_id` |
| `shard_size` | INTEGER NOT NULL | 既定 1000 |

登録元: Build 計画コマンド（§9）。更新なし。**完了の記録はこの表に持たず、`audits` の `build_verification` と staging の検証記録で持つ。** 完了の有無を可変な列にしないため。

### 4.11 `repro.audits`

| 列 | 型 | 備考 |
|---|---|---|
| `audit_id` | VARCHAR PK | `{subject_kind, subject_id, audit_kind, report}` の `json_hash` |
| `subject_kind` | VARCHAR NOT NULL | CHECK IN (`build`, `dataset`, `material_set`, `case_set`) |
| `subject_id` | VARCHAR NOT NULL | |
| `audit_kind` | VARCHAR NOT NULL | CHECK IN (`build_verification`, `count_check`, `list_hash_check`, `leak_check`, `distribution_check`, `hash_verify`) |
| `status` | VARCHAR NOT NULL | CHECK IN (`PASS`, `FAIL`, `WARN`, `UNKNOWN`)。**確認できなかったことを PASS にしない** |
| `report_json` | JSON NOT NULL | 条件、観測値、期待値、検査に使った版 |

追記のみ。同じ対象の最新の PASS を、Dataset の使用可否の判定に使う（§10.3）。登録元: 検証コマンド。

### 4.12 `repro.runs`

| 列 | 型 | 備考 |
|---|---|---|
| `run_id` | VARCHAR PK | metrics.jsonl の最初の `run` 記録の `json_hash` |
| `dataset_id` | VARCHAR NOT NULL | FK |
| `snapshot_hash` | VARCHAR NOT NULL | 使ったスナップショットのハッシュ |
| `config_hash` | VARCHAR NOT NULL | 学習設定のハッシュ |
| `git_commit` | VARCHAR NOT NULL | 来歴。ID には入れない |
| `world_size` | INTEGER NOT NULL | |
| `ddp_mode` | VARCHAR NOT NULL | CHECK IN (`single`, `real`, `virtual`) |
| `selection_uses_test` | BOOLEAN NOT NULL | Dataset の値の複写。比較用 |
| `data_state_json` | JSON NOT NULL | 最後の checkpoint の `data_state`（DATA_PIPELINE_DESIGN 9）|

学習が終わった（または止まった）後に、1 プロセスの書き込みで登録する。**学習中は書かない。** checkpoint ディレクトリの `data_provenance.json` が一次記録で、この表はそれを後から載せる（既存の `register_run` と同じ順序）。

P4 の登録関数は `register_training_run`（`python -m src.reproduction.cli register-run`）。学習プロセスは呼ばない。`--database` の既定値は無く、本番 `reproduction.duckdb` を暗黙には開かない。同じ `run_id` で dataset・snapshot・config・git・world size・`ddp_mode`・`selection_uses_test`・claim・データ計画が同じなら、checkpoint の位置（`epoch`、`update_in_epoch`、`plan_hash`）と checkpoint パスだけの変化は同じ run の更新である。それらが違う再登録は拒否し、行を変えない。別の `run_id` は別 run として挿入する。

`profile` は checkpoint の `data_state` に既にある。`claim_label`、`checkpoint_path`、`checkpoint_sha256` は表に列が無いので、スキーマを変えずに `data_state_json` へ足す。再開が読むのは checkpoint 本体の `data_state` であり、この足したキーは DB 側の登録情報である。

### 4.13 表の関係

```
catalog_snapshots ─┐
derivations ───────┼─< assets >─────────────┐ (clean/noisy/noise)
builds ────────────┘      ^                  │
   │                      │ member            │
   │              material_set_members        │
   │                      │                  │
   │               material_sets       cases >─ case_set_items >─ case_sets
   │                      ^              ^                           ^
   └──────────────────────┼──────────────┼───────────────────────────┤
                          └── dataset_bindings ── datasets ── protocols (parent)
                                                    │
audits (subject_kind/id)  runs ─────────────────────┘
```

---

## 5. JSON と型付きカラムの役割分担

| 型付きカラム（検索・結合・制約に使う） | JSON（文書として検証・ハッシュする） |
|---|---|
| ID、名前、版、種別、`role`、`split_role`、`profile`、`claim_label`、`teacher_kind` | `rule_json`（Protocol の規則）、`doc_json`（Case の文書） |
| 素材の属性（レート、フレーム数、話者、整合性） | `selection_rule_json`、`provenance_json`、`params_json`、`attrs_json` |
| 素材 ID への外部キー（`clean_asset_id` など） | `deviations_json`、`blocking_json`、`report_json`、`extra_json` |
| 集合の件数・ハッシュ | |

規則:

1. 「条件で絞る」「結合する」「CHECK する」列は型付きにする。それ以外は JSON。JSON の中身で SQL の WHERE を組まない。
2. JSON はすべて、`algorithm`（または `selection_rule.rule`）ごとの Python のデータクラスで検証し、**未知のキーを拒否**する。検証を通ったものを正規化してハッシュする。
3. JSON のスキーマ版は `algorithm_version`、`realization_version`、`recipe_version`、`rule` 名に持つ。
4. **EAV（キー・値の行）や、汎用の「分布 DSL」は採用しない。** DFN3 の `snrs`（7 値の一様）、雑音 2〜5 本、合成雑音 5 %、SEMamba++ の劣化確率、TF-Locoformer の整数 SNR などは、**各 `algorithm` の型付きデータクラスのフィールド**にする。公式の特殊処理を無理に共通化しない方針による。
5. 将来の Protocol 追加は、`algorithm` のレジストリへ実装とデータクラスを足し、テストの黄金文書を固定することで行う。スキーマ（表）は変えない。

---

## 6. ID・ハッシュ・版の管理

### 6.1 ハッシュの連鎖

```
file bytes ──sha256──► assets.sha256
assets (id, sha256|size) ──► material_sets.list_sha256 ──► material_set_id ─┐
cases.doc ──► case_id ──► case_sets.list_sha256 ──► case_set_id ────────────┤
protocol rule ──► protocol_id ──────────────────────────────────────────────┼─► dataset_id ─► snapshot_hash ─► run
builds (generator, params, seed, inputs) ──► build_id ──► (shard manifests) ─┘
```

| 対象 | ハッシュの入力 | 用途 |
|---|---|---|
| 素材 | ファイル内容の SHA-256 | 再生時の検証、重複検出、leakage 監査 |
| 素材リスト・ケース集合 | 順序付きメンバー（ID と素材の SHA-256 または大きさ） | 「同じリスト」の判定。公式ファイルの SHA-256 とは別に持つ |
| 出典ファイル | 公式リスト（`training.txt` など）の SHA-256（`provenance_json`） | 公式リストとの同一性 |
| Protocol・Case・Dataset | §3.2 | 版の識別 |
| Build の shard | 各 shard のマニフェスト（ファイルごとの SHA-256）のハッシュ | 生成物の同一性（§9） |
| スナップショット | 各テーブルの `json_hash` を連結した `snapshot_hash` | run と checkpoint の対応 |

### 6.2 版と来歴の区別

- **意味の版**（ID に入る）: `algorithm_version`、`realization_version`、`recipe_version`、`generator_version`、`version`（名前に付く整数）。出力が変わる変更では必ず上げ、古い版は残す。
- **来歴**（ID に入らない）: Git commit、ライブラリ版（`torch`、`torchaudio`、`librosa`、`soxr`、`numpy`）、環境。Git commit は無関係な変更で ID を変えないため入れない。代わりに、登録時、Build 時、run 時に `meta.json` と `data_provenance.json` へ記録する。
- 出力が変わる実装変更を版の更新なしに入れる事故を防ぐため、各 `algorithm` に**黄金文書**（固定した入力と seed に対する Case 文書）のテストを持たせる（IMPLEMENTATION_PLAN の受け入れテスト）。

### 6.3 「同じデータセットを再構築できる」保証

| レベル | 保証 | 根拠・検証 |
|---|---|---|
| DB の内容 | **ビット一致**: 同じ staging から作った DB は、登録順が違っても全表の内容が一致する | 内容アドレスの ID、位置の明示、時刻を持たない列。2 回の再構築で全表の `json_hash` が一致することをテストする（既存の staging のテストと同じ型） |
| 事前生成の計画 | **文書の一致**: Build 定義（生成器、パラメータ、seed、入力リスト）から、同じ `realization_v1` 文書の列が再び得られる | 各ファイルの seed を `gen_index` から導出し、共有カウンタに依存しない（§9.2）。計画だけを作り直して `case_id` が一致することをテストする |
| 事前生成の波形 | **一致を要求しない**: 同じ環境・版なら再生成で SHA-256 が一致するはずだが、保証しない。異なる環境では、フレーム数と統計の一致で確かめる | D-09。ライブラリ・並列の順序に依存する差は許容し、`meta.json` に版を残す |
| オンライン混合 | **実現値の文書が `(seed, epoch, index)` の関数**。波形は版が同じなら再現 | DATA_PIPELINE_DESIGN 9 |

---

## 7. Official / Research の分離

### 7.1 層ごとの分離

| 層 | Official-faithful | Research / Leakage-safe |
|---|---|---|
| データ定義の置き場 | `reproduction.duckdb` の `profile = official_faithful` の Dataset | `reproduction.duckdb` の `profile = leakage_safe`、または既存の `training.duckdb` / `voicebank.duckdb`（Legacy） |
| validation / test | 公式の使い方を保持する。test を validation とモデル選択に使ってよい（D-01）。`selection_uses_test = true` を記録 | train / validation / test は独立。test は選択に使わない |
| 評価用 DB | **持ち込まない**（`resource.duckdb` に公式再現の Dataset を入れない） | `resource.duckdb` の名前付きセット（既存）が Research Evaluation の入力 |
| 結果ディレクトリ | 学習: `data/results/training/<run>/`。公式評価: `data/results/se-official-eval/<run_id>/`（将来） | 学習: 同上。Research Evaluation: `data/results/se-evaluation/<run_id>/`（既存） |
| checkpoint | `data_provenance.json` に `profile`、`selection_uses_test`、選択に使った集合（case set ID と素材の SHA-256 の集合）を記録 | 同上（`selection_uses_test = false`） |
| 主張 | `claim_label` と §7.2 の表記規則 | `leakage_safe_comparison` |

### 7.2 主張ラベル（`claim_label`）

結果・文書・図は、この値から作った表記を使う。**他の言い方をしない。**

| ラベル | 条件 | 表記 |
|---|---|---|
| `official_reproduction` | 公式素材（または公式素材からの記録付きの派生）+ `official_effective` + `official_faithful` + 全素材 SHA-256。派生の差は `deviations_json` に必ず出す | 「公式再現」（派生の差を併記）|
| `protocol_conformant_substitute_materials` | 公式の混合分布・処理に従い、素材が代替（D-05、D-06） | 「公式プロトコル準拠・代替素材」。**公式データセットの再現と呼ばない** |
| `espnet_public_se_recipe_reproduction` | TF-GridNet。ESPnet の公開 SE レシピの再現（D-07） | 「ESPnet 公開 SE レシピの再現」。**論文の実験の再現ではない**。seed 固定は公式との差として併記 |
| `official_recipe_scaled_down` | 公式の生成式を縮小して生成（D-08 の段階生成の途中） | 「公式レシピの縮小版」。公式の規模・データとは別 |
| `minimal_repair_reproduction` | `minimal_launch_repair` の Protocol を使う（D-02、D-05） | 「最小補修を含む再現」。補修の一覧を併記 |
| `leakage_safe_comparison` | `leakage_safe` | 「leakage-safe 比較」 |

CHECK 制約と登録関数が、条件に合わないラベルを拒否する（§4.9）。**ラベルの上書きは、新しい版の Dataset 定義としてのみ可能**で、その版を使うかはユーザーが決める。

### 7.3 leakage の検査（`leak_check`）

`leakage_safe` Dataset は次を登録時に検査し、`audits` に記録する。1 つでも FAIL なら登録を拒否する。

1. train / valid / test の素材が `asset_id` で重ならない。
2. 同じ**話者**（`speaker_id`）が train と valid / test にまたがらない。話者が定義できない素材は検査不能として `WARN`。
3. 内容 SHA-256 が重ならない（`sha256` があるものだけ。無いものは `UNKNOWN`）。
4. 既存の評価用名前付きセット（`resource.duckdb` の `eval.*`、読み取り専用）の素材と重ならない。**`eval` スキーマが無いとき（現状）は `UNKNOWN` とし、PASS にしない**（確認: §2.3）。

`official_faithful` Dataset では、1〜3 のうち test と validation の重なりは**許す**（D-01）。許した結果は `selection_uses_test = true` と `selection_note` に必ず現れる。train と test の重なりは、公式でも許されないので検査して FAIL にする。

### 7.4 結果への伝播

- 公式再現の checkpoint で、モデル選択に使った集合（例: VoiceBank-DEMAND の公式 test 824）は、**Research Evaluation の独立 test として使わない**。評価の結果に `checkpoint_selection_overlap`（重なった評価セット名）と `independent: false` を記録する（IMPLEMENTATION_PLAN フェーズ 4）。重なりの判定は、評価セットのレシピに含まれる素材の SHA-256 と、`data_provenance.json` が持つ選択集合の SHA-256 の突き合わせで行う。名前の一致に頼らない。
- 既存の学習 API（`load_recipe_dataset`）の規則（評価用セットと test の学習・検証への使用拒否）は変えない。公式再現の例外は、新 Pipeline が Dataset 定義の `selection_uses_test = true` かつ学習設定の `data.allow_test_selection: true`（明示）の**両方**があるときだけ許す。

---

## 8. 登録・再構築

### 8.1 staging の構造（`data/staging/reproduction/`）

内容アドレスのディレクトリにして、書き込みは 1 回だけにする。既存のディレクトリと内容が違うときはエラー。

```
data/staging/reproduction/
  catalog/<catalog_id>/audio_files.parquet, meta.json        # 写した索引の行と出所
  assets/<batch_id>/assets.parquet, meta.json                # 素材（corpora / derived）
  materials/<material_set_id>/set.json, members.parquet
  protocols/<protocol_id>.json
  cases/<case_set_id>/cases.parquet, items.parquet, set.json
  datasets/<dataset_id>/dataset.json, bindings.json
  builds/<build_id>/build.json, plan/part-NNNN.parquet, shards/shard-NNNN.parquet, verified/…   # §9
  audits/<audit_id>.json
  INDEX.json                                                  # 追記専用の一覧（ID と種別）
```

- `meta.json`: 作成時刻、Git commit、環境、元ファイルのパス（ID に入れない情報）。
- 事前生成の音声は staging に入れない（`data/generated/`）。**staging には再構築に必要なマニフェストだけを置く**（音声の保存と DB 再構築元の責務を分ける）。

### 8.2 登録の手順

1. 準備コマンドが、素材と規則から staging の各ファイルを**先に**書く。
2. 登録関数（`src/data/reproduction/registry.py`）が staging を読み、全体を検証してから、**1 トランザクション**で `reproduction.duckdb` へ入れる。失敗したバッチは残らない。
3. 同じ ID・同じ内容は何もしない（冪等）。同じ ID で内容が違えばエラー。
4. Parquet からの一括登録（staging は DuckDB が書く Parquet）（`INSERT … SELECT … FROM read_parquet`）を使う。1 行ずつの INSERT は使わない。
5. 登録は書き込みプロセスが 1 つのときだけ。`data/db/.reproduction.lock` を `flock` で排他取得し、取れなければ失敗する。

### 8.3 再構築

- `reproduction-rebuild-db` は、staging から**新しいファイル**を作る。既存の DB を上書きしない（`bootstrap` の規則と同じ）。
- 検査: 再構築した DB と元の DB の全表の `json_hash` が一致する。
- `data/db/reproduction.duckdb`、`data/staging/reproduction/`、`data/generated/`、`data/db/snapshots/` は、AGENTS.md の「削除・上書き・再作成をしない」一覧へ追加する（フェーズ 1 で正本を更新する。今回は更新しない）。

---

## 9. 事前生成の保存・再開

対象: TF-GridNet（MS-SNSD）、TF-Locoformer（DNS 2020 の合成 3000 時間）。

2026-11-06 の変更: 学習に必要なのは Build の計画（`realization_v1`）と元素材の参照である。`rule_json.delivery = on_demand` の Case は `clean_asset_id` と `noisy_asset_id` を空にし、スナップショットの `docs.jsonl` に文書を書く。生成 WAV は資産にしない。`build_verification` は条件の検査で、`wav_files = 0` と `shard_ids = ["conditions"]` を記録する。公式規模は `source_scan: full` を明示した設定だけが全ファイルをハッシュする。その結果が `tf_gridnet.ms_snsd.espnet_recipe@1`（train 7,345 / valid 1,835 / test 305）である。明示が無い上限なし走査は拒否する。

下の WAV 保存の表は、保存済み WAV を読む旧 Dataset（`tf_gridnet.ms_snsd.smoke@1`）の履歴である。混合 WAV を書く経路は無く、新しい学習 Dataset は条件から PCM_16 を再構築する。

### 9.1 保存の責務

| 置くもの | 場所 | Git | 備考 |
|---|---|---|---|
| 生成した音声（noisy / clean / noise の wav） | `data/generated/<generator>/<build_id[:16]>/<stream>/s<NNNN>/fid<NNNNNNN>.wav`（`data/generated` は `data/corpora` と同様、容量の大きい外部領域へのシンボリックリンクを許す。`AUDIO_DB_GENERATED_ROOT`） | 管理外 | 1 ディレクトリ 1,000 ファイル（`shard_size`）。ファイル名は `gen_index` だけから決まる |
| Build 定義、計画（実現値の文書）、shard のマニフェスト、検証記録 | `data/staging/reproduction/builds/<build_id>/` | 管理外 | 再構築元。登録後も消さない。容量は小さい |
| DB の行（`assets`、`cases`、…） | `reproduction.duckdb` | 管理外 | マニフェストから登録 |

### 9.2 生成の決定性（D-07、D-08、D-09）

公式の生成は、未シード乱数、共有カウンタ、`Pool` の fork のため、同じ混合が再現しない（TARGETS 3.5、3.6）。audio-db は次の**追加の決定性**を導入し、公式との差として `deviations_json` に記録する。

- `gen_index = 0 … N-1` ごとに、`rng = Generator(PCG64(mix_seed("gen", build.seed, gen_index)))` を作る。ファイルの選択、SNR、レベル、切り出しは、この `rng` からだけ決める。共有カウンタや並列の実行順に依存しない。
- clean と noise の一覧の順序は、`build.seed` で 1 回 shuffle した順序とし、その順序（素材リストのハッシュ）を `build.json` に記録する。公式の「カウンタ % 件数」の巡回に相当する選択は、`gen_index` と一覧の順序から計算する。
- 公式の癖はそのまま再現する（D-02）: MS-SNSD の実 SNR は名目の半分（`snr_mixer` の平方根）、TF-Locoformer の SNR は整数 −5…14、最終レベルは整数 −35…−16 dBFS、出力は PCM_16、noise は 0.2 秒の無音を挟んで連結。公式のスクリプトの読み直しと一致確認は実装時に行う（UNKNOWN: ライブラリの版、クリップ時の挙動）。

### 9.3 計画 → 生成 → 検証の 3 段階

1. **計画（plan）**: 全 `gen_index` の `realization_v1` 文書を作り、`plan/part-NNNN.parquet` に保存する。**音声を作る前に完了する。** 計画だけで、件数、素材の使用回数、SNR・レベルの分布、予想容量が確定する。
2. **生成（render）**: shard（`shard_size` 件）を単位に、文書から wav を作る。
3. **検証（verify）**: shard の完了と整合を確認して `verified/` に記録する。

### 9.4 中断・再開・不完全な生成物

| 項目 | 規則 |
|---|---|
| 書き込み | 各 wav は `fid….wav.part` に書き、`fsync` してから名前を変える |
| shard の完了 | shard のマニフェスト `shards/shard-NNNN.parquet`（ファイルごとの相対パス、大きさ、SHA-256、フレーム数、レート、サンプル形式）を**一時名で書いてから原子的に名前を変える**。**このファイルがあることだけが、shard 完了の証拠**。wav が揃っていても、マニフェストが無ければ未完了 |
| 再開 | 完了した shard は飛ばす。未完了の shard は、`*.part` と、その shard が作った（マニフェストの無い）wav だけを削除して作り直す。他の shard と、マニフェストのあるファイルには触れない。生成は文書の純関数なので、作り直しの結果は同じ内容になる |
| Build の完了 | `verified/<K>.json`（検証した shard の ID の集合、マニフェスト全体のハッシュ、検査結果）が書かれるまで、Build は**完了と扱わない**。`ls` のファイル数や wav の存在を完了の根拠にしない |
| 検査 | 全ファイル: 存在、大きさ、レート 16 kHz、PCM_16、フレーム数が計画と一致、SHA-256 がマニフェストと一致（全件。標本ではない）。さらに統計: SNR とレベルの分布、混合の関係（`noisy ≈ clean + noise`、量子化の許容内）、clean の活動率閾値、クリップ保護 |
| DB 登録 | `verified` に含まれる shard のファイルだけを `assets`・`cases` に登録する。Dataset 定義は、参照する全 Case の shard が検証済みのときだけ登録できる。検証済みでない Build を指す Dataset はスナップショットを作らない |
| 不整合 | マニフェストと実ファイルが食い違えば `FAIL`（`audits`）。**自動で直さない**（削除も上書きもしない） |

### 9.5 既存データの保護・生成条件の変更

- 出力先は `build_id` を含む。`build_id` が同じ Build の `build.json` と内容が違えば、書き込まずにエラーにする。
- 生成条件（生成器の版、パラメータ、seed、入力リスト）を変えると、`build_id` が変わり、新しいディレクトリになる。**古い Build を上書きも削除もしない。** 不要な Build の削除は、ユーザーの明示の指示で行う。
- 書き込み前に、`df` で空き容量を確認する（計画から求めた容量の 1.2 倍が空いていること）。足りなければ開始しない。

### 9.6 段階生成（D-08）と容量

TF-Locoformer の公式規模: 30 秒 × 360,000 本 × 3 ストリーム。1 ストリームは 3000 h × 16,000 Hz × 2 バイト = 345.6 GB で、3 ストリームで約 **1.04 TB**（TARGETS 6 D-08 と同じ導出）。`noise` ストリームを学習が使うかは UNKNOWN（`noise1` が損失に入るかを実装時に確認する。使わないなら省略は容量の節約になるが、省略は公式との差として記録する）。ファイルシステム `/gs/bs`（Lustre）の空きは 2026-11-06 に 17 TB（54 TB 中 37 TB 使用、共有）。

Build の ID は件数を含まないので、**小さい段階の生成物は、大きい段階の先頭 K 件と同じ**（同じ `gen_index` が同じ文書・同じファイル）。小さい段階で検証した shard をそのまま大きい段階へ再利用する（再生成しない）。ただし分割（9:1 の train / cv）は件数に依存するので、分割は Build ではなくケース集合の規則（`sorted_ratio`）で決め、段階ごとに別のケース集合（別の Dataset）を作る。

| 段階 | 規模 | 内容 | 主な確認 | 承認 |
|---|---|---|---|---|
| S0 | 0 | 計画だけ（全 360,000 件）。音声は作らない | 文書の決定性、分布、容量見積り、素材の使用回数 | 不要（CPU・短時間） |
| S1 | 約 1 h（120 本） | 実生成 + 検証 + DB 登録 + 学習パイプラインの 1 epoch 通し | 生成式、マニフェスト、検証、再開（途中で止めて再開） | 不要（30 分未満の見込み。超えるなら要確認） |
| S2 | 約 30 h（3,600 本、約 10 GB × 3） | 生成・登録の規模確認、チャンク iterator の実データ長での確認 | 所要時間と I/O の測定、DB 登録の規模 | 30 分以上なら要 |
| S3 | 約 300 h（36,000 本、約 104 GB × 3） | 3000 h への外挿の根拠。段階の実運用リハーサル | 並列度、再開、失敗時の扱い | **要** |
| S4 | 3000 h（360,000 本） | 公式規模 | 同上 | **要**（所要時間は S2・S3 の実測から見積る。**UNKNOWN**） |

- S3・S4 は 30 分以上のジョブである。AGENTS.md に従い、投入前に、そのジョブを含むユーザーの明示の指示か承認が要る。優先度は `-p -5`。`h_rt` を 29 分にして回避しない。
- 生成コマンドは `APPROVAL=<記録>` を要求し、`build.json` に承認の記録を残す（承認の証跡。実際の承認は人が行う）。
- `cv_synthetic_small` の構成は **UNKNOWN のまま**（D-08）。ケース集合の規則 `fixed_subset`（`cv_synthetic` の固定部分集合）で表せるが、大きさと規則はユーザーが決める。S1〜S2 では、暫定の規則であることを `selection_rule_json` と `deviations_json` に明記し、Dataset の `claim_label` を `official_recipe_scaled_down` にする。

### 9.7 TF-GridNet の生成（MS-SNSD、D-07）

公式の `ms_snsd_create_mixture.sh` は `rm -r data/` で既存データを消す（確認: [tf_gridnet.md](../training/reproduction/tf_gridnet.md) C-1）。**audio-db は消さない。** 新しい Build ディレクトリに作る。

- 素材: `clean_train` 23,075 と `noise_train` 128（train 側）、`clean_test` 1,100 と `noise_test` 51（test 側）。train 側と test 側は別の Build 入力にする。
- 公式の生成式（C-1 の 2〜6）: `audio_length` 10 s、`silence_length` 0.2 s、各ファイルを −25 dBFS RMS で読む、名目 SNR は `linspace(0, 40, 5)` = [0, 10, 20, 30, 40] の 5 値すべてで同じクリップを作る（実 SNR は名目の半分）、`total_hours_train = 30`、`total_hours_test = 1`（5 SNR のコピーを数えた時間。一意なクリップは約 6 h / 0.2 h）。
- 分割: 生成した train 側 noisy をファイル名順に並べ、先頭 80 % を `tr_ms_snsd`、残りを `cv_ms_snsd`（同じクリップ N の 5 SNR は同じ側に入る）。ケース集合の規則 `sorted_ratio` で表す。
- 公式の生成は未シードで、実際の件数は UNKNOWN（TARGETS）。audio-db は §9.2 の決定性を足し、`deviations_json` に「seed 固定。公式の実物は再現不能」と書く（D-07）。**件数は公式の実測値が無いので、`total_hours` から生成を打ち切る規則**（`build.json` に記録）で決め、生成後の件数を `audits` に残す。
- 容量: 1 ストリームあたり 30 h × 16 kHz × 2 B ≈ 3.5 GB（noisy / clean / noise の 3 ストリームで約 10 GB）。段階を分ける必要は無く、1 回の生成でよい。30 分以上かかる場合は承認が要る。
- 再現ラベル: `espnet_public_se_recipe_reproduction`。論文の実験の再現ではない。

---

## 10. DuckDB の読み書き

### 10.1 書き込みと読み取りの分離（課題 12、13）

| 主体 | DB の使い方 |
|---|---|
| 準備・登録・生成・検証のコマンド | **書き込み。1 プロセスだけ**。`.reproduction.lock` を取得。学習のスナップショット作成と同時に実行しない |
| スナップショット作成 | **読み取り専用**で開き、Parquet + JSON を書き出して閉じる。rank 0 だけ |
| 学習プロセス（全 rank） | **DB を開かない。** スナップショットを読む |
| 評価 | `reproduction.duckdb` を開かない（`src/evaluation/` は `src.data.reproduction` を import しない。テストで固定） |
| run の登録 | 学習終了後、書き込み 1 プロセス |

### 10.2 スナップショット（`data/db/snapshots/<dataset_id>/`）

学習が使う不変の 1 式。内容:

- `dataset.json`: Dataset 定義、Protocol の `rule_json`、バインド、`claim_label`、`selection_uses_test`、`algorithm` と版。
- `assets.npz`: 参照される素材だけ（`asset_id`、`rel_path`、`sample_rate`、`frames`、`channels`、`sha256`、`integrity`）。`storage_root = generated` の素材は、行を持たず、`(build_id, stream, gen_index)` からパスを計算する（行は検証用の件数とハッシュだけ）。
- `cases.npz`、`case_set_items.npz`、`material_set_members.npz`: 参照されるものだけ。列は学習に要るものだけ（事前生成の Case は `doc_json` を含めない。`online_snapshot` と固定ペアは文書を `docs.jsonl` に持つ）。
- 形式: `numpy` の `.npz`（pickle を使わない。文字列は UTF-8 のバイト列とオフセットの配列）と JSON / JSON Lines。**ハッシュは形式でなく論理的な行の内容**（DB の `json_hash` と同じ定義）から計算するので、形式の変更でハッシュは変わらない。DuckDB は `db/snapshot.py`（書き出し）だけが使う。
- `manifest.json`: 各ファイルの論理ハッシュ、`snapshot_hash`、元 DB の `schema_version`、作成時の Git commit・環境（来歴。ハッシュには入れない）。

理由:

1. **DuckDB の版差と不在**。`DEVELOPMENT.md` は `.venvs/semamba` の DuckDB 1.4.3 が 2.0 のファイルを開けないと記録する（この作業ツリーに `.venvs/semamba` は無い）。`.venvs/deepfilternet` は DuckDB 自体を持たない（DFN3 の学習環境）。**スナップショットは `numpy` と JSON だけで読める形式にし、学習プロセスに DuckDB を要求しない。**
2. **ファイルロックと共有**。Lustre 上の同じ DB ファイルを複数 rank・複数ジョブが開くことを避ける。DuckDB は書き込みプロセスが 1 つあると他のプロセスが開けない。読み取りだけなら複数プロセスが同時に開けるが、学習中に準備コマンドが書き込むと失敗する。
3. **再現**。run が使ったデータが 1 つのハッシュで決まる。DB を後で更新しても、run が使った版を再現できる。

スナップショットは派生物であり再生成できるが、**書き込みは 1 回だけ**（同じ `dataset_id` のディレクトリがあり内容が違えばエラー）。

### 10.3 使用可否の判定

スナップショット作成コマンドは、次をすべて満たす Dataset だけを書き出す。

1. `declared_status = ready`。
2. 全バインドの素材が存在する（ファイルのサイズ検査）。`sha256` のある素材は、スナップショット作成時に**サイズが一致**すること（内容の検証は `verify --hash`。学習のたびに全ファイルのハッシュを計算しない）。
3. 該当する `audits` に、`count_check`、`list_hash_check`、`leak_check`（`leakage_safe` のみ）の最新が `PASS`（`UNKNOWN` は不可。ユーザーが明示して `--allow-unknown-leak-check` を付けた場合だけ許し、`manifest.json` に記録）。事前生成の Dataset は `build_verification` の PASS。

---

## 11. 6 モデルの表現（要約）

詳細な規則は各 Protocol の `rule_json` と DATA_PIPELINE_DESIGN 7。この節は DB 上の置き方だけを示す。

| モデル | Protocol（`algorithm`） | 方式 | Dataset 定義（`claim_label`） | 備考 |
|---|---|---|---|---|
| MP-SENet | `fixed_pair.mp_senet` | 固定ペア | 学習入口は `mp_senet.vb28.ondemand.official`（`official_reproduction`。train 11,572、valid = test 824、`selection_uses_test = true`）と `mp_senet.vb28.ondemand.safe`（`leakage_safe_comparison`。10,700 / 872 / test 824）。履歴の `mp_senet.vb28.official` と `mp_senet.vb28.safe` は 16 kHz 派生素材を指す | 学習入口の case は 48 kHz コーパスを指す。16 kHz PCM_16 は学習時の soxr HQ。公式の事前変換器は UNKNOWN。公式リストは `configs/data/lists/mp_senet/` |
| MossFormerGAN | `fixed_pair.mossformergan_se_16k` | 固定ペア | 学習入口は `mossformergan.vb28.ondemand.official` と `mossformergan.vb28.ondemand.safe`。件数は MP-SENet と同じ。履歴の `mossformergan.vb28.official` と `mossformergan.vb28.safe` は派生素材を指す | scratch 開始（D-04）は学習設定側。case は MP-SENet オンデマンドと共有し、`cases.protocol_id` は `mp_senet.fixed_pair.ondemand` |
| DeepFilterNet3 | `online.dfn3` | オンライン混合 | `dfn3.dns4_ptdb.official`（`declared_status = blocked`。`blocking_json`: DNS4、PTDB、`dataset.cfg`、RIR 集合、`sampling_factor`）。代替素材: `dfn3.substitute.<素材>`（`protocol_conformant_substitute_materials`） | 公式素材を入手するまで学習に使えない（D-06） |
| SEMamba++ | `online.semambapp_gsr`（HEAD の `config.yaml`。起動不能の最小補修は `deviation_class = minimal_launch_repair` の別 Protocol） | オンライン混合 | 公式素材（VCTK 0.92 と Arni RIR など）: 入手まで `blocked` または `unknown`。代替素材（手元の VCTK、WHAM48 と DNS2020 の雑音、DNS5 と OpenSLR の RIR）: `protocol_conformant_substitute_materials` | 手元の VCTK は 108 話者・44,085 本（確認: §2.3）。公式の 0.92（p280、p315 除外）と同一かは UNKNOWN |
| TF-GridNet | `pregenerated.ms_snsd_tfgridnet` | 事前生成の条件を先に確定し、学習は `delivery: on_demand` で元素材から再構築する。保存済み WAV の `tf_gridnet.ms_snsd.smoke@1` は旧経路 | `tf_gridnet.ms_snsd.espnet_recipe`（公式規模。train 7,345 / valid 1,835 / test 305）と `tf_gridnet.ms_snsd.ondemand`（`official_recipe_scaled_down`） | §9 |
| TF-Locoformer | `pregenerated.dns_tflocoformer` | 事前生成 | `tf_locoformer.dns.s1`〜`s4`（`official_recipe_scaled_down`。S4 は生成規模が公式だが、検証集合 `cv_synthetic_small` が公式と同じと確認できる（UNKNOWN の解消）までこのラベルのまま） | §9.6 |

16 kHz 化は P2 で実装した。公式の事前変換器は不明なままである。派生素材は soxr 1.1.0 の HQ、soundfile 0.14.0 の PCM_16 で、`data/derived/VoiceBank-DEMAND/soxr_hq_16000/` に置く。Q6 のとおり `official_reproduction` を名乗り、`deviations_json` に変換器が不明であることを書く。同じ 16 kHz ペアのレシピは 1 つの `case_id` になるため、MP-SENet と MossFormerGAN は case 行を共有する。`cases.protocol_id` は `mp_senet.fixed_pair` を指す。学習時の正規化・窓・順序は Dataset の protocol を読む。DuckDB は COPY 中に asset の外部キーを検査するので、親の corpora 素材を派生素材より先に挿入する。

---

## 12. この設計の未確定事項

実装前に決める事項は IMPLEMENTATION_PLAN の「着手前の確認」に集約した。この文書に関係するもの:

| ID | 内容 |
|---|---|
| U-DB-1 | `data/generated/` の実体の置き場所と容量の割り当て（TF-Locoformer S4 で約 1.04 TB） |
| U-DB-2 | `cv_synthetic_small` の大きさと規則（D-08 で未確定のまま維持） |
| U-DB-3 | MossFormerGAN の VoiceBank 16 kHz 版の変換法は公式 UNKNOWN のまま。`claim_label` は `official_reproduction` とし、変換器不明を `deviations_json` に書いた（P2） |
| U-DB-4 | DFN3 の公式素材（`dataset.cfg`、DNS4、PTDB、RIR 集合）。入手できるまで `blocked` |
| U-DB-5 | SEMamba++ の公式素材（VCTK 0.92、Arni RIR）と、VCTK-GSR の分割規則 |
| U-DB-6 | `resource.duckdb` に `eval` スキーマが無い現状（§2.3）。leakage 監査が `UNKNOWN` になる |
| U-DB-7 | TF-Locoformer の `noise` ストリームを学習が使うか |
| U-DB-8 | ESPnet の `speech_mix_shape` の発話順（分割・シャードの割り当てに影響。統計的には等価な範囲） |
| U-DB-9 | `data/generated`、`data/db/reproduction.duckdb`、`data/staging/reproduction/`、`data/db/snapshots/` を AGENTS.md の保護一覧へ加える時期（フェーズ 1 の完了時を提案） |
