# P10-C Preflight — Agent C（DB・Dataset・Data Pipeline）

調査日: 2026-11-07。Phase 1 の独立調査。証拠はコードと、読み取り専用で開いた実データだけである。過去の監査文書と `.agents/DECISIONS.md` は結論の根拠にしていない。`src/`、YAML、テスト、Dataset、Snapshot、DuckDB、checkpoint、実験結果は変更していない。DuckDB は `read_only=True` だけで開き、書き込みと `ATTACH` はしていない。

## 読んだ DB とクエリ

作業ディレクトリ `/net/spring/work/seki/exp/audio-db`。Python は `.venvs/se`（DuckDB 2.0.0）。

| ファイル | 結果 |
|---|---|
| `data/db/reproduction.duckdb` | 無い。`data/staging/reproduction` も無い |
| `data/db/reproduction-p8.duckdb` | 無い。DeepFilterNet3 のスナップショットも無い |
| `data/db/reproduction-p9.duckdb` | 12MB。`repro` スキーマを read-only で読んだ |
| `data/db/voicebank.duckdb` | 11MB。`eval` スキーマを read-only で読んだ |
| `data/db/resource.duckdb` | 195MB。スキーマ名、`eval.sets` の名前、`voicebank_demand_test` の件数と話者、`corpora.audio_files` の件数だけ読んだ |
| `data/db/snapshots/` | ディレクトリ 6 件。すべて `load_snapshot` で論理ハッシュを再計算した |
| `data/generated`、`data/derived` | ディレクトリ自体が無い |

`reproduction-p9.duckdb` で実行した読み取り:

- `information_schema.schemata` / `information_schema.tables`（`repro`）
- `repro.datasets` の `name, version, dataset_id, protocol_id, profile, material_origin, claim_label, declared_status, selection_uses_test, blocking_json`
- `repro.protocols` の `protocol_id, name, version, kind, algorithm, deviation_class, parent_protocol_id`
- 各 `repro.*` の `count(*)`
- `repro.runs`（0 件）、`repro.audits` の種別と status
- `repro.dataset_bindings` と `repro.material_sets`
- 分割をまたぐ `asset_id`、`sha256`、`speaker_id` の交差（material member）
- speech スロットだけの `speaker_id` 集合

続けて `read_graph`（内部も `read_only=True`）から `protocol_id_for` と `dataset_id_for` を再計算し、`db.snapshot._parts` の論理ハッシュをスナップショットの `load_snapshot` と比較した。`audits.status` は証拠に使っていない。

`voicebank.duckdb`:

- `eval.sets`、`eval.set_items` の件数
- `eval.cases.speaker_id` の分割間交差、`case_id` の交差
- レシピ `sources.*.path` と `sha256` の有無（全 11,572 件）

`resource.duckdb`:

- `eval.sets.name`
- `voicebank_demand_test` は 824 件、話者は `p232` と `p257`
- `corpora.audio_files` は 1,375,481 件

スナップショットは `dataset.json`、`manifest.json`、`cases.npz`、`case_set_items.npz`、`assets.npz`、`material_set_members.npz`、`docs.jsonl` を読んだ。音声ファイルは全 asset の存在確認と、各スナップショット 8 ファイルの SHA-256 再計算だけである。全音声バイトの再ハッシュはしていない（NOT_TESTED）。

## 件数表

スナップショットのディレクトリ名は `dataset_id`。`manifest.json` の `snapshot_hash` は別の値で、6 件とも `load_snapshot` の再計算と一致した。`dataset_id_for` の再計算も 6 件とも一致した。

| Dataset | claim | status | uses_test | protocol_id（dataset） | snapshot_hash | train / valid / test |
|---|---|---|---|---|---|---|
| `mp_senet.vb28.ondemand.official@1` | official_reproduction | ready | true | `69dcd7c2…269a7` | `265f256b…7d33d5` | 11,572 / 824 / 824 |
| `mossformergan.vb28.ondemand.official@1` | official_reproduction | ready | true | `f1b67cc1…c06d5` | `ba0c77c9…dff555` | 11,572 / 824 / 824 |
| `tf_gridnet.ms_snsd.espnet_recipe@1` | espnet_public_se_recipe_reproduction | ready | false | `9f5ffa7c…bf7e97` | `a69a477a…4f0757` | 7,345 / 1,835 / 305 |
| `tf_locoformer.dns.accept_recipe@1` | espnet_public_se_recipe_reproduction | ready | false | `cadda632…630116` | `3343a55e…9926f5` | 324,000 / 36,000 / 150 |
| `semambapp.gsr.substitute@1` | protocol_conformant_substitute_materials | ready | false | `ee7a448b…58a970` | `1b9be5fe…3f66d37` | speech 16 / 4 / 4（case は valid 4、test 4。train.cases は無い） |
| `semambapp.gsr.smoke@1` | protocol_conformant_substitute_materials | ready | false | 上と同じ | `dd57fd63…fb2802f` | speech 8 / 2 / 2（case は valid 2、test 2） |

`reproduction-p9.duckdb` の 3 Dataset。ID 再計算は一致。ready 2 件は DB から組んだ論理ハッシュとスナップショットが一致（case、asset、item、member、doc の件数も一致）。

| Dataset | claim | status | bindings | protocol |
|---|---|---|---|---|
| `semambapp.gsr.official@1` | official_reproduction | blocked | 0 | `semambapp.gsr`（`official_effective`、親なし）`02ca9647…975b7` |
| `semambapp.gsr.substitute@1` | protocol_conformant_substitute_materials | ready | 11 | `semambapp.gsr.audio-db`（`algorithm_change`、親は上） |
| `semambapp.gsr.smoke@1` | protocol_conformant_substitute_materials | ready | 11 | substitute と同じ |

公式 SEMamba の `blocking_json` は VCTK 0.92 でないこと、Arni が無いこと、公式 train manifest が空に近いこと、公式 validation pair が無いことを記録している。スナップショットは無い。`repro.runs` は 0 件なので、DB 側に run の `snapshot_hash` 列は無い。

asset 完整性（スナップショット）:

| Dataset | assets | integrity | sha256 欠落 | ファイル存在 | 再ハッシュ |
|---|---|---|---|---|---|
| MP-SENet official | 24,792 | sha256 のみ | 0 | 24,792 / 24,792 | 8/8 一致。全件 NOT_TESTED |
| MossFormerGAN official | 24,792 | sha256 のみ | 0 | 24,792 / 24,792 | 8/8 一致。全件 NOT_TESTED |
| TF-GridNet recipe | 24,354 | sha256 のみ | 0 | 24,354 / 24,354 | 8/8 一致。全件 NOT_TESTED |
| TF-Locoformer accept | 130,950 | sha256 のみ | 0 | 130,950 / 130,950 | 8/8 一致。全件 NOT_TESTED |
| SEMamba substitute | 60 | sha256 のみ | 0 | 60 / 60 | 8/8 一致。全件 NOT_TESTED |
| SEMamba smoke | 26 | sha256 のみ | 0 | 26 / 26 | 8/8 一致。全件 NOT_TESTED |

VoiceBank のレシピ文書 12,396 件は、`recipe_case_id` が `case_id` と一致し、文書内 SHA-256 が asset 表と一致した（MP-SENet と MossFormerGAN の両方）。これはファイルを読み直した結果ではない。再サンプル長 `(source_frames + 1) // 3` は 12,396 件とも case の `frames` と一致した。

TF-GridNet と TF-Locoformer の realization は、`digest(doc_json) == case_id` が全件一致（9,485 と 360,150）。

学習 YAML が指す Dataset:

| YAML の dataset | スナップショット |
|---|---|
| `mp_senet.vb28.ondemand.official@1` | ある |
| `mossformergan.vb28.ondemand.official@1` | ある |
| `tf_gridnet.ms_snsd.espnet_recipe@1` | ある |
| `tf_locoformer.dns.accept_recipe@1` | ある |
| `mp_senet.vb28.ondemand.safe@1` | 無い |
| `mossformergan.vb28.ondemand.safe@1` | 無い |
| `tf_gridnet.ms_snsd.smoke@1` | 無い |
| `tf_gridnet.ms_snsd.ondemand@1` | 無い |
| `tf_locoformer.dns.smoke@1` | 無い |
| `configs/training/deepfilternet3.yaml` と `semambapp.yaml` | reproduction Dataset を指さない。`data/db/voicebank.duckdb` の `voicebank_28_train` / `voicebank_28_valid` |

`configs/training/` に `dfn3.online.*` と `semambapp.gsr.*` を指す YAML は無い。スナップショット名は 6 件とも一意で、`find_snapshot_dir` が同名を複数拾う状態ではない。

`voicebank.duckdb`（公式 Protocol ではない）:

| 集合 | 件数 | 話者 |
|---|---|---|
| `voicebank_28_train` | 10,700 | 26。`p286` と `p287` は 0 件 |
| `voicebank_28_valid` | 872 | `p286`, `p287` |
| 公式 test 集合 | この DB には無い | — |

`resource.duckdb` の `voicebank_demand_test` は 824 件、話者 `p232` と `p257`。`voicebank_28_train` とも `voicebank_28_valid` とも話者交差は 0。

## リーク検査結果

Case ID が重ならないことと、素材が独立であることは分けて書く。

### MP-SENet / MossFormerGAN 公式（確認済み）

両方とも `selection_uses_test = true`。valid の case 集合と test の case 集合は一致する。

| 比較 | case ID | 話者 | 発話キー | clean asset | clean SHA-256 |
|---|---|---|---|---|---|
| train ∩ valid | 0 | 0 | 0 | 0 | 0 |
| train ∩ test | 0 | 0 | 0 | 0 | 0 |
| valid ∩ test | 824 | 2 | 824 | 824 | 824 |

valid と test の話者は `p232` と `p257` だけである。train は 28 話者 11,572 発話で、この 2 話者は含まない。`p286` と `p287` は train に含まれる。case 集合 ID 自体は valid と test で違う。これは `split_role` が case set ID に入るためで、中身の case ID は 824 件すべて同じである。

MP-SENet と MossFormerGAN の train / valid / test の case ID 集合は互いに等しい。case 行の `protocol_id` は両方のスナップショットで MP-SENet の `69dcd7c2…` だけである。MossFormerGAN の Dataset protocol は `f1b67cc1…`。asset 表も同一。正規化は Dataset 側で違い、MP-SENet は `noisy_rms_unit_v1`、MossFormerGAN は `independent_audio_norm_v1`。

雑音は別 asset になっていない（`noise_asset_id` は空、レシピの `operations` は空）。DEMAND のシーンが train と test で同じかは、このスナップショットからは UNKNOWN。

`voicebank.duckdb` の 10,700 / 872 は、この公式分割ではない。公式 test を validation に使っていない。

### TF-Locoformer `accept_recipe` の 36,000 件（確認済み。`cv_synthetic_small` ではない）

件数は train 324,000、valid 36,000、test 150。valid は小さい合成 CV 集合ではない。

保存されている train+valid の `utterance_key` を辞書順に並べると、先頭 `360000 * 9 // 10 = 324,000` が train、残りが valid と一致した。train の最大キーは valid の最小キーより小さい。case ID、`utterance_key`、`gen_index` の train∩valid は 0。duration で落ちた発話が登録前に存在したかは、スナップショットに無いので UNKNOWN。

素材:

| 比較 | case ID | 音声ファイル名 | 雑音ファイル名 | 話者列 |
|---|---|---|---|---|
| train ∩ valid | 0 | 2 | 43,823 | 列は全件 NULL |
| train ∩ test | 0 | 0（test は piece 名を持たない） | 0 | NULL |
| valid ∩ test | 0 | 0 | 0 | NULL |

音声ファイル数は train 47,354、valid 5,174。共有は `book_03992_chp_0020_reader_10635_21.wav` と `…_23.wav` の 2 件。ファイル名の `reader_(\d+)` は train+valid の音声 piece 360,000 件すべてにあり、読者トークンは train 1,653、valid 172、交差は `10635` の 1 件。これは `speaker_id` 列ではない。列は空なので、公式話者 ID としての一致は UNKNOWN。ファイル名トークンの交差は確認済み。

雑音ファイル名は train 54,691、valid 43,835、交差 43,823。asset の basename 衝突は 0 なので、同じ名前は同じ asset である。valid の雑音は train の雑音とほぼ同じ集合である。

test 150 件は `recorded_pair` / `recipe_v1` で、protocol は Dataset のものではなく `94f211d2…`。パスは `DNS2020/datasets/test_set/synthetic/no_reverb/`。clean の記録 SHA-256 150 件と、clean pool の記録 SHA-256 65,348 件の交差は 0。test の話者列は NULL で、ファイル名は `clean_fileid_N` なので話者は UNKNOWN。

renderer は `dns_pcm16_accept_v2`、retry は `deterministic_case_stream_v1`。train と valid の realization 360,000 件はこの retry と `pregenerated.dns_tflocoformer`。非 16 kHz の piece は 44,100 Hz が 1,962、48,000 Hz が 1,637、96,000 Hz が 110。学習の DNS reader は accept のときだけリサンプルする。

36,000 件をさらに小さくする場合の評価（方式は確定しない）: 今の valid は uttid 順の末尾であり、独立した話者ホールドアウトではない。短く切ると、境界の reader `10635` と、train とほぼ同じ雑音集合が残りうる。選択バイアスは、uttid の辞書順が音声ファイル名に偏っていることから来る。case 集合としては train と交わらない。

### TF-GridNet `espnet_recipe`（確認済み）

train 7,345、valid 1,835、test 305。case ID の交差は 0。`speaker_id` は 9,485 件すべて NULL。話者レベルの独立性は UNKNOWN。

保存済み発話だけで単純な `8/10` を切ると train 側は一致しない（切り位置 7,344）。同じ文書集合に `split_train_valid` を掛けると一致した。duration の脱落は 0。bash の `9180 * 8 / 10 = 7344` から、同一クリップを割らないために 1 件動かして train 7,345 になっている。valid は同じ生成プールの尾である。

音声ファイル名は train 4,918、valid 1,338、test 60。train∩valid は 234。valid の 1,338 名はすべて `clean_train` の basename に解決し、`clean_test` には 0。test 音声の basename は train と 0。clean ファイルの記録 SHA-256 は train 23,075 と test 974 で交差 0。

雑音ファイル名は train 128、valid 123、test 32。train∩valid は 123（valid の雑音名はすべて train にも出る）。train∩test の名前は 24、valid∩test は 22。`noise_train` と `noise_test` で basename が両方にあるものは 42 件あり、SHA-256 が同じものは 0。ディレクトリが違う別ファイルである。material member の asset ID 交差は train/valid/test とも 0。valid 用の material binding は無く、描画は valid を train ソース表で引く。未解決の valid 音声名は 0 だった。

`utterance_key` は train∩test が 285、valid∩test が 20。同じキー 305 件の realization を比べると、音声名・雑音名・長さ・build は 305 件とも違う。例: `noisy10_SNRdb_0.0_clnsp10` の train は `p298_097.wav` などと `SqueakyChair_7.wav`、test は `clnsp481.wav` と `AirConditioner_7.wav`。キーの一致は別 build の名前衝突で、同一波形ではない。

### SEMamba++（確認済み。公式再現ではない）

ready な 2 件の claim は `protocol_conformant_substitute_materials`。protocol は公式 `semambapp.gsr` ではなく、その子 `semambapp.gsr.audio-db`（`algorithm_change`）。smoke と substitute の `rule_json` は一致し、Dataset ID は binding が違うので別である。

speech の `speaker_id`（DB の asset 列）:

| Dataset | train | valid | test | 交差 |
|---|---|---|---|---|
| substitute | p225–p232 の 8 | p233, p234, p236, p237 | p238–p241 | 0 |
| smoke | p225–p228 の 4 | p233, p234 | p238, p239 | 0 |

asset ID と記録 SHA-256 の分割交差も 0。smoke の 26 asset は substitute の 60 asset の部分集合である。公式 Dataset は blocked で binding もスナップショットも無い。学習コードは `claim_label` を `official_reproduction` に書き換えない。この DB 上で、代替素材を公式再現と呼ぶ claim は無い。

雑音の `speaker_id` はファイル名 stem、RIR は部屋パスである。上の話者結論は speech スロットだけに限る。

### DeepFilterNet3（公式 blocked は既知。代替の中身は UNKNOWN）

`reproduction-p8.duckdb` と `dfn3` スナップショットは無い。登録された substitute の claim、分割、素材交差は、この作業ディレクトリの実データからは UNKNOWN。リポジトリの `configs/data/datasets/dfn3.online.yaml` は登録メモで、学習 YAML はそれを指さない。`configs/training/deepfilternet3.yaml` は `voicebank.duckdb` の 10,700 / 872 で、コメントどおりオンライン混合ではない。これを公式オンライン再現とは呼ばない。

## コード上の経路（実データと突き合わせたもの）

`resource.duckdb` を学習が開くか:

- 再現パイプラインの `open_training_data` はスナップショットだけを開き、DuckDB を開かない（`src/reproduction/pipeline/training_data.py` の `open_training_data`）。
- `load_recipe_dataset` はファイル名が `resource.duckdb` だと学習セットとして拒否する（`src/training/data/dataset.py`）。
- レシピ Dataset の `waveform_lengths` は、長さ取得のため既定で `DUCKDB_MAIN_PATH`（`resource.duckdb`）を `duckdb.connect(..., read_only=True)` で開く。再現スナップショットの学習経路では呼ばれない。書き込み接続は学習コードから見つかっていない。

事前生成 WAV の拒否:

- `open_chunk_datasets` は `rule_json.delivery != "on_demand"` で `ReproductionError`。
- `require_on_demand` は同じ条件で `TrainingError`。
- `/tmp` に作った `delivery: stored_wav` の最小スナップショットで、両方の拒否文を確認した。既存スナップショットは変更していない。歴史上の `tf_gridnet.ms_snsd.smoke@1` ファイルはこのディレクトリに無いので、その実ファイルでの拒否は NOT_TESTED。失敗は「スナップショットが無い」で閉じており、保存 WAV を黙って読む経路には入らない。

波形キャッシュ:

- `data/generated` は無い。今回の asset はすべて `corpora/` である。
- 学習時の DNS `render_realization` は `_render_pieces` を呼び、計画用の `load_16k` キャッシュは渡さない。MS-SNSD の on-demand モジュールは波形を保持しない、と実装されている。ディスク上の波形キャッシュは残っていない。

壊れたスナップショット:

- 6 件の `load_snapshot` は成功。DNS の再計算は 68.6 秒。
- `/tmp` の manifest で `snapshot_hash` を 64 個の `0` にすると `snapshot_hash does not match the logical rows` で拒否した。

オンデマンドの決定性:

- MP-SENet の train 先頭 case `4b2fc18f…c0d0fc` を、同じ resample 規則で 2 回描画した。16 kHz、長さ 36,480、波形は一致。
- 同じ valid case を MP-SENet の正規化と MossFormerGAN の正規化で描くと波形は一致しない。case 行の protocol は MP-SENet のままで、描画結果を分けているのは呼び出し側の `normalize` である。

別 protocol の文書:

- 固定ペアの `render_sample` は `cases.protocol_id` を見ない。MossFormerGAN の case 行は MP-SENet protocol を指す。文書は共有レシピで `operations` は空。学習は Dataset の `rule_json` を `validate_rule` してから正規化する。
- DNS の train/valid 360,000 件の protocol は Dataset protocol と一致する。test 150 件は別 protocol のレシピである。学習の DNS reader は train と valid だけを開く。test 文書の retry は空で、Dataset の `deterministic_case_stream_v1` とは一致しないので、その case を描こうとすると retry 検査で落ちる。algorithm フィールド自体は描画前に見ていない。今の train/valid では全件 `pregenerated.dns_tflocoformer` だった。

Dataset 再構築:

- SEMamba の ready 2 件は、`reproduction-p9.duckdb` の行から組んだ論理スナップショットとファイルが一致した。
- MP-SENet、MossFormerGAN、TF-GridNet、TF-Locoformer は、対応する `reproduction.duckdb` と `data/staging/reproduction` が無いので再構築比較はできない。スナップショット単体の論理ハッシュは一致している。

## Issue

### P10P-C-001

- 発見者: Agent C
- 対象: `data/db/reproduction.duckdb` と `data/staging/reproduction`
- 種別: レジストリ欠落
- 深刻度: 高
- 確度: 高
- 場所: `data/db/`。コードの既定パスは `src/reproduction/paths.py` の `DEFAULT_DB`
- 根拠: 2026-11-07 に `data/db/` を一覧し、`reproduction.duckdb` は無かった。`data/staging/` にあるのは `reproduction-p9` と `voicebank-current` だけである。一方 `data/db/snapshots/` には MP-SENet、MossFormerGAN、TF-GridNet、TF-Locoformer、SEMamba substitute、SEMamba smoke がある。この 4 本命について、DB 行とスナップショットの突合、protocol 式の再計算（親 protocol と `deviation_class` がスナップショットに無い）、staging からの再構築は実行できない。スナップショット自身の `load_snapshot` は 6 件とも成功している。
- 再現手順: `test -f data/db/reproduction.duckdb` が偽。`test -d data/staging/reproduction` が偽。`data/db/snapshots/*/dataset.json` の `name` は上の表のとおり存在する。
- 影響: 学習の再現経路は DuckDB を開かずスナップショットを読むので、今ある 6 ディレクトリの論理内容は読める。登録 DB と同一であることは未証明である。protocol ID が式と一致するかは、SEMamba（p9 で一致）以外は未検証である。Dataset ID はスナップショット内のフィールドから再計算して 6 件とも一致した。
- 対応方針: 既存の `reproduction.duckdb` を戻してから、ready Dataset ごとに論理ハッシュをスナップショットと比較する。この監査では DB を作っていない。
- P10-C影響: BLOCKER
- 検証状態: 確認済み（ファイル不在と、スナップショット側の自己一致）

### P10P-C-002

- 発見者: Agent C
- 対象: DeepFilterNet3 のオンライン Dataset
- 種別: データ欠落
- 深刻度: 高
- 確度: 高
- 場所: `data/db/reproduction-p8.duckdb`（不在）、`data/db/snapshots/`（`dfn3` の `dataset.json` が無い）
- 根拠: 公式 Dataset が blocked であることは既知であり、それ自体を新規の不具合にはしない。この作業ディレクトリには公式・代替・smoke のどれのスナップショットも DB も無い。したがって「代替素材なのに `official_reproduction` になっている」登録は、実データでは確認も否定もできない。比較として、存在する SEMamba の ready Dataset は claim が `protocol_conformant_substitute_materials`、protocol は `algorithm_change` の子であり、公式 protocol のスナップショットは無い。学習コードは claim を公式へ昇格しない。`configs/training/deepfilternet3.yaml` は `voicebank.duckdb` を指し、オンライン混合ではない。
- 再現手順: `test -e data/db/reproduction-p8.duckdb`。`data/db/snapshots/*/dataset.json` の `name` に `dfn3` が無いことを確認する。
- 影響: DFN3 のオンライン再現は、このディレクトリの実データからは開始できない。VoiceBank 28 話者のホールドアウト学習を、公式オンラインデータの再現として扱うことはできない。
- 対応方針: `reproduction-p8.duckdb` とスナップショットを戻し、claim と素材分割を read-only で再確認する。blocked の公式 Dataset を ready にしない。
- P10-C影響: BLOCKER
- 検証状態: 確認済み（ファイル不在）。登録内容そのものは UNKNOWN

## 反証して問題にならなかった仮説

- 公式 MP-SENet / MossFormerGAN の validation が test と別物である、という見方。実データでは valid と test は case 824 件、話者 `p232`/`p257`、clean SHA-256 まで同一で、`selection_uses_test` は true。safe 用スナップショットは無く、公式と取り違えて読むファイルは無かった。
- `voicebank.duckdb` の 872 件が公式 test である、という見方。valid は `p286`/`p287` の 872 件で、公式 test の `p232`/`p257` とは話者交差 0。
- TF-Locoformer の 36,000 件が `cv_synthetic_small` という小さい独立集合である、という見方。件数と uttid の 9/10 分割は上のとおりで、名前も `tf_locoformer.dns.accept_recipe@1` である。
- MossFormerGAN が MP-SENet の正規化を掛けてしまう、という見方。case 行の protocol は MP-SENet だが、同じ case を二つの `normalize` で描くと波形は違った。学習は Dataset の rule を渡す。
- 別 protocol の DNS test レシピを train として描いてしまう、という見方。test の protocol は別だが、train/valid の 360,000 件は Dataset protocol と retry が一致する。test は学習の DNS reader が開くスロットに入っていない。
- 事前生成 WAV の学習が拒否されず残っている、という見方。`/tmp` の `stored_wav` スナップショットは `open_chunk_datasets` と `require_on_demand` の両方で拒否された。`data/generated` は無い。
- 波形キャッシュが学習入力として残っている、という見方。該当ディレクトリは無く、学習時の描画はキャッシュを渡さない。
- `resource.duckdb` を学習が書き込みで開く、という見方。再現経路は開かない。レシピ経路の長さ取得は `read_only=True`。学習セットとしてのファイル名は拒否される。
- SEMamba の substitute が `official_reproduction` になっている、という見方。claim、protocol の `deviation_class`、binding の origin は substitute のまま。公式は blocked でスナップショットが無い。
- MS-SNSD の `utterance_key` 交差が同一波形のリークである、という見方。交差 305 キーのソース列はすべて違っていた。case ID 交差は 0。train と test の clean/noise 記録 SHA-256 交差も 0。
- 同名の `noise_*.wav` が train と test で同じバイトである、という見方。basename が両方にある 42 件の SHA-256 はすべて違った。
- スナップショットの論理ハッシュが manifest とずれている、という見方。6 件とも再計算が一致し、改ざんした `/tmp` manifest は拒否された。
- case ID の重複が分割内にある、という見方。6 スナップショットとも分割内の重複は 0。MP-SENet と MossFormerGAN の間で case ID を共有するのは、同じ 48 kHz ペアを二つの Dataset が指しているためで、train と test の共有ではない。

## 未監査

- 音声ファイル全件の SHA-256 再計算。存在確認は全 asset。内容の再計算は各スナップショット 8 ファイルだけ。
- `reproduction.duckdb` が無い Dataset の protocol 式（`protocol_id_for` に必要な親と `deviation_class` がスナップショットに無い）。
- DeepFilterNet3 の登録行、素材分割、claim。DB が無い。
- safe / smoke / 小規模 on-demand のスナップショット中身。ファイルが無いので分割は UNKNOWN。起動は「0 件」で失敗する実装である。
- DNS と MS-SNSD の公式話者 ID。case の `speaker_id` は空。DNS はファイル名トークンだけ確認した。
- VoiceBank の noisy WAV に焼き込まれた DEMAND シーンが、train と公式 test で同じ録音かどうか。
- オンライン混合（SEMamba、およびデータの無い DFN3）の描画を 2 回実行する決定性テスト。固定ペアの soxr 再サンプルだけ実行した。
- `reproduction-p9` の `audits.status` を証拠にした検証。交差と ID は SQL とハッシュ再計算で見ている。
- 本格学習、GPU、既存プロセスの停止。行っていない。
