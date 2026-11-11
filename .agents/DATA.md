# データ・DB・再現性

ここには、コードを読まないと分からない制約だけを書く。状態や履歴は書かない。

| 知りたいこと | 文書 |
|---|---|
| 評価用 DB の操作、staging の形 | `docs/db/SETUP.md` |
| レシピ JSON | `docs/db/RECIPE.md` |
| 再現用 DB の表、ID、claim、スナップショット | `docs/db/REPRODUCTION_DB_DESIGN.md` |
| どのファイルに何が入っているか、件数、ハッシュ、構築の記録 | `docs/db/INVENTORY.md` |
| コーパスの索引と内訳 | `docs/db/CORPORA.md`、`docs/db/DATASETS.md`、`docs/db/tables/AUDIO_FILES.md` |
| 評価の段階、WER の完全性、結果の形式 | `docs/evaluation/PIPELINE.md` |
| 学習の式と checkpoint | `.agents/ARCHITECTURE.md` |

## DB の構成

DuckDB は `data/db/resource.duckdb`（`src/data/constants.py`）。

- `corpora.audio_files`: `make db-ingest` が作る。`path` が主キーで、値は `data/corpora` からの相対パス（コーパス名を含む）。採番した ID は無い。
- `eval.cases` / `eval.sets` / `eval.set_items` / `eval.runs` / `eval.results`: 新しいスキーマを `make db-init` が空の DB に作る。`eval.cases` の生成条件は `recipe_json` だけに置き、SNR や切り出し位置の列は持たない。`corpora.audio_files` への外部キーは無く、レシピ内の相対パスと metadata の `corpus_path` で対応する。`case_id` は正規化レシピの SHA-256 で、採番や rowid ではない。
- `data/db/resource.duckdb` はこのレシピスキーマである。`recipe_json` の無い `eval.cases` を見つけると、`make db-init` は何も変更せず拒否する。バックアップの場所とハッシュは `docs/db/INVENTORY.md`。
- DB ファイル名の stem を `eval` にしない。DuckDB のカタログ名と衝突して `eval.cases` が曖昧になる。

## `make db-init` と `make db-ingest`

| | `make db-init` | `make db-ingest` |
|---|---|---|
| 実行 | `make db-init` | `make db-ingest`（いずれも `PYTHON` で環境を指定可能） |
| 役割 | `eval` スキーマを作り、`configs/db/init.yaml` の Manifest・コーパス選択・注釈を評価セットとして登録する | `data/corpora` を全走査し、`corpora.audio_files` を入れる |
| `corpora.audio_files` | 変更しない。前後の指紋（件数・path の最小/最大・ハッシュ）が違えばエラー | `DROP TABLE IF EXISTS` で消してから作り直す |
| 再実行 | 冪等。同じ内容は増えない。同じ ID で内容・順序が違う再登録はエラー（その set は一部だけ残らない） | 毎回、全削除して全投入 |
| 入力が無いとき | `required: false` は理由を出してスキップ。`required: true` は失敗 | — |

`make db-ingest` の危険:

- 実行した時点で既存の `corpora.audio_files` が消える。ユーザーが明示したときだけ、`data/db/resource.duckdb` を退避してから実行する。
- `eval` スキーマは消えない。レシピスキーマでは、相対パスが変わると別の `case_id` になり、ファイル内容が変わると SHA-256 の不一致で再生を拒否する。`recipe_json` の無い `eval.cases` への `db-init` は変更せず拒否する。
- `db-init` から `db-ingest` を呼ばない。

## 評価セット

- 評価の入力は名前付きセット（`eval.sets` の `name`）だけ。名前は `[A-Za-z0-9._-]{1,128}`。順序（`position`）は登録時のまま。同名で内容、順序、説明が違う登録は、上書きせずエラーにする。
- Manifest を評価設定の `dataset.source` に使えない。読むのは登録時（`make db-init` の `manifests:`、`EvaluationDB.import_manifest`）だけで、登録後は Manifest ファイルが無くても読める（音声は要る）。
- 音声パスは DB ファイルからの相対パスで、レシピの `sources.*.path` に保存する。絶対パスは保存しない。`../corpora/...` のように親へ戻る相対パスは、コーパスが `data/corpora` にあるときの形である。プロジェクトと `data/corpora` の外にある音声は登録できない。
- 生成する case の条件はレシピ JSON だけが正本である。`additive_noise` の SNR は区間全体の平均二乗電力の比（旧 `rms_v1` と同じ定義）で、`noise_start` は出力サンプリングレートへ変換し、短い雑音を繰り返したあとのサンプル位置である。seed はレシピに残るが、再生時に切り出し位置を seed から計算し直さない。`peak_normalize` はピークが `limit` を超えるときだけ共通ゲインで減衰する。`rir` バージョン 1 は因果的に畳み込み、入力長へ切り詰める（`trim_to_input`）。残響の尾は捨て、サンプル 0 は dry 入力のサンプル 0 と揃う。
- 指標が使う参照は `output.reference` で、`dry_clean`、`reverberant_clean`、`none` のいずれかである。RIR が無いときの合成は `dry_clean`。収録済み noisy は `none` で、PESQ / STOI / SI-SDR は `na`（`no_clean_reference`）。CHiME3 の real / simu は noisy として `none`。`tr05_org` は clean ファイルそのものを `dry_clean` として評価する。これは real 録音の時間整合した参照ではない。
- 同じレシピは同じ `case_id` になる。同じ `case_id` で transcript や metadata が違う再登録はエラーである。名前付きセットでは `mixing.enabled: false` にする。
- 名前付きセットを使わない `dataset.source: duckdb` ＋ `mixing.enabled: true` の経路も残っているが、比較実験には使わない。この経路はレシピを保存しない。
- `create_set_from_corpus(query=...)` は、`corpora.audio_files` を読む単一の SELECT だけを受け付ける。`protocol: rms_v1` は登録時に `additive_noise` バージョン 1 のレシピへコンパイルする。ハードクリップはレシピ操作にしない。
- 入力ファイルの SHA-256 は、その case が参照するファイルだけをレシピに保存する。再生時に食い違えばエラーにする。コーパス全件はハッシュしない。

## ステージングと再構築

- `data/staging/` が再構築元である。`metadata.json`、`corpora/audio_files.parquet`、`evaluation/cases.parquet`、`evaluation/sets.parquet`、`evaluation/set_items.parquet`、`results/runs.parquet`、`results/results.parquet`。Git 管理外。DB へ登録したあとも消さない。バックアップは DB ファイルと一緒に取る。
- 登録は `src.data.staging.register_staging`。検証してから 1 トランザクションで書く。同じ内容の再登録は増えない。同じ ID で内容が違うとエラーになり、そのバッチは残らない。不完全なファイルは DB を作る前に拒否する。
- 評価ランの一次記録は従来どおり `data/results/se-evaluation/<run_id>/` の `config.json` と `results.jsonl` である。DuckDB への登録は `register_run` で、そのあと行う。`run.json` が `completed` でない run（`incomplete`、`validation_failed`、`failed`、`running`）は登録しない。実行中の `results.jsonl` は最大 60 秒ぶん `records/` より古く、kill された run では古いまま残る。既存ランは自動では登録しない。
- `src.data.migrate_legacy` は、以前の `eval.cases` を一度だけレシピへ写す。実行時の互換読み取りではない。出力先が既にあるときは上書きしない。本番は切り替え済みなので、このコマンドを再実行する必要はない。旧 `case_id` は metadata の `legacy_case_id` に残り、新しい `case_id` はレシピハッシュである。

## 学習に使うセット

- `SnapshotDataModule` は、重みを渡さない epoch では snapshot の `rank_order` を使う。重み付きは研究用 `weighted_categorical_v1` で、公式 sampler ではない。`state_dict` の `weights` は、その `epoch` の計画に使った実効重みで、未設定は null である。再開は実行側の表と一致するときだけ許し、checkpoint の表で上書きしない。chunk と DeepFilterNet3 には掛けない。開いている epoch view の途中では変えない。DeepFilterNet3 の batch schedule は、world size 1 の epoch 先頭で `Dfn3IndexSampler` の `micro_batch` だけを付け替える。再開は、保存された `micro_batch` 以外の identity が一致するときその幅へ戻してから identity と `plan_hash` を比べる。他の供給は付け替えない。SEMamba++ は `online.semambapp_gsr` をそのまま使い、micro-batch は付け替えない。`group_size` は accumulation と一致させる。新規の実験 checkpoint は rank 0 が `data_provenance.json` と checkpoint 別 sidecar を書き、`provenance_sha256` にその SHA-256 を入れる。snapshot の素材 hash が無いデータは limitations に残し、評価は UNKNOWN で、独立の証明にしない。`provenance_sha256` が None の checkpoint は、後から同じディレクトリに来歴ファイルができても証明済みにしない。`dfn3.online.official@1` と `semambapp.gsr.official@1` は blocked のままである。
- ESPnet chunk の複数 rank では、checkpoint の `data_state.pipeline` が `espnet_chunk_v1` である。計画は `(seed, epoch, world_size)` から再計算し、`plan_hash` が違えば再開しない。world size または `virtual_world_size` が違う再開は拒否する。先読みした未使用バッチは位置に数えない。来歴 JSON は run に 1 つで、rank 0 だけが書く。rank ごとの消費位置は `data_state` に置き、来歴と混ぜない。
- 再現学習の rank 0 は、出力ディレクトリに `data_provenance.json` を一度だけ排他作成する。`data_state` は再開位置で、来歴 JSON は Dataset・snapshot・使用集合の内容ハッシュと派生元である。両者を混ぜない。`provenance_sha256` を持つ checkpoint は、横の JSON が欠けるかハッシュが違うと再開しない。来歴の無い古い checkpoint は再開できるが、評価の `independent` は `UNKNOWN` である。`UNKNOWN` は独立評価ではない。
- checkpoint は world size 1 でも `rng_by_rank` を持つ。各要素はその rank の Python、NumPy、Torch CPU と、学習に使う現在の CUDA device だけである。これが無い checkpoint は再開しない。
- 既存の checkpoint から新しい実験を初期化したとき（`initialize`）、元の来歴 sidecar を `initialization.data_provenance.json` に写し、新しい実験の来歴に引き継ぐ。初期化で選択集合の履歴は消えない。
- Latest と Archive、再開と初期化の条件、`status.json` の値は `docs/experiments/CHECKPOINT.md` と `docs/experiments/LAUNCHER.md`。
- 評価結果の `independent` は true / false / `UNKNOWN` である。`checkpoint_selection_overlap` は、選択に使った素材と内容または既知の派生元が重なった評価セット名である。情報が足りないときは null で、空リストは重なりが無いと確認できたときだけ使う。`selection_uses_test` は独立性とは別に残す。判定は checkpoint 横の JSON と評価レシピの SHA-256 だけで、`reproduction.duckdb` は開かない。

実験の学習はスナップショットを読む。評価用の名前付きセットを学習の入力にしない。

公開済み snapshot のディレクトリ名は `.` で始まらず、`.partial` で終わらない。export は `.{dataset_id}.partial` に書いてからその名前へ rename する。探索側はこの未公開ディレクトリを開かない。

- 教師はレシピの `output.reference` だけである。`dry_clean` と `reverberant_clean` を取り違えない。`none` のケースは学習に入れず、エラーにする。
- サンプルの識別子は `case_id`（レシピの SHA-256）である。rowid や登録順の採番は学習の並びに使わない。
- `configs/db/init.yaml` の評価セット（LibriSpeech test-clean、CHiME3 et05 を含む）は学習にも検証にも使わない。
- 学習セットの読み込みは `data/db/resource.duckdb` を使わない。このファイルは評価用の名前付きセットを持つ本番 DB だからである。`load_recipe_dataset` はこのファイル名を拒否する。スモーク用セットの登録先は `data/db/training.duckdb`（再構築元 `data/staging/training/`）で、VoiceBank-DEMAND の公式セットではない。内容は `docs/db/TRAINING_SETS.md`。
- チャンク計画の発話長だけは、`data/db/resource.duckdb` の `corpora.audio_files`（`path`、`frames`、`sample_rate`）を読み取り専用で一括取得する。`eval` は読まない。音声のデコードとリサンプリングはせず、torchaudio のリサンプリング後サンプル数と同じ整数を使う。学習セットの並び、クロップ、乱数は変えない。

## 指標の欠損と有効性

- 主指標の WER は、対象発話の全件が揃ったときだけ公式値として確定する。micro 集計（編集回数を足してから割る）で、対象は評価セットの全発話である。`status == "ok"` の発話だけを黙って集計しない。欠けた発話を 0 や 100% として補わない。
- WER を要求するのは `metrics.wer: true` の実行で、条件は `asr.conditions`（既定は noisy と enhanced）。noisy と enhanced は同じ対象集合で、片方が欠ければ両方とも公式値を出さない。
- 必須の WER が 1 件でも欠けた実行は `completed` にしない。`incomplete`（再計算できる失敗。ASR の例外など）か `validation_failed`（入力や参照の不備）で、CLI は終了コード 1 を返す。揃っている分の値は `wer_provisional` に置き、公式値 `wer_normalized` は null にする。
- SE が成功して ASR が失敗した発話は成功ではない。空文字の認識結果は正常な結果で、削除として数える。
- 参照テキストが空、部分音声（`partial_audio`）、長さ不明（`unknown_audio_span`）、入力音声の不備は `validation_failed` である。部分音声に全文テキストを対応させた WER は計算しない。部分に対応するテキストを作る処理は無い。
- 再開は、全条件が揃った発話だけを再利用し、失敗した発話を計算し直す。`evaluation` の無い以前の結果（スキーマ 1）は完全だったと示せないので、公式値にしない。
- 補助指標は別扱いである。値を出せないときは `na`（`reason` 付き）、ライブラリが無ければ `unavailable`。値は null で、集計に 0 として入れない。PESQ / STOI / SI-SDR は有限なスコアの平均だけで、欠損は公式 WER を止めない。clean 参照が無い評価セットでは `na`（`no_clean_reference`）。
- PESQ / STOI はゼロ波形を `na`、非有限入力をエラーとする。STOI のライブラリが発話不足の警告と返す固定値 `1e-5` はスコアにせず `na` とする。SI-SDR の無音推定と、参照に直交する推定は有限値にしない。
- PESQ（`pesq==0.0.4`、ITU-T P.862.2）は、16 kHz では長さが 64 サンプルの倍数をまたぐと内部の VAD 長が 1 増える。全文と直前の bin の末尾を比べて差が 0.1 超なら `na`（`pesq_utterance_split_discontinuity`）にし、生の値を `raw_value` に残す。補正値や欠損を平均に入れない。経緯は `docs/evaluation/PESQ_LENGTH_INVESTIGATION.md`。
- noisy と enhanced の補助指標の差は、`paired_signal_metrics` の同一 case で両方有効な集合を使う。異なる集合の単独平均を差として扱わない。
- WER の正規化（`wer-norm-v1`）などを変えると設定ハッシュが変わり、以前の結果は再利用されない。`--resume` が使えるのは、同じ `run_id` で設定ハッシュが同じときだけ。

## コーパスごとの注意

- CHiME3: 正解テキストは `corpora.audio_files` ではなく、公式注釈 `data/corpora/CHiME3/data/annotations/*.json` の `prompt` から評価セットに入る。
  - CH0 は近接マイクで、雑音環境の収録。clean 参照ではない。レシピの `output.reference` は `none` で、metadata の `clean_reference: false` も残る。CH1–CH6 はタブレット。
  - real と元 WSJ 音声は、時間整合したペアではない。`simu` は雑音付きの公式シミュレーションで、元 WSJ 波形とはペアにしていない。`tr05_org` は clean の元素材で、実録音の参照ではない。
- VoiceBank-DEMAND: 実体は `VoiceBank+DEMAND` で、リンク名は `VoiceBank-DEMAND`。clean/noisy は `clean_testset_wav` と `noisy_testset_wav`（および 28/56 話者の trainset）。`configs/db/lists/voicebank_demand_test.txt` は公式 test 824 件。28 話者 train のうち p286 と p287（872 件）を validation、残り 10,700 件を train とする。test 話者とは重ならない。transcript は同梱 txt があるときだけ索引する。SNR はファイルから推定しない。
- 標準リンクと索引対象は `src/data/corpora_catalog.py` の 14 コーパスだけである。MS-SNSD と DNS2020 を含む。索引は元の sample rate・フレーム数・チャンネルを保存し、学習用のリサンプリングや切り出しはしない。transcript が公式ファイルにあるコーパス（LibriSpeech、LibriTTS、VCTK、VoiceBank-DEMAND、CHiME3 の評価セット）は、学習で使うかどうかに関係なく原文を保存する。雑音と RIR に transcript は無い。
- `make db-bootstrap` は空の別ファイルを作る。出力先が既にあるときは `--rebuild` と明示的な `--backup` が無いと拒否し、別ファイルで再構築する。本番 `resource.duckdb` への切り替えは `--replace` だけで、検証とバックアップのあと `os.replace` で行う。
- 初期リンクと bootstrap は `--dataset-root` または `AUDIO_DB_DATASET_ROOT` を必須とし、引数が優先される。個別パスは `AUDIO_DB_CORPUS_<SOURCE>`（`src/data/corpora_catalog.py` の `corpus_env_var`）で、データセットルートより優先される。全リンクの事前検査後に作成し、競合する既存リンクは変更しない。`make db-link` の `--dry-run` は何も作らない。`--allow-partial` は欠けたコーパスを記録して残りをリンクする。既定は欠けたら一つも作らない。bootstrap の候補 DB は最終 DB と同じディレクトリで生成し、レシピの DB 相対パスを維持する。
- 現在の bootstrap は staging を先に生成せず DB に直接登録するため、上記の staging 必須制約を満たす本番再構築手順としては未完成である。
- 再現用 DB はモデルごとにファイルが分かれている（`reproduction.duckdb`、`reproduction-vb-ondemand.duckdb`、`reproduction-p8.duckdb`、`reproduction-p9.duckdb`）。互いに混ぜず、`resource.duckdb` にも入れない。どのファイルに何があるか、どの staging が再構築元かは `docs/db/INVENTORY.md`。文書のパスが在ると仮定せず、そこで確かめる。
- resource DB を渡さない leakage 監査は `UNKNOWN` であり、`PASS` ではない。
- `dfn3.online.official@1` と `semambapp.gsr.official@1` は blocked で、バインドもスナップショットも無い。代替素材を公式データセットの再現と呼ばない。混合 WAV、劣化 WAV は作らない。
- SEMamba++ の `kaiser_best` と `kaiser_fast` は `resampy==0.4.3`（`requirements/se.in`）を使う。`.venvs/asr` のパスは足さない。

## `data/` の扱い

`.gitkeep` 以外は Git 管理外（`.gitignore` の `/data/*/*`）。実データ・DB・チェックポイント・結果を `git add` しない。

| パス | 扱い |
|---|---|
| `data/corpora/` | コーパスの実体は、リポジトリ外へのシンボリックリンク。読み取り専用として扱い、書き込み・削除・名前変更をしない。リンク名は DB の `path` に入る |
| `data/checkpoints/` | 配布チェックポイント。取得が重い。削除・上書きをしない。ライブラリのコードにはパスを固定せず、呼び出し側が渡す。設定やテストにある `data/checkpoints/...` は例で、無ければテストは skip する |
| `data/results/` | `se-evaluation/<run_id>/` は評価結果の一次記録（`config.json` と `results.jsonl`）。`se-repro/` は公式再現の記録とビルドキャッシュ。新しい実行は新しい `run_id`。既存の run ディレクトリは `--resume` なしでは使えない。DuckDB へ載せるときは、このディレクトリを `register_run` で登録する |
| `data/staging/` | DB の再構築元。登録後も残す。Git 管理外 |
| `data/db/` | `resource.duckdb` は評価の本番。`reproduction*.duckdb` は再現用の `repro` スキーマ。`snapshots/` は学習が読む不変のスナップショット。`backups/` は退避。中身は `docs/db/INVENTORY.md` |
| `data/generated/` | 保存済みの事前生成 WAV の置き場。学習の必須データではない。オンデマンド Dataset はここへ混合 WAV を書かない |
| `data/derived/` | 派生素材の置き場。同じ変換条件の正常なファイルは再利用し、内容が違うファイルは上書きしない |

## DB の書き込みと保護

- 書き込む操作は、`make db-init` と `EvaluationDB` の登録系（`import_manifest`、`create_set_from_corpus`、`register_samples`）が `eval` へ、`make db-ingest` が `corpora` へ。読み取りは `read_only=True` で開く。`src/evaluation/` から `corpora.audio_files` を変更しない。
- DuckDB の書き込みは 1 プロセスだけ。ほかのプロセスが DB を開いている間は、書き込みを伴う操作をしない。
- 実験とテストは一時 DB を使う。テストは `data/` の下に書かない。本番の DB、staging、スナップショットを作る処理は Make ターゲットに置き、テストに入れない（`.agents/OPERATIONS.md`）。
- 再現用 DB の登録は staging を再構築元にし、同じ ID で内容が違う行は拒否する。再構築は新しいファイルへ書き、全表のハッシュが一致したときだけ置き換える。学習プロセスはこの DB を開かず、`data/db/snapshots/<dataset_id>/` を読む。学習後の `register_training_run` だけが `repro.runs` を書く。
- `research_checkpoint_selection` は研究用検証集合の claim で、公式再現の claim ではない。
- 再構成した DB（`schema_info.database_origin = reconstructed_from_verified_snapshot`）は元ファイルの復旧ではない。元との同一性は未検証として扱う。
- 破壊的な操作（`db-ingest`、DB の削除、スキーマの作り直し）の前に、`data/db/resource.duckdb` を退避する。
