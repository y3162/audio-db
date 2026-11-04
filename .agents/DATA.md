# データ・DB・再現性

仕様の詳細は `docs/db/SETUP.md`、`docs/db/RECIPE.md`、`docs/db/CORPORA.md`、`docs/db/tables/AUDIO_FILES.md`、`docs/evaluation/PIPELINE.md`。ここには、コードを読まないと分からない制約だけを書く。

## DB の構成

DuckDB は `data/db/resource.duckdb`（`src/data/constants.py`）。

- `corpora.audio_files`: `make db-ingest` が作る。`path` が主キーで、値は `data/corpora` からの相対パス（コーパス名を含む）。採番した ID は無い。
- `eval.cases` / `eval.sets` / `eval.set_items` / `eval.runs` / `eval.results`: 新しいスキーマを `make db-init` が空の DB に作る。`eval.cases` の生成条件は `recipe_json` だけに置き、SNR や切り出し位置の列は持たない。`corpora.audio_files` への外部キーは無く、レシピ内の相対パスと metadata の `corpus_path` で対応する。`case_id` は正規化レシピの SHA-256 で、採番や rowid ではない。
- `data/db/resource.duckdb` はこのレシピスキーマである。旧固定列のファイルは `data/db/backups/resource-legacy-20261031.duckdb`（SHA-256 `f3fdcd89b34cf44eda9853e71278806c1ecc4042f03173821805be054b61d156`）にある。`recipe_json` の無い `eval.cases` を見つけると、`make db-init` は何も変更せず拒否する。`data/db/rebuild-a.duckdb` と `data/db/rebuild-b.duckdb` は切り替え前の再構築コピーで、本番と同じ内容である。
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
- 評価ランの一次記録は従来どおり `data/results/se-evaluation/<run_id>/` の `config.json` と `results.jsonl` である。DuckDB への登録は `register_run` で、そのあと行う。既存ランは自動では登録しない。
- `src.data.migrate_legacy` は、以前の `eval.cases` を一度だけレシピへ写す。実行時の互換読み取りではない。出力先が既にあるときは上書きしない。本番は切り替え済みなので、このコマンドを再実行する必要はない。旧 `case_id` は metadata の `legacy_case_id` に残り、新しい `case_id` はレシピハッシュである。

## 学習に使うセット

学習は評価と同じ名前付きセットとレシピを読む。詳細は `docs/training/MP_SENET.md`。

- 教師はレシピの `output.reference` だけである。`dry_clean` と `reverberant_clean` を取り違えない。`none` のケースは学習に入れず、エラーにする。
- サンプルの識別子は `case_id`（レシピの SHA-256）である。rowid や登録順の採番は学習の並びに使わない。
- `configs/db/init.yaml` の評価セット（LibriSpeech test-clean、CHiME3 et05 を含む）は学習にも検証にも使わない。
- 学習セットの読み込みは `data/db/resource.duckdb` を使わない。このファイルは評価用の名前付きセットを持つ本番 DB だからである。学習セットは `data/db/training.duckdb` にあり、再構築元は `data/staging/training/` である。`libri_demand_smoke_train` は LibriSpeech train-clean-100、`libri_demand_smoke_valid` は dev-clean で、DEMAND のシーンは重ならない。VoiceBank-DEMAND の公式セットではない。接続確認に `rebuild-a.duckdb` を読み取り専用で開いてもよい。
- チャンク計画の発話長だけは、`data/db/resource.duckdb` の `corpora.audio_files`（`path`、`frames`、`sample_rate`）を読み取り専用で一括取得する。`eval` は読まない。音声のデコードとリサンプリングはせず、torchaudio のリサンプリング後サンプル数と同じ整数を使う。学習セットの並び、クロップ、乱数は変えない。

## 指標の欠損と有効性

- 値を出せないときは `na`（`reason` 付き）、ライブラリが無ければ `unavailable`。理由は `no_reference_transcript`、`partial_audio`、`unknown_audio_span`、`no_clean_reference` など。値は null で、集計に 0 として入れない。
- WER は編集回数を足してから割る micro 集計で、`N = 0` の発話は分母に入れない。PESQ / STOI / SI-SDR は有限なスコアの平均だけ。
- 部分音声: `max_seconds` などで発話の先頭だけを処理したとき、全文テキストと比べた WER は無効で、`na`（`partial_audio`）にする。元の長さが `max_seconds` 以下なら全文として計算する。長さが取れなければ `unknown_audio_span`。部分に対応するテキストを作る処理は無い。
- clean 参照が無い評価セットでは、PESQ / STOI / SI-SDR は `na`（`no_clean_reference`）。
- PESQ / STOI はゼロ波形を `na`、非有限入力をエラーとする。STOI のライブラリが発話不足の警告と返す固定値 `1e-5` はスコアにせず `na` とする。SI-SDR の無音推定と、参照に直交する推定は有限値にしない。
- PESQ（`pesq==0.0.4`、ITU-T P.862.2）は、16 kHz では長さが 64 サンプルの倍数をまたぐと内部の VAD 長が 1 増える。全文と直前の bin の末尾を比べる既存の保守的検出で差が 0.1 超なら `na`（互換の理由名 `pesq_utterance_split_discontinuity`）にし、生の値を `raw_value` に残す。この2長は最大64サンプル離れ、理由名だけでは発話分割の変化を証明しない。全文スコアが妥当と確認できない長さ依存も検出される。補正値や欠損を平均に入れない。
- `summary.json` の `groups` は各条件の有効値を別々に集計する。noisy と enhanced の差は `paired_signal_metrics` の同一 case で両方有効な集合を使い、件数・case ID・除外理由を残す。異なる集合の単独平均を差として扱わない。
- WER の正規化（`wer-norm-v1`）などを変えると設定ハッシュが変わり、以前の結果は再利用されない。`--resume` が使えるのは、同じ `run_id` で設定ハッシュが同じときだけ。

## コーパスごとの注意

- CHiME3: 正解テキストは `corpora.audio_files` ではなく、公式注釈 `data/corpora/CHiME3/data/annotations/*.json` の `prompt` から評価セットに入る。
  - CH0 は近接マイクで、雑音環境の収録。clean 参照ではない。レシピの `output.reference` は `none` で、metadata の `clean_reference: false` も残る。CH1–CH6 はタブレット。
  - real と元 WSJ 音声は、時間整合したペアではない。`simu` は雑音付きの公式シミュレーションで、元 WSJ 波形とはペアにしていない。`tr05_org` は clean の元素材で、実録音の参照ではない。
- VoiceBank-DEMAND: 実体は `VoiceBank+DEMAND` で、リンク名は `VoiceBank-DEMAND`。clean/noisy は `clean_testset_wav` と `noisy_testset_wav`（および 28/56 話者の trainset）。`configs/db/lists/voicebank_demand_test.txt` は公式 test 824 件。28 話者 train のうち p286 と p287（872 件）を validation、残り 10,700 件を train とする。test 話者とは重ならない。transcript は同梱 txt があるときだけ索引する。SNR はファイルから推定しない。
- 標準リンクと索引対象は `src/data/corpora_catalog.py` の 12 コーパスだけである。索引は元の sample rate・フレーム数・チャンネルを保存し、学習用のリサンプリングや切り出しはしない。transcript が公式ファイルにあるコーパス（LibriSpeech、LibriTTS、VCTK、VoiceBank-DEMAND、CHiME3 の評価セット）は、学習で使うかどうかに関係なく原文を保存する。雑音と RIR に transcript は無い。
- `make db-bootstrap` は空の別ファイルを作る。出力先が既にあるときは `--rebuild` と明示的な `--backup` が無いと拒否し、別ファイルで再構築する。本番 `resource.duckdb` への切り替えは `--replace` だけで、検証とバックアップのあと `os.replace` で行う。
- 初期リンクと bootstrap は `--dataset-root` または `AUDIO_DB_DATASET_ROOT` を必須とし、引数が優先される。個別パスは `AUDIO_DB_CORPUS_<SOURCE>`（`src/data/corpora_catalog.py` の `corpus_env_var`）で、データセットルートより優先される。全リンクの事前検査後に作成し、競合する既存リンクは変更しない。`make db-link` の `--dry-run` は何も作らない。`--allow-partial` は欠けたコーパスを記録して残りをリンクする。既定は欠けたら一つも作らない。bootstrap の候補 DB は最終 DB と同じディレクトリで生成し、レシピの DB 相対パスを維持する。
- 現在の bootstrap は staging を先に生成せず DB に直接登録するため、上記の staging 必須制約を満たす本番再構築手順としては未完成である。
- 現行の本番再構築元は `data/staging/resource-current`。`data/staging/` 直下、`data/staging/resource-full`、`data/staging/voicebank`、`data/staging/training` は過去の再構築元であり、現行の本番には使わない。VoiceBank 学習セットの現行再構築元は `data/staging/voicebank-current`。
- TSUBAME の初回 `resource.duckdb` は 2026-11-02 にコーパスから新規作成した。今回だけの staging 先行省略を含む条件と検証範囲は `docs/db/TSUBAME_INITIAL.md`。研究室側 DB のコピーや同一性の確認ではない。
- 学習セット `voicebank_28_train` / `voicebank_28_valid` は `data/db/voicebank.duckdb`（現行の再構築元 `data/staging/voicebank-current/`）。`data/db/training.duckdb` の LibriSpeech+DEMAND スモークとは別ファイルである。

## `data/` の扱い

`.gitkeep` 以外は Git 管理外（`.gitignore` の `/data/*/*`）。実データ・DB・チェックポイント・結果を `git add` しない。

| パス | 扱い |
|---|---|
| `data/corpora/` | コーパスの実体は、リポジトリ外へのシンボリックリンク。読み取り専用として扱い、書き込み・削除・名前変更をしない。リンク名は DB の `path` に入る |
| `data/checkpoints/` | 配布チェックポイント。取得が重い。削除・上書きをしない。ライブラリのコードにはパスを固定せず、呼び出し側が渡す。設定やテストにある `data/checkpoints/...` は例で、無ければテストは skip する |
| `data/results/` | `se-evaluation/<run_id>/` は評価結果の一次記録（`config.json` と `results.jsonl`）。`se-repro/` は公式再現の記録とビルドキャッシュ。新しい実行は新しい `run_id`。既存の run ディレクトリは `--resume` なしでは使えない。DuckDB へ載せるときは、このディレクトリを `register_run` で登録する |
| `data/staging/` | DB の再構築元。登録後も残す。Git 管理外 |
| `data/db/` | `resource.duckdb` はレシピスキーマの本番ファイル。`training.duckdb` は学習スモークセット。`backups/resource-legacy-20261031.duckdb` は切り替え前の固定列スキーマ。`rebuild-a.duckdb` と `rebuild-b.duckdb` は切り替え前に作った同じ内容の再構築コピー |

## DB の書き込みと保護

- 書き込む操作は、`make db-init` と `EvaluationDB` の登録系（`import_manifest`、`create_set_from_corpus`、`register_samples`）が `eval` へ、`make db-ingest` が `corpora` へ。読み取りは `read_only=True` で開く。`src/evaluation/` から `corpora.audio_files` を変更しない。
- DuckDB の書き込みは 1 プロセスだけ。ほかのプロセスが DB を開いている間は、書き込みを伴う操作をしない。
- 実験とテストは一時 DB を使う。単体テストは `resource.duckdb` を開いて書き込まない。
- 破壊的な操作（`db-ingest`、DB の削除、スキーマの作り直し）の前に、`data/db/resource.duckdb` を退避する。
