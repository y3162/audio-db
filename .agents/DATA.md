# データ・DB・再現性

仕様の詳細は `docs/db/SETUP.md`、`docs/db/CORPORA.md`、`docs/db/tables/AUDIO_FILES.md`、`docs/evaluation/PIPELINE.md`。ここには、コードを読まないと分からない制約だけを書く。次回実装予定の方針は `DECISIONS.md`。

## DB の構成

DuckDB は `data/db/resource.duckdb`（`src/data/constants.py`）。

- `corpora.audio_files`: `make db-ingest` が作る。`path` が主キーで、値は `data/corpora` からの相対パス（コーパス名を含む）。採番した ID は無い。
- `eval.cases` / `eval.sets` / `eval.set_items`: `make db-init` が作る。`eval.cases` から `corpora.audio_files` への外部キーは無く、`corpus_path` の文字列で対応している。
- DB ファイル名の stem を `eval` にしない。DuckDB のカタログ名と衝突して `eval.cases` が曖昧になる。

## `make db-init` と `make db-ingest`

| | `make db-init` | `make db-ingest` |
|---|---|---|
| 実行 | `.venvs/lab/bin/python -m src.data.init_eval` | `uv run python -m src.data.ingest_corpora`（環境を指定していない） |
| 役割 | `eval` スキーマを作り、`configs/db/init.yaml` の Manifest・コーパス選択・注釈を評価セットとして登録する | `data/corpora` を全走査し、`corpora.audio_files` を入れる |
| `corpora.audio_files` | 変更しない。前後の指紋（件数・path の最小/最大・ハッシュ）が違えばエラー | `DROP TABLE IF EXISTS` で消してから作り直す |
| 再実行 | 冪等。同じ内容は増えない。同じ ID で内容・順序が違う再登録はエラー（その set は一部だけ残らない） | 毎回、全削除して全投入 |
| 入力が無いとき | `required: false` は理由を出してスキップ。`required: true` は失敗 | — |

`make db-ingest` の危険:

- 実行した時点で既存の `corpora.audio_files` が消える。ユーザーが明示したときだけ、`data/db/resource.duckdb` を退避してから実行する。
- `eval` スキーマは消えない。しかし `eval.cases` の `case_id`・`corpus_path` は `corpora.audio_files` の `path` 文字列から作られている。コーパスの配置やシンボリックリンク名が変わって `path` が変わると、DB は何もエラーを出さずに対応が崩れる。
- `db-init` から `db-ingest` を呼ばない。

## 評価セット

- 評価の入力は名前付きセット（`eval.sets` の `name`）だけ。名前は `[A-Za-z0-9._-]{1,128}`。順序（`position`）は登録時のまま。同名で内容や順序が違う登録は、上書きせずエラーにする。
- Manifest を評価設定の `dataset.source` に使えない。読むのは登録時（`make db-init` の `manifests:`、`EvaluationDB.import_manifest`）だけで、登録後は Manifest ファイルが無くても読める（音声は要る）。
- 音声パスは DB ファイルからの相対パスで保存する。絶対パスは保存しない。プロジェクトと `data/corpora` の外にある音声は登録できない。
- 合成（`synthetic`）の case は、SNR・seed・`rms_v1`・雑音の切り出し位置・`max_seconds`・クリップ設定を case に保存し、評価時に同じ混合を再生成する。名前付きセットでは `mixing.enabled: false` にする。
- 名前付きセットを使わない `dataset.source: duckdb` ＋ `mixing.enabled: true` の経路も残っているが、比較実験には使わない。
- `create_set_from_corpus(query=...)` は、`corpora.audio_files` を読む単一の SELECT だけを受け付ける。
- 現在の `case_id` は、パス・SNR・seed などから作る文字列。`corpus:{path}`、`synthetic:{clean}|{noise}|{snr}|{seed}|{protocol}`、`chime3:...`、`voicebank:...`、Manifest は `utterance_id` そのまま。新しい方式は `DECISIONS.md`。

## 指標の欠損と有効性

- 値を出せないときは `na`（`reason` 付き）、ライブラリが無ければ `unavailable`。理由は `no_reference_transcript`、`partial_audio`、`unknown_audio_span`、`no_clean_reference` など。値は null で、集計に 0 として入れない。
- WER は編集回数を足してから割る micro 集計で、`N = 0` の発話は分母に入れない。PESQ / STOI / SI-SDR は有限なスコアの平均だけ。
- 部分音声: `max_seconds` などで発話の先頭だけを処理したとき、全文テキストと比べた WER は無効で、`na`（`partial_audio`）にする。元の長さが `max_seconds` 以下なら全文として計算する。長さが取れなければ `unknown_audio_span`。部分に対応するテキストを作る処理は無い。
- clean 参照が無い評価セットでは、PESQ / STOI / SI-SDR は `na`（`no_clean_reference`）。
- WER の正規化（`wer-norm-v1`）などを変えると設定ハッシュが変わり、以前の結果は再利用されない。`--resume` が使えるのは、同じ `run_id` で設定ハッシュが同じときだけ。

## コーパスごとの注意

- CHiME3: 正解テキストは `corpora.audio_files` ではなく、公式注釈 `data/corpora/CHiME3/data/annotations/*.json` の `prompt` から評価セットに入る。
  - CH0 は近接マイクで、雑音環境の収録。clean 参照ではない（case に `clean_reference: false`）。CH1–CH6 はタブレット。
  - real と元 WSJ 音声は、時間整合したペアではない。`simu` は雑音付きの公式シミュレーションで、元 WSJ 波形とはペアにしていない。`tr05_org` は clean の元素材で、実録音の参照ではない。
- VoiceBank-DEMAND: ID リスト（`configs/db/lists/`、MIT の表記を同梱）だけをリポジトリに置く。wav が `data/corpora/VoiceBank-DEMAND/` に揃うまで登録されない。transcript と SNR は無い。

## `data/` の扱い

`.gitkeep` 以外は Git 管理外（`.gitignore` の `/data/*/*`）。実データ・DB・チェックポイント・結果を `git add` しない。

| パス | 扱い |
|---|---|
| `data/corpora/` | コーパスの実体は、リポジトリ外へのシンボリックリンク。読み取り専用として扱い、書き込み・削除・名前変更をしない。リンク名は DB の `path` に入る |
| `data/checkpoints/` | 配布チェックポイント。取得が重い。削除・上書きをしない。ライブラリのコードにはパスを固定せず、呼び出し側が渡す。設定やテストにある `data/checkpoints/...` は例で、無ければテストは skip する |
| `data/results/` | `se-evaluation/<run_id>/` は評価結果（`results.jsonl` が再集計の入力）、`se-repro/` は公式再現の記録とビルドキャッシュ。新しい実行は新しい `run_id`。既存の run ディレクトリは `--resume` なしでは使えない。現在、結果は DuckDB に書かない |
| `data/db/` | `resource.duckdb` だけ |

## DB の書き込みと保護

- 書き込む操作は、`make db-init` と `EvaluationDB` の登録系（`import_manifest`、`create_set_from_corpus`、`register_samples`）が `eval` へ、`make db-ingest` が `corpora` へ。読み取りは `read_only=True` で開く。`src/evaluation/` から `corpora.audio_files` を変更しない。
- DuckDB の書き込みは 1 プロセスだけ。ほかのプロセスが DB を開いている間は、書き込みを伴う操作をしない。
- 実験とテストは一時 DB を使う。例外として、`tests/evaluation/test_evaldb.py` の `RealDatabaseInitTests` は、実 DB に `db-init` と同じ登録（冪等）を行う。
- 破壊的な操作（`db-ingest`、DB の削除、スキーマの作り直し）の前に、`data/db/resource.duckdb` を退避する。
