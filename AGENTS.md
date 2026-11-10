# audio-db

固定した ASR を使い、音声強調（SE）の出力を比べる研究基盤。主指標は WER（`wer-norm-v1`、micro）。PESQ / STOI は補助。ASR は学習しない。学習の唯一の入口は `src/experiments/`。

## 正本

- 共通ルールはこのファイルと `.agents/`。`.cursor/`、`.codex/`、`CLAUDE.md` はここを指すだけで、内容を複製しない。
- 人間向けの手順は `README.md` と `docs/`。詳細な検証ログは置かない。

## 必ず守る制約

- `data/db/resource.duckdb`、`data/db/reproduction.duckdb`、`data/db/snapshots/`、`data/corpora/`、`data/generated/`、`data/derived/`、`data/checkpoints/`、`data/results/`、`data/staging/` は Git 管理外。削除・上書き・再作成をしない。`make db-ingest` は `corpora.audio_files` を DROP して作り直す。ユーザーが明示したときだけ、DB を退避してから実行する。`make reproduction-init-db` は `reproduction.duckdb` だけを作り、`resource.duckdb` には触れない。
- 生成条件の正本はレシピ JSON。`case_id` は正規化レシピの SHA-256 で、レシピ自身には含めない。
- 登録データは DuckDB の前に `data/staging/` へ書く。登録後に消す一時ファイルではない。
- `make db-init` は `recipe_json` の無い `eval.cases` を見つけると、変更せず拒否する。
- 評価の入力は DuckDB の名前付き評価セットだけ。評価の実行時に Manifest を読まない。
- `na`・`unavailable`・skip を 0 として扱わない。部分音声に全文テキストを対応させた WER を計算しない。
- CHiME3 の CH0 を clean 参照とみなさない。CHiME3 real と元 WSJ を、時間整合したペアとみなさない。
- `src/se/` と `src/asr/` は他の `src/` を import しない。`transcribe()` と `enhance()` は推論専用。
- 独立 SE は `src/se/official/` に実行時依存しない。ESPnet 全体をコピーしない。
- チェックポイントのパスをライブラリに固定しない。
- 学習 checkpoint は `audio-db-experiment-v1`。`audio-db-training-v1` は読まない。
- `selection_uses_test` が true の checkpoint は、評価素材が離れていても independent にしない。不完全な provenance は UNKNOWN のままにする。
- GPU は共有資源。使う前に `nvidia-smi` で空きを確認する。CUDA の実験 runtime は、使用メモリが空いていない GPU を拒否する。`allow_busy_gpu: true` はその GPU が自分のジョブだと分かっているときだけ。
- TSUBAME で 30 分程度以上のジョブは、投入前に明示的な指示が必要。`h_rt` を 30 分未満にして回避しない。優先度は指示がなければ最低（`qsub` / `qrsh` は `-p -5`）。
- Mock、実モデルのスモーク、公式出力との一致を混ぜて「検証済み」と書かない。

## 作業別に読むファイル

| 作業 | 読む |
|---|---|
| 目的、モジュール責務、推論と学習の分離 | `.agents/PROJECT.md` |
| 依存方向、損失の順序、checkpoint | `.agents/ARCHITECTURE.md` |
| DB、コーパス、評価セット、欠損の扱い | `.agents/DATA.md` |
| 環境、GPU、テスト、公式コード | `.agents/OPERATIONS.md` |

DB・評価・音声生成・データ形式に触れる前は `.agents/DATA.md` を読む。学習の式を変える前は `.agents/ARCHITECTURE.md` を読む。

## ルールの更新

コード・設定・仕様を変えたら、次のセッションが知っておくべき制約が変わったときだけ正本を更新する。履歴の追記や、形式だけの更新はしない。

| 変更 | 更新先 |
|---|---|
| モジュールの責務・依存関係 | `.agents/PROJECT.md` |
| 損失・更新順・checkpoint の不変条件 | `.agents/ARCHITECTURE.md` |
| DB・データ形式・再現性 | `.agents/DATA.md` |
| 環境・テスト・実行方法 | `.agents/OPERATIONS.md` |
| 全エージェント共通の必須制約・読む順序 | `AGENTS.md` |

詳細の使い方は `docs/` を更新する。`.cursor/`、`.codex/`、`CLAUDE.md` は参照先が変わらない限り更新しない。
