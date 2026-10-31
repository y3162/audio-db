# audio-db

固定した ASR を使う、音声認識指向型の音声強調（SE）の研究基盤。SE と ASR を独立に差し替え、DuckDB の名前付き評価セットで WER（主）と PESQ/STOI（補助）を比べる。推論と評価に加え、SE の学習基盤がある。最初の学習対象は MP-SENet で、ASR は学習しない。

## 正本

- 共通ルールの正本は、このファイルと `.agents/`。`.cursor/`、`.codex/`、`CLAUDE.md` は正本を指すだけで、内容を複製しない。
- 詳細な仕様・使い方・検証記録は `docs/`（入口は `docs/INDEX.md`）、概要は `README.md`。`.agents/` に写さず、リンクで参照する。
- `.agents/DECISIONS.md` は「実装済み」と「採用決定済み・未実装」を分けて書く。後者をコードにあるものとして扱わない。

## 必ず守る制約

- `data/db/resource.duckdb`、`data/corpora/`（外部の実データへのシンボリックリンク）、`data/checkpoints/`、`data/results/`、`data/staging/` は Git 管理外で、再取得・再生成が重い。削除・上書き・再作成をしない。`make db-ingest` は `corpora.audio_files` を DROP して作り直す。ユーザーが明示したときだけ、DB を退避してから実行する。
- 生成条件の正本はレシピ JSON。`case_id` は正規化したレシピの SHA-256 で、レシピ自身には含めない。操作ごとの列を `eval.cases` に足さない。
- 登録用データは DuckDB へ書く前に `data/staging/` へ保存する。これは再構築元であり、登録後に消す一時ファイルではない。
- `data/db/resource.duckdb` はレシピスキーマである。旧固定列のファイルは `data/db/backups/resource-legacy-20261031.duckdb` に残してある。`make db-init` は `recipe_json` の無い `eval.cases` を見つけると、変更せず拒否する。
- 評価の入力は DuckDB の名前付き評価セットだけ。評価の実行時に Manifest を読まない。
- `na`・`unavailable`・skip を 0 として扱わない。部分音声に全文テキストを対応させた WER を計算しない。
- CHiME3 の CH0 を clean 参照とみなさない。CHiME3 real と元 WSJ を、時間整合したペアとみなさない。
- `src/se/` と `src/asr/` は他の `src/` モジュールに依存させない。ASR の `transcribe()` と SE の `enhance()` は推論専用で、学習処理を混ぜない。
- 独立 SE 実装を、`src/se/official/`（Git 管理外・将来削除予定）などの公式コードに実行時依存させない。ESPnet 全体をコピーしない。
- チェックポイントのパスをライブラリのコードに固定しない。
- GPU は共有資源。使う前に `nvidia-smi` で空きを確認する。
- Mock の成功、実モデルの推論スモーク、公式出力との一致検証を区別する。確認していないことを「検証済み」と書かない。

## 作業別に読むファイル

| 作業 | 読む |
|---|---|
| 目的、モジュールの責務・依存関係、推論と学習の分離 | `.agents/PROJECT.md` |
| DB、コーパス、評価セット、結果、指標の欠損の扱い | `.agents/DATA.md` |
| 仮想環境、GPU、テスト、公式コード、検証の区別 | `.agents/DEVELOPMENT.md` |
| 設計判断、次回実装予定（レシピ JSON など） | `.agents/DECISIONS.md` |
| SE の学習、再現性、勾配蓄積、checkpoint | `.agents/PROJECT.md`、`.agents/DATA.md`、`.agents/DEVELOPMENT.md`、`.agents/DECISIONS.md`、`docs/training/MP_SENET.md` |

DB・評価・音声生成・データ形式に触れる前は、`.agents/DATA.md` と `.agents/DECISIONS.md` の両方を読む。

## ルールの更新

コード・設定・仕様を変えたら、完了前に次を確認する。

> この変更で、次のセッションのエージェントが知っておくべき制約、前提、設計判断が変わったか。

変わったときだけ、該当する正本を更新する。変更履歴の追記や、形式だけの更新はしない。

| 変更 | 更新先 |
|---|---|
| モジュールの責務・依存関係 | `.agents/PROJECT.md` |
| DB・データ形式・再現性 | `.agents/DATA.md` |
| 環境・テスト・実行方法・検証の区別 | `.agents/DEVELOPMENT.md` |
| 重要な設計判断（採用・実装・撤回） | `.agents/DECISIONS.md` |
| 全エージェント共通の必須制約・読む順序 | `AGENTS.md` |

- 採用決定済みの事項を実装したら、`DECISIONS.md` で「実装済み」へ移し、現状の記述（`PROJECT.md`、`DATA.md`）も合わせる。
- 詳細仕様と検証記録は `docs/` を更新する。
- `.cursor/`、`.codex/`、`CLAUDE.md` は、参照先やエージェント固有の事情が変わらない限り更新しない。
