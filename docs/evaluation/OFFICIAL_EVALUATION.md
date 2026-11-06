# Official Evaluation（仕様のみ）

状態: **未実装**（P4、2026-11-06）。実装場所の決定だけを書く。コードは置かない。

Research Evaluation（`src/evaluation/` の本線、`python -m src.evaluation.cli`）は変えない。WER（`wer-norm-v1`）と PESQ / STOI を、`resource.duckdb` の名前付き評価セットで計算する経路である。

## 責務

Official Evaluation は、公式再現の checkpoint を、公式が使った指標で見るための別経路である。

| 項目 | 内容 |
|---|---|
| 置き場 | `src/evaluation/official/`（未作成）。本線の `pipeline.py` には公式スクリプトを入れない |
| 入力 | 再現用 Dataset の test 集合。評価時に公式リストを読んでよいのはこの層だけ |
| 指標 | 公式の計算（MP-SENet の `compute_metrics`、ClearerVoice の `speechscore`、ESPnet の `enh_scoring` など）。`wer-norm-v1` には置き換えない |
| 結果 | `data/results/se-official-eval/<run_id>/`（未作成）。Research Evaluation の `data/results/se-evaluation/` とは混ぜない |
| 主張 | `claim_label` と `selection_uses_test` を結果に書く |

`src/evaluation/official/` も `src.reproduction` を importしない。必要な Dataset・snapshot・集合ハッシュは、学習が書いた `data_provenance.json` と、公式リストのファイルから取る。

## 独立性

Official-faithful で `selection_uses_test` が true のとき、公式 test のスコアはモデル選択に使った集合のスコアである。独立した汎化性能とは書かない。

Leakage-safe のラベルだけでは、任意の評価セットからの独立は証明されない。Research Evaluation が付ける `independent` は、来歴 JSON の内容ハッシュおよび既知の派生元ハッシュと、評価素材のハッシュが突合できたときだけ true または false になる。情報が足りないときは `UNKNOWN` であり、true ではない。

## やらないこと

- 公式指標の実装
- `resource.duckdb` への公式 test の登録
- PESQ / STOI / WER の定義変更
- 本線の結果ディレクトリへの公式スコアの混在
