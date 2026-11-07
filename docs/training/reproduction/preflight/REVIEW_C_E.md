# P10-C Preflight Phase 2 — Agent C × Agent E

調査日: 2026-11-07。担当は Dataset identity と Checkpoint 再開。Phase 1 の `AGENT_C_DATA_PIPELINE.md` と `AGENT_E_CHECKPOINT_RESUME.md` を、現行ソースと `data/db/` のファイル一覧に照らした。DuckDB は開いていない。`src/`、YAML、snapshot、checkpoint、学習プロセスは変更していない。GPU は使っていない。

両文書に `DISPUTED` の項目は無い。このレビューで新たに `DISPUTED` へ置いた項目も無い。

## 判定

| ID | Phase 1 | このレビュー | 範囲 |
|---|---|---|---|
| P10P-C-001 | BLOCKER | 降格。本格開始は NON-BLOCKER | `reproduction.duckdb` と `data/staging/reproduction` が無くても、残っている snapshot の学習は開始できる。登録 DB との同一性は未証明のまま |
| P10P-C-002 | BLOCKER | BLOCKER のまま | DeepFilterNet3 のオンライン再現だけ。snapshot が 0 件なので開始できない。MP-SENet / MossFormerGAN / TF-GridNet / TF-Locoformer の開始条件には入らない |

Validation 途中停止で当該 epoch をやり直す記述と、再開後に Dataset が変わり得る記述は矛盾しない。やり直すのは、再開が受理されたときの制御位置である。公式の再現 snapshot は `snapshot_hash` が変わると再開を拒否する。validation 集合だけ差し替えられるのは、別経路の `RecipeDataset` で、かつ `strict_resume` が偽のときである。

## 学習は snapshot だけを読む

再現 YAML（`data.pipeline: reproduction`）の入口は `src/training/se/run.py` 118–166 行である。chunk は `open_chunk_datasets`、それ以外は `open_training_data`。どちらも `find_snapshot_dir` と `load_snapshot` だけを通り、`src/reproduction/paths.py` の `DEFAULT_DB`（`data/db/reproduction.duckdb`）を参照しない。

- `open_training_data`（`src/reproduction/pipeline/training_data.py` 306–327 行）の文書は「DuckDB stays closed」。実装は snapshot ディレクトリを `load_snapshot` する。
- `open_chunk_datasets`（`src/reproduction/pipeline/pregenerated.py` 18–46 行）も同じ `load_snapshot` のあと、on-demand の train / valid ビューを返す。
- `src/reproduction/snapshot_reader.py` 1–4 行は DuckDB を import しないモジュールである。`load_snapshot`（303–340 行）は JSON と npz から論理ハッシュを再計算し、manifest の `snapshot_hash` と違えば `snapshot_hash does not match the logical rows` で拒否する。
- `AssetStore`（`src/reproduction/pipeline/store.py` 1–5 行）はファイルパスを snapshot の prefix から解決し、DuckDB を開かない。
- `src/reproduction/pipeline/` と `src/reproduction/build/` に `duckdb.connect` は無い。`src/training/` で DuckDB を開くのはレシピ経路の `src/training/data/dataset.py` だけである。`src/training/se/job_chain.py` 56 行の `duckdb` は import 名の列挙であり、接続ではない。

したがって、再現の本格学習が読むのは snapshot と、そこが指す音声ファイルである。`reproduction.duckdb` が無いことは、この経路の開始を止めない。

レシピ経路は別である。`run.py` 167–189 行の `load_recipe_dataset` は YAML の `data.database` を開く。`load_dataset` は `read_only=True`（`src/evaluation/datasets/evaldb.py` 310–316 行）。`resource.duckdb` というファイル名は学習セットとして拒否する（`dataset.py` 247–248 行）。chunk の長さ取得 `waveform_lengths` は、レシピ Dataset に限って `DUCKDB_MAIN_PATH`（`resource.duckdb`）を `read_only=True` で開く（`dataset.py` 148–156 行、476–478 行）。再現の DNS / MS-SNSD の `waveform_lengths` は snapshot 上の計画長を返し、ファイルも DuckDB も読まない（`dns_ondemand.py` 71–73 行、`ondemand.py` 53–55 行）。P10-C 実行計画の 4 本命 YAML は `pipeline: reproduction` なので、こちらの接続には入らない。

`declared_status != ready` は開始時に拒否する（`training_data.py` 80–81 行、`dns_ondemand.py` 43–44 行、`ondemand.py` 31–32 行、`online_dfn3.py` 67 行、`online_semamba.py` 68–69 行）。ready の snapshot が 1 件だけあれば `find_snapshot_dir`（`training_data.py` 390–407 行）はそれを返す。0 件なら `found 0 snapshots` で開始前に落ちる。

## ファイルの再確認

2026-11-07 に `data/db/` を一覧した。読み取りはディレクトリ一覧と、各 snapshot の `dataset.json` だけである。

| パス | 結果 |
|---|---|
| `data/db/reproduction.duckdb` | 無い |
| `data/staging/reproduction` | 無い |
| `data/db/reproduction-p8.duckdb` | 無い |
| `data/db/snapshots/*/dataset.json` | 6 件。`dfn3` は無い |

6 件の `name` は `mp_senet.vb28.ondemand.official`、`mossformergan.vb28.ondemand.official`、`tf_gridnet.ms_snsd.espnet_recipe`、`tf_locoformer.dns.accept_recipe`、`semambapp.gsr.substitute`、`semambapp.gsr.smoke`。いずれも version 1、`declared_status` は ready。Agent C の件数表と同じ集合である。

## P10P-C-001 を降格する

Phase 1 の事実（DB と staging が無く、4 本命は DB 行との突合も protocol 式の再計算もできない）は維持する。変えるのは P10-C 影響である。

開始に必要なのは snapshot である。上の 6 件はディレクトリとして残っており、再現経路はそこだけを開く。`reproduction.duckdb` を戻さないと本格学習が起動できない、という BLOCKER にはしない。

残るのは検証の穴である。登録 DB と snapshot が同一であることは、この作業ディレクトリでは証明できない。protocol 式に必要な親 protocol と `deviation_class` は snapshot に無く、SEMamba 以外の式の再計算もできない。これは Agent C が影響の段に書いた「論理内容は読める。登録 DB と同一であることは未証明」のままである。開始条件へは昇格させない。

safe / smoke など、YAML が指して snapshot が無い Dataset は、`find_snapshot_dir` が 0 件で落ちる。公式 4 本命の YAML が指す名前は、上の 6 件に含まれる。

## P10P-C-002 は BLOCKER のまま

DeepFilterNet3 のオンライン再現は、読む snapshot が無い。`open_training_data` は `online.dfn3` のとき `OnlineDFN3TrainingData` を返すが（`training_data.py` 334–353 行）、その前に `find_snapshot_dir` が 0 件で失敗する。`reproduction-p8.duckdb` が無いことは、この失敗の原因ではない。snapshot を戻せば DB 無しで開く、という C-001 の分離は、ファイルが無い C-002 には当てはまらない。

`configs/training/deepfilternet3.yaml` は `pipeline: reproduction` を持たず、`data/db/voicebank.duckdb` の `voicebank_28_train` / `voicebank_28_valid` を指す。ファイル先頭のコメントはオンライン混合を使わないと書いている。この VoiceBank ホールドアウトは、公式オンラインデータの再現として開始できる状態ではない。`configs/training/` に `dfn3.online` を指す YAML も無い。

範囲は DFN3 オンラインに限る。実行計画に時間が書いてある MP-SENet、MossFormerGAN、TF-GridNet、TF-Locoformer は、それぞれ ready な snapshot がある。C-002 はそれらの開始を止めない。

## 再開は identity と snapshot hash が違うと拒否する

`Trainer._check_resume`（`src/training/core/trainer.py` 1365–1366 行）は、pickle の `dataset` と、今回渡した学習 Dataset の `identity()` が違うとき `checkpoint dataset identity does not match this run` で拒否する。pickle へ書く側は `_save` の `dataset.identity()`（1160 行）である。`data_state.json` は再開の入力ではない。位置は pickle の `data_state` と `epoch` / `micro_batches_consumed`（190–200 行）である。

再現経路の identity は `snapshot_hash` を含む。

| 経路 | identity | 追加の一致検査 |
|---|---|---|
| 固定ペア（MP-SENet / MossFormerGAN） | `training_data.py` 132–160 行。`snapshot_hash`、`train_ids_sha256`、`valid_ids_sha256` | `check_resume`（269–271 行）が `data_state` 全体を比べる。`data_state` も `snapshot_hash` を持つ（243–247 行）。不一致は `require_resume_match`（`src/reproduction/pipeline/state.py` 14–23 行） |
| オンライン（DFN3 / SEMamba） | `online_dfn3.py` 114–120 行、`online_semamba.py` 114–120 行。どちらも `snapshot_hash` | 同じ `check_resume`。DFN3 の `data_state` は 212–216 行で `snapshot_hash` を持つ |
| chunk の on-demand（TF-GridNet / TF-Locoformer） | train スロットの identity。`ondemand.py` 61–71 行、`dns_ondemand.py` 79–89 行。`snapshot_hash` は snapshot 全体 | `check_resume` は無い。`_check_resume` は `espnet_chunk_v1` なら `_check_chunk_state`（865–873 行）のあと return する（1378–1383 行）。chunk state の `dataset` は train identity（862 行） |

`snapshot_hash` の入力は `snapshot_reader.py` 20–28 行と 56–58 行の全 part（dataset、assets、cases、case_set_items、material_set_members、docs、generated）である。valid の case 集合が変わると `case_set_items` が変わり、train identity の `snapshot_hash` も変わる。chunk の train identity は train の `case_ids_sha256` だけを持ち、valid の case id ダイジェストは別フィールドには無い。拒否は、共有の `snapshot_hash` が変わることで起きる。train と valid は一つの `LoadedSnapshot` から作る（`ondemand.py` 142–174 行）。

provenance も同じ train identity を比べる。`assert_checkpoint_provenance`（`src/training/provenance.py` 202–205 行）は `dataset.identity()` と文書の `dataset_identity` が違うと拒否する。再現の `provenance_parts` は train / valid / test の case 集合を `sets` に入れる（`src/reproduction/pipeline/provenance.py` 18–22 行、41–54 行）。再開の直接比較は identity であり、その中の `snapshot_hash` が valid の行変化を含む。

Agent E の「Dataset identity の `size` を変えると `checkpoint dataset identity does not match this run`」は、この `payload["dataset"] != dataset.identity()` と同じ文である。`size` はレシピ identity のキー（`dataset.py` 113 行）で、再現 identity のキーは `train_size` である。拒否文は両方に共通する。

## Validation 途中停止と、再開後の Dataset

二つの記述は別の質問に答えており、食い違わない。

途中停止のやり直しは、checkpoint の位置の話である。chunk は epoch 末の Validation と Scheduler のあとで `step_*.pt` を書く。単一プロセスは `_run_chunk_epochs` 559–588 行、world が 1 でないときは 518–519 行から `_run_aligned_chunk_epochs` 766–796 行へ入る。出荷の TF-Locoformer は 4 rank なので後者である。順序は同じで、Validation の batch ループ（`_validate_aligned_chunks` 894–904 行、単一プロセスは 964 行）の途中では `step_*.pt` を書かない。`checkpoint_interval: 0` の定期保存は epoch 完了時だけである。

Validation 中にプロセスが消えると、最後の公開 pickle は前 epoch の終端である。`_save` は epoch を終えた cursor を「次 epoch、cursor 0」として書く（1147–1150 行）。再開はその位置から当該 epoch の Optimizer 更新を行い、Validation は先頭から 1 回である。完了済み epoch の更新は、その pickle に入っているのでやり直さない。Validation だけの再開分岐は無い。Agent E のミニ実行（単一プロセス）と、4 rank のソース順序は、この制御で一致する。4 rank の途中停止そのものは、このレビューでは実行していない。

Dataset が変わり得るのは、その再開が受理されたあとに、どのオブジェクトで Validation するかである。

- 再現 snapshot では、valid の論理行が変わると train identity の `snapshot_hash` が変わり、1365–1366 行で再開を拒否する。拒否された再開は epoch をやり直さない。公式 TF-Locoformer（`tf_locoformer.dns.accept_recipe@1`）はこの側である。Agent E の P10P-E-010 も「valid の case 集合が変われば hash が変わる」と書いており、P10P-E-002 のやり直しと反対の動作ではない。
- `RecipeDataset` の chunk（`configs/training/tf_locoformer_voicebank.yaml`）では、pickle に入る identity は train だけである（`dataset.py` 105–124 行）。validation は別の `load_recipe_dataset`（`run.py` 179–189 行）で、`_check_resume` に渡らない。train の case id と `size` が同じなら、validation セット名を変えても identity 検査は通る。次に完了する epoch の終端で、その新しい Validation が `chunk_history` に足される。途中停止の再開では、その「次に完了する epoch」が、やり直している当該 epoch である。完了済み epoch の history は置き換えない。
- この差し替えを拒否するのは `strict_resume` の設定比較である。`execute` の既定は `strict_resume=False`（`run.py` 72–74 行）。CLI は `execute(args.config, resume=args.resume)` だけで、strict を渡さない（`src/training/se/cli.py` 26 行）。`job_chain` は `strict_resume=True`（`job_chain.py` 163 行）で、`logging` と `verification_max_updates` を除いた config が違うと `checkpoint configuration does not match this experiment`（`run.py` 201–208 行）になる。`data.validation_set` はその config に含まれる。P10P-E-010 の「train identity が一致すれば拒否しない」は Trainer の identity 検査と、strict 無しの CLI では成り立つ。job chain の再開は、validation セット名の変更を config 不一致として拒否する。差し替え実験は Phase 1 と同様、このレビューでも実行していない。確度はソースの `CODE` のままである。

P10P-E-002 の「連続実行と重みが一致した」は、同じ Dataset のまま止めて再開したミニ実行の結果である。P10P-E-010 は、その一致を「validation を差し替えても history まで一致する」とは書いていない。差し替え時に一致するのは train identity の検査だけで、やり直した epoch の Validation 値は新しい集合のものになる。二つの文は同時に真である。
