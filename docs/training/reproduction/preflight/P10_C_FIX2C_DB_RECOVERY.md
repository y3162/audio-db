# P10-C Fix.2-C DB の探索と再構成

日付: 2026-11-08。ホストは `tensor`。開始時の HEAD は `b913fbb40daf6463ec28b2f162ca454f128d397c`、ブランチは `feature/reproduction`、`git status` は空だった。Python は 3.12.13、torch は 2.6.0+cu124、DuckDB は 2.0.0。

これは元の `reproduction.duckdb` の復旧ではない。親 snapshot から、内容アドレスが一致する範囲だけを再構成した。

## 探索した場所

- `.env` の `AUDIO_DB_DATASET_ROOT` は `/net/spring/work/seki/database`。中身はコーパスで、`reproduction.duckdb` は無い。
- `data/db/reproduction.duckdb`
- `data/db/backups/reproduction.duckdb`
- `data/staging/reproduction/reproduction.duckdb`
- `data/staging/reproduction/INDEX.json`
- `data/db` の `*.duckdb`（`resource.duckdb`、`reproduction-p9.duckdb`、`voicebank.duckdb`、テスト用 DB、`resource` のバックアップ）
- `/net/spring/work/seki/exp` の隣接チェックアウト `master`、`speech-enhancement`、`speech-enhancement-v2`
- `/tmp/audio-db-final-verify/base` の worktree
- `/tmp/reproduction.duckdb`

コーパスツリーの無制限走査はしていない。

## あったもの

親 snapshot はあった。

| 項目 | 値 |
|---|---|
| ディレクトリ | `data/db/snapshots/5469f0416329a8661df2cc91cd5bd9a149d6ea223c72e69ebdd5696bea6bdfbe` |
| `manifest.json` の snapshot hash | `3343a55efb4c57d9ffef9179fb2f39c3e0186d36f587a0f4ad9f0aa14e9926f5` |
| Dataset | `tf_locoformer.dns.accept_recipe@1` |
| `manifest.json` の mtime | 2026-11-07 09:30:31 +0900 |

同じ `snapshots/` には TF-GridNet、VoiceBank オンデマンド official 2 件、SEMamba++ 代替 2 件もある。SEMamba++ は `reproduction-p9` の系統で、親 DNS の登録グラフではない。

`data/staging/reproduction/research_valid_fixed_v1/dataset_definition.json` は選択定義の写しで、`INDEX.json` は無かった。登録グラフではない。

## 無かったもの

元の `reproduction.duckdb`（P10-A.1 の記録は 953,954,304 バイト、mtime 2026-11-07 09:29:08）は、上の場所に無かった。空の DB を正本として作ってはいない。

受理計画の shard（`data/staging/reproduction/workspaces/dns2020-accept-recipe`）も無かった。360,000 件の計画をコーパスから再実行してはいない。

## 再構成

`src/reproduction/prepare/reconstruct_accept.py` が、親 snapshot と現行の登録関数から親グラフを組む。次が snapshot の id と一致しなければ、DB ファイルを作る前に止める。

- protocol id と `rule_json`
- material set id（speech 65,348、noise 65,302）
- build id（seed 17、`excluded_non_16k_noise` は現行 `resource.duckdb` の数え方で 0）
- train / valid / test の case 列と文書
- dataset id `5469f0416329a8661df2cc91cd5bd9a149d6ea223c72e69ebdd5696bea6bdfbe`

メモリ上のグラフを export 相当の部品に戻すと、snapshot hash は `3343a55e…` のままである。DuckDB へ登録したあとも、親ディレクトリへは書かず、同じ hash になることを確認した。

`schema_info.database_origin` は `reconstructed_from_verified_snapshot`。`original_database_identity` は `unverified`。

## 未復元

- 元 DB ファイルのバイト列と、VoiceBank、MS-SNSD、P7 recipe / smoke、`repro.runs`
- 登録当時の `catalog_id`。再計算値は `d982440c72ce44c999e1ab4c056114df49f5da1203810118d1d00207c41f38e4`。snapshot には無いので、元の値との一致は未検証
- 全 130,950 asset の SHA-256 再計算。確認したのは等間隔 32 件で、不一致 0。残りは snapshot の記録のまま
- 受理監査が数えた描画件数、SNR 誤差、clip 件数。snapshot に無いので書いていない
- 話者 ID と録音 ID。案件の `speaker_id` は空のまま

素材の `sample_format` は登録関数の `PCM_16`、`extra_json` は `{}`、clean の `speaker_id` は `reader_id` である。これらは id には入らない。snapshot の公開列（path、rate、frames、channels、sha256、integrity）は一致している。

## スキーマ

新しい DB は現行スキーマで作った。`research_checkpoint_selection` は最初から CHECK にある。古い CHECK からの移行は、本番ファイルには適用していない。移行テストは一時 DB の `tests.reproduction.test_research_claim` で、行を書き換えずにラベルを足せることと、再実行が冪等であることを確認した。
