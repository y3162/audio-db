# P10-C Fix.2-C 実装

日付: 2026-11-08。本格 150 epoch は始めていない。commit も push もしていない。

## 追加した経路

`src/reproduction/prepare/reconstruct_accept.py`

- 親 snapshot から `tf_locoformer.dns.accept_recipe@1` のグラフを再構成する
- id が snapshot と違うときは DB を作らない
- 既存の `reproduction.duckdb` は上書きしない
- 再構成後に `register_research_validation_dataset` で研究用 Dataset を追記する
- 既存の `export_snapshot` で子 snapshot を書く。二回目は同じ hash でディレクトリを上書きしない
- 親 snapshot のファイル指紋が作業後に変わったら失敗にする

研究用の選択、claim、子の protocol / case set / dataset id は既存の `research_valid_fixed.py` のままである。子の `rule_json` は親のコピーで、validation の note は差し替えていない。

## 変更ファイル

- `src/reproduction/prepare/reconstruct_accept.py`（新規）
- `tests/reproduction/test_reconstruct_accept.py`（新規）
- `tests/reproduction/test_research_claim.py`（空のルートでは登録不能、という検査に変えた。作業木に DB が無いことを成功条件にしない）
- `configs/training/reproduction/tf_locoformer_dns_research_valid_smoke.yaml`（新規。本格 YAML とは別）
- `.agents/DATA.md`、`.agents/DECISIONS.md`
- `docs/training/reproduction/P10_C_EXECUTION_PLAN.md`
- `docs/training/reproduction/preflight/P10_C_PREFLIGHT_FIX_ISSUES.md`

本格 YAML `configs/training/reproduction/tf_locoformer_dns_research_valid_full.yaml` は変えていない。MossFormerGAN の `rng_by_rank` も変えていない。

## 登録結果

| 項目 | 値 |
|---|---|
| 親 dataset id | `5469f0416329a8661df2cc91cd5bd9a149d6ea223c72e69ebdd5696bea6bdfbe` |
| 子 dataset id | `e4a53415122aba24055f4ba7eef6367a7e41d062485d74de3d7ce94c07d7bdd0` |
| 子 protocol id | `9107163580bf9409d2c17012dd19df6703fee0197b0d59d2fe2b386a68818646` |
| 子 valid case set | `9a5a369bee9556178c4f86248229b18ad2d664446e4bff05f5414836aefd30eb` |
| 子 snapshot hash | `e4495fa6ac0048c6346cc65bc66fd71114fd1523f95230537e799610c8092024` |
| claim | `research_checkpoint_selection` |
| `selection_uses_test` | false |
| case id 一覧 SHA-256 | `73e8d44a3c2196af874f8cddb7a1cc9a6b1745a220d6f1b88fda7a9a502f7ad4` |

定義 JSON の未登録内容 id `8bd167b5a650baefb05ab24189d2c540c3fc3f9eb3a19900161e4a75e6d29782` は、validation note を差し替えた protocol の hash である。登録関数はその note を変えないので、発行した dataset id はこれと違う。case set id `9a5a369b…` は定義と一致した。未登録内容 id を登録済み id としては使っていない。

staging は `data/staging/reproduction/INDEX.json` と、`research_valid_fixed_v1/registration_result.json`。同じ内容の再登録は挿入 0 だった。

## テスト

`tests.reproduction.test_research_claim`、`tests.reproduction.test_reconstruct_accept`、`tests.reproduction.test_research_valid_fixed`、`tests.training.test_preflight_full`。27 件、6.6 秒、OK。

不足 snapshot では DB を作らないこと、既存 DB を置き換えないこと、古い CHECK からの移行、研究用登録の冪等、公開リストと親 validation の一致を含む。全 `tests/reproduction` は再実行していない。
