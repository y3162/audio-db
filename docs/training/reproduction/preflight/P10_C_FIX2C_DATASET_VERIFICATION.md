# P10-C Fix.2-C Dataset 検証

日付: 2026-11-08。親 snapshot のファイルは作業後も 2026-11-07 09:30:31 の `manifest.json` のままである。

## 識別子

| 項目 | 親 | 研究用 |
|---|---|---|
| 名前 | `tf_locoformer.dns.accept_recipe@1` | `tf_locoformer.dns.accept_recipe.research_valid_fixed@1` |
| dataset id | `5469f041…dfbe` | `e4a53415122aba24055f4ba7eef6367a7e41d062485d74de3d7ce94c07d7bdd0` |
| snapshot hash | `3343a55efb4c57d9ffef9179fb2f39c3e0186d36f587a0f4ad9f0aa14e9926f5` | `e4495fa6ac0048c6346cc65bc66fd71114fd1523f95230537e799610c8092024` |
| claim | `espnet_public_se_recipe_reproduction` | `research_checkpoint_selection` |
| train | 324,000 | 同じ case set `473e0bae…` |
| valid | 36,000 | 1,024。case set `9a5a369b…` |
| test | 150 | 同じ case set `52e7fea8…` |
| speech / noise | 65,348 / 65,302 | 同じ material set |

`selection_uses_test` はどちらも false。研究用は `leakage_safe` でも `cv_synthetic_small` でもない。

## 集合

- 子の train case id 列は親と一致
- 子の test case id 列は親と一致
- 1,024 件は親 valid の部分集合
- train との交差は 0、test との交差は 0
- 一覧 SHA-256 は `73e8d44a3c2196af874f8cddb7a1cc9a6b1745a220d6f1b88fda7a9a502f7ad4`
- 親 valid 36,000 件の case set は残っている

長さと rate:

| split | frames | sample rate |
|---|---:|---:|
| train | 480,000 が 324,000 件 | 16,000 |
| valid | 480,000 が 1,024 件 | 16,000 |
| test | 160,000 が 150 件 | 16,000 |

test は収録済み no-reverb ペアで、30 秒の生成案件ではない。

## リーク

case id の交差は 0。雑音素材は train と共有する。話者と録音 ID は UNKNOWN。clean の SHA-256 は案件文書に無い。`leakage_safe` ではない。checkpoint 選択にだけ使う。

## Reader と Render

子 snapshot を `load_snapshot` と `open_dns_datasets` で開いた。snapshot hash は `e4495fa6…`。

同じ案件を二回描画した最大絶対誤差は 0。

| slot | case id | samples | clean | noisy | finite |
|---|---|---:|---:|---:|---|
| valid | `00277e01b93f739d9b50fb89ec65a8672e68ad92a86d3474a6f25a736c4969f6` | 480,000 | 0 | 0 | true |
| train | `96a541ad80faf837ecda3495f0e6f146d3617b13a48dbea87122fa4d5f76033a` | 480,000 | 0 | 0 | true |

全 360,150 件の描画はしていない。asset SHA-256 の全件再計算もしていない。32 件の標本は snapshot の値と一致した。

子 snapshot の asset 部品 hash と material member 部品 hash は親 snapshot と同じである。cases、docs、case set items、dataset 文書は valid を 1,024 件にしたので違う。

## 冪等

同じ 1,024 件での再登録は、全表の挿入件数 0、claim は `research_checkpoint_selection`、dataset id は変わらない。親 snapshot ディレクトリは上書きしていない。二回目の export も `e4495fa6…` のままである。
