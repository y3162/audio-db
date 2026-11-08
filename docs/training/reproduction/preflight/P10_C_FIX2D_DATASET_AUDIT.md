# P10-C Fix.2-D Dataset 監査

日付: 2026-11-08。ホスト `tensor`。既存 snapshot と `reproduction.duckdb` は書き換えていない。音声の事前生成もしていない。

## DB の来歴

`data/db/reproduction.duckdb` を読み取り専用で開いた。

| キー | 値 |
|---|---|
| `database_origin` | `reconstructed_from_verified_snapshot` |
| `original_database_identity` | `unverified` |
| `reconstructed_parent_snapshot_hash` | `3343a55efb4c57d9ffef9179fb2f39c3e0186d36f587a0f4ad9f0aa14e9926f5` |
| `reconstructed_parent_dataset_id` | `5469f0416329a8661df2cc91cd5bd9a149d6ea223c72e69ebdd5696bea6bdfbe` |

登録されている Dataset は次の 2 件だけである。

| name | dataset id | claim |
|---|---|---|
| `tf_locoformer.dns.accept_recipe@1` | `5469f0416329a8661df2cc91cd5bd9a149d6ea223c72e69ebdd5696bea6bdfbe` | `espnet_public_se_recipe_reproduction` |
| `tf_locoformer.dns.accept_recipe.research_valid_fixed@1` | `e4a53415122aba24055f4ba7eef6367a7e41d062485d74de3d7ce94c07d7bdd0` | `research_checkpoint_selection` |

VoiceBank、MS-SNSD、P7 の recipe / smoke、過去の training run はこの DB に無い。このファイルを、それらが入っていた歴史上の `reproduction.duckdb` と同一視しない。

学習は DB を開かず snapshot を読む。VoiceBank と MS-SNSD の学習入口は、下の snapshot がディスク上にあることで足りる。今回それらを DB へ登録していない。

## Snapshot

`load_snapshot` は論理行のハッシュと manifest の `snapshot_hash` を照合する。4 モデルと親 DNS snapshot はすべて一致した。件数に重複 case id は無い。

| モデル | Dataset | dataset id | snapshot | train / valid / test |
|---|---|---|---|---|
| MP-SENet | `mp_senet.vb28.ondemand.official@1` | `36ab4590b113d0aa00884c44041eaa97d65085cee57bcc9c9e06135741fc76d7` | `265f256b938a89746dcaa62cd44daa2c6bcca3bce4cc1b1719eb2170717d33d5` | 11,572 / 824 / 824 |
| MossFormerGAN | `mossformergan.vb28.ondemand.official@1` | `153096c312f8027bba93fa635224bff8681170841bbe40164d189f923d16aebe` | `ba0c77c94babbbc2c3e70b149554f8a922a68cf6fbe71ccf833927dc71dff555` | 11,572 / 824 / 824 |
| TF-GridNet | `tf_gridnet.ms_snsd.espnet_recipe@1` | `13c6058881ef86fe7506f705b1bd5a5bc68b313181410d8fdaaee20c33ef3fc6` | `a69a477af70bc43dbcabdafbd5c97ec8a26aac4debd51158079545de1e4f0757` | 7,345 / 1,835 / 305 |
| TF-Locoformer 研究 | `tf_locoformer.dns.accept_recipe.research_valid_fixed@1` | `e4a53415122aba24055f4ba7eef6367a7e41d062485d74de3d7ce94c07d7bdd0` | `e4495fa6ac0048c6346cc65bc66fd71114fd1523f95230537e799610c8092024` | 324,000 / 1,024 / 150 |
| TF-Locoformer 親 | `tf_locoformer.dns.accept_recipe@1` | `5469f0416329a8661df2cc91cd5bd9a149d6ea223c72e69ebdd5696bea6bdfbe` | `3343a55efb4c57d9ffef9179fb2f39c3e0186d36f587a0f4ad9f0aa14e9926f5` | 324,000 / 36,000 / 150 |

| モデル | claim | profile | `selection_uses_test` | delivery | renderer |
|---|---|---|---|---|---|
| MP-SENet | `official_reproduction` | `official_faithful` | true | `on_demand` | 固定ペア。48 kHz を学習時に soxr HQ と PCM_16 |
| MossFormerGAN | `official_reproduction` | `official_faithful` | true | `on_demand` | 同じ固定ペア経路。正規化は Dataset の protocol |
| TF-GridNet | `espnet_public_se_recipe_reproduction` | `official_faithful` | false | `on_demand` | `pregenerated.ms_snsd_tfgridnet`。固定 `mix_seed` |
| TF-Locoformer 研究 | `research_checkpoint_selection` | `official_faithful` | false | `on_demand` | `dns_pcm16_accept_v2` |

研究用 manifest の provenance は `database_origin=reconstructed_from_verified_snapshot`、`original_database_identity=unverified`、親 dataset と親 snapshot は上の親と一致する。

## 研究用と親

case id 列を snapshot の case set から比べた。

| 比較 | 結果 |
|---|---|
| train | 一致。324,000 |
| test | 一致。150 |
| 親 valid | 36,000 のまま |
| 子 valid | 1,024。ユニーク 1,024。親 valid の部分集合。重複 0 |

研究用 Dataset は `leakage_safe` でも `cv_synthetic_small` でもない。独立した汎化評価として扱わない。

## 1 件の再描画

全件の波形再構築ではない。各入口で 1 件を二度ではなく 1 回描画し、有限値と形状を見た。

| Dataset | 結果 |
|---|---|
| MP-SENet valid 1 件 | clean / noisy とも長さ 27,861。有限。全長で、hop 倍数への切り詰めではない |
| MossFormerGAN valid 1 件 | 長さ 32,000。有限。validation の 2 秒 crop |
| TF-GridNet valid 1 件 | 長さ 16,000 の chunk。有限。valid 件数 1,835 |
| TF-Locoformer 研究 valid 1 件 | 長さ 16,000 の chunk。有限。valid 件数 1,024 |

全 asset SHA-256 の再計算はしていない。

## Leakage-safe

`mp_senet.vb28.ondemand.safe@1` と `mossformergan.vb28.ondemand.safe@1` は、現在の DB にも `data/db/snapshots/` の dataset 名にも無い。今回の統合 GO の対象に加えていない。未登録のまま残る。

## 不変性

親 snapshot と 4 モデルの snapshot へは書き込んでいない。`load_snapshot` の再計算ハッシュは manifest と一致した。
