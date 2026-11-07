# P10-C Preflight 最終監査

日付: 2026-11-07。本格学習は開始していない。

## 総合

CONDITIONAL GO。MP-SENet、MossFormerGAN の 1 GPU、TF-GridNet は開始できる。MossFormerGAN の 2 GPU、TF-Locoformer の研究用 1,024 件、元の 36,000 件検証は NO-GO である。独立検証は [AGENT5_RED_TEAM.md](AGENT5_RED_TEAM.md)。2 GPU の判定はそちらに合わせた。短い GPU forward が両 rank で終わったことは、公式との数値一致ではない。

## 公式との一致

- MP-SENet の検証入力は、CPU で公式の magnitude 経路と最大絶対誤差 0 だった。切り詰めとの差は 1.5627 だった。
- 公式 PESQ 平均は -1 を残す。研究用集約は残さない。
- MossFormerGAN の初期 G1 と D4 は、担当テストが公式 state_dict と最大絶対差 0 と報告した。このセッションではその重いテストを再実行していない。
- PESQ 欠測は学習を止めない。DDP では全 rank が同じ可否になる。
- 2 GPU の実 Discriminator は、SyncBatchNorm と DDP で両 rank の forward が終わり、running mean が一致した。公式学習ステップとの一致は NOT_TESTED。

## 意図した差

- 失敗した optimizer 群は、公式 1 GPU のように後続 batch で蓄積窓を埋めない。
- Discriminator の DDP は `broadcast_buffers=False`。
- TF-GridNet は固定 `mix_seed`。ビット一致と統計的同等は主張しない。
- TF-Locoformer の選択集合は研究用 Protocol である。公式 `cv_synthetic_small` ではない。
- 公式忠実と leakage-safe は別実験である。

## 研究用 Protocol

`tf_locoformer.dns.accept_recipe.research_valid_fixed@1`。規則 `research_valid_fixed_v1`。seed `20261107`。1,024 件。case id リストの SHA-256 は `73e8d44a3c2196af874f8cddb7a1cc9a6b1745a220d6f1b88fda7a9a502f7ad4`。親は `tf_locoformer.dns.accept_recipe@1`、snapshot `3343a55efb4c57d9ffef9179fb2f39c3e0186d36f587a0f4ad9f0aa14e9926f5`。親は変更していない。登録と新しい snapshot hash は BLOCKED。`selection_uses_test` は false。`leakage_safe` ではない。毎 epoch、`valid.loss` で選ぶ設定は YAML にある。その YAML の Preflight は data gap で FAIL である。

外挿時間は、検証 1 回 574 秒、150 epoch で 23.92 時間、学習 384.4 時間、checkpoint 書き込み 343.5 秒、合計約 408.4 時間。再計測ではない。

## 残存リスク

- 実 VoiceBank 824 件の PESQ 失敗件数は未計測。
- MossFormer の GPU 検証ループ、公式 2 GPU との損失一致、clip が 10 を超える頻度は NOT_TESTED。
- 研究用 Dataset を DB と snapshot に追記するまで、Locoformer の本格 YAML は開けない。
- `tests/reproduction` の全件結果は、この監査を書いた時点では未完了であり得る。未完了を PASS とはしない。
- 独立検証の文書がこの監査より後に違う分類を書いた場合は、そちらの証拠を優先する。

## 4 モデルの開始条件

| モデル | 判定 | 理由 |
|---|---|---|
| MP-SENet 公式忠実 | CONDITIONAL GO | 検証入力と PESQ 集約は CPU で確認した。GPU での全長検証は NOT_TESTED。`official_faithful` は test を選択に使う |
| MP-SENet leakage-safe | CONDITIONAL GO | 別 YAML と別 claim がある。今回の数値修正は公式損失の検証経路であり、safe 分割の再計測はしていない |
| MossFormerGAN 1 GPU | CONDITIONAL GO | 初期化と PESQ スキップは CPU。crop の乱数源は未修正 |
| MossFormerGAN 2 GPU | NO-GO | 短い SyncBN forward は両 rank で終わった。検証 batch は全 rank で同一であり、公式の DistributedSampler ではない。DDP は `broadcast_buffers=False`、`find_unused_parameters=False` |
| TF-GridNet | CONDITIONAL GO | 本格 YAML と world size 拒否と Preflight の実行条件 PASS。固定 seed は意図した差。学習は未開始 |
| TF-Locoformer 研究用 1,024 | NO-GO | 定義とリーク監査はある。snapshot が無く Preflight は FAIL |
| TF-Locoformer 36,000 を毎 epoch | NO-GO | 選択方式として採用しない。検証時間が学習を上回る見積のまま |

## 本格学習

開始しない。
