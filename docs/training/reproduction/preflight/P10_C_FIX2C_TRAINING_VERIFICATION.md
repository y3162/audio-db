# P10-C Fix.2-C 学習前検証

日付: 2026-11-08。ホスト `tensor`。150 epoch は始めていない。本格 YAML は書き換えていない。

## Preflight

```text
python -m src.training.se.cli \
  --config configs/training/reproduction/tf_locoformer_dns_research_valid_full.yaml \
  --preflight --world-size 4 \
  --expect-dataset tf_locoformer.dns.accept_recipe.research_valid_fixed@1 \
  --expect-snapshot e4495fa6ac0048c6346cc65bc66fd71114fd1523f95230537e799610c8092024
```

終了コード 0。`status` は PASS。失敗リストは空。

| 項目 | 値 |
|---|---|
| dataset id | `e4a53415122aba24055f4ba7eef6367a7e41d062485d74de3d7ce94c07d7bdd0` |
| snapshot hash | `e4495fa6ac0048c6346cc65bc66fd71114fd1523f95230537e799610c8092024` |
| claim | `research_checkpoint_selection` |
| world | 4 |
| global / micro / accumulation | 4 / 1 / 1 |
| `max_steps` / `max_epochs` | 2,100,000 / 150 |
| counts | train 324,000 / valid 1,024 / test 150 |
| `keep_nbest` | 5 |
| checkpoint | epoch 末。`keep_latest` 3、best、n-best を保護 |
| 保存先 | `data/results/training/reproduction/tf_locoformer.dns.accept_recipe.research_valid_fixed-full` |
| disk | PASS。空き約 1.89 TB。見積 1,979,011,804 バイト |
| smoke cap | 本格 YAML には無い |
| config SHA-256 | `2002547bfb6118d3d1cb85caac87e66030cf9536f77a20b5895368c29544beb0` |

`ave_5best` は、有限な `valid.loss` が 5 件以上あるときの既存 Trainer の平均ファイルである。本格 YAML の `keep_nbest: 5` がその入力である。Preflight PASS は公式再現の成功ではない。

## World size 4 の計画

本格 YAML の cache 1,024、seed 0、epoch 1。計画 hash は二回同じだった。

| 集合 | rank あたりの発話 | 更新 | 破棄 batch | 選択の交差 | 選択の和 |
|---|---:|---:|---|---:|---:|
| train（1,000 / rank） | 1,000 | 14,000 | 0, 0, 0, 0 | 0 | 4,000 |
| valid（全 1,024） | 256 | 3,584 | 0, 0, 0, 0 | 0 | 1,024 |

valid の保持 chunk に現れた case id は 1,024 で、欠落と rank 間の重複は無い。train の計画 hash は `4bd4f9b0e9295c218256878102e483cc62eb03f0b806d790afa0854823da4941`。valid は `c3d9de17ef756c15a58a4319ebefa496b02f30f086450d7e8eb2c0ac13c1fe32`。

これは計画である。1,024 件の validation を学習ループで最後まで走らせてはいない。

## 短時間 GPU

物理 GPU 1, 2, 6, 7。`CUDA_VISIBLE_DEVICES=1,2,6,7`。設定は `configs/training/reproduction/tf_locoformer_dns_research_valid_smoke.yaml`。公式幅、world 4、global 4、micro 1、accumulation 1。`num_cache_chunks` は 128 で、本格 YAML の 1,024 とは違う。この smoke の train plan hash は `433ef66e9c04e0bc38be75e2c7c830d60d815267aa9b2ec623a35c4cc5f026f4`。

`verification_max_updates: 3`、`chunk_iterator.max_updates: 3`、`validation_max_updates: 1`。出力は `data/results/training/reproduction/fix2c-tf-locoformer-smoke`。最初の完走は `fix2c-tf-locoformer-smoke-run1` に残した。

| 回 | 結果 | peak allocated | 壁時計の目安 |
|---|---|---:|---|
| 初回 | exit 0。`INTERRUPTED`。`complete.json` は無い | 16,470,292,992 バイト | 12:38:43 から 12:40:36 まで |
| step 1 から再開 | exit 0。再び step 3 で `INTERRUPTED` | 16,531,107,840 バイト | 12:43:28 から 12:45:13 まで |

学習損失は `mr_l1_tfd`。step 1 の generator 損失は 3779.84521484375。validation は rank あたり 1 batch、集約の count は 4、`valid.loss` は 7095.0062255859375。注記は「iterator_stop のあとの各 rank の接頭辞であり、validation 全体ではない」。

Checkpoint には `rng_by_rank` が 4 要素ある。各要素は Python、NumPy、Torch CPU、その rank の CUDA device。`valid.loss.ave_1best.pt` は `1epoch.pt` へのシンボリックリンクで、`valid.loss.ave.pt` はそれを向く。有限な `valid.loss` は 1 件なので `valid.loss.ave_5best.pt` は無い。

`deterministic: false` なので、再開後の重みが初回とビット一致することは確認していない。再開は同じ設定で step 1 の checkpoint から step 3 まで進み、`INTERRUPTED` で終わった、という確認である。

## NOT_TESTED

- 1,024 件すべての validation 完走
- 150 epoch
- 再開重みのビット一致
- 本格 YAML の cache 1,024 での GPU 実行（GPU で使ったのは smoke の cache 128）
- 全 asset SHA-256 の再計算
