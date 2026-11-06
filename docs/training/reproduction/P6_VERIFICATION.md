# P6 検証（2026-11-06）

ESPnet 公開 SE recipe（commit `0d136c5`、`egs2/ms_snsd/enh1`）の MS-SNSD 混合を、学習時に再構築する。論文実験の再現ではない。全量 WAV は作っていない。

## 停止した事前生成

対話プロセス。qsub のジョブ ID は無い。`h_rt` は無い。

| 項目 | 値 |
|---|---|
| コマンド | `.venvs/se/bin/python -u -m src.reproduction.cli ms-snsd --config configs/data/builds/ms_snsd_tfgridnet.yaml --step register` |
| PID | Python 3040394、ラッパー 3040365 |
| 開始 | 2026-11-06T03:46:42Z |
| 終了 | 2026-11-06T04:13:42Z（約 27.0 分、`aborted`、終了コード不明） |
| 計画 | train 9180、test 305、計画 895.7 秒 |
| 停止時の WAV | build `45662315c17acbf0` の完成 15,182 本、5,693,221,236 バイト。未完成 `.part` 1 本、355,458 バイト。shard 0000–0004 が完成し、`fid0005060` を書いていた |

このセッションで確認した時点ではプロセスは残っていなかった。ユーザーの Python に同じコマンドは無く、`qstat -u` にもジョブは出ていない。

## 削除と保持

削除したのは未登録の全量生成だけである。

- `data/generated/ms_snsd_synth/45662315c17acbf0/`（上の WAV と `.part`）
- `data/staging/reproduction/workspaces/ms_snsd/plans/45662315c17acbf0e9fb697403d74aad69808ccd69eb1fee917cb17b66b0f459/`
- 同じ workspaces の test 計画 `de71bc74be29aee1995bbd0ed436a5e7cafd5d0357762a4d2b15b6579c56d1d9/`
- `data/staging/reproduction/workspaces/ms_snsd/builds/45662315c17acbf0e9fb697403d74aad69808ccd69eb1fee917cb17b66b0f459/`

保持したもの:

- `tf_gridnet.ms_snsd.smoke@1` が参照する WAV 90 本、33,724,350 バイト（train build `047023837a46f7b9` の 75 本、test build `0f27826b220ccb7c` の 15 本）。DB 登録済みなので消していない。
- そのスナップショット `6a6335376cac2cde7659651438629a1cf444f3bff5cf8816684034b36a434f09`
- VoiceBank の 4 Dataset とスナップショット
- `resource.duckdb`、`data/corpora/`、`data/derived/`、既存 checkpoint

## オンデマンド Dataset

`tf_gridnet.ms_snsd.ondemand@1`

| 項目 | 値 |
|---|---|
| dataset_id | `d8298f35a3f4da1fcf5a1966163716eb15460602f4767989f8ba86fbae880f92` |
| snapshot_hash | `78152930aa7b3ba5f0cfa061cc6d4afaa9c82cef016102ad9d2a9532c2dbe7a5` |
| claim | `official_recipe_scaled_down` |
| selection_uses_test | false |
| train / valid / test | 計画 25 実現値から 20 / 5。test は 5。最大クリップは train 5、test 1 |
| train build | `ad7c41d19225990083f767659b6f7324cfd70664562915ee53957bf272150932` |
| test build | `5db0e4c8638a37ff86c93670906a085e979c907789b7181aa95588d761cd0b2d` |
| 条件検査 | train 75 ストリーム、test 15。クリップしたサンプル 0。peak 0.622 と 0.525。`wav_files` 0 |
| 所要 | 計画 5.2 秒、メモリ上の再構築 0.17 秒、コマンド全体約 26 秒 |
| 来歴 | 元素材 60 本が `source`。`derived` と `unknown` は 0 |

公式規模（train 30 h / test 1 h、実現値 9180 + 305）の素材ハッシュは、`clean_limit` が無いため CLI が拒否する。登録していない。

## 公式との比較

`tests/reproduction/test_p6_ms_snsd.py` が、コーパスの `audiolib.snr_mixer` と独立実装を同じ入力で配列一致させた。測った SNR は名目の半分。10 秒未満は 0.2 秒の無音でつなぎ、長い素材は切らない。

オンデマンドの 1,000 実現値（合成音、200 クリップ）は、2 回の再構築が一致し、測った SNR が名目の半分だった。公式に seed が無いので、公式ダンプとのビット一致は主張しない。

## CPU で確認したこと

- スナップショットの全 Case について、`docs.jsonl` の長さと `frames` が一致する（小規模 Dataset と合成テスト）。
- 実スナップショットの DataLoader が chunk `(1, 48000)` を返し、同じ chunk の再構築が一致した。
- world size 1・2・4 で `plan_chunk_epoch` が通り、同じ入力の `plan_hash` が一致した。実データの更新数は 7、13、26（`num_samples=4`、global batch 4）。
- `make reproduction-build` と、`delivery` の無い設定、上限の無い設定は WAV を作らず拒否する。
- `tests/reproduction` 59 件が PASS（約 115 秒）。`tests.training.test_p5_chunk` と `test_protocol` の 19 件が PASS（約 9.5 秒）。

## 確認していないこと

- オンデマンド Dataset での Trainer の GPU ステップ、checkpoint 再開、validation。以前の GPU スモークは保存済み WAV の `tf_gridnet.ms_snsd.smoke@1` であり、この Dataset ではない。
- 公式規模の 9180 + 305 件を DB に登録すること。
- DeepFilterNet3 と SEMamba++ のオンライン抽選、TF-Locoformer の DNS 合成。仕様だけ更新し、実装はしていない。
- MP-SENet と MossFormerGAN の 16 kHz 化を学習時へ移すこと。現行は P2 の派生素材を読む。移すなら P2・P3 の再検証が要る。
