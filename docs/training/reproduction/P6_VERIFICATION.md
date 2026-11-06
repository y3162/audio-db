# P6 検証（2026-11-06）

ESPnet 公開 SE recipe（commit `0d136c5`、`egs2/ms_snsd/enh1`）の MS-SNSD 混合を、学習時に再構築する。論文実験の再現ではない。全量 WAV は作っていない。

## 公式規模のオンデマンド登録（2026-11-06）

対話プロセス。qsub は使っていない。

| 項目 | 値 |
|---|---|
| コマンド | `.venvs/se/bin/python -m src.reproduction.cli ms-snsd --config configs/data/builds/ms_snsd_tfgridnet.yaml --step register` |
| 計画 | 再計算の結果、train の実現 9,180、test 305。以前に中断した WAV 生成の計画件数と一致した。式の一致であり、公式ダンプとのビット一致ではない |
| 分割 | train 7,345 / valid 1,835 / test 305。`count*8/10` のあと、1 クリップの 5 SNR を割らない |
| 監査 | train は 1,000 件をメモリで混合（`render_scope=sample`）。test 305 件は全件。`wav_files=0`。train のクリップサンプル 2,191、test 2,759 |
| Dataset | `tf_gridnet.ms_snsd.espnet_recipe@1`。snapshot `a69a477af70bc43dbcabdafbd5c97ec8a26aac4debd51158079545de1e4f0757` |
| 監査状態 | `count_check` PASS、`list_hash_check` PASS、`leak_check` WARN（話者 ID が無く、資産 SHA-256 の重複は空） |
| chunk 計画 | 公式 train 7,345 件の長さ列。`num_samples=5000`、seed 17、chunk 48,000。world 1 は updates 7,915、hash `424cf2d605938c6108df4b591dfde25bd71ea958123f0067f26e7d78c192bb6e`。world 2 は 15,824、`995a33f69abaee36766a3494b485ba9d1b7f481f165b7d0d9a81774ee7543c2a`。world 4 は 31,597、`46d57c9d0f5224f3590e84dfed077d04a4ddbdaa7232b40a467ff43f56b0ec87`。音声は混合していない |

元素材は clean_train 23,075、noise_train 128、clean_test 1,100、noise_test 51。既に登録済みの同じファイルは行を増やさず、新しいファイルだけを足した。

## 停止した事前生成（履歴）

この節は、オンデマンドへ切り替える前に中断した WAV 生成である。生成物は削除済みで、現在の学習経路はこれを読まない。

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

公式規模は上の節で登録済みである。`source_scan: full` が無い上限なし走査だけを CLI が拒否する。

## 公式との比較

`tests/reproduction/test_p6_ms_snsd.py` が、コーパスの `audiolib.snr_mixer` と独立実装を同じ入力で配列一致させた。測った SNR は名目の半分。10 秒未満は 0.2 秒の無音でつなぎ、長い素材は切らない。

オンデマンドの 1,000 実現値（合成音、200 クリップ）は、2 回の再構築が一致し、測った SNR が名目の半分だった。公式に seed が無いので、公式ダンプとのビット一致は主張しない。

## CPU で確認したこと

- スナップショットの全 Case について、`docs.jsonl` の長さと `frames` が一致する（小規模 Dataset と合成テスト）。
- 実スナップショットの DataLoader が chunk `(1, 48000)` を返し、同じ chunk の再構築が一致した。
- world size 1・2・4 で `plan_chunk_epoch` が通り、同じ入力の `plan_hash` が一致した。小規模データの更新数は 7、13、26（`num_samples=4`、global batch 4）。公式規模の計画は上の表。
- `make reproduction-build` は削除した。`delivery` の無い設定と、`source_scan: full` の無い上限なし走査は WAV を作らず拒否する。
- `tests/reproduction` 61 件が PASS（約 136 秒）。`tests/training` の `test_p5*.py` 15 件が PASS（約 11 秒）。

## CPU Trainer

`tf_gridnet.ms_snsd.ondemand@1`、device cpu、virtual world 2、seed 17。公式規模の 7,345 件では回していない。

| 実行 | 結果 |
|---|---|
| `num_samples_per_epoch=4`、max_steps 2 | `segment_steps=1` で step 1 を保存して停止し、その checkpoint から再開して step 2 まで進んだ。generator loss は 11830.85986328125 から 8750.265625。checkpoint は `step_00000001.pt` と `step_00000002.pt`。plan hash `91761e0677bae0ff39fe8c8c68663339c4609778334d06580550d90392a73be2`。epoch は 13 updates あるので、2 step では validation しない |
| `num_samples_per_epoch=1`、max_steps 3 | updates は 3。step 3 の loss 7935.341796875、`validated: true`。validation の generator / mr_l1_tfd は 9015.006917317709（count 24、status ok）。SI-SNR は 16.337975629694238（status ok）。PESQ と STOI は `mr_l1_tfd` では `na`。checkpoint `data/results/se-repro/p6-tfgridnet-ondemand-cpu-epoch/step_00000003.pt`。終了後の data_state は epoch 1、update 0、plan hash `af9d05ecbfb90b4587083ee6cafe3660762e689959a748f8358048f3c4f141b7` |

chunk 経路は prefetch スレッドを使わない。同じ chunk の再構築が一致することは単体テストで確認した。

## 確認していないこと

- オンデマンド Dataset での GPU ステップ。公式規模 Dataset での Trainer step も未実施。
- DeepFilterNet3 と SEMamba++ のオンライン抽選、TF-Locoformer の DNS 合成。仕様だけ更新し、実装はしていない。
