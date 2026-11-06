# P8 検証（2026-11-06）

DeepFilterNet3 のオンライン混合を、ピン留め `Rikorose/DeepFilterNet` `d375b2d8309e0935d165700c91da9de862a99c31` の `TdDataset::get_sample` に対応する式で実装した。公式の DNS4、PTDB、出荷 `dataset.cfg`、学習 RIR 集合は手元に無い。ready な Dataset は代替素材で、公式素材の Dataset は `blocked` のままである。混合 WAV は書いていない。

この pin は出荷 checkpoint の学習時コードだとは証明していない。波形のビット一致は主張しない。乱数は NumPy PCG64 で、公式の Rust `StdRng` ではない。リサンプルは `scipy.signal.resample` で、公式の rubato `FftFixedInOut` ではない。

## 総合

オンライン混合の式、分布、Trainer 接続は代替素材で通った。公式データセットの再現は未達である。SEMamba++ は実装していない。

## 登録

`data/db/reproduction.duckdb` と `data/db/resource.duckdb` は変更していない。登録先は `data/db/reproduction-p8.duckdb`、staging は `data/staging/reproduction-p8/`。スナップショットは `data/db/snapshots/<dataset_id>/` に新しく書いた。

| Dataset | 状態 | claim | dataset_id | snapshot_hash |
|---|---|---|---|---|
| `dfn3.online.substitute@1` | ready | `protocol_conformant_substitute_materials` | `ee6223405c19bef4e8f39c9609b830273c422d7ed32b17fbaad0b0554ce1328b` | `861992f6a520823bbe8fc75faf9ffa32740881c89b65eca21bc1160a1de7401b` |
| `dfn3.online.smoke@1` | ready | `protocol_conformant_substitute_materials` | `283030adc4a81a90d833338fbd2d9e8677ec387b0cd2b7f7d222efb3aa0340e8` | `dec95878ea5854a54656a7a547511cd95809bb383166af94eb402ef41bcbeca6` |
| `dfn3.online.official@1` | blocked | `official_reproduction` | `be8e03d6f583c14c5667e6a86a09f6d3cae26b510c5a1aa872bed106ee8f8f60` | 無し |

`count_check`、`list_hash_check`、`leak_check` は 3 Dataset とも PASS である。公式 Dataset はバインドが空なので、leak は空集合の PASS である。スナップショットの容量は substitute が 288,198 バイト、smoke が 49,715 バイト。中身は索引で、波形ではない。

代替素材の件数:

| 役割 | substitute train / valid / test | smoke train / valid / test |
|---|---|---|
| speech | LibriSpeech train-clean-100 200 / dev-clean 40 / test-clean 40 | 8 / 4 / 4 |
| noise | DEMAND 10 シーン × 4、4 シーン × 4、4 シーン × 4 | 8 / 4 / 4 |
| RIR | OpenSLR28 simulated small / medium / large room、40 / 10 / 10 | 4 / 4 / 4 |

ユニークな素材ファイルは 412。speech は話者ディレクトリ、noise はシーン、RIR は `smallroom/Room…` のように部屋サイズを話者 ID に含めた。split をまたぐ資産・話者・SHA-256 の重複は登録時の `leak_check` が PASS だった。

`blocking_json` は DNS4、PTDB、公式 `dataset.cfg`、公式 RIR 集合、VCTK の factor 10 とファイルリスト。DNS2020 と DNS5 は DNS4 ではない。VCTK のファイルは手元にあるが、公式の採用リストは無い。

子 Protocol `dfn3.online.audio-db` は `algorithm_change` で、親 `dfn3.online`（`official_effective`）を指す。代替 Dataset は子を、blocked の公式 Dataset は親を指す。数値の正本は `src/reproduction/ids.py` の `DFN3_ONLINE_V1`。DuckDB とスナップショットへ書き戻した `rule_json.online` は、この辞書と等しい。`configs/data/datasets/dfn3.online.yaml` は登録名の記録で、混合器は読まない。

`official_reproduction` は ready のとき全素材 SHA-256 を要求する。blocked または unknown でバインドが空なら登録できる。ready な公式再現がバインド無しになる経路は無い。

## 式

出荷 ini の SNR は `-100,-5,0,5,10,20,40`。gain は ini に無く、`train.py` の既定 `[-6, 0, 6]`。雑音本数は `uniform(2, 6)` の整数で 2, 3, 4, 5。train の合成雑音は 5%。`p_reverb` は 0.1。窓は 48 kHz、144,000 サンプル。RIR が空で `p_reverb > 0` のときはエラーにし、確率を 0 にしない。

短い speech は別発話を連結してから切り出す。零埋めしない。無音の差し替えは 64 回で打ち切る。成功した連結は数えない。雑音はゲインを掛けて足し、チャンネル数 1 で割る。本数では割らない。長さが speech 以下の雑音は、等長も含めて繰り返してから切り出す。

SNR −100 は clean を零にし、雑音を正規化する。通常の低 SNR ではない。clip ガードは clean、雑音、混合の最大絶対値が 1 を超えるとき、三つを同じ係数で縮める。残響の教師は `0.3 * dry + 0.7 * 後期抑制した畳み込み` で、元の dry RMS に戻す。そのあとゲインと clip ガードを通す。dry そのものでも、完全な残響波形でもない。

確率 0 の augmentation（speech の clipping、zeroing、air absorption、bandwidth extension、干渉話者、biquad）は乱数を消費しない。実装していない。

valid と test のサンプル種は epoch を含まない。学習の world size が 2 以上でも、検証は rank 0 が valid 全体を 1 プロセスで描画する。batch は検証用の 64 で、端数 batch を残す。Trainer はバッチごとの損失を非加重平均する。公式の `run_epoch` がバッチ平均を等しく平均するのと対応する。VoiceBank 固定ペアは 1 発話のままである。

順序 `dfn3_online_ceil_v1` は、world size 1 では `ceil(件数 / batch)` で端数を残す。world size が 2 以上のときは、rank の長さが揃うまで順列の先頭を繰り返す。同じ sample index の波形は rank に依存しない。公式の学習は単一プロセスで、このパディングは勾配の一致ではない。公式のメインスレッド RNG（seed 42、epoch で再開しない）には依存しない。

batch schedule の実効サイズは `min(schedule, 64)`。128 と 9999 は 64。`identity()` に現在の batch は入れない。epoch の `set_batch` のあと `data_state` を比べる。蓄積は 1。

## 実行した検証

コマンドは `.venvs/se/bin/python -m unittest`。GPU と `qsub` は使っていない。

| 項目 | 結果 |
|---|---|
| 連結が零埋めでないこと、雑音を本数で割らないこと、SNR と clip ガード、SNR −100、RIR 欠如のエラー、教師式の独立再実装、合成雑音の傾き、同一種の再現、valid が epoch に依らないこと | PASS |
| 分布（seed `20261106`、4,000 混合。窓は実行前に固定。雑音本数 800–1200、SNR 421–721、gain 1133–1533、残響 280–520、合成雑音率 0.038–0.062、非 noise-only の実 SNR 誤差 0.05 dB 未満） | PASS（約 115 秒） |
| world size 1 / 2 / 4 の計画（端数、同じ更新回数、重複はパディングだけ、plan hash の再現） | PASS |
| `dfn3.online.smoke@1` の実音声。3 秒、48 kHz、有限、ピークはガード後、world size 1 と 2 で同じ index が一致、valid の epoch 0 と 3 が一致 | PASS |
| Trainer。stub で 1 step 保存、再開、epoch 0 の batch 2 から epoch 1 の batch 4、検証、`data_provenance.json` の claim | PASS |
| `DeepFilterNetTask`。出荷構成（2,135,484 パラメータ）に、実現した混合の先頭 8,192 サンプルを 1 step。損失は有限 | PASS |
| 3 秒全長の出荷モデル step | NOT_TESTED（CPU で 8,192 サンプルが約 10 秒。全長は掛けていない） |
| 公式 `libdfdata.PytorchDataLoader` との波形比較 | NOT_TESTED。モジュールは import できた。ローダは HDF5 の公式 Dataset を要し、波形だけの関数は無い |
| P1〜P4（51 件）、P6・P7・P7.1・`test_batch_schedule`・`test_protocol`（28 件） | PASS |
| `test_df_validation_reduction`、`test_df_reference_spectral`、`test_df_rfft_kernel`、`test_checkpoint_atomic`、`test_mock_trainer`、`test_p5_chunk` | PASS |
| `test_df_libdf_frontend` の 2 件 | ERROR。`.venvs/se` の `libdf.DF()` が引数を取らない。P8 の変更箇所ではない |

## 公式との差

- NumPy PCG64。Rust `StdRng` のビット列ではない。一様分布の端もビット一致しない。
- `scipy.signal.resample`。rubato ではない。ゲートと 500 Hz の格子は対応する。
- epoch 順は `mix_seed("dfn3-order", ...)`。公式の seed 42 の履歴には依存しない。
- 1,440,000 サンプルを超える RIR は切る。ファイルは channel 0。短い代替 RIR はこの上限に当たらない。
- 無音の差し替えは 64 回でエラーにする。公式のループは無制限。
- world size が 2 以上のパディングは audio-db の計画である。
