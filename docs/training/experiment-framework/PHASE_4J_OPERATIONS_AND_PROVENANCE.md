# Phase 4-J: 運用・provenance・分散 checkpoint

判定: **CONDITIONAL GO。** 6 モデルの損失、Optimizer 順、DataModule の生成は変えていない。このホストでは、追跡中の `libdf_rfft.so` を置き換えずに、読めるローカル再ビルドを自動で使う。新規の実験 checkpoint は `audio-db-data-provenance-v1` に結ぶ。`provenance_sha256` が None の旧ファイルは来歴なしのまま再開できる。公式データ全量の学習と論文スコアの再現はしていない。

無条件 GO にしない理由は、追跡バイナリがこの glibc 2.31 では読めないこと、DeepFilterNet3 の `batch_schedule` 付き複数 GPU を実装していないこと、CUDA の bit 一致を主張しないことである。

## 1. 変更ファイル

学習式は変えていない。足した接続は次である。

- `src/se/models/deepfilternet3/dsp.py`。ライブラリ選択。
- `src/experiments/provenance.py`。既存の provenance 契約への接続。
- `src/experiments/checkpoint.py`。保存と resume の検査。
- `src/experiments/signals.py` と各 Trainer の停止位置。`SIGTERM` を既存の停止判定で見る。
- `src/experiments/mossformergan.py`。world size 2 以上では、保存前に蓄積窓のログを rank 0 のリストへ揃える。勾配は変えない。
- `src/experiments/deepfilternet3.py`。`rfft.json` を rank 0 が書く。
- `scripts/setup/doctor.py`、`scripts/setup/env_roles.sh`、`scripts/experiments/show_status.py`。
- `tests/experiments/test_phase4j_operations.py` と、既存試験の provenance 期待、`tests/experiments/test_cuda_amp.py` の NCCL。

## 2. Native Library の構築・配備

追跡ファイル `src/se/models/deepfilternet3/libdf_rfft.so` は置き換えていない。`make df-rfft` と `make setup` の `build_df_rfft` は、内容が違うとき追跡ファイルへコピーしない。成果物は `build/df-rfft/release/libdf_rfft.so` で、Git 管理外である。

選択順は次である。

1. `DF_RFFT_LIBRARY` があればそのパス。
2. 無ければ追跡ファイルを試し、ABI 1 で読めればそれを使う。
3. 追跡ファイルが存在して読めず、ローカル成果物が ABI 1 ならそれを使う。
4. 追跡ファイルが読めず、ローカル成果物も無ければ `RuntimeError`。PyTorch fallback にはしない。
5. ファイル自体が無いときだけ、従来どおり `torch.fft.rfft` へ戻る。`DF_RFFT_REQUIRED=1` の欠落は `FileNotFoundError`。

2026-11-10。このホストの glibc は 2.31。`DF_RFFT_LIBRARY` を外したプロセスの記録は次である。

| 項目 | 値 |
|---|---|
| source | `local_build` |
| kernel | `native` |
| abi | 1 |
| sha256 | `e1eb0815e5242191e98ca5785ca2a868a01b421d91be7b0fc6e79b994dc77e77` |
| path | `build/df-rfft/release/libdf_rfft.so` |
| 追跡ファイル | `GLIBC_2.33` が無く、ロードしない |

`make doctor` の `df_rfft` は、環境変数を外した起動で source、ABI、SHA-256、有限スペクトルを見る。不正なバイト列の共有ライブラリは `RuntimeError` で、fallback にしない。欠落パスを指定したときは従来の fallback のままである。

別ホストでは `make df-rfft` または `make setup PLATFORM=tsubame` / `shinoda_lab` を実行する。Cargo が無いときは `build_df_rfft` が失敗し、doctor も native を PASS にしない。

## 3. 環境依存性

この測定の Python は `.venvs/se`、torch 2.6.0+cu124 である。NCCL は物理 GPU 6 と 7。GPU 0、3、4、5 は使っていない。DeepFilterNet3 の学習試験は、上のローカル成果物を環境変数なしで選んだ。

`rfft.json` は DeepFilterNet3 の実験ディレクトリに、実際に選んだ kernel を書く。他モデルには書かない。

## 4. Provenance 形式

形式は既存の `audio-db-data-provenance-v1` である。生成は `src.training.provenance.build_provenance` と `publish_provenance`、ハッシュは `canonical_bytes` を使う。新しい schema は作っていない。

`SnapshotDataModule.lineage()` があるときは、その素材、set、claim、profile、snapshot hash を入れる。pipeline は `reproduction`。DeepFilterNet3 の `align_scheduled_batch` が変える `micro_batch` は checkpoint の dataset identity に残り、provenance の run identity からは外す。幅が変わったあとの保存が、同じ run の来歴を拒否しないためである。`lineage()` が無い `TensorPairData` などは pipeline `experiment`、materials は空、limitations に「素材 hash が無い」と「ファイルがあるだけでは独立ではない」を書く。素材 hash は推測しない。

rank 0 だけが `data_provenance.json` を排他作成する。同じ run identity の再保存はバイト列を書き換えない。別の identity は `CheckpointError` で、完成済み `.pt` は置き換えない。

## 5. Checkpoint Hash 契約

形式は `audio-db-experiment-v1` のままである。必須キーは足していない。`provenance_sha256` は、新規保存ではディレクトリ JSON の SHA-256 である。sidecar は `step_NNNNNNNN.data_provenance.json` で、同じバイト列である。

保存順は、ディレクトリ JSON、一時 `.pt` の検証、sidecar、最後に `.pt` の replace である。replace の前に失敗した一時ファイルは捨てる。replace 自体が失敗した一時ファイルは残し、直前の完成 `.pt` は残す。複数ファイルを一つの原子操作にはできない。

Resume は sidecar とディレクトリ JSON の両方が要る。欠落、ハッシュ不一致、sidecar 間の不一致、破損、dataset identity の不一致は拒否する。Initialize はソース側のハッシュを検査し、新しい実験の dataset 一致までは要求しない。来歴を新しい run の証明としてコピーしない。

## 6. 旧 Checkpoint 互換性

`provenance_sha256 is None` は旧実験である。Resume と Initialize は来歴ファイルを要求しない。後から同じディレクトリへ `data_provenance.json` を書いても、`judge_experiment_checkpoint` はその旧ファイルを UNKNOWN のままにする。既存ファイルの一括書き換えはしていない。

試験は、1 step の新規 checkpoint からハッシュを外して JSON を消し、2 step まで再開した。旧ファイルの `provenance_sha256` は None のままで、metrics は logical step 1 と 2 が 1 行ずつだった。

## 7. 分散 State 比較

`CheckpointManager.save` は保存前に `_assert_replicated_states` を呼ぶ。比較対象は登録済み state の dtype、shape、値である。module parameter、buffer、optimizer、scheduler、trainer、data、runtime の GradScaler を含む。RNG は rank ごとに集めるだけで、値の一致検査には入れない。

不一致と、比較中の例外は、完成 `.pt` を replace しない。試験は weight、buffer に加え、optimizer、scheduler、trainer の片 rank 改変で `ranks disagree` を確認した。1 個の float32 テンソル 16 バイトの不一致検出は 2.22e-4 秒だった。これは公式幅の通信量ではない。公式幅の全 state gather は測っていない。比較を hash や reduce へ置き換えていない。検出を弱めないためである。

`parameter_ranks_agree: True` は、この比較を通過したあとに書く。フラグだけを後から読んで一致の証明にはしない。

## 8. CUDA / NCCL

2026-11-10。物理 GPU 6 と 7。PyTorch 2.6.0+cu124。小さい構成。公式幅ではない。

MossFormerGAN。`num_channel` 64、長さ 2,400、`deterministic: false`。world size 2。

- 全 rank 成功。accumulation 2、logical step 1。SyncBatchNorm は 0 より大きい。parameter と buffer の SHA-256 は rank 間で一致した。
- 片 rank の PESQ 失敗は両 rank が backward しない。logical step は 0、失敗 micro-batch は 2、`average_calls` は 0。data epoch は 1 へ進む。
- 成功、失敗、成功。updates 6、accumulation 2。logical step 2、失敗 2、成功 4。`official_step` は logical step より大きい。
- 部分蓄積。1 micro-batch のあと `step_00000000.pt` を書き、次の generator step の注入失敗で止めた。別プロセスの resume で logical step 1、`accu_count` 0、rank の parameter と buffer は一致した。保存前に蓄積窓のログを rank 0 へ揃えないと、`window_discriminator` の rank 差で完成ファイルを書けなかった。

SEMamba++ fused。`hid_feature` 8、`num_tfmamba` 1、2,048 サンプル、accumulation 1。`configured=fused`、`executed=fused`。1 step で止めて別プロセスで 2 step まで再開した。generator、mrd、cqt、両 optimizer、両 scheduler、`rng_by_rank` がある。連続 2 回の自己再実行差は 4.887e-4、再開差は 4.240e-4 だった。再開差は自己再実行差より小さい。bit 一致ではない。`reflection_pad1d` の CUDA backward は決定的でない。

DeepFilterNet3 の `batch_schedule` は world size 1 以外を `RuntimeFailure` のままにする。スケジュール幅を rank へ割ると、rank ごとのサンプル数、epoch の plan hash、cosine の index が公式の 1 プロセス規則と別物になる。拒否を外して試験だけを通していない。

MP-SENet、TF-GridNet、TF-Locoformer の既存の短い NCCL は、このセッションでは繰り返していない。

## 9. 中断・再開

`SIGTERM` はフラグにする。各 Trainer は、これまで `stop_after_logical_steps` を見ていた位置で、全 rank の論理和を見る。GAN の 2 optimizer の途中では保存しない。最後の完成 checkpoint から再開する。

`InterruptProbe` で、step ファイルが出たあと `SIGTERM` を送った。status は `INTERRUPTED`、`complete.json` は無い。同じ YAML で `step_00000001.pt` から再開し、metrics の logical step は 1 から 30 まで重複が無かった。6 Trainer の `_stop_requested` は、フラグが立っているとき True だった。モデルごとの長い `SIGTERM` は、この短い試験とは別に、既存の `stop_after_logical_steps` が同じ位置で止まる。

`max_logical_steps` への到達は `COMPLETED` と `complete.json` である。checkpoint 間隔が広く、epoch の途中で 1 step だけ終わる実行は `.pt` を書かない。その違いは変えていない。

## 10. 学習運用

入口は既存の CLI である。

```bash
.venvs/se/bin/python -m src.experiments.cli --config CONFIG --preflight
.venvs/se/bin/python -m src.experiments.cli --config CONFIG
.venvs/se/bin/python -m src.experiments.cli --config CONFIG --resume CHECKPOINT
.venvs/se/bin/python scripts/experiments/show_status.py DIRECTORY
```

状態は `status.json`、完了は `complete.json`、失敗は `FAILED` とプロセスの stderr である。`show_status.py` はディレクトリを書き換えない。新しいジョブ管理フレームワークは作っていない。30 分以上の本学習は開始していない。

TSUBAME と shinoda_lab は `make setup PLATFORM=...` の既存経路である。`build_df_rfft` は両方の `install_roles` から呼ばれる。

## 11. ログ

`metrics.jsonl` が正本である。再開は保存済み logical step より後の行を捨てる。MossFormerGAN の skip は `skip_log.jsonl` で、logical step が進まない失敗でも data cursor は進む。全モデルに無い TensorBoard や best ファイルは必須にしていない。SEMamba++ に `best_generator.pt` は無い。

## 12. 回帰テスト

`DF_RFFT_LIBRARY` は外した。選択された library は第 2 節のローカル再ビルドである。`CUDA_VISIBLE_DEVICES=""`。

```bash
.venvs/se/bin/python -m unittest discover -s tests/experiments -v
.venvs/se/bin/python -m unittest discover -s tests/training -v
.venvs/se/bin/python -m unittest discover -s tests/reproduction -v
```

Phase 4-I の基準は experiments Ran 209 / skip 1、training Ran 297 / skip 12、reproduction Ran 135 / skip 2 である。

2026-11-10 01:53–02:37（JST）。上の 3 コマンドを、`DF_RFFT_LIBRARY` を外し、`CUDA_VISIBLE_DEVICES=""` で順に実行した。選ばれた kernel は `local_build`、ABI 1、SHA-256 `e1eb0815e5242191e98ca5785ca2a868a01b421d91be7b0fc6e79b994dc77e77` である。

| suite | 結果 | 基準との差 |
|---|---|---|
| experiments | Ran 220、OK、skipped=1、1069.436s、exit 0 | Ran は 209 から 220。増分 11 は `tests/experiments/test_phase4j_operations.py`。skip は 1 のまま |
| training | Ran 297、OK、skipped=12、936.474s、exit 0 | 基準と同じ |
| reproduction | Ran 135、OK、skipped=2、600.565s、exit 0 | 基準と同じ |

失敗、エラー、新規 skip は無い。

## 13. 6 モデル別の運用可能範囲

| モデル | 範囲 |
|---|---|
| MP-SENet | 既存の CPU 一致と短い NCCL。新規 checkpoint は provenance を持つ。公式全量は含まない |
| MossFormerGAN | 上に加え、このセッションの小さい NCCL で PESQ skip の混在、蓄積 2、部分蓄積の resume。公式幅ではない |
| TF-GridNet | 既存の小さい NCCL 1 step。公式幅の NCCL ではない |
| TF-Locoformer | 同じ。研究用 validation profile は `tf_locoformer.dns.accept_recipe.research_valid_fixed@1` |
| DeepFilterNet3 | 同じ FFT 経路の CPU 一致。追跡 library はこのホストで読めず、ローカル再ビルドを自動で使う。`batch_schedule` の複数 GPU は未実装。`dfn3.online.official@1` は blocked |
| SEMamba++ | fused の NCCL 再開をこのセッションで測った。自己再実行差の内側だった。bit 一致ではない。`semambapp.gsr.official@1` は blocked。best generator は無い |

## 14. 未解決事項

- 追跡中の `libdf_rfft.so` を、この glibc で読める成果物へ差し替える配備。追跡ファイルの無断置換はしない。
- DeepFilterNet3 の `batch_schedule` を付けた複数 GPU。公式の幅、plan hash、cosine index を同時に保つ設計が要る。
- 公式幅モデルでの rank 比較の通信量。検出能力を落とす置換はしていない。
- 公式データ全量の本学習、中断、ログ。30 分以上の投入は別の承認が要る。
- world size を変える resume、AMP BF16、複数 Optimizer の AMP FP16、FSDP、sharded checkpoint。
- 実評価セットとの突合。`judge_checkpoint` は呼んだ。Official Evaluation 自体は未実装のままである。

## 15. Phase 4 全体の最終判定

**CONDITIONAL GO。** 6 モデルは同じ Experiment Framework から構築でき、検証した CPU 条件では現行 Task と一致し、新規 checkpoint はデータ来歴のハッシュを持つ。このホストの DeepFilterNet3 は、追跡バイナリではなくローカル再ビルドで native kernel を使う。CUDA の bit 一致、公式全量、論文スコア、`batch_schedule` の複数 GPU はこの GO に含めない。
