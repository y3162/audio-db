# Phase 4-G: DeepFilterNet3

判定: **CONDITIONAL GO。** `DfNet`、公式損失、AdamW、DeepFilter cosine、online DFN3 の epoch 管理は Experiment Framework に接続した。CPU / FP32 で、同じ FFT 経路の旧 `DeepFilterNetTask` との損失、更新後 Parameter、AdamW state の最大絶対誤差は 0 だった。

この判定は公式 DNS4 全量の学習でも論文スコアの再現でもない。`dfn3.online.official@1` は blocked のままである。検証に使った ready な snapshot は `dfn3.online.smoke@1` で、`claim_label` は `protocol_conformant_substitute_materials` である。

条件付きにした理由は、追跡中の `src/se/models/deepfilternet3/libdf_rfft.so` がこのホストで読めないことである。ローカル再ビルドは読め、追跡ファイルは置き換えていない。GPU / NCCL の DDP は未実施である。

基準 commit は `e62e373`。作業ツリーは Phase 4-G の差分を含む。

## 1. 環境と Native Library

ホストの glibc は 2.31 である。追跡中の `libdf_rfft.so` は `GLIBC_2.33` と `GLIBC_2.34` を要求し、`ldd` はロードに失敗する。Rust は 1.76.0、Cargo は 1.76.0 で、`make df-rfft` は `build/df-rfft/release/libdf_rfft.so` を作った。Makefile は追跡ファイルと違うので置き換えなかった。

`DF_RFFT_LIBRARY` をそのローカル成果物にすると、ABI は 1 でロードできる。ファイルが無い経路では `_rfft_kernel()` は None になり、`torch.fft.rfft` に戻る。追跡ファイルをそのまま読むと `RuntimeError` で、fallback には入らない。

長さ 960 の 1 フレームで、ローカル native FFT と PyTorch fallback の最大絶対誤差は 1.1444091796875e-05 だった。これは実装差であり、損失式の変更ではない。旧 Task と新 Trainer を同じ経路で比べた最大絶対誤差は、native でも fallback でも 0 だった。fallback の 0 は native 再現ではない。

解析 FFT、ERB、unit norm は CPU 上の kernel または PyTorch である。モデルを CUDA に載せても、この DSP は CPU のままである。

## 2. Model と DSP

`DeepFilterNet3Model` は `src.se.build_model("deepfilternet3")` の `DfNet` を 1 組だけ持つ。`DFStream` は Trainer が持ち、波形ごとに `reset()` してから clean、そのあと noisy を解析する。学習は `enhance()` を呼ばない。Forward は enhanced complex spectrum、mask、local SNR、deep-filter coefficients を返す。

`df_fc_a` は state dict に残り、forward では未使用である。prepare は `find_unused_parameters: true` である。Factory と Builder にモデル名の分岐は無い。

公式幅の `--preflight` は PASS だった。`training_executed` は false。state id は data、model、optimizer、runtime、trainer である。4.67 秒。

## 3. Loss

再利用したのは既存の `_multi_resolution_spectral`、`_local_snr_db`、`_vorbis_istft` である。`DeepFilterNetTask.train_update` は呼んでいない。

- multi-resolution spectral。FFT 256 / 512 / 1024 / 2048。magnitude 500、complex 500、gamma 0.3。Hann、`normalized=True`、hop は FFT / 4。
- local SNR。重み 0.001。20 ms。clamp は `lsnr_min - 1` から `lsnr_max + 1`。MSE。
- mask、SI-SDR、ASR は足さない。

小さい合法構成（`conv_ch` 16、`nb_erb` 8、`emb_num_layers` 2）と 4,800 サンプルで、合計損失の差は 0 だった。

## 4. Optimizer

AdamW。学習率 0.001、betas `(0.9, 0.999)`、weight decay `1e-12`、`amsgrad` true、勾配クリップ 1。Factory が Optimizer を作り、Parameter はモデルと同一オブジェクトである。1 更新後の Parameter と AdamW state の最大絶対誤差は 0 だった。

## 5. Cosine Schedule

`deepfilter_cosine_schedules` を呼ぶ。式は複製していない。warmup 開始は 0.0001、基本は 0.001、最終は 0.000001、warmup は 3 epoch、総 epoch は 120、weight decay の終点は 0.01 である。

index は `epoch * steps_per_epoch + index_in_epoch` である。過去 epoch の更新数の累積ではない。`steps_per_epoch` が変わると表を作り直す。最初の更新の学習率は 0.0001 だった。weight decay が正のときだけ cosine の値で上書きする。

## 6. Batch Schedule

出荷の閾値は `scheduled_batch_size` のままである。上限 64 では epoch 0 が 16、2 が 24、5 が 32、10 以降が 64 である。128 と 9999 は 64 になる。これは VoiceBank の 10,700 発話の loader 長ではない。smoke snapshot の発話数は train 8、valid 4 で、epoch 長は現在の micro-batch から作り直す。

`SnapshotDataModule.rebind_micro_batch` は `Dfn3IndexSampler` だけを受け付ける。開いている epoch view は拒否する。幅が変わると計画キャッシュを捨てる。chunk、固定ペア、SEMamba の plan hash は変えない。world size が 1 でないとき、batch schedule 付きの学習は拒否する。これは現行 Trainer と同じ制限である。

resume は、保存された `micro_batch` 以外の identity が一致するときだけ、その幅へ戻してから identity と plan hash を比べる。幅を変えたあとの checkpoint が、YAML の初期幅の計画を再利用することはない。

smoke で幅 1 の epoch 0 は 8 更新、幅 2 はそれより少ない。epoch 0 を幅 1 で閉じた次の logical step は epoch 1、幅 2、`schedule_steps` 4、`update_index` 1 だった。`valid.loss.best.pt` を書いた。

## 7. Validation

epoch の途中では走らない。epoch 末に rank 0 が valid 全体を見る。モデルは `eval()`、Optimizer は更新しない。発話ごとの損失を等重みで平均する。短い最後の batch を 1 件として他の batch と同じ重みにはしない。RNG と学習 cursor は戻す。

best は validation loss の厳密な改善だけである。同点は前の checkpoint を残す。early stopping は非改善が `early_stopping_patience` に達すると止まり、改善するとカウンタを 0 に戻す。宣言 YAML の patience は 25 である。

## 8. Checkpoint / Resume

形式は `audio-db-experiment-v1`。`provenance_sha256` は None である。smoke の 1 step のあと別のプロセスで 2 step まで再開し、metrics は 2 行だった。保存するのは generator、optimizer、cosine の `schedule_steps`、trainer の epoch 境界、data の cursor と plan hash、RNG である。rank ごとの損失値は trainer state に入れない。入れると DDP の checkpoint が rank 不一致になる。

## 9. CPU 数値比較

比較経路はローカル native kernel と、ライブラリを欠いた PyTorch fallback の両方である。どちらも旧 Task と新 Trainer の差は 0 だった。比較したのは DSP features、合計損失、クリップ後の更新、Parameter、AdamW state、学習率、weight decay である。公式幅の DfNet でこの 1 更新比較はしていない。公式幅は preflight の構築だけである。

## 10. GPU 検証

2026-11-09。物理 GPU 1。小さい構成、4,800 サンプル、1 更新、CUDA / FP32。`deterministic` は false。損失は 8.998255729675293 で有限だった。解析 FFT は CPU のままである。CPU とのビット一致は主張しない。NCCL は使っていない。

## 11. DDP

CPU / Gloo、world size 2、`find_unused_parameters` true、logical step 1。checkpoint の `parameter_ranks_agree` は true、`world_size` は 2 である。BatchNorm の running stats は rank の発話が違うので forward の直後には揃わない。保存の前に rank 0 の buffer を broadcast する。これは次の DDP forward が rank 0 から配るのと同じで、SyncBatchNorm でも二つの batch の平均でもない。

## 12. Dataset Profile

`dfn3.online.smoke@1` は ready、`official_faithful`、`protocol_conformant_substitute_materials` である。48 kHz、144,000 サンプル、教師は `official_mixed_target`、順序は `dfn3_online_ceil_v1`、`group_size` は 1、短い最終 micro-batch は残す。DataModule の 1 波形と `render_mixture` の最大絶対誤差は 0 だった。`dfn3.online.official@1` は blocked のままである。`dfn3.online.substitute@1` も ready だが、この Phase の実行には smoke を使った。

## 13. 回帰

変更前。`CUDA_VISIBLE_DEVICES=""`。`DF_RFFT_LIBRARY` は空である。

| Suite | 件数 | 結果 |
|---|---:|---|
| `tests/training` | 297（906.045 秒） | failure 1、error 6、skip 12 |
| `tests/reproduction` | 135（454.696 秒） | error 1、skip 2 |

失敗は DeepFilterNet3 の追跡 `libdf_rfft.so` が要求する `GLIBC_2.33` である。`tests/experiments` の同じ実行は 171 件、855.027 秒で、failure 1、skip 1 だった。その 1 件は、実装の途中の `checkpoint.py` を別プロセスが読んだ `NameError` である。完成した関数を読む再実行とは別である。

変更後。`CUDA_VISIBLE_DEVICES=""`。`DF_RFFT_LIBRARY` は `build/df-rfft/release/libdf_rfft.so`。追跡ファイルは置き換えていない。

| Suite | 件数 | 結果 |
|---|---:|---|
| `tests/experiments` | 181（875.838 秒） | OK、skip 1 |
| `tests/training` | 297（947.161 秒） | OK、skip 12 |
| `tests/reproduction` | 135（454.934 秒） | OK、skip 2 |

変更前の `tests/training` の failure 1 と error 6、`tests/reproduction` の error 1 は、このローカル成果物を読むと消えた。変数を外すと追跡ファイルは引き続き読めない。`tests/experiments` は 171 に Phase 4-G の 10 件を足した 181 で、skip 1 は Phase 2-C の CUDA 非表示のままである。新規の Failure、Error、Skip は無い。

## 14. 未解決の差分

- 追跡中の `libdf_rfft.so` は glibc 2.31 で読めない。ローカル再ビルドは未インストールである。
- native FFT と PyTorch FFT の 1 フレーム差 1.1444091796875e-05。
- 公式 DNS4 全量、論文スコア、120 epoch の完走。
- VoiceBank 10,700 発話の `steps_per_epoch` 668 を、この online snapshot の epoch 長として使っていない。
- GPU / NCCL の DDP。CUDA は world size 1 の 1 更新である。
- BatchNorm は SyncBatchNorm にしていない。
- 公式幅 DfNet の 1 更新の数値比較はしていない。一致を測ったのは小さい合法構成である。
- SEMamba++ は未接続である。
