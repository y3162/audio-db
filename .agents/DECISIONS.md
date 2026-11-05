# 設計判断

変更履歴ではなく、決定と理由の記録。状態は 2 つに分ける。

- **実装済み**: コードとテストにある。
- **採用決定済み・未実装**: 方針は決まったが、コード・スキーマには無い。実装済みとして扱わず、現状の記述（`PROJECT.md`、`DATA.md`、`docs/`）も書き換えない。実装したら「実装済み」へ移す。

## 採用決定済み・未実装

（なし）

## 実装済み

1. **評価の入力は DuckDB の名前付きセットだけ。Manifest は登録時にしか読まない。** 同じ名前なら入力・順序・混合条件が同じになり、Manifest や元ファイルの変更に評価が左右されない。
2. **`make db-init` と `make db-ingest` を分ける。`db-init` は `corpora.audio_files` に触れず、前後の指紋で確認する。** 評価セットの登録で、コーパス由来のテーブルを壊さない。
3. **合成混合は `rms_v1`。条件を case に保存して再生成する。** 区間全体の平均二乗電力の比で、VoiceBank-DEMAND・URGENT の手順でも、ITU-T P.56 でも、VAD でもない。seed は SHA-256 で作り（Python の `hash()` は使わない）、雑音の対応はセットの seed と path で決める（実験 seed では変わらない）。SE を変えても同じ入力で比べるため。
4. **WER は micro 集計と `wer-norm-v1`。出せない値は `na` と理由で残し、0 にしない。** 部分音声と全文の組は無効にする。不正な WER を作らないため。
5. **CHiME3 の正解は公式注釈の `prompt`。CH0・real・simu・`tr05_org` を clean 参照にしない。** 時間整合した clean 参照が無いのに PESQ / STOI を出さないため。
6. **SE と ASR は独立モジュール。評価は共通 API だけを呼ぶ。** ASR は使う系列のバックエンドだけを import し、環境も `.venvs/se` と分ける（NeMo の依存が合わない）。
6a. **プラットフォームはセットアップの入力だけにする。** 仮想環境は `.venvs/se`、`.venvs/asr`、`.venvs/official`、`.venvs/semambapp`、公式 `libdf` 用の `.venvs/deepfilternet` である。`tsubame` と `shinoda_lab` は環境名に入れない。旧称 lab と `.venvs/lab` は SHINODA-LAB を指していた。`make setup` は DB を初期化しない。依存の固定は `requirements/*.lock.txt`、公式ソースと重みは `requirements/official-assets.toml`。
7. **推論 API は推論専用。** `transcribe()` と `enhance()` に学習や勾配の経路を持たせない。
8. **独立 SE 実装は、公式コードに実行時依存しない。一致は最終波形の最大絶対誤差で判定する。** MP-SENet、TF-GridNet、DeepFilterNet3、MossFormerGAN、SEMamba++ は `docs/se/VALIDATION.md`。TF-Locoformer の `dns` は standalone separator と CPU で差 0、公開重みの GPU 波形差は 1.11e-9（`docs/training/OTHER_MODELS.md`）。未検証の範囲もそこに残してある。
9. **ASR は、ASR 用に fine-tune されたチェックポイントを、そのデコーダのまま使う。** 自己教師ありだけのチェックポイントは拒否する。Parakeet は TDT / CTC / RNN-T をモデル ID のデコーダのまま使い、Whisper は `transcribe` だけにする。系列やデコーダの取り違えを防ぐため。
10. **結果は run 単位のディレクトリに JSON / JSONL で保存し、`results.jsonl` から再集計する。** 書き込みは原子的で、再開は成功レコードだけを再利用する。DuckDB への登録は、そのファイルを `register_run` で後から載せる。
11. **DB はレシピ JSON を管理し、音声生成は Python が解釈する。** 操作は順序付きの `additive_noise`、`rir`、`peak_normalize`（いずれも version 1）。未知の操作と version は拒否する。詳細は `docs/db/RECIPE.md`。
12. **`case_id` は正規化レシピの SHA-256 である。** キー順はソートし、浮動小数は小数 6 桁へ丸める。`case_id` 自身はハッシュに入れない。同じレシピは再構築しても同じ ID になる。
13. **登録データは `data/staging/` に保存してから DuckDB へ入れる。** 参照は `case_id` と `path` で持ち、rowid や採番に依存しない。同じステージングから作った DB は、登録順が違っても内容が一致する。
14. **旧 `eval.cases` の固定列を実行時に読まない。** 一度きりの変換は `src.data.migrate_legacy` で、ステージングと検証用 DB を作る。本番 `resource.duckdb` は、その検証のあとレシピスキーマへ置き換えた。旧ファイルは `data/db/backups/resource-legacy-20261031.duckdb` に残す。`migrate_legacy` は既存のステージングと再構築 DB を上書きしない。
15. **SE の学習は `src/training/` に置き、`enhance()` と `transcribe()` には入れない。** 最初のモデルは MP-SENet。Generator は推論実装をそのまま使い、損失と Metric Discriminator は学習側にある。公式ツリーは実行時に import しない。
16. **Global batch は厳密に保つ。** `micro_batch_size_per_gpu` は上限で、その範囲の最大の micro batch と整数の蓄積回数で global batch を作る。作れない構成はエラーにする。Global step は Optimizer 更新の回数である。MP-SENet の ExponentialLR は、公式と同じく、そのエポックの使える micro-batch を終えたときだけ進む。蓄積回数では進まない。DDP の all-reduce は forward 時に決まるので、`no_sync` は micro-batch の forward を包む。最後の micro-batch だけ同期する。SEMamba++ の損失は平均なので、micro-batch の backward は蓄積回数で割る。割らないと公式の batch 平均ではなく、micro-batch 勾配の総和になる。蓄積が 1 のときは Generator を 1 回だけ forward し、その波形を Discriminator と Generator の両方に使う。蓄積が 2 以上のときは全 micro-batch の計算グラフを保持しない。Generator に dropout は無いので、Discriminator を更新したあと micro-batch ごとに forward し直す。その forward は `accumulation_scope` の中で行い、遅延 backward の手動 all-reduce は使わない。参照の selective scan は学習時だけ再計算し、3 秒の逆伝播を 16 GB に収める。推論は再計算しない。
17. **学習の乱数は用途ごとに分ける。** 初期化、dropout、サンプル順、切り出しは独立である。切り出しと順序はサンプル ID と seed から再現し、消費順には依存させない。`deterministic: true` では決定的アルゴリズムを有効にし、決定的に計算できない CUDA 逆伝播は非決定的カーネルへ戻さず、同じ順伝播値の別実装にする。DeepFilterNet3 の spectral loss はそれ以外では公式の反射パディングを使う。決定的 CUDA の代替実装は勾配の加算順が変わり得るため、同一実装として扱わない。

   MP-SENet の STFT も CPU と通常の CUDA では公式の native centered reflection を使い、厳密な決定的 CUDA だけ明示反射にする。反射の実装が異なると、forward が一致しても backward の加算順が変わり、AdamW の更新が分岐するためである。
18. **学習データは名前付きセットのレシピだけを使う。** 教師が定義できないケースと、LibriSpeech test-clean・CHiME3 et05 を含む評価用セットは拒否する。ケースの識別はレシピハッシュで、採番 ID ではない。本番 `resource.duckdb` は評価セットの本番ファイルなので、学習からは名前で拒否する。
19. **SE の学習タスクは `src/training/se/` に分け、共通 Trainer に損失を書かない。** TF-Locoformer の `dns` は AdamW と `mr_l1_tfd`（時間重み 0.5、勾配クリップ 5）で、standalone separator との CPU 1 step の差は 0。 MP-SENet は公式ステップとの数値比較まで行った。AdamW 更新後の重み差は、初回ステップの閉形式で説明できる。TF-GridNet の学習損失は ESPnet の `mr_l1_tfd` である。SEMamba++ の参照ツリーはコピーしていない。Discriminator は公開仕様の multi-scale sub-band CQT（`nnAudio` の `CQT2010v2`）と 3 解像度 STFT を独立に実装している。参照コンストラクタが層へ渡していない幅 32 は使わず、実行される幅 128 に合わせる。MossFormerGAN の Discriminator は公開版由来の NISQA AdaptCNN、2 層 Conformer、attention pool と LearnableSigmoid を `src/training/se/mossformer_discriminator/` に持つ。DeepFilterNet3 の学習損失は出荷 `config.ini` の multi-resolution スペクトルと local SNR で、mask と SI-SDR は 0 である。`scheduler.name: deepfilter_cosine` のとき、YAML の `steps_per_epoch` と `epochs` で出荷の per-step cosine（warmup 3 epoch、学習率 1e-4 から 1e-3、下限 1e-6、weight decay は 1e-12 から 0.01）を Optimizer 更新の前に適用する。ステップ数はデータローダから推測しない。スモーク YAML は 8 ケース・batch 1 なので `steps_per_epoch: 8` である。
21. **標準コーパスは `src/data/corpora_catalog.py` の 14 件だけをリンクし、空の DB を `db-bootstrap` で作る。** DNS2020 と MS-SNSD は汎用索引で、初期カタログに含まれる。追加挿入の別経路は持たない。 VoiceBank-DEMAND の学習分割は公式 28 話者ファイルを話者単位で train 10,700 / validation 872（p286、p287）にし、公式 test 824 とは話者が重ならない。
20. **独立 SEMamba++ は参照 scan を既定とし、明示指定で外部 CUDA 融合演算を使う。** `scan_backend: fused` は CUDA 上で外部 `mamba-ssm` の `mamba_inner_fn` と Triton RMSNorm だけを使う。PyTorch 畳み込みのあと selective scan だけを融合する経路は、公式 fast path と backward がずれて学習が離れるので使わない。`mamba_inner_fn` の `xz` は公式と同じ非連続 stride `(time, batch * time, 1)` の view にし、contiguous コピーはしない。ネットワークと checkpoint キーは独立実装のまま保つ。CPU は reference に戻る。20 step の勾配スナップショットは公式の再実行でも反転し得る。学習の同等性はその再実行差と損失・重みで判断し、融合版の存在だけを長時間学習後の品質一致とみなさない。
23. **MossFormerGAN の AdamW は、公開 YAML の `weight_decay: 1e-5` ではなく、コンストラクタが渡さないときの PyTorch 既定 0.01 である。** 識別器の学習率は生成器の 2 倍。betas は (0.9, 0.999)、eps は 1e-8。検証損失が直前の記録より下がらない回が 5 回になると、最後に改善した checkpoint の重みと Optimizer を戻し、生成器の学習率だけを半分にする。その回数は半分にしたあとリセットしない。10 回で学習を止める。悪化した損失は記録上の最良値を上書きする。この回数と最良 snapshot は独立実装の checkpoint に残す。公開 Solver の再起動は `val_no_impv` と `best_val_loss` を初期値に戻す。
24. **H100 MIG 4g.47gb の SEMamba++ 本試験は micro 4 × 蓄積 2 である。** 3 秒・公式幅の micro 8 は 45.56 GiB を確保したあと discriminator で OOM した。長さ 8,000 の fused CUDA では batch 8 が載り、peak allocated は 9,610,847,232 バイトだった。同じ初期重みで batch 8 の自己再実行は音声最大差 0、generator 勾配の相対 L2 は 2.90e-4、更新後重みの相対 L2 は 1.39e-5 だった。micro 4 × 蓄積 2 は音声最大差 0.001338、generator 勾配の相対 L2 0.0282、重みの相対 L2 2.01e-4 で、自己再実行より大きい。勾配 norm は約 16.00 と 16.02 で、蓄積回数で割る縮尺は合っている。差は fused kernel の batch 形状に再現する。研究用本格学習はこの差を明示した micro 4 × 蓄積 2 で、`configs/training/semambapp_seed17.yaml` を使う。公式の 3 秒・1 forward の batch 8 とは別判定にする。MP-SENet の検証は hop 倍数へ切り、更新経路の長さ検査は変えない。
25. **学習入力の先読みは、crop と順序の定義を変えない。** `data.load_workers`（既定 4）は発話の読み込みをスレッドで並べ、`data.prefetch`（既定 true）は次の Optimizer 更新分を GPU 更新と重ねる。切り出しとサンプル順は case id・seed・epoch の関数のままである。MP-SENet と MossFormerGAN の wideband PESQ は、行が 4 以上のとき spawn したプロセスで同じ `pesq` を行順のまま計算する。モデル、損失、batch、Optimizer、STFT、selective scan は変えていない。`torch.compile`、CUDA Graph、AMP、TF32 の変更は入れていない。
22. **SEMamba++ の GAN 更新は公式の系列順を使う。** 各サブ discriminator は real と fake を交互に処理し、CQT と multi-resolution の各損失を別に集計してから加算する。mel 損失と multi-resolution discriminator は公式の CPU 生成 Hann 窓を使い、mel basis は librosa の Slaney フィルタを使う。通常経路の反射パディングは公式に合わせ、厳密な決定的 CUDA では明示パディングを使う。処理順や項の結合を変えると浮動小数点の累算順が変わる。


26. **TF-GridNet の公式学習は ESPnet の可変チャンク iterator で、発話 RMS は掛けない。** `num_iters_per_epoch: 5000` は元発話数であり、optimizer 更新数ではない。1 epoch はその iterator が尽きるまでで、境界で検証、ReduceLROnPlateau、early stopping（patience 5、valid loss）、best 5（valid loss 最小と valid si_snr 最大）を進める。記録される si_snr は負の SI-SDR なので、max は SI-SDR が低い方を選ぶ。ピン留め ESPnet は `speech_volume_normalize` を None のままにする。公式モードの `configs/training/tf_gridnet_chunk_seed17.yaml` は `data.rms_normalize: false` である。他モデルと `tf_gridnet_fixed5000_seed17.yaml` は発話 RMS を維持し、固定 5,000 更新の checkpoint から公式ディレクトリへは再開しない。VoiceBank train 10,700 の実長では、seed 17 の epoch 1–5 の optimizer 更新数が 575、574、593、566、588 だった。短い発話の破棄と、batch 未満の端数チャンクの破棄を含む。複数 process の chunk 分割はしない。詳細は [公式設定監査](../docs/training/OFFICIAL_TRAINING_AUDIT.md)。

27. **DeepFilterNet3 の公式学習は出荷 ini の batch schedule と 120 epoch である。** 実効 batch は `min(schedule, 64)`。128 と 9999 は 64 になる。cosine の表は loader 長が変わった epoch で作り直し、`epoch * steps + step` で引く。10,700 発話の 120 epoch は optimizer 更新 22,711 回で、`max_steps: 22711` がその回数である。次の更新は cosine 表の外を指す。早期停止は validation loss が 25 epoch 改善しないときで、改善するとカウンタを戻す。勾配蓄積で公式 batch を置き換えない。batch 32 と 64 はこの MIG に載る。研究用の検証は 1 発話ずつ計算した損失の非加重平均で、公式のパディング込み batch 平均とは別である。固定 VoiceBank レシピは公式のオンライン混合とは別である。設定は `configs/training/deepfilternet3_schedule_seed17.yaml`。

28. **長時間学習の管理経路は更新完了境界で保存し、前ジョブの成功を確認してから続ける。** `job_chain` は設定・source・commit・DB・環境と checkpoint の指紋を記録し、前状態と qacct の正常終了を要求する。保留数の上限を超える自動投入や、古い checkpoint への自動巻き戻しはしない。実装と対話 GPU スモークはあるが、qsub ジョブ間の動作は未検証であり本学習 GO ではない。検証範囲と残件は [ジョブ継続の検証](../docs/training/JOB_CONTINUATION_AUDIT.md)。
29. **研究室の本学習は、時間制限が無いホストではジョブを分割しない。** 定期 checkpoint と再開を使う。`SIGTERM` は現在の optimizer 更新を終えてから checkpoint を書き、`CHECKPOINTED` で戻る。`SIGKILL` は書き込み中の一時ファイルを捨て、直前の完成 checkpoint を残す。同じ `checkpoint_dir` の二重起動は `.lock` で拒否する。完了時の `complete.json` がある実験は再開しない。破損した checkpoint は `TrainingError` にして再開元にしない。TSUBAME の `qsub` 連鎖はこの決定の対象外である。
30. **noisy と enhanced の音質比較は指標ごとに共通の有効 case だけで行う。** 条件別の単独平均は従来どおり残し、同一 record の両値が有限かつ `ok` の case の平均・差と除外理由を `summary.json` に別途記録する。PESQ の長さ依存があるときは値を埋めず、詳細は [調査](../docs/evaluation/PESQ_LENGTH_INVESTIGATION.md) に記録する。
