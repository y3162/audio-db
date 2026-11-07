# Agent 5 独立レッドチーム（P10-C）

判定日は 2026-11-07。対象は branch `feature/reproduction` の現行ソースである。エージェントの記述は証拠にしていない。本番コード、テスト、YAML、DuckDB は変更していない。学習は開始していない。GPU は使っていない。

preflight の PASS は、このモジュールが見る起動条件の一致である。公式再現の成功ではない。テストだけの追加は RESOLVED にしない。研究用 validation の新設は、公式の不具合修正ではない。

## 起動判定

| 対象 | 判定 |
|---|---|
| MP-SENet（`mp_senet_vb28_official.yaml`、world 2） | CONDITIONAL GO |
| MossFormerGAN 1 GPU（`mossformergan_vb28_official.yaml`） | CONDITIONAL GO |
| MossFormerGAN 2 GPU（`mossformergan_vb28_official_2gpu.yaml`） | NO-GO |
| TF-GridNet（`tf_gridnet_ms_snsd_full.yaml`） | CONDITIONAL GO |
| TF-Locoformer 研究プロトコル（`tf_locoformer_dns_research_valid_full.yaml`） | NO-GO |
| TF-Locoformer 元の 36000 件検証 | NO-GO |

CONDITIONAL GO は、元の開始ブロッカーのうちコードで閉じたものと、残る公式差を分けたうえでの起動可否である。数値の公式一致が済んだ、という意味ではない。

## このレビューが自分で見たもの

CPU、`.venvs/se/bin/python`、torch 2.6.0+cu124。`CUDA_VISIBLE_DEVICES=""`。

- `src/training/se/mp_senet.py` の `validation_report` と `_generate`、公式 `src/se/official/mp-senet/train.py` の検証ループと `steps` の増分、`models/model.py` の `eval_pesq`。
- `src/training/se/preview.py` の `predict`。
- `src/training/se/mossformergan.py` の初期化、`distribute`、`train_update`、公式 `clearervoice-studio/train/speech_enhancement/train.py`、`networks.py`、`models/mossformer_gan/generator.py`、`solver.py`。
- `src/training/core/trainer.py` の 4 つの学習ループ、`_accept_optimizer_update`、`_validate`、`_validate_training_data`、`_validate_aligned_chunks`、`_finish_resumed_validation`、保持。
- `src/training/se/run.py` の `execute`、`src/training/se/cli.py`、`src/training/se/preflight.py` の allowlist。
- 本格 YAML 2 件と `configs/training/reproduction/p10b/` 4 件。p10b は作業ツリーで差分なし。`checkpoint_interval: 2` は残っている。
- 研究用リストの case id を自分で SHA-256 した。親スナップショット先頭ファイルの指紋を、リストに記録された指紋と比べた。
- 単体テスト 30 件は 145.5 秒で OK。`tests.training.test_preflight_full`、`tests.training.test_checkpoint_retention`、`tests.reproduction.test_research_valid_fixed`、`tests.training.test_mp_senet_validation_full_length`。
- 一時ディレクトリへの preflight。GridNet 本格、MP-SENet official、MossFormerGAN 1 GPU、MossFormerGAN 2 GPU は PASS。Locoformer 研究 YAML は FAIL。この PASS は公式再現の成功ではない。

## NOT_TESTED

- GPU 上の SyncBatchNorm の running stats と公式 DDP の数値一致。
- `tests/training/test_mossformer_official_init.py` の再実行。著者のテストは max abs 0 を主張している。このレビューは再実行していない。
- VoiceBank の実波形に対する PESQ。824 件、および「100 の倍数でない 807 件」の再計数。807 は PESQ 失敗件数ではない。
- 学習の開始、optimizer の実更新、実 checkpoint からの再開。
- 片 rank の例外や SIGTERM で他 rank が collective に残るか。
- 元の 36000 件検証の所要時間の再計測。テストが固定している 841.05 時間は外挿のままである。

## 重点確認

### 1. MP-SENet の検証入力と PESQ

`validation_report` は `align_validation_length` で切らず、clean と noisy の全長を `_generate(..., allow_official_istft_length=True)` に渡す。学習の `_reconstruction` と `_official` はフラグなしで、iSTFT 長が波形長と違うと `TrainingError` になる。公式 `train.py` の検証も全長を `mag_pha_stft` に入れている。

`loss == "official"` の `_validation_pesq` は `official_eval_pesq` の失敗を -1 にし、`combine_official_pesq_values` が平均に残す。研究経路は `_annotate_research_pesq` が `na` を平均から外す。欠測ライブラリは両方 `unavailable` である。

再実行した CPU forward（同じ初期重みを公式ネットからコピー）では、長さ 1600 と 1650 の magnitude 最大絶対誤差はどちらも 0.0 だった。同じテストが、1650 を 1600 に切った入力との magnitude 差 1.5627447366714478 を残している。これは切り詰めをした場合の差であり、現行の検証入力の差ではない。実データの PESQ 件数は測っていない。

`preview.predict` は MP-SENet で同じフラグ付き全長生成をする。TensorBoard 用の戻り値は、生成後に重なり長へ揃える。生成前の切り詰めは現行ソースに無い。

### 2. MossFormerGAN の初期化

公式 `train.py` は `network_wrapper` を 2 回作る。各 wrapper は `MossFormerGAN_SE_16K` を 2 回作り、train モードでは SyncANet のあと Discriminator を作る。順は G1 D1 G2 D2、保持した G1 の `.to(device)`、G3 D3 G4 D4 で、残すのは G1 と D4 である。

現行 `MossFormerGANTask.__init__` は同じ順を `seed_model_init` の中で行う。`generator.to(device)` はそのブロックの中にあり、3 回目の `_mossformer_generator` より前である。

CPU で、すでに CPU 上にある `Linear` と `BatchNorm1d` の `.to(cpu)` を実行した。パラメータ storage は同一、torch / Python / NumPy の RNG state は不変だった。この確認の範囲では、CPU の `.to(cpu)` は RNG を消費しない。CUDA の `.to` は測っていない。

著者の `test_mossformer_official_init.py` は、残した generator と discriminator の max abs が 0 であることを要求する。このレビューは再実行していない。旧 checksum `7729.488281` は使っていない。

### 3. PESQ スキップと global_step

`train_update` は PESQ 失敗や非有限の generator 損失で例外にせず、`update_applied` 0 を返す。DDP では先に `_agree_skip` が `all_reduce` の MAX で群全体をスキップする。Trainer 側の一致は `_accept_optimizer_update` の `all_gather` である。値が rank 間で違うと `TrainingError`。0 のときは rank 0 が `pesq_skip` を書き、`global_step` を増やさず `continue` する。

この `continue` は次の 4 ループにある。`_run_epochs`、`_run_data_epochs`、`_run_chunk_epochs`、`_run_aligned_chunk_epochs`。

公式 `solver.py` は失敗 batch を backward も step もしない一方、`state=='train'` の中では `self.step += 1` を失敗時も実行する。公式の蓄積は成功 batch だけで窓を埋める。現行は固定グループ全体を捨て、`global_step` も進めない。検証の非有限 generator 損失はまだ例外になる。

### 4. SyncBatchNorm

`world_size > 1` のとき Discriminator だけ `convert_sync_batchnorm` する。Generator 側の `ConvModule` には BatchNorm が無い。`Conformer_ConvModule` の BatchNorm は、generator が使っているクラスとは別である。実行時にモジュールを数えてはいない。

CUDA では Discriminator を DDP で包み、`broadcast_buffers=False`、`find_unused_parameters=False` である。公式 DDP は `find_unused_parameters=True` で、`broadcast_buffers` は既定の True である。CPU では Discriminator を DDP で包む前に戻る。torch 2.6 の `SyncBatchNorm.forward` は、学習中かつ process group 初期化済みで入力が CUDA でないとき、collective の前に `ValueError` を出す。ソース上、その raise は `all_gather` より前である。`tests/training/test_mossformer_syncbn_collective.py` の gloo 試験は `forward` を差し替えた all_gather であり、`nn.SyncBatchNorm.forward` そのものではない。

`validation_requires_all_ranks` が真のとき、`_validate` と `_validate_training_data` は全 rank が forward に入る。件数は `_require_same_validation_count` が forward の前に `all_gather` し、違えば例外にする。

MossFormer の `TrainingData.validation` は `rank` を受けたあと捨て、全 rank に同じ連結順を返す。SyncBN が見る検証 batch は rank ごとに違わない。公式の検証 DataLoader は `distributed` のとき `DistributedSampler` で rank ごとに分かれる。chunk 検証の `_validate_aligned_chunks` は `rank_batches[rank]` で分かれるが、MossFormer の経路ではなく、`validation_requires_all_ranks` も見ていない。

GPU 上の数値一致は NOT_TESTED。

### 5. 本格 YAML、world size、再開

`tf_gridnet_ms_snsd_full.yaml` は `required_world_size: 2`、`max_steps: 553521`、`checkpoint_interval: 0`。`verification_max_updates`、`max_updates`、`validation_max_updates`、`checkpoint_interval: 2` は無い。preflight は PASS、dataset は ready。これは起動条件であり、公式再現の成功ではない。

`tf_locoformer_dns_research_valid_full.yaml` は `required_world_size: 4`、`max_steps: 2100000`、`checkpoint_interval: 0`、dataset は `tf_locoformer.dns.accept_recipe.research_valid_fixed@1`。同じ cap は無い。preflight は snapshot 0 件で FAIL。

p10b 4 ファイルは作業ツリーで未変更で、`checkpoint_interval: 2` と `verification_max_updates: 2` が残る。

`execute` は `setup_process` のあと、`build_task` と snapshot を開く前に `required_world_size` を拒否する。world 2、global 4、micro 上限 4 の MP-SENet 計画は micro 2、蓄積 1。MossFormer 2 GPU は micro 1、蓄積 2。同じ global 4 を world 1 で計画すると蓄積 4 になるが、YAML の `required_world_size: 2` が先に拒否する。GridNet world 4 を計算すると蓄積 1 のままになる。本格 YAML が固定しているのは world 2 である。

CLI の `--resume` は `strict_resume=True` で `execute` に渡る。`RESUME_ALLOWLIST` は `logging` だけである。実 checkpoint での再開はしていない。

### 6. 研究用 validation

`data/db/reproduction.duckdb` は無い。リストの `registration_status` は BLOCKED、`snapshot_hash` は null、`leakage_safe` は false、`claim_label` は `research_checkpoint_selection` である。`cv_synthetic_small` ではない。

case id は 1024 件、重複なし、辞書順。SHA-256（UTF-8、辞書順、末尾改行）は `73e8d44a3c2196af874f8cddb7a1cc9a6b1745a220d6f1b88fda7a9a502f7ad4` で、フィールドと一致した。親スナップショット `5469f0416329...` の先頭 8 ファイルの size と mtime は、リスト内の `manifest_fingerprint` と一致した。この比較の範囲では、記録後に親スナップショットの先頭ファイルは変わっていない。親の `selection_note` は、36000 件が `cv_synthetic_small` ではなく、その名前は UNKNOWN だと書いている。

`tests.reproduction.test_research_valid_fixed` は、親が存在するとき公開リストと再選択が一致することを含めて通った。登録は BLOCKED のままである。

### 7. checkpoint 保持と検証のやり直し

`checkpoint_retention` が dict のときだけ `_remember_owned` が削除する。未設定の run は削らない。本格 YAML は `keep_latest: 3`、`keep_best`、`protect_nbest` を持つ。

`validation_pending` は、optimizer 更新のあと、検証と scheduler の前に checkpoint へ入る。再開の `_finish_resumed_validation` は、その検証を先頭 batch からやり直す。検証の途中カーソルは無い。途中再開は追加されていない。

## 全 ID

| ID | 状態 | 根拠 |
|---|---|---|
| P10P-A-001 | ACCEPTED_DIFFERENCE | 公式は更新後に `steps += 1` する前の `steps` で保存する。5000 番の保存は 5001 回目の更新のあと。現行は `global_step` を増やしたあと `% checkpoint_interval` で保存する。 |
| P10P-A-002 | RESOLVED | 検証と preview は全長を Generator に入れる。学習更新の長さ拒否は残る。CPU で公式実装との magnitude 差は 1600 と 1650 で 0.0。実 PESQ は NOT_TESTED。807 は再計数しておらず、PESQ 失敗件数ではない。 |
| P10P-A-003 | RESOLVED | `loss: official` は -1 を平均に残す。研究経路は `na` を外したままである。欠測は `unavailable`。807 を失敗件数と呼ばない。 |
| P10P-A-004 | ACCEPTED_DIFFERENCE | 現行再開は `ExponentialLR` の `state_dict` を戻す。公式は `last_epoch` で作り直す。 |
| P10P-A-005 | ACCEPTED_DIFFERENCE | 公式 discriminator は `AdaptiveMaxPool2d`。同値タイの勾配は再測定していない。 |
| P10P-A-006 | RESOLVED | 初期化順は公式の G1 D1 G2 D2、device 移動、G3 D3 G4 D4 と一致し、G1 と D4 を残す。CPU の `.to(cpu)` は確認した範囲で RNG を進めない。重みの max abs 0 は著者テストの主張で、再実行していない。 |
| P10P-A-007 | MITIGATED | 学習中の PESQ 失敗は停止しない。`update_applied` 0 で `global_step` は増えない。公式は失敗 batch でも `self.step` を増やし、成功 batch で蓄積窓を埋める。検証の非有限損失はまだ例外。 |
| P10P-A-008 | MITIGATED | world 2 以上で Discriminator だけ SyncBN。CUDA DDP は `broadcast_buffers=False`。検証は全 rank だが batch は同一で、公式の rank 分割ではない。GPU 数値は NOT_TESTED。 |
| P10P-A-009 | ACCEPTED_DIFFERENCE | `loss >= best` で履歴最小を上書きする。公式 `solver.train` と同じ規則である。 |
| P10P-A-010 | ACCEPTED_DIFFERENCE | 公式 MP-SENet は `cudnn.benchmark = True`。公式 MossFormer は deterministic True、benchmark False。現行 `set_determinism(False)` は benchmark False、deterministic False。本格 YAML は `deterministic: false`。 |
| P10P-A-011 | ACCEPTED_DIFFERENCE | 検証 crop は `mix_seed`。公式は `random.randint`。半減 epoch の実測は無い。 |
| P10P-A-012 | ACCEPTED_DIFFERENCE | `_validate` と chunk 検証は `finally` で RNG を戻す。公式の検証 dropout は学習列を進めたままにする。乱数列の比較は NOT_TESTED。 |
| P10P-A-013 | ACCEPTED_DIFFERENCE | safe YAML は leakage-safe 用で、検証間隔と dataset が公式 YAML と違う。`mp_senet.vb28.ondemand.safe` の snapshot はこの作業木の列挙に無かった。 |
| P10P-A-014 | ACCEPTED_DIFFERENCE | 定数波形の `independent_audio_norm` は空の peak 集合で例外にする。公式平均は未定義になる。 |
| P10P-A-015 | ACCEPTED_DIFFERENCE | 公式の半減は checkpoint から `self.epoch` を戻す。現行は重みを戻し、Trainer の epoch は単調である。 |
| P10P-B-001 | ACCEPTED_DIFFERENCE | chunk 計画は最短 rank で止める。公式 `iterator_stop` と同じ打ち切りで、全件検証ではない。 |
| P10P-B-002 | ACCEPTED_DIFFERENCE | GridNet の `si_snr` は負の SI-SDR。`valid.si_snr.best.pt` の整列は高い `si_snr` を悪い側に置く。本格 YAML の選択は `valid.loss`。Locoformer の検証 `si_snr` は正の SI-SDR。 |
| P10P-B-003 | OPEN | `cv_synthetic_small` の件数と選択規則は、親 snapshot の note でも UNKNOWN。36000 件の親集合は変えていない。研究用 1024 件はその名前ではない。 |
| P10P-B-004 | ACCEPTED_DIFFERENCE | chunk 開始は端が窓内に収まるよう `randint` で動く。登録集合で何件動いたかは再計数していない。 |
| P10P-B-005 | MITIGATED | GridNet の cap 無し本格 YAML はある。Locoformer の本格ファイルは研究プロトコルで、元の 36000 件検証の YAML ではない。p10b は未変更。 |
| P10P-B-006 | ACCEPTED_DIFFERENCE | TF-Locoformer は標準偏差が `1e-6` 未満だと forward を拒否する。公式は割る。 |
| P10P-B-007 | OPEN | README hash `239f516` の tree は、このレビューで再取得していない。 |
| P10P-C-001 | BLOCKED | `reproduction.duckdb` は無い。4 モデルの official / recipe snapshot は ready として読めた。DB と snapshot の同一性は証明できない。 |
| P10P-C-002 | BLOCKED | `reproduction-p8.duckdb` は無い。4 モデルの開始条件ではない。 |
| P10P-D-001 | ACCEPTED_DIFFERENCE | 蓄積の最後以外は、rank 間平均の前に局所勾配を clip する。コードは未変更と書いている。2 GPU の実勾配 norm は NOT_TESTED。 |
| P10P-D-002 | MITIGATED | P10P-A-008 と同じ。 |
| P10P-D-003 | ACCEPTED_DIFFERENCE | world 1 の MossFormer 順は `rank_mod_fixed_v1` の注記どおり、公式 `DataLoader(shuffle=True)` の列ではない。 |
| P10P-D-004 | MITIGATED | P10P-A-007 と同じ。 |
| P10P-D-005 | OPEN | 片 rank の失敗で他 rank が collective に残る実測は無い。 |
| P10P-D-006 | OPEN | DeepFilterNet3 の `distribute` は SyncBN に変換しない。`validation_requires_all_ranks` も立てない。4 モデルの開始条件ではない。 |
| P10P-D-007 | ACCEPTED_DIFFERENCE | 学習ログは rank 0 の `_log_train` だけが書く。 |
| P10P-D-008 | OPEN | 他 rank の `unavailable` がスコアを消す経路は、このレビューで発火させていない。 |
| P10P-D-009 | OPEN | Discriminator DDP は `find_unused_parameters=False`。公式は True。実学習形状での未使用パラメータは NOT_TESTED。 |
| P10P-E-001 | MITIGATED | 検証前に `validation_pending` 付きの step checkpoint を書く。best ファイルは検証中に先へ進むことがある。再開は検証を先頭からやり直す。 |
| P10P-E-002 | MITIGATED | 検証中の死は、保存済みの optimizer 更新を繰り返さない。検証は先頭からやり直す。途中 batch からの再開は無い。 |
| P10P-E-003 | MITIGATED | 1 個の checkpoint pickle の中の optimizer と scheduler は、その保存時点の組である。ディスク上の best は、完了 checkpoint より先に置き換わることがある。 |
| P10P-E-004 | MITIGATED | CLI の再開は logging 以外の config 差を拒否する。`execute` を `strict_resume=False` で直接呼ぶ経路は残る。実再開は NOT_TESTED。 |
| P10P-E-005 | MITIGATED | `step_*.pt` の replace のあと、`generator.pt` と status は別の `save_atomic` である。同じ `max_steps` の再開は `fit` が拒否する。 |
| P10P-E-006 | OPEN | `truncate_after` は各行を `json.loads` する。壊れた行は再開を止める。引き裂き書き込みは再試験していない。 |
| P10P-E-007 | ACCEPTED_DIFFERENCE | `save_atomic` は一時ファイルのあと `os.replace` する。失敗時は直前の checkpoint が残る。保持テストは通った。 |
| P10P-E-008 | ACCEPTED_DIFFERENCE | epoch 途中の checkpoint は、まだ実行していない検証を含まない。`validation_pending` は検証境界だけである。 |
| P10P-E-009 | ACCEPTED_DIFFERENCE | 非 chunk の ExponentialLR は、データ epoch の終端で進む。 |
| P10P-E-010 | OPEN | chunk の再開照合は学習 dataset の identity を見る。検証 dataset の identity は別オブジェクトのままである。 |
| P10P-F-001 | MITIGATED | P10P-B-005 と同じ。 |
| P10P-F-002 | MITIGATED | 本格 YAML の間隔は 0 で、保持は opt-in。p10b の間隔 2 はそのままなので、p10b を本格へコピーすると件数が増える。 |
| P10P-F-003 | RESOLVED | 本格 YAML と MossFormer の official YAML は `required_world_size` を持ち、`execute` が `build_task` の前に拒否する。不一致の単体テストは通った。p10b にはこのキーが無い。 |
| P10P-F-004 | RESOLVED | `--resume` は `strict_resume=True`。差を許すキーは `logging` だけ。実 checkpoint での再開は NOT_TESTED。 |
| P10P-F-005 | MITIGATED | `build_provenance` は `git_dirty` を書く。preflight も git dirty を出す。 |
| P10P-F-006 | OPEN | `complete.json` は status、global_step、checkpoint だけである。chunk の early stop も `COMPLETED` になり得る。一次記録は JSONL。 |
| P10P-F-007 | OPEN | 36000 件 × 150 epoch の 841.05 時間は外挿のまま。研究用 1024 件は別プロトコルで、この所要時間を消していない。 |
| P10P-F-008 | MITIGATED | P10P-E-002 と同じ。途中再開は追加されていない。 |
| P10P-F-009 | OPEN | 容量見積は保持ファイル数 × checkpoint バイト数である。provenance sidecar は入っていない。 |
| P10P-G-001 | RESOLVED | P10P-A-003 と同じ。公式経路の -1 平均は入った。研究経路の除外は残る。 |
| P10P-G-002 | ACCEPTED_DIFFERENCE | P10P-A-001 と同じ。 |
| P10P-G-003 | ACCEPTED_DIFFERENCE | P10P-B-002 と同じ。`valid.loss` の n-best を公式と違う規則だとは読まない。 |
| P10P-G-004 | ACCEPTED_DIFFERENCE | P10P-A-011 と同じ。半減 epoch がずれた測定は無い。 |

## 判定の理由

MP-SENet は CONDITIONAL GO。検証入力の切り詰めは本番コードから外れ、CPU のランダム初期重みでは公式 Generator と magnitude が一致した。公式 PESQ の -1 平均は `loss: official` にある。残る step 番号、cuDNN、再開時の学習率、crop 乱数は既知の差である。preflight PASS は公式再現の成功ではない。

MossFormerGAN 1 GPU は CONDITIONAL GO。初期化順は公式と揃い、PESQ 失敗は run を止めない。world 1 は通常 BatchNorm のままである。失敗時に `global_step` を進めないことと、グループ全体の skip は公式の step カウンタと違う。重み一致の再実行はしていない。

MossFormerGAN 2 GPU は NO-GO。SyncBN 変換と全 rank 検証はコードにある。GPU の数値一致は無く、検証 batch は公式のように rank で分かれず、DDP の `broadcast_buffers` と `find_unused_parameters` も公式と違う。preflight PASS はこの差を消さない。

TF-GridNet は CONDITIONAL GO。cap 無しの本格 YAML があり、snapshot は ready、preflight は PASS だった。WAV のビット一致や統計的同等は主張できない。`valid.si_snr.best.pt` を公式成果物にしない。p10b は使わない。

TF-Locoformer の研究プロトコルは NO-GO。YAML の dataset は未登録で、preflight は snapshot 0 件の FAIL だった。1024 件のリストは研究用の選択であり、公式の `cv_synthetic_small` でも 36000 件検証でもない。

TF-Locoformer の元の 36000 件検証は NO-GO。親の 36000 件は残っている。検証時間の 841 時間は再計測していない。研究用部分集合は、この検証を公式手順として置き換えない。
