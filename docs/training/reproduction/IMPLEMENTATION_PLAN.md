# 再現用 DB・Data Pipeline の実装計画

状態: **P1〜P3 は実装済み。** P1 は合成データの単体テスト。P2 は VoiceBank-DEMAND の実ファイルで、MP-SENet と MossFormerGAN の 4 Dataset とスナップショットまで完了している。P3 は固定ペアの Pipeline、Legacy アダプタ、Trainer 接続まで完了している。検証は [P3_VERIFICATION](P3_VERIFICATION.md)。**P4〜P10 は未着手。** フェーズ 4 以降は、この計画の受け入れ条件をそのまま使う。

設計の正本: [REPRODUCTION_DB_DESIGN](../../db/REPRODUCTION_DB_DESIGN.md)（DB）、[DATA_PIPELINE_DESIGN](DATA_PIPELINE_DESIGN.md)（Pipeline）。再現対象の正本: [TARGETS](TARGETS.md)。採用決定: [DECISIONS](../../../.agents/DECISIONS.md) の D-01〜D-09。

## 0. 検証の区別（AGENTS.md）

各フェーズの受け入れ条件は、次の 3 種類を区別して書く。確認していないことを「検証済み」と書かない。

| 区分 | 意味 |
|---|---|
| **単体（Mock）** | 合成データ・一時 DB・CPU のテスト。実データ・実モデルを使わない |
| **実データ smoke** | 実コーパス・実モデルで動かす小規模な通し。公式出力との一致は主張しない |
| **公式一致** | 公式コードの出力（リスト、サンプル、分布）との照合 |

GPU を使うテストは、`nvidia-smi` で空きを確認してから行う。TSUBAME で 30 分以上のジョブは、投入前にユーザーの明示の指示か追加承認が要る（優先度は `-p -5`、`h_rt` を 30 分未満にして回避しない）。

---

## 1. 着手前の確認（ユーザーへ）

フェーズ 1 の前に確認したい事項。**答えが無くても着手できるものには「既定」を書いた。** 既定で進めてよければ、そのまま進める。

| ID | 質問 | 既定（答えが無い場合）| 影響 |
|---|---|---|---|
| Q1 | 本計画のフェーズ順（§3）と、フェーズ 1 の着手の可否 | 承認が無ければ着手しない | 全体 |
| Q2 | `data/generated/` の実体の置き場所と容量（TF-Locoformer S4 で約 1.04 TB。`/gs/bs` の空きは 2026-11-06 に 17 TB で共有）| S3 まで（約 0.3 TB）は `data/generated/` 直下。S4 は別途確認 | フェーズ 7 |
| Q3 | `resource.duckdb` に `eval` スキーマが無い現状（DB 設計 §2.3）は意図したものか | 変更しない。leak 監査は `UNKNOWN` | フェーズ 2 |
| Q4 | 公式リスト（MP-SENet の `training.txt` / `test.txt` など）を `configs/data/lists/` へ複製してよいか（ライセンスは各リポで確認）| 複製する前にライセンスを確認し、不可なら SHA-256 だけを記録して、取得手順を文書化 | フェーズ 2 |
| Q5 | `cv_synthetic_small` の大きさと規則（D-08 で未確定）| 未確定のまま。S1・S2 は暫定規則（`provisional` と明記）| フェーズ 7 |
| Q6 | 16 kHz 化した派生素材で `official_reproduction` を名乗ってよいか（公式の変換器は UNKNOWN）| 名乗る。ただし `deviations_json` に「変換器は公式不明」を必ず併記 | フェーズ 2 |
| Q7 | 16 kHz 化の既定の変換器（候補: `soxr` の HQ。現行の audio-db は `torchaudio.functional.resample`）| `soxr` HQ。選んだ変換器と版を `derivations` に記録 | フェーズ 2 |
| Q8 | DFN3・SEMamba++ は、公式素材の入手を待つか、代替素材（別プロトコル準拠）を先に進めるか | 代替素材を先に進める（D-05、D-06）。ラベルを分ける | フェーズ 8・9 |
| Q9 | 新しい最上位パッケージ `src/reproduction/` の追加と、保護一覧（AGENTS.md）への `reproduction.duckdb`、`data/generated`、`data/staging/reproduction/`、`data/db/snapshots/` の追加 | 追加する（フェーズ 1 の完了時に正本を更新）| 全体 |
| Q10 | 実機の複数 GPU の台数・利用可否と、S2〜S4 の生成ジョブの承認の段取り | 最初は virtual rank で検証し、実 DDP は GPU 確保後 | フェーズ 5 |
| Q11 | TF-Locoformer の `noise` ストリームを省略してよいか（学習が使うか未確認。省略で約 0.35 TB 減）| 公式どおり 3 ストリームを生成する | フェーズ 7 |

---

## 2. 未解決事項（解決できない・まだ決められないもの）

**解決できない（一次資料・素材・実機が無い）**

| ID | 内容 | 扱い |
|---|---|---|
| UN-1 | DFN3 の `dataset.cfg`、DNS4、PTDB、RIR 集合、`sampling_factor` | 公式素材の Dataset は `declared_status = blocked`。代替素材は別ラベル（D-06）|
| UN-2 | SEMamba++ の公式素材（VCTK 0.92、Arni RIR）、VCTK-GSR の分割規則、検証データの作り方 | `blocked` / `unknown`。代替素材は別ラベル（D-05）|
| UN-3 | MP-SENet・MossFormerGAN の 16 kHz 版の公式の変換器 | `derivations` と `deviations_json` に選んだ方法を明記（Q6、Q7）|
| UN-4 | TF-GridNet の公式生成物（未シード、再現不能）| seed 固定は公式との差として記録（D-07）|
| UN-5 | TF-Locoformer の `cv_synthetic_small`、実学習の所要時間・停止理由 | Q5 |
| UN-6 | MS-SNSD の `sf.write` のクリップ時の挙動 | 生成時に実測し、`audits` に記録 |
| UN-7 | ESPnet の validation の `iterator_stop` 相当、DDP の dropout の seed、基底の発話順（shape file）| ESPnet のコードを実装時に確認（DATA_PIPELINE_DESIGN O-4〜O-6）|
| UN-8 | 公式と audio-db の torch / librosa / soundfile / soxr の版差（公式は librosa 0.7.2 など）| `meta` に記録。差の大きさは測れる範囲で測る |

**まだ決めない（実装時・測定後）**

| ID | 内容 |
|---|---|
| OP-1 | スナップショット `.npz` の列構成の詳細（形式は決定済み）|
| OP-2 | 読み込みのスレッド / プロセスの選択、prefetch の深さ（測定後）|
| OP-3 | TF-Locoformer の生成の並列度と所要時間（S2・S3 の実測後）|
| OP-4 | 公式評価（Official Evaluation）の実装場所（`src/evaluation/official/` の案）。フェーズ 4 は仕様だけ |

---

## 3. フェーズ

### 3.1 一覧と依存

```
P1 DB 基盤 ──┬─► P2 固定ペアの準備（MP-SENet・MossFormerGAN）──┐
             │                                                 ├─► P4 来歴の伝播・評価分離 ──► P10 統計的同等性・公式一致の検証
             └─► P3 Pipeline 中核 + Legacy アダプタ ─────────────┤
                       │                                       │
                       ├─► P5 chunk / DDP / virtual rank ─► P6 TF-GridNet（生成 + 学習）──► P7 TF-Locoformer（段階生成）
                       └─► P8 DFN3 オンライン ─────────────────────────────────────────────┤
                       └─► P9 SEMamba++ オンライン ────────────────────────────────────────┘
```

- P2 と P3 は P1 のあと**並行**できる（P3 は P1 のスナップショット形式に依存するが、テストでは合成のスナップショットを使える）。
- P8・P9 は P3 だけに依存し、P5〜P7 と**並行**できる。ただし素材（UN-1、UN-2）が無い間は代替素材だけ。
- 推奨の着手順: P1 → P2 ∥ P3 → P5 → P6 → P4 → P7 → P8 ∥ P9 → P10。P4 は P2・P3 の後ならいつでもよい。

### 3.2 フェーズ 1: DB 基盤

| 項目 | 内容 |
|---|---|
| 目的 | `reproduction.duckdb`、staging、登録、再構築、スナップショットを作る。モデルの実データはまだ入れない |
| 変更するモジュール | 新規: `src/reproduction/{errors,ids}.py`、`src/reproduction/db/{schema,registry,snapshot,audit}.py`。`Makefile`（`reproduction-doctor`、`reproduction-init-db`、`reproduction-snapshot`、`reproduction-rebuild-db`、`reproduction-verify`）。`tests/reproduction/` |
| 機能 | 全表の作成（DB 設計 §4）、不変条件の検査、staging の書き込み（内容アドレス、書き込み 1 回）、1 トランザクションの冪等な登録、`flock` による書き込みの排他、新しいファイルへの再構築と全表のハッシュの比較、スナップショットの書き出し（`.npz` + JSON）、監査の記録 |
| 受け入れ（単体）| (1) 同じ staging から 2 回作った DB の全表の `json_hash` が一致し、登録順を変えても一致する。(2) 同じ ID で内容が違う登録を拒否する。(3) CHECK と登録関数の不変条件（`claim_label` と `profile` の組み合わせ、`selection_uses_test` の計算、`generated` 素材の必須列）の違反を拒否する。(4) 既存の `resource.duckdb` を開かず、変更しない（ハッシュ比較）。(5) スナップショットが `numpy` と JSON だけで読め、同じ `snapshot_hash` を再生成する。(6) DuckDB の無い環境（`.venvs/deepfilternet`）で、スナップショットの読み込みが import できる（読み込み側が `duckdb` を import しない）。(7) 書き込みロックが取れないとき失敗する。(8) `src/evaluation/` が `src.reproduction` を import しないことの import テスト |
| 完了条件 | 受け入れの全項目が通る。`data/db/reproduction.duckdb` が空のスキーマだけで作られる。`AGENTS.md`、`.agents/DATA.md`、`.agents/PROJECT.md` に、新 DB・新パッケージ・保護一覧を反映（Q9）|
| 依存 | Q1、Q9 |
| 更新する正本 | `AGENTS.md`（保護一覧）、`.agents/DATA.md`、`.agents/PROJECT.md`（依存方向）、`.agents/DECISIONS.md`（D-01〜D-09 のうち DB の部分を「実装済み」へ）|

### 3.3 フェーズ 2: 固定ペアの準備（MP-SENet・MossFormerGAN）

| 項目 | 内容 |
|---|---|
| 目的 | VoiceBank-DEMAND の公式分割と leakage-safe 分割を、DB に登録してスナップショットまで作る |
| 変更するモジュール | `src/reproduction/prepare/{materials,lists,derive,datasets}.py`、`configs/data/{protocols,datasets,lists}/`、`Makefile`（`reproduction-prepare`）、`tests/reproduction/` |
| 機能 | `corpora.audio_files` から必要な行を `assets` へ写す（読み取り専用）、公式リストの取り込み（SHA-256 付き）、16 kHz 派生素材の作成（`derivations` に記録）、ケース集合（official 11,572 / 824、safe 10,700 / 872 / 824）、Dataset 定義（`official_faithful` と `leakage_safe`）、監査（件数、リストの SHA-256、leak）|
| 受け入れ（単体）| (1) 公式 `training.txt`（11,572）と `test.txt`（824）の件数とリストのハッシュが一致する。(2) `leakage_safe` の train / valid / test が素材・話者で重ならない。(3) `official_faithful` が `selection_uses_test = true` を計算して記録する。(4) `leakage_safe` で `selection_uses_test = true` の Dataset を拒否する。(5) `eval` スキーマが無い環境で leak 監査が `UNKNOWN` になり、`PASS` にならない。(6) `reproduction-prepare` が、素材が無いとき `BLOCKED`（終了コード 3）を返し、何も作らない |
| 受け入れ（実データ smoke）| VoiceBank の実ファイルで、素材の件数・レート・フレーム数が `corpora.audio_files` の値と一致する。派生素材の長さが `resampled_frame_count` と一致する |
| 受け入れ（公式一致）| 公式リストの各 ID が、実コーパスのファイルに解決される（欠落ゼロ）。公式の `MP-SENet` が `librosa.load(sr=16000)` で得る長さと、派生素材の長さの差を測って記録する（UN-3、UN-8）|
| 完了条件 | 上記の通過。`mp_senet.vb28.official`、`mp_senet.vb28.safe`、`mossformergan.vb28.official`、`mossformergan.vb28.safe` のスナップショットが作れる |
| 依存 | P1、Q3、Q4、Q6、Q7 |
| 更新する正本 | `.agents/DATA.md`（新 DB の現状）、`DECISIONS.md` |

P2 の実施結果（2026-11-06）:

- 公式リストは `configs/data/lists/mp_senet/` と `configs/data/lists/mossformergan/` に、固定 commit のバイト列のまま置いた。両リストの発話 ID と順序は一致する。
- 実ファイル 24,792 本（clean/noisy × train 11,572 と test 824）を `repro.assets` に写し、soxr 1.1.0 HQ の 16 kHz PCM_16 を同じ本数だけ `data/derived/VoiceBank-DEMAND/soxr_hq_16000/` に作った。容量は 2.3 GB。クリップしたサンプルは 0。
- official は 11,572 / 824 / 824 で `selection_uses_test = true`。safe は 10,700 / 872 / 824 で `selection_uses_test = false`。safe の asset・話者・SHA-256 の分割間重複は PASS。`eval` が無いため `eval_sets` だけ `UNKNOWN` で、全体の `leak_check` も `UNKNOWN` である。PASS にはしていない。
- インストール済み librosa 1.0.0 の `librosa.load(sr=16000)`（既定 `soxr_hq`）と、派生素材を 16 ファイル（最短・最長を含む clean/noisy）で比べた。長さの差は 0。soxr の float 出力との最大絶対誤差は 0。保存した PCM_16 との最大絶対誤差は 3.05e-5。公式 MP-SENet がピンする librosa 0.7.2 の既定 `kaiser_best` は、`resampy` が無いため実行していない。
- スナップショットの再書き出しは同じ `snapshot_hash` になった。`resource.duckdb` の SHA-256 は作業前後で `bbd3fe040bf8da67888749cd6410058c742d43c3cf0c0b33d7921366c2da9029` のままである。

### 3.4 フェーズ 3: Pipeline 中核と Legacy アダプタ

| 項目 | 内容 |
|---|---|
| 目的 | `TrainingData` の境界を作り、固定ペアを新経路で、既存の学習を Legacy アダプタで動かす |
| 変更するモジュール | 新規: `src/reproduction/pipeline/{types,store,order,planner,transform,renderer,assemble,loader,state,training_data}.py`、`src/training/data/legacy.py`。変更: `src/training/core/trainer.py`（`TrainingData` を受ける、`data_state`、`update_in_epoch`）、`src/training/se/config.py`・`run.py`（`data.pipeline`、`data.dataset` など、Legacy のキーとの併用拒否）。`tests/reproduction/`、`tests/training/` |
| 機能 | `OrderPolicy` 5 種（§8.2）、固定ペアの Planner、`FileRenderer`、`noisy_rms_unit_v1` / `independent_audio_norm_v1` の正規化、crop / pad、`BatchAssembler`、スレッド loader（prefetch、位置のコミット）、`data_state` の保存と照合、再開、`selection_uses_test` と `allow_test_selection` の検査 |
| 受け入れ（単体）| (1) **Legacy の黄金テスト**: 現行の `RecipeDataset` 経路と Legacy アダプタ経路で、同じ seed の `plan`・バッチ・波形が一致する（現行のテストを変えずに通す）。(2) `global_shuffle_stride_v1` が現行の `epoch_indices` と一致する。(3) `torch_distributed_sampler_v1`・`fixed_single_shuffle_v1`・`rank_mod_fixed_v1`・`per_rank_independent_v1` の順序が、公式の式を再実装した参照テスト（合成データ）と一致する。(4) 計画が純関数であること: 同じ入力で同じ `plan_hash`、`world_size` を変えると変わる。(5) 再開: epoch の途中で保存・再開した場合と、通しで回した場合で、以降のバッチが一致する。(6) prefetch の破棄: 先読み済みを破棄して再開しても、以降のバッチが一致する。(7) `data.pipeline` のキーの併用拒否。(8) test の validation 使用が、`selection_uses_test` と `allow_test_selection` の両方が無いとき拒否される |
| 受け入れ（実データ smoke）| VoiceBank の 1 epoch を CPU で通し、MP-SENet の既存タスクの 1 step を実行できる（精度は主張しない）|
| 完了条件 | 上記の通過。Legacy の既存テストがすべて通る。`.agents/PROJECT.md` の依存方向に新 import を反映 |
| 依存 | P1（スナップショット形式）。P2 は不要（合成スナップショットで可）|
| 更新する正本 | `.agents/PROJECT.md`、`.agents/DEVELOPMENT.md`（テストの実行方法）、`DECISIONS.md` |

### 3.5 フェーズ 4: 来歴の伝播と評価の分離

| 項目 | 内容 |
|---|---|
| 目的 | 公式再現の checkpoint が「独立 test を使っていない」ことを、評価結果まで伝える |
| 変更するモジュール | `src/training/core/checkpoint.py`（`data_provenance.json`）、`src/reproduction/db/registry.py`（`runs`）、`src/evaluation/`（**評価セットのレシピの素材 SHA-256 と、checkpoint が持つ選択集合の SHA-256 を突き合わせる読み取りだけ。`src.reproduction` は import しない。`data_provenance.json` を読むだけ**）|
| 機能 | `data_provenance.json` の書き込み、`runs` への登録、評価結果の `independent` と `checkpoint_selection_overlap` の記録 |
| 受け入れ（単体）| (1) 選択に test を使った checkpoint を、同じ素材を含む評価セットで評価すると `independent: false` と重なった集合名が記録される。(2) 素材が重ならなければ記録されない。(3) 名前が同じで素材が違うセットを、重なりと誤認しない |
| 完了条件 | 通過。Official Evaluation の仕様（OP-4）が文書化されている（実装はしない）|
| 依存 | P2、P3 |
| 更新する正本 | `.agents/DATA.md`（評価結果の列）、`docs/evaluation/`（該当する文書）|

### 3.6 フェーズ 5: チャンク iterator の DDP と virtual rank

| 項目 | 内容 |
|---|---|
| 目的 | TF-GridNet（2 rank × 2）・TF-Locoformer（4 rank × 1）の計画を作り、virtual rank と実 DDP で検証する |
| 変更するモジュール | `src/reproduction/pipeline/chunking.py`、`planner.py`、`src/training/core/trainer.py`（`_run_chunk_epochs` の `world_size != 1` の拒否の緩和、validation の集約）、`src/training/data/chunk_plan.py`（`world = 1` の特殊形として残す）|
| 機能 | §8.3 の rank 対応の計画、`iterator_stop`（全 rank の最小）、virtual rank、validation の `all_reduce`、`plan_hash` の rank 間の一致確認、`data_state` の拡張 |
| 受け入れ（単体）| (1) `world = 1` で現行の `epoch_chunk_batches` と一致する（黄金）。(2) `world = 2, 4` で、shard が `[rank::world]`、rank あたりの batch が `global_batch // world`、`updates = min(...)`、余りを捨てる、が ESPnet の式を再実装した参照と一致する。(3) epoch 間で発話の消費位置が進む。(4) virtual rank と、`world` 個のプロセスを模した逐次実行で、各 rank のバッチ列が一致する。(5) validation の合計・件数の集約が、1 プロセスで全体を処理した結果と一致する。(6) `plan_hash` が全 rank で一致する |
| 受け入れ（実データ smoke）| 実 DDP（GPU が確保できる場合）で 2 rank の 1 epoch を完走し、virtual rank の計画と一致する。**GPU 確保は `nvidia-smi` で確認し、Q10 の結果に従う** |
| 受け入れ（公式一致）| ESPnet のチャンク iterator の出力（バッチ列）と、同じ入力・seed での照合。可能な範囲（ESPnet の import が可能なら。公式コードへの**実行時依存は作らず、テスト内だけ**）。不可なら、式のコード読みの一致を「公式一致」ではなく「コード読み」と書く |
| 完了条件 | 通過。UN-7 の 3 項目が ESPnet のコードで確認され、結果が DATA_PIPELINE_DESIGN に反映される |
| 依存 | P3、Q10 |
| 更新する正本 | `.agents/DECISIONS.md`（#26 の単一プロセス記述を更新）、DATA_PIPELINE_DESIGN |

### 3.7 フェーズ 6: TF-GridNet（生成 + 学習）

| 項目 | 内容 |
|---|---|
| 目的 | ESPnet 公開 SE recipe の再現として、MS-SNSD の事前生成、登録、学習の入口までを作る |
| 変更するモジュール | `src/reproduction/build/{plan,render,manifest,verify}.py`、`build/generators/ms_snsd.py`、`protocols/pregenerated.py`、`configs/data/builds/ms_snsd_tfgridnet.yaml`、`Makefile`（`reproduction-plan-build`、`reproduction-build`、`reproduction-register`）|
| 機能 | 計画 → 生成 → 検証 → 登録（DB 設計 §9）。`rm -r` に当たる削除は**しない**。再開、`.part`、shard マニフェスト、`verified` |
| 受け入れ（単体）| (1) 計画だけを 2 回作って `case_id` の列が一致する。(2) shard の途中で中断（`.part` と未完了の shard）→ 再開 → 完了した shard に触れない。(3) マニフェストの無い shard は登録されない。(4) 同じ `build_id` で内容が違うとエラー。(5) 生成条件を変えると別のディレクトリになり、古い Build を変更しない。(6) 容量不足で開始しない。(7) 完了していない Build を指す Dataset のスナップショットが拒否される |
| 受け入れ（実データ smoke）| MS-SNSD で S0（計画）と、数 shard の生成・検証・登録・学習入口（1 epoch の通し）。**実生成は 30 分未満の見込みを確認してから**。超える見込みなら承認を取る |
| 受け入れ（公式一致）| 生成式（`snr_mixer` の平方根、−25 dBFS、0.2 秒の無音、10 秒の窓、5 SNR）を、公式スクリプトの小さな固定入力で照合する（公式スクリプトを実行できる場合。できなければ式の読み合わせとし、「公式一致」と書かない）。SNR と長さの分布が公式の式から期待される値に一致する（統計）|
| 完了条件 | 通過。TF-GridNet の Dataset が `espnet_public_se_recipe_reproduction` として登録され、公式との差（seed 固定、件数の決め方）が `deviations_json` にある |
| 依存 | P1、P3、P5（学習入口の検証）|
| 更新する正本 | `.agents/DATA.md`、`DECISIONS.md`、`docs/training/`（TF-GridNet の記録）|

### 3.8 フェーズ 7: TF-Locoformer（段階生成）

| 項目 | 内容 |
|---|---|
| 目的 | DNS 2020 の合成 3000 h を、段階的に生成・検証する。**今回の設計では生成しない** |
| 変更するモジュール | `build/generators/dns_synth.py`、`configs/data/builds/tf_locoformer_dns.yaml`、`Trainer`（`valid.loss.ave_5best` の実装）|
| 機能 | S0〜S4（DB 設計 §9.6）。生成の並列度の調整。`ave_5best`（上位 5 epoch の平均）の保存と、各 checkpoint の epoch の記録 |
| 受け入れ（単体）| (1) S0 の計画（360,000 件）の決定性、SNR（整数 −5…14）とレベル（−35…−16 dBFS）の分布が公式の式に一致する。(2) 小さい段階の shard が大きい段階の先頭と同一（`build_id` が件数を含まない）。(3) `ave_5best` が、合成の検証履歴で、上位 5 epoch の重みの平均になる（D-09: 厳密に検証する項目）。(4) 容量の見積り（計画から）が 345.6 GB × 3 に一致する |
| 受け入れ（実データ smoke）| S1（約 1 h = 120 本）の生成・検証・登録・学習入口の通し。S2 以降は承認後 |
| 完了条件 | S1 の通過。S2〜S4 は、**ユーザーが承認したものだけ**実施し、実測（所要時間、I/O）を `audits` と文書に記録する |
| 依存 | P6、Q2、Q5、Q11、S2 以降は TSUBAME の承認 |
| 承認の扱い | S2（30 分以上の見込み）・S3・S4 は、そのジョブを含む明示の指示か追加承認が要る。`-p -5`。`h_rt` を 30 分未満にして回避しない |

### 3.9 フェーズ 8: DFN3 のオンライン混合

| 項目 | 内容 |
|---|---|
| 目的 | DFN3 のオンライン混合を専用実装し、代替素材で動かす。公式素材は入手後 |
| 変更するモジュール | `src/reproduction/protocols/online_dfn3.py`、`pipeline/renderer.py`（オンライン用の接続）、`configs/data/{protocols,datasets}/dfn3.*`、`src/training/se/deepfilternet.py`（データの接続のみ）|
| 機能 | `realize`（SNR、gain、雑音 2〜5 本、合成雑音 5 %、残響 `p_reverb = 0.1`、clip ガード、教師 `official_mixed_target`）、窓 3 秒（短い素材は別発話の連結）、固定 validation（`online_snapshot`）、代替素材の素材リスト |
| 受け入れ（単体）| (1) `realize` が `(seed, epoch, index)` の純関数。world_size を変えても同じサンプル。(2) 抽選した SNR・gain・雑音の本数の分布が、公式の分布（一様、`uniform(2,6)` の整数化など）に統計的に同等（固定 seed の十分な標本、有意水準を固定したテスト）。(3) 混合後の実 SNR の誤差が許容内。clip ガードの不変条件。(4) 教師が `0.3 · dry + 0.7 · 後期減衰` の定義と一致（**厳密**）。(5) validation の実現値が毎回同じ |
| 受け入れ（公式一致）| 公式 Rust の `libDF` を実行できる環境があれば、固定入力・固定 seed の混合出力と比べる（できない場合は、式の読み合わせとし、「公式一致」と書かない）|
| 完了条件 | 代替素材の Dataset が `protocol_conformant_substitute_materials` として登録・スナップショットまで作れる。公式素材の Dataset は `blocked` のまま、欠けているものが `blocking_json` に列挙される |
| 依存 | P3、Q8、素材（UN-1）|

### 3.10 フェーズ 9: SEMamba++ のオンライン混合

| 項目 | 内容 |
|---|---|
| 目的 | SEMamba++ の GSR 劣化を専用実装し、代替素材で動かす |
| 変更するモジュール | `protocols/online_semamba.py`、`configs/data/{protocols,datasets}/semambapp.*`、`src/training/se/semambapp.py`（データの接続のみ）|
| 機能 | 劣化の順序（reverb → noise → clipping → bandwidth）、VAD 基準の SNR、窓 48,000、`independent_peak_v1`、`per_rank_independent_v1`、**公式の最小補修（起動不能の 4 点）は実装しない**（D-05。必要になった時点で別の Protocol として決める）|
| 受け入れ（単体）| (1) `realize` が純関数。(2) 劣化の適用順が固定。劣化の確率（0.5 / 1.0）と SNR の分布が公式の `config.yaml` に統計的に同等。(3) VAD 基準の SNR が、固定入力で、公式の `utils.py` の式を再実装した参照と一致（**厳密**）。(4) `per_rank_independent_v1` で rank 間の重複・欠落が起きる |
| 完了条件 | 代替素材の Dataset が登録・スナップショットまで作れる。公式素材は `blocked` / `unknown` |
| 依存 | P3、Q8、素材（UN-2）|

### 3.11 フェーズ 10: 統計的同等性と公式一致の検証、再現 run

| 項目 | 内容 |
|---|---|
| 目的 | D-09 の「厳密に確認する項目」と「統計的同等性で受け入れる項目」を、実装後にまとめて検証し、再現 run を行う |
| 受け入れ（厳密）| 固定ペアの対応と件数（11,572 / 824）、分割、サンプリングレート、教師波形の定義、損失、optimizer・scheduler の更新規則、checkpoint 選択規則（MP-SENet の `best_pesq`、MossFormerGAN の halving、TF-GridNet の `valid.loss`、TF-Locoformer の `ave_5best`）を、各モデルの参照テストと公式の式で確認する |
| 受け入れ（統計的）| オンライン混合と事前生成の分布（SNR、レベル、長さ、使用回数）を、公式の分布と、固定 seed・固定標本・固定の有意水準で比べる。結果を `audits` に記録 |
| 再現 run | MP-SENet（D-03: 2 GPU、global batch 4、400 epoch、段階実行）を最初に行う。400 epoch は 30 分以上のジョブで、投入前に承認が要る。**この計画は run の承認を含まない** |
| 完了条件 | 厳密項目がすべて通る。統計項目の結果が文書に残る。run を行った場合は、`data_provenance.json` と結果の `independent: false`（公式再現）が揃う |
| 依存 | P2〜P9 のうち対象モデルの分 |

---

## 4. 正本の更新の予定

| 変更 | 更新先 | タイミング |
|---|---|---|
| D-01〜D-09 の採用（本設計）| `.agents/DECISIONS.md`（「採用決定済み・未実装」）、`TARGETS.md` | **完了（本作業）** |
| 新 DB・新パッケージ・保護一覧 | `AGENTS.md`、`.agents/DATA.md`、`.agents/PROJECT.md` | P1 の完了時（Q9）|
| 実装した項目を「実装済み」へ | `.agents/DECISIONS.md`（現状の記述も `PROJECT.md`、`DATA.md` に合わせる）| 各フェーズの完了時 |
| テストの実行方法、新環境の規則 | `.agents/DEVELOPMENT.md` | P3、P5 |
| 詳細な検証記録 | `docs/training/VERIFICATION.md`（既存の形式）| 各フェーズの完了時 |
| 設計の変更（実装で判明）| 本書と DB 設計・Pipeline 設計 | 判明した時点。「設計時の想定」を消さず、変更の理由を書く |

---

## 5. リスク

| ID | リスク | 影響 | 緩和 |
|---|---|---|---|
| R-1 | 公式素材の入手不能（UN-1、UN-2）| DFN3・SEMamba++ の「公式データセット再現」が成立しない | ラベルの分離（代替は別プロトコル準拠）。`blocked` を明示 |
| R-2 | 16 kHz 変換器の差（UN-3）| MP-SENet・MossFormerGAN の数値の差 | `derivations` に記録。差を測って文書化。Q6 の確認 |
| R-3 | 3000 h 生成の I/O・容量・時間（約 1.04 TB、17 TB 共有の空き）| S4 が完了しない | 段階生成（S0〜S4）、再開、容量確認、承認ゲート |
| R-4 | スレッド並列でオンライン混合が間に合わない（OP-2）| 学習が遅い | 測定し、必要ならプロセス並列。分布・順序を変えない |
| R-5 | ESPnet の rank 挙動の読み違い（UN-7）| TF 系の epoch 境界・validation の差 | 実装時に ESPnet のコードで確認し、参照テストを作る |
| R-6 | 公式の乱数列と audio-db の torch の版差（OP の O-7）| 順序の一致が保証されない | 構造の再現を目標にし、版を記録（D-09）|
| R-7 | `resource.duckdb` の現状と正本の食い違い（Q3）| 評価側の前提が崩れる | 変更せず確認する。leak 監査は `UNKNOWN` |
| R-8 | 再現 run の長さ（MP-SENet 400 epoch など）| GPU 時間・承認 | 段階実行（D-03）。P10 で承認を取る |
| R-9 | 設計の未実装部分が実装済みとして読まれる | 誤った前提 | `DECISIONS.md` の「未実装」に置く。本書と設計書の状態表記を維持 |

---

## 6. この計画の外（今回やっていないこと）

- 実装、設定（YAML）の変更、Makefile の変更、DB の作成・変更、音声の生成、`make db-ingest`、GPU の使用、学習・評価の実行。
- `AGENTS.md`、`.agents/DATA.md`、`.agents/PROJECT.md`、`.agents/DEVELOPMENT.md` の更新（実装の完了時に行う。§4）。
- Official Evaluation の実装（OP-4）。
- 公式素材の入手。
