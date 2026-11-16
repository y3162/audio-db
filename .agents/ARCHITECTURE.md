# 学習の不変条件

学習式を「整理」で変えない。6 モデルの比較用学習 Task は `tests/numerics/reference/` の `mp_senet_task.py`、`tf_gridnet_task.py`、`tf_locoformer_task.py`、`deepfilternet_task.py`、`semamba_task.py`、`mossformer_task.py` にあり、本番はそれらを import しない。共有の STFT・スペクトル損失・scheduler は `src/experiments/calc/` にある。学習が backward する損失は YAML の Loss Term と Objective で組み立て、Trainer は注入された Objective を呼ぶ。参照ステップは自分の係数を持ち、テストが両方を比べる。第二の学習 CLI は作らない。仕様は `docs/experiments/LOSS.md`。

## Checkpoint

仕様は `docs/experiments/CHECKPOINT.md`、プロセスの監視は `docs/experiments/LAUNCHER.md`。ここは変えてはいけない条件だけを書く。

- 形式は `audio-db-experiment-v1`。実装は `src/experiments/execution/checkpoint.py`。`audio-db-training-v1` は読まない。
- 保存の方針は Selector（`src/experiments/selectors.py`）だけが持つ。Trainer は自分の更新順の中で状態が整合した点を選び `lifecycle.save_if_needed(...)` を呼ぶ。ファイルを書くかは Selector が決める。Lifecycle（`src/experiments/lifecycle.py`）の `start` と `finish` は Trainer の `fit` が一度ずつ呼ぶ。書く点では `CheckpointManager.offer` を呼ぶ。Selector が断った点は何も書かない。Trainer に保存間隔の引数を足さない。Trainer は CheckpointManager を受け取らない。出力先は `lifecycle.output_directory`、設定同一性は `lifecycle.config_identity` から読み、研究評価にも Manager は渡さない。`lifecycle.restored` は resume のときだけ真で、initialize は別実験なので偽である。YAML は `checkpoint.selector` だけを書く。標準 Selector は、間隔を省くと毎 step では保存しない。間隔の既定値は `docs/experiments/CHECKPOINT.md`。Experiment は build し、Lifecycle を注入し、`fit` を一度呼び、資源を解放する。
- Latest は `latest.pt` の 1 つだけ。共通基盤が一時ファイルに書き、検証してから `os.replace` する。失敗した保存が以前の Latest を壊す経路を作らない。Selector は Latest の名前、形式、削除に触れない。
- Archive は `archive/` の中だけ。Selector の削除は `archive/` の外に届かない。Latest と Archive はどちらも完全な checkpoint で、どちらからも再開できる。
- 公式の推論用の重み（`best_generator.pt`、`valid.loss.best.pt` など）は公式 Trainer の成果物で、checkpoint ではない。Selector に移さない。
- 保存前に rank 間の parameter、buffer、optimizer、scheduler、trainer、data、Selector の決定と状態を比べる。不一致なら書かない。
- 保存点は、Trainer が安全と宣言した点だけである。logical step を持つ Trainer では、完了した logical step の境界がそれである。蓄積窓の途中と GAN の optimizer 対の途中では書かない。`partial: true` は構築時に拒否する。開いた窓の勾配は checkpoint に入れない。中断がその窓の途中なら、最後に完了した logical step からやり直す。Core は安全地点を推定しない。
- payload は `progress` を必須にする。欠けるファイルは再開を拒否し、`logical_step` だけから progress を作らない。`progress` に `logical_step` があるとき、payload の `logical_step` はその値と一致する。キーが無い保存点では `logical_step` は null である。Core は学習単位のカウンタを Trainer に強制しない。`progress_keys` を宣言した Trainer は、Selector が要するキーを attach 時に検査する。宣言が無い Trainer は検査しない。間隔がキーを数える点でそのキーが無いときは拒否する。`boundary`、`stop`、`final` はカウンタが無くても Latest を選べる。
- 停止ポリシーは Selector と独立し、指標を評価しない。Trainer は境界処理後に指標を配送し、停止状態を更新してから保存点を `save_if_needed` する。停止設定は実験同一性に含み、状態を全 rank で照合して再開する。`TopKSelector` の順位と `PatienceStopPolicy` の監視指標は別の設定である。
- 学習ログの正本は `metrics.jsonl` である。`tensorboard/` は rank 0 がその有限スカラーから作る派生で、再開は JSONL を巻き戻したあと作り直す。イベントファイルは再開の入力にしない。
- 研究用の評価とモデル選択は、公式の検証・scheduler・公式の推論用重みのあとで動く。指標の計算、集約、選択判定は Python が行い、YAML は指標、データ、タイミング、選択基準を宣言する。公式の検証式と推論用ファイルは置き換えない。`official_faithful` と `official_reproduction` は研究用の metric、evaluator、selection を拒否する。仕様は `docs/experiments/EVALUATION.md`。
- 再開は同じ実験の続きである。`config_sha256` は `output`、`checkpoint`、`stop_after_logical_steps` を除いた設定の SHA-256 である。Optimizer と Scheduler の学習率、betas、weight decay、eps、momentum、gamma、step size などが保存時の解決済み設定と違うときだけ、状態を戻したあと現在の YAML で上書きする。有限な数値だけの Objective `weights` は構造ハッシュに入れない。係数だけの変更は、戻したモデルと Optimizer を保ち、現在の YAML の係数を使う。減衰した学習率は巻き戻さない。クラス、項の構成、形状、データ計画、出力ディレクトリが違う再開は拒否する。world size は epoch の境界（`update_index` が 0、またはその epoch の更新回数と一致する）で、`micro_batch_size * world_size * accumulation_steps` が同じときだけ変えられる。境界以外、積が変わる変更、per-rank 幅を identity に持たないデータ計画の world size 変更は拒否する。増えた rank は他 rank の RNG をコピーしない。`initialize` と `publication` は書かれているときだけハッシュに入る。無い文書は以前と同じハッシュになる。Latest より前の Archive から続けることも拒否する。古い状態から始めるのは、`initialize` を使う別の実験である。
- `initialize` は module、optimizer、scheduler の状態だけを読み、元の実験のファイルを変更しない。データ位置、Trainer の履歴、RNG は引き継がない。
- 再開は、checkpoint に保存した解決済み設定と構造ハッシュが一致する YAML だけを受ける。構造ハッシュは Optimizer と Scheduler のハイパーパラメータと、有限な数値だけの Objective `weights` を除く。Selector のクラス ID と設定はハッシュとは別に照合し、停止ポリシーもクラス ID と状態を照合する。旧 checkpoint 形式のハッシュを再開時に受け入れない。
- `COMPLETED` と `complete.json` は、全プロセスが正常終了したあとに書く。Launcher の下では Launcher が書く。一部の rank だけが学習を続ける経路を作らない。
- `provenance_sha256` が None の checkpoint は来歴なしのまま再開できる。後から同じディレクトリに JSON ができても、その checkpoint を証明済みにしない。
- FSDP と sharded checkpoint は未実装。`fp16` は CUDA autocast と GradScaler、`bf16` は CUDA autocast だけで Scaler は無い。CPU への黙った降格はしない。複数 Optimizer の `fp16` は拒否する。

## 公式再現と研究

- `official_faithful` は、その Trainer の公式条件として設定に書いた Dataset ID、Protocol ID、snapshot hash、profile、claim、`selection_uses_test`、sample rate、teacher、pipeline が、開いたスナップショットの identity と一致するときだけ許す。`selection_uses_test` は true。profile や identity が欠けたもの、`WavePairData`、別の snapshot は拒否する。独立な最終評価とは呼ばない。
- `official_faithful` の Sampler は、設定した order とその world size で解決した `order_policy` の両方が公式であるときだけ許す。MP-SENet の order は `fixed_single_shuffle_v1` と `torch_distributed_sampler_v1` の組で、world size 1 の解決は `fixed_single_shuffle_v1`、2 以上は `torch_distributed_sampler_v1` である。MossFormerGAN は `rank_mod_fixed_v1` である。`max_epoch_samples` と `max_validation_batches` は未設定だけを許す。`research` はこの制限を受けない。publication のスキーマには足さない。
- `leakage_safe` は profile が `leakage_safe` で `selection_uses_test` が false のときだけ許す。不明な値は適合にしない。snapshot の train / valid / test の重なり検査はデータモジュールが開くときに行う。
- `research` は適合する音声ペアを使え、公式再現としては記録しない。
- 公式再現設定は `configs/reproduction/experiments/official/`（`official_faithful`）。1 step の宣言は `tests/fixtures/configs/experiments/`。Dataset と Protocol の定義は `configs/data/`。評価 DB の登録は `configs/db/init.yaml`。
- 構造ハイパーパラメータは実験 YAML の `kwargs` または `kwargs.architecture` に書く。省略したキーは公開コンストラクタの既定値で埋め、YAML の値をその既定値で上書きしない。未知のキーは構築時に拒否する。仕様は `docs/experiments/MODELS.md`。
- `official_faithful` の MP-SENet と MossFormerGAN は、公開構造定数と違う architecture を構築時に差異として記録し、実行は拒否しない。比較相手はモデル自身の重みではない。`leakage_safe` と `research` はこの比較をしない。チャネルが割れない、型が違う、などの形の検査は残る。
- DeepFilterNet3 の公式データは 48 kHz のままである。モデル、DFStream、損失は同じ `DFN3Params` を使う。`scan_backend` の fused と reference は構造の変更ではない。
- 構造を変えた設定は `config_sha256` が変わり、変える前の checkpoint から再開できない。shape が違う state_dict の strict load は失敗する。
- `dfn3.online.official@1` と `semambapp.gsr.official@1` は blocked。代替素材を公式データの再現と呼ばない。
- 追跡中の `libdf_rfft.so` は置き換えない。ファイルが在るのに読めないときは、ABI が合うローカル再ビルドだけを使い、無ければエラーにする。PyTorch に戻るのは、ファイル自体が無く `DF_RFFT_REQUIRED=1` でもないときだけで、そのとき libdf とのビット一致は成り立たない（`src/models/deepfilternet3/dsp.py`）。

## 損失

- Loss Term は 1 項を計算する。Objective は注入した項を、その並びの左から加重和する。Trainer は forward、backward、蓄積、Optimizer、GAN の更新順、DDP、AMP、再開を持つ。項も Objective も backward と Optimizer を呼ばない。
- 加算順は Objective に注入した項の順である。同じ損失式を複数の学習経路へ複製しない。参照ステップは自分の係数を残し、学習経路はその定数を黙って読まない。
- 重み `1.0` は乗算しない。`0.0` でも項は実行し、その勾配は 0 である。有限な負の重みは、項が禁じていない限り許す。非有限、未知のキー、重複名、欠ける context は拒否する。実行時に YAML を書き換えない。
- `config_sha256` は損失コンポーネントの class、kwargs、inject を含む。構造ハッシュは、値が全部有限な数値である Objective `weights` だけを外す。クラス、項、数値でない重み表が違う再開は拒否する。損失の内部状態は保存しない。重みのスケジュールは未実装。学習 forward が名前付き mapping を返すときは、Trainer がそのキーを読む。`output_features` は forward が既に計算した任意テンソルの名前だけで、未知の名前は構築時に拒否する。要求していない任意テンソルは残さない。
- `official_faithful` と `leakage_safe` の MP-SENet と MossFormerGAN は、公開係数と違う Objective を構築時に差異として記録し、実行は拒否しない。`research` はこの比較をしない。SEMamba++ の `official_reproduction` と DeepFilterNet3 の公式データセットは、損失とは別に拒否されたままである。AdamW であることは更新手順の前提として残し、学習率、betas、weight decay、eps、amsgrad、ExponentialLR の gamma の違いは記録して実行する。
- モデルの STFT 幅、sample rate、local SNR の上下限は、損失の YAML 定数に複製せず、モデルから context へ渡す。

## モデルごとの更新

損失の足し方と Optimizer の順序は、`tests/numerics/reference/` の比較 Task と一致させる。Task は自分の式を持ち、本番の損失を呼んで一致したことにはしない。係数の正本は実験 YAML の Objective である。検証の統計、順位、patience は、そのモデルの `evaluator.py` に置く。DeepFilterNet3 の一連の損失は `src/models/deepfilternet3/loss.py` に置く。MP-SENet の GAN 対と進捗は `src/models/mp_senet/trainer.py`、検証式と選択は `evaluator.py` に置く。Trainer はいつ更新・検証・scheduler・保存を呼ぶかを持つ。統合済みの MP-SENet、TF-GridNet、TF-Locoformer、DeepFilterNet3、SEMamba++ は、その更新順を `trainer.py` に持つ。MossFormerGAN の更新順は `src/models/mossformergan/trainer.py` にあり、plateau の式は `evaluator.py` の `mossformer_validation_decision`、重みの復帰と学習率の変更は Trainer が行う。意味が一致する終了処理、蓄積窓、rank 停止だけを `src/experiments/training/control.py` に置く。意味が違う `_prepare_run` と `_offer` は Trainer に残す。

- `StandardTrainer` は、全 rank の micro-batch 件数が同じ正の数で窓が accumulation ちょうどのとき、損失を accumulation で割る。短い末尾や rank 間で件数が違う窓は、各 micro-batch の損失に `n_i * world_size / n_global` を掛ける。TF-GridNet、TF-Locoformer、DeepFilterNet3、MP-SENet、SEMamba++ も、窓の行数が全部同じ正の数のときはその整数で割る。行数が違うときは同じ係数を掛ける。TF-GridNet、TF-Locoformer、DeepFilterNet3 の行数 0 は forward せず、パラメータに接続した 0 を backward する。MP-SENet と SEMamba++ の行数 0 も forward せず、同じ gather と backward の回数でパラメータに接続した 0 を流す。MossFormerGAN は計画から窓の行数を読み、違うときだけ同じ係数を掛ける。計画が無い view は窓の件数で割ったままである。行数を知るために先の micro-batch は描画しない。行数 0 は forward せず、音声がある rank と同じ 2 回の agree と 2 回の backward でパラメータに接続した 0 を流す。
- 6 公式 Trainer は Preflight で 1 窓を backward し、Optimizer は踏まない。戻し処理は読み取り専用の進捗プロパティへ代入しない。MossFormerGAN でその epoch の micro-batch がすべて skip されたときは backward が無く、検査は `skipped` を残して学習を続ける。DeepFilterNet3 の検査は cosine 表を作り直さない。

- MP-SENet GAN: Discriminator を先に更新し、同じ Generator forward のグラフで Generator を更新する。この対の入口は `src/models/mp_senet/trainer.py` の `MPSENetUpdate.run` にあり、CPU/DDP の順は同じファイルの `accumulated_step`、CUDA 単一 rank で蓄積 1 の PESQ 重なりは `cuda_overlapped_step` にある。信号項と metric 項は別の Objective で、CUDA の重なり経路は信号、Discriminator、metric の順に backward する。`ExponentialLR` は epoch の micro-batch が尽きたときだけ進む。検証の式は `src/models/mp_senet/evaluator.py` にあり、参照学習 Task は使わない。rank 0 の実行、RNG、学習カーソル、shadow も `evaluator.py` にあり、いつ検証するかは Trainer が持つ。step の保存要求は両 Optimizer のあとの安全点で、ログの物理 I/O だけを次の CUDA backward に重ねてよい。`states["trainer"]` のキーと値は `src/models/mp_senet/trainer.py` の `MPTrainingProgress` が持つ。Trainer はそのオブジェクトを一つ持ち、同じフィールドを属性へ複製しない。復元は payload を検査し終えてから代入する。失敗しても既存の進捗と、そのネストした dict や list の中身と参照は変わらない。`optimizer_updates` の加算は登録した Generator と Discriminator の id だけで、payload に残った余分な id は加算しない。登録 id が欠ける payload は位置で読み替えず拒否する。`8f54ab8` より前と同じく、カウンタ値は `int` で変換する。
- MossFormerGAN: RI、magnitude、time、GAN は 1 回の再構成が作るスカラーを Context Term が読む。time は波形 L1 ではない。Objective が公開係数を掛ける。Generator の backward のあと Discriminator の backward。両 AdamW はその後。PESQ が無い micro-batch は分母に残り、損失 0 にはしない。plateau は 5 回で Generator の学習率を半分にし、10 回で止める。検証の本番の式は `src/models/mossformergan/evaluator.py` にあり、比較 Task は同じ式を自分で持つ。
- TF-GridNet: `mr_l1_tfd` を重み 1 の Objective で包む。Adam、epoch 末の `ReduceLROnPlateau`、`valid.loss` の n-best。検証は注入した Objective の total である。進捗の実体は `src/models/tf_gridnet/trainer.py` の `GridNetProgress` 一つで、Trainer は同じ値を複製しない。蓄積更新と epoch 境界も同じファイルにあり、検証と n-best は `evaluator.py` にある。復元は検査後に代入する。`optimizer_updates` の加算は登録した id だけで、余分な id は残して加算しない。
- TF-Locoformer: 同じスペクトル損失、AdamW、`WarmupReduceLROnPlateau`。warmup は Optimizer 更新のあと。最初の更新は構築時の学習率。SI-SDR は監視だけで backward に入れない。検証は注入した Objective の total である。進捗の実体は `src/models/tf_locoformer/trainer.py` の `LocoformerProgress` 一つである。蓄積更新と epoch 境界も同じファイルにあり、検証と n-best は `evaluator.py` にある。`step_batch` は適用された更新の直後、`step_validation` は epoch の検証のあとで、`scheduler.step` は使わない。`learning_rates` は使った学習率、`scheduled_rates` は次の学習率である。
- DeepFilterNet3: multi-resolution スペクトルは 1 項で、local SNR の Objective 重みは `1e-3` である。FFT、hop、sample rate、SNR の上下限はモデルの context から渡す。cosine は Optimizer 更新の前。batch schedule は world size 1 の micro-batch 幅だけを付け替える。検証は注入した Objective の total である。保存する進捗は `src/models/deepfilternet3/trainer.py` の `DFProgress` 一つで、`losses` は journal のリストであり checkpoint に入れない。cosine 表は同じファイルの `CosineTable` が持ち、epoch 長が変わったときだけ作り直す。損失手順は `loss.py`、等重み検証は `evaluator.py` にある。比較用学習 Task は `tests/numerics/reference/deepfilternet_task.py` にあり、本番は import しない。登録 Optimizer ID だけを増やし、欠けた登録 ID は復元時に拒否する。
- MossFormerGAN: Generator と Discriminator の backward のあと、Generator の AdamW、Discriminator の AdamW の順。PESQ と非有限は micro-batch を skip し、データ位置は進む。plateau は `mossformer_validation_decision` で、5 回の停滞で best を戻して Generator の学習率を半分にし、10 回で止める。保存する進捗は `src/models/mossformergan/trainer.py` の `MossProgress` 一つで、開いた蓄積窓の勾配は checkpoint に入れない。境界の保存の前にその窓は捨て、中断がその窓の途中なら最後に完了した logical step を残す。登録した 2 つの Optimizer ID だけを増やす。
- SEMamba++: Discriminator を先に更新する。蓄積 1 は Generator グラフを再利用する。蓄積 2 以上は再 forward し、DataModule は再消費しない。CQT 族と MRD 族はそれぞれ 1 項にまとめ、外側の重みは族の和へ一度だけ掛かる。波形 L1 は学習の和に入らない。`best_generator.pt` は書かない。検証は `src/models/semambapp/evaluator.py` の shadow が波形 L1 を採点する。比較 Task は同じ採点式を自分で持つ。`resampy==0.4.3` は `kaiser_best` と `kaiser_fast` に必要。保存する進捗は `src/models/semambapp/trainer.py` の `SEMambaProgress` 一つで、forward 回数と同期トレースは同じファイルの更新側の非保存観測である。両 Optimizer は登録 ID だけを増やし、欠けた登録 ID は復元時に拒否する。

Global batch は micro-batch、world size、蓄積回数の積と一致しない構成を拒否する。logical step は、蓄積 1 回ぶんの Optimizer 更新を試みた回数。AMP が更新を skip した step も logical step とデータ位置を進め、`optimizer_updates` は適用された更新だけを数える。

## データ供給

- 学習はスナップショットと純関数だけを読む。`(seed, epoch, world_size, rank, index)` から再計算し、データ側の乱数状態は保存しない。
- 固定ペア、chunk、DeepFilterNet3 のオンライン混合、SEMamba++ のオンライン劣化は別の描画。共通の混合エンジンにはまとめない。Adapter の選択はデータ供給の準備時に行い、Trainer は dataset algorithm で分岐しない。
- Trainer がデータに要求するのは sample rate、teacher kind（`dry_clean`、`reverberant_clean`、`official_mixed_target`）、波形の形状と有効長である。正規化、crop、chunk、順序、validation の rank 分割は実験 YAML の pipeline 引数であり、スナップショットの `rule_json` や生成アルゴリズム名から選ばない。公開済み `rule_json` に残る同名のフィールドは登録時の記録であり、学習はそれを上書きに使わない。サンプルレートが違うときは、明示した resample があるときだけ変換する。
- 重み付き抽選 `weighted_categorical_v1` は研究用。重みを渡さない epoch の `rank_order` は変えない。
- 実験 YAML のバッチ名は `global_batch_size` と `micro_batch_size` である。`accumulation_steps = global_batch_size / (micro_batch_size * world_size)` が正の整数でない設定は拒否する。`accumulation`、`batch_size`、`micro_batch` は受けない。コンストラクタへは、そのクラスが宣言している引数だけを渡す。起動と再開は `metadata.json` の launch に実効バッチを残す。
- world size が違う再開は、epoch の境界かつ `micro_batch_size * world_size * accumulation_steps` が同じときだけ許す。`TensorPairData` の identity は per-rank 幅を `batch_size` として持つ。chunk や online 計画などその幅を identity に持たない供給は、world size が違う再開を拒否する。
