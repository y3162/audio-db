# 学習の不変条件

学習式を「整理」で変えない。モデル固有の参照ステップは `src/experiments/official/<model>/reference.py` にあり、共有の STFT・スペクトル損失・scheduler は `src/experiments/calc/` にある。学習が backward する損失は YAML の Loss Term と Objective で組み立て、Trainer は注入された Objective を呼ぶ。参照ステップは自分の係数を持ち、テストが両方を比べる。第二の学習 CLI は作らない。仕様は `docs/experiments/LOSS.md`。

## Checkpoint

仕様は `docs/experiments/CHECKPOINT.md`、プロセスの監視は `docs/experiments/LAUNCHER.md`。ここは変えてはいけない条件だけを書く。

- 形式は `audio-db-experiment-v1`。実装は `src/experiments/execution/checkpoint.py`。`audio-db-training-v1` は読まない。
- 保存の方針は Selector（`src/experiments/selectors.py`）だけが持つ。Trainer は整合した点で `lifecycle.save_if_needed(...)` を呼ぶだけで、いつ保存するかを決めない。Lifecycle（`src/experiments/lifecycle.py`）が `start` と `finish` を一度ずつ行い、書く点では `CheckpointManager.offer` を呼ぶ。Selector が断った点は何も書かない。Trainer に保存間隔の引数を足さない。YAML は `checkpoint.selector` だけを書く。標準 Selector は、間隔を省くと毎 step では保存しない。間隔の既定値は `docs/experiments/CHECKPOINT.md`。
- Latest は `latest.pt` の 1 つだけ。共通基盤が一時ファイルに書き、検証してから `os.replace` する。失敗した保存が以前の Latest を壊す経路を作らない。Selector は Latest の名前、形式、削除に触れない。
- Archive は `archive/` の中だけ。Selector の削除は `archive/` の外に届かない。Latest と Archive はどちらも完全な checkpoint で、どちらからも再開できる。
- 公式の推論用の重み（`best_generator.pt`、`valid.loss.best.pt` など）は公式 Trainer の成果物で、checkpoint ではない。Selector に移さない。
- 保存前に rank 間の parameter、buffer、optimizer、scheduler、trainer、data、Selector の決定と状態を比べる。不一致なら書かない。
- 保存点は、Trainer が安全と宣言した点だけである。logical step を持つ Trainer では、完了した logical step の境界がそれである。GAN の optimizer 対の途中では書かない。Core は安全地点を推定しない。
- payload は `progress` を必須にする。欠けるファイルは再開を拒否し、`logical_step` だけから progress を作らない。`progress` に `logical_step` があるとき、payload の `logical_step` はその値と一致する。キーが無い保存点では `logical_step` は null である。Core は学習単位のカウンタを Trainer に強制しない。`progress_keys` を宣言した Trainer は、Selector が要するキーを attach 時に検査する。宣言が無い Trainer は検査しない。間隔がキーを数える点でそのキーが無いときは拒否する。`boundary`、`stop`、`final` はカウンタが無くても Latest を選べる。
- 停止ポリシーは Selector と独立し、指標を評価しない。Trainer は境界処理後に指標を配送し、停止状態を更新してから保存点を `save_if_needed` する。停止設定は実験同一性に含み、状態を全 rank で照合して再開する。
- 研究用の評価とモデル選択は、公式の検証・scheduler・公式の推論用重みのあとで動く。指標の計算、集約、選択判定は Python が行い、YAML は指標、データ、タイミング、選択基準を宣言する。公式の検証式と推論用ファイルは置き換えない。`official_faithful` と `official_reproduction` は研究用の metric、evaluator、selection を拒否する。仕様は `docs/experiments/EVALUATION.md`。
- 再開は同じ実験の厳密な続きである。`config_sha256` は `output`、`checkpoint`、`stop_after_logical_steps` を除いた設定の SHA-256 で、違う設定、違う出力ディレクトリ、違う world size を拒否する。`initialize` と `publication` は書かれているときだけハッシュに入る。無い文書は以前と同じハッシュになる。Latest より前の Archive から続けることも拒否する。古い状態から始めるのは、`initialize` を使う別の実験である。
- `initialize` は module、optimizer、scheduler の状態だけを読み、元の実験のファイルを変更しない。データ位置、Trainer の履歴、RNG は引き継がない。
- 再開は現行の設定ハッシュだけを受ける。Selector のクラス ID と設定はハッシュとは別に照合し、停止ポリシーもクラス ID と状態を照合する。旧設定世代のハッシュを再開時に受け入れない。
- `COMPLETED` と `complete.json` は、全プロセスが正常終了したあとに書く。Launcher の下では Launcher が書く。一部の rank だけが学習を続ける経路を作らない。
- `provenance_sha256` が None の checkpoint は来歴なしのまま再開できる。後から同じディレクトリに JSON ができても、その checkpoint を証明済みにしない。
- FSDP、sharded checkpoint、AMP BF16 は未実装。複数 Optimizer の `amp_fp16` は拒否する。

## 公式再現と研究

- `official_faithful` は、その Trainer の公式条件として設定に書いた Dataset ID、Protocol ID、snapshot hash、profile、claim、`selection_uses_test`、sample rate、teacher、pipeline が、開いたスナップショットの identity と一致するときだけ許す。`selection_uses_test` は true。profile や identity が欠けたもの、`WavePairData`、別の snapshot は拒否する。独立な最終評価とは呼ばない。
- `official_faithful` の Sampler は、設定した order とその world size で解決した `order_policy` の両方が公式であるときだけ許す。MP-SENet の order は `fixed_single_shuffle_v1` と `torch_distributed_sampler_v1` の組で、world size 1 の解決は `fixed_single_shuffle_v1`、2 以上は `torch_distributed_sampler_v1` である。MossFormerGAN は `rank_mod_fixed_v1` である。`max_epoch_samples` と `max_validation_batches` は未設定だけを許す。`research` はこの制限を受けない。publication のスキーマには足さない。
- `leakage_safe` は profile が `leakage_safe` で `selection_uses_test` が false のときだけ許す。不明な値は適合にしない。snapshot の train / valid / test の重なり検査はデータモジュールが開くときに行う。
- `research` は適合する音声ペアを使え、公式再現としては記録しない。
- 公式再現設定は `configs/reproduction/experiments/official/`（`official_faithful`）。1 step の宣言は `tests/fixtures/configs/experiments/`。Dataset と Protocol の定義は `configs/data/`。評価 DB の登録は `configs/db/init.yaml`。
- 構造ハイパーパラメータは実験 YAML の `kwargs` または `kwargs.architecture` に書く。省略したキーは公開コンストラクタの既定値で埋め、YAML の値をその既定値で上書きしない。未知のキーは構築時に拒否する。仕様は `docs/experiments/MODELS.md`。
- `official_faithful` の MP-SENet と MossFormerGAN は、公開構造定数と違う architecture を構築時に拒否する。比較相手はモデル自身の重みではない。`leakage_safe` と `research` はこの構造ロックを使わない。
- DeepFilterNet3 の公式データは 48 kHz のままである。モデル、DFStream、損失は同じ `DFN3Params` を使う。`scan_backend` の fused と reference は構造の変更ではない。
- 構造を変えた設定は `config_sha256` が変わり、変える前の checkpoint から再開できない。shape が違う state_dict の strict load は失敗する。
- `dfn3.online.official@1` と `semambapp.gsr.official@1` は blocked。代替素材を公式データの再現と呼ばない。
- 追跡中の `libdf_rfft.so` は置き換えない。ファイルが在るのに読めないときは、ABI が合うローカル再ビルドだけを使い、無ければエラーにする。PyTorch に戻るのは、ファイル自体が無く `DF_RFFT_REQUIRED=1` でもないときだけで、そのとき libdf とのビット一致は成り立たない（`src/se/models/deepfilternet3/dsp.py`）。

## 損失

- Loss Term は 1 項を計算する。Objective は注入した項を、その並びの左から加重和する。Trainer は forward、backward、蓄積、Optimizer、GAN の更新順、DDP、AMP、再開を持つ。項も Objective も backward と Optimizer を呼ばない。
- 加算順は Objective に注入した項の順である。同じ損失式を複数の学習経路へ複製しない。参照ステップは自分の係数を残し、学習経路はその定数を黙って読まない。
- 重み `1.0` は乗算しない。`0.0` でも項は実行し、その勾配は 0 である。有限な負の重みは、項が禁じていない限り許す。非有限、未知のキー、重複名、欠ける context は拒否する。実行時に YAML を書き換えない。
- `config_sha256` は損失コンポーネントの class、kwargs、inject を含む。係数が違う checkpoint は再開できない。損失の内部状態は保存しない。重みのスケジュールは未実装。
- `official_faithful` と `leakage_safe` の MP-SENet と MossFormerGAN は、公開係数と違う Objective を構築時に拒否する。`research` は項の追加、削除、重み変更ができる。SEMamba++ の `official_reproduction` と DeepFilterNet3 の公式データセットは、損失とは別に拒否されたままである。
- モデルの STFT 幅、sample rate、local SNR の上下限は、損失の YAML 定数に複製せず、モデルから context へ渡す。

## モデルごとの更新

損失の足し方と Optimizer の順序は、対応する `src/experiments/official/<model>/reference.py` の参照ステップと一致させる。係数の正本は実験 YAML の Objective である。

- MP-SENet GAN: Discriminator を先に更新し、同じ Generator forward のグラフで Generator を更新する。信号項と metric 項は別の Objective で、CUDA の重なり経路は信号、Discriminator、metric の順に backward する。`ExponentialLR` は epoch の micro-batch が尽きたときだけ進む。検証は参照タスクの式のままである。
- MossFormerGAN: RI、magnitude、time、GAN は 1 回の再構成が作るスカラーを Context Term が読む。time は波形 L1 ではない。Objective が公開係数を掛ける。Generator の backward のあと Discriminator の backward。両 AdamW はその後。PESQ が無い micro-batch は分母に残り、損失 0 にはしない。plateau は 5 回で Generator の学習率を半分にし、10 回で止める。検証は参照タスクの式のままである。
- TF-GridNet: `mr_l1_tfd` を重み 1 の Objective で包む。Adam、epoch 末の `ReduceLROnPlateau`、`valid.loss` の n-best。検証は注入した Objective の total である。
- TF-Locoformer: 同じスペクトル損失、AdamW、`WarmupReduceLROnPlateau`。warmup は Optimizer 更新のあと。最初の更新は構築時の学習率。SI-SDR は監視だけで backward に入れない。検証は注入した Objective の total である。
- DeepFilterNet3: multi-resolution スペクトルは 1 項で、local SNR の Objective 重みは `1e-3` である。FFT、hop、sample rate、SNR の上下限はモデルの context から渡す。cosine は Optimizer 更新の前。batch schedule は world size 1 の micro-batch 幅だけを付け替える。検証は注入した Objective の total である。
- SEMamba++: Discriminator を先に更新する。蓄積 1 は Generator グラフを再利用する。蓄積 2 以上は再 forward し、DataModule は再消費しない。CQT 族と MRD 族はそれぞれ 1 項にまとめ、外側の重みは族の和へ一度だけ掛かる。波形 L1 は学習の和に入らない。`best_generator.pt` は書かない。検証は参照タスクの波形 L1 のままである。`resampy==0.4.3` は `kaiser_best` と `kaiser_fast` に必要。

Global batch は micro-batch、world size、蓄積回数の積と一致しない構成を拒否する。logical step は、蓄積 1 回ぶんの Optimizer 更新を試みた回数。AMP が更新を skip した step も logical step とデータ位置を進め、`optimizer_updates` は適用された更新だけを数える。

## データ供給

- 学習はスナップショットと純関数だけを読む。`(seed, epoch, world_size, rank, index)` から再計算し、データ側の乱数状態は保存しない。
- 固定ペア、chunk、DeepFilterNet3 のオンライン混合、SEMamba++ のオンライン劣化は別の描画。共通の混合エンジンにはまとめない。Adapter の選択はデータ供給の準備時に行い、Trainer は dataset algorithm で分岐しない。
- Trainer がデータに要求するのは sample rate、teacher kind（`dry_clean`、`reverberant_clean`、`official_mixed_target`）、波形の形状と有効長である。正規化、crop、chunk、順序、validation の rank 分割は実験 YAML の pipeline 引数であり、スナップショットの `rule_json` や生成アルゴリズム名から選ばない。公開済み `rule_json` に残る同名のフィールドは登録時の記録であり、学習はそれを上書きに使わない。サンプルレートが違うときは、明示した resample があるときだけ変換する。
- 重み付き抽選 `weighted_categorical_v1` は研究用。重みを渡さない epoch の `rank_order` は変えない。
- world size が違う再開は拒否する。
