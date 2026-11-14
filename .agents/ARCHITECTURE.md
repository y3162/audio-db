# 学習の不変条件

学習式を「整理」で変えない。モデル固有の参照ステップは `src/experiments/official/<model>/reference.py` にあり、共有の STFT・スペクトル損失・scheduler は `src/experiments/calc/` にある。テストが公式実装や連続実行と比べる。Trainer はその関数を呼ぶ。第二の学習 CLI は作らない。

## Checkpoint

仕様は `docs/experiments/CHECKPOINT.md`、プロセスの監視は `docs/experiments/LAUNCHER.md`。ここは変えてはいけない条件だけを書く。

- 形式は `audio-db-experiment-v1`。実装は `src/experiments/execution/checkpoint.py`。`audio-db-training-v1` は読まない。
- 保存の方針は Selector（`src/experiments/selectors.py`）だけが持つ。Trainer は整合した点を `checkpoint.offer(...)` で差し出すだけで、いつ保存するかを決めない。Trainer に保存間隔の引数を足さない。YAML は `checkpoint.selector` だけを書く。標準 Selector は、間隔を省くと毎 step では保存しない。間隔の既定値は `docs/experiments/CHECKPOINT.md`。
- Latest は `latest.pt` の 1 つだけ。共通基盤が一時ファイルに書き、検証してから `os.replace` する。失敗した保存が以前の Latest を壊す経路を作らない。Selector は Latest の名前、形式、削除に触れない。
- Archive は `archive/` の中だけ。Selector の削除は `archive/` の外に届かない。Latest と Archive はどちらも完全な checkpoint で、どちらからも再開できる。
- 公式の推論用の重み（`best_generator.pt`、`valid.loss.best.pt` など）は公式 Trainer の成果物で、checkpoint ではない。Selector に移さない。
- 保存前に rank 間の parameter、buffer、optimizer、scheduler、trainer、data、Selector の決定と状態を比べる。不一致なら書かない。
- 保存点は完了した logical step の境界と、Trainer が安全と宣言した点だけ。GAN の optimizer 対の途中では書かない。
- 停止ポリシーは Selector と独立し、指標を評価しない。Trainer は境界処理後に指標を配送し、停止状態を更新してから保存点を offer する。停止設定は実験同一性に含み、状態を全 rank で照合して再開する。
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
- `dfn3.online.official@1` と `semambapp.gsr.official@1` は blocked。代替素材を公式データの再現と呼ばない。
- 追跡中の `libdf_rfft.so` は置き換えない。ファイルが在るのに読めないときは、ABI が合うローカル再ビルドだけを使い、無ければエラーにする。PyTorch に戻るのは、ファイル自体が無く `DF_RFFT_REQUIRED=1` でもないときだけで、そのとき libdf とのビット一致は成り立たない（`src/se/models/deepfilternet3/dsp.py`）。

## モデルごとの更新

損失の足し方と Optimizer の順序は、対応する `src/experiments/official/<model>/reference.py` の参照ステップと一致させる。

- MP-SENet GAN: Discriminator を先に更新し、同じ Generator forward のグラフで Generator を更新する。`ExponentialLR` は epoch の micro-batch が尽きたときだけ進む。
- MossFormerGAN: Generator の backward のあと Discriminator の backward。両 AdamW はその後。PESQ が無い micro-batch は分母に残り、損失 0 にはしない。plateau は 5 回で Generator の学習率を半分にし、10 回で止める。
- TF-GridNet: `mr_l1_tfd`、Adam、epoch 末の `ReduceLROnPlateau`、`valid.loss` の n-best。
- TF-Locoformer: 同じスペクトル損失、AdamW、`WarmupReduceLROnPlateau`。warmup は Optimizer 更新のあと。最初の更新は構築時の学習率。
- DeepFilterNet3: 出荷の multi-resolution スペクトルと local SNR。cosine は Optimizer 更新の前。batch schedule は world size 1 の micro-batch 幅だけを付け替える。
- SEMamba++: Discriminator を先に更新する。蓄積 1 は Generator グラフを再利用する。蓄積 2 以上は再 forward し、DataModule は再消費しない。`best_generator.pt` は書かない。`resampy==0.4.3` は `kaiser_best` と `kaiser_fast` に必要。

Global batch は micro-batch、world size、蓄積回数の積と一致しない構成を拒否する。logical step は、蓄積 1 回ぶんの Optimizer 更新を試みた回数。AMP が更新を skip した step も logical step とデータ位置を進め、`optimizer_updates` は適用された更新だけを数える。

## データ供給

- 学習はスナップショットと純関数だけを読む。`(seed, epoch, world_size, rank, index)` から再計算し、データ側の乱数状態は保存しない。
- 固定ペア、chunk、DeepFilterNet3 のオンライン混合、SEMamba++ のオンライン劣化は別の描画。共通の混合エンジンにはまとめない。Adapter の選択はデータ供給の準備時に行い、Trainer は dataset algorithm で分岐しない。
- Trainer がデータに要求するのは sample rate、teacher kind（`dry_clean`、`reverberant_clean`、`official_mixed_target`）、波形の形状と有効長である。正規化、crop、chunk、順序、validation の rank 分割は実験 YAML の pipeline 引数であり、スナップショットの `rule_json` や生成アルゴリズム名から選ばない。公開済み `rule_json` に残る同名のフィールドは登録時の記録であり、学習はそれを上書きに使わない。サンプルレートが違うときは、明示した resample があるときだけ変換する。
- 重み付き抽選 `weighted_categorical_v1` は研究用。重みを渡さない epoch の `rank_order` は変えない。
- world size が違う再開は拒否する。
