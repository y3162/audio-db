# 学習の不変条件

学習式を「整理」で変えない。モデル固有の参照ステップは `src/experiments/official/<model>/reference.py` にあり、共有の STFT・スペクトル損失・scheduler は `src/experiments/calc/` にある。テストが公式実装や連続実行と比べる。Trainer はその関数を呼ぶ。第二の学習 CLI は作らない。

## Checkpoint

- 形式は `audio-db-experiment-v1`。必須の意味は `src/experiments/execution/checkpoint.py`。公開 import は `src.experiments.checkpoint`。
- `config_sha256` は `output` と `stop_after_logical_steps` を除いた設定の SHA-256。クラスパスは YAML の公開 import の文字列である。`core/`、`execution/`、`training/` への分割は、この文字列を変えないので新しい世代のハッシュを足さない。違う設定の resume は拒否する。
- `WarmupReduceLROnPlateau` のクラスパスは `src.experiments.calc.optim.WarmupReduceLROnPlateau`。再開時だけ、移動前の `src.training.se.optim.WarmupReduceLROnPlateau` で計算したハッシュも同じ設定として受ける。公式コンポーネントへ移した Model と Trainer は、移動前の `src.experiments.<model>` と `src.experiments.models.*` のハッシュも受ける。scheduler の旧パスと公式コンポーネントの旧パスを同時に持つ世代も受ける。新しい保存は現在のパスのハッシュを書く。既存の `data_provenance.json` は、その世代のクラスパスで計算したときだけ同じ run として残す。
- `provenance_sha256` が None の checkpoint は来歴なしのまま再開できる。後から同じディレクトリに JSON ができても、その checkpoint を証明済みにしない。
- 保存前に rank 間の parameter、buffer、optimizer、scheduler、trainer、data を比べる。不一致の完成ファイルは置かない。
- SIGTERM は完了した logical step の境界で書く。GAN の optimizer 対の途中では書かない。
- FSDP、sharded checkpoint、AMP BF16 は未実装。複数 Optimizer の `amp_fp16` は拒否する。

## 公式再現と研究

- `official_faithful` は、公式が test を選択に使うなら使う。`selection_uses_test` は true。独立な最終評価とは呼ばない。
- `leakage_safe` は test を選択に使わない。profile または `selection_uses_test` が claim と矛盾する設定は拒否する。
- 本格候補は `configs/reproduction/experiments/official/`（`official_faithful`）と `configs/reproduction/experiments/variants/`（`leakage_safe` と比較用の代替条件）。1 step の宣言は `tests/fixtures/configs/experiments/`。Dataset と Protocol の定義は `configs/data/`。評価 DB の登録は `configs/db/init.yaml`。
- `dfn3.online.official@1` と `semambapp.gsr.official@1` は blocked。代替素材を公式データの再現と呼ばない。
- 追跡中の `libdf_rfft.so` は置き換えない。読めないときは、ABI が合うローカル再ビルドだけを使い、PyTorch fallback にしない。

## モデルごとの更新

損失の足し方と Optimizer の順序は、対応する `src/experiments/official/<model>/reference.py` の参照ステップと一致させる。

- MP-SENet GAN: Discriminator を先に更新し、同じ Generator forward のグラフで Generator を更新する。`ExponentialLR` は epoch の micro-batch が尽きたときだけ進む。
- MossFormerGAN: Generator の backward のあと Discriminator の backward。両 AdamW はその後。PESQ が無い micro-batch は分母に残り、損失 0 にはしない。plateau は 5 回で Generator の学習率を半分にし、10 回で止める。
- TF-GridNet: `mr_l1_tfd`、Adam、epoch 末の `ReduceLROnPlateau`、`valid.loss` の n-best。
- TF-Locoformer: 同じスペクトル損失、AdamW、`WarmupReduceLROnPlateau`。warmup は Optimizer 更新のあと。最初の更新は構築時の学習率。
- DeepFilterNet3: 出荷の multi-resolution スペクトルと local SNR。cosine は Optimizer 更新の前。batch schedule は world size 1 の micro-batch 幅だけを付け替える。
- SEMamba++: Discriminator を先に更新する。蓄積 1 は Generator グラフを再利用する。蓄積 2 以上は再 forward し、DataModule は再消費しない。`best_generator.pt` は書かない。`resampy==0.4.3` は `kaiser_best` と `kaiser_fast` に必要。

Global batch は micro-batch、world size、蓄積回数の積と一致しない構成を拒否する。logical step は Optimizer 更新の回数。

## データ供給

- 学習はスナップショットと純関数だけを読む。`(seed, epoch, world_size, rank, index)` から再計算し、データ側の乱数状態は保存しない。
- 固定ペア、chunk、DeepFilterNet3 のオンライン混合、SEMamba++ のオンライン劣化は別の描画。共通の混合エンジンにはまとめない。
- 重み付き抽選 `weighted_categorical_v1` は研究用。重みを渡さない epoch の `rank_order` は変えない。
- world size が違う再開は拒否する。
