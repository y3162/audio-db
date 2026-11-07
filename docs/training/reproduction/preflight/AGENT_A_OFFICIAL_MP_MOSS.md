# P10-C Preflight Agent A: 公式実装監査（MP-SENet / MossFormerGAN）

発見者は Agent A。根拠は公式 checkout の実行コード、現行 `src/` の実行コード、実 YAML、この監査で実行した CPU テストだけである。過去の監査メモや「一致」と書かれた文書は使っていない。

CUDA は使っていない。`CUDA_VISIBLE_DEVICES=""` で起動し、`torch.cuda.is_available()` は False、`device_count` は 0 だった。GPU 上の既存プロセスには触れていない。リポジトリの `src/`、YAML、テスト、Dataset、Snapshot、DuckDB、checkpoint、実験結果は変更していない。一時スクリプトは `/tmp/p10a_audit.py` と `/tmp/p10a_audit2.py`。

## 1. 参照した公式 commit（実測 HEAD）

| 対象 | パス | 実測 HEAD | 期待 | git status |
|---|---|---|---|---|
| MP-SENet | `src/se/official/mp-senet` | `89932cfe90d1dacb8e170e4a331d762462c21792` | 一致 | 未追跡は `__pycache__/` と `models/__pycache__/` のみ |
| MossFormerGAN (ClearerVoice-Studio) | `src/se/official/clearervoice-studio` | `6b3774dc79c46ae8bed2a4fa5f706f0ac8c75c61` | 一致 | 作業ツリーはクリーン |

Python は `/net/spring/work/seki/exp/audio-db/.venvs/se/bin/python`、torch `2.6.0+cu124`。CPU テスト中は CUDA を無効化した。

## 2. 監査済み領域と読んだファイル

監査した実行経路は、モデル構築、初期化乱数、forward、損失、Discriminator、Optimizer、Scheduler、Validation、Checkpoint、前処理、train/eval の切替、乱数を消費する順序である。

公式 MP-SENet:

- `src/se/official/mp-senet/train.py`
- `src/se/official/mp-senet/config.json`
- `src/se/official/mp-senet/dataset.py`
- `src/se/official/mp-senet/models/model.py`
- `src/se/official/mp-senet/models/discriminator.py`
- `src/se/official/mp-senet/models/transformer.py`
- `src/se/official/mp-senet/utils.py`（`LearnableSigmoid2d`。matplotlib は import 時だけ）

現行 MP-SENet:

- `src/se/models/mp_senet/model.py`
- `src/se/models/mp_senet/transformer.py`（公式と `diff -q` で同一）
- `src/se/models/mp_senet/sigmoid.py`
- `src/se/api.py` の `_build_mp`
- `src/training/se/mp_senet.py`
- `src/training/se/losses.py`
- `src/training/se/discriminator.py`
- `src/training/se/stft.py`
- `src/training/core/trainer.py` の通常 epoch と reproduction の `_run_data_epochs`、best、resume、validation 集約
- `src/training/se/config.py` の optimizer / reproduction データ解釈
- `src/training/data/dataset.py` の RMS と crop
- `src/evaluation/metrics/pesq.py`
- `configs/training/mp_senet.yaml`
- `configs/training/reproduction/mp_senet_vb28_official.yaml`
- `configs/data/protocols/mp_senet.fixed_pair.ondemand.yaml`
- `src/reproduction/pipeline/renderer.py`
- `src/reproduction/pipeline/transform.py`
- `src/reproduction/ids.py` の `fixed_pair.mp_senet`

公式 MossFormerGAN:

- `train/speech_enhancement/train.py`
- `train/speech_enhancement/train.sh`
- `train/speech_enhancement/networks.py`
- `train/speech_enhancement/solver.py`（`train` と `_run_one_epoch_mossformergan_se_16k`、checkpoint）
- `train/speech_enhancement/losses/loss.py` の `loss_mossformergan_se_16k`
- `train/speech_enhancement/utils/misc.py` の `stft` / `istft` / `power_compress` / `power_uncompress` / `batch_pesq`
- `train/speech_enhancement/dataloader/dataloader.py` の `audio_norm` と `Wave_Processor`
- `train/speech_enhancement/models/mossformer_gan/generator.py` の `MossFormerGAN_SE_16K` と `SyncANet.forward`
- `train/speech_enhancement/models/mossformer_gan/discriminator.py` の `Discriminator` と `BatchNorm2d`
- `train/speech_enhancement/config/train/MossFormerGAN_SE_16K.yaml`
- インストール済み `pesq/__init__.py` の `pesq()` シグネチャ

現行 MossFormerGAN:

- `src/training/se/mossformergan.py`
- `src/training/se/mossformer_discriminator/__init__.py`
- `src/training/se/optim.py` の `build_optimizer`
- `src/se/models/mossformergan/generator.py` の `SyncANet.forward`
- `src/se/models/mossformergan/inference.py` の `build`
- `src/se/models/mossformergan/conv_module.py` の `BatchNorm1d` 返却
- `src/training/core/distributed.py` の `wrap_module`
- `src/training/core/rng.py` の `seed_model_init` と `set_determinism`
- `src/training/core/batch.py` の batch 分解
- `configs/training/mossformergan.yaml`
- `configs/training/reproduction/mossformergan_vb28_official.yaml`
- `configs/training/reproduction/mossformergan_vb28_official_2gpu.yaml`
- `configs/data/protocols/mossformergan_se_16k.fixed_pair.ondemand.yaml`
- `src/reproduction/pipeline/planner.py` の validation crop
- `src/reproduction/pipeline/training_data.py` の `epoch` / `validation` / `data_state`

PyTorch 2.6.0 の `torch/optim/lr_scheduler.py` にある `LRScheduler._initial_step`、`ExponentialLR.get_lr`、`AdamW` のデフォルトも読んだ。

## 3. 未監査・UNKNOWN

- DuckDB と Snapshot は開いていない。YAML コメントにある 2,893 や 5,786 updates/epoch は実測していない。plateau が検証と同じ step で進むかは、この件数と `validation_interval` が一致するときに限る（後述）。
- 48 kHz 実音声を soxr と librosa で比べていない。波形の数値差は UNKNOWN。
- 2 GPU の DDP / SyncBatchNorm は起動していない。差は呼び出しの有無まで。
- MP-SENet Generator の全パラメータ勾配は測っていない。出力とスカラー損失は誤差 0。勾配差を実測したのは MaxPool の同値タイだけ。
- MossFormer の全ブロックを行単位で diff していない。同一重み・短いスペクトログラムの forward は誤差 0。
- 公式 `joblib` PESQ と現行 `metric_targets` を、同じ 4 件以上の実発話で比べていない。無音 1 件では両方とも「スコアなし」だった。
- `score_pesq` の境界ガードが VoiceBank で発火するかは UNKNOWN。試した 1 秒正弦とランダム 8 本では発火しなかった。
- DDP の `find_unused_parameters=True` が現行で必要かは UNKNOWN。未使用パラメータの有無は見ていない。
- 公式 `train.sh` が指す finetune checkpoint はこの checkout に無い。ファイルが有る場合の finetune 経路は実行していない。

## 4. 実行したテストと結果

共通の起動は次のとおり。

```text
CUDA_VISIBLE_DEVICES="" /net/spring/work/seki/exp/audio-db/.venvs/se/bin/python -u /tmp/p10a_audit.py
CUDA_VISIBLE_DEVICES="" /net/spring/work/seki/exp/audio-db/.venvs/se/bin/python -u /tmp/p10a_audit2.py
```

環境: torch 2.6.0+cu124、`cuda_available False`、`device_count 0`。

### 4.1 AdamW

`torch.optim.AdamW([param], lr=5e-4)` の `weight_decay` は `0.01`、`betas` は `(0.9, 0.999)`、`eps` は `1e-8`。`weight_decay=1e-5` を明示すると `1e-05` になる。公式 `train.py` は `weight_decay` を渡さない。

### 4.2 ExponentialLR

初期 `lr=0.0005`、`gamma=0.99`。

- コンストラクタ直後: `lr=0.0005`、`last_epoch=0`。
- `step()` を 3 回: `lr=0.0004851495`（`0.0005 * 0.99**3`）、`last_epoch=3`。
- その optimizer を読み、公式と同じく `ExponentialLR(..., last_epoch=5)` を作り直す: `lr=0.000480298005`（直前の lr をさらに 0.99 倍）、`last_epoch=6`。
- 現行と同じく optimizer を読んだあと `load_state_dict`: `lr=0.0004851495`、`last_epoch=3` のまま。

`ExponentialLR.get_lr` は `last_epoch==0` 以外で現在の lr を `gamma` 倍する。コンストラクタは `_initial_step` から `step()` を 1 回呼ぶ。再開時に `last_epoch>=0` を渡すと、checkpoint 済みの lr がもう 1 回 decay する。

### 4.3 MP-SENet STFT、forward、損失、Pool

長さ 1600、1650、12345、`n_fft=400`、`hop=100`、`win=400`、`compress=0.3`。公式 `mag_pha_stft` / `mag_pha_istft` と現行の最大絶対誤差はすべて `0`。

iSTFT 長は 1600、1600、12300。入力長が hop の倍数でないとき、公式も現行も元の長さには戻らない。`align_validation_length(1650, 100, 400)` は 1600、`12345` は 12300。

同一 state_dict の公式 `MPNet` と現行 `build_model("mp_senet")`、eval、入力長 1600:

- state_dict のキーは一致。
- magnitude / phase / complex の最大絶対誤差は `0`。
- 長さ 1650 とその先頭 1600 は、STFT フレーム数がどちらも `(1, 201, 17)` でも、Generator 出力 magnitude の最大絶対誤差は `1.563`。

同一 state_dict の公式 `MetricDiscriminator` と現行、同じ dropout シード: forward 最大絶対誤差 `0`。キーも一致し、`load_state_dict` の missing / unexpected は空。

`AdaptiveMaxPool2d(1)` と `amax` :

- タイのないランダム入力: forward 誤差 `0`、勾配誤差 `0`。
- 全要素 0 の 2x2: forward は一致。勾配は公式側 `[1, 0, 0, 0]`、現行側 `[0.25, 0.25, 0.25, 0.25]`。最大絶対誤差 `0.75`。

同じテンソルで現行 `generator_losses` と公式の式を計算: total `3.06674790`、差 `0`。PESQ 目標ありの Discriminator 損失の差 `0`。目標 `None` の損失は real 項だけと差 `0`。

公式ループのカウンタだけを写した確認（`train.py` 自体は動かしていない）:

- 公式の条件は更新のあと、`steps += 1` の前。検証が走る更新番号は 5001 と 10001 で、そのときログ上の `steps` は 5000 と 10000。
- 現行は `global_step += 1` のあと `global_step % 5000 == 0`。検証は 5000 と 10000 番目の更新。

### 4.4 PESQ、RMS、audio_norm

`pesq` のシグネチャは `(fs, ref, deg, mode='wb', on_error=PesqError.RAISE_EXCEPTION)`。

- float32 の 1 秒正弦と 0.8 倍: raw も `score_pesq` も `4.63562536239624`、差 `0`。
- 長さ 16050 と 16000: 公式呼び出しも `score_pesq` も `4.643888473510742`。この版は長さ差だけでは例外にしない。
- 無音: raw は `NoUtterancesError`。`score_pesq` は `na` / `zero_energy_audio`。`metric_targets` は `None`。
- ランダム 2 秒 8 本では `score_pesq` の境界ガードは発火しなかった。

`audio_norm` と同じ式の現行 `independent_audio_norm`: 長さ 4000 の正規乱数で最大絶対誤差 `5.313e-9`（現行が float32 で返す分）。定数波形は公式が NaN、現行は `ReproductionError`。

`noisy_rms_unit` と `sqrt(length / sum(square))` の torch 実装: 最大絶対誤差約 `4.8e-7`。

### 4.5 MossFormerGAN の信号処理、Discriminator、初期化

長さ 3200、batch 2、CPU。

- noisy のエネルギー正規化スケールの差 `0`。
- `stft(..., center=True)` と現行 `_stft` の差 `0`。shape `(2, 201, 33, 2)`。
- `power_compress` の差 `0`。
- clean 側の圧縮 real / magnitude の差 `0`。
- `power_uncompress` の差 `0`。
- `istft` は両方 `center=False`、shape `(2, 3600)`、差 `0`。時間損失 `1.0644199848175049`、差 `0`。iSTFT は入力 3200 より長く、時間損失は先頭を min 長で切る。公式も現行も同じ。

Discriminator、seed 777 でそれぞれを最初の乱数消費者として構築: checksum は両方 `7299.69091796875`。重みコピー後の forward 差 `0`、shape `(1, 1)`。`train()` のまま `torch.no_grad()` で 1 回 forward すると buffer が動く（先頭 3 本の最大変化 `0.0210`、`0.0796`、`0.2051`）。その後 `eval()` の forward では buffer 変化の最大が `0`。

`MossFormerGAN_SE_16K` を seed 777 から 2 回構築し、現行の `SyncANet` と `Discriminator` を同じ seed から 1 回ずつ構築した。

| 重み | checksum | パラメータ数 |
|---|---|---|
| 公式 1 回目の Generator（公式が残す方） | `114801.781250` | 3,128,639 |
| 公式 1 回目の Discriminator（捨てる方） | `7127.819336` | 8,125,011 |
| 公式 2 回目の Generator（捨てる方） | `114509.164062` | 同じ |
| 公式 2 回目の Discriminator（公式が残す方） | `7729.488281` | 同じ |
| 現行の 1 回初期化 Generator | `114801.781250` | 差 0 |
| 現行の 1 回初期化 Discriminator | `7127.819336` | 公式が残す D との差 `601.67` |

現行 Generator に公式 1 回目の state_dict を載せて、`[1, 2, 6, 201]` を eval で forward: real / imag の最大絶対誤差 `0`。初期化 2 回とこの forward で約 92 秒。

### 4.6 Plateau 関数

`mossformer_validation_decision` に次の損失列を入れた。

`1.0, 1.0, 0.9, 1.2, 1.1, 1.3, 1.4, 0.5, 0.5, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6`

結果の要点:

- 最初の `1.0` は `improve`。次の同値 `1.0` は `hold`、stall 1。同値は改善ではない。
- `0.9` のあと `1.2` は `hold` で、記録上の best が `0.9` から `1.2` に置き換わる。
- その次の `1.1` は `1.2` より小さいので `improve` になる。履歴最小の `0.9` より悪い。
- `0.5` で改善したあと、非改善が 5 回目（`1.1`）で `halve`。stall は 5 のまま残る。
- さらに非改善が続き、stall 10（`1.6`）で `stop`。

## 5. 一致した範囲（証拠付き）

### MP-SENet

- ハイパーパラメータ。公式 `config.json` は `lr=0.0005`、`adam_b1=0.8`、`adam_b2=0.99`、`lr_decay=0.99`、`segment_size=32000`、`n_fft=400`、`hop_size=100`、`win_size=400`、`compress_factor=0.3`、`beta=2.0`、`dense_channel=64`、`seed=1234`。再現 YAML `configs/training/reproduction/mp_senet_vb28_official.yaml` はこれと同じ値を optimizer、scheduler、model に持つ。`batch_size` は 4。
- AdamW。公式 `train.py` 65–66 行は `weight_decay` を渡さない。PyTorch 2.6 のデフォルトは 0.01。現行 YAML も 0.01 で、`MPSENetTask` は `optimizer["weight_decay"]` を渡す。
- ブロック数。公式 `MPNet` のデフォルトは 4。`config.json` の `num_tsconformers` はコンストラクタに渡っていない。現行 `_build_mp` は YAML の 4 を渡す。監査した YAML では両方 4。
- STFT と iSTFT。セクション 4.3 で最大絶対誤差 0。圧縮、`atan2` の微小項、Hann 窓を含む。
- Generator forward。同一重み・長さ 1600 で誤差 0。`transformer.py` はファイルが同一。
- Discriminator forward。同一重み・同じ dropout シードで誤差 0。構造の違いは `AdaptiveMaxPool2d(1)` と `amax` で、タイがなければ勾配も一致した。
- 更新順。公式 `train.py` 132–184 行は Discriminator を更新してから Generator。現行 `_official` も `optim_d.step()` のあと Generator backward。PESQ が無いときの Discriminator 損失は real 項だけになり、テストでも差 0。
- 損失の式。magnitude 0.9、phase 0.3、complex と consistency は MSE を 2 倍してから 0.1、metric 0.05、time 0.2。スカラー損失の差は 0。
- 学習セグメントに対する前処理の回数。公式 `Dataset.__getitem__` は全長を noisy RMS で 1 回正規化し、そのあと 32000 点を切る。現行 reproduction は `render_sample` が `noisy_rms_unit_v1` を 1 回かけ、そのあと window。`MPSENetTask._generate` は波形 STFT を 1 回だけ行い、データ側では STFT しない。32000 は hop 100 の倍数なので、学習更新では `align_validation_length` が長さを変えない。
- fresh start の ExponentialLR。両方とも epoch 終了で 1 回 `step()`。公式は `train.py` 277–278 行、loader の外。現行 reproduction は `Trainer._run_data_epochs` 465–466 行で、`epoch_done` のときだけ。`validation_interval` の step では scheduler を進めない。検証間隔と scheduler 間隔は分離されている。
- best の epoch 番号。公式は 0 始まりの `epoch >= best_checkpoint_start_epoch`（`train.py` 266 行、デフォルト 40）。表示だけ `epoch+1`。現行も 0 始まりの epoch を `_consider_best` に渡す。比較は両方ともより大きい PESQ だけを採用する（公式は `>`、現行は `<=` で棄却）。
- 学習中のモード。両方とも更新前に Generator と Discriminator を `train()`。公式の検証は Generator だけ `eval()` にし、Discriminator は呼ばない。

### MossFormerGAN

- 公式が実際に Optimizer へ渡す値。`train.py` 54–56 行は `AdamW(parameters, lr=init_learning_rate)` と、Discriminator にはその 2 倍。`betas` も `weight_decay` も渡さない。デフォルトは `(0.9, 0.999)` と `0.01`。YAML の `weight_decay: 0.00001` はこのコンストラクタに届かない。現行は `OFFICIAL_ADAMW_WEIGHT_DECAY = 0.01` を強制し、YAML の betas は `(0.9, 0.999)`、Discriminator の lr は 2 倍。監査した再現 YAML の `weight_decay` も 0.01。
- 損失の係数と時間項のペア。公式 `loss.py` 49–57 行は clean の real と予測 imag で iSTFT し、`0.1*RI + 0.9*mag + 0.2*time + 0.05*GAN`。現行 `_generator_terms` は同じ組み合わせ。STFT、圧縮、iSTFT、時間損失のテスト誤差は 0。
- 前処理は 2 段で、3 段目は無い。公式 loader の `audio_norm`（-25 dBFS を 2 回）のあと、`solver.py` 266–268 行が noisy だけを `sqrt(T / (sum(x^2)+1e-6))` で再スケールし、`loss.py` 29–30 行が同じ係数を clean に掛ける。現行 reproduction は `independent_audio_norm_v1` のあと `_generator_terms` が同じ係数を掛ける。式の誤差は `audio_norm` で `5.3e-9`、スケールで 0。
- 更新順。公式 YAML は `accu_grad: 1`、`batch_size: 1`、`effec_batch_size: 2` なので蓄積分岐。`solver.py` 277–291 行は Generator backward、Discriminator backward、そのあと Generator step、Discriminator step。損失は `(effec/batch)` で割る。現行 1 GPU YAML は global 2、micro 1 なので `batch.py` の分解は蓄積 2。`train_update` も Generator backward、Discriminator backward、G step、D step。
- Gradient clip は 10。公式は各 micro-batch の backward の直後。現行も各 micro-batch の backward の直後。
- 検証損失は学習と同じ Generator 目的。公式は `state!='train'` でも `loss_gen` を足し、`None` のバッチは足さず分母 `(i+1)` に残す（`solver.py` 326–329 行）。現行 `validation_report` は PESQ 失敗を 0 にして status `ok` のまま返す。`_combine_metric` は status `ok` の平均なので、バッチ 1 件なら 0 が分母に残る。
- 検証中の Discriminator モード。公式 `solver.train` 197 行は `self.model.eval()` だけで、Discriminator は `train()` のまま。`torch.no_grad()` でも `BatchNorm2d` の running stats は更新されることを、公式 Discriminator で確認した。現行 `eval_mode` と `validation_report` は Generator だけ `eval()`、Discriminator は `train()`。
- Plateau の比較。公式 `solver.py` 215–236 行は `val_loss >= best` を非改善とし、その損失で `best_val_loss` を上書きし、5 回目で Generator の lr だけ半分、10 回目で停止する。half のあと stall は戻さない。現行 `mossformer_validation_decision` は同じで、CPU の数列でも同値は hold、悪化で best が上書きされ、5 で halve、10 で stop だった。`_restore_best_and_halve_generator` が掛けるのは `optim_g` だけ。
- 1 GPU の実効バッチ。公式 YAML の per-GPU effective batch は 2。現行 `mossformergan_vb28_official.yaml` は global 2、micro 1、world 1 で蓄積 2。
- 2 GPU の本数と global batch。`train.sh` は `n_gpu=2`。各 GPU の effective batch 2 を DDP が平均すると global 4。現行 `mossformergan_vb28_official_2gpu.yaml` は global 4、micro 1。world 2 なら蓄積 2。このファイルを 1 プロセスで起動すると蓄積 4 になる、という YAML の注意は `batch.py` の式と一致する。
- この checkout では from-scratch の初期 lr が 0.0005。`train.sh` は finetune パスを渡すが、指している checkpoint ファイルはツリーに無い。`_load_pretrained_model` はファイルが無いとき初期 lr を finetune 値へ書き換えない。

## 6. Issue 一覧

### P10P-A-001

- 発見者: Agent A
- 対象: MP-SENet の検証と checkpoint の更新番号
- 種別: BUG
- 深刻度: MEDIUM
- 確度: CONFIRMED
- 場所: 公式 `train.py` の `train`、223–275 行。現行 `trainer.py` の `_run_data_epochs`、450–466 行、および `_should_save`、1139–1144 行。`_run_epochs` 344–358 行も同じ比較。
- 根拠: 公式は更新のあと、`steps += 1` の前に `steps % validation_interval == 0 and steps != 0` を見る。カウンタ確認では、ログ上の steps 5000 は 5001 番目の更新のあと。現行はインクリメント後の `global_step % 5000 == 0` なので 5000 番目。checkpoint も同じカウンタ。
- 再現手順: セクション 4.3 のカウンタ。公式ループは 5001 と 10001、現行は 5000 と 10000。
- 影響: 再現 YAML の 5000 間隔は、公式より 1 更新早い重みを検証し保存する。損失の式は変わらない。best の PESQ も 1 更新ずれた重みで比較される。
- 対応方針: 公式に合わせるなら、検証と保存を「その番号の更新が終わったあと、カウンタを進める前」に置く。実験の意図が「5000 更新ごと」なら現状を維持し、公式ログの steps とは 1 ずれると記録する。
- P10-C影響: NON-BLOCKER
- 検証状態: FAIL

### P10P-A-002

- 発見者: Agent A
- 対象: MP-SENet 検証入力の hop 切り詰め
- 種別: OFFICIAL_DIFF
- 深刻度: HIGH
- 確度: CONFIRMED
- 場所: 現行 `mp_senet.py` の `validation_report`、159–166 行、`align_validation_length`、356–366 行。公式 `train.py` の検証、231–245 行と `dataset.py` の `mag_pha_istft`、22–28 行。検証 `Dataset` は `split=False`（`train.py` 88–89 行）で全長を渡す。
- 根拠: 長さ 1650 の iSTFT は両方とも 1600 になる。公式は 1650 のまま STFT してモデルへ入れる。現行は先に 1600 へ切る。フレーム数はどちらも 17 でも、Generator magnitude の最大絶対誤差は 1.563。学習長 32000 はこの分岐に入らない。
- 再現手順: セクション 4.3。`align_validation_length(1650)` は 1600。同一モデルで全長と切り詰めの出力差は 1.563。
- 影響: 検証 PESQ と `g_best` に使う波形が公式と違う。hop 100 で割り切れない検証発話が対象。best の選択が変わる。
- 対応方針: 公式の検証に合わせるなら、モデルには全長を入れ、iSTFT 後の長さ差は PESQ 側の扱いに任せる。切り詰めを残すなら、best は公式の PESQ 最大化ではないと実験記録に書く。
- P10-C影響: BLOCKER
- 検証状態: FAIL

### P10P-A-003

- 発見者: Agent A
- 対象: MP-SENet 検証 PESQ の失敗サンプルの平均
- 種別: OFFICIAL_DIFF
- 深刻度: HIGH
- 確度: CONFIRMED
- 場所: 公式 `models/model.py` の `eval_pesq` と `pesq_score`、177–194 行。`train.py` 257 行と 266–268 行。現行 `mp_senet.py` の `_reference_score` と `_combine_scores`、400–425 行。`trainer.py` の `_combine_metric`、1737–1743 行。`pesq.py` の `score_pesq`、68–93 行。
- 根拠: 公式は例外を -1 にし、`np.mean` に残す。無音は `NoUtterancesError` なので -1 になる。現行の無音は `na` で、平均から除外される。`score_pesq` は全長と 1 ビン手前の差が 0.1 を超えると、raw が計算できても `na` にする。float32 正弦では raw と `score_pesq` は差 0 で、ランダム 8 本ではこのガードは発火しなかった。長さ 16050 と 16000 は、この pesq では両方とも同じスコアになり、-1 にはならなかった。
- 再現手順: セクション 4.4。無音は raw が例外、`metric_targets` は `None`、`score_pesq` は `na`。
- 影響: 失敗が混ざる検証セットでは、best に使う平均が公式より高くなりうる。失敗が無ければ、float32 の通常スコアは raw PESQ と一致しうる。VoiceBank でガードが何件発火するかは未計測。
- 対応方針: best 用の平均を公式の `pesq_score` と同じく、例外を -1 として全件平均にするかを決める。境界ガードは評価指標としては別に残し、checkpoint 選択には使わない、が公式の選択に近い。
- P10-C影響: NON-BLOCKER
- 検証状態: FAIL

### P10P-A-004

- 発見者: Agent A
- 対象: MP-SENet の中断と再開
- 種別: OFFICIAL_DIFF
- 深刻度: HIGH
- 確度: CONFIRMED
- 場所: 公式 `train.py` の `train`、50–73 行と 104 行、207–209 行。`utils.py` の `scan_checkpoint`、86–91 行。現行 `mp_senet.py` の `load_state_dict`、111–125 行。`trainer.py` の `fit`、190–200 行と `_save`、1146–1165 行。PyTorch `LRScheduler._initial_step` と `ExponentialLR.get_lr`。
- 根拠: 公式は `do_*` の `epoch`（進行中の 0 始まり epoch）で `ExponentialLR` を作り直す。テストでは 3 回 step 済みの lr `0.0004851495` が、作り直しで `0.000480298005` になった。データは `range(last_epoch, ...)` でその epoch の先頭からやり直し、`steps` は保存値 +1。現行は scheduler state を読み、lr は `0.0004851495` のまま。保存 cursor の続きから再開する。`best_pesq` も現行は checkpoint に残る。公式は再開のたびに `best_pesq = 0` なので、epoch 40 以降の次の検証で、以前の best より悪くても `g_best` を上書きしうる。
- 再現手順: セクション 4.2。
- 影響: 1 回でも再開すると、公式は lr が余分に 0.99 倍され、同じ epoch の先頭データを再学習する。現行は lr とデータ位置を保つ。長時間を分割する実行では、公式の軌跡には乗らない。
- 対応方針: 公式のバグまで再現するなら、再開時に ExponentialLR を `last_epoch=保存 epoch` で作り直し、その epoch を先頭から回す。学習を正しく続けるなら現行のままにし、公式再開とは別だと記録する。
- P10-C影響: BLOCKER
- 検証状態: FAIL

### P10P-A-005

- 発見者: Agent A
- 対象: MP-SENet Discriminator の global max の勾配
- 種別: OFFICIAL_DIFF
- 深刻度: LOW
- 確度: CONFIRMED
- 場所: 公式 `models/discriminator.py` の `MetricDiscriminator`、57 行 `AdaptiveMaxPool2d(1)`。現行 `discriminator.py` の `_GlobalMaxPool`、25–33 行と 52 行。
- 根拠: forward はタイなし入力で誤差 0、Discriminator 全体の forward も誤差 0。全要素が同じ 2x2 では勾配が `[1,0,0,0]` と `[0.25]*4` に分かれた。
- 再現手順: セクション 4.3 の pool 比較。
- 影響: 最大値が厳密に同値の位置があると、Discriminator とその先の Generator への勾配が公式と違う。通常のランダム活性ではタイが無く、測った範囲では勾配も一致した。
- 対応方針: 決定的 backward を優先するなら現状を維持する。公式の tie break（先頭だけ）に合わせるなら `AdaptiveMaxPool2d` に戻す。GPU の非決定性と引き換えになる。
- P10-C影響: NON-BLOCKER
- 検証状態: FAIL

### P10P-A-006

- 発見者: Agent A
- 対象: MossFormerGAN Discriminator の初期化乱数
- 種別: BUG
- 深刻度: HIGH
- 確度: CONFIRMED
- 場所: 公式 `train.py` の `main`、29–33 行。`networks.py` の `network_wrapper.__init__`、15–18 行。`generator.py` の `MossFormerGAN_SE_16K.__init__`、21–27 行。現行 `mossformergan.py` の `MossFormerGANTask.__init__`、97–99 行。
- 根拠: `network_wrapper` は呼ぶたびに Generator と Discriminator の両方を作る。公式は 1 回目の `.se_network` を残し、2 回目の `.discriminator` を残す。seed 777 では、現行の 1 回ずつの初期化は公式 1 回目（Generator checksum 差 0、Discriminator checksum `7127.819336`）と一致し、公式が学習に使う 2 回目の Discriminator（`7729.488281`、差 `601.67`）とは一致しない。Generator の forward は、公式が残す重みと誤差 0。
- 再現手順: セクション 4.5 の double init。
- 影響: Generator の初期値は公式が残すものと一致する。Discriminator は学習の初手から別の重みになる。GAN の勾配は最初の更新から公式と分岐する。
- 対応方針: 公式に合わせるなら、同じ seed で wrapper 相当を 2 回構築し、1 回目の Generator と 2 回目の Discriminator を残す。1 回初期化が意図なら、公式の初期 Discriminator ではないと記録する。
- P10-C影響: BLOCKER
- 検証状態: FAIL

### P10P-A-007

- 発見者: Agent A
- 対象: MossFormerGAN の PESQ 失敗と非有限損失
- 種別: BUG
- 深刻度: HIGH
- 確度: CONFIRMED
- 場所: 公式 `loss.py` の `loss_mossformergan_se_16k`、58–75 行。`solver.py` の `_run_one_epoch_mossformergan_se_16k`、277–303 行。現行 `mossformergan.py` の `train_update`、235–240 行と `_finite`、388–390 行。`losses.py` の `metric_targets`、113–114 行。
- 根拠: 公式は NaN または PESQ が `None` のとき `None, None` を返し、`optimizer.step` は `loss_gen is not None` の中だけ。ループは次のバッチへ進む。`self.step` の加算は失敗時も 303 行で行われる。無音の `metric_targets` は `None` だった。現行はそのとき `TrainingError` を送出する。検証側は 0 を分母に残す実装で、公式の検証平均とは揃っている。学習の継続だけが違う。
- 再現手順: 無音で `metric_targets` が `None` になることはセクション 4.4。そのあと `train_update` が例外になることは、236–240 行が `None` で必ず raise するコード。フルの `train_update` は実行していない。
- 影響: 無音や PESQ 例外が 1 バッチでも出ると、公式はその更新を飛ばして学習を続け、現行は run 全体が失敗する。公式コメントは無音で PESQ が落ちることを想定している。
- 対応方針: 学習では公式どおり、そのバッチは backward も step もせず次へ進む。0 を Discriminator の目標にはしない、という現行の方針は公式の `return None` と両立する。
- P10-C影響: BLOCKER
- 検証状態: NOT_TESTED

### P10P-A-008

- 発見者: Agent A
- 対象: MossFormerGAN の 2 GPU、BatchNorm と DDP buffer
- 種別: OFFICIAL_DIFF
- 深刻度: HIGH
- 確度: CONFIRMED
- 場所: 公式 `solver.py` の `__init__`、29–34 行。`DDP(...)` は `broadcast_buffers` を渡さない。現行 `mossformergan.py` の `distribute`、112–117 行、`train_update` の buffer broadcast、220–224 行。`distributed.py` の `wrap_module`、67–80 行。Generator 側の `BatchNorm1d` は `src/se/models/mossformergan/conv_module.py` 182 行。公式 Discriminator の `BatchNorm2d` は `discriminator.py` 125–165 行。
- 根拠: 公式は Generator と Discriminator を `SyncBatchNorm.convert_sync_batchnorm` してから DDP する。`find_unused_parameters=True`。`broadcast_buffers` はデフォルト True。現行は SyncBatchNorm を呼ばず、Discriminator だけ `broadcast_buffers=False` にして、micro-batch の前に rank 0 の buffer を 1 回 broadcast する。1 GPU は `wrap_module` が DDP しない。2 GPU の数値は未実行。
- 再現手順: コード比較のみ。1 プロセスの BN 更新はセクション 4.5。2 GPU は起動していない。
- 影響: `mossformergan_vb28_official.yaml` の 1 プロセスではこの差は出ない。`train.sh` の 2 GPU と `mossformergan_vb28_official_2gpu.yaml` では、batch 統計の GPU 間同期と、forward ごとの buffer broadcast が公式と違う。Discriminator の running stats は rank 0 の検証更新が次 step で配られる形になり、SyncBN の全 GPU 平均ではない。
- 対応方針: 2 GPU 再現では両方を SyncBatchNorm にし、`broadcast_buffers` の公式デフォルトとの差を、backward に必要な箇所だけに限って記録する。1 GPU 実験には適用しない。
- P10-C影響: BLOCKER
- 検証状態: NOT_TESTED

### P10P-A-009

- 発見者: Agent A
- 対象: MossFormerGAN の plateau が履歴最小を保たないこと
- 種別: RISK
- 深刻度: HIGH
- 確度: CONFIRMED
- 場所: 公式 `solver.py` の `train`、214–236 行と 249–250 行。現行 `mossformergan.py` の `mossformer_validation_decision`、63–80 行、`step_schedulers`、135–153 行。
- 根拠: `>=` の枝が `best_val_loss` を今の損失で上書きする。CPU では `0.9` のあと `1.2` が best になり、次の `1.1` が improve になった。同値は改善ではない。5 回目で halve、stall は残る。10 回目で stop。half は Generator の lr だけ。現行関数はこの公式挙動を再現している。バグの新規導入ではない。
- 再現手順: セクション 4.6。
- 影響: 「best」は履歴最小ではない。悪化のあと少し戻ると、より悪かった重みが last best として保存され、5 回目の half でその重みへ戻る。公式も同じなので、公式の選択を再現するならこの挙動が正しい。最小検証損失の重みを残す実験には使えない。
- 対応方針: 公式再現なら変えない。実験が履歴最小を欲するなら、比較と保存を分ける変更が必要で、その時点で公式とは別実験になる。
- P10-C影響: NON-BLOCKER
- 検証状態: PASS

### P10P-A-010

- 発見者: Agent A
- 対象: cuDNN の決定性フラグ
- 種別: OFFICIAL_DIFF
- 深刻度: MEDIUM
- 確度: CONFIRMED
- 場所: 公式 MP-SENet `train.py` 22 行 `cudnn.benchmark = True`。公式 MossFormer `train.py` 18–19 行は `deterministic = True` かつ `benchmark = False`。現行 `rng.py` の `set_determinism`、29–42 行。呼び出しは `run.py` 80 行。再現 YAML は両方 `deterministic: false`。
- 根拠: 現行は `benchmark` を常に False にし、`deterministic: false` では `cudnn.deterministic` を False にする。MP-SENet 公式は benchmark True。MossFormer 公式は deterministic True。CPU テストはどちらのフラグも使わない。
- 再現手順: コード比較。GPU の同一入力比較はしていない。
- 影響: GPU では畳み込みアルゴリズムの選択が公式と違いうる。CPU で確認した誤差 0 は、この差を消していない。
- 対応方針: 公式の GPU 挙動に寄せるなら、モデルごとに公式の 2 行をそのまま適用する。再現性を優先するなら現行のまま、GPU のビット一致は目標外だと書く。
- P10-C影響: NON-BLOCKER
- 検証状態: NOT_TESTED

### P10P-A-011

- 発見者: Agent A
- 対象: 学習サンプルの crop と、48 kHz からの正規化順
- 種別: OFFICIAL_DIFF
- 深刻度: HIGH
- 確度: CONFIRMED
- 場所: 公式 MP-SENet `dataset.py` 76–87 行。公式 MossFormer `dataloader.py` の `audio_norm` と `audioread`、25–42 行、`Wave_Processor.process` 64–76 行。現行 `transform.py` 26–40 行と 43–66 行、`renderer.py` 58–71 行、`planner.py` の `_crop`、236–238 行。ondemand protocol の `resample.order`。
- 根拠: 公式 MP-SENet は全長 RMS のあと `random.randint(0, length-segment)`。現行の crop 開始は `mix_seed(...) % (length-window+1)` で、範囲の個数は同じだが列は別。公式 MossFormer の検証も毎回 `random.randint` で 2 秒を切る。現行 protocol は `per_epoch_random` で epoch ごとに決定的。公式 `audioread` は `audio_norm` のあと、sample rate が違うときだけ `librosa.resample`。ondemand protocol は soxr、clip、PCM16 のあと `protocol_normalize`。式そのものは `audio_norm` で `5.3e-9`、RMS で約 `4.8e-7`。実ファイルの波形差は未計測。
- 再現手順: 式はセクション 4.4。crop 列と実音声は未実行。
- 影響: ミニバッチの中身と、48 kHz ソースの正規化対象が公式 loader と違う。学習ステップの数値を公式実行とビット比較はできない。プロトコルとして別実装にするなら、モデル更新則の差とは分けて記録できる。
- 対応方針: モデル側の再現とデータ列の再現を分けて判定する。データ列を公式乱数に合わせるなら `random` の消費順を写す必要がある。48 kHz 変換を公式不明のままにするなら、現行の soxr 順を実験条件として固定する。
- P10-C影響: NON-BLOCKER
- 検証状態: NOT_TESTED

### P10P-A-012

- 発見者: Agent A
- 対象: MossFormerGAN 検証が学習の dropout 乱数を進めるか
- 種別: OFFICIAL_DIFF
- 深刻度: MEDIUM
- 確度: CONFIRMED
- 場所: 公式 `solver.py` 196–200 行。検証でも `loss_mossformergan_se_16k` が Discriminator を `train()` のまま呼ぶ。現行 `trainer.py` の `_validate_training_data`、1259–1323 行は検証の前後で RNG を保存して戻す。
- 根拠: 公式に RNG の保存は無い。Discriminator は dropout を持つ。現行は検証で消費した乱数を戻す。数値の乱数列比較はしていない。
- 再現手順: コード比較のみ。
- 影響: 最初の検証のあと、公式の学習 dropout は検証で進んだ状態から続く。現行は検証前の状態に戻る。初期化差（P10P-A-006）が無い場合でも、1 epoch 後のマスクは公式とずれる。
- 対応方針: 公式の乱数消費に合わせるなら、MossFormer の検証中は RNG を戻さない。他モデルの決定性を守るなら、この差をモデル固有として残す。
- P10-C影響: NON-BLOCKER
- 検証状態: NOT_TESTED

### P10P-A-013

- 発見者: Agent A
- 対象: 研究用 YAML と、公式の 1000 step checkpoint
- 種別: OFFICIAL_DIFF
- 深刻度: MEDIUM
- 確度: CONFIRMED
- 場所: `configs/training/mp_senet.yaml` の `validation_interval: 2675`。`configs/training/mossformergan.yaml` は `data.rms_normalize` を書かず、`config.py` 433 行のデフォルト True。公式 MossFormer `train.sh` 13 行は `checkpoint_save_freq=1000`。`solver.py` 323–324 行と 248 行。
- 根拠: 公式 MP-SENet の検証間隔は 5000 更新で、scheduler は epoch 終わり。研究 YAML は検証も checkpoint も 2675 で、コメント上の 1 epoch と一致する設定であり、5000 ではない。研究 MossFormer YAML は legacy の noisy RMS に加え、`_generator_terms` がもう一度エネルギー正規化する。公式 loader は `audio_norm` であって noisy RMS ではない。公式は epoch 末尾に加えて 1000 step ごとにも保存する。再現 YAML の間隔は 1 pass 分だけ。
- 再現手順: YAML と該当行の照合。学習は起動していない。
- 影響: 研究用 `configs/training/*.yaml` は公式の検証間隔、前処理、保存間隔ではない。再現用 `configs/training/reproduction/*official*.yaml` はこの 3 点を公式側に寄せている。1000 step ごとの公式 checkpoint は現行の再現 YAML には無い。中断の粒度が違う。
- 対応方針: 本再現は reproduction YAML を正とし、研究 YAML を公式比較に使わない。1000 step 保存が必要なら `checkpoint_interval` を別に持たせる。
- P10-C影響: NON-BLOCKER
- 検証状態: NOT_TESTED

### P10P-A-014

- 発見者: Agent A
- 対象: 定数波形の `audio_norm`
- 種別: DATA_GAP
- 深刻度: LOW
- 確度: CONFIRMED
- 場所: 公式 `dataloader.py` の `audio_norm`、34–42 行。現行 `transform.py` の `independent_audio_norm`、57–61 行。
- 根拠: 定数 32 サンプルでは、公式は空の peak 集合の mean で NaN。現行は例外。通常の乱数波形では差 `5.3e-9`。
- 再現手順: セクション 4.4。
- 影響: 完全に平坦な発話だけ。公式は NaN 損失としてバッチをスキップしうる。現行はデータ読み出しで止まる。
- 対応方針: 平坦な入力を公式どおり NaN にして学習側のスキップに渡すか、例外のまま不正データとして落とすかを決める。通常の VoiceBank では分岐しない。
- P10-C影響: NON-BLOCKER
- 検証状態: PASS

### P10P-A-015

- 発見者: Agent A
- 対象: MossFormerGAN の half が公式では epoch 番号を巻き戻すこと
- 種別: OFFICIAL_DIFF
- 深刻度: MEDIUM
- 確度: CONFIRMED
- 場所: 公式 `solver.py` の `_load_model`、104–105 行と `train`、229–252 行。`train` の for は 187 行で、sampler は 188 行のループ変数 `epoch`。現行 `mossformergan.py` の `_restore_best_and_halve_generator`、165–172 行。trainer の epoch は `_run_data_epochs` 509 行で 1 増えるだけで、half では戻さない。
- 根拠: half の `_load_model` は best checkpoint の `epoch` と `step` を `self.epoch` と `self.step` に書き戻す。そのあと 252 行で `self.epoch + 1`。sampler の `set_epoch` は for の変数なのでデータ順は巻き戻らない。現行は重みと optimizer を戻して Generator lr を半分にするが、trainer の epoch は進み続ける。実行は関数と分岐の読み取りで、half の統合テストはしていない。
- 再現手順: コード比較。plateau の判定自体はセクション 4.6。
- 影響: データ順は公式も巻き戻らない。公式の checkpoint ファイル名に入る epoch は half のあと過去の番号に戻る。現行の `step_*.pt` の epoch は単調。再開位置の解釈が公式 checkpoint と違う。
- 対応方針: 公式のファイル名と `self.epoch` まで合わせる必要が無ければ、現行の単調な epoch を維持する。合わせるなら half のあと checkpoint 内の epoch だけ公式同様に戻し、データ計画の epoch とは別フィールドにする。
- P10-C影響: NON-BLOCKER
- 検証状態: NOT_TESTED

## 7. 反証を試みて問題にならなかった仮説

- 公式 MossFormer の AdamW は YAML の `1e-5` を使う。反証: `train.py` 54–56 行は引数を渡さない。実測デフォルトは 0.01。現行も 0.01 を渡す。
- 検証で Discriminator を `eval()` にして BN を止める。反証: 公式は `model.eval()` だけ。公式 Discriminator の `train()` + `no_grad` で running stats が変わり、`eval()` では変化 0。現行も検証中は Discriminator を `train()` のままにする。
- Plateau は履歴最小を保持し、同値は改善、half は stall を戻し、Discriminator の lr も半分にする。反証: 公式 215–236 行と現行関数の数列。同値は hold、非改善で best を上書き、5 で halve したあと stall は残り、10 で stop。lr を半分にするのは Generator だけ。
- PESQ 失敗を 0 にして optimizer を進める。反証: 学習では公式は `None` を返して step しない。検証では 0 を足して分母に残す。現行の検証も 0 を残す。学習の現行は step ではなく例外（P10P-A-007）。
- MossFormer の時間項は予測の real/imag を対にして戻す。反証: 公式 49 行は clean real と予測 imag。現行も同じ。iSTFT と時間損失の誤差は 0。
- STFT が二重、または `center` の向きが違う。反証: データ側は波形のまま。タスク側の STFT は 1 回。MossFormer の center=True STFT と center=False iSTFT は現行と誤差 0。MP-SENet の STFT / iSTFT も誤差 0。
- Generator 構造が別物。反証: MP-SENet は同一重みで出力誤差 0。MossFormer も公式が残す Generator 重みで forward 誤差 0。
- 現行 Discriminator 初期化は公式が学習する Discriminator と一致する。反証: checksum は捨てられる 1 回目と一致し、残る 2 回目とは差 `601.67`（P10P-A-006）。
- 長さが hop で割り切れないと公式 PESQ は必ず -1。反証: pesq 0.0.4 は 16050 と 16000 で例外にせず、`score_pesq` と同じ `4.643888473510742` を返した。-1 になるのは例外時で、無音は例外になった。
- `score_pesq` は float32 の通常スコアを raw PESQ からずらす。反証: 1 秒正弦で差 0。境界ガードは別経路で、試した入力では発火しなかった。
- ExponentialLR は検証 5000 回ごとに進む。反証: 公式は epoch ループの外で 1 回。現行も `epoch_done` のときだけで、`validation_interval` の分岐では scheduler を呼ばない。
- 公式再開は scheduler state をそのまま戻す。反証: state は保存していない。作り直しで lr が余分に 0.99 倍された。
- 1 GPU の再現 YAML が `train.sh` の 2 GPU global batch 4 と違うこと自体が、YAML の分解バグ。反証: 1 GPU YAML は per-GPU effective 2 と一致する。2 GPU YAML は global 4。`train.sh` の GPU 数に対応するのは 2 GPU 側。

問題が 0 件ではない。上記の一致範囲は、損失式、1 回の forward、fresh start の scheduler 間隔、1 GPU の BN モードに限る。初期 Discriminator、検証の入力長、再開、PESQ 失敗時の継続、2 GPU の SyncBatchNorm は公式の実行と揃っていない。
