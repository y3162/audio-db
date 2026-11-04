# 公式学習設定の監査（途中、2026-11-04）

判定は公式の実効動作との比較。GO は当該行に限定し、学習全体の GO ではない。INTENTIONAL は研究条件の差、DIFF は差異、UNKNOWN は未追跡。公式ディレクトリは読み取り専用。独立側の比較対象は `configs/training/trial100/` と `src/training/`。検証の範囲は [VERIFICATION.md](VERIFICATION.md)。以下は全項目の監査完了を意味しない。

VoiceBank-DEMAND の共通 train/valid/test 分割は研究上の共通条件であり、各公式実験のデータセットを再現した主張ではない。train 10,700・valid 872、test 824、話者分離。入力生成・乱数・公式評価プロトコルの一致は別途必要。

## MP-SENet

| 項目 | 公式設定値 | 公式実効値 | 独立実装値 | 判定 | 根拠 |
|---|---|---|---|---|---|
| 構造・音声 | 64 channel、4 blocks、16 kHz、32000、STFT 400/100 | 設定使用 | 同値 | GO | `src/se/official/mp-senet/config.json`、`configs/training/trial100/mp_senet.yaml` |
| optimizer | AdamW 0.0005、beta .8/.99 | weight decay 既定 .01 | 同値 | GO | `src/se/official/mp-senet/train.py:65`、`src/training/se/optim.py` |
| 更新順・scheduler | — | D 更新後 G、epoch 終了後 exponential .99 | 同じ順 | GO | `src/se/official/mp-senet/train.py:133`、`:164`、`:277`、`src/training/se/mp_senet.py` |
| best / PESQ | start epoch 40 | validation PESQ 最大 | 同じ | GO | `src/se/official/mp-senet/train.py:257`、`:266` |
| validation 長 | — | hop 倍数に揃える | 対応済み | GO | `docs/training/VERIFICATION.md` |
| 損失各項・切出し・再開乱数 | — | 詳細再監査未完 | 詳細再監査未完 | UNKNOWN | `src/se/official/mp-senet/train.py`、`src/training/se/mp_senet.py` |

## SEMamba++

| 項目 | 公式設定値 | 公式実効値 | 独立実装値 | 判定 | 根拠 |
|---|---|---|---|---|---|
| 構造・データ | hid 48、4 TFMamba、16 kHz、48000 | fused scan、batch 8 | 同じ構造、micro 4 × 2 | INTENTIONAL | `src/se/official/semambapp/config.yaml`、`configs/training/trial100/semambapp.yaml`。batch 8 は MIG で OOM。勾配蓄積と一括 forward の等価性は未確定 |
| optimizer / scheduler | AdamW .0002、beta .8/.99、.99 | AdamW 既定 decay .01、epoch 毎 exponential | 同値 | GO | `src/se/official/semambapp/train.py:163`、`:435` |
| GAN、損失、PESQ、乱数、validation | — | 詳細再監査未完 | 詳細再監査未完 | UNKNOWN | `src/se/official/semambapp/train.py:206`、`src/training/se/semambapp.py` |

## TF-GridNet

| 項目 | 公式設定値 | 公式実効値 | 独立実装値 | 判定 | 根拠 |
|---|---|---|---|---|---|
| 構造 / optimizer | 4 層、LSTM 128、Adam .001、clip 5 | 設定を使用 | 同値 | GO | `build/clean-20261104/sources/espnet/egs2/ms_snsd/enh1/conf/tuning/train_enh_tfgridnet.yaml`、`configs/training/tf_gridnet_fixed5000_seed17.yaml` |
| epoch | `num_iters_per_epoch: 5000`、chunk iterator | 5000 **元発話**を選び、長さ依存のチャンクを生成。更新数は可変 | 固定 5000 更新 | DIFF Critical | `espnet2/tasks/abs_task.py:2194`、`espnet2/iterators/chunk_iter_factory.py:34`、`:60`。パスはいずれも `build/clean-20261104/sources/espnet/` 以下 |
| chunk | 48000、batch 4 | 50% overlap、短い発話を除外、cache 内 shuffle | 発話毎に crop/pad | DIFF Critical | `espnet2/iterators/chunk_iter_factory.py:44`、`:184`、`:211`、`src/training/data/dataset.py` |
| validation / scheduler | valid loss、factor .7、patience 1 | epoch 終了後に更新、既定 threshold 1e-4 relative | 固定 5000 更新境界で更新 | INTENTIONAL | `espnet2/train/trainer.py:377`、`src/training/core/trainer.py`、`src/training/se/optim.py` |
| best / stop | valid si_snr 最大と loss 最小、各上位 5、patience 5、最大 35 | 2 指標で保存、早期停止あり | loss 最小 1 件、早期停止なし、175000 更新上限 | DIFF Major | 上記公式 YAML、`espnet2/train/trainer.py:446`、`:525` |
| data | MS-SNSD | MS-SNSD | VoiceBank-DEMAND | INTENTIONAL | 上記公式レシピ、独立設定 |
| 再開 | epoch checkpoint | 公式 reporter、model、optimizer、scheduler 等 | 論理 epoch とデータ周位置・RNG を保存。旧 checkpoint は拒否 | INTENTIONAL | `espnet2/train/trainer.py`、`src/training/core/trainer.py` |

固定 5000 更新はユーザーが指定した研究スケジュールであって公式 epoch ではない。公式と同じ iterator、早期停止、best 5 と SI-SNR の選択は未実装。独立設定は実験案であり本格学習の確定条件ではない。

## DeepFilterNet3

| 項目 | 公式設定値 | 公式実効値 | 独立実装値 | 判定 | 根拠 |
|---|---|---|---|---|---|
| batch schedule | 0/16, 2/24, 5/32, 10/64, 20/128, 40/9999 | epoch 毎に変更 | 固定 16 の trial | DIFF Critical | `src/se/official/deepfilternet/DeepFilterNet/df/train.py:170`、`:233`、`configs/training/trial100/deepfilternet3.yaml` |
| optimizer | AdamW、amsgrad、lr .001 | 実装の config/default に従う | .001、decay 1e-12、amsgrad | UNKNOWN | `src/se/official/deepfilternet/DeepFilterNet/df/train.py:480`、`:515`。配布設定と実際の読込を再追跡要 |
| epoch 数 | 明示設定を要確認 | default 10 | schedule 120（配布重み名由来） | DIFF Major | `src/se/official/deepfilternet/DeepFilterNet/df/train.py:186`、trial YAML |
| dataset / augmentation / loss | 音声合成と拡張 | 詳細再監査未完 | 固定 VoiceBank 対 | UNKNOWN | `src/se/official/deepfilternet/DeepFilterNet/df/train.py`、`src/training/se/deepfilternet.py` |

batch 24 は測定済み、32 以上は未測定、9999 は MIG に載らない。固定 batch のまま 120 epoch を公式 schedule と扱わない。

## MossFormerGAN

| 項目 | 公式設定値 | 公式実効値 | 独立実装値 | 判定 | 根拠 |
|---|---|---|---|---|---|
| AdamW | lr G .0005、weight decay .00001 | G .0005、D .001、weight decay 既定 .01 | 同値 | GO | `src/se/official/clearervoice-studio/train/speech_enhancement/train.py:56`、同ディレクトリ `config/train/MossFormerGAN_SE_16K.yaml` |
| 蓄積 | micro 1、effective 2 | 2 micro 更新 | 同値 | GO | 同 YAML、`solver.py:278`、`configs/training/trial100/mossformergan.yaml` |
| plateau 再開 | — | 独自 plateau、再起動でカウンタ初期化 | カウンタを保存 | INTENTIONAL | `solver.py`、`src/training/se/mossformergan.py`、`docs/training/VERIFICATION.md` |
| 構造・損失・切出し・検証の全項目 | — | 詳細再監査未完 | 詳細再監査未完 | UNKNOWN | 同 `solver.py`、`src/training/se/mossformergan.py` |

## 共通の未完了項目

各モデルの loss の全重み・reduction、データ拡張・分割、validation/checkpoint 頻度、公式 resume と RNG の詳細、worker・CUDA の差を一次ソースまで追跡し終えていない。既存の短時間比較 GO はその試験範囲に限られ、監査完了を意味しない。公式と同じ TF-GridNet epoch を求めるか固定更新の研究条件を採用するか、DF の大 batch を実行する資源をどう確保するかの判断が必要。
