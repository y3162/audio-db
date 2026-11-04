# 公式学習設定の監査（2026-11-04）

判定は公式コードの実効動作との比較である。GO はその行に限り、学習全体の合格ではない。

- **GO**: 実効値が一致する
- **INTENTIONAL**: 理由を説明できる意図的な差
- **DIFF**: 不一致。Critical は学習結果に直接効く。Major は比較条件に効き得る。Minor は影響が限られる
- **UNKNOWN**: この監査では一次ソースまで閉じていない
- **N/A**: 対象外

公式ディレクトリは読み取り専用。共通の VoiceBank-DEMAND（train 10,700 / valid 872 / test 824、話者分離）は研究上の共通条件であり、各論文のデータセット再現ではない。実装の数値比較の範囲は [VERIFICATION.md](VERIFICATION.md)。本格学習の設定は `configs/training/tf_gridnet_chunk_seed17.yaml`、`configs/training/deepfilternet3_schedule_seed17.yaml`、`configs/training/semambapp_seed17.yaml`、`configs/training/mp_senet_seed17.yaml`、`configs/training/mossformergan_seed17.yaml` である。`trial100/` は 100 step 試験であり本格学習の条件ではない。

固定 ESPnet commit は `0d136c5456d403cb2c0a4c13e76e8136e32e9b7a`（`build/clean-20261104/sources/espnet/`）。

## 総括

| モデル | 更新式の既存 GO | 学習プロトコル | 本格学習 |
|---|---|---|---|
| MP-SENet | 検証した step の範囲で GO | 損失・AdamW・ExponentialLR・PESQ best は一致。切り出し順は case id 決定的で、公式 DataLoader の消費順とは別 | 設定は固定済み。開始は別承認 |
| SEMamba++ | 同じ範囲で GO | 公式の 3 秒 batch 8 は PARTIAL。研究設定は micro 4 × 蓄積 2 で、fused の batch 形状差を測定済み | GO。`semambapp_seed17.yaml`。この MIG に収まり、100 step と再開は既存 |
| TF-GridNet | 同じ範囲で GO | RMS なしの chunk iterator。VoiceBank は MS-SNSD と別。epoch 1–5 の更新数を実長で数えた | GO。`tf_gridnet_chunk_seed17.yaml`。4 step と別プロセス再開まで |
| DeepFilterNet3 | 同じ範囲で GO | schedule と cosine は一致。検証集約とオンライン混合は公式と別 | GO。batch 32 と 64 は測定済み。`deepfilternet3_schedule_seed17.yaml` の `max_steps` は 22711 |
| MossFormerGAN | 同じ範囲で GO | AdamW の実効 weight decay 0.01、独自 plateau は checkpoint に残す | 設定は固定済み。開始は別承認 |

## MP-SENet

| 項目 | 公式設定値 | 公式実効値 | 独立実装値 | 判定 | 根拠 |
|---|---|---|---|---|---|
| 構造・音声 | 64 channel、4 blocks、16 kHz、segment 32000、STFT 400/100 | 設定を使用 | 同値 | GO | `src/se/official/mp-senet/config.json`、`configs/training/mp_senet.yaml` |
| optimizer | AdamW 0.0005、beta 0.8/0.99 | weight decay はコンストラクタ既定 0.01。eps は PyTorch 既定 1e-8 | 同値 | GO | `src/se/official/mp-senet/train.py:65`、`src/training/se/optim.py` |
| 損失 | mag 0.9、phase 0.3、complex 0.1、consistency 0.1、metric 0.05、time 0.2。complex と consistency は 2 倍してから 0.1 | その式 | 同値 | GO | `train.py:167-181`、`src/training/se/losses.py` |
| 更新順 | D のあと G | 同じ。PESQ が None のとき D の fake 項は 0 | 同じ | GO | `train.py:130-183`、`src/training/se/mp_senet.py` |
| scheduler | `lr_decay` 0.99 | epoch 終了後に ExponentialLR | 同じ。蓄積回数では進まない | GO | `train.py:72`、`:277` |
| epoch | `training_epochs` 既定 400 | データ 1 周 | データ 1 周。batch 4 で 2,675 更新 | GO | `train.py:296` |
| best | `best_checkpoint_start_epoch` 既定 40 | validation PESQ 最大 | 同じ | GO | `train.py:266-269` |
| validation 長 | — | hop 倍数に揃える | 対応済み | GO | [VERIFICATION.md](VERIFICATION.md) |
| データ | VoiceBank-DEMAND の公式リスト | ファイルリストを読む | 名前付きセットのレシピ。事前混合で、オンライン混合はしない | INTENTIONAL | 共通分割。論文プロトコルそのものではない |
| 切り出し順 | DataLoader shuffle | 消費順の RNG | case id・seed・epoch の関数 | INTENTIONAL | `.agents/DECISIONS.md` の学習乱数。公式バッチ会員とは DIFF Major になり得る |
| 再開 | last epoch と重み | プロセス再起動は DataLoader の位置を公式 checkpoint の範囲で戻す | model、optimizer、scheduler、RNG、データ位置 | GO（独立実装の再開） | `src/training/core/trainer.py` |

## SEMamba++

| 項目 | 公式設定値 | 公式実効値 | 独立実装値 | 判定 | 根拠 |
|---|---|---|---|---|---|
| 構造・データ | hid 48、4 TFMamba、16 kHz、segment 48000、batch 8 | fused scan、1 forward の batch 8 | 同じ構造。研究設定は micro 4 × 蓄積 2 | DIFF（公式 1 forward） / 研究設定は明示 | 3 秒は OOM。長さ 8,000 の fused 比較は [VERIFICATION.md](VERIFICATION.md)。`semambapp_seed17.yaml` |
| 損失 | adversarial 1、feature matching 1、mel 0.1、mag 0.9、phase 0.3、complex 0.1、consistency 0.1。waveform L1 はコメントアウト | その和 | 同値 | GO | `src/training/se/semambapp.py` 冒頭、公式 `train.py` の損失組み立て |
| optimizer / scheduler | AdamW 0.0002、beta 0.8/0.99、decay 0.99 | AdamW 既定 weight decay 0.01。epoch ごとに ExponentialLR | 同値 | GO | `config.yaml`、`train.py:163`、`:435` |
| 更新順 | D のあと G | サブ discriminator は real と fake を交互 | 同じ順 | GO | `.agents/DECISIONS.md` |
| データ拡張 | 公式ローダの混合 | オンライン混合があり得る | 固定レシピ | INTENTIONAL | 共通 VoiceBank |
| 再開 | epoch checkpoint | — | 独立 checkpoint。optimizer は名前で対応 | GO（独立実装） | パラメータ順は公式と違う |

## TF-GridNet

| 項目 | 公式設定値 | 公式実効値 | 独立実装値 | 判定 | 根拠 |
|---|---|---|---|---|---|
| 構造 / optimizer | 4 層、LSTM 128、Adam 0.001、eps 1e-8、weight decay 0、clip 5 | YAML を使用。betas は Adam 既定 (0.9, 0.999) | 同値 | GO | `egs2/ms_snsd/enh1/conf/tuning/train_enh_tfgridnet.yaml`、`src/training/se/optim.py` |
| epoch | `max_epoch` 35、`num_iters_per_epoch` 5000 | 5000 は元発話数。チャンク数と更新数は可変 | `chunk_iterator.num_samples_per_epoch: 5000`、`max_epochs: 35` | GO | `espnet2/tasks/abs_task.py:2194`、`src/training/data/chunk_plan.py` |
| chunk | 48000、shift 0.5、cache 1024、短い発話は捨てる、batch 4 | 単一カテゴリ・単一サンプルレートでその通り | 同じ計画。CPU で公式 iterator と ID・位置・波形スライス・バッチ順が一致 | GO | `espnet2/iterators/chunk_iter_factory.py`、`tests/training/test_chunk_official.py` |
| 固定 5000 更新 | — | 公式 epoch ではない | `logical_steps_per_epoch` と `tf_gridnet_fixed5000_seed17.yaml` | INTENTIONAL | 研究用。公式学習には使わない |
| 損失 | `mr_l1_tfd` 重み 1、`si_snr` 重み 0 | 学習損失は mr_l1_tfd のみ | 同値 | GO | 公式 YAML `criterions`、`src/training/se/tf_gridnet.py` |
| validation | chunk iterator、shuffle なし、`num_iters_per_epoch` なしで全発話 | 検証もチャンク | 同じ計画 | GO | `abs_task.py:1737`、`trainer.py` の `_planned_chunks` |
| scheduler | ReduceLROnPlateau、factor 0.7、patience 1、mode min | threshold 1e-4、relative（PyTorch 既定） | 同じ値を明示 | GO | 公式 YAML `scheduler_conf`、`src/training/se/optim.py` |
| best | valid `si_snr` max と valid `loss` min、各上位 5 | `si_snr` の記録値は SI-SDR 損失（負の SI-SDR）。max は SI-SDR が低い方を選ぶ | 同じ規則。`valid.loss.best.pt` と `valid.si_snr.best.pt` | GO | `espnet2/enh/loss/criterions/time_domain.py:290`、`FixedOrderSolver` |
| early stop | patience 5、valid loss min | `epoch - best_epoch > patience` | 同じ | GO | `espnet2/train/reporter.py:404` |
| data | MS-SNSD | MS-SNSD | VoiceBank-DEMAND | INTENTIONAL | 共通分割 |
| 正規化 | `speech_volume_normalize` 既定 None | 音量正規化なし | `tf_gridnet_chunk_seed17.yaml` は `rms_normalize: false`。固定 5,000 更新と他モデルは発話 RMS | GO（公式モード） | `enh` の既定、`tests/training/test_dataset.py` |
| 再開 | epoch 末の `checkpoint.pth`（model、optimizer、scheduler、reporter） | 公式は epoch 途中の iterator を保証しない | epoch 途中も、計画を seed から再生成してバッチ位置を飛ばす。model、optimizer、scheduler、RNG、論理 epoch を戻す | INTENTIONAL | 途中再開は独立実装の追加保証 |
| 複数 GPU | サンプルを rank で割ってからチャンク | バッチの中身が変わる | 1 process のみ。複数 process はエラー | INTENTIONAL | 公式レシピは batch 4 の単一プロセスを想定 |

`best_checkpoint_start_epoch: 40` は MP-SENet の規則であり、この 35 epoch には使わない。公式学習 YAML は 0。

## DeepFilterNet3

出荷 ini は `data/results/training/final-h100-20261102-cpu/df-config.ini`（batch 64、max_epochs 120、patience 25）。Python の `MAX_EPOCHS` 既定 10 は、そのキーが無いときだけである。

| 項目 | 公式設定値 | 公式実効値 | 独立実装値 | 判定 | 根拠 |
|---|---|---|---|---|---|
| batch schedule | `0/16,2/24,5/32,10/64,20/128,40/9999` | `min(schedule, batch_size 64)`。epoch 0–1 は 16、2–4 は 24、5–9 は 32、10 以降は 64 | 同じ | GO | `df/train.py:233-242`、`src/training/core/batch_schedule.py` |
| 蓄積による代替 | なし | その batch の 1 forward | batch が `micro_batch_size_per_gpu` を超えるとエラー。蓄積に置き換えない | GO | `trainer.py` の `_apply_batch_schedule` |
| optimizer | AdamW、lr 0.001、betas 0.9/0.999、decay 1e-12、amsgrad | `load_opt` が amsgrad=True | 同値 | GO | `df/train.py:480-498`、ini `[optim]` |
| cosine | warmup 3 epoch、lr 1e-4 から 1e-3、下限 1e-6、decay は 1e-12 から 0.01 | loader 長が変わると表を作り直し、`epoch * len(loader) + step` で引く | 同じ | GO | `df/train.py:244-246`、`:352-360`、`DeepFilterNetTask.retarget_deepfilter_schedule` |
| epoch | ini `max_epochs` 120 | 120 | `epochs: 120`。`max_steps: 22711` は 10,700 発話の optimizer 更新の合計 | GO | `2*668+3*445+5*334+110*167`。最終 cosine 添え字は `120*167-1` |
| early stop | ini patience 25、valid loss min | 改善でカウンタを 0 に戻し、25 回改善しなければ停止 | 同じ | GO | `df/checkpoint.py:156`、`trainer.py` の `_batch_schedule_should_stop` |
| best | valid loss min | 改善時だけ置き換える | validation loss 最小の `best_generator.pt`。開始 epoch の待ちは無し | GO | `df/train.py:285` |
| 損失 | multi-res factor 500、complex 500、gamma 0.3、FFT 256/512/1024/2048、local SNR 1e-3、mask と SI-SDR は 0 | その重み | 同値 | GO | ini `[multiresspecloss]`、`[localsnrloss]`、`src/training/se/deepfilternet.py` |
| データ | オンライン混合、SNR と gain のリスト、reverb 確率 | 毎 epoch 合成 | 固定 VoiceBank 対。3 秒 crop | INTENTIONAL | 共通分割。公式の合成分布ではない |
| 検証 batch | `batch_size_eval` 64 | 学習 batch とは別に 64。batch 内は要素平均で、パディングを含む | 1 発話ずつ計算し、発話間は非加重平均。eval の BatchNorm は running stats | DIFF Major。研究条件として固定 | `tests/training/test_df_validation_reduction.py`。公式の batch 64 検証とは呼ばない |
| GPU | batch 64 まで | — | batch 32 の定常中央値 3.471 秒、reserved 6.49 GiB。batch 64 は 7.025 秒、reserved 12.88 GiB | GO | `data/results/training/go-20261104/df-batch32-rerun/`、`df-batch64/` |
| 再開 | model と optimizer の epoch checkpoint。patience ファイル | schedule の位置は epoch から再現 | 同じ。cosine の添え字は epoch と epoch 内 step から再計算 | GO | `trainer.py` |

## MossFormerGAN

| 項目 | 公式設定値 | 公式実効値 | 独立実装値 | 判定 | 根拠 |
|---|---|---|---|---|---|
| AdamW | lr G 0.0005、YAML の weight decay 1e-5 | G 0.0005、D 0.001、weight decay は渡さないので 0.01。betas (0.9, 0.999)、eps 1e-8 | 同実効値 | GO | `train.py:56`、`MossFormerGAN_SE_16K.yaml`、`.agents/DECISIONS.md` |
| 損失 | 0.1 RI + 0.9 mag + 0.2 time + 0.05 GAN | その式 | 同値 | GO | `losses/loss.py:57`、`src/training/se/mossformergan.py` |
| 蓄積 | micro 1、effective 2 | 2 micro 更新 | 同値 | GO | solver と `configs/training/mossformergan.yaml` |
| plateau | 5 回で学習率を半分、10 回で停止 | プロセス再起動でカウンタ初期化 | カウンタと最良 snapshot を checkpoint に残す | INTENTIONAL | `solver.py`、`src/training/se/mossformergan.py` |
| データ | 公式ローダのランダム切り出し | 消費順の乱数 | case id 決定的 | INTENTIONAL | 上の MP-SENet と同じ方針 |
| 再開 | solver checkpoint | plateau は初期化され得る | 独立 checkpoint は plateau を保持 | INTENTIONAL | [VERIFICATION.md](VERIFICATION.md) |

## まだ閉じていないもの

- TF-GridNet の公式コーパスは MS-SNSD であり、この研究は VoiceBank-DEMAND を使う。`enh_train.py` の 35 epoch は起動していない。固定 5,000 更新の 9.9 時間は公式 chunk の見積もりに使わない。epoch 1–5 の更新数は計測済み。
- SEMamba++ の公式 3 秒・batch 8 の 1 forward は、この MIG では OOM のままである。
- DeepFilterNet3 の検証 227 秒は、8 件からの古い外挿のままである。全件の再測定はしていない。batch 32 と 64 の step 時間は計測済み。
- 公式 DataLoader の消費順と、case id 決定的な順序のバッチ差は、MP-SENet、SEMamba++、MossFormerGAN、DeepFilterNet3 の固定レシピ経路に残る。TF-GridNet の公式モードだけ iterator 順を合わせた。
- オンラインのノイズ混合・RIR・SNR 分布は、VoiceBank の固定レシピでは再現しない。
