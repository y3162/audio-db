> 訂正（2026-11-04）：TF-GridNet の 5,000 更新は固定更新の研究案であり、ESPnet chunk iterator の公式 epoch ではない。35 × 5,000 の時間はその研究案の外挿。詳細は [監査](OFFICIAL_TRAINING_AUDIT.md)。その後、5 モデルの研究用 seed 17 は GO にした。開始に残るのは、30 分を超える各ジョブの承認である。

# 本格学習と ASR 評価の計画（案）

実行承認用の計画であり、長時間ジョブはまだ開始していない。最終受入で PARTIAL の条件は解消してから対応モデルを本試験に入れる。まず TF-GridNet と固定 ASR で評価パイプラインを確認し、残りのモデルを順に追加する。ASR の重みとデコーダは固定する。第一 ASR は Whisper Large v3、Parakeet TDT 0.6B v2 と ASR fine-tuned Wav2Vec 2.0 は第2段階とし、各 checkpoint の取得可否と推論メモリを先に確認する。

## データと比較

VoiceBank+DEMAND の公式 28 話者のうち train 26 話者 10,700 発話、validation p286/p287 872 発話、公式 test 824 発話を用いる。話者・元音声・雑音の分割を case ID とファイル指紋で監査する。評価は transcript 全文と完全な発話を対応させ、固定した DuckDB 名前付きセットからのみ読み、WER は `wer-norm-v1` の micro 集計とする。参照 clean が時間整合した場合だけ PESQ、STOI、SI-SDR を集計し、算出不能は理由付き欠損とする。Noisy、Clean、各モデルの公式 pretrained、独立実装の新規学習 checkpoint を同じケースで比べる。公式コードの同条件 scratch 学習は資源に余裕がある場合の対照とする。

標準 test に加え、分離した雑音から負の SNR を含む固定評価セットを作る。例えば -5、0、5、10、15 dB を事前指定し、レシピと staging を保存し登録する。これは標準 VoiceBank test とは別集計とし、テスト発話や雑音を学習・モデル選択に使わない。各 SNR と全体の WER、発話別の誤り数差・改善率・悪化率を保存する。発話を単位とする paired bootstrap で差の区間を報告する。clean は上限の目安であり、WER が必ず改善する保証はない。

公式条件再現と、同一データ・分割・validation・評価セットを用いた公平比較は別実験として記録する。公式で異なるサンプルレートや入力長は明記し、共通比較でもモデル固有の前処理を固定する。入力 resampling、出力長、遅延補正、ASR 設定、PESQ 帯域を事前に固定する。validation のみで checkpoint を選び、test の繰り返し選択をしない。

## 学習プロトコル

各モデルは scratch、seed 3・17・29、公式幅と公式損失・optimizer・scheduler を起点とする。初回は seed 17 の短い pilot で有限損失、勾配、GPU メモリ、I/O と checkpoint 再開を確認し、その結果で資源要求を改定する。epoch ごとに validation と完全な再開 checkpoint、最新と validation 最良を保存する。予定はまず最大 100 epoch、validation 改善なし 15 epoch で early stopping（最初の 40 epoch は停止しない）；公式再現系列では公式停止条件を優先する。発散、非有限値、欠損率、更新速度低下を監視し異常時は停止して原因を記録する。再開時は optimizer、scheduler、RNG、順序・crop を復元する。現在の smoke YAML をそのまま本学習に用いず、学習データセットと公式幅を指定した新規設定を記録する。

共通比較では validation PESQ を checkpoint 選択の事前指定基準とし、同じ validation の WER も追跡する。ASR を使う WER に切り替えて事後選択しない。現行 Trainer の validation・保存間隔は step 指定で、best は PESQ である。上記の epoch ごとの実行、patience 15 の early stopping、自動的な発散検出は本計画の要件であり、現行コードで実装済みと仮定しない。各 epoch の更新数と端数処理を固定し、必要な機能を実装・再開試験してから本学習を開始する。

100 step 試験で固定した設定は `configs/training/trial100/`。本学習では `max_steps` を epoch 数に合わせ、`checkpoint_dir` を空の新しいディレクトリにする。`validation_interval` と `checkpoint_interval` は、そのモデルの `steps_per_epoch` の約数で、かつ最終 step に当たるようにする。当たらないと、epoch 末の plateau は検証値を見ない。patience 15 の共通 early stopping は、MossFormerGAN の公開ルール以外は未実装である。

| モデル | 試験で動いた設定 | 公開条件との差 |
|---|---|---|
| MP-SENet | global 4、2 秒、16 kHz、AdamW 5e-4、betas 0.8/0.99、WD 0.01、ExponentialLR 0.99、400 epoch が公開既定 | 読み込み workers 4 と次 step の先読み。検証の端数サンプルは hop 倍数へ切る |
| SEMamba++ | global 8、3 秒、16 kHz、hid 48、fused、AdamW 2e-4、100 epoch。実行は micro 4 × 蓄積 2 | 公式 batch 8 の 1 forward はこの MIG で OOM。蓄積 2 は Generator を forward し直す |
| TF-GridNet | `tf_gridnet_chunk_seed17.yaml`。chunk 48,000、batch 4、RMS なし、35 epoch | 固定 5,000 更新は使わない。seed 17 の epoch 1 は 575 更新 |
| DeepFilterNet3 | `deepfilternet3_schedule_seed17.yaml`。120 epoch、`max_steps` 22711 | batch 32 は 3.471 秒、batch 64 は 7.025 秒。検証は発話ごとの非加重平均 |
| MossFormerGAN | global 2、2 秒、16 kHz、micro 1 × 2、AdamW 5e-4、D はその 2 倍、WD 0.01、clip 10、120 epoch | 1 step で G と D を各 1 回。公開プロセスの再起動は plateau 回数を戻す |

## 資源と順序

H100 MIG 4g.47gb、VoiceBank 実音声、定常 step の中央値からの見積もりは [GPU_BUDGET.md](GPU_BUDGET.md)。内訳は [PERFORMANCE.md](PERFORMANCE.md)。旧 A100 の表は [GPU_COST.md](GPU_COST.md) に残し、ここでは使わない。研究用本格学習の 1 seed は、MP-SENet 190.8 時間、SEMamba++ 67.0 時間、MossFormerGAN 143.1 時間、TF-GridNet の公式 chunk は約 1.2–1.3 時間、DeepFilterNet3 の 120 epoch は学習 38.8 時間に古い検証外挿 7.6 時間を足して約 46.4 時間である。合計は約 449 時間。3 seed は約 1,346 時間。TF-GridNet の 9.9 時間は固定 5,000 更新の研究案であり、この合計には入れない。ASR の全件 WER はこの合計に入っていない。

実行順は次のとおり。各本学習ジョブは 30 分を超えるので、投入のたびに明示の承認を得る。

1. TF-GridNet の seed 17 は `configs/training/tf_gridnet_chunk_seed17.yaml`。epoch 1 は 575 更新。固定 5,000 更新の checkpoint からは再開しない。
2. DeepFilterNet3 の seed 17 は `configs/training/deepfilternet3_schedule_seed17.yaml`。batch 32 と 64 は測定済みで、この MIG に収まる。
3. SEMamba++ の seed 17 は `configs/training/semambapp_seed17.yaml`。micro 4 × 蓄積 2。
4. MossFormerGAN、MP-SENet の seed 17 は `configs/training/mossformergan_seed17.yaml` と `configs/training/mp_senet_seed17.yaml`。
5. 同じ checkpoint を固定 ASR の小規模な名前付きセットへ通し、WER の所要時間を測る。全件評価はその時間を見てから承認する。
6. seed 17 の test 評価が揃ってから seed 3 と 29 を出す。

いずれも 30 分を超えるジョブは、投入の前にそのジョブを含む明示の承認が要る。詳細は [OFFICIAL_TRAINING_AUDIT.md](OFFICIAL_TRAINING_AUDIT.md)。

Whisper Large v3 は、同じ test セットの登録順先頭 4 発話で clean / noisy / enhanced を通した。全 824 件は、1 checkpoint あたり約 0.9 時間の比例見積もりであり、まだ承認していない。

5 モデルの研究用 seed 17 設定は固定した。残る開始条件は、30 分を超える各ジョブの明示承認である。別ノードでの環境再構築と、公式プロセスとの Adapter 差は、学習を始める条件にはしない。コミット済み YAML の `allow_busy_gpu` は false なので、選択した GPU の `memory.used` が 1024 MiB 未満である必要がある。

GPU は使用前に空きを確認する。この計画はジョブ投入の承認ではない。
