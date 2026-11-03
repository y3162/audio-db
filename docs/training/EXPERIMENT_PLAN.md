# 本格学習と ASR 評価の計画（案）

実行承認用の計画であり、長時間ジョブはまだ開始していない。最終受入で PARTIAL の条件は解消してから対応モデルを本試験に入れる。まず TF-GridNet と固定 ASR で評価パイプラインを確認し、残りのモデルを順に追加する。ASR の重みとデコーダは固定する。第一 ASR は Whisper Large v3、Parakeet TDT 0.6B v2 と ASR fine-tuned Wav2Vec 2.0 は第2段階とし、各 checkpoint の取得可否と推論メモリを先に確認する。

## データと比較

VoiceBank+DEMAND の公式 28 話者のうち train 26 話者 10,700 発話、validation p286/p287 872 発話、公式 test 824 発話を用いる。話者・元音声・雑音の分割を case ID とファイル指紋で監査する。評価は transcript 全文と完全な発話を対応させ、固定した DuckDB 名前付きセットからのみ読み、WER は `wer-norm-v1` の micro 集計とする。参照 clean が時間整合した場合だけ PESQ、STOI、SI-SDR を集計し、算出不能は理由付き欠損とする。Noisy、Clean、各モデルの公式 pretrained、独立実装の新規学習 checkpoint を同じケースで比べる。公式コードの同条件 scratch 学習は資源に余裕がある場合の対照とする。

標準 test に加え、分離した雑音から負の SNR を含む固定評価セットを作る。例えば -5、0、5、10、15 dB を事前指定し、レシピと staging を保存し登録する。これは標準 VoiceBank test とは別集計とし、テスト発話や雑音を学習・モデル選択に使わない。各 SNR と全体の WER、発話別の誤り数差・改善率・悪化率を保存する。発話を単位とする paired bootstrap で差の区間を報告する。clean は上限の目安であり、WER が必ず改善する保証はない。

公式条件再現と、同一データ・分割・validation・評価セットを用いた公平比較は別実験として記録する。公式で異なるサンプルレートや入力長は明記し、共通比較でもモデル固有の前処理を固定する。入力 resampling、出力長、遅延補正、ASR 設定、PESQ 帯域を事前に固定する。validation のみで checkpoint を選び、test の繰り返し選択をしない。

## 学習プロトコル

各モデルは scratch、seed 3・17・29、公式幅と公式損失・optimizer・scheduler を起点とする。初回は seed 17 の短い pilot で有限損失、勾配、GPU メモリ、I/O と checkpoint 再開を確認し、その結果で資源要求を改定する。epoch ごとに validation と完全な再開 checkpoint、最新と validation 最良を保存する。予定はまず最大 100 epoch、validation 改善なし 15 epoch で early stopping（最初の 40 epoch は停止しない）；公式再現系列では公式停止条件を優先する。発散、非有限値、欠損率、更新速度低下を監視し異常時は停止して原因を記録する。再開時は optimizer、scheduler、RNG、順序・crop を復元する。現在の smoke YAML をそのまま本学習に用いず、学習データセットと公式幅を指定した新規設定を記録する。

共通比較では validation PESQ を checkpoint 選択の事前指定基準とし、同じ validation の WER も追跡する。ASR を使う WER に切り替えて事後選択しない。現行 Trainer の validation・保存間隔は step 指定で、best は PESQ である。上記の epoch ごとの実行、patience 15 の early stopping、自動的な発散検出は本計画の要件であり、現行コードで実装済みと仮定しない。各 epoch の更新数と端数処理を固定し、必要な機能を実装・再開試験してから本学習を開始する。

| モデル | 初期計画（global batch / segment） | 更新・scheduler |
|---|---|---|
| MP-SENet | 4 / 2秒・16 kHz | AdamW 5e-4、betas 0.8/0.99、WD 0.01、epoch ExponentialLR 0.99 |
| SEMamba++ | 8 / 3秒・16 kHz、必要なら micro 4×2 | AdamW 2e-4、betas 0.8/0.99、epoch ExponentialLR 0.99、fused。蓄積経路の最終受入が必要 |
| TF-GridNet | 4 / 3秒・16 kHz、ms-snsd | Adam、clip 5、validation に基づく ReduceLROnPlateau。公式設定を起動前に固定 |
| DeepFilterNet3 | 初期 16 / 3秒・48 kHz | AdamW、出荷 cosine と batch schedule。可変 batch の実装・steps_per_epoch を事前確認 |
| MossFormerGAN | 2 / 2秒・16 kHz、micro 1×2 | AdamW、G 5e-4・D 1e-3、公式の更新順。scheduler は公式設定を確認 |

## 資源と順序

下表は [GPU_COST.md](GPU_COST.md) の旧 commit・A100 PCIe 40 GB、合成 noisy 入力、3〜4 定常 step の中央値からの外挿であり、H100 や実データの保証値ではない。epoch は train 10,700 件、I/O・validation・保存を除く。時間は 100 epoch・1 seed の概算で、early stopping と DF の batch schedule は含まない。チェックポイントは当時の実測値。評価時間は全件・ASR とも未実測なので pilot で計測する。

| モデル | 秒/step | 時間/epoch | VRAM GiB | checkpoint MB | 100 epoch GPU 時間 |
|---|---:|---:|---:|---:|---:|
| MP-SENet | 1.065 | 0.791 | 31.33 | 29.7 | 79.1 h |
| SEMamba++ | 3.294 | 1.224 | 26.64 | 90.3 | 122.4 h |
| TF-GridNet | 0.216 | 0.161 | 4.98 | 31.1 | 16.1 h |
| DeepFilterNet3 | 1.659 | 0.308 | 3.03 | 34.4 | 30.8 h（batch 16 固定仮定） |
| MossFormerGAN | 1.483 | 2.204 | 16.96 | 137 | 220.4 h |

1. 受入残件を解消し、名前付き評価セット・固定 ASR の小規模 end-to-end を監査する。
2. TF-GridNet の pilot と seed 17 本学習、validation 選択、固定 test 評価を実施する。
3. 残りのモデルを同じ test に追加し、追加 seed と追加 ASR を実施する。3 seed の単純合計は上表の約3倍であり、資源・I/O・validation を再測定してから個別にジョブを承認する。

GPU は使用前に空きを確認し、TSUBAME の30分程度以上の各ジョブは明示的な指示または追加承認を得る。本計画はジョブ投入の承認ではない。
