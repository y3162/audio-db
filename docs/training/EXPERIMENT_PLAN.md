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
| TF-GridNet | global 4、3 秒、16 kHz、ms-snsd、Adam 1e-3、WD 0、clip 5、plateau 0.7/1 | 公開 epoch は 5,000 iter × 35。1 走査は 2,675 step |
| DeepFilterNet3 | global 16、3 秒、48 kHz、公式幅、AdamW 1e-3、amsgrad、clip 1、cosine | batch 24 は同じ MIG で 40 step 完了。epoch 時間は batch 16 とほぼ同じ。batch 32 以上は未測定。`MAX_EPOCHS` は ini を読み直していない |
| MossFormerGAN | global 2、2 秒、16 kHz、micro 1 × 2、AdamW 5e-4、D はその 2 倍、WD 0.01、clip 10、120 epoch | 1 step で G と D を各 1 回。公開プロセスの再起動は plateau 回数を戻す |

## 資源と順序

H100 MIG 4g.47gb、VoiceBank 実音声、定常 step の中央値からの見積もりは [GPU_BUDGET.md](GPU_BUDGET.md)。内訳は [PERFORMANCE.md](PERFORMANCE.md)。旧 A100 の表は [GPU_COST.md](GPU_COST.md) に残し、ここでは使わない。高速化後の標準シナリオの 1 seed は、公開 epoch 数で MP-SENet 191 時間、SEMamba++ 67 時間、TF-GridNet 9.9 時間、MossFormerGAN 143 時間、DeepFilterNet3 は batch 16 の 2 epoch で 0.65 時間である。合計約 411 時間。3 seed は約 1,234 時間。最適化前の同じ範囲は 597 時間と 1,792 時間だった。DeepFilterNet3 の batch 32 以上と ASR の WER はこの合計に入っていない。batch 24 の epoch 2–4 を足しても 1 seed は約 1 時間しか増えない。

実行順は次のとおり。各本学習ジョブは 30 分を超えるので、投入のたびに明示の承認を得る。

1. TF-GridNet の seed 17 を、公式の 5,000 iter × 35 epoch で 1 GPU に出す。検証間隔は 5,000 step。
2. 同じ checkpoint を固定 ASR の小規模な名前付きセットへ通し、WER の所要時間を測る。全件評価はその時間を見てから承認する。
3. MossFormerGAN、MP-SENet、SEMamba++（蓄積 2）の順に seed 17 を出す。DeepFilterNet3 は batch 16 の epoch 0–1 と、測定済みの batch 24 の epoch 2–4 までを先に承認する。batch 32 以上は未測定なので、その epoch の前に step 時間とメモリを測る。
4. seed 17 の test 評価が揃ってから seed 3 と 29 を出す。

本格学習の開始判断は **PARTIAL** のままである。読み込みの並列化と次 step の先読みは採用し、100 step の case ID は最適化前と一致した。TF-GridNet の seed 17 は、30 分を超えるジョブの承認と、epoch 末に当たる validation interval の YAML を書けば開始できる。開始前に残るのはその承認、SEMamba++ を蓄積 2 のまま使うことの確認、DeepFilterNet3 の batch 32 以上である。別ノードでの環境再構築と、公式プロセスとの Adapter 差は、学習を始める条件にはしない。

GPU は使用前に空きを確認する。この計画はジョブ投入の承認ではない。
