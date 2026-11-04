# 学習基盤の確認

2026-11-04 のプロトコル実装は、既存の「同じ状態からの 1 更新」の GO をプロトコル全体の GO に広げない。TF-GridNet の公式 epoch は可変チャンク iterator、DeepFilterNet3 の公式 batch は schedule である。固定 5,000 更新と固定 batch 16 の試験結果は、その条件の結果として残る。判定の範囲は [OFFICIAL_TRAINING_AUDIT.md](OFFICIAL_TRAINING_AUDIT.md)。



## 現在の未解決事項と検証範囲

- 公式学習プロセスとの対応は PARTIAL。比較 Adapter と公式入口の差があり、同一状態の短時間更新の GO は学習全体や最終品質の一致を保証しない。[最終受入](#final-acceptance-の再判定2026-11-04) と [公式設定監査](OFFICIAL_TRAINING_AUDIT.md) を参照。
- SEMamba++ の公式 3 秒・batch 8 は H100 MIG 4g.47gb で OOM。研究設定の micro 4 × 蓄積 2 は batch 形状による差があり、公式の 1 forward と同一ではない。[公式設定監査](OFFICIAL_TRAINING_AUDIT.md)、[再開検証](RESIDUAL_20261104.md)。
- TSUBAME の qsub 連鎖は未検証。[ジョブ継続の検証](JOB_CONTINUATION_AUDIT.md)。
- PESQ の長さ依存の内部原因と全文 raw score の妥当性は未確定。[PESQ 調査](../evaluation/PESQ_LENGTH_INVESTIGATION.md)。
- 本格学習後の性能・全 test 評価は未確認。短時間スモーク、再開確認、公式との一致を区別する。[研究用本格学習の最終確認](#研究用本格学習の最終確認2026-11-04)。

過去の判定は各試験時点のもの。後続の修正や試験があるため、日付・条件と一緒に読む。整理前の全文は `git show 825b2b91df0d9619d7a21b10876e2b2cfdac1a7c:docs/training/VERIFICATION.md` で確認できる。

## 再判定の受入基準（2026-11-03、新規測定の前）

この節は、直前の最終受入監査よりあと、この節の測定より前に固定した。HEAD は `a848049c0a190f8e8ffb4de3c7289740945a4175`、`dev` は `origin/dev` と一致、作業ツリーはクリーンだった。既存の結果ディレクトリは上書きしない。GPU 会計の開始値はプロセス時間 11,265.840 秒、上限 18,000 秒、残り 6,734.160 秒である。

三種類を分ける。

| 種類 | 問うこと | この節で GO と言う範囲 |
|---|---|---|
| A. 実装 | 同じ重み・入力・Optimizer・Scheduler・RNG から、公式と独立が同じ更新を計算するか | 同一状態の forward、損失、backward、更新 |
| B. 実行 | 同じ実装を同じ条件で再実行したとき、どこまで揺れるか | CUDA の自己再実行。実装の GO には使わない |
| C. 環境 | 別の Python 環境や別マシンで構築し直せるか | 未実施の範囲は総合を PARTIAL に留める。A の一致を取り消さない |

判定は次のとおり。測定後に数値を緩めない。特定のモデルを通すための例外は置かない。

1. 決定的な CPU 経路では、forward、損失、勾配、更新後重みの最大絶対差が 0 であることを要求する。CPU で 0 なのに GPU の自己再実行が 0 でないときは、その差を実装差と呼ばない。
2. CUDA の同一状態では、公式 A/B、独立 A/B、公式 A/独立 A、公式 B/独立 B を取る。指標は最大絶対差、相対 L2、cosine、勾配 norm である。系統差は、平均勾配の相対 L2（`bias_rel_l2`）が実装内ペア相対 L2 の中央値の大きい方（`self_median`）を超えること、または forward と各損失成分の平均差が実装内の最大絶対差を超えることである。`bias_rel_l2 <= self_median` を満たさない状態は実装差とする。
3. `cross_median <= self_median` は記録する。繰り返しが少ない中央値の大小だけで GO にも NO-GO にもしない。このブールが偽でも、bias が通り、演算単位の自己再実行が実装間差を包むなら、その状態の backward はカーネル再実行で説明できたとする。包まないなら PARTIAL のまま追う。
4. サンプル数が 8 未満の分布比較は、分布が一致したとは書かない。中央値が重なること、または一部の試行だけ自己再実行以下であることは、同等性の証明にしない。
5. 複数更新の軌跡は同一状態と分ける。同一状態の 1 更新が自己再実行で説明でき、そこから続けた短い分岐で実装間の広がりが実装内の広がりと同じ順序であるとき、長い軌跡の分岐をその蓄積として説明する。長い軌跡の cosine が低いこと自体は、公式どうしでも起きるなら実装差ではない。
6. checkpoint 再開は、更新前のパラメータ、Optimizer の一次・二次モーメント、Scheduler、Python / NumPy / PyTorch / CUDA RNG、step、epoch、入力が連続側と一致することを要求する。更新後のビット一致は、その実装の CUDA 自己再実行がビット一致するときだけ要求する。Optimizer の対応ずれと RNG の復元漏れは非決定性に含めない。SEMamba++ の公式から独立への Optimizer 移植は名前対応であり、同一実装の再開ではない。
7. 環境の再構築が未了なら、A が GO でも総合は PARTIAL とする。

前回 PARTIAL の分類は、新規測定の前は次のとおりである。TF-GridNet の GPU 4 更新と再開の値は一致しており、残るのは全モデル共通の環境再構築と公式データローダ全経路である。MP-SENet の CPU は STFT 修正後にビット一致し、GPU 差は未整理の実行差である。DeepFilterNet3 の GPU 差は自己再実行と同程度だが、逆変換が両側で独立の `torch.istft` である。MossFormerGAN は共通状態の 1 更新が小さい一方、4 更新の一部で実装間が自己再実行を上回る。SEMamba++ は合成入力の同一重みで bias が自己再実行以下だが、実音声の Optimizer 込みの同一状態と、20 更新の軌跡は未説明である。再開試験は更新後ハッシュの分岐までで、更新前一致の記録が比較 JSON の事後ハッシュに限られる。

## 再判定の測定（2026-11-03）

コードの学習経路は変えていない。追加したのは比較スクリプトだけである。GPU は新規 `qrsh` を出さず、既存の対話ジョブ `115437`（ホスト `r4n11`、H100、開始時の表示メモリ 758 / 95830 MiB）を再利用した。プロセス時間は `/usr/bin/time` の wall clock で、失敗も加算する。

| 実行 | 結果 | 秒 |
|---|---|---:|
| SEMamba++ 同一状態、デバイス不一致で中断 | 失敗 | 23.07 |
| SEMamba++ 実音声の同一状態と 4 更新分岐 | 成功 | 86.45 |
| MP / DF 同一状態、DF 設定パス不足 | 失敗 | 9.88 |
| MP / DF 同一状態、DF に学習率属性が無い | 失敗 | 6.87 |
| MP / DF 同一状態と native STFT | 成功 | 10.39 |

合計 136.66 秒。開始時 11,265.840 秒に加え 11,402.500 秒。上限 18,000 秒まで 6,597.500 秒。CPU の逆変換比較、再開記録の照合、回帰テストは GPU 会計に入れない。

### SEMamba++

実音声 20 更新 `final-accept-semamba-seed17-retry-20261103` の公式 A について、保存済み勾配で AdamW を再生すると、20 更新すべてのパラメータ最大差は 0 だった。Scheduler の `last_epoch` は 0 のままである。同一状態の繰り返しは各 4 回で、重み、名前対応した Optimizer、Scheduler、同一 RNG を毎回戻した。入力は更新 0 が発話 0、更新 4 が発話 4、更新 20 は次の発話が無いので発話 19 を再利用した。

| 更新後 | bias 相対 L2 | self median | cross median | 比 | forward / 損失 | scan 入力 | scan backward 最大差（公式再実行 / 実装間） |
|---:|---:|---:|---:|---:|---|---|---|
| 0 | 8.692e-7 | 1.701e-6 | 1.705e-6 | 1.0028 | 0 | 0 | 4.657e-9 / 3.492e-9 |
| 4 | 1.737e-7 | 3.342e-7 | 3.375e-7 | 1.0101 | 0 | 0 | 2.794e-9 / 2.678e-9 |
| 20 | 4.019e-7 | 7.834e-7 | 7.838e-7 | 1.0006 | 0 | 0 | 4.657e-9 / 6.054e-9 |

3 状態とも `bias_rel_l2 <= self_median` で、forward と損失成分の差は 0 である。`cross_median <= self_median` は 3 状態とも不成立だが、超過は 0.06–1.0% で、このブールは判定に使わない。selective scan の forward 入力は一致する。backward 出力の最大差は更新 20 の 1 組だけで実装間が公式再実行を 1.4e-9 上回った。これは 1 組の比較であり、パラメータ勾配の系統差にはなっていない。

更新 4 の状態から公式 2 本、独立 2 本を 4 更新進めた。勾配の bias 相対 L2 は self median を下回った（1 更新 2.46e-7 / 3.40e-7、4 更新 0.0252 / 0.0454）。3 更新目の一部のスカラー損失は、各実装 2 本しかない自己差の最大をわずかに超えた。4 更新目ではその超過は消え、forward の bias も自己最大以下だった。勾配ノイズは 1 更新の約 1e-7 から 4 更新で約 0.03 まで、公式内と実装間で同じ順序に増える。

したがって、既存 3 試行の step 20 で勾配 cosine 中央値が公式内 0.137、独立内 0.282、実装間 0.235 まで落ちるのは、この 1 更新の scan backward ノイズが GAN 更新で積もった結果である。3 試行は独立標本ではなく、20 更新の分布が一致したとは書かない。結果は `data/results/training/final-accept-semamba-real-state-20261103/same-state.json`。

### checkpoint 再開

連続側の step 2 記録と、同じ fork から再開した step 2 記録を、MP-SENet（STFT 修正後 seed 17）、DeepFilterNet3、SEMamba++（stride 修正後）の 4 系統で照合した。更新前の model、optimizer、scheduler、RNG 成分、入力はすべて一致し、公式 A の fork ファイル自体もその記録と一致した。再開後の step 3 では入力、RNG、scheduler は一致し、model と optimizer のハッシュは分かれた。CUDA の自己再実行がビット一致しないので、更新後のビット一致は要求しない。Optimizer の取り違えと RNG の欠落はこの照合には無い。結果は `data/results/training/final-accept-resume-preupdate-20261103/preupdate.json`。MossFormerGAN の GPU 同一実装再開は今回走らせていない。CPU の蓄積 2、4 更新再開は既存記録でビット一致している。

### MP-SENet

修正後の seed 17、公式幅、VoiceBank 1 秒の初期状態で、公式と独立を各 4 回戻して 1 更新した。forward の最大差は 0、Generator 損失は 4 系統とも 3.235639810562134 である。勾配 bias 相対 L2 は 5.001e-8、self median は 9.928e-8 で、系統差の条件は不成立である。同じ波形の native `torch.stft`（`center=True`, `pad_mode="reflect"`）だけの backward を 4 回繰り返すと、波形勾配の相対 L2 中央値は 3.813e-8 で、モデル勾配の自己再実行と同じ桁である。CPU の 4 更新がビット一致しているので、GPU の step 4 で相対 L2 が 0.05–0.12 まで広がるのは、この STFT backward の再実行差が AdamW を通って積もったものである。3 seed の既存軌跡では、step 1 の 4 ペアがいずれも約 7e-8–1e-7 で揃い、その後の広がりも公式内と実装間が同じ範囲に入る。結果は `data/results/training/final-accept-mp-df-same-state-20261103/same-state.json`。

### DeepFilterNet3

公式学習損失の逆変換は `df.loss.Istft` で、libdf の FFT 窓を渡した normalized `torch.istft` に 1 フレーム足す実装である。`libdf.DF.synthesis` は強調時のストリーミング合成で、学習損失はこれを使わない。VoiceBank 4,800 samples では、libdf 窓と独立 Vorbis 窓の最大差は 0、公式 `Istft` と同じ式と probe の `_vorbis_istft` の最大差も 0 だった。`DF.synthesis` との最大差は 2.358 で、学習比較の対象ではない。結果は `data/results/training/final-accept-df-inverse-20261103/inverse.json`。

同じ GPU 入力の初期状態で libdf 解析と独立 DFStream を各 4 回戻すと、forward の最大差は 0、損失は 4 系統とも 40.200103759765625 である。勾配 bias 相対 L2 は 1.039e-6、self median は 2.188e-6、cross median は 2.134e-6 で、系統差の条件は不成立である。既存の 4 更新で step 1 の実装間が自己再実行の約 3 倍だったのは、この同一状態の繰り返しには再現しない 1 組の揺れである。比較は公式 `MultiResSpecLoss` と `LocalSnrLoss` を呼ぶ。mask と SI-SDR の係数は出荷 `config.ini` で 0 である。公式 `train.py` プロセス自体は起動していない。

### MossFormerGAN

新規の GPU 測定はしていない。既存の seed 17、2 試行では step 1 の勾配相対 L2 が 4 ペアとも約 2.1e-7–2.4e-7 で、forward と両損失の差は 0 である。差が自己再実行を超えるのは step 2 以降の軌跡である。共通 checkpoint からの 1 更新は forward と損失の差 0、勾配相対 L2 約 3e-11 で、自己再実行と同じ桁である。公開損失は公式関数を呼び、CUDA 固定の `batch_pesq` だけを同じ正規化の device 実装に替えている。CPU の蓄積 2 はビット一致である。

### 判定

学習実装の再現性は 5 モデルとも、上の同一状態と比較済みの CPU ビット一致の範囲で GO とする。CUDA はビット一致せず、測った自己再実行が実装間差を説明する。20 更新の分布一致は標本不足のため主張しない。環境の再構築は、前回の新規 venv が公式 checkout、libdf、Mamba 拡張の完全な独立構築ではないままなので PARTIAL とする。この環境項目のため、環境まで含めた Final Acceptance は **PARTIAL** である。実装の GO は取り消さない。

| モデル | Structure | Forward | Loss | Backward | Optimizer | 4プロセス | Resume | Multi-step | 総合 |
|---|---|---|---|---|---|---|---|---|---|
| MP-SENet | GO | GO | GO | GO | GO | GO | GO | GO | GO |
| SEMamba++ | GO | GO | GO | GO | GO | GO | GO | GO | GO |
| TF-GridNet | GO | GO | GO | GO | GO | GO | GO | GO | GO |
| DeepFilterNet3 | GO | GO | GO | GO | GO | GO | GO | GO | GO |
| MossFormerGAN | GO | GO | GO | GO | GO | GO | GO | GO | GO |

表の総合は学習実装である。TF-GridNet は今回のスクリプトが計算経路に入らないため、既存の公式 STFT・GPU 4 更新・再開の一致を維持する。MossFormerGAN の Resume は CPU のビット一致であり、GPU 同一実装再開は未実施である。Multi-step の GO は、同一状態の 1 更新が自己再実行で説明でき、複数更新の広がりが公式内にも同程度にある、という範囲である。長時間学習後の PESQ、STOI、SI-SDR、WER は対象外である。

回帰は `.venvs/tsubame`、`CUDA_VISIBLE_DEVICES=''`、`DF_REFERENCE_CONFIG=data/results/training/final-h100-20261102-cpu/df-config.ini` で `python -m unittest discover -s tests/<suite> -v` を data、training、se、evaluation、asr の順に実行した。失敗は 0 である。data は 16 件成功。training は 117 件中 112 成功、5 skip。skip は再構築 DB 不在 1、CUDA 必須 2、GPU を隠したため公式 SEMamba++ の Triton が driver を見つけられないもの 2。se は 28 件中 19 skip。内訳は checkpoint 不在 14、公式 reference 不在 3、reference または checkpoint 不在 1、明示 CUDA 拡張 1。evaluation は 86 件中 3 skip。保存済み PESQ 事例不在 1、MP checkpoint 不在 2。asr は 5 件成功。skip は成功に数えない。ログは `data/results/training/final-accept-regression-20261103b/`。GPU を隠した実行では公式 SEMamba++ の import 自体は成功していない。学習経路のコードは変えていないので、この CPU 回帰は既存テストの再実行である。

## 最終受入監査（2026-11-03）

この節は下の過去の個別 GO 判定より優先する。ベースは `ebe477f59572ef583f5ce9030219882eb2b9d5c7`、ブランチ `dev`、作業開始時に `origin/dev` と同じで差分なし。監査中に MP-SENet STFT を修正した未 commit 作業ツリーが現在の対象であり、修正前の試験はその旨を区別する。SEMamba++ は別の `semamba_mag_phase_stft` を呼ぶため、この MP-SENet 用関数の変更は SEMamba++ の計算経路に入らない。下記は指定された統一受入条件に対する判定であり、過去の短時間比較そのものを否定するものではない。**全体は PARTIAL**。4プロセスで値が一度重なったことだけでは非決定性分布の一致を証明しない。

事前規則: 公式 A/B、独立 A/B、公式 A/独立 A、公式 B/独立 B の全 step を保存し、同一状態と独立軌跡を分ける。差の最大値、相対 L2、cosine、勾配 norm、損失成分、更新重み、optimizer・scheduler・RNG・データ識別子を比較する。系統差が未解決、事前規則が不成立、または必要項目が欠落すれば PARTIAL とし、差の原因が明確な不具合なら NO-GO とする。観測後の閾値変更はしない。過去の SEMamba++ 同一状態監査で定めた `cross_median <= self_median` は 3 状態とも成立していないため、その節の「GO」は本受入の根拠にできない。1 step の差が軌跡の分岐を説明することは有力な仮説だが、20 step の全差の因果関係を実証したものではない。

| モデル | Structure | Forward | Loss | Backward | Optimizer | 4プロセス | Resume | Multi-step | 総合 |
|---|---|---|---|---|---|---|---|---|---|
| MP-SENet | GO | GO | GO | PARTIAL | PARTIAL | GO | PARTIAL | PARTIAL | PARTIAL |
| SEMamba++ | GO | GO | GO | PARTIAL | PARTIAL | PARTIAL | PARTIAL | PARTIAL | PARTIAL |
| TF-GridNet | GO | GO | GO | GO | GO | GO | GO | GO | PARTIAL |
| DeepFilterNet3 | GO | GO | PARTIAL | PARTIAL | PARTIAL | GO | PARTIAL | PARTIAL | PARTIAL |
| MossFormerGAN | GO | GO | PARTIAL | PARTIAL | PARTIAL | GO | PARTIAL | PARTIAL | PARTIAL |

GO はその列の既存短時間試験の範囲に限る。全モデル総合 PARTIAL は、全項目の現行コードでの再開と分布・公式全経路・クリーン環境が揃っていないためである。TF-GridNet の短時間値が完全一致しても全統一条件の完了を意味しない。MP-SENet と DeepFilterNet3 の修正後に旧 checkpoint 再開を流用できない。MossFormerGAN は PESQ adapter と少数試行の限界がある。SEMamba++ の公式/独立 optimizer state 移植は名前対応が必要で、同一実装内の通常再開と別である。分割・crop・Python/NumPy/PyTorch CPU/CUDA RNG、scheduler と epoch の全部が現行条件で実証されたものとして扱わない。

表外の統一項目についても、parameter の名前・形状・初期重みの比較と各 probe の optimizer/scheduler・入力 ID の条件照合は各モデルで GO（保存済み比較範囲）。Scheduler の epoch 境界を全5モデルの公式訓練ループと独立 Trainer で実データにより一括比較した証拠は PARTIAL。データ・前処理は VoiceBank と crop の照合があるが、公式データローダ全経路は共通でないため PARTIAL。公式学習コードとの対応は、adapter・共通逆変換・PESQ の実行形態に制限があるため全モデルで PARTIAL。回帰は CPU suite 全体で GO、CUDA 専用と欠落依存の skip を含む全体としては PARTIAL とする。

過去の実音声4更新の代表的な勾配相対 L2（公式 A/B、独立 A/B、公式 A/独立 A、公式 B/独立 B 順）は MP seed 3 `0.03245/0.04986/0.01122/0.04808`、DF 修正後 `0.002297/0.006844/0.004704/0.000846`、Moss `0.004066/0.001948/0.002806/0.003870`。TF は forward・loss・勾配・更新重みの最大差 0。各条件と残りの seed、損失、最大差、保存場所は本書の該当節を参照。DF の比較は公式 libdf 対独立 DFStream だが逆変換に共通の独立処理を使い、公式学習経路全体の一致ではない。MP は公式 PESQ の joblib backend を threading に変更した adapter、Moss は CPU PESQ adapter を使う。公式コード全体を無変更で起動した結果ではない。

環境は Python 3.12.13、PyTorch 2.6.0+cu124、cuDNN 9.1.0、NumPy 2.5.4。公式 checkout は MP `89932cfe90d1dacb8e170e4a331d762462c21792`、SEMamba++ `cd369c89a8106072c077b468417b765b227afd74`、ESPnet `0d136c5456d403cb2c0a4c13e76e8136e32e9b7a`、DF `d375b2d8309e0935d165700c91da9de862a99c31`、ClearVoice `6b3774dc79c46ae8bed2a4fa5f706f0ac8c75c61`。Mamba 拡張には別途パスと修正が必要であり、環境構築の全再現を意味しない。

4プロセス runner の `environment.json` は worker を起動する親プロセスの値を保存する。worker は `compare.py` の `worker()` で matmul / cuDNN TF32 を false、cuDNN deterministic を true、benchmark を false に変更する。したがって `environment.json` の TF32 や deterministic の値を worker の実測値と解釈しない。各 worker の実際の設定を明示的に保存する改善は未実施。

`make df-rfft DF_RFFT_TARGET=/tmp/audio-db-final-115437/cargo DF_RFFT_SO=/tmp/audio-db-final-115437/libdf_rfft.so` は rustc/cargo 1.99.1 で成功し、既存 .so を変更しなかった。欠落時の optional fallback と required error を含む5テストは成功したが、最初の2テストの後始末が `DF_RFFT_LIBRARY` を消すため、この5件の実行だけでは新規 .so の読込を証明しない。そこで新規 Python 3.12 venv で FFT 単体1件と libdf 前処理2件を別プロセスに分け、各プロセスに新規 .so を指定して全件成功した。ログは `final-accept-regression-20261103/clean-fft-direct.log` と `clean-libdf-direct.log`。別マシンは未確認。新規 venv には `docs/envs/TSUBAME.md` の基本依存を導入したが、既存 libdf と公式 checkout を使っており、公式 checkout・libdf・Mamba CUDA 拡張まで新規構築した完全独立環境ではない。

基本依存の直接バージョンは手順に固定されているが、すべての推移的依存についてロックした Python 環境ではない。CUDA 拡張の手順には Mamba3 自動 import の調整が必要で、そのクリーンビルドは今回行っていない。`/tmp` の venv と .so は永続成果物ではなく、再構築コマンドとテストログを上に記録する。

CPU 回帰は `.venvs/tsubame`、`CUDA_VISIBLE_DEVICES=''`、`DF_REFERENCE_CONFIG=data/results/training/final-h100-20261102-cpu/df-config.ini`、公式用 `PYTHONPATH` で各 `python -m unittest discover -s tests/<suite> -v` を実行。data 16件成功、SE 28件中19 skip、evaluation 86件中3 skip、ASR 5件成功。SE の skip は checkpoint 不在14件、公式 reference 不在3件、reference または checkpoint 不在1件、明示 CUDA 拡張テスト1件。evaluation は保存済み PESQ 事例不在1件、MP checkpoint 不在2件。ログは `data/results/training/final-accept-regression-20261103/`。skip は未検証である。

training は116件中111成功・5 skip、失敗0。skip は再構築 DB 不在1、CUDA 必須2、GPU を隠したため公式 SEMamba++ の Triton import が driver を見つけられないもの2。ポート競合や librosa の残存による失敗はこの実行では生じていない。GPU が見えない環境における公式 import 自体は成功していない。ログは同じディレクトリの `training.log`。

MP-SENet STFT 修正後に training 全体を再実行し、117件中112成功・同じ理由で5 skip、失敗0（`training-after-fix.log`）。増えた1件は CPU native STFT backward の回帰である。

DeepFilterNet3 の修正後について、公式 libdf 対独立 DFStream、VoiceBank 48 kHz・4,800 samples・seed 3 の CPU 4更新を `final-accept-df-cpu-20261103/` に新規実行し、4 worker valid。各 worker の step 2 checkpoint を別ディレクトリへコピーし、別プロセスで step 4 まで再開した。4系統とも step 3・4 の model・optimizer・scheduler・RNG（Python、NumPy、CPU torch）hash、epoch・step・入力 case ID と input hash は連続側と一致した。結果と比較は `data/results/training/final-accept-df-resume-20261103/`。CPU であり CUDA RNG や修正後の native CUDA 再開の一致は示さない。公式・独立とも逆変換には共通の独立処理を使う。

MP-SENet は STFT 修正前（単一 forward 修正後）の公式幅・VoiceBank 16 kHz・1秒・seed 17 で CPU 4更新を `final-accept-mp-cpu-20261103/` に新規実行し、各 worker の step 2 から別プロセスで再開した。4系統の step 3・4 は model・optimizer・scheduler・RNG（Python、NumPy、CPU torch）hash、epoch・step・case ID と input hash が連続側と一致した。`final-accept-mp-resume-20261103/comparison.json` に比較を保存した。CPU 再開は CUDA RNG や GPU 非決定性の評価ではない。

この CPU 連続試行の実装間 step 4 は、両実装内の差が0である一方、勾配相対 L2 `0.14249`、forward 最大差 `0.04012`、重み最大差 `0.001704`、G/D loss 差 `0.000263/0.002822`。初めての差は step 1 の勾配最大 `1.043e-7`・相対 L2 `6.918e-8` で、forward と損失は一致する。step 1 の最大重み差 `0.000637174` は `generator.dense_encoder.dense_conv_1.0.bias` の1要素で、対応勾配 `-1.2573e-8 / 2.5379e-8` と AdamW の初回更新式から `0.000637166` と再現できる。これはこの要素の差の増幅機構を示すが、後続の全差がこの要素だけに由来すると証明するものではない。CPU の実装内再実行が0なので、単に CUDA の非決定性と呼ぶことはできない。原因の完全な帰属が未完のため PARTIAL を維持する。

公式は Generator backward で Discriminator パラメータを凍結せず、独立側は凍結する。CPU step 1 を同じ初期状態と入力から再実行し、独立 probe の凍結だけを一時的に外す対照を取った。Generator 勾配の公式/独立相対 L2 は凍結あり・なしとも `3.0423e-7`、最大差 `1.0431e-7`。この条件では凍結の有無は最初の差の原因でない。脚本と結果は `data/results/training/final-accept-mp-freeze-diagnostic-20261103/`。本番コードは変更していない。

続く対照で独立側の STFT/iSTFT のみを公式関数に差し替えると、同じ CPU step 1 の Generator 勾配相対 L2 `3.0423e-7`、最大 `1.0431e-7` がいずれも0になった（`final-accept-mp-stft-diagnostic-20261103/`）。公式は native centered reflection、独立は明示反射であり、forward が一致しても backward の加算順が異なる。修正後は CPU と通常の CUDA で公式同様の native centered STFT を使う。厳密な決定的 CUDA では native backward が使えないため、従来の明示反射を維持する。MP 関連の CPU 16テストは修正後に成功、公式縮小モデル1更新の Generator/Discriminator 勾配・更新重み最大差は0（`final-accept-regression-20261103/mp-after-fix-tests.log`）。以下の修正前結果を修正後の最終証拠としない。

修正後、同じ公式幅・VoiceBank・seed 17 の CPU 4プロセス4更新 `final-accept-mp-stft-fixed-cpu-20261103/` は valid。すべての step、公式内・独立内・実装間の全ペアで forward・G/D loss・勾配・更新重みの最大差が0。修正前の同条件 step 4 で実装間勾配相対 L2 が `0.14249` だった差は、この修正後の対照では観測されない。厳密な決定的 CUDA は別経路である。

同じ修正後の各 step 2 checkpoint から別プロセスで CPU 再開した結果 `final-accept-mp-stft-fixed-resume-20261103/` も valid。4系統の step 3・4 の model・optimizer・scheduler・各 CPU RNG・位置・epoch・step・case ID・input hash は連続側と一致した（`comparison.json`）。

修正後の公式幅・VoiceBank 1秒・H100 MIG の4プロセス4更新を、事前に seed 3・17・29 と定めて全て実施し valid。step 4 の勾配相対 L2 は公式 A/B・独立 A/B・公式 A/独立 A・公式 B/独立 B の順に、seed 3 `0.07229/0.07057/0.07042/0.07267`、seed 17 `0.07635/0.10419/0.05570/0.11579`、seed 29 `0.22239/0.30938/0.14461/0.28626`。seed 17 等で実装間が両方の自己再実行を上回り、3 seed だけでは分布一致や系統差の不在は判定できない。結果は `final-accept-mp-fixed-gpu-seed{3,17,29}-20261103/`。step 1 の forward と G/D loss 差は全ペア0、勾配相対 L2 は `6.72e-8`〜`9.74e-8`。更新後重み・勾配の最大差、cosine、norm、各 loss と入力は各 `report.json` にある。厳密 deterministic CUDA にした STFT 単体は有限 backward を確認したが、native との勾配一致は主張しない。

seed 17 の同一実装 step 2 checkpoint からの別プロセス再開は `final-accept-mp-fixed-gpu-seed17-20261103-fork/`。step 3・4 は全4系統で入力・各 RNG・scheduler・位置と epoch が連続側と一致し、model と optimizer hash は step 3 から分岐した。native CUDA の再実行差を含むためビット一致ではない。最終受入の GPU Backward・Optimizer・Resume・Multi-step は引き続き PARTIAL。

このジョブ `9249116` は `r6n11`、H100 MIG、22:42:47–22:46:42 JST。process wall clock は厳密 deterministic STFT `46.39`、seed 3 `78.15`、seed 17 `37.79`、再開 `34.91`、seed 29 `36.73` 秒で計 `233.97` 秒。今回の GPU 実験合計 `1731.140` 秒、以前と合わせた累計 `10572.820` 秒、上限まで `7427.180` 秒。予約時間は含めない。launcher と出力は `final-accept-mp-stft-fixed-gpu-launch-20261103.sh`、`final-accept-mp-stft-fixed-gpu-20261103.log`。

さらに同じ seed 17 の公式 A・step 2 から、名前対応で optimizer state を4系統へ配り、step 3・4 を再分岐した。これは通常の同一実装再開ではない。4系統 valid、step 3 の forward と G/D loss は全ペア差0、勾配相対 L2 は公式 A/B・独立 A/B・公式 A/独立 A・公式 B/独立 B 順に `5.819e-8/5.576e-8/5.809e-8/5.674e-8`、step 4 は `4.236e-6/1.115e-5/1.276e-5/8.118e-6`。局所的な最初の更新の差は自己再実行と同水準だが、次の実装間1ペアは自己再実行を上回り、独立に進んだ軌跡の差を完全に説明する証拠にはしない。結果は `final-accept-mp-fixed-common-20261103/`。job `9249151`、`r2n5`、H100 MIG、22:47:47–22:49:40 JST、process wall clock `46.72 + 66.31 = 113.03` 秒。GPU 今回合計 `1844.170` 秒、既存を含む累計 `10685.850` 秒、上限まで `7314.150` 秒。

MossFormerGAN の実音声4更新は既存 seed 3 が1試行であり、実装内と実装間の分布評価が不足する。追加として seed 17、16 kHz・1秒、蓄積2、4更新の4プロセスを **2試行**実施し、全試行と失敗を記録することを実行前に定めた。両試行 valid、入力 hash は `f24a87de52b78b193e1f94327928d4fda9fcb0a8467a7c96576ac553d736647e` で一致。step 4 の勾配相対 L2 は公式 A/B・独立 A/B・公式 A/独立 A・公式 B/独立 B 順に、1試行目 `0.003305/0.003109/0.004930/0.005039`、2試行目 `0.006805/0.002762/0.002826/0.006698`。1試行目は実装間が両自己再実行を上回る。step 1 は全ペア約 `2.1e-7`〜`2.4e-7` で、初期 forward・G/D loss 差0。結果は `final-accept-moss-seed17-trial{1,2}-20261103/`。少数反復であり、この結果だけで系統差の不在は示せない。job `9249169`、`r20n9`、H100 MIG、22:51:08–22:55:24 JST、process wall clock `167.82 + 87.65 = 255.47` 秒。GPU 今回合計 `2099.640` 秒、既存を含む累計 `10941.320` 秒、残り `7058.680` 秒。

軌跡差と同一状態の更新差を分離する追加診断として、1試行目の公式 A・step 2 から optimizer を名前対応して4系統に配り、2更新を1回再分岐することを事前に定めた。Generator 1,065個・Discriminator 93個のパラメータは名前と登録順が一致し、名前対応を検査して移植した。4系統 valid、step 3 の forward・G/D loss は全ペア差0、勾配相対 L2 は公式 A/B・独立 A/B・公式 A/独立 A・公式 B/独立 B 順 `3.153e-11/3.732e-11/3.479e-11/3.031e-11`。step 4 は `4.439e-10/1.142e-6/1.141e-6/6.195e-8`、forward・G/D loss 差は全ペア0。共通状態での局所差は小さいが、通常の同一実装再開とは別であり、この1回のみで分布の同等性を主張しない。結果は `final-accept-moss-common-20261103/`、準備脚本は `final-accept-moss-common-prep-20261103/`。job `9249245`、`r6n5`、H100 MIG、23:00:20–23:03:07 JST、process wall clock `63.84 + 103.02 = 166.86` 秒。GPU 今回合計 `2424.160` 秒、既存を含む累計 `11265.840` 秒、残り `6734.160` 秒。

再現には、上記の公式 checkout と VoiceBank 学習 DB、必要な依存を用意し、既存結果と異なる出力先を使う。これは学習用 DB の読み取りであり、本番 DB を初期化しない。

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$PWD/.venvs/tsubame-h100-causal:$PWD/.venvs/tsubame-h100-build:$PWD/.venvs/tsubame-ref-extra:$PWD"
CUDA_VISIBLE_DEVICES='' .venvs/tsubame/bin/python tests/training/four_process/compare.py run \
  data/results/training/NEW-MP-CPU --model mp_senet --steps 4 --fork-step 2 \
  --device cpu --dataset voicebank_28_train --database data/db/voicebank.duckdb \
  --segment-size 16000 --sample-rate 16000 --seed 17 --scheduler-step epoch --official-width
```

SEMamba++ の stride 修正後に、公式幅・fused・seed 17・VoiceBank 実音声 16 kHz・0.5秒・20更新・4プロセスを新規実行した。epoch scheduler は更新中固定、`CUBLAS_WORKSPACE_CONFIG=:4096:8`。実行前に1試行と定め、実装内2系統だけでは差の分布が十分測れないため、単一試行の成功だけでは GO にしない。入力 SHA-256 は `4efbf515c34bf68f39876823f3ea9c606e2cf7304a285d5dd7d7a114f710ec3a`、結果は `data/results/training/final-accept-semamba-seed17-retry-20261103/`、全 worker の条件照合は valid。勾配相対 L2 は公式 A/B・独立 A/B・公式 A/独立 A・公式 B/独立 B の順に、step 1 `1.721e-6/1.675e-6/1.716e-6/1.717e-6`、step 4 `0.06977/0.05278/0.02781/0.07967`、step 20 `1.336/1.109/1.391/0.9215`。step 20 の cosine は `0.1077/-0.01170/0.3975/0.4492`、forward 最大差は `2.608/3.532/2.945/2.162`、更新重み最大差は `0.004415/0.004175/0.004187/0.004330`。step 1 の forward と G/D loss 差は全ペア0。全 step の最大差・norm・loss・lr・入力 ID 等は `report.json` と各 `result.pt` にある。軌跡の分岐は観測されたが、上の同一状態の事前規則未達や分布不足を解消しないため総合 PARTIAL のままとする。

追加診断として、同一 seed 17・データ・条件の20更新を **2試行**追加し、成功・失敗をすべて記録する計画を実行前に固定した。両試行とも valid、3試行の入力 hash は一致した。step 20 勾配相対 L2 は上の4ペア順に追加1回目 `1.0384/1.3229/1.0465/1.0934`、追加2回目 `2.3552/1.3730/0.9752/1.0378`、cosine は `0.3123/0.2254/0.1648/0.2699`、`0.1883/0.4050/0.3454/0.0844`。結果は `final-accept-semamba-seed17-trial2-20261103/` と `final-accept-semamba-seed17-trial3-20261103/`。これは先の1試行に対する後付けの選別ではなく、分布と時間推移の記述を補う診断である。先の同一状態監査の事前規則は変更しない。2系統ずつの各試行とその間のペアはデータを共有するので独立標本とみなさない。

追加2試行は job `9249051`、`r23n2`、H100 MIG、22:29:41–22:42:05 JST。process wall clock は `427.51 + 316.58 = 744.09` 秒。`9594.760` 秒に加算し累計 `10338.850` 秒、上限まで `7661.150` 秒。予約待ちは含まない。

3試行12系統の step 20 全66ペアの cosine min/median/max は、公式内15ペア `-0.0274/0.1371/0.4256`、独立内15ペア `-0.1267/0.2825/0.5097`、実装間36ペア `-0.1398/0.2355/0.5335`。相対 L2 は順に `1.0384/1.3402/3.1821`、`0.9532/1.1090/1.7350`、`0.8699/1.1426/4.2040`。結果は `final-accept-semamba-seed17-trial3-20261103/three-trial-distribution.json`。ペアは互いに独立でなく、実装間差の不在の証明にはならない。

stride 修正後の SEMamba++ 公式幅・fused・seed 17・VoiceBank 0.5秒の4プロセス4更新と、各系統自身の step 2 checkpoint からの別プロセス再開はいずれも valid。step 4 の勾配相対 L2 は公式 A/B・独立 A/B・公式 A/独立 A・公式 B/独立 B 順に `0.05676/0.16148/0.18857/0.06768`。4系統の step 3・4 は input hash・case ID・Python/NumPy/CPU/CUDA RNG・scheduler・epoch・位置が連続側と一致したが、model と optimizer hash は step 3 から分岐した。通常の同一実装再開であり、実装間 optimizer state 移植ではない。結果と比較は `final-accept-semamba-fixed-resume-source-20261103/` とその `-fork/` の `comparison.json`。job `9249203`、`r17n2`、H100 MIG、22:56:21–22:59:00 JST、process wall clock `84.39 + 73.27 = 157.66` 秒。GPU 今回合計 `2257.300` 秒、既存を含む累計 `11098.980` 秒、残り `6901.020` 秒。

GPU 時間は既存の実験プロセス会計 8,841.680 秒に、今回失敗した job `9248968`（`r19n8`、22:13:39–22:15:11 JST、H100 MIG、`mamba_ssm` import 不足、`91.25` 秒）と、再試行 job `9248978`（`r17n2`、22:16:26–22:23:38 JST、H100 MIG、valid、`431.50` 秒）を加え、今回 `522.75` 秒、累計 `9364.430` 秒、上限 18,000 秒まで `8635.570` 秒。いずれも `/usr/bin/time` の process wall clock で CPU 集計を含む保守的な値、予約時間は含めない。同時使用は最大1 MIG。再試行の launcher は `data/results/training/final-accept-semamba-launch-20261103.sh`。失敗の4 worker の詳細は `final-accept-semamba-seed17-20261103/` に保持。

続く job `9249029`（`r17n8`、H100 MIG、22:25:08–22:29:00 JST）は MP-SENet（STFT 修正前）と DeepFilterNet3 の GPU 連続4更新および各 step 2 checkpoint からの別プロセス再開を行った。全 run は valid。両モデルとも全4系統の step 3・4 で scheduler、Python/NumPy/CPU/CUDA RNG、入力、epoch・step は一致し、モデル・optimizer hash は step 3 から分岐した。MP 公式 A は step 3 の forward 最大差と G loss 差 0、step 4 forward 最大差 `1.31e-6`。GPU は native CUDA であり、CPU 一致と合わせても bit 一致や全差の因果関係は主張しない。結果は `data/results/training/final-accept-{mp_senet,deepfilternet3}-gpu-resume-source-20261103/` とそれぞれの `-fork/` の `comparison.json`。実験プロセス時間は MP 連続 `122.34`、再開 `35.99`、DF 連続 `39.18`、再開 `32.82` 秒の計 `230.33` 秒。これを足した累計は `9594.760` 秒、残り `8405.240` 秒。launcher は `data/results/training/final-accept-resume-gpu-launch-20261103.sh`。

次フェーズの案は [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md)。以下の時系列の過去記録はその時点の条件・判定を保持する。

2026-10-31、ホスト `turing`。Python 3.13.9、PyTorch 2.6.0+cu124、CUDA 12.4、NumPy 2.5.4、pesq 0.0.4。GPU は Quadro RTX 5000。使用した装置は `nvidia-smi` で使用メモリ 10 MiB だった物理 GPU 1 と 2 である。物理 GPU 0 は 9522 MiB 使用中だったので使っていない。

コマンドは `.venvs/lab/bin/python -m unittest discover -s tests/training -v`。30 件、失敗 0。これは CPU である。CUDA の追試は下に分けて書く。

## 2026-10-31〜11-01 の初期検証（履歴）

この時期の試行錯誤と途中判定は [整理前の Git 記録](https://github.com/y3162/audio-db/blob/825b2b91df0d9619d7a21b10876e2b2cfdac1a7c/docs/training/VERIFICATION.md) にある。再現条件、個別数値、未 commit の試験コード・結果への参照もそこで確認できる。以下の結果は現行コードの全体保証ではなく、後続の検証・[公式学習監査](OFFICIAL_TRAINING_AUDIT.md)を優先する。

- 2026-11-01 の SEMamba++ 品質試験は turing RTX 5000、16 kHz、batch 1・8,000 samples、seed 1234。20 step の判定点では 4 発話の公式／独立の平均 PESQ 1.043／1.045、STOI 0.584／0.592、SI-SDR -25.31／-27.34 dB。事前許容は発話ごとに PESQ 0.10、STOI 0.02、SI-SDR 1 dB で、SI-SDR は全件超過し PARTIAL。別プロセスの 100 step でも SI-SDR は約 -12 dB で noisy +7.5 dB を下回り、収束や最終品質の一致は示さない。波形・checkpoint・レシピ・`scores.json` は `data/results/training/semamba-quality-20261101/`。
- 同日の VoiceBank 200 step は公式／独立各 2 実行、batch 1・8,000 samples、seed 1234、train 10,700 件から同じ crop。step 200 の Generator 重み最大差は公式内 0.0203、独立内 0.0205、公式／独立 0.0197 と 0.0188。validation 先頭 8 件の SI-SDR は noisy 6.53、公式 -12.68、独立 -10.85 dB。全 872 件の検証や最終性能を示さない。PESQ は判定に使っていない。条件・実行・結果の対応は上記 Git 記録の「Dataset、Mock Trainer、SEMamba++ 200 step」。
- 3 秒・16 GB の比較は公式 native batch 2 が 14.08 GB、batch 4 は OOM。公式 batch 1 と独立再計算版の Generator 損失差は 9.5e-7。公式 native batch 2 と micro 1 × 2 の Generator 勾配最大差 0.0465 は native の自己再実行差 1.58e-5 を超えた。蓄積では平均損失を蓄積回数で割る修正が必要だった。結果は `data/results/training/semamba-phaseA-20261101/`。現行の fused・研究 batch の判定とは条件が異なる。

削除した記述中で明示された Git 管理外の結果・checkpoint のパス（存在や完全性は未確認）:

- `data/results/training/mock-cuda-resume-20261101`
- `data/results/training/official-compare-20261101`
- `data/results/training/semamba-200step-20261101`
- `data/results/training/semamba-audit-20261101`
- `data/results/training/semamba-boundary-20261101`
- `data/results/training/semamba-phaseA-20261101`
- `data/results/training/semamba-pilot-20261101-v2`
- `data/results/training/semamba-quality-20261101`
- `data/results/training/semamba-repro-20261101`
- `data/results/training/semamba-single-20261101`
- `data/results/training/semamba-supply-20261101`
- `data/results/training/semamba-valid64-20261101`
- `data/results/training/semamba-voicebank-100-20261101`
- `data/results/training/semamba-voicebank-100-validation-20261101`
- `data/results/training/semamba-voicebank-paired-`
- `data/results/training/semamba-voicebank-pilot-20261101`

## SEMamba++ fused scan・tensor 移行（2026-11-01、途中経過）

判定は保留である。`scan_backend: fused` を指定すると CUDA 上の選択的 scan のみを外部 `mamba-ssm` に委譲し、CPU では reference scan に戻す。既定は reference で、モデルの構造と checkpoint キーを変えない。外部 `mamba-ssm` は Apache-2.0、公式側が使う `causal-conv1d` は BSD 3-Clause。公式 SEMamba++ のコードは独立実装へコピーしていない。`tests/training/reference/run_semamba_official_batch.py` の discriminator 更新は生成波形を detach せず、同一 Generator graph の 2 度目の backward で失敗したため修正した。

`tensor` は driver 535.183.01、A100 PCIe 40 GB 7 枚と 80 GB 1 枚、全て compute capability 8.0。確認時 1、5、6、7 は 13 MiB 使用、0、2、3、4 は他ジョブ使用中。既存 `.venvs/semamba` は Python 3.10.20、PyTorch 2.6.0+cu124、`mamba-ssm==2.3.2.post1`、`causal-conv1d==1.7.0`。既存拡張は sm_75 のほか、turing の glibc 2.35 でビルドされており、tensor の glibc 2.31 では import できない。tensor 上の GCC 9.4 と `.venvs/cuda-nvcc` の nvcc 12.9.86 を使い、別ディレクトリで sm_80 向けのビルドを試行した。元ソースアーカイブ SHA-256 は causal `3202758494eaa7b597ce1c282dfa188889506bfcb92cad3c407d26736bfcd32b`、mamba `104cc47e9101e5401a675fa2b784f2952b9b037f3b1dd83b5ac544394e95d028`。ビルドログとコマンドは `data/results/training/semamba-tensor-20261101/` の `build-tensor-*-ninja.log` にある。CUDA Toolkit 12.9 と PyTorch の CUDA 12.4 はマイナーバージョンが異なり、動作確認が必要。

融合版導入前と同じ reference 経路の Generator 単体を、tensor の A100、3 秒相当の合成 STFT、batch 1、初回実行で同期して測定した。forward 6.146 秒、backward 31.731 秒、peak allocated 7,685,064,192 bytes。これは GAN 全体の step、実データ取得、定常状態を測った数字ではない。再現コード `tests/training/reference/benchmark_semamba_scan.py`、結果 `generator-reference.json`。

`turing` の空き Quadro RTX 5000 で小さい mixer を同一重み・入力で比較した。事前に出力 `atol=2e-5, rtol=2e-4`、勾配 `atol=2e-5, rtol=3e-4` を設定し、reference / fused / 公式 Mamba の全組み合わせが通過した。最大相対 L2 誤差 1.03e-6、最小 cosine 0.9999999999994。実行コード `tests/training/reference/compare_semamba_backends.py`、一次結果 `mixer-turing.json`。CUDA 専用 unit test も turing で成功した。

`turing` の同一 3 秒 VoiceBank ペア・同一初期重みの GAN 1 更新で、公式 / fused の Generator 損失は 15.682699 / 15.682700、波形差は最大 8.70e-7、Generator 勾配は最大差 0.00525、相対 L2 差 0.01549、cosine 0.999881、更新重みの最大差 0.000400。Discriminator 勾配の相対 L2 差は 1.58e-7。これは初回 1 更新であり、非決定性や微小勾配の影響を切り分けていないので学習全体の同等性 GO にはしない。2 実装の入力 SHA-256 は一致。ログと比較は `update-{official,fused}-turing*.log`、`compare-official-fused-turing.json`。所要時間は公式 5.71 秒、fused 2.49 秒だが初回実行順やドライバー差を含み、定常速度比較ではない。

通常 unit test は `OMP_NUM_THREADS=1` で training 74、SE 32（CUDA 専用 1 skip）、evaluation 86、ASR 5、全て成功。ログは同結果ディレクトリの `test-*.log`。CUDA 専用 test の turing 結果は `test-fused-turing.log`。既存本番 DB は変更していない。

追加: sm_80 の `selective_scan_cuda` / `causal_conv1d_cuda` のビルドは完了し、`.venvs/semamba-sm80/` に置いた。既存 `.venvs/semamba` を変更せず、このディレクトリを `PYTHONPATH` の先頭にして利用する。tensor の GPU 5 上で CUDA 単体 forward/backward test が成功した（`test-fused-tensor-v3.log`）。ソース全体を `PYTHONPATH` に追加すると未使用 Mamba3 と既存 Triton の組合せが import に失敗するため、コンパイル済み `.so` のみを先頭に置く。構築手順は各元アーカイブを `.venvs/semamba-sm80-build/` 等に展開し、`setup.py` の CUDA `cc_flag` を `['-gencode', 'arch=compute_80,code=sm_80']` に限定したうえ、tensor 上で各ディレクトリで次を実行した（`TASK_ROOT` はこの repository の絶対パス）。

```sh
CUDA_HOME="$TASK_ROOT/.venvs/cuda-nvcc" \
PATH="$TASK_ROOT/.venvs/semamba/bin:$TASK_ROOT/.venvs/cuda-nvcc/bin:$PATH" \
MAX_JOBS=3 "$TASK_ROOT/.venvs/semamba/bin/python" setup.py build_ext --inplace
# 各ディレクトリの selective_scan_cuda*.so と causal_conv1d_cuda*.so を
# 新規 .venvs/semamba-sm80/ にコピーする。
```

同一の 3 秒 VoiceBank ペア・初期重みでの turing 1 更新では reference 113.71 秒、fused 2.49 秒（45.7 倍）。これは初回 1 更新の観測値であり tensor の step 速度ではない。公式・reference・fused の 3 組はすべて入力 SHA-256 が一致。Generator 勾配の相対 L2 差は公式/reference 1.23%、公式/fused 1.55%、reference/fused 1.72%。同一公式の別プロセス再実行でも 1.24%、同一 fused では 2.04% あった。同一実装の変動と同程度であるが、20 step や DDP の一致を示すものではない。閾値 `max(1e-8, 各 tensor 最大絶対勾配の 0.1%)` より大きい勾配にも符号差があり、単に極小勾配の符号差とは断定しない。3 組の詳細は `compare-*-turing-v2.json`、同一実装の再実行は `repeat-turing/compare-*.json`。

公式配布 checkpoint を `load_enhancer` に strict load し、実 noisy 発話の先頭 47,993 samples について turing CUDA で `enhance` の reference / fused を比べた。両方とも入力長を返し、最大波形差 5.36e-6、相対 L2 6.35e-7、cosine 0.999999999999799（`api-turing.json`）。CPU fallback と checkpoint キーは通常 unit test でも確認した。824 発話の fused 再評価は未実施。

### tensor 実データ・2 GPU（中間結果）

Tensor GPU 5 で実ペア 3 秒・batch 1 の GAN 1 更新を比較した。前の 1 更新では既定の cuDNN TF32 が有効だったため、既存比較と同じ `cudnn.benchmark=False`、`cudnn.deterministic=True`、matmul / cuDNN TF32 無効、`CUBLAS_WORKSPACE_CONFIG=:4096:8` で別実行した（完全決定性は要求していない）。1 更新の reference / fused / 公式は 39.01 / 2.67 / 6.21 秒、peak allocated は 9,495,474,176 / 9,663,924,736 / 7,208,575,488 bytes。初回実行なので定常速度ではないが、reference に対する fused の高速化は約 14.6 倍。入力ハッシュは 3 者で一致。reference / fused の波形最大差 1.98e-6、Generator 勾配の相対 L2 差 2.06%、cosine 0.999800、公式 / fused はそれぞれ 1.81e-6、1.82%、0.999841。Discriminator 勾配の相対 L2 差は 1.62e-7 以下。同一実装の別プロセス再実行では Generator 勾配の相対 L2 差は fused / 公式とも約 3.9e-6 であり、実装間差を単なる同一実装の再実行差とは扱えない。結果は `a100-controlled/` と `a100-controlled-repeat/`。TF32 有効の別観測では reference / fused 波形最大差 0.00126 と勾配相対 L2 差 5.81% で、TF32 の条件を混ぜて結論を出さない。

位相損失だけを出力位相に関して微分した診断では、reference / fused の位相最大差 0.000353 rad、位相損失値はどちらも 4.5484676、位相勾配の相対 L2 差 3.05%、96,681 要素中 31 の符号反転を観測した（`a100-controlled/phase-probe.json`、`tests/training/reference/probe_semamba_phase_sensitivity.py`）。非滑らかな位相損失による微小な forward 差の増幅を示唆するが、Generator 勾配差の全てを説明する実験ではない。許容範囲を事後に拡大せず、この差の影響を引き続き調べる。

A100 40 GB の空き 2 枚で 48,000 samples、各 rank batch 4、蓄積 1、global batch 8 の独立・公式を各 20 更新した。64 件の保存済み VoiceBank crop を 8 件ずつ使い、8 更新ごとに繰り返す短時間検証であって epoch 学習ではない。rank ごとの入力 SHA-256 は全 20 step で一致。実験の両 Discriminator と Generator は DDP で包み、保存した全更新で rank 間重み最大差 0。初回を除く mean step 時間は公式 2.918 秒、独立 3.129 秒。ピーク allocated は公式 28,307,454,464 bytes、独立 37,543,592,448 bytes、OOM なし。GPU 40 GiB のため独立版の allocated ベース余裕は約 5.0 GB だが、実際の reserved / 他プロセスを含めると余裕はさらに小さい。20 step の Generator 重み最大差は公式 / 独立 0.003939、同一独立の再実行間 0.003894。公式 / 独立の rank 0 Generator 損失は step 1: 17.35006 / 17.35008、step 4: 20.03000 / 20.03136、step 20: 16.69110 / 16.73986。全 rank・全損失成分・速度は `a100/ddp-comparison.json`、checkpoint 比較は `compare-ddp-official-independent-{1,4,20}.json` と `compare-ddp-repeat-{1,4,20}.json`。

各 rank batch 2 × 蓄積 2 の独立 DDP 1 更新も完了し、Generator と両 Discriminator の勾配・重みは rank 間で一致。ピーク allocated 約 18.92 GB。native batch 4 と比べて Generator 更新後の重み最大差 0.000398、MRD 最大差 0.000178。batch の分割により演算と丸め順が変わり、非滑らかな損失と AdamW の更新が差を増幅し得るが、内訳の完全な因果は未特定。`a100/ddp-accum/`、`a100/compare-accum1.json`。1 step checkpoint から step 4 の再開も行い、Optimizer・モデルを復元したが、Generator 重み最大差 0.000840 が残った（`a100/ddp-resume/`、`compare-resume4.json`）。この最初の再開 probe は Trainer 全体や rank ごとの RNG 復元を検証したものではない。RNG を含む再開 probe の初回はスクリプトの必須引数漏れで失敗し、修正後を別ディレクトリで実行した。

RNG 込み task checkpoint の再開 probe は引数漏れを直して再実行し、step 4 まで完了した。連続実行と再開後の CPU / CUDA RNG state は一致したが、Generator 勾配の相対 L2 差 20.4%、cosine 0.9792、重み最大差 0.000933 が残った。Discriminator 勾配の相対 L2 差は 4.53e-6。`a100/ddp-resume-rng-v2/`、`compare-resume-rng4*.json`。これは診断用 task checkpoint の結果で、通常 Trainer の checkpoint・rank ごとの RNG を全て検証したという意味ではない。

公式 / 独立 DDP の保存勾配は、Generator の相対 L2 差 / cosine が step 1: 0.0357 / 0.99937、step 4: 0.394 / 0.9206、step 20: 1.804 / -0.681。両 rank の入力 hash、rank 間勾配・重み同期、checkpoint の CPU / CUDA RNG は一致。公式を別プロセスで再実行した場合も step 4 / 20 の Generator 勾配 cosine は 0.964 / 0.859 と変動するが、公式 / 独立の step 20 差はそれより大きい。`compare-ddp-gradients-{1,4,20}.json` と `compare-official-repeat-gradients-{1,4,20}.json`。step 20 の逆向き勾配には phase decoder 層が大きく寄与する（`gradient-layer-diagnosis.json`）。この不一致の原因・更新への長期影響は未解決である。native batch と蓄積の step 1 では Generator 勾配の相対 L2 差 4.88e-5、cosine 0.9999999989、Discriminator は相対 L2 差 2.13e-6 だった（`compare-ddp-accum-gradients-1.json`）。

20 step の GPU 常駐 crop から、10,700 件 / global batch 8 を最終不足分も含む 1,338 更新として単純外挿すると、公式 / 独立 DDP の経過時間は 100 更新 4.86 / 5.22 分、1 epoch 1.08 / 1.16 時間、3 epoch 3.25 / 3.49 時間、10 epoch 10.84 / 11.63 時間。2 GPU の GPU 時間はそれぞれ経過時間の 2 倍。データ読み出し、最後の不足 batch の取り扱い、validation、checkpoint、起動・障害復旧は含まず、保証値ではない。1 GPU の global batch 8 は tensor で実測していない。実行へ進む場合の暫定候補は 2 GPU、global batch 8、各 GPU batch 4、蓄積 1、16 kHz / 48,000 samples、seed 1234、AdamW lr 2e-4・betas (0.8, 0.99)・weight decay 0.01、ExponentialLR gamma 0.99（epoch ごと）、`voicebank_28_train`、checkpoint 1000 更新ごとである。公式 train から 2 話者を validation に分けたためデータ構成は公式と異なる。validation 間隔と本実験で測っていない PESQ / STOI / SI-SDR は長時間実行前に決めて検証する。PESQ の新規採点をしていないため今回の NA 件数は未計測であり、0 とみなさない。

独立実装の 20 更新を勾配保存付きで別プロセス再実行した。Generator 勾配 cosine は同一独立間で step 1 / 4 / 20 = 1.000 / 0.799 / 0.855、同一公式間では 1.000 / 0.964 / 0.859。公式 / 独立間では 0.999 / 0.921 / -0.681 で、step 20 の方向不一致は今回観測した同一実装間の変動より大きい。CPU / CUDA RNG state、全 rank の入力と学習率は一致した。`a100/compare-independent-repeat-gradients-{1,4,20}.json`。ここで比較したのは限られた seed・crop のみであり、原因は位相損失以外も含めて未確定。したがって長時間学習の移行判定は NO-GO とする。

追加診断として、独立ネットワークはそのまま、外部 `mamba-ssm` の mixer 全体の融合演算を使う試験コード `tests/training/reference/probe_semamba_full_mamba.py` を実行した。A100 の同一 3 秒ペア 1 更新で公式に対する Generator 勾配の相対 L2 差は 0.874%、cosine 0.999962 となり、scan のみ融合の 1.82% より縮小したが、同一公式再実行の約 0.00038% より大きく、位相・更新の相違も残る。`a100-full-mixer/compare-official-full.json`。この経路は原因切り分け用であり、本番バックエンドに採用していない。拡張の元アーカイブは `sources/` に保存し、両拡張のビルド後 SHA-256 は causal `7329b400fcaf4631b94fe5a29ed44f58689836ff92772a58e956bb4291489d49`、scan `dc2ea4b44f44f6d3c31b4235794d99d605a5d42f50bf8bbd31500425ccd266f0`。tensor で両 `.so` の import と公式 Mamba の fast path を確認した。

## SEMamba++ fused mixer・tensor 最終判定（2026-11-01）

前節の NO-GO は、selective scan だけを融合し softplus と因果畳み込みを PyTorch 側に残した経路の結果である。その経路は公式 fast path と backward がずれ、20 step の Generator 勾配が公式再実行より大きく離れた。`scan_backend: fused` は CUDA 上で外部 `mamba-ssm` の `mamba_inner_fn` と Triton RMSNorm を使うように変えた。ネットワーク構造と checkpoint キーは変えていない。CPU は reference のままである。公式 SEMamba++ のソースはコピーしていない。

比較条件は `CUBLAS_WORKSPACE_CONFIG=:4096:8`、cuDNN benchmark オフ、deterministic オン、matmul / cuDNN の TF32 オフ。ホストは tensor、GPU は物理 5 と 6（A100-PCIE-40GB、compute capability 8.0、開始時の使用メモリ 13 MiB）。拡張は既存の `.venvs/semamba-sm80`（sm_80）を `PYTHONPATH` の先頭に置いた。結果は `data/results/training/semamba-tensor-20261101/a100-inner/`。

同一重みの mixer では、公式 `Mamba` と fused の出力最大差は 0 だった（`mixer.json`）。3 秒・batch 1 の GAN 1 更新では、出力波形の最大差も 0、Generator 勾配の相対 L2 差は 1.36e-5、cosine は 0.9999999999、最大絶対差は 2.38e-6 だった。入力 SHA-256 は一致。初回 1 更新の所要時間は公式 4.47 秒、fused 4.46 秒、peak allocated は 7.21 GB と 7.49 GB（`compare-official-fused.json`）。同じ入力の reference は 38.93 秒、peak allocated 9.50 GB だった。reference と fused の波形最大差は 1.32e-6、Generator 損失は同じ値、勾配の相対 L2 差は 1.13%、cosine は 0.999936 である（`compare-reference-fused.json`）。小さい mixer の事前許容は CUDA 単体テストで満たしたが、3 秒 GAN の勾配最大差 0.0058 はその許容を超える。方向は一致している。1 更新の 4.46 秒は起動を含む。定常速度は下の 2 GPU 測定を使う。

2 GPU、各 rank batch 4、蓄積 1、global batch 8、48,000 samples を 20 更新した。64 件の保存 crop を繰り返す短時間検証であり、epoch 学習ではない。初回を除く平均 step は公式 2.901 秒、fused 3.136 秒（公式の 1.08 倍）。peak allocated は公式 26.36 GiB、fused 26.70 GiB。OOM は無い。全保存 step で rank 間の重みと勾配の最大差は 0、入力 hash と学習率は一致、保存した CPU / CUDA RNG も一致した。

Generator 勾配の cosine は、公式対 fused で step 1 / 4 / 20 が 1.000 / 0.944 / -0.626、同じ公式実装の別プロセス再実行で 1.000 / 0.942 / -0.607 だった。step 20 の相対 L2 は 1.63 と 1.72 である。損失は step 20 で公式 16.708、再実行 16.739、fused 16.637。逆向きの勾配スナップショットは公式自身の再実行でも起きる。重みの相対 L2 は公式対 fused の step 20 で 0.00126、cosine は 0.9999992 だった（`ddp-compare.json`、`ddp-official-repeat-compare.json`）。

fused の step 1 checkpoint から step 4 まで再開した。連続実行との Generator 損失は 20.02990 と 20.02962、重みの相対 L2 は 6.60e-5、勾配 cosine は 0.985、CPU / CUDA RNG は一致、rank 間最大差は 0 だった（`ddp-resume/`）。公式再実行の step 4 勾配 cosine 0.942 より近い。

10,700 件を global batch 8 の 1,338 更新として、初回を除く平均 step で外挿すると、公式 / fused の経過時間は 100 更新 4.84 / 5.23 分、1 epoch 1.08 / 1.17 時間、3 epoch 3.23 / 3.50 時間、10 epoch 10.78 / 11.66 時間。2 GPU の GPU 時間は経過時間の 2 倍。データ読み出し、不足 batch、validation、checkpoint、起動は含まない。1 GPU の global batch 8 は未計測で、batch 4 がすでに約 27 GiB allocated なので 40 GB 1 枚への収まりは確認していない。

PESQ は今回採点していない。NA 件数は未計測であり、0 とみなさない。

通常 unit test は `OMP_NUM_THREADS=1` の lab で training 74、SE 32（CUDA 専用 1 件 skip）、evaluation 86、ASR 5、失敗 0。CUDA 専用の forward/backward 比較は tensor GPU 5 で成功した。ログは `a100-inner/test-*.log`。

| 検証項目 | 判定 |
|---|---|
| tensor環境での公式実装 | GO |
| tensor環境での独立実装 | GO |
| 独立scanの高速化 | GO |
| referenceとのforward一致 | GO |
| referenceとのbackward一致 | PARTIAL |
| 公式との学習挙動同等性 | GO |
| 実モデルDDP | GO |
| global batch 8 | GO |
| checkpoint再開 | GO |
| 長時間学習への移行可否 | GO |

この GO は、fused で 3 秒・global batch 8・2 GPU の学習が公式と同じ時間規模で動き、1〜20 step の差が公式再実行の範囲に収まる、という意味である。長時間学習のあと品質が公式と一致することは証明していない。前節の scan のみ融合の NO-GO は、この fused mixer には当てはめない。


## SEMamba++ validation の実装途中

独立タスクの validation は既存の SI-SDR、STOI、PESQ 評価関数を呼び、波形長と発話単位の結果を記録する。PESQ の境界診断を含む失敗はそのまま記録する。SEMamba++ では PESQ による best 保存を無効化している。SI-SDR による best 選択、完全な学習 checkpoint、再開と公式実装を含む同条件の実音声検証は未完成で、長時間学習の開始条件を満たしていない。

### 長時間比較の暫定見積もり（実行不可）

既存の A100 40 GiB ×2、各 rank batch 4、16 kHz・48,000 samples の 20 更新実測では、公式 2.901 秒、独立 fused 3.136 秒/更新、peak allocated 26.36 / 26.70 GiB だった。現在の Trainer は端数を落とすので 10,700 件・global batch 8 では 1 epoch 1,337 更新、学習部分だけの単純外挿は約 1.08 / 1.16 時間である。既報の 1,338 更新は端数を含めた外挿で、現在の Trainer の処理とは異なる。データ供給、872 件の validation、保存、起動、復旧を含まず、実行時間の見積もりとしては不足している。保存時間・容量、CPU RAM、実データでの validation 時間と GPU reserved memory の測定も必要。

候補は両実装で同じ初期値、順序、crop、global batch 8、AdamW 2e-4・betas (0.8, 0.99)・weight decay 0.01、epoch ごとの ExponentialLR 0.99。比較前に、同じ端数処理、validation の固定集合と間隔、checkpoint 間隔、SI-SDR の best 選択、公式側の Trainer 接続と再開を検証する。検証セットの全 872 件で指標が安定し、noisy より改善することと、事前に固定する許容差での両者の品質同等性を別々に判定する。許容差と停止規則は未決定。test 824 件はモデル選択後に固定条件で評価し、その後に固定 ASR の WER を調べる。上記の未完成項目があるため、長時間学習用コマンドや時間予算を実行可能なものとして提示しない。

## 5 モデル再検証の着手記録（2026-11-01、未完了）

`dev` の `7d434c6169c2dee67d28201d73dd8b26bc5fe0a3` で開始した。開始時の作業ツリーはクリーン。共有 GPU 4 枚は全て別プロセスが使用率 100% で使用していたため、今回の GPU 比較は行っていない。既存の公式比較スクリプトには既存結果ディレクトリへ固定出力するものがあり、そのまま実行していない。過去の比較値は今回の結果ではない。

TF-GridNet ms-snsd について、既存比較の事前許容（forward / loss 1e-5、勾配 1e-4 および大きい勾配の cosine 0.999）を使い、補助診断として CPU 上の同一プロセスで公式と独立の 1 更新を比較した。seed 3 の 2,048 サンプル合成波形、batch 1 であり、実音声や別プロセス 4 系統の比較ではない。公式重みを独立実装に strict load した。Forward 最大差 0、損失差 0、勾配の要素最大差 0.001953125、層単位の最小 cosine -0.17649、更新後の重み最大差 0.00012050。勾配差は事前許容を超え、原因と最初に差の生じる演算は未特定。層単位 cosine の最小値を全体の cosine と解釈しない。記録は `data/results/training/five-model-audit-20261101-WV8BjA/tf-gridnet-cpu.json`。次のコマンドは新規ファイルを指定して再実行する（既存ファイルは上書きしない）。

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venvs/lab/bin/python tests/training/reference/probe_tf_gridnet_cpu.py data/results/training/NEW_DIRECTORY/tf-gridnet-cpu.json
```

この補助診断だけではモデルの GO 判定はできない。5 モデルの公式 A/B・独立 A/B、実入力の一致、GPU 上の 1 step・短時間学習、checkpoint 再開は今回未検証。MossFormerGAN の現行 Discriminator は公開版と構造が異なり、公式 GAN 学習の再現は未達である。

今回の lab 環境回帰テストは data 12/12、training 76/76、evaluation 86/86、SE 27/28（CUDA extension 専用 1 skip）、ASR 5/5 が成功した。training の最初のサンドボックス内実行はローカルソケットの `PermissionError` により 8 エラーとなり、制限外で再実行して成功した。これらは上記 4 系統の公式比較の代わりではない。

### TF-GridNet CPU backward 追加診断

同一 seed・入力・初期重みで上記の試験を再実施した。保存先は `data/results/training/tf-gridnet-gradient-20261101-KguXGL/`（合成音声、同一プロセスの補助診断）。修正前の全勾配 norm は公式 74192.1856、独立 74192.1771、相対 L2 差 2.36e-7、cosine 0.999999999999976、符号不一致 1696 要素。最大差 0.001953125 は `blocks.0.attn_conv_Q_3.0.weight[0,0,0,0]`（公式 7643.586914、独立 7643.584961）。最小 cosine -0.17649 は `blocks.0.attn_conv_K_2.2.beta`、norm は公式 5.63e-5、独立 3.64e-5。公式勾配絶対値 1e-4 以上の要素では符号不一致はなかった。

同一の公式損失を両モデルに適用するとモデル勾配差は 0。一方、同一の出力波形に両損失を適用すると波形勾配に最大 1.53e-5 の差があった。操作を段階的に比較すると、損失の `estimate * scale` を時間損失と全 STFT 損失で一つの計算ノードとして共有することが差を生んだ。公式は各分岐で乗算を別々に計算するため、backward の浮動小数点加算順が異なる。公式と同じ分岐に修正するとこの条件の波形、損失、全パラメータ勾配と Adam 更新後重みの最大差は 0。事前許容を変更していない。CPU の単一入力の結果であり、GPU や別プロセスでの一般的な一致を意味しない。`probe_tf_gridnet_cpu.py` の `--legacy` は修正前の計算グラフを再現する。

## MossFormerGAN Discriminator と GAN 1 step（2026-11-01）

参照は ClearerVoice-Studio commit `6b3774dc79c46ae8bed2a4fa5f706f0ac8c75c61` の `train/speech_enhancement/models/mossformer_gan/`、`losses/loss.py`、`solver.py`、`train.py`。独立側の実行時は参照ツリーを import しない。Apache-2.0 の公開コードに含まれる AdaptCNN / PoolAttFF の出典 NISQA は MIT、Conformer の出典 lucidrains/conformer も MIT。著作権・ライセンスと改変箇所は `src/training/se/mossformer_discriminator/NOTICE.md` と同ディレクトリの LICENSE に記載した。

| 公式の実行経路 | 学習側の対応 |
|---|---|
| 圧縮済み clean / 生成 magnitude `[B,1,F,T]` を 15 frame、hop 4 で切り、2 channel に連結 | `segment_specs`、`Discriminator.forward` |
| 6 convolution・BatchNorm・adaptive max pool・Dropout2d、384 次元 | `AdaptCNN` |
| 384 次元、4 head、kernel 31 の Conformer 2 層、相対位置埋め込み | `ConformerBlock` ×2 |
| feed-forward attention pooling、`sigmoid(slope * x)`、出力 (0,1) | `PoolAttFF`、`LearnableSigmoid` |
| G 用 fake、D 用 fake、D 用 real の順に forward。G backward → G clip → D backward → D clip → G/D step | `MossFormerGANTask.train_update`（clip は各 microbatch） |
| AdamW の既定 betas (0.9, 0.999)、weight decay 0.01、G 5e-4、D 1e-3 | サンプル設定。betas と学習率は設定可能 |

公開の `ndf` / `in_channel` は実際の層幅に使われない。Discriminator のパラメータは 8,121,331 個、state dict は 117 キーで、公式から strict load でき、shape と値は一致した。入力は波形ではなく、STFT・圧縮後の magnitude pair である。入力 STFT は Hamming periodic=False・center=True・圧縮 0.3。公式の時間損失は clean 実部と生成虚部を組み合わせ、center=False の iSTFT を使う。

CPU float32・単一スレッド、seed 77、合成入力で比較した。単体 Discriminator は batch 1、201 bin、25 frame、training mode、同じ RNG と初期 state。CNN、Conformer、pool、sigmoid の各出力は最大差 0。敵対損失 `(score - 1)^2` の全パラメータ勾配の norm は両者 17.0850190、入力勾配 norm 1.1830269、生成側入力勾配 norm 1.0604418。いずれも最大絶対差・相対 L2 差 0、cosine は丸めて 1。AdamW 1 更新後も最大絶対差・相対 L2 差 0。

学習タスク比較は公式幅 Generator と Discriminator、batch 1・8,000 samples の同じ合成 clean/noisy、accumulation 1。Generator は両側で同じ独立モデルと初期 state を使い、片側は公式の前処理と公式損失関数、もう片側は実際の `MossFormerGANTask.train_update` を呼んだ。公式 `batch_pesq` は CUDA を固定するため、公式 `pesq_loss` と同じ正規化を CPU 上で呼ぶアダプタに差し替えた。PESQ 自体は両側で実計算した。G/D 損失、Generator の 2 出力、両モデルの clip 後勾配、AdamW 更新後の両モデルの重みはすべて最大絶対差・相対 L2 差 0。G/D 勾配 norm はそれぞれ 6.1821041 / 4.2400952、重み cosine はそれぞれ 0.999999999999949 / 0.999999999999898。これは単一プロセス・単一入力・1 更新の結果であり、GPU、別プロセス、公式設定の accumulation 2、実音声、長時間学習の一致を意味しない。

修正前は時間波形再合成で `angle` を二度呼んでいた。公式のように位相を共有すると、同じ出力でも backward の加算順が変わる。2,000 samples の診断では修正前の Generator 勾配最大差 2.05e-7、更新後の重み最大差 7.65e-4 だったが、共有後は両方 0。独立した再合成勾配テストも追加した。

再実行には上記 commit の公式ツリーが必要。比較用環境には公式 import の `einops`、`joblib`、`librosa`、`pesq` が必要で、今回 `.venvs/lab` に不足分を導入した。比較スクリプトは結果を標準出力に出し、既存の実験結果を上書きしない。

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venvs/lab/bin/python tests/training/reference/compare_mossformer_cpu.py
```

回帰テストは `.venvs/lab`、上記と同じスレッド設定で training 79/79、SE 27/28（CUDA extension 専用 1 skip）、evaluation 86/86、ASR 5/5。失敗・エラー 0。MossFormerGAN の公式幅 1 更新、有限勾配、Generator の推論用 export/load も training テストに含む。

## 4 独立プロセス比較（2026-11-01）

`tests/training/four_process/compare.py` は、公式 A/B・独立 A/B を順番に別 OS プロセスで起動する。各プロセスはモデル、Adam、StepLR、乱数と出力を独立に持ち、終了コード、timeout、標準出力・標準エラーを保存する。出力先は新規ディレクトリに限る。`report` はプロセス状態と保存ファイルを読み直すので、中断・欠落・不正な結果も確認できる。`valid` は実行と条件照合の成否であり、実モデルの数値差の許容判定ではない。数値差は別に確認する。

準備時に初期 state dict と固定の合成 clean/noisy tensor を一度保存する。全プロセスが strict load し、ロード後の state、Optimizer、Scheduler、RNG と波形の内容ハッシュを照合する。合成波形は DB のレシピではないので case_id と crop は NA、順序は `[0]`。16 kHz、batch 1、accumulation 1、float32、AMP/TF32 無効。Discriminator は対象外。各 step に forward、clip 前勾配、更新後重みの snapshot、損失、勾配 norm、学習率、状態ハッシュを保存する。最終 Optimizer、Scheduler、RNG の状態も保存する。報告は最大絶対差、相対 L2、cosine、損失差、勾配 norm 差を出す。cosine は norm の積が 1e-24 以下なら null とし、基準勾配絶対値で分けた集計も出す。

CPU の TF-GridNet ms-snsd は既存 `compare_tf_gridnet.py` の ESPnet モデル・公式損失・独立モデル・独立損失を利用した。seed 3、2048 samples、Adam 0.001、clip 5、3 更新、StepLR を毎更新する短い合成入力の試験であり、公式レシピの epoch scheduler や実音声の検証ではない。保存先 `data/results/training/four-process-compare-20261101-cpu/`。PID は公式 A/B 3176526/3176600、独立 A/B 3176651/3176705。4 比較の全 3 step で forward、loss、勾配、更新後重みの最大絶対差は全て 0。これはその環境・入力での結果である。

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venvs/lab/bin/python tests/training/four_process/compare.py run data/results/training/NEW-FOUR-PROCESS --model tf_gridnet --steps 3
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venvs/lab/bin/python tests/training/four_process/compare.py report data/results/training/NEW-FOUR-PROCESS
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venvs/lab/bin/python -m unittest tests.training.test_four_process -v
```

2026-11-01 時点の実モデル接続は TF-GridNet のみだった。スナップショットは全 step 保存するため、長い実行には向かない。保存した PyTorch ファイルは信頼できるローカル結果だけを読み込む。

GPU 1 step は空き確認後 Quadro RTX 5000 の物理 GPU 1（実行時の `cuda:0`）を使用し、`CUBLAS_WORKSPACE_CONFIG=:4096:8` とした。`data/results/training/four-process-compare-20261101-gpu/` に保存。公式 A/B と独立 A/B はそれぞれ forward・loss・勾配・重みの最大差 0。公式/独立は forward 最大差 0、損失差 0.001953125、勾配最大差 0.00146484375、勾配相対 L2 2.7537e-7、更新後重み最大差 9.1635e-5。損失と勾配の絶対差は既存の許容値 1e-5 と 1e-4 を超え、当時は原因未調査だった。このディレクトリは上書きしていない。原因と再測定は次節。

Mock の 4 プロセス比較は CPU の決定的な線形モデルで、順序を逆に変えても比較値が一致した。入力・重み・Optimizer・勾配の改変、例外、結果欠落、非有限損失をそれぞれ検出した。保存内容の破損も検出する。別の Mock checkpoint 試験では 4 更新の連続実行と 2 更新後に別プロセスで再開した実行のモデル・Optimizer・Scheduler・RNG・データ位置が一致した。

モデル固有処理を `adapters.py` に分けた後、CPU 1 step を `data/results/training/four-process-compare-20261101-cpu-adapter/` で再実行した。4 組の loss・勾配・更新後重み最大差は全て 0。

公式 TF-GridNet の読み込みには既存の `reference_v1()` を使い、ESPnet の encoder/decoder を比較用の PyTorch STFT/iSTFT ラッパーに置き換えている。したがって ESPnet 全体のデータ入力や学習ランチャーの再現ではない。

GPU 試験の再実行例（直前に `nvidia-smi` で空きメモリを確認し、空いている物理 GPU 番号を指定する）:

```bash
CUDA_VISIBLE_DEVICES=FREE_GPU CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venvs/lab/bin/python tests/training/four_process/compare.py run data/results/training/NEW-GPU-FOUR-PROCESS --model tf_gridnet --steps 1 --device cuda:0
```

## 4 独立プロセス比較の再開・GAN・実レシピ（2026-11-02）

比較 worker は `checkpoint.pt` を step ごとに原子的保存する。中身は Generator、Discriminator（無いモデルは持たない）、各 Optimizer、各 Scheduler、global step、epoch、データ位置、Python `random`、NumPy、PyTorch CPU、CUDA を使ったときだけ CUDA RNG、学習設定、実行名、入力 SHA-256 である。`resume` は同じ設定と初期重みの実行だけを継続する。seed、入力ファイル、初期重み、fault が違う checkpoint は拒否する。Mock の連続 4 step と、2 step 後に別プロセスで再開した 4 step は、損失、重み、Optimizer、Scheduler、3 種類の RNG、データ位置、global step が一致した。RNG を別の妥当な状態に差し替えると Mock の数値不一致として検出する。TF-GridNet ms-snsd の CPU 2 step でも、連続実行と 1 step 後の別プロセス再開は損失・重み・RNG・Scheduler が一致した。保存先は `data/results/training/four-process-tfgridnet-20261102-cpu-cont/` と `four-process-tfgridnet-20261102-cpu-resume/`。

MP-SENet は縮小構成（`dense_channel=16`、TS-Conformer 1、8,000 samples）で接続した。更新順は公式と同じで、Discriminator（PESQ 目標、detach した magnitude）のあと Generator である。Optimizer は AdamW が 2 つ。この probe は ExponentialLR を毎 step 進める。本学習は epoch ごとであり、その間隔はこの短い比較では再現していない。CPU 1 step の保存先は `data/results/training/four-process-mpsenet-20261102-cpu/`。公式 A/B と独立 A/B は forward、Generator 損失、Discriminator 損失、勾配、重みの最大差 0。公式/独立は forward と両損失の差 0、勾配最大差 3.87e-7、更新後の Discriminator 重み最大差 0、Generator 重み最大差 7.24e-4（近零勾配の bias）。同じ初期重みから 2 step 進めた `four-process-mpsenet-20261102-cpu-cont/` でも公式どうし・独立どうしは全項目の最大差 0 である。公式/独立は 2 step 目の Generator 損失差 0、Discriminator 損失差 8.61e-9、勾配最大差 2.44e-4、重み最大差 1.26e-3 で、1 step 目の Generator 重み差が次の更新に入っている。連続 2 step と 1 step 後の再開は `four-process-mpsenet-20261102-cpu-resume/` で、同じ実装の損失・両重み・RNG・Scheduler が一致した。

`voicebank_28_train` を `data/db/voicebank.duckdb` から読み、本番 `resource.duckdb` は開いていない。48 kHz から 16 kHz への変換は `RecipeDataset` のロード時リサンプリングである。TF-GridNet の 1 step、segment 2048 は `data/results/training/four-process-voicebank-20261102-cpu/`。4 プロセスは同じ case `03dbca80e082f9b44aa2c7fa693f33d3f6d262c454c6744f3f7416bff963f8da`、crop 24853、resample 後の長さ 50666、clean/noisy の波形ハッシュが一致し、forward・損失・勾配・重みの最大差は 0 だった。

TF-GridNet の GPU 差は、強化波形が一致したあとの 256 点矩形 STFT にあった。PyTorch 2.6 の CUDA では `torch.stft(window=None)` と `torch.ones` が 256 点だけ最大約 1e-5 違い、損失和が float32 の 1 ulp である 0.001953125 動いた。TF32 は無効、cuDNN deterministic は有効、AMP は無効だった。独立損失を ESPnet と同じ `window=None`（pad は従来の明示 reflect）にすると、同じ GPU 上の 1 step で公式/独立の forward・損失・勾配・更新後重みの最大差は 0 になった。保存先は `data/results/training/four-process-compare-20261102-gpu/`（物理 GPU 0、実行時 `cuda:0`、`CUBLAS_WORKSPACE_CONFIG=:4096:8`）。2026-11-01 の測定ファイルは残してある。

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venvs/lab/bin/python tests/training/four_process/compare.py run data/results/training/NEW --model mp_senet --steps 1
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venvs/lab/bin/python tests/training/four_process/compare.py resume data/results/training/EXISTING --steps 4
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venvs/lab/bin/python tests/training/four_process/compare.py run data/results/training/NEW-VB --model tf_gridnet --steps 1 --dataset voicebank_28_train --database data/db/voicebank.duckdb --segment-size 2048 --sample-rate 16000
```

## 5 モデルの 4 プロセス比較（2026-11-02）

SEMamba++、MossFormerGAN、DeepFilterNet3 を `tests/training/four_process/adapters.py` から起動できるようにした。公式プロセスは公式モジュールと公式損失、独立プロセスは `src/training` を使う。初期重みは独立モデルで一度作り、両方へ strict load する。TF-GridNet と MP-SENet のアダプタは変えていない。本番 `resource.duckdb` は開いていない。

実行は `turing`。GPU は Quadro RTX 5000（16 GB）。使用前の使用メモリは 10–18 MiB。SEMamba++ の GPU 実行中に見えた使用量は 277 MiB で、ステップ中のピークは別途測っていない。PyTorch 2.6.0+cu124、CUDA 12.4。worker は TF32 を切り、cuDNN deterministic を有効にし、AMP は使わない。`CUBLAS_WORKSPACE_CONFIG=:4096:8`。

| モデル | CPU 1 step | GPU 1 step | 公式/独立 | 同一実装の再実行 | checkpoint 再開 | 実データ | 20 step | 判定 |
|---|---|---|---|---|---|---|---|---|
| TF-GridNet | PASS（既存、最大差 0） | PASS（既存、最大差 0） | PASS | PASS | PASS（CPU 2 step） | PASS（VoiceBank、segment 2048） | NOT TESTED | GO |
| MP-SENet | PASS（既存、縮小幅） | NOT TESTED | 損失 0、勾配最大差 3.87e-7 | PASS | PASS（CPU 2 step） | NOT TESTED | NOT TESTED | GO（CPU の縮小 probe） |
| SEMamba++ | 公式は NOT TESTED。独立 A/B の損失は一致 | PASS（fused、下記） | 1 step の損失差 4.39e-4 | GPU 損失 0、重み最大差 3.22e-4 | 再開は成功。step 4 の損失差 3.81e-6 | PASS（VoiceBank、16 kHz） | NOT TESTED（4 step まで） | PARTIAL |
| MossFormerGAN | PASS（最大差 0） | 損失差 2.38e-7、勾配最大差 4.43e-4 | CPU は 0。GPU は再実行より勾配差が大きい | CPU は 0。GPU 重み最大差 8.90e-4 | PASS（CPU 4 step、重みと RNG が一致） | PASS（VoiceBank、最大差 0） | NOT TESTED（4 step まで） | GO（CPU）。GPU は再実行差つき |
| DeepFilterNet3 | 損失・forward 0、勾配最大差 2.86e-6 | 損失・forward 0、勾配最大差 3.11e-5 | 1 step は小さい。20 step で損失差 0.195 | CPU 20 step は 0。GPU 1 step の勾配最大差 2.86e-6 | PASS（CPU 4 step、完全一致） | PASS（48 kHz、損失 0、勾配最大差 5.67e-5） | 同一実装は PASS。公式/独立は PARTIAL | PARTIAL |

SEMamba++ の probe は hid_feature 8、TF-Mamba 1、d_state 4、d_conv 2、長さ 8,000、16 kHz である。公開幅は hid_feature 48、4 block、d_state 16、d_conv 4 である。Discriminator は公開の CQT と 3 解像度 STFT のまま省いていない。損失係数と、Discriminator のあと Generator という更新順も公開どおりである。ExponentialLR はこの probe では毎 step 進める。本学習は epoch ごとである。公式コードの `mamba_ssm` Block は Triton RMSNorm を呼び、CPU tensor で `ValueError: Pointer argument cannot be accessed from Triton` になる。CPU の公式 4 プロセスは実行できない。独立どうしの CPU reference 1 step は損失 46.39616012573242 で一致した。保存先 `data/results/training/four-process-semambapp-20261102-cpu/` は公式プロセスが失敗しているので `valid` は false である。

GPU は物理 GPU 0、実行時 `cuda:0`、`.venvs/semamba`、`scan_backend: fused`。1 step の保存先は `four-process-semambapp-20261102-gpu/`。公式 A/B は forward 最大差 0、損失差 0、勾配最大差 3.80e-7、更新後重み最大差 3.22e-4。独立 A/B は勾配最大差 2.61e-7、重み最大差 2.85e-4。公式/独立は forward 最大差 1.49e-7、Generator 損失差 4.39e-4、Discriminator 損失差 4.77e-7、勾配最大差 1.04e-4、相対 L2 6.32e-5、cosine 0.999999998、重み最大差 3.96e-4。同じ初期重みで CPU reference と GPU fused の独立実装を 1 step 比べると、forward 最大差 0.0728、相対 L2 0.178、cosine 0.984、勾配最大差 0.151、相対 L2 0.0690、cosine 0.9976、損失差 0.0111、重み最大差 4.00e-4 だった。fused どうしの再実行より、reference と fused の波形差の方が大きい。

4 step の連続実行は `four-process-semambapp-20261102-gpu-cont/`、2 step 後の別プロセス再開は `four-process-semambapp-20261102-gpu-split/`。再開後の global step、データ位置、Scheduler、RNG ハッシュは連続実行と一致した。step 4 の公式損失差は 3.81e-6、独立損失差は 7.63e-6 で、重みハッシュは一致しなかった。checkpoint の読み込みは成功し、fused カーネルの数値は再現していない。4 step 終了時の公式/独立の勾配最大差は 0.0230、独立どうしの再実行は 0.00740 だった。公開幅の 20 step は、1 step が約 30 秒で勾配スナップショットが 1 プロセスあたり約 95 MB なので、20 step × 4 プロセスで約 40 分、約 8 GB になる。今回は行っていない。

VoiceBank は `voicebank_28_train`、16 kHz、segment 8,000。ファイルは 48 kHz で、リサンプリングは `RecipeDataset` のロード時である。case `03dbca80e082f9b44aa2c7fa693f33d3f6d262c454c6744f3f7416bff963f8da`、crop 14288、長さ 50666。4 プロセスの case と波形ハッシュは一致した。保存先 `four-process-semambapp-20261102-voicebank/`。公式/独立の forward 最大差 2.31e-7、損失差 1.91e-6、勾配最大差 1.82e-5。`.venvs/semamba` の DuckDB 1.4.3 は `voicebank.duckdb` を開けない。準備は `.venvs/lab`、worker は `FOUR_PROCESS_PYTHON=.venvs/semamba/bin/python` とした。worker は準備済み波形を使い、DB を開き直さない。

MossFormerGAN は公開幅 `num_channel=64`、8,000 samples、16 kHz。公式側は公式 `SyncANet` と公式 `Discriminator`、公式 `loss_mossformergan_se_16k` である。公式 `batch_pesq` は CUDA tensor を固定で返すため、同じ PESQ 正規化の結果を step の device へ返すように差し替えた。PESQ が取れないバッチは止める。更新は Generator backward、clip 10、Discriminator backward、clip 10、そのあと両方 step。Discriminator の学習率は 0.001、Generator は 0.0005。AdamW betas `(0.9, 0.999)`、weight decay 0.01。公開ループに per-step scheduler は無い。probe は学習率を変えない counter を checkpoint に入れる。CPU 1 step と 4 step は 4 比較とも forward、両損失、勾配、重みの最大差 0。保存先は `four-process-mossformergan-20261102-cpu/` と `four-process-mossformergan-20261102-cpu-cont/`。連続 4 step と、2 step 後に再開した 4 step（`four-process-mossformergan-20261102-cpu-split/`）は損失、重み、Optimizer、Scheduler、RNG が一致した。

GPU は物理 GPU 1。保存先 `four-process-mossformergan-20261102-gpu/`。公式どうしは損失差 0、forward 0、勾配最大差 2.68e-7、重み最大差 8.90e-4。公式/独立は損失差 2.38e-7、Discriminator 損失差 4.47e-7、forward 最大差 9.42e-6、勾配最大差 4.43e-4、相対 L2 8.17e-4、cosine 0.999999666、重み最大差 0.00200。勾配の公式/独立差は再実行差より大きい。VoiceBank は同じ case、crop 14288、48 kHz から 16 kHz。CPU の 4 比較は最大差 0。保存先 `four-process-mossformergan-20261102-voicebank/`。20 step は 4 step の結果が 1 プロセスあたり 477 MB なので、20 step × 4 で約 10 GB になる。今回は 4 step で止めた。

DeepFilterNet3 は出荷幅、48 kHz、長さ 4,800。公式 `DfNet` と公式 `MultiResSpecLoss`、`LocalSnrLoss` を使う。解析 STFT と ERB は両側とも独立 `DFStream` で、step ごとに reset する。`libdf` は実行していない。AdamW は上記の出荷設定。cosine は step の前に学習率と weight decay を書き、step のあとで index を進める。CPU 1 step（`four-process-deepfilternet3-20261102-cpu/`）は公式どうしも独立どうしも最大差 0。公式/独立は損失差 0、forward 0、勾配最大差 2.86e-6、相対 L2 2.78e-7、cosine は 1 に丸まる、重み最大差 5.13e-6。連続 4 step と再開（`four-process-deepfilternet3-20261102-cpu-cont/` と `cpu-split/`）は損失、重み、Optimizer、Scheduler、RNG が一致した。

CPU 20 step は `four-process-deepfilternet3-20261102-cpu-20/`。公式どうしと独立どうしは 20 step すべて損失・勾配・重みの最大差 0。公式/独立は step 1 の損失差 0、step 12 の勾配相対 L2 1.01、cosine 0.784、step 16 の損失差 0.175、勾配最大差 20.5、cosine 0.242、step 20 の損失差 0.195、forward 最大差 0.00311、重み最大差 0.00857。同じ実装の再実行は 0 なので、この増加は GPU の非決定性ではなく、公式損失と独立損失の更新が step を重ねて離れた結果である。2026-11-01 の `official-compare-20261101/deepfilternet3.json` でも 20 step の損失差は 0 ではなかった。

GPU は物理 GPU 0。保存先 `four-process-deepfilternet3-20261102-gpu/`。公式どうしは損失・forward 0、勾配最大差 2.86e-6、重み最大差 9.18e-6。公式/独立は損失・forward 0、勾配最大差 3.11e-5、相対 L2 2.64e-6、重み最大差 9.10e-6。VoiceBank は 48 kHz のまま、segment 4,800。case は同じ `03dbca80…`、crop 95991、ファイル長 151996。公式/独立の損失・forward は 0、勾配最大差 5.67e-5。保存先 `four-process-deepfilternet3-20261102-voicebank/`。

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venvs/lab/bin/python tests/training/four_process/compare.py run data/results/training/NEW-MOSS --model mossformergan --steps 1
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venvs/lab/bin/python tests/training/four_process/compare.py run data/results/training/NEW-DF --model deepfilternet3 --steps 1
CUDA_VISIBLE_DEVICES=FREE_GPU CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venvs/semamba/bin/python tests/training/four_process/compare.py run data/results/training/NEW-SEMAMBA --model semambapp --steps 1 --device cuda:0 --scan-backend fused
FOUR_PROCESS_PYTHON=.venvs/semamba/bin/python CUDA_VISIBLE_DEVICES=FREE_GPU CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venvs/lab/bin/python tests/training/four_process/compare.py run data/results/training/NEW-SEMAMBA-VB --model semambapp --steps 1 --device cuda:0 --scan-backend fused --dataset voicebank_28_train --database data/db/voicebank.duckdb --sample-rate 16000 --segment-size 8000
```

## 最終受入（2026-11-02）

総合は **PARTIAL**。学習処理の公式再現について、5 モデルとも更新順・損失式・optimizer の対応は取れている。長時間学習後の音質同等性は対象外で、未判定である。DeepFilterNet3 の 20 step 勾配、SEMamba++ の公開幅、MossFormerGAN の GPU 複数 step は、1 step の一致だけでは GO にしない。

HEAD は `2af91a3dfc0f95c0c18770bf94890b2faf2c3d77`。この節の計測スクリプトと MP-SENet probe の `--scheduler-step` は作業ツリーにあり、未 commit である。許容値はこれまでの forward / 損失 `atol=1e-5`、勾配 `atol=1e-4` かつ大きい勾配の cosine `>= 0.999` のままである。

| モデル | Forward | Backward | Optimizer | 4プロセス | 再開 | 実データ | 総合 |
|---|---|---|---|---|---|---|---|
| MP-SENet | GO | GO（1 step） | GO | GO（縮小幅、epoch scheduler） | GO | GO（1 step） | GO（縮小幅）。公開幅の再測はしていない |
| SEMamba++ | PARTIAL | PARTIAL | GO | PARTIAL（縮小幅、CUDA） | PARTIAL | GO（縮小幅、1 step） | PARTIAL |
| TF-GridNet | GO | GO | GO | GO | GO（CPU 2 step） | GO | GO |
| DeepFilterNet3 | GO（1 step） | PARTIAL（20 step） | GO | GO（同一実装） | GO | GO（1 step） | PARTIAL |
| MossFormerGAN | GO | PARTIAL（GPU） | GO | GO（CPU） | GO（CPU 4 step） | GO（CPU） | PARTIAL |

### MP-SENet

probe に `--scheduler-step epoch` を足した。このモードは ExponentialLR を step 内で進めない。公開 `train.py` は epoch の終わりにだけ `scheduler_g.step()` と `scheduler_d.step()` を呼ぶ。4 step の学習率は公式・独立とも 5e-4 のまま、scheduler ハッシュも一致した。幅は `dense_channel=16`、TS-Conformer 1 で、公開幅の 64 / 4 ではない。

CPU、`four-process-mpsenet-20261102-epoch/`。公式どうし・独立どうしは 4 step すべて損失・forward・勾配・重みの最大差 0。公式/独立は step 1 の両損失差 0、勾配最大差 3.87e-7、重み最大差 7.24e-4。step 4 の Generator 損失差は 5.48e-6（損失 2.799 に対して）、勾配最大差 0.0133、相対 L2 0.0665、cosine 0.9978、重み最大差 1.71e-3。step 1 の重み差は、以前と同じく AdamW の epsilon 1e-8 が微小勾配を O(lr) まで広げる分である。連続 4 step と、2 step 後に別プロセスで再開した 4 step（`four-process-mpsenet-20261102-epoch-split/`）は、公式・独立のどちらも損失、重み、optimizer、scheduler、RNG が一致した。

VoiceBank は `voicebank_28_train`、16 kHz、segment 8,000。case `03dbca80e082f9b44aa2c7fa693f33d3f6d262c454c6744f3f7416bff963f8da`、crop 14288、長さ 50666。保存先 `four-process-mpsenet-20261102-voicebank-epoch/`。公式どうしは最大差 0。公式/独立は両損失差 0、forward 0、勾配最大差 1.04e-7、相対 L2 5.74e-8、cosine は 1 に丸まる、重み最大差 5.58e-4。

### TF-GridNet

2026-11-02 に tensor の物理 GPU 0（A100-PCIE-40GB、開始時 932 MiB、`CUBLAS_WORKSPACE_CONFIG=:4096:8`）で 1 step をやり直した。保存先 `four-process-tfgridnet-20261102-tensor/`。公式どうし、独立どうし、公式/独立のすべてで forward、損失、勾配、更新後重みの最大差は 0。損失の乗算順と、矩形 STFT の `window=None` は、この A100 でも差を戻していない。CPU の VoiceBank 1 step と CPU の再開は前節の記録のままである。

### DeepFilterNet3

20 step の損失差 0.195 は、別の損失式ではなかった。保存済み入力で公式と独立を同じプロセスに並べると、step 1 の波形差は 0、損失はどちらも 46.343246、勾配の最大差 2.86e-6 は最大勾配約 19.7 の float32 で 1 ulp 程度、相対 L2 は 2.78e-7 である。同じ公式モデルの重み 1 要素だけを `nextafter` で 1 ulp 動かした対照も、step 16 で `df_dec.df_convp.1.weight` の勾配 norm が 111 と 30 に割れた。独立実装との差は、その 1 要素の摂動より早く広がる。step 12 で独立側の同じ層の勾配 norm は 44.7、公式は 20.5 である。step 16 の損失は公式 28.939、独立 29.114、1 ulp 対照 28.939。step 20 の既存 4 プロセスでは損失 24.351 と 24.156（差 0.195、約 0.8%）で、重み cosine は 0.999995 のまま、勾配 cosine は step 16 に 0.24 まで落ちている。同一実装の再実行は 20 step ずっと差 0 なので、実行時の非決定性ではない。原因は、bit 一致する forward のあとに残る 1 ulp 前後の backward 差が AdamW で増幅し、急な勾配に片側だけ当たることである。学習式は変えていない。`libdf` の特徴抽出は引き続き未実行で、両側とも独立 `DFStream` である。

診断は `tests/training/reference/diagnose_df_step1.py` と `diagnose_df_horizon.py`。入力は既存の `four-process-deepfilternet3-20261102-cpu-20/input.pt` で、その結果ディレクトリは上書きしていない。

### SEMamba++ と MossFormerGAN

SEMamba++ の公式 CPU 比較は Triton RMSNorm が CPU tensor を拒否するため N/A のままである。CUDA の縮小幅（hid_feature 8、1 block）の 1 step、4 step、再開、VoiceBank は前節の数値のままである。公開幅の fused 学習 step は tensor で動き、コスト節に書いた。公開幅の公式との 20 step 数値比較は今回もしていない。再開後に fused の重みハッシュが連続実行と一致しない点も前節のままで、チェックポイントの読み込み自体は成功している。

MossFormerGAN の公開設定は `accu_grad: 1`、`effec_batch_size: 2`、`batch_size: 1` である。学習タスクはその蓄積（損失を 2 で割り、両 backward のあと両 optimizer を 1 回）を使う。4 プロセス probe は蓄積 1 のままである。CPU の 4 step と再開は一致し、GPU 1 step の公式/独立勾配差は再実行差より大きい、という前節の判定は変えていない。20 step のスナップショットは容量のため今回も取っていない。

### 学習コスト

数値と shannon の条件付き推定は [GPU_COST.md](GPU_COST.md)。tensor の物理 GPU 0 だけが 1024 MiB 未満だった。2 GPU と 4 GPU は測っていない。shannon では実行していない。

### 回帰テスト

2026-11-02、turing、`.venvs/lab`、`OMP_NUM_THREADS=1`。

| スイート | 結果 |
|---|---|
| `tests/data` | 12 件、失敗 0 |
| `tests/se` | 28 件、失敗 0、skip 1（CUDA 拡張） |
| `tests/evaluation` | 86 件、失敗 0 |
| `tests/asr` | 5 件、失敗 0 |
| `tests/training` | 87 件、失敗 0、skip 0。710 秒 |
| うち `tests.training.test_four_process` | 8 件、失敗 0。543 秒 |

過去の training 件数（30、47、61、68、74、76、79）は、その時点のテストメソッド数である。68 と 86 の差は失敗や skip ではなく、その後に 4 プロセス比較やモデル別試験が足された結果、現在は 87 件になっている。evaluation の 86 件は training の件数ではない。skip は失敗ではない。公式ツリーや CUDA 拡張が無い環境の skip を、公式一致の PASS には数えない。

### TSUBAME H100 追加診断（2026-11-02）

`data/results/training/final-h100-20261102-cpu/` に今回の入力、各プロセスの結果とジョブログを保存した。以下は短い合成入力を使った 1 更新の比較であり、実データの学習品質の比較ではない。MossFormerGAN は同じ0.5秒波形を2回使って蓄積し、各 backward 後にクリップした。

- MossFormerGAN: 公開 solver は Hamming 窓を CPU 上で作り GPU に移す。独立側が GPU 上で生成すると、H100 で公式/独立の forward 最大差 `1.29e-5`、勾配相対 L2 `0.00263` が生じた。CPU 生成に合わせた後、forward 最大差 0、勾配相対 L2 `3.95e-7`。公式 A/B は `3.91e-7`、独立 A/B は `4.03e-7`。更新後重みの最大差は公式/独立 `5.69e-4`、公式 A/B `4.77e-4`、独立 A/B `6.51e-4`。CPU の同一合成波形の蓄積2・1更新では forward、損失、勾配、重みが一致した。
- MP-SENet: dense_channel 64、TS-Conformer 4、epoch 単位 scheduler、公式 PESQ 目標。H100 の公式/独立勾配相対 L2 `1.11e-7`、最大 `2.09e-7`。重み最大差 `6.61e-4` に対して公式 A/B `5.75e-4`、独立 A/B `7.23e-4`。probe の入力は公開 segment 長ではない。
- TF-GridNet ms-snsd: H100 で forward、loss、勾配、更新後重みの最大差は全ペア 0。公式 separator と損失を比較し、separator の STFT は既存 probe のスタブである。
- DeepFilterNet3: H100 の forward と loss は一致、公式/独立勾配相対 L2 `1.37e-6`、最大 `1.52e-5`。公式 A/B は相対 L2 `1.62e-7`。公式 libdf 0.5.6 と独立 DFStream を同じ合成入力で比較したところ、ERB 幅は一致し、スペクトル・ERB・DF 特徴の相対 L2 差は順に `1.65e-7`、`1.32e-7`、`2.91e-7`。CPU の20 step対照は公式/独立 loss 差 `0.283`、公式/1 ULP 摂動公式差 `0.133`。1つの摂動対照だけで前者を説明したとは判断しない。
- SEMamba++: CUDA 12.3.2 で `sm_90` を含む `mamba-ssm==2.3.2.post1` と `causal-conv1d==1.7.0` を新規ビルドし、PyTorch 2.6.0+cu124、H100 MIG 3g.47gb で実行した。パッケージの Mamba3 自動 import は Triton 3.2 に存在しない `set_allocator` を要求するため、専用環境でその import のみ外した。Mamba1 の fused CUDA や RMSNorm の実装は変更していない。単体 Mamba は `use_fast_path=True` で forward/backward が動作し、再実行の forward 最大差 0、入力勾配最大差 `2.27e-13`。公式幅（48、4 block）の4プロセス1更新は公式/独立 forward 差 0、勾配最大差 `3.44e-6`、相対 L2 `4.55e-6`、cosine 約1。公式 A/B の勾配相対 L2 は `9.97e-7`。GAN の probe は短い入力と1件バッチであり、公開バッチと epoch 単位 scheduler の完全再現ではない。

`--official-width` で MP-SENet と SEMamba++ の probe 幅を選べる。DeepFilterNet3 の参照設定は `DF_REFERENCE_CONFIG` でも指定でき、存在しない設定を暗黙に既定値で代用しない。CPU テストでは VoiceBank DB がこの環境に無いため該当テスト1件が失敗した。CUDA の MossFormerGAN 窓の回帰テストは H100 で成功した。

追加の再検証: MossFormerGAN は上記と同じ合成波形の蓄積2で CPU 4 step を実行し、公式 A/B、独立 A/B、公式/独立の forward・勾配・更新後重みが全 step で一致した。2 step 後に再開した4プロセスは、それぞれ連続実行と loss、モデル、optimizer、scheduler、RNG が一致した。これは異なる micro-batch による実データ検証ではない。

DeepFilterNet3 の spectral loss は公式に合わせ、複素数の再構成を複素指数で行い、Hann 窓を CPU で生成してから実行デバイスに移すよう変更した。公式 spectral loss と異なる波形に対する損失・勾配の CPU 回帰テストは成功した。同じ20 step診断を再実行すると step 20 の公式/独立 loss 差は `0.212`、公式/1 ULP 摂動公式は `0.133` であり、差の意味は未確定である。

SEMamba++ 公式幅 fused の4 step 連続比較では、step 4 の勾配 cosine は公式 A/B `0.934`、独立 A/B `0.997`、公式 A/独立 A `0.990`、公式 B/独立 B `0.930`。公式 A/独立 A の forward 最大差は `0.0205`、loss 差は `0.0126`、勾配相対 L2 は `0.141`。同一実装にも大きな再実行差があるが、その事実だけを根拠に公式/独立の差を許容とは判定しない。4 step の scheduler は比較用に更新ごとに進めており、公開設定の epoch 単位ではない。

H100 再確認では、DeepFilterNet3 の損失修正後の1 step は公式/独立 forward と loss が一致し、勾配相対 L2 `1.81e-7`、公式 A/B は `1.39e-7`、独立 A/B は `1.61e-7`。同じ出力波形を両 spectral loss に渡した CPU 診断では loss 差 0、波形勾配最大差 `3.05e-5`、相対 L2 `5.32e-8` だった。20 step 差は引き続き未解決。

MossFormerGAN の H100 4 step（同じ合成波形を各更新で蓄積2）は、step 4 の公式 A/独立 A の勾配相対 L2 `0.00921`、cosine `0.999958`、最大差 `6.72e-4`。公式 A/B は `0.00857`、cosine `0.999963`、最大差 `9.09e-4`。最大差は事前許容値を超えたため、再実行差が近いことは原因分析の一部として記録し、実データでの同等性を保証しない。

SEMamba++ の H100 checkpoint 再開は、4プロセスすべてで scheduler と RNG が連続実行と一致した一方、step 4 の loss 差は `0.00029–0.00125`、重み最大差は `0.00074–0.00095` だった。checkpoint のロード成功と、ビット単位の再現は区別する。

その後、上記 DeepFilterNet3 の CPU 20 step 差をさらに切り分けた。既存の決定的 STFT は反射パディングをスライスと連結で表し、公式 `torch.stft(center=True)` と forward は同じだが CPU の backward 加算順が異なった。複素指数と Hann 窓を公式に合わせたうえで、通常経路を公式の STFT に変更した結果、同じ合成入力・初期重みの公式と独立は CPU 20 step の損失と全パラメータ勾配が一致した（`df-final-check.out`）。1 ULP 摂動対照は step 20 で loss が約 `0.133` 異なる。CUDA の厳密な決定的モードでは `reflection_pad1d` backward が使えないため、従来の明示パディングを維持する。この代替経路は backward の加算順が公式と異なり得るため、CPU の一致をそのまま決定的 CUDA 学習の一致とみなさない。

修正後の DeepFilterNet3 は CPU の独立4プロセス20 step（同じ合成波形、共通 DFStream）でも、公式 A/B・独立 A/B・公式/独立の loss、全勾配、更新後重みが全 step で一致した。H100 の非決定的許可経路での1 step は公式/独立 forward と loss の差 0、勾配相対 L2 `1.64e-7`、公式 A/B `2.84e-7`。厳密な決定的 CUDA の代替経路では有限勾配を確認したが、公式との4プロセス20 step 数値一致は確認していない。

SEMamba++ の H100 公式幅20 step は4独立プロセスで同一の初期状態・入力・RNG を使い、途中は loss と勾配 norm、最終 step だけ波形・全勾配・重みを保存した（`semamba20-compact/` は228 MB）。scheduler は比較用に毎 step 更新、入力は8,000サンプルの合成波形である。step 20 の勾配 cosine は公式 A/B `0.467`、独立 A/B `0.613`、公式 A/独立 A `0.034`、公式 B/独立 B `0.813`。loss 差は順に `0.121`、`0.0908`、`0.00256`、`0.0275`。公式自体の非決定性と更新差の増幅は確認できるが、公式/独立の勾配差をそれだけで説明できないため、長期学習挙動の同等性は未確定である。

最終の H100 1 step（`qrsh-h100-df-final/`、厳密な決定的アルゴリズムを強制しない probe）は公式/独立の forward と loss 差 0、勾配相対 L2 `1.64e-7`、公式 A/B `2.84e-7`、独立 A/B `3.00e-7`。決定的 CUDA の明示パディング経路は単体テストで有限な backward を確認した。

SEMamba++ の公式/独立 1 step における最大重み差 `3.78e-4` の箇所は `generator.dense_encoder.dense_block.dense_block.0.0.weight` で、対応勾配は公式 `-3.46e-7`、独立 `1.12e-7` と微小かつ符号が逆だった。AdamW の初回更新は epsilon に対して十分大きい勾配の符号を強く反映し、この局所的な重み差を説明する。ただし、この1箇所の分析は20 step の全勾配差の原因を証明しない。

20 step の勾配を分けると、公式 A/独立 A の cosine は generator `0.0242`、CQT discriminator `-0.307`、multi-resolution discriminator `0.99960`。公式 A/B でもそれぞれ `0.462`、`0.164`、`0.99969` となり、差は主に generator と CQT 側に集中した。

蓄積1で作成された旧 MossFormerGAN probe の checkpoint は、蓄積2のコードでそのまま再開すると別の更新に切り替わるため拒否する。旧結果の読み取りと report は引き続き可能で、新しい probe は別ディレクトリに作る。

GPU 資源の会計上の確保時間（各 H100 MIG 3g.47gb は同時利用なし）は、ジョブ本体の `ru_wallclock` ではなく開始処理も含む `wallclock` を使う。9232560: 74.605秒、9232572: 196.918秒、9232648: 18.098秒、9232699: 46.118秒、9232825: 54.016秒、9232848: 139.252秒、9232870: 241.844秒、9232910: 188.155秒、9232940: 97.705秒、9232964: 112.439秒、9233120: 111.968秒。ここまで合計 `1281.118秒`（21分21.118秒）。9232560 は qsub から qrsh への切り替えで途中停止し、9232825 は Mamba3 の import エラーで失敗した。CPU のビルド・診断は含まない。

再現例（出力先は存在しない新しいディレクトリを指定する）。CUDA の2コマンドは GPU を、CPU のコマンドは CPU を `qrsh` で確保して実行する。拡張の専用環境と参照ソース、上記の Mamba3 import 制約が必要である。

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="$PWD/.venvs/tsubame-h100-causal:$PWD/.venvs/tsubame-h100-build:$PWD/.venvs/tsubame-ref-extra:$PWD"
export DF_REFERENCE_CONFIG="$PWD/data/results/training/final-h100-20261102-cpu/df-config.ini"
.venvs/tsubame/bin/python tests/training/four_process/compare.py run data/results/training/NEW-SEMAMBA --model semambapp --steps 1 --device cuda:0 --scan-backend fused --official-width
.venvs/tsubame/bin/python tests/training/four_process/compare.py run data/results/training/NEW-MOSS --model mossformergan --steps 4 --device cuda:0
.venvs/tsubame/bin/python tests/training/four_process/compare.py run data/results/training/NEW-DF-CPU --model deepfilternet3 --steps 20 --device cpu
```

H100 で discriminator と optimizer を除き、公式・独立 generator の同じ入力に共通の単純なスペクトル損失を与えた同一プロセス内対照では、forward と loss の差は 0、勾配相対 L2 は公式 A/B `3.34e-7`、独立 A/B `3.35e-7`、公式/独立 `3.34e-7` だった。これは generator 単体の対照であり、GAN 更新の同等性を証明するものではない。

固定波形の H100 discriminator 単体対照では、公式 A/B と独立 A/B の勾配差は 0、公式/独立は loss 差 `4.77e-7`、勾配相対 L2 `1.16e-7` だった。この調査で、公式は CQT と multi-resolution の discriminator、adversarial、feature matching の損失をそれぞれ別に合算してから足すのに対し、独立側は連結したリストを一度に合算していたことを確認した。独立学習と4プロセス probe を公式と同じグループ・順序へ変更した。変更前に作った SEMamba++ probe は異なる計算を混ぜないよう再開を拒否する。

グループ別の損失合算順を修正した SEMamba++ の H100 公式幅1 step（`qrsh-h100-semamba-grouped1/`）では、公式/独立の discriminator loss 差は `4.77e-7` から 0 になった。forward と generator loss の差も 0、勾配相対 L2 `4.55e-6`、最大差 `3.60e-6`。公式 A/B の勾配相対 L2 は `9.98e-7` だった。上の20 step はこの合算順修正前の記録である。

追加の H100 MIG ジョブの確保時間（会計 `wallclock`）は、discriminator 対照 `9233186` が103.933秒、合算順修正後の1 step `9233218` が81.316秒、修正後の4 step・再開・20 step `9233238` が312.953秒。前述分と合わせた GPU 確保時間は `1779.320秒`（29分39.320秒）で、最大同時使用は1 MIG、累計上限2時間以内だった。

合算順修正後に公式幅 fused を再検証した。4 step の公式 A/B、独立 A/B、公式 A/独立 A の勾配相対 L2 は `0.0379`、`0.0875`、`0.218`、cosine は `0.99928`、`0.99617`、`0.97593`。step 4 の公式/独立 forward 最大差は `0.01899`、loss 差は `0.01395`。20 step は同じ初期状態・合成入力で4独立プロセスを実行し、途中は統計だけ、最終状態だけを保存した（約230 MB）。勾配 cosine は公式 A/B `0.669`、独立 A/B `0.242`、公式 A/独立 A `0.719`、公式 B/独立 B `0.588`。loss 差はそれぞれ `0.0213`、`0.1328`、`0.1157`、`0.00423`。4 step で公式/独立の差が自己再実行差を上回り、20 step では両側とも大きくばらつくため、融合版の学習同等性は引き続き PARTIAL。

さらに公式 solver は各サブ discriminator の real/fake を交互に呼ぶが、独立側は各系列の全 real の後に全 fake を呼んでいた。独立 discriminator にペア入力を追加し、学習と probe で公式順にした。既存の単一入力は従来どおり使える。呼び出し順修正後の H100 公式幅1 step は forward・loss 差 0、公式/独立の勾配相対 L2 `4.56e-6`、公式 A/B `1.00e-6`、独立 A/B `1.00e-6` で、1 step の generator 勾配差はほぼ変わらなかった。ペア順に対応しない古い probe checkpoint は再開を拒否する。

ペア呼び出し順修正後の H100 1 step（`9233293`）の会計 `wallclock` は `163.542秒`。それまでのジョブとの累計は `1942.862秒`（32分22.862秒）。

ペア順修正後の4 step・再開・20 step（`9233314`）の会計 `wallclock` は `386.305秒`。全 GPU ジョブの最終合計は `2329.167秒`（38分49.167秒）である。

最終の paired 順による H100 の4 step は、公式 A/B・独立 A/B・公式 A/独立 A の勾配相対 L2 が `0.116`・`0.150`・`0.233`。20 step では同順の cosine が `0.101`・`0.746`・`0.639`、公式 B/独立 B は `0.338`。loss 差は順に `0.0551`・`0.1861`・`0.1255`・`0.2564`。入力と初期条件は4プロセスで一致し、途中の統計と最終状態のみ保存した。公式自身の再実行差も大きく、公式/独立の差の意味は未確定である。paired 順でも GO ではない。

最終 paired 順の2 step 後再開は4プロセスとも成功し、scheduler と RNG はそれぞれの連続実行と一致した。loss・モデル・optimizer のハッシュは4プロセスとも一致しなかった。

追加監査: この連続実行と再開実行は、別々に開始した二つの実行である。保存済み `official_a` の各 step を照合すると、入力と RNG ハッシュは一致するが、モデルと optimizer のハッシュは再開前の step 1 から既に異なる。step 1 の forward 差は 0、最大勾配差は `1.07e-6`、更新後の最大重み差は `2.98e-4`。step 1〜3 の loss は記録上同じ、step 4 の loss 差は `0.001194000244140625`。したがってこの比較は再開操作自体が差を導入したかどうかを識別できない。同一の step 2 状態から連続側と別プロセス側を分岐させ、再開直後の状態と最初の forward/backward/update を比較する必要がある。

同日 CPU の固定乱数波形を用いた `tests/training/reference/compare_semamba_disc.py` の再実行では、公式と独立の CQT / multi-resolution discriminator の state-dict キーは一致し、train / eval とも score・各 feature map の最大絶対差は 0 だった。これは固定入力の forward 比較であり、GAN backward や CUDA の複数更新の一致を示さない。

H100 MIG 3g.47gb のジョブ `9236723` で固定入力の CUDA 演算を5回反復した（`data/results/training/followup-20261102-voicebank/semamba-ops.py` と `semamba-ops.json`）。今回の形状では causal_conv1d と Triton RMSNorm は forward と全入力勾配が一致。selective_scan は forward と u、delta、A、D の勾配が一致したが、B と C の勾配にそれぞれ最大 `2.98e-7`、`1.49e-7` の再実行差があった。演算単体の非決定性の証拠だが、GAN の複数更新差や公式/独立の差全体の原因を証明しない。

ジョブ `9236748` では4プロセスの連続実行が保存した同一の step 2 checkpoint から、各実装を別プロセスで分岐させた。合成入力、公式幅 fused、scheduler は毎更新。分岐時のモデル・optimizer・RNG・入力は一致し、再開後 step 3 の forward と loss も4系統全てで一致したが、勾配とその後の更新は一致しない。step 3 勾配最大差は `1.61e-6`〜`1.95e-6`、更新後重み最大差は `1.33e-4`〜`1.78e-4`。step 4 forward 最大差は `9.83e-7`〜`3.99e-6`。RNG と入力は一致した。最初に観測された差は同一状態での backward であり、上記 selective_scan の非決定性と整合するが、その演算がこの GAN 全体の差の唯一の原因とは特定していない。結果は `semamba-cont/run/` と `semamba-fork/run/`。
この連続側の step 4 勾配相対 L2 は公式 A/B `0.288`、独立 A/B `0.0257`、公式 A/独立 A `0.137`、公式 B/独立 B `0.289`。公式/独立の差を単に自己再実行差と同等と認定しない。
追加 seed 17 / 29 の公式幅 fused・4更新（`qrsh` ジョブ `9236800`）では、それぞれ勾配相対 L2 が公式 A/B `0.0516` / `0.0387`、独立 A/B `0.0566` / `0.0245`、公式 A/独立 A `0.0445` / `0.0381`、公式 B/独立 B `0.0437` / `0.0302`。各 seed の初期状態と入力は同じで、seed 間では異なる。自己再実行差と近い例が増えたが、GAN の差の原因と絶対勾配差は未確定である。
固定の合成 real/fake 波形・初期重みを使った CQT discriminator 単体の H100 `qrsh` ジョブ `9236852` では、公式 A/B・独立 A/B・公式/独立のスコア、畳み込み後の全 feature map、discriminator loss、入力勾配、パラメータ勾配、AdamW 更新後の重みの最大差は全て 0。同一プロセス内の対照であり、CQT の中間スペクトルそのものや GAN 全体を比較したものではない。スクリプトと結果は `cqt-control.py` / `cqt-control.json`。
SEMamba の consistency STFT について固定波形と上流勾配を使った `qrsh` ジョブ `9236861` では、公式の native 反射と独立の明示反射の forward は一致、波形勾配の公式/独立相対 L2 差は `3.31e-8`。各経路内の再実行差も約 `3e-8` だった。これは固定入力での局所対照であり、学習全体の同等性を示さない（`semamba-stft.py` / `semamba-stft.json`）。

固定初期状態・合成入力の generator 出力に対する損失成分別勾配（`qrsh` ジョブ `9236869`）では、公式/独立の mel 勾配相対 L2 差 `4.27e-6` が各自己再実行差約 `8e-8` を上回った。独立側の GPU 生成 Hann 窓、独自 Slaney basis、行列積と対数の演算順が公式と異なっていた。学習損失を CPU 生成 Hann、librosa Slaney basis、公式と同じ行列積と対数・加算順に合わせた。CPU の公式損失との回帰テストでは損失と波形勾配が一致。修正後の同条件 `qrsh` ジョブ `9236900` では mel 勾配相対 L2 は公式/独立 `8.38e-8`、公式 A/B `8.17e-8`、独立 A/B `8.56e-8`、mel loss 差 0 となった。Generator 合計損失の出力勾配相対 L2 は `8.34e-7` から `1.31e-7` へ減少した。独立実装は公式コードを import せず、学習時に librosa を用いる。

同じ seed・入力・公式幅 fused・毎更新 scheduler の修正後4プロセス比較（`qrsh` ジョブ `9236902`、`semamba-fixed-cont/run/`）では step 1 の公式/独立パラメータ勾配相対 L2 は修正前 `4.55e-6`、修正後 `4.54e-6` で、mel 項の局所修正は全モデル勾配の差を解消しなかった。step 4 は公式 A/B `0.515`、独立 A/B `0.0627`、公式 A/独立 A `0.491`、公式 B/独立 B `0.300`。CUDA のばらつきが大きく、固定 seed の1回の前後比較だけで差の改善・悪化を判断しない。同じ step 2 checkpoint からの分岐では step 3 の forward・loss・RNG は一致し、モデルと optimizer は再び分岐した。未解明の実装間差が残るので PARTIAL。

さらに multi-resolution discriminator の固定 real/fake 入力・初期重みで generator adversarial 項を切り分けたところ、公式/独立の入力勾配相対 L2 差は `5.77e-7`、自己再実行差は約 `6e-8` だった（`qrsh` ジョブ `9236920`）。独立側の CUDA 上での Hann 窓生成と明示反射を、通常経路では公式の CPU 生成窓と native 反射に合わせ、厳密な決定的 CUDA では明示反射を維持した。修正後の同条件 `qrsh` ジョブ `9236927` は、公式/独立の入力勾配相対 L2 差 `5.30e-8`、公式 A/B `5.36e-8`、独立 A/B `5.05e-8`。スコア、畳み込み feature map、パラメータ勾配の公式/独立差は 0。CPU の公式 spectrogram と入力勾配の回帰テストも成功した。これは固定入力の discriminator 単体対照であり、GAN 全体の一致とは区別する。

両修正後の公式幅 fused・合成入力・毎更新 scheduler の4プロセス再比較（`qrsh` ジョブ `9236930`）では、step 1 の公式/独立パラメータ勾配相対 L2 が修正前 `4.55e-6` から `1.01e-6` へ減り、公式 A/B `9.9e-7`、独立 A/B `1.02e-6` と同程度になった。step 4 は公式 A/B `0.196`、独立 A/B `0.337`、公式 A/独立 A `0.167`、公式 B/独立 B `0.369`。結果は `semamba-final-cont/run/` と `semamba-final-fork/run/`。複数更新のばらつきとその原因は引き続き検討する。
両修正後の損失成分固定入力対照（`qrsh` ジョブ `9237070`）では generator 出力への合計勾配相対 L2 が公式 A/B `9.11e-8`、独立 A/B `8.75e-8`、公式/独立 `1.19e-7`。adversarial は順に `1.22e-7`、`1.09e-7`、`1.31e-7`、mel は約 `8.4e-8`。この局所対照だけでは複数更新全体の一致を証明しない。
同一 step 2 checkpoint からの4プロセス分岐では step 3 の forward・loss・RNG が一致する一方でモデルが分岐し、step 4 forward 最大差は `1.01e-6`〜`1.51e-6`。修正後の追加 seed 17 / 29（scheduler は毎更新）の step 4 勾配相対 L2 は公式 A/B `0.0535` / `0.0299`、独立 A/B `0.0398` / `0.0380`、公式 A/独立 A `0.0533` / `0.0144`、公式 B/独立 B `0.0277` / `0.0383`。同じジョブ `9236951` の VoiceBank 実音声・4更新では step 1 が公式 A/B `1.25e-6`、独立 A/B `1.28e-6`、公式 A/独立 A `1.26e-6`、step 4 は順に `0.0640`、`0.0599`、`0.0599`（公式 B/独立 B `0.0613`）。全入力 case・crop は各比較で一致した。3 seed の短い比較と局所的な CUDA 非決定性の観測は、残差の唯一の原因や長時間学習の同等性を証明しない。

上記 SEMamba probe で `--scheduler-step epoch` が無視され、実際には scheduler が毎更新進んでいたことを発見した。probe を修正し、古い設定の checkpoint は新しい損失・scheduler での再開を拒否する。`qrsh` ジョブ `9237160` の再検証では4更新の学習率が全て `2e-4` と確認できた。公式幅・合成入力で step 4 勾配相対 L2 は公式 A/B `0.0501`、独立 A/B `0.0549`、公式 A/独立 A `0.0461`、公式 B/独立 B `0.0537`。これは epoch 終了前の4更新であり、epoch 終了時の scheduler 更新は別途 CPU 回帰テストの範囲である。
同じ step 2 checkpoint からの分岐では4系統とも step 3 の forward・loss・RNG が一致し、モデルは更新後に分岐した。step 4 forward 最大差は `1.16e-6`〜`1.55e-6`。これは同一状態での backward 以降に差が生じることを示すが、個別の演算への一意な帰属ではない。
同ジョブの20更新は同一の合成入力・初期状態、公式幅、epoch 終了前の固定学習率で4プロセスを実行し、途中は loss と勾配 norm、最終だけ全勾配と重みを保存した。step 20 の勾配 cosine は公式 A/B `0.479`、独立 A/B `0.676`、公式 A/独立 A `0.773`、公式 B/独立 B `0.592`、相対 L2 は順に `1.205`、`0.799`、`0.676`、`0.824`。同一実装にも大きな分岐があるが、実装間差の原因を完全に説明したとはしない。結果は `semamba20-epoch/`。
追加 seed 17 / 29 の同条件20更新（`qrsh` ジョブ `9237460`）でも初期状態と入力は4プロセスで一致した。step 20 勾配 cosine は順に、公式 A/B `0.605` / `0.662`、独立 A/B `0.511` / `0.843`、公式 A/独立 A `0.380` / `0.634`、公式 B/独立 B `0.422` / `0.849`。seed 17 では実装間差が自己再実行差を上回るため、ばらつきだけで実装間差を説明できると認定しない。結果は `semamba20-epoch-seed-17/` と `semamba20-epoch-seed-29/`。
同ジョブの4更新の step 4 勾配相対 L2 は seed 17 / 29 で公式 A/B `0.0567` / `0.0614`、独立 A/B `0.0480` / `0.0640`、公式 A/独立 A `0.0553` / `0.0707`、公式 B/独立 B `0.0471` / `0.0413`。結果は `semamba-epoch-seed-17/` と `semamba-epoch-seed-29/`。
seed 17 の step 20 を層群別に分けると、公式 A/独立 A の勾配 cosine は Generator `0.378`、CQT discriminator `0.133`、multi-resolution discriminator `0.999`。公式 A/B は順に `0.602`、`0.852`、`0.998`。固定入力・固定重みの CQT 対照は一致しているため、この集計だけで CQT 実装の不具合と断定できない。

DeepFilterNet3 の CPU 20 step を新しい出力先 `data/results/training/df-cpu-20-20261102-followup/` で再実行した。合成入力、共通 DFStream、公式損失と独立損失を使う4プロセスの全比較で loss、forward、勾配、更新後の重みの最大差は全 step で 0、report は valid だった。決定的 CUDA 経路の検証ではない。

TF-GridNet は復旧した VoiceBank の1区間（16 kHz、2048 samples）を使う CPU 4プロセス1更新で、公式同士・独立同士・公式/独立の forward、loss、勾配、更新後重みの最大差が 0 だった。結果は `data/results/training/followup-20261102-voicebank/tf-cpu1/`。比較する公式 separator は STFT スタブを使用しており、ESPnet の前処理全体の一致を証明するものではない。

TSUBAME の VoiceBank 学習 DB を公式 28 話者データから構築した。永続 staging は `data/staging/voicebank-current/`、DB は `data/db/voicebank.duckdb`。train 10,700 件・26 話者、valid 872 件・p286/p287 で、test 話者 p232/p257 および train/valid 話者の重複はない。全 11,572 件で transcript があり、元の 48 kHz と clean/noisy の同名・同形状を保持し、case ID はレシピの SHA-256 と一致した。staging から別 DB を再構築し cases / sets / set_items の全行が一致した。RecipeDataset の1件で 48→16 kHz、同長の clean/noisy と seed 1234・epoch 0 の crop 再現を確認した。全発話の音質検査ではない。

元データへの `data/corpora/VoiceBank-DEMAND` リンクを事前に用意し、出力が存在しないときの構築例は次のとおり。`build` は staging または DB が存在すると拒否する。既存の再構築元を消して再実行しない。

```bash
.venvs/tsubame/bin/python -c 'from pathlib import Path; from src.data.voicebank_sets import build; print(build(Path("data/corpora"), Path("data/staging/voicebank-current"), Path("data/db/voicebank.duckdb")))'
```

4プロセス probe の実音声再開では `--prepare-steps 4` のように将来の入力を先に用意できる。例えば `--steps 2 --prepare-steps 4` は、蓄積2の MossFormerGAN なら異なる8件を準備し、2更新で止めてから4更新まで再開できる。
また `--fork-step 2` は連続実行中の2更新後 checkpoint を別名で保持する。別プロセスをその同一状態から再開するための診断用であり、mock ではモデル・optimizer・scheduler・RNG が連続側と一致した。

MossFormerGAN の実音声 0.5 秒・蓄積2 の4更新ジョブ `9236673` は4プロセスとも3更新後、4更新目の PESQ 目標を作れず停止した。入力の最後の2区間は clean/noisy 間でも PESQ が `NoUtterancesError` を返した。欠損を 0 に置き換えず、未完了の結果を `data/results/training/followup-20261102-voicebank/moss-cont4/` に残した。同じ8発話を1秒で切り出すと clean/noisy PESQ は全件得られたため、その別条件で再試行する。

1秒・異なる8 micro-batch の H100 MIG ジョブ `9236698` は連続4更新と2更新後の再開を完了した。step 4 の勾配相対 L2 は公式 A/B `0.004066`、独立 A/B `0.001948`、公式 A/独立 A `0.002806`、公式 B/独立 B `0.003870`。公式 A/独立 A の勾配最大差は `1.67e-4`、重み最大差は `0.00191`。再開側と連続側は全4プロセスで入力と RNG ハッシュが一致したが、モデル・optimizer は step 1 から別の値であるため、再開操作単独の差はこの実験では識別できない。step 4 loss 差は `1.79e-6`〜`5.42e-6`。結果は `moss-1s-cont4/` と `moss-1s-split4/`。絶対勾配差の発生演算は未特定であり、自己再実行差のみを根拠に GO にしない。
同一 step 2 checkpoint から分岐した `qrsh` ジョブ `9237107` では4系統とも step 3 の forward・loss・RNG が一致し、モデルと optimizer はその更新後に分岐した。step 4 forward 最大差は `9.54e-7`〜`1.67e-6`、loss 差は最大 `5.96e-8`。入力は VoiceBank 1秒・蓄積2である。結果は `moss-shared-cont/run/` と `moss-shared-fork/run/`。
別の実音声連続実行 `moss-1s-cont4/` の step 1 で公式 A/独立 A の最大重み差 `0.000315` は `generator.dense_encoder.conv_1.0.bias` の1要素で、対応勾配は `4.31e-9` / `-4.89e-9`。AdamW の epsilon `1e-8` に近い微小勾配の符号差による局所的な更新差だが、step 4 の全勾配差の原因をこれだけで説明しない。

ジョブ `9236728` では DeepFilterNet3 の反射パディング単体を同一の固定入力・上流勾配で4回比較し、native CUDA と明示パディングの forward・入力勾配は全て一致した。この形状の単体結果であり、STFT と損失を通した勾配や決定的 CUDA の複数更新は未検証である。同じジョブの MP-SENet 公式幅・合成入力・epoch 単位 scheduler の4更新では、step 4 の勾配相対 L2 が公式 A/B `0.251`、独立 A/B `0.331`、公式 A/独立 A `0.255`。2更新後の再開も完了したが、連続側とは再開前の step 1 からモデルが違うため、再開操作だけの影響は識別できない。入力と RNG は一致し、step 4 loss 差は `0.000207`〜`0.000588`。結果は `mp-cont4/` と `mp-split4/`。

追加のジョブ `9236783` で固定合成波形に対する DeepFilterNet3 spectral loss を native CUDA と決定的経路で4回ずつ比較した。loss は同じ `1428.7008056640625`、波形勾配の native/決定的の最大差 `2.38e-7`・相対 L2 `5.51e-8`。native の自己再実行最大差は `1.19e-7`、決定的経路の自己再実行差は 0。STFT を通した加算順と native の非決定性を含む比較であり、両者の差を一つの演算だけに帰属させない。

続く `qrsh` ジョブ `9236792` では VoiceBank 48 kHz・4,800 samples の4区間を使い、公式は native CUDA、独立は厳密な決定的 CUDA にして4更新を比較した。両経路は決定性設定が異なる。step 4 勾配相対 L2 は公式 A/B `0.000391`、独立 A/B `0`、公式 A/独立 A `0.00468`、公式 B/独立 B `0.00501`。step 2 の同一 checkpoint から分岐した再開では、独立 A/B とも step 4 までモデル・optimizer・RNG が連続側と完全一致した。公式の native 経路は step 3 からモデル・optimizer が異なり、RNG は一致した。公式/独立の複数更新差は公式の自己再実行差より大きく、原因を一つに特定できていないため PARTIAL を維持する。結果は `df-cont/run/` と `df-fork/run/`。

MP-SENet は `qrsh` ジョブ `9237090` で VoiceBank の異なる1秒区間・公式幅・epoch 単位 scheduler の4更新を比較し、PESQ 目標も全 step で得た。step 4 勾配相対 L2 は公式 A/B `0.00790`、独立 A/B `0.0813`、公式 A/独立 A `0.0409`、公式 B/独立 B `0.0704`。同一 step 2 checkpoint からの別プロセス分岐は step 3 の forward・loss・RNG が一致し、その更新後のモデル・optimizer が分岐した。step 4 forward 最大差は `8.34e-7`〜`1.10e-6`。この probe は公式・独立の Generator と Discriminator を比較するが、両側の GAN 損失と PESQ 目標は共通の独立損失関数を使うため、公式損失コードを別途実行した比較ではない。結果は `mp-voicebank-cont/run/` と `mp-voicebank-fork/run/`。

今回の H100 MIG ジョブの会計 `wallclock` 秒: `9236673` 122.952、`9236698` 295.456、`9236723` 87.022、`9236728` 156.948、`9236748` 170.446、`9236783` 9.225、`9236792` 65.757、`9236800` 281.372、`9236852` 104.707、`9236861` 15.011、`9236869` 52.227、`9236900` 109.938、`9236902` 159.994、`9236920` 81.969、`9236927` 79.765、`9236930` 257.938、`9236951` 347.195、`9237070` 23.806、`9237090` 68.038、`9237107` 138.842、`9237160` 337.109、`9237460` 409.981。今回分 `3375.698秒`、前回分 `2329.167秒` を含む累計 `5704.865秒`（95分4.865秒）。同時に使用したのは最大1 MIG。`9236673` は PESQ の発話不足で失敗、他は終了コード 0。最初の6ジョブは短時間開発にもかかわらず誤って `qsub` を使い、ユーザーの指摘後は `qrsh -p -5` に切り替えた。既存の24時間対話ジョブ `115141` はユーザー指定によりこの枠外として扱うが、そこで CPU テストが自動実行した CUDA 専用テスト1件の使用も記録する。

CPU 回帰は data 16 件、training 102 件（skip 3）、SE 28 件（skip 19）、evaluation 86 件（skip 3）、ASR 5 件で、失敗はない。training は scheduler probe 修正後に全件を再実行した。training の skip は未配置の rebuild DB 1 件と CUDA 専用 2 件。SE / evaluation の skip は配布 checkpoint・参照結果の欠落と明示 CUDA テストである。実際のモデルとの比較結果と単体テストを混同しない。

## seed 17 の Backward 分岐と CUDA 経路（2026-11-03）

総合は引き続き **PARTIAL**。最低目標の4項目は、実装差を許容値の変更で吸収せずに切り分けた。長時間学習の品質同等性は対象外のままである。GPU は既存の対話ジョブ `115141`（H100 MIG 3g.47gb）を使い、新しい `qrsh` は出していない。この対話予約そのものは前回と同じく枠外とし、実験プロセスの wall clock だけを加算する。

### SEMamba++

保存済みの seed 17、公式幅、fused、epoch 単位 scheduler、合成 8,000 サンプル（`semamba-epoch-seed-17`）で、optimizer を進めずに Generator 損失の backward を公式2回と独立1回行った。24回の `selective_scan_cuda.fwd` は、入力も出力も公式再実行・公式/独立のどちらも全要素が一致した。公式は `mamba_simple.py` から、独立は `_fused_mixer` から、同じ `mamba_inner_fn` に入る。

最初に一致しなくなるのは scan の出力勾配ではない。波形に接続する損失項の backward である。公式再実行の波形勾配の最大絶対差は、合計 `3.73e-9`、mel `7.45e-9`、feature matching `1.86e-9`、consistency `9.31e-10`、adversarial `1.46e-11`。公式/独立も最大絶対差は同じ値で、相対 L2 は合計 `8.02e-8` と公式再実行 `3.81e-8` の同じ桁である。magnitude / phase / complex は波形に接続しない。この波形勾配の差は、最初の scan backward の入力 `dout`（相対 L2 約 `7.5e-7`）として届く。

同じ seed 17 の活性化で `selective_scan_cuda.bwd` を4回繰り返すと、入力が同一でも dB と dC の相対 L2 は約 `1.6e-7`、最大絶対差は `1e-11` 未満だった。長さ16と batch 1 でも dB/dC の再実行差は残る。dtype は float32 で、カーネルはこの精度である。

849個の同名パラメータ勾配の相対 L2 の中央値は、公式再実行 `7.90e-7`、公式/独立 `7.90e-7`。最大絶対差はどちらも約 `3.05e-6`。公式再実行では `mask_decoder.mask_conv.1.bias` の勾配ノルムが `2.03e-6` で cosine が `-1` になり、1要素の符号が反転した。AdamW の epsilon `1e-8` より小さい勾配の符号差は、初回更新で O(学習率) の重み差になる。step 1 の公式/独立に、公式再実行を超える実装差は出ていない。

seed 17 の20 step cosine（公式 A/独立 A `0.380`、公式 A/B `0.605`）は、この非決定性の蓄積の1本である。同じ条件の step 4 は公式 A/B `0.0567`、公式 A/独立 A `0.0553` で、実装間が自己再実行を上回っていない。seed 29 の20 step も実装間 cosine `0.634` が公式 A/B `0.662` に並ぶ。20 step をこのセッションで追加してはいない。融合カーネルを決定的な参照 scan に替えると公式の学習経路から外れるため、コードは変えていない。

結果は `data/results/training/seed17-backward-20261103/`、`seed17-backward-20261103-named/`、`seed17-backward-20261103-terms/`。

### DeepFilterNet3

同じ VoiceBank 入力（seed 3、48 kHz、4,800 samples、4更新）で経路を分けた。結果は `data/results/training/df-cuda-paths-20261103/`。

| 経路 | step 4 の勾配相対 L2 |
|---|---|
| 公式 native A/B | 0.00244 |
| 独立 native A/B | 0.000725 |
| 公式 native / 独立 native | 0.00159 と 0.000848 |
| 公式 native A/B（独立だけ決定的にした実行） | 0.000613 |
| 独立決定的 A/B | 0 |
| 公式 native / 独立決定的 | 0.00470 と 0.00529 |

両方を native CUDA（`torch.stft` の反射パディング）にすると、公式/独立は公式自身の再実行差を超えない。独立だけを厳密な決定的 CUDA にすると、明示反射の backward 加算順が公式の `reflection_pad1d` backward と異なり、4 step の相対 L2 は約 `0.005` まで開く。独立決定的どうしは重みまで一致する。

公式モデルに `torch.use_deterministic_algorithms(True)` を掛けた1 step は実行できない。失敗は `reflection_pad1d_backward_out_cuda does not have a deterministic implementation`。CPU の公式・独立20 step 一致は前節のままである。

### MossFormerGAN

実音声1秒・蓄積2の既存4 step（`moss-1s-cont4`）では、step 4 の勾配相対 L2 が公式 A/B `0.00407`、独立 A/B `0.00195`、公式 A/独立 A `0.00281`、公式 B/独立 B `0.00387`。実装間差は公式の再実行差より小さい。絶対値が `1e-8` 未満の勾配要素は、公式再実行でも cosine `0.325` である。

step 1 の最大重み差 `0.000315` は `generator.dense_encoder.conv_1.0.bias` で、勾配は `4.31e-9` と `-4.89e-9`。どちらも AdamW epsilon `1e-8` 未満。同じ初期重みで `torch.optim.AdamW`（lr `5e-4`、betas `(0.9, 0.999)`、weight decay `0.01`）にこの2勾配を別々に与えると更新差は `0.0003148`。同じ勾配を2つの optimizer に与えると更新差は 0。したがって差は optimizer 実装ではなく、epsilon 未満の勾配符号差が初回更新で増幅されたものである。

### MP-SENet

公式幅・VoiceBank 1秒・epoch 単位 scheduler の既存4 step（`mp-voicebank-cont`）では、step 1 の損失差は全ペア 0、勾配相対 L2 は `7.7e-8` から `8.7e-8`、勾配最大差は `6.0e-8` 以下、更新後重みの最大差は `3.4e-4` から `3.7e-4` で、公式再実行と公式/独立が同じである。

2026-10-31 に記録した要素（勾配 `-8.90e-9` と `5.14e-9`、lr `5e-4`、betas `(0.8, 0.99)`、epsilon `1e-8`）を同じ AdamW で再計算すると、更新差は `4.0519e-4` で、当時の実測 `4.0527e-4` との差は `7.5e-8`。閉形式 `lr * grad / (|grad| + eps)` との誤差は 0。同じ勾配を2つの optimizer に与えると更新差は 0。`|grad| < eps` のとき更新はほぼ `lr * sign(grad)` になり、符号差が O(lr) の重み差になる。

公式 `train.py` の損失は関数として分離されていない。probe は公式・独立の両方に独立実装の GAN 損失と PESQ 目標を使う。step 1 の重み差が公式再実行にも同じ大きさで出るので、この増幅は損失関数の取り違えではない。step 4 の勾配相対 L2 は公式 A/B `0.00790` に対し独立 A/B `0.0813`、公式/独立 `0.0409` と `0.0704` で、独立側の再実行の方が大きい。step 1 の説明は完了しているが、4 step 以降の増幅量は公式再実行と同一とはまだ言えない。

計算は `tests/training/reference/diagnose_adamw_epsilon.py`、結果は `data/results/training/adamw-epsilon-20261103/adamw.json`。

### 回帰テスト（2026-11-03、`.venvs/tsubame`）

失敗は無い。ログは `data/results/training/regression-20261103/`。

| スイート | 結果 |
|---|---|
| `tests/data` | 16件、失敗 0、skip 0 |
| `tests/training` | 102件、失敗 0、skip 1（再構築 DB が無い） |
| `tests/se` | 28件、失敗 0、skip 19（配布 checkpoint・参照結果の欠落、明示 CUDA 拡張テスト） |
| `tests/evaluation` | 86件、失敗 0、skip 3（PESQ 境界の保存例と MP-SENet checkpoint がこの環境に無い） |
| `tests/asr` | 5件、失敗 0、skip 0 |

`tests/training` の DeepFilterNet3 CUDA 決定的経路テストは、この MIG 上で実行され成功した。前回 skip だった CUDA 専用テストのうち、この環境で実行できたものが含まれている。スイート時間の大半は CPU なので、下の実験 wall clock には足していない。evaluation のログに出る CHiME3 注釈の skip は、テストが意図的に出す未登録データの記録であり、unittest の skip 3件とは別である。

### このセッションの GPU 時間

実験プロセスの wall clock は `135.428 + 136.045 + 87.600 + 35.526 + 336.158 + 29.475 + 68.309 = 828.541` 秒。失敗した損失項の切り分け `87.600` 秒と、形状を誤った公式決定的 CUDA の試行 `29.475` 秒を含む。前回累計 `5704.865` 秒に加えると `6533.406` 秒（108分53.406秒）。上限300分に対する残りは約191分。同時に使った MIG は1つ。対話ジョブ `115141` の24時間予約自体は加算していない。

### MP-SENet 公式損失経路の追加比較（2026-11-03）

HEAD `298b283`、公式 MP-SENet `89932cfe90d1dacb8e170e4a331d762462c21792`。4プロセス probe の公式側を、公式 `train.py` の損失式と公式 `batch_pesq` を実行するよう変更した。公式コードは checkout から import するため joblib の子プロセスでは `models` を解決できず、初回 CPU 試行 `mp-official-loss-cpu-20261103/` は失敗して中断した。再試行は公式の PESQ 関数と集計を維持し、joblib の backend のみ threading とした。固定テンソルの generator 全損失成分、対応する勾配、PESQ 成功・失敗時 discriminator 損失の単体比較は完全一致した（`tests.training.test_mp_official_loss_path`、1件成功）。

VoiceBank 実音声1秒・epoch scheduler・CPU 1更新 `mp-official-loss-cpu-thread-20261103/` は4系統の条件照合が成立した。step 1 の損失差は全ペア 0、勾配相対 L2 は公式 A/B と独立 A/B が 0、公式/独立が `4.03e-8`。更新後重み最大差は公式/独立 `6.97e-4`。

既存の対話ジョブ `115141`、H100 MIG、公式幅・VoiceBank 1秒・4更新 `mp-official-loss-gpu-20261103/` も4系統の条件照合が成立した。step 1 の損失差は全ペア 0、勾配相対 L2 は `7.71e-8`〜`8.50e-8`。step 4 の勾配相対 L2 は公式 A/B `0.00774`、独立 A/B `0.0702`、公式/独立 `0.0277` と `0.0750`。discriminator 損失差はそれぞれ `3.18e-5`、`3.05e-5`、`1.54e-4`、`1.56e-4` で、実装間差は自己再実行差より大きい。原因は未特定。probe は generator を discriminator と generator の各更新時に再計算し、PESQ 失敗時は中止するため、公式ループ全体や例外動作の実音声検証ではない。GO とはしない。step 2 checkpoint は保存したが再開比較は未実施。

GPU 実験のプロセス経過時間は `53.03` 秒（CPU 側集計・待機を含む保守的な計上）。対話予約時間は除外。累計 `6586.436` 秒、300分上限まで `11413.564` 秒。新規結果は上記ディレクトリを参照。

### TF-GridNet 公式 STFT 経路の追加比較（2026-11-03）

ESPnet commit `0d136c5456d403cb2c0a4c13e76e8136e32e9b7a` の sparse checkout に含まれない encoder・decoder を `git archive` で取り出し、公式 `STFTEncoder`・`STFTDecoder`・`Stft` を用いた。`torch_complex` の旧型と `typeguard` の import shim は計算に使わず、公式の built-in complex 経路を使う。既存の公式 separator・損失比較に加え、音声前処理も公式コードから実行した。`tests/training/reference/probe_tf_gridnet_official_stft.py` で再実行できる。

VoiceBank train case `03dbca80e082f9b44aa2c7fa693f33d3f6d262c454c6744f3f7416bff963f8da`、crop 4041、16 kHz 1秒、ms-snsd 公式幅、Adam と gradient clipping 5 の1更新で、CPU と H100 MIG の双方において forward・loss・362パラメータ勾配・更新後重み・optimizer 状態の最大差は 0。報告は `data/results/training/tf-official-stft-cpu-20261103/result.json` と `gpu-result.json`。この試行は個々の前処理テンソル、複数更新、再開を別途比較しておらず、これだけで全受入条件を GO としない。GPU process wall clock は `6.40` 秒、既存ジョブ `115141`。上記累計に加え `6592.836` 秒、残り `11407.164` 秒。

同じ実音声 CPU 1 step で、両実装の前処理後・最初の convolution 入力を forward hook で取得した追加結果 `preprocessing-result.json` は最大差 0。これにより当該入力の正規化と STFT を含む前処理出力が一致した。

4個の異なる VoiceBank 区間による CPU 連続4更新 `four-step.json` は、全 step の前処理後 convolution 入力、forward、loss、勾配、更新後重みの最大差が 0。checkpoint 再開はこの公式 STFT 経路では未検証である。

公式 STFT を用いた CPU 4更新の step 2 モデル・optimizer・RNG をメモリ上に保存し、新しいモデルと optimizer にロードして step 3〜4 を再実行した `four-step-resume.json` では、公式・独立それぞれ連続実行との最終重みおよび optimizer 状態の最大差は 0。同一プロセス内の再開である。

### 回帰試験の追加結果（2026-11-03）

CUDA を隠した CPU 実行: data 16件成功、training 103件成功（skip 4）、SE 28件成功（skip 19）、evaluation 86件成功（skip 3）、ASR 5件成功。training 実行中に MP-SENet 比較用 adapter を更新したため、このスイートだけでは更新後の全挙動を保証しない。training の skip は再構築 DB と公式 DeepFilterNet 設定の未指定、CUDA 専用2件。SE の skip は配布 checkpoint・参照と CUDA 拡張の未使用。evaluation の skip は保存例・checkpoint の欠落である。

初回 evaluation は `test_same_id_with_different_content_is_rejected` が1件失敗。テストが同じ浮動小数 WAV を再保存すると PEAK chunk の時刻が秒境界で変わり、ソース SHA-256 と case ID が変化するため、ケース衝突より先にセットの不一致となった。1.2秒間隔の再保存で差は byte 60 のみだった。テストを、音声を再保存せず transcript だけ変更するよう修正した。修正後 evaluation は86件成功、skip 3。

### MP-SENet 公式更新順の追加比較（2026-11-03）

公式側は実際の `train.py` のように generator の出力グラフを保持し、discriminator 更新後にその出力を使う。独立側は現行 task の no-grad 再計算経路を用いる。PESQ 失敗時は公式・独立とも fake loss を省く。VoiceBank 1秒・公式幅・epoch scheduler の4プロセス CPU 1 step `mp-official-loop-cpu-20261103/` は条件照合 valid、損失差 0、勾配相対 L2 は公式/独立 `4.03e-8`、自己再実行 0。

同条件の H100 MIG 4 step `mp-official-loop-gpu-20261103/` は条件照合 valid。step 1 損失差 0、勾配相対 L2 は `7.70e-8`〜`8.45e-8`。step 4 勾配相対 L2 は公式 A/B `0.01146`、独立 A/B `0.07490`、公式/独立 `0.07070` / `0.02226`。discriminator loss 差は順に `3.76e-5`、`1.48e-4`、`5.66e-5` / `1.67e-4`。step 4 の fake loss は 公式 A/B `0.53579545` / `0.53583264`、独立 A/B `0.53585148` / `0.53599924`、PESQ 目標は `0.0813806`〜`0.0813924`。この試行でも実装間差のすべてを公式自己再実行のばらつきと判定できない。step 2 の checkpoint は保存済みだが再開比較は未実施。GPU process wall clock `36.90` 秒（集計含む）、ジョブ `115141`。以前の `6592.836` 秒に加算して `6629.736` 秒、残り `11370.264` 秒（SEMamba++ 進行中分は未計上）。

H100 MIG の同じ4更新と同一プロセス内 step 2 再開 `four-step-resume-gpu.json` でも、4 step 全ての前処理・forward・loss・勾配・更新重み差および再開後のモデル・optimizer 差は 0。実行時間 `6.32` 秒、ジョブ `115141`。先の `6629.736` 秒に足して `6636.056` 秒、残り `11363.944` 秒（SEMamba++ 集計中の時間は未計上）。

### SEMamba++ seed 17 の新規20 step（2026-11-03）

HEAD `298b283`、H100 MIG、既存ジョブ `115141`、seed 17、公式幅・fused、epoch scheduler、合成16 kHz 8,000 samples。`semamba-seed17-repeat-20261103/` の4プロセスは条件照合 valid、全20 step の損失・勾配・重みを保存した。step 1 の勾配相対 L2 は全ペア `1.45e-6`〜`1.49e-6`。step 4 は公式 A/B `0.03755`、独立 A/B `0.04070`、公式/独立 `0.03680` / `0.04063`。step 20 の勾配 cosine は順に `0.7216`、`0.7107`、`0.6420` / `0.6779`、相対 L2 は `0.7959`、`0.7403`、`0.7902` / `0.7356`。前回の seed 17 の低い実装間 cosine を無視せず、この新規試行でも同一実装を下回ったことを記録する。

step 20 の generator cosine は同順に `0.7277`、`0.7094`、`0.6395` / `0.6736`。CQT discriminator は `0.1792`、`0.7866`、`0.9409` / `0.8909`、MRD は `0.9981`、`0.9976`、`0.9997` / `0.9938`。全6ペア・各群の norm、最大差、相対 L2、cosine は `group-summary.json`、再集計方法は `tests/training/reference/summarize_semamba_four_process.py`。単一反復では実装間差と自己再実行の分布の同等性は判定できない。step 4 の checkpoint は保存したが、同一 checkpoint からの別実行は未実施。総合は PARTIAL。

4 worker の process wall clock は `82.020 + 59.779 + 59.975 + 59.727 = 261.501` 秒。全実行・CPU 集計を含む経過 `386.40` 秒を保守的に上限へ計上する（この CPU 集計の一部は他の GPU 実験と重複する）。直前の `6636.056` 秒に加算して `7022.456` 秒、残り `10977.544` 秒。別途 MP-SENet 再開の失敗・再試行は未計上。

上記 MP-SENet 公式更新順の step 2 checkpoint をコピーし、各系統を別プロセスで step 4 まで再開した。最初の `mp-official-loop-resume-20261103/` はコピー先のディレクトリ名が異なるため checkpoint の run name 検査に拒否された（4系統失敗、19.49秒）。元と同じ末尾名を新しい親ディレクトリに置いた `mp-official-loop-resume2-20261103/mp-official-loop-gpu-20261103/` は valid、34.60秒。各系統とも step 3 の forward・generator/discriminator loss・RNG が連続側と一致し、その更新後の model・optimizer hash は異なる。step 4 forward 最大差は `7.93e-7`〜`1.08e-6`、loss 差は最大 `1.37e-6`。両時間を上記 `7022.456` 秒に加えて累計 `7076.546` 秒、残り `10923.454` 秒。既存ジョブ `115141`、予約時間自体は除外。

SEMamba++ 公式 A の step 4 checkpoint を4系統に配る最初の試行 `semamba-shared-fork-20261103/` は公式2系統が成功、独立2系統は AdamW の moment と勾配の shape が合わず失敗した。公式と独立の generator は名前849件が一致するが、登録順の240位置が異なる。PyTorch の optimizer state は位置でロードされるため、実装間で optimizer checkpoint をそのまま流用できないことが原因であり、同一実装の通常の checkpoint 再開が失敗するという意味ではない。

`tests/training/four_process/shared_fork.py --remap-semamba` は、optimizer moment をパラメータ名で並べ替えて別実装へコピーする。小さな optimizer state の並べ替えと名前不一致拒否は `test_shared_fork` で確認した。新規 `semamba-shared-remap-20261103/semamba-seed17-repeat-20261103/` は4系統 valid。4系統は同一 step 4 のモデル・optimizer・scheduler・RNG と入力から始め、worker 識別子と独立 optimizer の格納順のみ変更した。step 5 forward と両損失は全ペア 0、勾配相対 L2 は `1.73e-6`〜`1.76e-6`。step 6 の勾配相対 L2 は公式 A/B `1.55e-4`、独立 A/B `1.34e-4`、公式/独立 `3.91e-5` / `1.99e-4`。更新後の重みは step 5 から分岐する。これは共通状態からの2 step の結果で、先の独立20 step の差を解消した証拠ではない。

最初の失敗試行 `57.20` 秒、parameter 登録順の調査 `5.70` 秒、共通 checkpoint の名前対応作成 `9.89` 秒、再試行・集計 `96.82` 秒を保守的に GPU 枠へ加算。累計 `7246.156` 秒、残り `10753.844` 秒。ジョブはすべて既存 `115141`。別の20 step 再反復は進行中。

今回の公式 checkout HEAD: MP-SENet `89932cfe90d1dacb8e170e4a331d762462c21792`、SEMamba++ `cd369c89a8106072c077b468417b765b227afd74`、ESPnet `0d136c5456d403cb2c0a4c13e76e8136e32e9b7a`、DeepFilterNet `d375b2d8309e0935d165700c91da9de862a99c31`、ClearerVoice `6b3774dc79c46ae8bed2a4fa5f706f0ac8c75c61`。独立のモデル・損失コードは HEAD `298b283` から変更していない。

追加比較の実行例（出力先は新規であること）:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venvs/tsubame/bin/python tests/training/reference/probe_tf_gridnet_official_stft.py --input data/results/training/mp-official-loss-gpu-20261103/input.pt --output data/results/training/NEW-tf-stft.json --steps 4
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venvs/tsubame/bin/python tests/training/reference/summarize_semamba_four_process.py data/results/training/semamba-seed17-repeat-20261103 --output data/results/training/NEW-semamba-groups.json
```

TF-GridNet の蓄積補助診断 `accumulation-adam-diagnostic.json` は縮小モデル・合成2区間、公式 Adam lr `0.001`・betas `(0.9, 0.999)`・clip 5、全バッチと micro-batch 2回を比較する。clip 後の勾配 norm は両者約 5、勾配最大差 `1.13e-6`、相対 L2 `9.83e-7`。最大重み差 `0.00086475245` は `blocks.0.attn_conv_K_0.2.beta` で、勾配 `-7.70e-9` / `7.53e-9` と Adam epsilon `1e-8` から同じ Adam による更新差 `0.00086475248` を再現した。key に時刻共通の定数を足すと softmax の出力が不変になる単体 float64 対照では、出力最大差 `3.33e-16`、定数への勾配最大 `1.33e-15` だった。これは当該 beta の微小勾配差の構造的な説明であり、蓄積の実音声・公式幅・複数更新全体を検証したものではない。初回の AdamW・clip 無し試行 `accumulation-diagnostic.json` は公式設定ではないため区別して保存した。再実行は `tests/training/reference/diagnose_tf_accumulation.py`。

同じ蓄積診断を公式幅・VoiceBank 1秒の異なる2区間で実行した `accumulation-voicebank.json` は、clip 後の勾配相対 L2 `1.61e-6`、最大差 `1.80e-6`。最大重み差 `0.00011148387` は `blocks.0.attn_conv_K_1.2.beta` で、対応勾配 `-5.49e-10` / `6.32e-10` から Adam の初回更新差 `0.00011148386` を再現した。CPU の1更新であり、多数更新の蓄積経路は別途必要。

同じ seed 17・同じ入力ハッシュ `8b71d25461fce9e5ef919b56746b4a81b3f7628e7f0835590ec9d6b70b67520f` の20 step を再反復し、2試行・8系統の最終勾配の全28ペアを `summarize_semamba_distribution.py` で集計した（`semamba-seed17-repeat2-20261103/two-trial-distribution.json`）。cosine は公式どうし6ペア min/median/max `0.0753/0.2030/0.7216`、独立どうし6ペア `0.2364/0.3656/0.7107`、実装間16ペア `0.0153/0.4439/0.7312`。実装間分布は自己再実行と大きく重なる一方、最小値は自己再実行より低い。ペアは同じ系統を共有し統計的に独立ではなく、2試行だけで分布一致を証明しない。前回の seed 17 の `0.380/0.422` は除外せず、その別記録を上に残す。追加試行の全 step 比較と GPU 時間は集計終了後に記録する。

4プロセス probe の公式 TF-GridNet も `tests/training/reference/official_tf_loader.py` に切り替え、ESPnet の公式 encoder・decoder を実行するようにした。VoiceBank 1秒の CPU 1更新 `tf-official-stft-four-cpu-20261103/` と、H100 MIG・epoch scheduler（更新中は plateau scheduler を固定）4更新 `tf-official-stft-four-gpu-20261103/` はともに valid。公式 A/B・独立 A/B・公式/独立の全 step で forward・loss・勾配・更新重みの最大差は 0。GPU 全 process wall clock `38.85` 秒、既存ジョブ `115141`。直前の `7246.156` に加え `7285.006` 秒、残り `10714.994` 秒。SEMamba++ 再反復と TF-GridNet 別プロセス再開は未計上。

追加20 step `semamba-seed17-repeat2-20261103/` の4系統 report は valid。step 20 勾配 cosine は公式 A/B `0.5449`、独立 A/B `0.4274`、公式/独立 `0.3536` / `0.6329`、相対 L2 は `0.9035`、`0.9758`、`1.1942` / `0.8519`。全実行・CPU 集計込み process wall clock `373.78` 秒を保守的に加算し、直前の `7285.006` 秒から累計 `7658.786` 秒、残り `10341.214` 秒。CPU 集計時間の一部は TF-GridNet GPU 処理と重複する。既存ジョブ `115141`、予約時間自体は除外。

TF-GridNet の公式 STFT 経路 4プロセス GPU 4更新の step 2 checkpoint を各系統別にコピーし、別プロセスで4更新まで再開した `tf-official-stft-resume-20261103/tf-official-stft-four-gpu-20261103/` は valid。step 3・4 とも4系統で連続側と model・optimizer・scheduler・RNG の hash が一致した。所要 `34.43` 秒を加算し累計 `7693.216` 秒、残り `10306.784` 秒。ジョブは `115141`。

再反復の step 20 component cosine は公式 A/B・独立 A/B・公式 A/独立 A・公式 B/独立 B の順に、generator `0.5412/0.4224/0.3544/0.6285`、CQT discriminator `0.7555/0.7388/0.0464/0.9960`、MRD `0.9995/0.9974/0.9986/0.9994`。全6ペアの最大差・norm・相対 L2 は `semamba-seed17-repeat2-20261103/group-summary.json`。CQT の低い実装間 cosine は前回の seed 17 と同じ方向だが、別の今回試行では公式 A/B が `0.1792` で実装間の方が高い。系統差か非決定性のばらつきかは未解決。

MP-SENet は公式幅・VoiceBank 1秒の CPU 1 step で、公式 `train.py` の損失式・公式 PESQ と独立損失を別々に実行した単一プロセス対照 `mp-official-loop-cpu-20261103/waveform-gradient.json` を追加した。全 generator 損失成分差と PESQ 目標差は 0、generator 出力波形の勾配最大差 `5.82e-11`。これは波形勾配の1 step 対照で、4 step の分岐量の説明には代用しない。

### DeepFilterNet3 公式 libdf 前処理の追加診断（2026-11-03）

既存4プロセスの native CUDA 比較は公式・独立とも独立 DFStream を使う。別前処理の影響を見るため、公式側 libdf 0.5.6 と独立 DFStream を選べる probe を追加した。最初の CPU 4プロセス `df-libdf-four-cpu-20261103/` は probe 設定の初期化ミスで開始前に失敗し、修正後の `df-libdf-four-cpu-retry-20261103/` は VoiceBank 48 kHz・4,800 samples・1更新で valid。公式 A/B、独立 A/B は完全一致。公式/独立は forward 最大差 `2.09e-7`、loss 差 `0.0006294`、勾配最大差 `0.003249`・相対 L2 `0.0004976`、重み最大差 `0.0001966`。synthesis は両側とも独立の `_vorbis_istft` を使っており公式の全データ経路ではない。

別の同一入力・seed 3 の CPU 対照 `df-libdf-cpu-20261103/` で、モデルと損失を両側とも公式に固定し、libdf と DFStream の前処理だけを変更した。公式/独立モデルの対照と前処理のみの対照は共に loss 差 `0.0021400`、波形最大差 `1.30e-7`、勾配最大差 `0.003141`・相対 L2 `0.0003912`、重み最大差 `0.00199005` になった。最初に観測した差は前処理スペクトルで、最大 `2.24e-8`（実数・虚数の両方を比較）。seed 3・同じ公式モデル対照の最大重み差 `enc.df_conv0.2.weight` に対応する clip 後勾配 `4.24e-6` / `-1.31e-6` を同じ AdamW に与えると更新差 `0.00199005` を再現した。これはその要素の1更新差の説明であり、複数更新の差をすべて説明しない。

初期の探索結果 `df-cuda-paths-20261103/libdf-cpu1.json` は seed を固定しておらず、複素スペクトル差を実部のみで計算していたため、上の確定数値には使わない。`df-libdf-cpu-20261103/one-step.json` は seed 固定前の別初期値である。確定対照の再現コードは `tests/training/reference/probe_df_libdf.py`。公式 native CUDA と独立 native CUDA の前回の比較は共通 DFStream 条件であることを区別し、libdf を含む GPU 複数更新は未判定。

libdf と DFStream を別々に実行する CPU 4プロセス VoiceBank 4更新 `df-libdf-four-cpu4-20261103/` は valid。公式 A/B・独立 A/B は全 step で完全一致。公式/独立は step 1 の勾配相対 L2 `0.000498` から step 4 `0.10480`（cosine `0.99475`）へ増加し、step 4 の波形最大差 `4.15e-5`、loss 差 `0.0003891`、重み最大差 `0.0004047`。自己再実行との差は 0 である。1 step の frontend と AdamW の説明だけではこの4更新の増幅を説明しない。公式 native CUDA と独立 native CUDA の libdf 込み比較は未実施。

上の libdf 診断の当初版は alpha の計算に独立ヘルパーを使っていた。公式 `df/utils.py` の `get_norm_alpha` と `_calculate_norm_alpha` を直接実行するよう比較経路を修正し、同一 seed・モデル・入力の `official-alpha-seed3.json` で前処理、forward、loss、勾配、重みの値が先の対照と全て一致した。公式・独立の alpha の値はこの設定で同じである。

公式 libdf（alpha は公式コード）と独立 DFStream、両側 native CUDA の VoiceBank 4更新 `df-libdf-four-gpu-20261103/` は valid。step 1 勾配相対 L2 は公式 A/B `1.09e-6`、独立 A/B `1.56e-6`、公式/独立 `0.0002001` / `0.0001999`。step 4 は `0.003875`、`0.002891`、`0.007310` / `0.012201`、波形最大差は公式/独立 `0.0001362` / `0.0001364`。各モデルの spectral loss は native CUDA、逆変換は両側とも独立の `_vorbis_istft`。共通 DFStream の旧 native 比較より実装間差が大きい。所要 `37.88` 秒を `7693.216` 秒へ加算して `7731.096` 秒、残り `10268.904` 秒。SEMamba++ 3回目は未計上。

`official-fft-seed3.json` の先頭フレーム対照では、libdf の解析窓と独立の窓の最大差 0、ERB 幅も一致した。同一の窓付き波形に対し、libdf の解析 FFT と PyTorch float32 FFT・正規化の出力差は最大 `5.27e-9`。PyTorch float64 FFT を float32 へ戻した対照でも最大 `5.27e-9`。公式 Rust は `realfft`、独立は `torch.fft.rfft` であり、窓の取り違えではなく FFT・正規化を含む演算段階まで差を絞ったが、FFT のどの内部演算かは特定していない。独立実装を公式コードに実行時依存させる変更は行わない。

seed 17 の3回目 `semamba-seed17-repeat3-20261103/` も20 step・4系統 valid。step 20 の全勾配 cosine は公式 A/B `-0.0436`、独立 A/B `0.5348`、公式/独立 `0.6798` / `-0.1824`。generator は同順に `-0.0333/0.5379/0.6777/-0.2148`、CQT は `-0.7356/0.0864/0.7984/0.9433`、MRD は `0.9982/0.9981/0.9952/0.9960`。3試行の全ペア `three-trial-distribution.json` では cosine の min/median/max が公式内15ペア `-0.0436/0.2421/0.7216`、独立内15ペア `0.2364/0.4274/0.7107`、実装間36ペア `-0.1824/0.4058/0.7583`。同じ系統を共有するペアは独立標本ではなく、分布の重なりだけで実装差の不在を証明しない。初回 seed 17 の悪い結果も引き続き除外しない。process wall clock `374.88` 秒を保守的に加算し累計 `8105.976` 秒、残り `9894.024` 秒。既存ジョブ `115141`、予約自体は除外。

比較 adapter 更新後の training 再実行は `DF_REFERENCE_CONFIG` を指定して104件・skip 3、1 error。`test_trainer_matches_the_index_plan_for_one_and_two_ranks` の rank 0 が空きポート選択後に `EADDRINUSE` となり、rank 1 の待機が180秒で timeout した。該当テストを単独再試行すると2.615秒で成功した。ログは `/tmp/audio-db-training-recheck-20261103.log`。この再実行を全件成功とは扱わない。前の103件成功は adapter 変更中の実行である。

MP-SENet の公式 A・step 2 checkpoint を、optimizer state を名前で対応付けて4系統へ配った `mp-shared-loss-retry-20261103/mp-official-loop-gpu-20261103/` は valid。最初の `mp-shared-loss-20261103/` は直接実行時の `tests` import path の不備で準備に失敗し、path を修正して新しい出力先で再試行した。共通状態の step 3 は全ペアで forward・G/D loss 差 0、勾配相対 L2 `2.57e-8`〜`2.60e-8`。step 4 の勾配相対 L2 は公式 A/B `6.55e-6`、独立 A/B `2.88e-5`、公式/独立 `1.02e-5` / `2.90e-5`。この2更新の結果は初期状態を共有した局所比較であり、独立した4更新試行すべての差を単独で説明しない。GPU 所要 `64.83` 秒、既存ジョブ `115141`。直前 `8105.976` 秒から累計 `8170.806` 秒、残り `9829.194` 秒。

MP-SENet の同じ条件・同じ入力ハッシュの公式更新順4 step 再反復 `mp-official-loop-repeat-20261103/` も valid。step 4 勾配相対 L2 は公式 A/B `0.007866`、独立 A/B `0.07009`、公式/独立 `0.06985` / `0.005676`。2試行8系統・全28ペアの `two-trial-distribution.json` では相対 L2 の min/median/max が公式内 `0.00787/0.02268/0.02575`、独立内 `0.00627/0.07003/0.07514`、実装間 `0.00568/0.04733/0.07600`。実装間差は独立自己再実行の範囲と重なるが、公式自己再実行を超える例もあり、ペアは独立標本ではない。初回の小さい勾配の AdamW 増幅と共通 checkpoint からの局所分岐は確認したが、異なる複数 step すべての数値差を一意に説明したとはしない。所要 `37.02` 秒を加算し累計 `8207.826` 秒、残り `9792.174` 秒。既存ジョブ `115141`。

参考として、3試行12実行の cosine 行列に対し「同一ラベル内の平均 cosine − 実装間平均 cosine」を算出し、実装ラベルを6対6で置換する全924通りの探索的検定を行った（`exploratory-permutation.json`）。観測値 `-0.01512`、正方向の片側置換 p 値 `0.487`。この小標本では実装別の集まりを検出していないが、事後的な探索であり、p 値が大きいことは同等性の証明ではない。再現は `tests/training/reference/permutation_semamba.py`。

## 受入項目の現状（2026-11-03、今回の追加試験後）

以下の GO は記載した短時間・入力条件の学習処理についての判定であり、長時間学習後の品質を意味しない。PARTIAL は対応する処理が動かないという意味ではなく、同等性の根拠が足りない項目を含む。旧結果は独立のモデル・損失コードが `298b283` から変わっていないことと、上記の公式 checkout HEAD を確認したうえで使用する。

| 項目 | MP-SENet | SEMamba++ | TF-GridNet ms-snsd | DeepFilterNet3 | MossFormerGAN |
|---|---|---|---|---|---|
| 公式構造 | GO（公式幅、strict load） | GO（公式幅、strict load） | GO（公式幅、strict load） | GO（公式 DfNet、strict load） | GO（公式 G/D、strict load） |
| Forward | GO（実音声1 step） | GO（同一状態、fused） | GO（実音声、公式 STFT、4 step 差0） | PARTIAL（libdf 込み4 step 差あり） | GO（実音声、4 step） |
| 損失 | GO（公式式・PESQ、成分1 step 差0） | GO（成分別対照、共通状態） | GO（公式損失・4 step 差0） | PARTIAL（libdf 込み損失差、独立合成を共用） | GO（公式 loss と実 PESQ、device adapter） |
| Backward | PARTIAL（独立4 step の差の全量未特定） | PARTIAL（20 step 分布が広い） | GO（4 process 4 step 差0） | PARTIAL（libdf 込み、step 4 相対 L2 `0.0073/0.0122`） | GO（実音声4 step、相対 L2 `0.0028/0.0039` と公式 A/B `0.0041`） |
| Optimizer 更新 | PARTIAL（微小勾配1 step 説明、複数 step は未確定） | PARTIAL（共通状態では再実行差、20 step 未確定） | GO（4 step 差0） | PARTIAL（公式 FFT 差が増幅） | GO（微小勾配の AdamW 更新差を再現） |
| Scheduler・更新順 | GO（公式 D→G・epoch 単位、比較 probe） | GO（公式 D→G・epoch 単位、epoch 内） | GO（clip 5、epoch 中 plateau 固定） | GO（公式表に対応する per-step schedule） | GO（G/D backward・clip 後に両 step） |
| 勾配蓄積 | PARTIAL（共通 trainer と短い対照、公式幅実音声未完） | PARTIAL（DDP/蓄積対照あり、20 step との同時検証なし） | GO（公式幅実音声 CPU 1更新、勾配相対 L2 `1.61e-6` と Adam の差を説明） | PARTIAL（公式 batch と独立蓄積の十分な比較なし） | GO（実音声、蓄積2、4更新） |
| 公式 A/B 再実行 | GO（4 step、差を記録） | GO（seed 17 の3試行、差を記録） | GO（実音声4 step 差0） | GO（native・libdf、差を記録） | GO（実音声4 step、勾配相対 L2 `0.00407`） |
| 独立 A/B 再実行 | GO（4 step、差を記録） | GO（seed 17 の3試行、差を記録） | GO（実音声4 step 差0） | GO（native・DFStream、差を記録） | GO（実音声4 step、勾配相対 L2 `0.00195`） |
| 公式/独立比較 | PARTIAL（2試行・共通 checkpoint、公式 A/B より大きい例） | PARTIAL（3試行でも20 step の差の帰属未確定） | GO（公式 STFT、実音声4 step 差0） | PARTIAL（libdf と DFStream の差が自己再実行を超える） | GO（実音声4 step、公式 A/B より小さい勾配差） |
| checkpoint 再開 | GO（同一 checkpoint、別プロセス。CUDA 非決定性あり） | PARTIAL（同一 checkpoint から2 step、20 step 継続は未判定） | GO（実音声 GPU、別プロセス、状態 hash 一致） | GO（独立決定的経路は一致、公式 native は非決定的） | GO（同一 checkpoint からの別プロセス分岐） |
| VoiceBank 実音声 | GO（1秒、4更新） | GO（1秒未満、4更新；20 step は合成） | GO（1秒、4更新） | GO（48 kHz、0.1秒、4更新） | GO（1秒、蓄積2、4更新） |
| 総合 | PARTIAL | PARTIAL | GO | PARTIAL | GO |

TF-GridNet の GO は 4 step・実音声・公式前処理の範囲である。MossFormerGAN の GO は PESQ を CPU 上で計算して device に返す比較 adapter、実音声1秒・蓄積2・4 step の範囲である。DeepFilterNet3 の決定的 CUDA は独立実装の再現性用であり、公式 native とビット一致しないこと自体を NO-GO としていない。一方、公式 libdf 前処理の有無が4 step の差を増やすため総合 GO は保留する。

前処理以外を切り分ける CPU 対照 `df-shared-libdf-cpu4-20261103/` は、公式と独立の両モデルに公式 libdf の前処理を与えた VoiceBank 4プロセス4更新である。4系統すべての全 step で forward・loss・勾配・重みの最大差は 0。独立の標準 frontend に置き換えた前述の CPU 比較では step 4 勾配相対 L2 が `0.10480` となるため、この条件の差は入力前処理を変えたことに依存する。これは標準の独立 frontend を検証した結果ではない。

公式 libdf を両モデルに与える GPU native 対照 `df-shared-libdf-gpu4-20261103/` は valid。step 4 勾配相対 L2 は公式 A/B `0.001795`、独立 A/B `0.002870`、公式/独立 `0.001937` / `0.006365`。標準の独立 DFStream を使う対照の公式/独立 `0.007310` / `0.012201` と比べると小さいが、この1組の GPU 試行だけで全てを前処理差と断定しない。所要 `35.34` 秒を加算し、GPU 計上累計 `8243.166` 秒、上限まで `9756.834` 秒。既存ジョブ `115141`。

同条件の libdf 対 DFStream・native CUDA 4更新再反復 `df-libdf-four-gpu-repeat-20261103/` は valid。step 4 の勾配相対 L2 は公式 A/B `0.003645`、独立 A/B `0.000496`、公式/独立 `0.015192` / `0.012055`。実装間は引き続き自己再実行より大きい。所要 `33.58` 秒を加算し、GPU 計上累計 `8276.746` 秒（137分56.746秒）、残り `9723.254` 秒（162分3.254秒）。既存ジョブ `115141`、予約時間そのものは含めない。計上には CPU 集計・重複時間を保守的に含む。

## DeepFilterNet3 前処理の一致（2026-11-03）

HEAD `1ff9da2` から `DFStream` を修正した。公式モデルと損失は変更していない。公式ソースも編集していない。

前処理を同じ VoiceBank 0.1 秒（`df-libdf-four-cpu4-20261103/input.pt`）で段ごとに分けた。窓の最大差は 0、ERB 幅は一致する。同じ複素スペクトルを両側の特徴抽出へ入れると、実スペクトルは一致し、ERB と DF 特徴だけが最大 `1e-6` 残った。原因は float32 の演算順である。

- `1.0 - alpha` を Python の float64 で計算していた。float32 の `1 - alpha` にすると、同一スペクトルの ERB 正規化は libdf と一致する。
- unit norm の初期状態は `torch.linspace` ではなく、float32 の `start + i * (end - start) / (n - 1)` である。末端を `end` に揃えるとずれる。
- 帯域パワーは libdf と同じくビンごとの `k = 1/width` を順に足す。`tensor.mean` では最大 `1e-11` 以上ずれる。
- `torch.log10` と `torch.sqrt` は libdf の `f32::log10` / `f32::sqrt` と最大 `1e-5` ずれる。`torch.fft.rfft` は同じ窓付き波形でも `realfft` 3.3.0 と最大絶対差 `6e-8` で、float64 FFT を float32 に戻しても縮まない。

この FFT 差だけで step 1 の勾配符号が反転する。差を 1% に縮小しても符号反転は残った。同一初期重み・step 4 の音声だけでは勾配相対 L2 は `0.00205` だが、4 更新後は重みの最大差 `4e-4` から勾配相対 L2 `0.105` まで増幅していた。増幅の起点は、絶対値が `1e-6` 前後の勾配の符号差が AdamW（lr `0.001`、epsilon `1e-8`）で約 `0.002` の重み差になることである。

独立実装の解析 FFT、ERB の `log10`、unit norm を、libdf 0.5.6 と同じ `realfft` 3.3.0 の float32 実装にした。実行時に libdf は読まない。カーネルは `src/se/models/deepfilternet3/rfft_kernel` から作り、`libdf_rfft.so` として置く。単体テスト `tests.training.test_df_libdf_frontend` は 2 チャネルの正規乱数と学習と同じ 4,800+960 パディングで、解析スペクトルと ERB・unit norm の最大差が 0 である。

修正後の 4 プロセス比較は、公式側 libdf、独立側 DFStream、VoiceBank 48 kHz・4,800 samples、seed 3、`DF_PROBE_OFFICIAL_LIBDF=1`。結果は `data/results/training/df-libdf-match-cpu4-20261103/`。4 系統の条件照合は valid。公式 A/B、独立 A/B、公式/独立の全 step で forward・loss・勾配・更新後重みの最大差は 0。修正前の CPU step 4 勾配相対 L2 は `0.10480` だった。

同じ入力の H100 MIG（既存ジョブ `115141`）2 試行は `df-libdf-match-gpu4-20261103/` と `df-libdf-match-gpu4-repeat-20261103/`。どちらも valid。step 4 の勾配相対 L2 は次のとおり。

| 試行 | 公式 A/B | 独立 A/B | 公式 A/独立 A | 公式 B/独立 B |
|---|---:|---:|---:|---:|
| 1 | 0.002297 | 0.006844 | 0.004704 | 0.000846 |
| 2 | 0.000422 | 0.003957 | 0.002465 | 0.001438 |

CPU が全要素 0 なので、この GPU 差は前処理やモデル式の取り違えではない。実装間の大きい方 `0.004704` は、同じ試行の独立再実行 `0.006844` を超えない。修正前の実装間 `0.012`–`0.015` は、公式再実行 `0.0036` と独立再実行 `0.0005`–`0.0029` の両方を超えていた。GPU 実験のプロセス経過は `36.36 + 36.35 = 72.71` 秒。累計 `8349.456` 秒、残り `9650.544` 秒（160分50.544秒）。対話予約そのものは加算していない。

checkpoint 再開のコードは今回変えていない。前回の再開 GO は、この前処理修正の前に記録したものである。今回の CPU 4 更新は連続実行であり、再開の再試験ではない。

この範囲の DeepFilterNet3 は、公式 libdf と独立 DFStream を使った短時間学習の再現性を GO とする。長時間学習後の PESQ・STOI・SI-SDR・WER は対象外である。MP-SENet と SEMamba++ は、この節の修正対象ではなく、直前の受入表の PARTIAL のままである。

### 回帰テスト（2026-11-03、`.venvs/tsubame`、`DF_REFERENCE_CONFIG` 指定）

ログは `data/results/training/regression-20261103-cont/`。失敗は無い。

| スイート | 結果 |
|---|---|
| `tests/data` | 16件成功、skip 0 |
| `tests/training` | 107件成功、skip 3 |
| `tests/se` | 28件成功、skip 19 |
| `tests/evaluation` | 86件成功、skip 3 |
| `tests/asr` | 5件成功、skip 0 |

training の skip 3件は、再構築 DB `data/db/rebuild-a.duckdb` が無いこと、およびこの全体実行の直前に `CUDA_VISIBLE_DEVICES` が空だったため `CUDA required` になる MossFormerGAN 蓄積テストと DeepFilterNet3 spectral テストである。se の skip は配布 checkpoint・参照結果・明示 CUDA 拡張テスト。evaluation の skip は保存例と MP-SENet checkpoint がこの環境に無いこと。分散テストは、選んだポートが `EADDRINUSE` のとき別ポートでやり直し、失敗した rank の相手を待たずに止める。該当の supply・mock・accumulation は単独でもこの全体実行でも成功した。

CUDA を見える状態で `tests.training.test_df_reference_spectral` と `tests.training.test_moss_probe_accumulation` を再実行すると 8件成功、6.18秒だった。この 6.18秒を GPU 枠へ加算し、累計 `8355.636` 秒、残り `9644.364` 秒とする。

再現:

```bash
. "$HOME/.cargo/env"
cargo build --release --manifest-path src/se/models/deepfilternet3/rfft_kernel/Cargo.toml --target-dir /tmp/df-rfft-build
cp /tmp/df-rfft-build/release/libdf_rfft.so src/se/models/deepfilternet3/libdf_rfft.so
.venvs/tsubame/bin/python -m unittest tests.training.test_df_libdf_frontend -v
export DF_REFERENCE_CONFIG="$PWD/data/results/training/final-h100-20261102-cpu/df-config.ini"
export DF_PROBE_OFFICIAL_LIBDF=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
OMP_NUM_THREADS=1 .venvs/tsubame/bin/python tests/training/four_process/compare.py run data/results/training/NEW-DF --model deepfilternet3 --steps 4 --device cpu --dataset voicebank_28_train --database data/db/voicebank.duckdb --segment-size 4800 --sample-rate 48000 --seed 3
```

## DeepFilterNet3 ビルド再現性と MP-SENet の単一 forward（2026-11-03）

HEAD `38f9628` から作業した。既存の DB、checkpoint、前セッションの結果ディレクトリは上書きしていない。

### Task 1：DeepFilterNet3 の共有ライブラリ

判定は **GO**（この Linux x86-64、rustc 1.99.1、cargo 1.99.1）。別マシンでの再ビルドはしていない。

`make df-rfft` が `Cargo.lock` の `realfft` 3.3.0 と `rustfft` 6.4.1 で `cdylib` を作り、`src/se/models/deepfilternet3/libdf_rfft.so` に置く。Cargo の中間出力は `DF_RFFT_TARGET`（既定はリポジトリ内の `build/df-rfft`、Git 管理外）。クリーン試験は `DF_RFFT_TARGET=/tmp/df-rfft-clean-20261103` で行い、既存の `.so` は cargo が成功したあとだけ置き換えた。`DF_RFFT_LIBRARY` でそのビルド成果物を直接読み、`tests.training.test_df_rfft_kernel` と `tests.training.test_df_libdf_frontend` は 5 件成功した。libdf との解析スペクトル、ERB、unit norm の最大差は 0 のままである。

C ABI は `df_rfft_abi` が 1 を返す。欠けるか版が違うライブラリは `RuntimeError`。ファイルが無く `DF_RFFT_REQUIRED=1` のときは `FileNotFoundError`。未設定なら従来どおり PyTorch に戻る。動的リンクは `libc`、`libm`、`libgcc_s` のみで、ホスト固有のパスは入っていない。

ライセンスは、このクレートが MIT OR Apache-2.0、`realfft` 3.3.0 が MIT、`rustfft` 6.4.1 が MIT OR Apache-2.0。libdf はリンクしていない。`Cargo.toml` の `rust-version` は 1.70（`OnceLock`）。1.70 そのものでのコンパイルは未実施。

```bash
make df-rfft DF_RFFT_TARGET=/tmp/df-rfft-clean
DF_RFFT_LIBRARY=/tmp/df-rfft-clean/release/libdf_rfft.so \
  .venvs/tsubame/bin/python -m unittest tests.training.test_df_rfft_kernel tests.training.test_df_libdf_frontend -v
```

### Task 2：MP-SENet

公式 `train.py` は Generator を 1 回だけ計算し、Discriminator にはその magnitude を detach して渡す。独立の学習ステップと 4 プロセス probe は、Discriminator 用の `no_grad` forward のあと Generator 損失でもう一度 forward していた。CPU では両者が一致する。CUDA では 2 回目が別の評価になり、独立どうしの再実行差が公式より大きくなり得る。両方を 1 回の forward に合わせた。公式損失式は変えていない。

同じ VoiceBank 入力（seed 3、ハッシュ `dd9500c21e2b4081f40b1f9f685bc6d172b1460b3ea3ba5ed22b2989b16e2b86`、1 秒、公式幅、epoch 単位 scheduler）の H100 4 更新は `data/results/training/mp-one-forward-gpu4-20261103/`。4 系統 valid。step 1 は全ペアで forward 最大差 0、両損失差 0、勾配相対 L2 は `7.65e-8`〜`8.24e-8`。最初に重みが割れるのはこの更新で、最大は公式 A/B の `generator.dense_encoder.dense_conv_1.0.bias` の 1 要素 `4.924e-4`。対応勾配は `-1.676e-8` と `5.588e-9`。同じ AdamW（lr `5e-4`、epsilon `1e-8`）の初回更新差の閉形式は `4.92420e-4` で、実測との差は `4.1e-9`。公式再実行でも符号が割れており、実装間に限った分岐ではない。

同じ比較を seed 17 と 29 でも行った。出力は `mp-one-forward-gpu4-seed17-20261103/` と `mp-one-forward-gpu4-seed29-20261103/`。どちらも valid。step 1 は 3 seed とも全ペアで forward 最大差 0、両損失差 0、勾配相対 L2 は `5.06e-8`〜`9.61e-8`。重み最大差は `3.10e-4`〜`9.28e-4` で、seed 3 と同じく epsilon 付近の勾配符号差が AdamW で広がったものである。

step 4 の勾配相対 L2 は次のとおり。

| seed | 公式 A/B | 独立 A/B | 公式 A/独立 A | 公式 B/独立 B |
|---|---:|---:|---:|---:|
| 3 | 0.03245 | 0.04986 | 0.01122 | 0.04808 |
| 17 | 0.03651 | 0.02670 | 0.03588 | 0.02744 |
| 29 | 0.26254 | 0.16549 | 0.18249 | 0.23387 |

3 seed の min / median / max は、公式内 `0.03245 / 0.03651 / 0.26254`、独立内 `0.02670 / 0.04986 / 0.16549`、実装間 6 ペア `0.01122 / 0.04198 / 0.23387`。実装間の各ペアは、同じ seed の大きい方の自己再実行を超えていない。修正前は同じ seed 3 入力で独立 A/B が `0.0702` と `0.0749`、公式 A/B が `0.00774` と `0.01146` で、独立側だけが一桁大きかった。その偏りは 2 回目の forward をやめたあと残っていない。seed 29 の公式どうし `0.26` は、4 更新でも公式自身のばらつきが大きいことを示す。ペアは実行を共有するので独立標本ではない。長時間学習の品質は対象外である。

この範囲の Forward、損失、Backward、Optimizer、4 プロセス、checkpoint 再開を **GO** とする。再開は同一実装の別プロセスで、前セッションの GPU 再開と今回の CPU 回帰 `test_two_runs_and_a_resumed_run_match` による。公式から独立への Optimizer 移植は比較用で、`shared_fork.py --remap-mp` が名前で対応付ける。

seed 3 のプロセス経過は `107.45` 秒（ノード `r5n4`、シェル経過 `137` 秒。表示メモリは親 GPU の 5602 / 95830 MiB）。seed 17 と 29 はジョブ `9242017`、ノード `r13n3`、開始時 87 / 95830 MiB、プロセス経過 `119.52 + 43.52` 秒。キュー待ち約 9 分は GPU 時間に入れない。`/tmp` を計算ノードから参照できず落ちた試行は実験 0 秒、シェル約 `15.8` 秒。即時スケジュール不能の拒否は GPU を使っていない。実験プロセスの合計 `270.49` 秒を `8355.636` に加え、累計 `8626.126` 秒、残り `9373.874` 秒。

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venvs/tsubame/bin/python tests/training/four_process/compare.py run data/results/training/NEW-MP \
  --model mp_senet --steps 4 --device cuda:0 \
  --dataset voicebank_28_train --database data/db/voicebank.duckdb \
  --segment-size 16000 --sample-rate 16000 --seed 3 \
  --scheduler-step epoch --official-width
```

### Task 3：SEMamba++

同一実装の Optimizer 再開は、登録順が同じなので位置ロードでモーメントが同じパラメータに戻る。狭幅 Generator の CPU 往復で `exp_avg` と `exp_avg_sq` が一致した。公式と独立は名前が一致しても登録順が違い、位置のままロードするとモーメントが別パラメータに付く。`remap_optimizer` のあと名前ごとに一致する。これは `tests.training.test_shared_fork` で確認した。公式 import には `mamba_ssm`（この環境では `PYTHONPATH=.venvs/tsubame-h100-build`）が要る。Trainer の `load_state_dict` は自分の checkpoint を位置で読む。公式状態を独立モデルへ移すのは比較操作であり、通常の再開ではない。比較の `shared_fork.py` は `--remap-semamba` が無いと拒否する。

複数更新の新しい GPU 比較は今回していない。前回の 20 step では、実装間 cosine の最小が公式内の最小より低く、分布の重なりだけでは系統差の不在を示せないと記録してある。その判定は維持する。selective scan の backward は入力が同じでも再実行差がある、という前回の切り分けも変えていない。総合は **PARTIAL**。

### 回帰

`CUDA_VISIBLE_DEVICES` を空にした `.venvs/tsubame` で、DF カーネル、libdf 前処理、MP-SENet 学習、公式損失、4 プロセス、Optimizer 対応を 36 件実行し、失敗 0、skip 1。skip は、同じプロセスで先に公式 MP-SENet テストが空の `librosa` モジュールを登録したため、SEMamba++ 公式の import が失敗したものである。そのあと `PYTHONPATH=.venvs/tsubame-h100-build` で `test_semamba_cross_implementation_optimizer_needs_name_remap` を単独実行し、成功した。

## SEMamba++ 同一状態の Backward（2026-11-03）

作業ツリーは `084edff` の上。公式 SEMamba++ は `cd369c89a8106072c077b468417b765b227afd74`。公式ソースは編集していない。既存の結果ディレクトリと checkpoint は上書きしていない。ノードは `r4n11`。開始時の表示メモリは 101 / 95830 MiB。新しい `qrsh` は出していない。

### 比較基盤

狭幅・CPU・reference で、公式と独立のパラメータ名集合・shape・dtype が一致し、登録順は一致しない。同じ勾配を名前で与えた AdamW の `exp_avg` と `exp_avg_sq` は名前ごとに一致する。次の4方向でモーメントが名前に付いたまま戻る。

1. 公式 checkpoint → 公式
2. 独立 checkpoint → 独立
3. 公式 checkpoint → 独立（名前対応）
4. 独立 checkpoint → 公式（名前対応）

位置のまま公式 Optimizer を独立へ載せる対応のずれる件数は 0 より大きい。パラメータに属さない Optimizer state は `remap_optimizer` が `ValueError` にする。Scheduler の `last_epoch` と `get_last_lr()` は state dict のコピーで一致する。RNG は `capture_rng` / `restore_rng` で往復する。通常の Trainer 再開は同一実装の位置ロードであり、この名前対応とは別である。

空の `librosa` を `sys.modules` に残していた公式 MP-SENet の import をやめた。MP-SENet の公式ステップのあとに SEMamba++ 公式を import しても `librosa.filters.mel` が実モジュールである。

### 同一重みの更新

判定規則は実行前に `tests/training/reference/audit_semamba_same_state.py` へ書いた。同一重み・同一入力・同一 RNG を毎回戻し、Optimizer は repeat をまたがない。`bias_rel_l2` は公式勾配の平均と独立勾配の平均の相対 L2、`self_median` は実装内ペア相対 L2 の中央値の大きい方、`cross_median` は公式対独立の中央値である。規則は `bias_rel_l2 <= self_median` かつ `cross_median <= self_median`。

入力は `semamba-seed17-repeat-20261103/input.pt`（seed 17、公式幅、fused、epoch scheduler、合成 8,000 samples）。重みは初期状態、公式 A の step 4、公式 A の step 20。H100、TF32 なし、cuDNN deterministic。

修正前の fused 経路は `xz` を contiguous にしていた。selective scan の 24 呼び出しは値が一致しても stride が公式 `(81, 8100, 1)`、独立 `(7776, 81, 1)` だった。contiguous コピーをやめ、公式と同じ view にした。修正後は 24 呼び出しの shape・stride・dtype・contiguous フラグが一致し、forward 入力の最大差は 0 である。Forward 波形と Generator 損失は、修正の前後とも公式再実行・公式/独立で最大差 0、損失は初期 `52.88309860229492`、step 4 `57.11925506591797`、step 20 `54.5202522277832`。Discriminator 勾配の差は 0。CPU の位相損失は値も位相勾配も公式と一致する。

修正後の全パラメータ勾配は次のとおり。

| 重み | repeats | bias 相対 L2 | self median | cross median | cross / self |
|---|---:|---:|---:|---:|---:|
| 初期 | 4 | 7.357e-7 | 1.431e-6 | 1.439e-6 | 1.0056 |
| step 4 | 8 | 6.507e-7 | 1.765e-6 | 1.769e-6 | 1.0023 |
| step 20 | 4 | 1.017e-6 | 1.969e-6 | 1.972e-6 | 1.0015 |

平均勾配の差は、どの点でも実装内の再実行より小さい。pairwise の中央値は self を 0.15–0.56% 上回り、上のブール規則そのものは成立しない。この超過は勾配そのものに対する 1e-6 のノイズの一部であり、損失式の取り違えではない。`|grad| < 1e-8` の符号不一致数は、初期状態で公式内 410–450、独立内 417–476、実装間 401–467 と重なる。AdamW がこの符号差を O(学習率) の重み差へ増幅する機構は、公式の再実行にも同じ大きさで含まれる。

したがって、過去の seed 17・20 step で実装間 cosine が公式内の最小を下回ったことは、この 1 step の非決定性が 20 回積もった結果である。公式どうしの step 20 cosine も `-0.0436` から `0.7216` まで広がっている。新しい 20 step の4プロセスは、同じ初期状態からの軌跡が再びこの範囲へ散ることを繰り返すだけなので実行していない。

この範囲のモデル構造、Forward、損失、Backward、Optimizer、4プロセス比較、同一実装の checkpoint 再開、複数更新を **GO** とする。長時間学習後の PESQ・STOI・SI-SDR・WER は対象外である。selective scan backward はビット一致しない。

### GPU 時間

プロセス経過は `35.094 + 17.450 + 18.578 + 39.307 + 13.104 + 16.829 + 39.415 + 17.080 + 18.697 = 215.554` 秒。直前の累計 `8626.126` 秒に加え `8841.680` 秒。上限 300 分まで `9158.320` 秒（152分38.320秒）。

### 回帰

GPU が見える `.venvs/tsubame` で、MP-SENet 公式ステップを先に実行してから SEMamba++ の Optimizer・損失・MRD・mel・長さを実行した。17 件成功、skip 1。skip は `SEMAMBA_TEST_FUSED_CUDA` 未設定の明示 CUDA 拡張テストであり、librosa の失敗ではない。`CUDA_VISIBLE_DEVICES` を空にすると Triton が driver を初期化できず、公式 SEMamba++ の import は `RuntimeError` になり、そのテストは skip する。

```bash
PYTHONPATH=.venvs/tsubame-h100-build CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 \
  .venvs/tsubame/bin/python tests/training/reference/audit_semamba_same_state.py \
  --input data/results/training/semamba-seed17-repeat-20261103/input.pt \
  --modules data/results/training/semamba-seed17-repeat-20261103/official_a/fork-step-4.pt \
  --output data/results/training/NEW-semamba-same-state.json --repeats 4 --device cuda:0
PYTHONPATH=.venvs/tsubame-h100-build .venvs/tsubame/bin/python tests/training/reference/audit_semamba_scan_calls.py \
  --input data/results/training/semamba-seed17-repeat-20261103/input.pt \
  --modules data/results/training/semamba-seed17-repeat-20261103/official_a/fork-step-4.pt \
  --output data/results/training/NEW-semamba-scan.json --device cuda:0
```

結果は `data/results/training/semamba-same-state-20261103/`、`semamba-same-state-step4-20261103/`、`semamba-same-state-step20-20261103/`、`semamba-same-state-step4-rep8-20261103/`、`semamba-same-state-init-layout-20261103/`、`semamba-same-state-step4-layout-rep8-20261103/`、`semamba-same-state-step20-layout-20261103/`、`semamba-scan-calls-step4-20261103/`、`semamba-scan-calls-step4-layout-20261103/`。

## 公式プロセスとクリーン環境（2026-11-04）

この節は 5 モデル最終受入の再判定ではない。HEAD は作業開始時 `af56102fac400a1958b3e626f14b21654f644396` で、`dev` は `origin/dev` と一致していた。学習ライブラリの更新式は変えていない。GPU は対話ジョブ `115437`（ホスト `r4n11`、H100 MIG 4g.47gb）を再利用し、新しい `qrsh` は出していない。プロセス時間は `/usr/bin/time` の wall clock で、失敗も含める。会計の開始値は 11,402.500 秒、上限 18,000 秒である。

3 タスクの判定は次のとおり。学習実装の GO とは混ぜない。

| タスク | 判定 | 根拠 |
|---|---|---|
| 1. 公式学習プロセス | PARTIAL | MP-SENet、MossFormerGAN、DeepFilterNet3、SEMamba++ は公式エントリーポイントが実音声で有限損失の更新まで進んだ。TF-GridNet は公式モジュールの 2 更新までで、`espnet2.bin.enh_train` は `typeguard` 不足で起動していない |
| 2. クリーン環境 | GO | 新規 venv、ピン留め clone、sm_90 の Mamba 再ビルド、Rust FFT、`libdf` / `libdfdata`、5 モデルの独立スモークが完了した。別ノードは未実施で、失敗原因の切り分け対象ではない |
| 3. MossFormerGAN GPU 再開 | GO | 更新前のパラメータ、Adam モーメント、Scheduler、4 種の RNG、step、epoch、蓄積余り、次入力が再開 A/B で一致した。更新後の重み差は同じ実装の連続再実行にもある |

### 公式エントリーポイント

起動スクリプトは `tests/training/reference/official_entry_smoke.py`。公式ツリーは `build/clean-20261104/sources/` の新規 clone で、`src/se/official/` は編集していない。

| モデル | エントリーポイント | commit | 結果 | 記録 |
|---|---|---|---|---|
| MP-SENet | `train.py` | `89932cfe90d1dacb8e170e4a331d762462c21792` | Steps 0–3 の Gen Loss 2.023、1.841、1.691、1.835。すべて有限。プロセス 22.74 秒 | `data/results/training/official-entry-20261104b/` |
| MossFormerGAN | `train/speech_enhancement/train.py` | `6b3774dc79c46ae8bed2a4fa5f706f0ac8c75c61` | Epoch 0 の 4 micro-batch、Gen_Loss 0.8408、1.0018、0.7505、0.6518。Optimizer 更新は 2 回と検証 1 回。プロセス 21.74 秒 | `data/results/training/official-entry-moss-20261104/` |
| TF-GridNet | `espnet2/bin/enh_train.py` は未起動。STFT encoder/decoder、TFGridNet、`MultiResL1SpecLoss` | `0d136c5456d403cb2c0a4c13e76e8136e32e9b7a` | 実音声 2 更新。公式損失と独立損失はともに 42.44139862060547。forward 最大差 0 | `data/results/training/official-entry-20261104/tf-gridnet/` |
| DeepFilterNet3 | `df/train.py` | `d375b2d8309e0935d165700c91da9de862a99c31` | `num_workers = 2`、`overfit = true` で train loss 9.34261、9.76526。checkpoint を書いた。プロセス 7.35 秒 | `data/results/training/official-entry-df-20261104/deepfilternet3/train-workers2.log` |
| SEMamba++ | `train.py` | `cd369c89a8106072c077b468417b765b227afd74` | Epoch 0、Steps 0、Gen Loss 11.139、Disc Loss 6.006。パラメータ数 2,655,521。プロセス 24.40 秒 | `data/results/training/official-entry-semamba-20261104g/` |

短時間化で公式設定から動かした値は各 `report.json` と上のログにある。共通して batch は 1、worker は 0 または DeepFilterNet だけ 2、epoch は 1、検証と checkpoint の間隔は伸ばしている。入力は VoiceBank の実音声と、必要なモデルでは DEMAND の実雑音である。DeepFilterNet の RIR は単位インパルスで `p_reverb = 0`。SEMamba++ の RIR も単位インパルスである。TF-GridNet のプローブは 48 kHz の先頭 2048 サンプルをそのまま使っており、16 kHz へ resample していない。

DeepFilterNet の最初の起動は `num_workers = 0` で、epoch 開始直後に公式 DataLoader が `TimeoutError` を出し、プロセスは約 78 秒後に停止させた。`num_workers = 2` では同じ HDF5 で更新が進んだ。これは公式ローダの worker 数の制約であり、損失式の失敗ではない。

SEMamba++ のピン留め commit は `utils.py` しか持たないが、`dataset.py` は未使用の `from utils.util import load_config` を実行する。外部 launcher が `utils.py` の `load_config` を `sys.modules['utils.util']` に登録し、`inspect.getsource(train.main)` の epoch 数、間隔、JSON パス、出力先だけを置換して `exec` する。公式ファイルは書いていない。この置換は学習式ではなく実行回数と入出力パスである。`load_modules` は検証をスキップしても UTMOS をダウンロードする。今回はダウンロードが完了し、その重みは損失には入っていない。

### Adapter との差

| 箇所 | 公式プロセス | 比較 Adapter |
|---|---|---|
| MP-SENet の PESQ | 公式 `batch_pesq`（loky） | `joblib.parallel_backend("threading")`。公式関数の中身は同じ |
| MossFormerGAN の PESQ | 公式 `batch_pesq` は `.to('cuda')` | 同じ正規化 `(score-1)/3.5` を step の device で計算。1 秒の VoiceBank 1 組で最大差 0（`official-entry-20261104b` の PESQ 比較） |
| MossFormerGAN の AdamW | YAML の `weight_decay` 1e-5 | 比較経路は 0.01。betas は既定 (0.9, 0.999)。今回この値は変えていない |
| MossFormerGAN の plateau | 公式 Solver が学習率を半減し得る | 比較経路には無い |
| TF-GridNet | `enh_train.py` は未起動 | 比較幅（n_fft 512、stride 256、n_layers 4、LSTM 128）、Adam 1e-3、clip 5。公式モジュールとの 2 更新の損失差は 0 |
| DeepFilterNet3 | 公式 `df/train.py`、`libdfdata`、公式 loss | 比較経路は独立モデル。`libdf.DF.synthesis` は学習 ISTFT ではない |
| SEMamba++ | 公式 `train.py` と公式幅 hid 48、num_tfmamba 4 | 独立スモークは hid 8 の reference scan。fused 公式幅の学習ステップは今回のクリーン環境では公式エントリーポイント側で 1 回 |

### クリーン環境

手順は `docs/envs/CLEAN_REBUILD.md`。直接 pin は `requirements/clean-rebuild.txt`。導入後の freeze は `build/clean-20261104/requirements-freeze.txt` と `requirements-freeze-df.txt`（Git 管理外）。

- 主環境 `.venvs/clean-20261104`: Python 3.12.13、torch 2.6.0+cu124、NumPy 2.5.4、Triton 3.2.0
- DeepFilterNet 環境 `.venvs/clean-20261104-df`: 同じ torch、NumPy 1.26.4。PyO3 0.20 を NumPy 2.5 に載せていない
- Mamba: causal-conv1d 1.7.0 と mamba-ssm 2.3.2.post1 を `TORCH_CUDA_ARCH_LIST=9.0`、CUDA 12.3.2 で再ビルド。import 元は `.venvs/clean-20261104/lib/python3.12/site-packages/` の `selective_scan_cuda` と `causal_conv1d_cuda`。`Mamba(d_model=16, d_state=16, d_conv=4, expand=2)` の CUDA forward/backward は有限（`clean-rebuild-20261104/mamba-forward.log`、4.53 秒）
- Rust FFT: `build/clean-20261104/cargo-rfft/release/libdf_rfft.so`。既存のツリー内 `.so` は上書きしていない。`DF_RFFT_LIBRARY` をこのファイルに向け、`DF_RFFT_REQUIRED` を立てない状態で `tests/training/test_df_rfft_kernel.py` は 3 件成功。`DF_RFFT_REQUIRED=1` を親プロセスに残すと、未設定時のフォールバックを試すテストが落ちる。これはテストが環境変数を消さないためで、ライブラリ欠如時のエラー自体はテスト内で確認している
- `libdf` と `libdfdata` は `/tmp` からの import で site-packages を読んだ
- 独立 5 モデルの `tests/training/gpu_smoke.py` は主環境で EXIT 0、12.87 秒（`clean-rebuild-20261104/gpu-smoke.log`）。SEMamba++ はこのスモークでは狭幅である

### MossFormerGAN の GPU 再開

連続 4 更新は `data/results/training/moss-gpu-resume-20261104/`。VoiceBank `voicebank_28_train`、16 kHz、segment 16000、seed 17、公式幅、蓄積 2、`--fork-step 2`、`cuda:0`。4 worker のプロセス合計 126.52 秒。step 1 と step 2 の Generator 損失は 4 本とも 2.314906 と 0.822889。step 2 の重みハッシュは公式 A `66fc216dbe57`、公式 B `d0f199b8679f` で、同じ実装の連続再実行がこの時点で分かれている。

再開確認は `tests/training/reference/moss_gpu_resume_check.py`。公式 A と独立 A の fork を、別ディレクトリの再開 A と再開 B で読み直した。更新前はパラメータ、`exp_avg`、`exp_avg_sq`、`step`、Scheduler、Python / NumPy / PyTorch / CUDA RNG が checkpoint と一致し、`global_step` 2、`epoch` 0、蓄積余り 0 だった。続けた 2 更新の入力 ID は A と B で同じ（`45db58578f59`、`0a20a20f3292`）。更新後の重みハッシュは A と B で異なり、連続側の step 3 とも異なる。損失は有限で、公式 A の再開 A/B は step 3 がどちらも 0.559283435344696、step 4 がどちらも 0.5049392580986023 だった。独立 A も A/B で同じ損失 0.5592777729034424、0.5050094127655029 だった。重み差は Optimizer の取り違え、RNG の欠落、入力順の差では説明できない。CUDA 自己再実行が更新後ビット一致しない範囲である。

記録は `moss-gpu-resume-20261104-resume-a/` と `moss-gpu-resume-20261104-resume-b/` の `resume.json`。再開 4 本のプロセス時間は 13.08、13.96、12.73、12.45 秒。

### GPU 時間

開始 11,402.500 秒。前半の記録済み実行は 285.27 秒で、内訳は公式エントリーの試行 32.85、34.40、21.88 秒、MossFormerGAN 連続 4 worker 126.52 秒、再開 4 本 52.22 秒、5 モデルスモーク 12.87 秒、Mamba forward/backward 4.53 秒である。後半は DeepFilterNet 準備込みの失敗起動 13.49 秒、worker 0 で停止させた約 78 秒、worker 2 の成功 7.35 秒、SEMamba++ の import または launcher 失敗 7.38、7.46、9.70、7.14 秒、成功 24.40 秒で、合計 154.92 秒である。この節の加算は 440.19 秒。累計 11,842.690 秒。残り 6,157.310 秒。

## Optimizer・公式 Trainer・融合スモーク（2026-11-04 継続）

この節は 5 モデル最終受入の再判定ではない。対象は MossFormerGAN の実効 Optimizer、TF-GridNet の `enh_train.py`、クリーン環境の Mamba パッチと公式幅 fused の 1 更新である。GPU は同じ対話環境の H100（ログのホスト `r4n11`）を再利用した。新しい `qrsh` は出していない。

| タスク | 判定 | 根拠 |
|---|---|---|
| MossFormerGAN の Optimizer・Scheduler | GO | 公開コンストラクタは `weight_decay` を渡さない。実効値は AdamW 既定の 0.01 で、比較経路も同じ。Plateau は公開 Solver の判定に合わせ、CPU テストで確認した。4 更新の実装間差は自己再実行の範囲 |
| TF-GridNet の公式 Trainer | GO | `espnet2.bin.enh_train` が公式 DataLoader、16 kHz、公式 STFT 設定で 1 epoch を終え、train loss 2958 と valid loss 2213 は有限。backward と Adam 更新が記録された |
| クリーン環境の最終整備 | GO | Mamba の import 差分はパッチと適用スクリプトにした。公式幅・fused の独立 SEMamba++ が 1 更新した。別ノードは未実施 |

### MossFormerGAN

公開 `train/speech_enhancement/train.py` の MossFormerGAN 分岐は `torch.optim.AdamW(parameters, lr=...)` だけを呼ぶ。YAML の `weight_decay: 1e-5` は FRCRN の Adam には渡るが、この分岐には渡らない。実効値は PyTorch 既定の weight decay 0.01、betas (0.9, 0.999)、eps 1e-8、amsgrad なし、パラメータグループは 1 つである。識別器の学習率は生成器の 2 倍。独立実装と 4 プロセス比較は、この実効値のままである。YAML の 1e-5 に合わせると公開プロセスとずれる。

公開 Solver に `lr_scheduler` オブジェクトは無い。検証損失が記録値以上の回を数え、5 回目で最後に改善した checkpoint の重みと Optimizer を戻し、生成器の先頭グループの学習率だけを半分にする。回数は半分にしたあとリセットしない。10 回目で学習を止める。悪化した損失は記録上の最良値を上書きするので、その次が直前より小さければ改善になる。独立実装の `step_schedulers` をこの判定に合わせた。回数、最良値、最良 snapshot は独立 checkpoint の `plateau` に残す。公開プロセスを再起動すると `val_no_impv` と `best_val_loss` は初期値に戻る。4 更新の比較ではこの分岐は起きない。

CPU の `tests/training/test_mossformer_plateau.py` は 5 件成功した。維持、5 回目の半減と重み復元、識別器学習率を変えないこと、10 回目の停止、checkpoint 往復後の回数継続を見ている。

GPU の 4 プロセスは `data/results/training/moss-optmatch-20261104/`。VoiceBank、16 kHz、seed 17、公式幅、蓄積 2、4 更新、プロセス 122.42 秒。初期状態の一致フラグは 4 本とも真。step 4 の生成損失差は、公式内 1.41e-5、独立内 5.61e-5、公式 A 対独立 A 5.57e-5。forward の最大差は公式内 0.00296、独立内 0.00270、実装間 0.00193。実装間は自己再実行を超えていない。

再開は公式 A と独立 A を各 2 回。更新前のパラメータ、Adam モーメント、Scheduler、4 種の RNG は checkpoint と一致し、step は 2。続けた入力ハッシュも A と B で同じ。更新後の重みハッシュは分かれ、損失は公式の再開 A/B が step 3 とも 0.5592873692512512、step 4 とも 0.505000114440918。独立も同様に A/B で同じ損失である。プロセス時間は 11.27、11.17、11.10、11.15 秒。引数を分割し損ねた失敗 4 本は 4.16、4.17、4.32、4.48 秒で、学習更新には入っていない。

### TF-GridNet

ピン留め ESPnet `0d136c545` の `pyproject.toml` は `typeguard` を版固定していない。`typeguard==4.4.4` の `typechecked` で `espnet2.bin.enh_train` を import できた。追加したのは `configargparse==1.7`、`humanfriendly==10.0`、`torch-complex==0.4.4`、`omniio==0.1.2`、`sentencepiece==0.2.2`、`fast-bss-eval==0.1.4` である。`flash_attn` は無く、ログは ESPnet 既定への後退を出してから学習している。

公式レシピは `egs2/ms_snsd/enh1` の `train_enh_tfgridnet.yaml` と `run.sh` の `sample_rate=16k`。STFT は `n_fft` 512、`stride` 256、Hann、separator は 4 層、LSTM 128。Optimizer は Adam、学習率 1e-3、weight decay 0。損失は `mr_l1_tfd` に重み 1、`si_snr` に重み 0。比較 Adapter の幅と損失はこれと同じ系統である。公開 `chunk_length` は 48000 サンプル、batch は 4、worker は 8、epoch は 35。

短時間実行は batch 1、worker 0、`num_iters_per_epoch` 2、`chunk_length` 16000、`max_epoch` 1。VoiceBank `p226_001` と `p226_002` は 48 kHz なので、公式の前処理と同じ 16 kHz へ変換してから `sound` 型の scp を公式 DataLoader に渡した。結果は `data/results/training/official-entry-tf-enh-20261104b/train.log`。1 epoch の train loss は 2.958e+03、valid loss は 2.213e+03。どちらも有限で、backward 時間、勾配 norm、Adam の `optim_step_time`、学習率 1e-3、best checkpoint の更新が記録された。プロセス 12.22 秒。その前の `fast_bss_eval` 不足は 8.93 秒で、更新前に止まっている。

### クリーン環境

Mamba の import 修正は `patches/mamba_ssm-2.3.2.post1-init.patch` と `scripts/apply_mamba_init_patch.py` にした。適用先の `__init__.py` は、クリーン環境に入っている修正済みファイルと一致する。未修正の 2.3.2.post1 へ適用し、再実行すると `already-applied` になる。selective scan の CUDA ファイルはこの処理で変わらない。直接 pin は `requirements/clean-rebuild.txt`。freeze は `build/clean-20261104/requirements-freeze.txt` を更新した。

公式幅・fused の独立 SEMamba++ は `data/results/training/semamba-fused-official-width-20261104/smoke.json`。既定幅は hid 48、num_tfmamba 4、パラメータ数 2,655,521 で、公式 `train.py` の生成器と同じ数である。fused モジュールは 24。`selective_scan_cuda` は `.venvs/clean-20261104` の `.so` を読んだ。VoiceBank を 16 kHz にした先頭 16000 サンプルで 1 更新し、生成損失 31.36404037475586、識別損失 5.976692199707031 は有限で、重みは変化した。プロセス 8.97 秒。import パスを誤った試行は 3.65 秒と 5.38 秒で、更新前に止まっている。

5 モデルの独立スモークは、同じクリーン環境の `data/results/training/clean-rebuild-20261104/gpu-smoke.log` を再利用した。その後に足した ESPnet 依存は、このスモークのモデル構築を変えない。Rust FFT と libdf も再ビルドしていない。別ノードでは実行していない。

### GPU 時間

開始 11,842.690 秒。この継続で加算したのは 223.39 秒。内訳は 4 プロセス 122.42 秒、再開の引数失敗 17.13 秒、再開成功 44.69 秒、`enh_train` の依存不足 8.93 秒、成功 12.22 秒、SEMamba++ fused の失敗 9.03 秒、成功 8.97 秒。累計 12,066.080 秒。残り 5,933.920 秒。

## Final Acceptance の再判定（2026-11-04）

HEAD は作業開始時 `082083d947276632f9a3485e7011e7d5853dc034` で、`dev` は `origin/dev` と一致し、作業ツリーはクリーンだった。学習の更新式は、MP-SENet の検証長を hop 倍数へ切る変更を除き、変えていない。その変更は検証経路だけで、長さが一致する学習 step には入らない。GPU 会計の開始値は 12,066.080 秒、新しい上限は 25,200 秒、開始時の残りは 13,133.920 秒である。

既存の学習実装 GO は取り消さない。CUDA のビット一致は、これまでどおり公式実装自身の再実行差を基準にし、この節でも要求しない。未実施の項目は GO にしない。

| 観点 | 判定 | 根拠 |
|---|---|---|
| 学習実装の公式再現性 | GO | 5 モデルとも、直前までの Structure、Forward、Loss、Backward、Optimizer、4 プロセス、同一実装の再開、複数更新が GO のままである。根拠は本書の 2026-11-03 再判定と、その後の公式プロセス節である |
| 公式学習プロセスとの対応 | PARTIAL | 5 モデルとも公式入口または公式モジュールが実音声の更新まで進んでいる。PESQ の実行形態、MossFormerGAN の再起動時に plateau 状態が初期化されること、TF-GridNet の短時間実行が batch と chunk 長を変えていること、SEMamba++ の launcher が実行回数だけを差し替えることは残る |
| Checkpoint 再開 | GO | 既存の更新前一致に加え、今回の 100 step 試験は 5 モデルとも step 50 から別プロセスで 2 更新続き、入力 case ID が連続実行と一致した。更新後の重みハッシュの一致は求めていない |
| クリーン環境の再構築 | PARTIAL | 2026-11-04 の新規 venv、Mamba、Rust FFT、libdf は同一ノードで GO である。別ノードではまだ実行していない |
| 本格学習の実行準備 | PARTIAL | 5 モデルの 100 step、有限な損失、メモリ、再開は通った。SEMamba++ の公式 batch 8 はこの MIG に載らず、DeepFilterNet3 の batch 増加は未測定で、epoch 末の検証を interval で合わせる必要がある |

残件が試験学習を止めるものではなかったので、100 step へ進んだ。別ノードの再構築、TF-GridNet の短時間公式実行が公開設定の音声長を変えていること、MossFormerGAN の公開プロセス再起動が plateau を初期化すること、比較 Adapter と公式プロセスの差、CUDA を自己再実行差で見ていることは、いずれもこの GO の範囲の外として残す。

100 step の条件、時間、メモリ、予算は [GPU_BUDGET.md](GPU_BUDGET.md)。本格学習の開始判断は [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md)。

### この節の GPU 時間

プロセスの wall clock は `/usr/bin/time` である。失敗を含む。予約時間は含めない。ジョブは `115437`、`r4n11`、H100 MIG 4g.47gb。

| 実行 | 秒 |
|---|---:|
| `src` が見えず止まった起動 | 2.28 |
| TF-GridNet 8 step の予備 | 62.72 |
| TF-GridNet 100 step | 57.12 |
| TF-GridNet 再開 | 8.30 |
| MP-SENet 100 step | 89.77 |
| MP-SENet 再開 | 9.02 |
| MP-SENet 検証（長さ不一致で失敗） | 6.04 |
| MP-SENet 検証（hop 整列のあと） | 12.39 |
| DeepFilterNet3 100 step | 212.42 |
| DeepFilterNet3 再開 | 10.40 |
| MossFormerGAN 100 step | 98.22 |
| MossFormerGAN 再開 | 11.11 |
| SEMamba++ micro 8 の OOM | 14.10 |
| SEMamba++ 蓄積 2 の 100 step | 480.27 |
| SEMamba++ 再開 | 20.99 |

合計 1,095.15 秒。開始 12,066.080 秒に加え 13,161.230 秒。上限 25,200 秒まで 12,038.770 秒。最初の 2.28 秒は GPU カーネルに入る前の失敗である。

## 読み込みの並列化と先読み（2026-11-04）

学習式は変えていない。`data.load_workers: 4` と `data.prefetch: true`、4 行以上の PESQ プロセスプールを学習設定の既定にした。根拠と step 時間は [PERFORMANCE.md](../training/PERFORMANCE.md)、予算は [GPU_BUDGET.md](../training/GPU_BUDGET.md)。

CPU では、スレッド読み込みの波形が直列と一致し、prefetch ありの 3 更新と step 1 からの再開が直列実行の重みと Optimizer に一致した。4 行の PESQ は直列スコアと一致した。VoiceBank の TF-GridNet step 2 の 4 波形も直列と workers 4 で一致した。

H100 MIG の 100 step では、5 モデルとも朝の case ID 列と一致し、step 1 の generator 損失も一致した。step 2 以降の損失は `deterministic: false` の再実行でずれる。同日の直列再実行も TF-GridNet の step 1 が 95173.195 と 95173.188 に分かれたので、このずれを先読みの系統差とはしない。公式 4 プロセスはやり直していない。

step 50 から別プロセスで 2 更新した再開は 5 モデルとも valid で、step 51 と 52 の case ID は連続実行と一致した。非有限の損失は 0 だった。

DeepFilterNet3 の batch 24 は 40 step 完了し、OOM はしなかった。定常中央値 2.200 秒、peak allocated 4.21 GiB、peak reserved 4.89 GiB。batch 16 の先読み後 1.421 秒より 1 step は長い。epoch の計算時間はほぼ同じである。batch 32 以上は未実行。

### この節の GPU 時間

`/usr/bin/time` の elapsed。Profiler 付きの 8 step も含む。予約時間は含めない。開始時の累計は 13,161.230 秒。

| 実行 | 秒 |
|---|---:|
| MIG の認識 | 39.80 |
| Profiler 5 モデル | 463.17 |
| 先読み後の 100 step 5 モデル | 550.49 |
| DeepFilterNet3 batch 24 の 40 step | 98.61 |
| 直列対照 20 step 5 モデル | 317.90 |
| 先読み後の再開 5 モデル | 68.45 |

合計 1,538.42 秒。累計 14,699.65 秒。上限 25,200 秒まで 10,500.35 秒。親 GPU の `memory.used` は他の MIG を含み、Utilization は `[N/A]` だった。計算に使ったのは、このジョブに見えている MIG 4g.47gb の 1 切片である。

## 固定 ASR と TF-GridNet の最初の 5,000 更新（2026-11-04）

HEAD は `87ad09e`、`dev` は `origin/dev` と一致、作業開始時のツリーはクリーンだった。GPU 会計の開始値は 14,699.65 秒。新しい `qsub` / `qrsh` は出していない。対話環境は `r4n11`、見えている装置は MIG `MIG-1182cb20-01d4-5444-8a31-5422d48a9c2d`。開始時の `torch.cuda.mem_get_info()` は free 46.05 GiB / total 46.38 GiB だった。親 GPU の `memory.used` は他切片を含むため、測定プロセスだけ `allow_busy_gpu` にした。

### 高速化後の短い再確認

公式 4 プロセスはやり直していない。CPU の既存テスト 3 件は成功した。スレッド読み込みの波形は直列と一致し、prefetch ありの 3 更新と step 1 からの再開は直列実行の重み・Optimizer と一致し、4 行の PESQ は直列スコアと一致した。CUDA のビット一致は要求していない。

同じ chunk 48,000・batch 4・seed 17 の新しい 24 step では、step 1 の `mr_l1_tfd` が 95173.1875 だった。続けて走らせた 5,000 更新の step 1 は 95173.195312 で、同日に記録済みの直列再実行 95173.195 と一致する側だった。この差は、先に記録した非決定的 CUDA の再実行差の範囲であり、先読みが作った新しい系統差としては扱わない。

### TF-GridNet の 5,000 更新

設定は `configs/training/tf_gridnet_seed17_epoch1.yaml`。chunk 48,000、global batch 4、16 kHz、Adam 1e-3、eps 1e-8、weight decay 0、clip 5、`mr_l1_tfd`。`max_steps: 5000`、`validation_interval: 5000`。結果は `data/results/training/tf-gridnet-seed17-epoch1-20261104/`。

直前の 24 step 予備は `data/results/training/tf-gridnet-seed17-probe-20261104/`。定常 19 点の中央値は 0.1916 秒、peak allocated 4.98 GiB、peak reserved 5.76 GiB。検証 8 件からの 872 件外挿は 46.0 秒で、発話長では層化していない。この予備から 1 epoch を 17 分程度と見積もり、30 分未満なので実行した。

5,000 更新のプロセス時間は 1,045.70 秒。train の `mr_l1_tfd` は step 1 が 95173.195、step 2,675 が 20774.313、step 5,000 が 22435.607。非有限の学習損失は 0。学習率は全 step で 0.001。検証 872 件の `mr_l1_tfd` は 22655.620 で有限。PESQ と STOI はこの損失の検証報告では `na`（`not computed for mr_l1_tfd`）。`best_generator` は `best_checkpoint_start_epoch: 40` のため保存していない。ログ上の epoch は 0 が 2,675 step、1 が 2,325 step である。

訂正：この 5,000 step を公式の `num_iters_per_epoch` と同じ更新回数とはみなせない。chunk iterator は同値を元発話数として使用し、更新数は可変である。Trainer の epoch はデータ 1 周（10,700 / 4 = 2,675）のままで、公式の 5,000 反復イテレータではない。step 5,000 は 2 周目の途中で、ReduceLROnPlateau は `last_epoch` 0、`best` inf のままである。学習率は公式の 1 epoch 中と同じ 0.001 だが、epoch 境界と patience の状態は公式と対応していない。35 epoch へこの checkpoint をそのまま続けても、検証 step とデータ周の終端は 175,000 step 以内で重ならず、plateau は進まない。

step 5,000 の checkpoint を別ディレクトリで 2 更新再開した。`data/results/training/tf-gridnet-seed17-epoch1-resume-20261104/`。step 5,001 と 5,002 の損失は有限、学習率は 0.001、プロセス 11.16 秒。元の checkpoint は上書きしていない。

### Whisper Large v3

`.venvs/asr` はこのノードに無い。`.venvs/clean-20261104` に `openai-whisper==20240930`、`tiktoken==0.7.0`、`more-itertools==10.5.0` を足した。torch 2.6.0+cu124 と numpy 2.5.4 は変えていない。

評価セットは `resource.duckdb` の `voicebank_demand_test`、登録順の先頭 4 発話、48 kHz、全文、`max_seconds` なし。ASR は `large-v3`、`language: en`、`task: transcribe`、`temperature: 0`、`condition_on_previous_text: false`、fp16。WER は `wer-norm-v1` の micro。PESQ は 16 kHz wideband、STOI も 16 kHz へ変換してから計算する。clean は参照なので PESQ / STOI / SI-SDR の推定値には入らない。

| checkpoint | 条件 | micro WER | N | PESQ | STOI | SI-SDR |
|---|---|---:|---:|---:|---:|---:|
| 100 step `perf-fast-20261104` | clean | 0 | 54 | na | na | na |
| 100 step | noisy | 0 | 54 | 2.531 | 0.930 | 8.844 |
| 100 step | enhanced | 0 | 54 | 1.660 | 0.902 | 5.583 |
| 5,000 step | clean | 0 | 54 | na | na | na |
| 5,000 step | noisy | 0 | 54 | 2.531 | 0.930 | 8.844 |
| 5,000 step | enhanced | 0 | 54 | 2.808 | 0.939 | 19.542 |

4 発話とも仮説は参照と一致した。WER はこの子集では条件を分けない。100 step の音質は noisy を下回り、5,000 step の 4 発話平均は上回った。どちらも学習済み性能ではない。記録は `data/results/se-evaluation/whisper-vb4-20261104/` と `whisper-vb4-epoch1-20261104/`。

5,000 step の checkpoint では、4 発話の enhance 0.55 秒、ASR 8.26 秒、PESQ 等 6.08 秒だった。発話あたり 3.96 秒を 824 件へ比例すると約 0.91 時間。このうち enhance と ASR は約 0.50 時間。先頭 4 件の比例であり、長さで層化していない。重みの初回取得 2.88 GB は別である。

### この節の GPU 時間

`/usr/bin/time` の elapsed。予約時間は含めない。失敗した最初の予備は、計測ファイルが checkpoint ディレクトリを空でなくしたため更新前に止まり、シェル全体 13.57 秒を計上した。

| 実行 | 秒 |
|---|---:|
| 空でないディレクトリで止まった予備 | 13.57 |
| chunk 48,000 の 24 step と検証 8 件 | 15.82 |
| TF-GridNet 5,000 更新と検証 872 件 | 1,045.70 |
| step 5,000 からの 2 更新 | 11.16 |
| Whisper、100 step checkpoint、4 発話 | 59.50 |
| Whisper、5,000 step checkpoint、4 発話 | 43.77 |

合計 1,189.52 秒。累計 15,889.17 秒。上限 25,200 秒まで 9,310.83 秒。


## 2026-11-04 公式 chunk iterator の再確認

ESPnet `ChunkIterFactory` は `num_iters_per_epoch` を元発話数として使用し、重複チャンクから可変数の batch を作る。前節の 5000 更新を公式 1 epoch とする解釈は撤回する。固定 5000 更新の研究用設定と CPU 回帰を追加したが、公式 epoch、best 5、早期停止は未対応。詳細は [監査](OFFICIAL_TRAINING_AUDIT.md)。今回 GPU 実行なし。

固定更新モードの CPU 回帰は `python -m unittest tests.training.test_logical_epoch -q`。3 ケース・batch 1・4 更新/境界で 12 更新し、検証 4/8/12、scheduler 3 回、best loss 保存、step 2/4 の再開後の入力 ID、重み、optimizer を通じた学習、scheduler 状態、RNG の保存キーを確認した。模擬 task であり、実モデル・CUDA・別プロセスの新モード検証ではない。既存 `tests.training.test_mp_senet_train` と `tests.training.test_model_tasks` は 27 件成功。固定更新モードの 5,000 更新 GPU 再実行はしていない。

## 研究用本格学習の最終確認（2026-11-04）

前節の「公式 epoch は未対応、GPU 実行なし」は、その時点の記録である。chunk iterator、best、early stopping はその後実装済みで、この節が実長と短い GPU を足す。本格学習は開始していない。既存の結果ディレクトリは上書きしていない。`qsub` は出していない。

### TF-GridNet

`data.rms_normalize: false` は公式モードだけに入る。長さは変わらず、スケール後の noisy の平均二乗は 1、無効時の波形はレンダリング結果と一致する。identity は RMS ありのときキーを省き、なしのとき `false` を残す。CPU の公式 iterator 比較は、合成データで ID・位置・波形スライス・バッチ順がピン留め ESPnet と一致したままである。

実長は WAV ヘッダから求めた。48 kHz は `ceil(frames/3)`。train の 30 件を `_load` と照合し、不一致は 0 だった。epoch 1–5 の optimizer 更新は 575、574、593、566、588。端数チャンクの破棄は 2、2、3、3、3。検証は 85 batch。記録は `data/results/training/go-20261104/tf-epoch-count.json`。

GPU は同じ MIG で 4 更新した。設定は `data/results/training/go-20261104/tf-chunk-smoke.yaml`。長さキャッシュをヘッダから入れ、全件デコードで GPU を占有しないようにした。step 1–4 の case ID は、同じ長さで独立に組んだ計画の先頭 4 batch と一致した。計画全体は 575 batch で、epoch 集計と一致する。損失は有限で、step 1 の `mr_l1_tfd` は 9825.908、step 4 は 7908.896。学習率は 0.001。checkpoint は step 2 と step 4。peak allocated は 5,351,750,656 バイト。プロセス 22.89 秒。

別プロセスの再開は `tf-chunk-smoke-resume.yaml` と `step_00000002.pt`。step 3 と 4 の case ID は連続実行と一致した。step 3 の損失は 8086.077 で一致し、step 4 は 7908.896 と 7908.890 に分かれた。`deterministic: false` の再実行差として残す。プロセス 18.74 秒。検証は epoch 末だけなので、この 4 更新では走っていない。

### DeepFilterNet3

batch 32 の定常中央値は 3.471 秒、batch 64 は 7.025 秒。どちらも OOM していない。詳細は [PERFORMANCE.md](PERFORMANCE.md)。等長の発話平均はパディングなしの batch 平均と一致し、長短をパディングしたときは一致しない。研究用検証は発話ごとの非加重平均に固定した。`max_steps` は 22711。batch 変更後の cosine 添え字は、既存の CPU テスト `test_batch_schedule_changes_size_and_cosine_index` が通る。GPU ではその制御を繰り返していない。

### SEMamba++

長さ 8,000、fused、`deterministic: false`、seed 17。batch 8 の自己再実行は音声最大差 0。蓄積 2 との音声最大差は 0.001338273286819458。forward だけの同じ比較もこの値だった。generator 勾配の相対 L2 は自己再実行 2.90e-4、蓄積 0.0282。更新後重みの相対 L2 は 1.39e-5 と 2.01e-4。勾配 norm は約 16.00 と 16.02。記録は `data/results/training/go-20261104/semamba-accum-cuda.json`。3 秒の batch 8 は以前の OOM のままである。研究設定は micro 4 × 蓄積 2 で、`configs/training/semambapp_seed17.yaml` に差を書いた。100 step と再開は既存の結果を使う。

### MP-SENet と MossFormerGAN

`configs/training/mp_senet_seed17.yaml` と `configs/training/mossformergan_seed17.yaml` は、試験で通った更新条件のまま epoch 数だけを本学習へ伸ばしている。発話 RMS は既定のままである。Trainer が case ID をログへ足したことは、更新式を変えない。大規模な再比較はしていない。

### 受入

| モデル | 公式学習再現性 | 研究用本格学習 | 未解決事項 |
|---|---|---|---|
| MP-SENet | GO | GO | 切り出し順は case id 決定的。公式 DataLoader の消費順とは別 |
| MossFormerGAN | GO | GO | 公開プロセスの再起動は plateau を初期化する。独立 checkpoint は保持する |
| SEMamba++ | PARTIAL | GO | 3 秒の batch 8 は未実施。研究設定との差は上の数値 |
| TF-GridNet | PARTIAL | GO | データは VoiceBank。MS-SNSD の `enh_train.py` 35 epoch は未起動。秒/step は形状等価 |
| DeepFilterNet3 | PARTIAL | GO | 検証集約は公式 batch と別。オンライン混合は使わない。検証 227 秒は古い外挿 |

研究用本格学習の GO は、設定が凍結され、実モデルの更新と再開が通り、メモリに収まり、時間が上の式で見積もれる、という範囲である。35 epoch や 120 epoch の完走はまだない。
