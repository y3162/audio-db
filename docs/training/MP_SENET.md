# MP-SENet の学習

学習の入口は `python -m src.training.se.cli --config <yaml>` である。Generator は `src.se` の `build_model("mp_senet")` を使う。損失、Metric Discriminator、更新順序は `src/training/se/` にあり、`enhance()` は呼ばない。`src/se/official/` は実行時に import しない。

参照した公式コードは commit `89932cfe90d1dacb8e170e4a331d762462c21792`（`src/se/official/mp-senet/` の `train.py`、`dataset.py`、`models/discriminator.py`、`models/model.py` の `phase_losses`、`config.json`）である。

## 公式と一致している処理

- Generator は振幅と位相を受け、マスクされた振幅、位相、圧縮複素スペクトルを返す。
- Metric Discriminator は参照振幅と推定振幅を channel 方向に重ね、spectral norm 付き Conv、InstanceNorm、PReLU を 4 段、全体最大値、Dropout 0.3、LearnableSigmoid で 1 次元のスコアを出す。
- STFT は Hann、`n_fft=400`、`hop=100`、`win=400`、`center` の reflect pad、振幅は `sqrt(power + 1e-9)` のあと `compress_factor` 乗、位相は `atan2(imag + 1e-10, real + 1e-5)`。iSTFT は圧縮を戻してから Hann で戻す。
- 損失の係数は公式 `train.py` のままである。振幅 MSE × 0.9、位相（瞬時・群遅延・瞬時角周波数）× 0.3、複素 MSE × 2 × 0.1、STFT 整合 MSE × 2 × 0.1、時間 L1 × 0.2、Discriminator を 1 に近づける MSE × 0.05。
- Discriminator は実ペアを 1 へ、生成振幅を wideband PESQ の `(score - 1) / 3.5` へ近づける。PESQ が失敗したバッチは生成側の項を足さない。
- 更新順は Discriminator、そのあと Generator。Generator の metric 項は Discriminator を通すが、Discriminator の重みは凍結する。
- Optimizer は AdamW、学習率 `5e-4`、betas `(0.8, 0.99)`、weight decay `0.01`（公式が引数を省略したときの PyTorch 既定）。Scheduler は `ExponentialLR`、`gamma=0.99` で、エポックの学習ステップが終わってから 1 回進む。
- 入力は noisy のエネルギーで正規化し、その係数を教師にも掛ける。短い区間は 0 で `segment_size` まで埋める。正規化は切り出しの前である。
- 複数 GPU は DistributedDataParallel。勾配を累積する間は最後の micro-batch 以外で `no_sync()` を使う。

`loss: reconstruction` は時間波形の L1 だけである。これは基盤テスト用で、公式の学習再現ではない。

## 意図的に変えた処理

| 項目 | 公式 | ここ |
|---|---|---|
| バッチ | `batch_size` を GPU 数で整数除算し、余りは捨てて実効バッチが変わる | `global_batch_size = micro × world_size × accumulation` を厳密に保つ。割れない構成はエラー。micro は上限以下の最大の約数。Global step は Optimizer 更新の回数で、Scheduler とは別 |
| 再現性 | `cudnn.benchmark = True`。seed は CUDA に一度置く | `deterministic: true` で `use_deterministic_algorithms(True)`、`cudnn.benchmark = False`、`cudnn.deterministic = True`、TF32 オフ。`CUBLAS_WORKSPACE_CONFIG=:4096:8` |
| STFT の reflect | `torch.stft(center=True, pad_mode="reflect")` | 同じ値になる index による reflect のあと `center=False`。CUDA の `reflection_pad1d` 逆伝播に決定的実装が無い |
| 全体最大 | `AdaptiveMaxPool2d(1)` | `amax`。順伝播の値は一致する。CUDA の `adaptive_max_pool2d_backward` に決定的実装が無い |
| 切り出し | `random.randint` | `SHA-256(seed, epoch, case_id)`。呼び出し順に依存しない |
| シャッフル | 初期化時の `random.seed(1234)` と DistributedSampler | `(seed, epoch)` から作り直す順列。末尾の端数は rank 間で揃うように捨てる |
| DataLoader | `num_workers=4` | プロセス内で読む。worker の乱数を増やさない |
| PESQ | `joblib` のプロセスプール | 同じ `pesq(..., "wb")` を入力順に計算する |
| Checkpoint | `g_*` と `do_*` を別ファイル。RNG と Scheduler は保存しない | 1 ファイルにモデル、Optimizer、Scheduler、global step、epoch、消費済み micro-batch、RNG、設定、環境、データセット識別子。一時ファイルへ書いてから rename する。推論用に `generator` キーの `generator.pt` も書く |
| データ | 公式ファイルリストの clean/noisy wav | DuckDB の名前付きセットとレシピ。教師は `output.reference`。`none` と評価用セットは拒否する |
| 起動 | `mp.spawn` と設定ファイル内の GPU 数 | CPU はプロセスグループなし。GPU は `CUDA_VISIBLE_DEVICES` と `torchrun`。物理 GPU 番号はコードに書かない |

モデル初期化の乱数、dropout の乱数、データの順列、切り出しは別々に種を付ける。切り出しと順列はグローバルな PyTorch 乱数を消費しない。

## 検証と best checkpoint

検証は rank 0 だけが、勾配を切って評価モードで計算する。終わったら学習モードに戻し、学習用の乱数状態は検証の前に保存して戻す。指標を出せないときは 0 ではなく `na` または `unavailable` と理由を残す。PESQ と STOI はレシピの教師波形を参照にする。

公式は epoch 40 以降、検証 PESQ が改善したときだけ Generator を `g_best` に保存する。ここも `best_checkpoint_start_epoch`（既定 40）以降で、検証 PESQ の平均が改善したときだけ `best_generator.pt` を書く。中身は `generator` キーだけで、`load_enhancer()` が読む。ステップ、指標、設定は隣の `best_generator.json` に書く。再開用の `step_XXXXXXXX.pt` とは別である。

学習ログは `checkpoint_dir/metrics.jsonl` である。先頭は run レコード、その後に step と validation のレコードが続く。再開時は、checkpoint の global step より後の行を消してから続ける。DuckDB には書かない。

## 未実装

- TensorBoard。
- VoiceBank-DEMAND 全件での学習と、論文のスコア比較。
- 公式 Discriminator checkpoint の読み込み検証。
- 複数ノード。
- 公式学習ステップとの数値比較（L3）。
- ASR の学習、ASR 特徴の損失。

## 学習データ

`data.database` はレシピスキーマの DuckDB である。`data/db/resource.duckdb` は評価セットの本番ファイルなので、名前でも拒否する。`train_set` と `validation_set` は別名にする。

教師が `dry_clean` または `reverberant_clean` のケースだけを使う。`none` はエラーにする。`dry_clean` を残響教師の代わりには使わない。ケースの識別子はレシピの SHA-256 で、採番 ID は使わない。

次の名前は学習にも検証にも使わない。`example`、`libri_demand_snr0`、`libri_testclean_full_smoke`、`chime3_et05_real_ch0`、`chime3_et05_real_ch5`、`chime3_et05_simu_ch5`、`chime3_tr05_org`、`voicebank_demand_test`。LibriSpeech test-clean と CHiME3 et05 は、セット名が違ってもデータセット名と split から拒否する。学習 split に `test` / `valid` / `dev` は使わない。

入力と教師の長さまたはサンプルレートがレシピの出力と一致しないときはエラーにする。無音の noisy もエラーにする。

## 実行

CPU:

```bash
.venvs/lab/bin/python -m src.training.se.cli --config configs/training/mp_senet.yaml
```

空き GPU を `nvidia-smi` で確認し、使用メモリが 1024 MiB 未満のものだけを見せる。例は物理 GPU 1 と 2 が空いているときである。

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_VISIBLE_DEVICES=1,2
.venvs/lab/bin/python -m torch.distributed.run --nproc_per_node=2 \
  -m src.training.se.cli --config configs/training/mp_senet.yaml
```

`configs/training/mp_senet.yaml` の `CHANGE_ME_TRAIN` は置き換える。評価セットの名前のままでは拒否される。チェックポイントは `checkpoint_dir` で、`data/checkpoints/`、`data/corpora/`、`data/db/`、`data/staging/` には書けない。再開は `--resume path/to/step_XXXXXXXX.pt`。

追加の Python パッケージは作っていない。`.venvs/lab`（PyTorch 2.6.0+cu124、PyYAML、pesq）を使う。

テスト:

```bash
.venvs/lab/bin/python -m unittest discover -s tests/training -v
```

確認した数値は [VERIFICATION.md](VERIFICATION.md)。
