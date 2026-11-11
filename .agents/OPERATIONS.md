# 環境とテスト

依存の正本は `requirements/<role>.lock.txt`。`make setup` は DB を初期化しない。

| 環境 | 用途 |
|---|---|
| `.venvs/se` | Python 3.12、torch 2.6.0+cu124。SE、評価、実験、通常のテスト |
| `.venvs/asr` | NeMo、Whisper、Transformers。実 ASR |
| `.venvs/official` | 公式実装との比較 |
| `.venvs/semambapp` | SEMamba++ の fused CUDA |
| `.venvs/deepfilternet` | 公式 `libdf`。独立実装の実行時依存ではない |

`src/se/official/` は Git 管理外。独立実装はここを import しない。テストの公式比較だけが読む。

## GPU

- 使う前に `nvidia-smi` を見る。実験 runtime は CUDA を要求したとき、空いていない GPU を `PrepareError` にする。
- `runtime.kwargs.allow_busy_gpu: true` はその装置が自分のジョブだと分かっているときだけ。
- `CUDA_VISIBLE_DEVICES` で隠すと、見える番号は 0 から振り直される。評価と学習の空き確認は、未設定と空文字をどちらも「見える番号を物理番号のまま見る」として扱う。UUID と MIG の文字列は、別の物理 GPU の番号に読み替えない。
- CPU テストは `CUDA_VISIBLE_DEVICES=""` で実行する。CPU の成功を GPU テストの PASS にしない。

```bash
make experiment CONFIG=configs/reproduction/experiments/official/mp_senet_gan.yaml
make experiment-preflight CONFIG=tests/fixtures/configs/experiments/mp_senet_gan.yaml
make test-cpu
```

`make experiment` は学習を始める。preflight は Optimizer を更新しない。

## テスト

`make test-cpu` は下の 7 ディレクトリを CPU だけで実行する。`make test` は同じ CPU 実行に加え、親プロセスの `CUDA_VISIBLE_DEVICES` の範囲で空いている GPU があるときだけ GPU のテストも実行する。変数が未設定なら全物理 GPU、空文字なら GPU レーンは NOT_RUN で、CPU テストは実行する。CPU 側は `CUDA_VISIBLE_DEVICES=""` なので、CPU の成功は GPU の成功ではない。GPU が無い、nvidia-smi が使えない、または範囲内に 1024 MiB 未満かつ計算プロセスの無い GPU が無いときは、その GPU テストは NOT_RUN であり PASS ではない。

モジュールはプロセスプールで並列になる。同時数は `AUDIO_DB_TEST_WORKERS`、プロセス内スレッドは `AUDIO_DB_TEST_THREADS`。未指定なら、affinity と cgroup の CPU 上限のうち小さい方から決める。`os.cpu_count()` は、そのどちらも取れないときの代替である。明示したワーカー数・スレッド数・`AUDIO_DB_DFN3_WORKERS` は、その数をそのまま使う。スレッドの既定上限は 8 で、全コアを 1 プロセスに渡すと実チェックポイントの評価が遅くなるためである。GPU レーンも走るときは、CPU ワーカーと GPU スロットでこのスレッド幅を分ける。`deterministic: true` は従来どおり `torch.set_num_threads(1)` にする。使う GPU の物理番号は `AUDIO_DB_TEST_GPUS` で、親の可視範囲の外は選ばない。この明示リストは空き確認を省く。整数と `GPU-` UUID は物理 GPU で、MIG やそれ以外の表記は別の GPU に読み替えず、GPU レーンを NOT_RUN にする。見える GPU が 1 枚のとき、2 枚必要なテストは NOT_RUN である。同一 GPU 群で同時に走らせる GPU モジュール数は `AUDIO_DB_TEST_GPU_SLOTS` で、既定は 1。DeepFilterNet3 の分布確認のプロセス数は `AUDIO_DB_DFN3_WORKERS`。未指定でスイートから走っているときは、利用可能な CPU を `AUDIO_DB_TEST_WORKERS` で割った数にする。各プロセスは 1 スレッドである。1 モジュールの上限秒は `AUDIO_DB_TEST_TIMEOUT`（既定 1800）。`tests/data/reproduction/test_p1.py`、`test_p9_semamba.py`、`tests/experiments/test_phase4i_integration.py` は `data/db` の本番ファイルを書くか、そのディレクトリ名を見る。この 3 つは同時に走らない。`test_p9_semamba.py` と `tests/experiments/test_semambapp.py` も同時に走らない。p9 が smoke snapshot を公開する前に、semambapp がそのディレクトリを走査するからである。GPU レーンは選んだ物理番号を `CUDA_VISIBLE_DEVICES` と `AUDIO_DB_CUDA_GPU0` / `AUDIO_DB_CUDA_GPU1` の両方に入れる。CUDA の子プロセスは後者を自分の `CUDA_VISIBLE_DEVICES` に書き戻す。

ディレクトリ単体は今までどおり次のコマンドでよい。この経路はプールしない。スレッドは起動前の `OMP_NUM_THREADS` に従い、実モデルの評価スモークは自分でも上限を掛ける。

```bash
CUDA_VISIBLE_DEVICES="" .venvs/se/bin/python -m unittest discover -s tests/se -v
CUDA_VISIBLE_DEVICES="" .venvs/se/bin/python -m unittest discover -s tests/asr -v
CUDA_VISIBLE_DEVICES="" .venvs/se/bin/python -m unittest discover -s tests/data -v
CUDA_VISIBLE_DEVICES="" .venvs/se/bin/python -m unittest discover -s tests/evaluation -v
CUDA_VISIBLE_DEVICES="" .venvs/se/bin/python -m unittest discover -s tests/experiments -v
CUDA_VISIBLE_DEVICES="" .venvs/se/bin/python -m unittest discover -s tests/numerics -v
CUDA_VISIBLE_DEVICES="" .venvs/se/bin/python -m unittest discover -s tests/setup -v
```

`tests/numerics/` は損失、STFT、scheduler、参照ステップの数値比較。`tests/experiments/` は組み立て、checkpoint、6 モデルの学習ループ。再現 DB とスナップショットのテストは `tests/data/reproduction/` にあり、`tests/data` の discover がまとめて実行する。チェックポイントや公式ツリーが無いテストは skip する。SKIP は PASS ではない。

CUDA が要るテストは、見える GPU が無いとき、理由が `NOT_RUN` で始まる skip にする。ランナーはそれを SKIP と分けて NOT_RUN と数える。CUDA / NCCL の `tests/experiments/test_cuda_amp.py` と、GPU レーンに入る実モデルのテストがこれに当たる。見えるのが 1 枚のときは、2 枚必要なテストだけ NOT_RUN になる。CPU スイートの成功には数えない。`make test` は、親の可視範囲で空いている GPU の物理番号を `CUDA_VISIBLE_DEVICES` に並べ、このファイルと、GPU がないと実行できない実モデルの少数のテストを別レーンで実行する。評価の GPU 使用前検査は、見える番号を `nvidia-smi` の物理番号へ戻してからメモリを見る。空の `CUDA_VISIBLE_DEVICES` では番号を物理番号のまま見る。

## 検証の区別

| 呼び方 | 言えること |
|---|---|
| Mock | 配線、集計、再開、設定の拒否 |
| 実モデルのスモーク | その環境と入力で形状と有限値が正しい |
| 公式一致 | 記録したモデル、長さ、sample rate、device、seed での最大絶対誤差 |

未計測の条件での一致や、論文スコアの再現とは書かない。
