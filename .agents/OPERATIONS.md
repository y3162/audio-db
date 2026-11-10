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
- `CUDA_VISIBLE_DEVICES` で隠すと、見える番号は 0 から振り直される。
- CPU テストは `CUDA_VISIBLE_DEVICES=""` で実行する。CPU の成功を GPU テストの PASS にしない。

```bash
make experiment CONFIG=configs/reproduction/experiments/official/mp_senet_gan.yaml
make experiment-preflight CONFIG=tests/fixtures/configs/experiments/mp_senet_gan.yaml
make test-cpu
```

`make experiment` は学習を始める。preflight は Optimizer を更新しない。

## テスト

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

CUDA / NCCL の `tests/experiments/test_cuda_amp.py` は、見える GPU が無いとき NOT_RUN になる。CPU スイートの成功には数えない。

## 検証の区別

| 呼び方 | 言えること |
|---|---|
| Mock | 配線、集計、再開、設定の拒否 |
| 実モデルのスモーク | その環境と入力で形状と有限値が正しい |
| 公式一致 | 記録したモデル、長さ、sample rate、device、seed での最大絶対誤差 |

未計測の条件での一致や、論文スコアの再現とは書かない。
