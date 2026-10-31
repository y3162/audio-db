# 開発環境と検証

環境の構築手順・固定バージョン・実行前の環境変数は `docs/envs/LAB.md`、`docs/envs/ASR.md`、`docs/envs/TSUBAME.md`。検証の記録は `docs/se/VALIDATION.md`、`docs/se/IMPLEMENTATION_STATUS.md`、`docs/asr/MODELS.md`。

## 仮想環境

| 環境 | 用途 |
|---|---|
| `.venvs/lab` | Python 3.13、torch 2.6.0+cu124、NumPy 2.5.4、DuckDB 2.0.0。SE、評価、`make db-init`、通常のテスト |
| `.venvs/asr` | Python 3.12、NeMo、Whisper、Transformers、NumPy 2.1.3。実 ASR を使う評価・スモーク。SE と ASR を同じプロセスでつなぐため、DuckDB・PESQ・STOI も入っている |
| `.venvs/ref`、`.venvs/semamba`、`.venvs/cuda-nvcc` | 公式実装との比較とそのためのビルド専用。公式の依存を含む。`semamba` は `mamba-ssm` と `causal-conv1d` を sm_75（turing の Quadro RTX 5000）向けにソースビルドしてある。ライブラリ本体はこれらを import しない |

- NeMo は Python 3.13・NumPy 2.5.4 の lab と依存関係が合わないため、ASR の環境を分けている。NeMo などを lab に入れない。lab にパッケージを足すときは NumPy を上げない（`requirements/evaluation.txt` の手順）。
- Numba と NeMo は、`.venvs/*` の `nvidia/cuda_nvcc`（ptxas 12.2）を使う。システムの `/usr/local/cuda` は使わない。
- TSUBAME 4.0 は別の環境（`docs/envs/TSUBAME.md`）。

## サーバーと GPU

- 研究室サーバーは ampere（RTX 3090 ×4）、tensor（A100 40GB ×7、80GB ×1）、turing（Quadro RTX 5000 ×4）、shannon（A100 80GB ×8）。`.venvs/lab` は共通。
- GPU は他のユーザーも使う。使う前に `nvidia-smi` で空きを確認し、使用メモリが 1024 MiB 未満の GPU だけを指定する。評価 CLI は、使用メモリが 1024 MiB 以上の CUDA GPU を拒否する。`runtime.allow_busy_gpu: true` は、その GPU が自分のジョブだと分かっているときだけ使う。
- `CUDA_VISIBLE_DEVICES` で隠すと、見える GPU の番号は 0 から振り直される。`device` の指定はその番号。
- 設定やスモークに書かれた `cuda:1`（`configs/evaluation/e2e.yaml`、`tests/asr_smoke/run_smoke.py`）は固定の例。そのまま実行せず、空きを確認する。多くの試験は CPU で足りる。

## テスト

```bash
.venvs/lab/bin/python -m unittest discover -s tests/se -v          # SE
.venvs/lab/bin/python -m unittest discover -s tests/asr -v         # ASR。重みをダウンロードしない
.venvs/lab/bin/python -m unittest discover -s tests/evaluation -v  # 評価。実 DB への冪等な登録テストを含む（DATA.md）
.venvs/lab/bin/python -m src.evaluation.cli --config configs/evaluation/smoke.yaml --limit 2 --device cpu   # 合成音声 + mock
.venvs/asr/bin/python -m tests.asr_smoke.run_smoke parakeet whisper wav2vec2   # 実 ASR のスモーク（GPU）
```

- チェックポイントや公式コードが無いテストは skip される。SKIP は PASS ではない。
- 実 SE と実 ASR をつなぐ接続確認は `configs/evaluation/e2e.yaml`。`.venvs/asr` で実行する。

## 公式コードと独立実装

- `src/se/official/` は公式実装の置き場。Git 管理外で、将来的に削除する予定（未実施）。
- 現在、公式一致の比較（`tests/se/reference/`、`tests/se/test_mp_senet.py`、`tests/se/test_tf_gridnet.py`）がこれを参照する。削除するときは、これらの扱いと `docs/se/VALIDATION.md` を同時に決める。参照を前提とする新しいコード・テスト・設定を増やさない。
- 独立 SE 実装は、公式コードに実行時依存しない。`tests/se/test_independence.py` は、`src/se/official/` を除いたコピーで MP-SENet と TF-GridNet が推論できることを確認する（ほかのモデルは範囲外）。`make db-init` も `official/` を読まない。
- ESPnet 全体をコピーしない。TF-GridNet は、推論に要る GridNet ブロックと正規化層だけを、出典 commit とライセンスを付けて引用している。ライセンスが不明なソース（SEMamba++）はコピーしない。
- 新しいモデルを足すときは、出典 commit とライセンスを `docs/se/MODELS.md` に記録し、公式出力との比較を行う。
- チェックポイントのパスをコードに固定しない。呼び出し側が渡す。

## 検証の区別

| 呼び方 | 内容 | 言えること | 言えないこと |
|---|---|---|---|
| Mock | `name: mock`。SE は入力のコピー、ASR は試験用 | 配線・集計・再開・設定検証が通る | 実モデルが動くこと、性能 |
| 実モデルの推論スモーク | 実チェックポイントで短い入力が完了し、形状と有限値が正しい | その環境と入力でモデルが動く | 公式と一致すること、論文用のスコア |
| 公式一致検証 | 公式実装と公式チェックポイントの最終波形を、同じ入力・seed で比べ、最大絶対誤差を記録する | 記録した条件（モデル・長さ・サンプリングレート・device・seed）での一致 | 未計測の条件での一致 |

- 独立実装の CPU と CUDA の差は、公式との差ではない。公式の一致の証拠に使わない。
- ASR の「動作検証」は `docs/asr/MODELS.md` の表が正本。カタログに載っていることは検証済みを意味しない。
- 検証を報告するときは、何を（モデル、入力、device、seed、環境）確認したかを書く。
