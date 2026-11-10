# audio-db

固定 ASR の WER を主指標として音声強調モデルを学習・評価する研究基盤。詳細の入口は [ドキュメント案内](docs/INDEX.md)、開発上の制約は [AGENTS.md](AGENTS.md)。

## セットアップ

```bash
cp .env.example .env
make setup PLATFORM=tsubame
make doctor
```

SHINODA-LAB では `PLATFORM=shinoda_lab` にする。仮想環境は `.venvs/se`、`.venvs/asr`、`.venvs/official`、`.venvs/semambapp`、`.venvs/deepfilternet` で、プラットフォーム名は入らない。手順は [セットアップ](docs/envs/SETUP.md)。

## 音声強調

`available_enhancers()` は MP-SENet、TF-GridNet、DeepFilterNet3、MossFormerGAN、SEMamba++ を返します。チェックポイントパスを渡します。公式波形との対応と、CLI のファイル出力など波形以外で未確認の範囲は `docs/se/VALIDATION.md` にあります。

```python
from src.se import available_enhancers, build_model, load_enhancer

print(available_enhancers())
model = build_model('mp_senet')  # 未学習の nn.Module
enc = load_enhancer('mp_senet', checkpoint='/path/to/g_best_vb', device='cpu')
enhanced = enc.enhance(noisy, sample_rate=16000)
```

設定は辞書または JSON で渡せます。詳細と未対応事項は [設計](docs/se/ARCHITECTURE.md)、[モデル](docs/se/MODELS.md)、[検証](docs/se/VALIDATION.md) を参照してください。

TF-GridNet は以下の variant を選択できます（現時点では検証途中です）。

```python
model = build_model('tf_gridnet', variant='urgent24')
enhancer = load_enhancer('tf_gridnet', checkpoint='/path/to/21epoch.pth', variant='urgent24')
model_ms = build_model('tf_gridnet', variant='ms-snsd')
```

可変長バッチは `enhancer.enhance_batch(padded_waveforms, 16000, lengths)` で処理できます。

## 学習

SE の学習は `src/experiments` だけである。`enhance()` は推論のままである。公式再現の本格候補は `configs/experiments/official/`、研究用は `configs/experiments/research/`。式と再開条件は [.agents/ARCHITECTURE.md](.agents/ARCHITECTURE.md)。

```bash
make experiment-preflight CONFIG=configs/experiments/mp_senet_gan.yaml
make experiment CONFIG=configs/experiments/official/mp_senet_gan.yaml
```

## 公式再現用 DB

`data/db/reproduction.duckdb` は評価用の `resource.duckdb` とは別のファイルです。P1 では空の `repro` スキーマだけを作ります。実コーパスの Dataset はまだ登録しません。

```bash
make reproduction-doctor
make reproduction-init-db
make reproduction-verify DATASET=name@version HASH=1
make reproduction-snapshot DATASET=name@version
make reproduction-rebuild-db OUT=data/db/reproduction-rebuilt.duckdb
```

`DATASET` が無い snapshot、`OUT` が無い rebuild、`DATASET` と `BUILD` のどちらも無い verify は失敗します。同じスキーマへの `reproduction-init-db` は何もしません。違うスキーマや未完了のファイルは変更せず拒否します。再構築は `OUT` が既存なら拒否します。leakage 監査が `UNKNOWN` の Dataset を書き出すときだけ `ALLOW_UNKNOWN_LEAK_CHECK=1` を付けます。このフラグはスナップショットの来歴に残り、ハッシュには入りません。

学習プロセスは `reproduction.duckdb` を開かない。スナップショットを読む。

## 評価データ

評価の入力は DuckDB の名前付きセットです。Manifest は登録するときだけ読み、評価の実行時には使いません。`make db-init` は `eval` スキーマを作り、`configs/db/init.yaml` の対象を登録します。`corpora.audio_files` は削除も再作成もしません。

```bash
make db-init
```

```python
from src.evaluation.datasets import EvaluationDB

db = EvaluationDB("data/db/resource.duckdb")
dataset = db.load_dataset("libri_demand_snr0")
for sample in dataset:
    ...
```

コーパスからセットを作るときも Manifest は要りません。

```python
db.create_set_from_corpus(
    "libri_demand_snr0",
    role="synthetic",
    filters={"corpus": "LibriSpeech", "subset": "test-clean"},
    limit=2,
    noise_filters={"corpus": "DEMAND"},
    noise_limit=1,
    snr_db=0,
    seed=0,
    max_seconds=2.0,
)
```

同じレシピからは同じ `case_id` と、同じ波形が再生成されます。条件の形は [レシピ](docs/db/RECIPE.md)、取り出し方は [評価パイプライン](docs/evaluation/PIPELINE.md) と [DB の準備](docs/db/SETUP.md) です。

`data/db/resource.duckdb` はレシピスキーマです。`make db-init` は同じ内容を足さず、`corpora.audio_files` も変えません。切り替え前のファイルは `data/db/backups/resource-legacy-20261031.duckdb` です。

## 音声認識

Parakeet、Whisper、Wav2Vec2、HuBERT、WavLM は `src.asr.load_recognizer()` から使います。モデル本体は `src/asr/` にあり、評価パッケージには依存しません。一覧と検証状況は [ASR](docs/asr/MODELS.md)、環境は [ASR 環境](docs/envs/ASR.md) です。

```python
from src.asr import load_recognizer

asr = load_recognizer(
    "whisper",
    model_id="tiny",
    device="cpu",
    language="en",
)
result = asr.transcribe(waveform, sample_rate=16000)
```

評価設定では `asr.name` と `asr.model_id` を指定します。`mock` はテスト用です。

## 評価パイプライン

データセット、SE、ASR、指標は差し替えられます。実コーパスや実チェックポイントが無いときは、合成音声と `mock` で一連の処理を確認できます。`mock` は音声強調モデルではありません。設計と SNR の定義は [評価パイプライン](docs/evaluation/PIPELINE.md) にあります。`rms_v1` は VoiceBank-DEMAND や URGENT の混合手順ではありません。

```bash
python -m src.evaluation.cli \
  --config configs/evaluation/smoke.yaml \
  --dry-run

python -m src.evaluation.cli \
  --config configs/evaluation/smoke.yaml \
  --limit 2 \
  --device cpu
```

実チェックポイントを使う例は `configs/evaluation/example.yaml` です。結果は `data/results/se-evaluation/<run_id>/` に保存されます。中断した実験は同じ `run_id` で `--resume` します。
