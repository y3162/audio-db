# 評価パイプライン

データセット、音声強調（SE）、音声認識（ASR）、評価指標を組み合わせて実験するための実行基盤です。SE の推論本体は `src.se.load_enhancer()` のままです。このパッケージはモデルを学習しません。コーパスの取り込みも、`corpora.audio_files` の再作成もしません。

全コーパスの推論や論文用の WER 比較は、この基盤の対象外です。合成音声と Mock でパイプラインが通ることと、チェックポイントがあるときの短い実モデル実行を分けて扱います。

## ディレクトリ

```text
src/evaluation/
├── cli.py                 # python -m src.evaluation.cli
├── config.py              # 設定の検証
├── pipeline.py            # 段階の接続
├── datasets/              # DuckDB の評価セット / synthetic
├── audio/                 # 読み込みと rms_v1 混合
├── models/                # SE は src.se、ASR は src.asr を呼ぶ
├── metrics/               # WER / PESQ / STOI / ESTOI / SI-SDR
└── results/               # JSONL・CSV・再開用のレコード

src/asr/                    # 認識モデル本体。評価には依存しない
configs/db/init.yaml         # make db-init が登録する Manifest と評価セット
configs/evaluation/
├── smoke.yaml             # 合成音声 + mock
└── example.yaml           # 名前付き評価セット + MP-SENet

examples/manifests/          # DuckDB へ登録する Manifest の例
tests/evaluation/
```

`src/data/` と `src/se/` の処理は複製していません。波形のデコードは torchaudio、データベースの列定義は `src.data.models.audio_file` を使います。

## 処理の段階

各段階は単独でも呼べます。

1. 入力の選択（DuckDB の評価セット、または合成音声）
2. 波形の読み込み
3. 必要なときだけ雑音混合
4. SE
5. ASR（無効ならここを飛ばす）
6. 指標（正解テキストが無ければ WER を記録しない。clean が無ければ PESQ / STOI を記録しない）
7. 結果の保存と集計

ASR が未導入でも SE までは実行できます。指標ライブラリが無い場合も、その指標だけ `unavailable` になり、ほかの指標は続きます。

## データ入力

評価時の入力は、DuckDB に登録した名前付きセットか、メモリ上の合成音声です。Manifest を評価の `dataset.source` には使えません。

共通の単位は発話です。発話 ID、データセット名、split、音声パス、サンプリングレート、書き起こし、clean / noisy / noise のパス、SNR、話者 ID、任意のメタデータを持てます。波形は、その発話を処理するときだけ読みます。

| 種類 | 内容 |
|---|---|
| clean | クリーン音声 |
| noisy | 既存の雑音付き音声 |
| paired | clean と noisy の対応 |
| synthetic | clean と noise から `rms_v1` で作る音声 |

```yaml
dataset:
  source: duckdb
  path: data/db/resource.duckdb
  corpora_root: data/corpora
  set: libri_demand_snr0
```

```python
from src.evaluation.datasets import EvaluationDB

db = EvaluationDB("data/db/resource.duckdb")
dataset = db.load_dataset("libri_demand_snr0")
```

セットの順序は登録時のままです。同じセットを使う限り、SE モデルが違っても入力は同じです。

### DB 初期化

```bash
make db-init
```

1. `eval.cases`、`eval.sets`、`eval.set_items` を作る。
2. `configs/db/init.yaml` の Manifest をインポートする。
3. 同じファイルに書いたコーパス選択で評価セットを作る。
4. 件数を表示し、`corpora.audio_files` が変わっていないことを確認する。

このコマンドは `make db-ingest` を呼びません。2 回実行しても case は増えません。同じ ID で内容が違う再登録はエラーです。Manifest や音声が無い任意項目は、何をスキップしたかを表示します。`required: true` の入力に不備があると、コマンドは失敗します。

現在の任意項目は、例示用の `examples/manifests/example.jsonl`、LibriSpeech test-clean の 2 発話を DEMAND と 0 dB で混ぜる `libri_demand_snr0`、同じ test-clean から path 順に 8 発話を全長のまま取る `libri_testclean_full_smoke`、CHiME3 の公式 JSON から作る評価セット、MP-SENet の VoiceBank-DEMAND テストリストです。音声が無い項目はスキップします。VoiceBank-DEMAND の発話 ID は `configs/db/lists/voicebank_demand_test.txt` にあり、公式コードの配置には依存しません。wav はこのチェックアウトには無く、無いときは登録をスキップします。

CHiME3 は `data/annotations/*.json` の `prompt` を正解テキストにします。評価セットはチャンネルを分けます。

| セット | 内容 |
|---|---|
| `chime3_et05_real_ch5` | et05 の実録音。タブレットの CH5。clean 参照は無い |
| `chime3_et05_real_ch0` | 同じ実録音の近接マイク CH0。雑音環境の収録であり、clean 参照ではない |
| `chime3_et05_simu_ch5` | et05 の公式シミュレーション。CH5。元の WSJ 波形とはペアにしていない |
| `chime3_tr05_org` | 学習用の元音声。clean。実録音の時間整合した参照ではない |

CH1–CH6 はタブレット、CH0 は近接マイクです。インポータはほかの split とチャンネルも受け付けます。登録は `make db-init` の `annotation_sets` です。

### Manifest のインポート

JSONL（空行は無視）または JSON 配列です。相対パスは Manifest のあるディレクトリ、または `root` から解決します。登録後は Manifest ファイルが無くてもセットを読めます。音声ファイルは必要です。

```json
{"utterance_id": "u1", "dataset": "toy", "split": "test", "clean_path": "clean.wav", "noise_path": "noise.wav", "transcript": "hello", "speaker_id": "s1", "snr_db": 0}
```

`clean_path` と `noise_path` と `snr_db` がある行は synthetic です。`clean_path` と `noisy_path` は paired、`noisy_path` だけは noisy、clean または `audio_path` だけは clean です。既知のキー以外は metadata に残します。`corpora.audio_files` に同じ音声があるときは、Manifest に無い書き起こしと話者をそこから補います。音声そのものを `corpora.audio_files` へは追加しません。

パスはデータベースファイルからの相対パスで保存します。プロジェクトと `data/corpora` の外にあるファイルは登録できません。

```python
db.import_manifest("examples/manifests/example.jsonl", "example")
```

### コーパスからの評価セット

Manifest を経由せず、`corpora.audio_files` を選んでセットにできます。並びは `path` です。

```python
db.create_set_from_corpus(
    "libri_clean",
    role="clean",
    filters={"corpus": "LibriSpeech", "subset": "test-clean"},
    limit=2,
)
```

SQL を渡す場合は、`corpora.audio_files` を読む単一の SELECT に限られます。

```python
db.create_set_from_corpus(
    "libri_clean",
    role="clean",
    query="SELECT * FROM corpora.audio_files WHERE corpus = ? AND subset = ? ORDER BY path",
    params=["LibriSpeech", "test-clean"],
    limit=2,
)
```

`role: synthetic` は clean ごとに雑音を 1 つ割り当て、SNR、seed、`rms_v1`、clean の先頭秒数、雑音の切り出し位置を case に保存します。雑音の対応はセットの seed と path から決まるので、実験の seed では変わりません。サンプリング周波数が違うときは切り出し位置を保存せず、同じ seed で再計算します。

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
    prevent_clip=True,
)
```

`dataset.set` を使う設定では `mixing.enabled` を false にします。混合条件は case 側にあります。

名前付きセットを使わない DuckDB 読み取りも残しています。`filters` と `noise_filters` で `corpora.audio_files` を読み取り専用のまま選び、`mixing.enabled: true` のときにその場で混ぜます。比較実験は名前付きセットを使ってください。`pair: hash` は発話 ID から雑音を選び、`pair: cycle` は並び順に巡回します。

### 合成音声

`dataset.source: synthetic` はメモリ上の振幅変調ノイズを作ります。コーパスが無くても `configs/evaluation/smoke.yaml` が動きます。

### データが無いとき

- 評価セットも DuckDB も無い: 実行はエラー。合成設定には影響しません。
- セットが未登録、またはクエリが 0 件: エラー。
- 行はあるが音声ファイルが無い: その発話は `skipped`。

## 雑音混合（rms_v1）

VoiceBank-DEMAND とも URGENT とも同じ手順ではありません。活性音声レベル（ITU-T P.56）も使いません。

定義は、リサンプリングと長さ合わせのあと、成分の平均二乗電力の比です。

```text
SNR_dB = 10 * log10(mean(clean^2) / mean(scaled_noise^2))
```

- 雑音が短ければ繰り返してから、seed で開始位置を決めて切り出します。
- 雑音が長ければ、同じ方法で開始位置を決めます。seed は SHA-256 から作り、Python の `hash()` は使いません。
- 無音を含む区間も平均に入れます。VAD では切りません。
- clean 全体の電力が `1e-12` 未満なら混合せず、発話は `clean_has_zero_energy` で skip します。
- 雑音の電力が 0 ならエラーです。SNR が有限でない値もエラーです。
- `prevent_clip: true` は、ピークが 0.99 を超えるとき clean と雑音へ同じゲインを掛けます。SNR は変わりません。
- `hard_clip: true` は混合後に [-1, 1] へ切ります。記録する達成 SNR は、このクリップの前の値です。

SNR のリストは設定で変えられます。`-10, -5, 0, 5` は例であり、固定値ではありません。

## SE

```yaml
models:
  - name: mp_senet
    variant: vb
    checkpoint: data/checkpoints/mp-senet/g_best_vb
  - name: unipase
    checkpoints:
      dewavlm: /path/DeWavLM-Omni.pt
      adapter: /path/Adapter.pt
      vocoder: /path/Vocoder.pt
      postnet: /path/PostNet.pt
```

`name` と `variant` は `src.se` の registry で検証します。チェックポイントが無いモデルに、別の重みは割り当てません。複数 variant があるモデルは variant を必須にします。

モデルは発話ごとにロードし直しません。発話の一部だけが未完了のとき、そのモデルを 1 回だけロードします。複数モデルは 1 つずつ処理し、終わったら参照を捨てます。

UniPASE や StuPASE のように乱数を使うモデルのため、強調の直前に発話ごとの seed で `torch.manual_seed` と `random.seed` を設定します。seed は実験 seed、発話 ID、モデル名、variant、SNR の SHA-256 です。処理順を変えても同じ発話の seed は同じです。

出力は 1 次元で、入力と同じ長さで、有限でなければなりません。長さが変わった場合、その発話は `error` になり、PESQ や STOI を 0 としては記録しません。

`name: mock` は入力をコピーする試験用です。強調モデルの性能ではありません。

## ASR

認識本体は `src.asr` です。評価パッケージはそれを呼びます。

```python
from src.asr import load_recognizer

asr = load_recognizer("parakeet", model_id="nvidia/parakeet-tdt-0.6b-v2", device="cuda:1")
result = asr.transcribe(waveform, sample_rate=16000)
```

```yaml
asr:
  enabled: true
  name: parakeet
  model_id: nvidia/parakeet-tdt-0.6b-v2
  device: cuda:1
  language: en
```

`name` は `mock`、`parakeet`、`whisper`、`wav2vec2`、`hubert`、`wavlm` です。`mock` 以外は `model_id` が必要です。デバイスを省略すると CPU です。SE の `runtime.device` とは別に指定できます。GPU を使うときは、その番号が空いていることを実行前に確認します。

Whisper は `transcribe` だけで、翻訳にはしません。Parakeet の TDT / CTC / RNN-T はモデル ID のデコーダのままです。HuBERT と WavLM の自己教師ありチェックポイントは ASR として読みません。

仮説の `text` は正規化前です。WER は `wer-norm-v1` を参照と仮説の両方に掛けます。モデル一覧は [ASR](../asr/MODELS.md) です。

## 指標

WER は発話ごとの編集回数を足してから割ります。

```text
WER = (S + D + I) / N
```

`N` は参照の語数です。発話 WER の平均は出しません。`N = 0` の発話は分母に入れず、値は null です。0 にはしません。

認識に使った音声が元の発話より短い接頭辞で、その区間に対応する部分テキストが無いとき、WER は `na`、理由は `partial_audio` です。`max_seconds` があるだけでは切り出し済みにしません。元の長さがその秒数以下なら全文として計算します。長さが取れないときは `unknown_audio_span` で `na` にします。正解テキストが無いときも `na`（`no_reference_transcript`）です。これらの発話は micro 集計の `N` に入れません。

正規化は `wer-norm-v1` です。NFKC、小文字化、Unicode の句読点を空白に置換、空白の圧縮、です。生の空白区切りも同時に保存します。正規化バージョンを変えると設定ハッシュが変わり、以前の結果は再利用しません。

PESQ は 16 kHz なら wideband、8 kHz なら narrowband です。それ以外のサンプリングレートは 16 kHz へ変換してから wideband を計算し、変換したことを記録します。STOI も 16 kHz 以外は 16 kHz へ変換します。長さが違う参照と推定は短い方に揃え、削ったサンプル数を記録します。時間ずれの推定はしません。

clean が無いとき、PESQ / STOI / SI-SDR は `na` / `no_clean_reference` で、値は null です。`pesq` または `pystoi` が未インストールのときは `unavailable` とインストール方法を記録し、WER などは続けます。

ESTOI は `metrics.estoi: true`、SI-SDR は `metrics.si_sdr: true` です。既定はオフです。DNSMOS と CER は未実装です。

## 設定と CLI

未知のキー、チェックポイントの欠落、有限でない SNR、WER だけ有効、は実行前にエラーになります。

```bash
make db-init
python -m src.evaluation.cli --config configs/evaluation/smoke.yaml --dry-run
python -m src.evaluation.cli --config configs/evaluation/smoke.yaml --limit 2 --device cpu --seed 0
python -m src.evaluation.cli --config configs/evaluation/example.yaml --run-id libri-2s --device cpu
python -m src.evaluation.cli --config configs/evaluation/example.yaml --run-id libri-2s --resume
python -m src.evaluation.cli --summarize data/results/se-evaluation/libri-2s
```

`--limit`、`--device`、`--seed`、`--output`、`--run-id`、`--models` は設定ファイルを上書きします。`--limit` は実行件数の上限で、設定ハッシュには含まれません。再開時に件数だけ増やせます。データセットの絞り込みや seed を変えるとハッシュが変わり、同じ `run_id` では再開できません。

Dry-run はモデルをロードしません。対象発話、モデル、チェックポイントの有無、SNR、出力先、予定件数を表示します。準備ができていなければ終了コード 2 です。

CUDA を指定した実行は `nvidia-smi` を見て、使用メモリが 1024 MiB 以上の GPU を拒否します。空いている GPU だけを指定してください。`runtime.allow_busy_gpu: true` は、その GPU が自分のジョブだと分かっているときだけ使います。試験の多くは CPU で足ります。

## 結果

```text
data/results/se-evaluation/<run_id>/
├── config.json
├── run.json           # run id、git commit、設定ハッシュ、環境
├── results.jsonl      # その実行で採用した発話結果
├── summary.json
├── summary.csv
├── records/<id>.json  # 発話ごとの原子的書き込み
└── audio/<id>.wav     # save_audio: true のとき強調波形だけ
```

Run ID を省略すると、実験名、UTC 時刻、設定ハッシュから作ります。同じディレクトリが既にある状態で `--resume` を付けないとエラーです。

各レコードには、run ID、git commit、設定ハッシュ、データセット、発話 ID、パス、SNR、seed、モデル名、variant、チェックポイント SHA-256、ASR 設定、認識結果、正解、WER の編集数、PESQ / STOI、処理時間、エラーまたは skip 理由、Python / torch / デバイスが入ります。チェックポイントの SHA-256 はサイズと mtime が変わるまで再計算しません。

`results.jsonl` が再集計の入力です。WER は micro 集計、PESQ / STOI / SI-SDR は有限な発話スコアの平均です。欠けたスコアは平均に 0 として入れません。

### 再開

成功したレコードだけを再利用します。再利用の条件は、設定ハッシュ、torch の版、発話、モデル、variant、チェックポイント内容、SNR、サンプリングレート、入力波形、clean 参照、強調 seed が一致することです。失敗と skip はやり直します。音声やチェックポイントの内容が変わると、同じ枠を上書きして計算し直します。

レコードは一時ファイルへ書いてから `os.replace` します。JSONL と summary も同様です。

`output.save_audio` の既定は false です。true のときは強調波形だけを 32-bit float WAV で保存します。noisy と clean は保存しません。

## テスト

```bash
cd /net/spring/work/seki/exp/audio-db
.venvs/lab/bin/python -m unittest discover -s tests/evaluation -v
```

合成音声、Mock SE、Mock ASR で、混合、SNR、WER の micro 集計、指標の skip、保存、再開、設定変更によるキャッシュ無効化、dry-run を確認します。GPU も巨大チェックポイントも要りません。

`tests/evaluation/test_real_smoke.py` は、MP-SENet の `g_best_vb` と DuckDB と音声ファイルがあるときだけ、LibriSpeech test-clean の 1 発話を先頭 2 秒、DEMAND の先頭 2 秒、SNR 0 dB、CPU で実行します。これは接続の確認であり、論文用のスコアではありません。Mock の成功とも別です。チェックポイントが無い環境では skip します。

既存の SE 試験は今までどおりです。

```bash
.venvs/lab/bin/python -m unittest discover -s tests/se -v
```

## 依存関係

実行環境は `.venvs/lab` です。評価のために追加したものは `requirements/evaluation.txt` です。

| パッケージ | 用途 |
|---|---|
| PyYAML | 設定ファイル |
| pesq | PESQ。無いときはその指標だけ unavailable |
| pystoi / scipy | STOI と ESTOI。無いときはその指標だけ unavailable |

torch、torchaudio、duckdb、numpy は既存の lab 環境のものを使います。numpy は 2.5.4 のままにします。

```bash
uv pip install --python .venvs/lab/bin/python "numpy==2.5.4" -r requirements/evaluation.txt
```

## まだやらないこと

全コーパスの推論、7 モデルの一括ベンチマーク、論文用 WER、ASR の学習、SE の再学習、結果テーブルの DuckDB への書き込み、公式混合プロトコルへの置き換えは、この基盤の対象外です。ASR の推論入口は [ASR](../asr/MODELS.md) にあります。結果ファイルは、あとから DuckDB へ登録できる JSON のままにしています。
