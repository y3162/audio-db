# ASR

`src/asr` は音声認識の独立モジュールです。評価パイプラインにも、ほかの研究コードからも同じ入口で使います。学習と fine-tuning はしません。

```python
from src.asr import load_recognizer

asr = load_recognizer(
    "parakeet",
    model_id="nvidia/parakeet-tdt-0.6b-v2",
    device="cuda:1",
    language="en",
)
result = asr.transcribe(waveform, sample_rate=16000)
print(result.text)
```

`waveform` は 1 次元の float テンソルです。サンプリングレートが 16 kHz でなければ 16 kHz へ変換します。`result.text` は正規化前の認識テキストです。WER の正規化は評価側の `wer-norm-v1` で行います。

使わない系列の重みは読みません。`load_recognizer("whisper", ...)` は NeMo も Wav2Vec2 も初期化しません。

## 系列

| 系列 | Backend | 入口の `name` |
|---|---|---|
| Parakeet | NeMo | `parakeet` |
| Whisper | openai-whisper | `whisper` |
| Wav2Vec2 | Transformers CTC | `wav2vec2` |
| HuBERT | Transformers CTC | `hubert` |
| WavLM | Transformers CTC | `wavlm` |

TDT、CTC、RNN-T は同じ関数ではデコードしません。Parakeet はモデル ID からデコード方式を決め、チェックポイントが持つデコーダを使います。Whisper の `task` は `transcribe` です。翻訳にはしません。

自己教師ありの HuBERT / WavLM / Wav2Vec2 は拒否します。ASR 用に fine-tune された ID だけを受け付けます。

カタログに無い `nvidia/parakeet-*` は、ID に TDT / CTC / RNN-T のどれか一つだけが含まれるときに読めます。複数が含まれる ID は拒否します。

重みの保存先はライブラリのキャッシュです。コードにはパスを書いていません。

| Backend | キャッシュ |
|---|---|
| NeMo | `NEMO_CACHE_DIR`、未設定なら `~/.cache/torch/NeMo` |
| Whisper | `~/.cache/whisper` |
| Transformers | `HF_HOME`、未設定なら `~/.cache/huggingface` |

## モデル

動作検証は、この作業環境で短い音声の `transcribe()` が完了したかを表します。カタログに載っていることと、検証済みであることは別です。

| 系列 | モデル ID | Backend | チェックポイント | 言語 | デコード | 動作検証 | 備考 |
|---|---|---|---|---|---|---|---|
| Parakeet | `nvidia/parakeet-tdt-0.6b-v2` | NeMo | NVIDIA / Hugging Face。`from_pretrained` | 英語 | TDT | 済（cuda:1） | 2 秒の LibriSpeech で英語テキストを得た。CUDA graph は無効のまま greedy TDT |
| Parakeet | `nvidia/parakeet-tdt-0.6b-v3` | NeMo | 同上 | 多言語 | TDT | 未実施 | 言語はモデルカードに従う |
| Parakeet | `nvidia/parakeet-tdt-1.1b` | NeMo | 同上 | 英語 | TDT | 未実施 | |
| Parakeet | `nvidia/parakeet-ctc-0.6b` | NeMo | 同上 | 英語 | CTC | 済（cuda:1） | `EncDecCTCModelBPE` で読んだ。TDT デコーダは使っていない |
| Parakeet | `nvidia/parakeet-ctc-1.1b` | NeMo | 同上 | 英語 | CTC | 未実施 | |
| Parakeet | `nvidia/parakeet-rnnt-0.6b` | NeMo | 同上 | 英語 | RNN-T | 済（cuda:1） | `EncDecRNNTBPEModel`。CTC / TDT には切り替えない |
| Parakeet | `nvidia/parakeet-rnnt-1.1b` | NeMo | 同上 | 英語 | RNN-T | 未実施 | |
| Whisper | `tiny` / `tiny.en` | Whisper | OpenAI。`whisper.load_model` | 多言語 / 英語 | seq2seq transcribe | `tiny` を cuda:1 で済 | `tiny.en` は未実施。task は transcribe |
| Whisper | `base` / `base.en` | Whisper | 同上 | 多言語 / 英語 | seq2seq transcribe | 未実施 | |
| Whisper | `small` / `small.en` | Whisper | 同上 | 多言語 / 英語 | seq2seq transcribe | 未実施 | |
| Whisper | `medium` / `medium.en` | Whisper | 同上 | 多言語 / 英語 | seq2seq transcribe | 未実施 | |
| Whisper | `large` / `large-v1` / `large-v2` / `large-v3` | Whisper | 同上 | 多言語 | seq2seq transcribe | 未実施 | |
| Whisper | `turbo` / `large-v3-turbo` | Whisper | 同上 | 多言語 | seq2seq transcribe | 未実施 | |
| Wav2Vec2 | `facebook/wav2vec2-base-960h` | Transformers | Meta。`AutoModelForCTC` | 英語 | CTC | 済（cuda:1） | 代表モデル。未使用の `masked_spec_embed` について Transformers が警告する |
| Wav2Vec2 | `facebook/wav2vec2-large-960h` | Transformers | 同上 | 英語 | CTC | 未実施 | |
| Wav2Vec2 | `facebook/wav2vec2-large-960h-lv60` | Transformers | 同上 | 英語 | CTC | 未実施 | |
| Wav2Vec2 | `facebook/wav2vec2-large-960h-lv60-self` | Transformers | 同上 | 英語 | CTC | 未実施 | |
| HuBERT | `facebook/hubert-large-ls960-ft` | Transformers | Meta。fine-tune 済み | 英語 | CTC | 済（cuda:1） | `hubert-large-ls960` は事前学習のみなので拒否 |
| HuBERT | `facebook/hubert-xlarge-ls960-ft` | Transformers | 同上 | 英語 | CTC | 未実施 | |
| WavLM | `patrickvonplaten/wavlm-libri-clean-100h-base-plus` | Transformers | Microsoft WavLM の CTC fine-tune | 英語 | CTC | 済（cuda:1） | `microsoft/wavlm-base-plus` は ASR ではないので拒否 |

ライセンスは、Parakeet が CC-BY-4.0、Whisper と Meta の Wav2Vec2 / HuBERT が MIT、WavLM の fine-tune も公開カード上は MIT です。

`openai/whisper-tiny` は `tiny` として扱います。英語専用チェックポイントに英語以外の `language` を渡すと、ダウンロード前にエラーになります。`language` を省略した Whisper は英語の `transcribe` です。

## 評価からの利用

```yaml
asr:
  enabled: true
  name: parakeet
  model_id: nvidia/parakeet-tdt-0.6b-v2
  device: cuda:1
  language: en
```

`src/evaluation/models/asr.py` は設定を `src.asr.load_recognizer()` に渡すだけです。`mock` は評価のテスト用で、モデル本体ではありません。認識テキストは仮説として保存し、WER を計算するときだけ `wer-norm-v1` を掛けます。

例は `configs/evaluation/e2e.yaml` です。SE は CPU、Parakeet は空いている GPU を指定します。

## 環境

NeMo は Python 3.13 の `.venvs/lab` には入れません。ASR 用は `.venvs/asr`（Python 3.12.13）です。再構築は [ASR 環境](../envs/ASR.md) にあります。
