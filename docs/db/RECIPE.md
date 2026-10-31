# 音声生成レシピ

生成条件の正本はレシピ JSON である。DuckDB はそれを保存し、波形は `src.evaluation.audio.render.render_recipe` が作る。未知の `type` と `version` は拒否する。

`case_id` は、この文書を正規化した UTF-8 の SHA-256 である。正規化はキーのソート、空白なし、浮動小数の小数 6 桁への丸めである。`case_id` 自体はレシピに入れない。

## 形

```json
{
  "recipe_version": 1,
  "sources": {
    "clean": {"path": "../corpora/LibriSpeech/test-clean/example.flac", "sha256": "<64 hex>"},
    "noise": {"path": "../corpora/DEMAND/DKITCHEN/ch01.wav", "sha256": "<64 hex>"},
    "rir": {"path": "rir/room001.wav", "sha256": "<64 hex>"}
  },
  "operations": [
    {"type": "rir", "version": 1, "source": "rir", "channel": 0, "length": "trim_to_input"},
    {
      "type": "additive_noise",
      "version": 1,
      "source": "noise",
      "snr_db": -5.0,
      "noise_start": 160000,
      "noise_start_unit": "sample",
      "seed": 7
    },
    {"type": "peak_normalize", "version": 1, "limit": 0.99}
  ],
  "output": {"sample_rate": 16000, "reference": "reverberant_clean", "max_seconds": 2.0}
}
```

`path` はデータベースファイルからの相対パスである。`sha256` はそのファイルの内容で、再生時に再計算して食い違えばエラーにする。コーパス全件はハッシュしない。

`output.reference` は指標が使う参照で、`dry_clean`、`reverberant_clean`、`none` のいずれかである。RIR を含むレシピでは、どちらを使うかをこのフィールドに書く。収録済みの noisy は `none` とし、PESQ / STOI / SI-SDR は `na`（`no_clean_reference`）のままである。

操作が空のレシピは、ファイルを読むだけである。clean、noisy、clean と noisy の paired がそれにあたる。

## 操作

| type | version 1 |
|---|---|
| `additive_noise` | 平均二乗電力の比で SNR を作る。無音区間も平均に入れる。短い雑音は繰り返す。`noise_start` は、雑音を `output.sample_rate` へ変換して繰り返したあとのサンプル位置である。seed は保存するが、再生時に位置を計算し直さない |
| `rir` | 指定チャンネルを因果的に畳み込み、`trim_to_input` で入力長へ切る。サンプル 0 は dry 入力と揃う。入力より後ろの残響尾は捨てる。多チャンネル出力や、部屋寸法からの RIR 生成はしない |
| `peak_normalize` | 観測信号のピークが `limit`（0 より大きく 1 以下）を超えるときだけ、観測信号と参照へ同じゲインを掛ける。超えていなければゲインは 1 |

同じ操作を複数回書ける。実行順は配列の順である。ハードクリップは操作にしない。

`max_seconds` があるときは、処理の前に clean の先頭だけを使う。元の発話より短いときは WER を `na`（`partial_audio`）にする。

## ケースの列

`eval.cases` は `case_id`、`kind`、`dataset`、`split`、`sample_rate`、`transcript`、`speaker_id`、`recipe_json`、`content_hash`、`metadata` だけを持つ。`sample_rate` は `output.sample_rate` と同じ値である。SNR やパスは列にしない。

`content_hash` は、`case_id` を除いたケース本体のハッシュである。同じ `case_id` で transcript や metadata が違う登録はエラーになる。例外は、旧 DB からの移行で metadata に残る `legacy_case_id` と、その旧 ID を写した `source_id` だけである。この差では行を書き換えず、再登録は unchanged になる。合成ケースの混合 seed は、操作名ではなく歴史的なプロトコル名 `rms_v1` を入力にする。こうすると、保存済みの `noise_start` と seed を `db-init` が計算し直しても同じレシピになる。
