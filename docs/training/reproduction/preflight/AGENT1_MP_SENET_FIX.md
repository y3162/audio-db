# Agent 1: MP-SENet 検証長と PESQ 平均

対象は P10P-A-002 と P10P-A-003 だけである。コミットも push もしていない。本学習は始めていない。GPU は使っていない。

## 変更

`validation_report` は、生成前に hop 倍数へ切らない。clean と noisy の全長を STFT と Generator に渡す。`align_validation_length` は残してあるが、検証はこの関数でモデル入力を切らない。

学習の `_generate` は、波形長と iSTFT 長が違うとこれまでどおり `TrainingError` にする。検証だけ `allow_official_istft_length=True` で、公式の `center=True` iSTFT 長を許す。その長さは `(length // hop) * hop` で、入力以下の最大 hop 倍数である。

magnitude / phase / complex は全長 STFT のままである。consistency は、その全長 STFT から出した複素スペクトルと、iSTFT 波形を公式と同じくもう一度 STFT したものを比べる。iSTFT 波形が反射パディングできないほど短いときだけ consistency を作らず、その損失は `na` にする。時間 L1 は `min(clean, audio)` の重なりだけを比べ、両方の長さを `audio_lengths` に残す。切った長さをモデル入力としては記録しない。

`loss == "official"` の PESQ は `eval_pesq` に合わせる。例外、非有限、ライブラリが返す負のエラーコードは -1 になり、平均に残る。`validation_report` の `pesq.value` がその平均で、`status` は `ok` である。Trainer の `_combine_metric` は `status=ok` の値を平均するので、-1 が落ちない。検証は 1 件ずつ collate する。

`loss == "reconstruction"` は従来の研究側のままである。`na` と error は平均から外す。ポリシー名は `research_exclude_na` で、公式平均には混ぜない。ライブラリが無いときは両経路とも `unavailable` で、値は null である。-1 にも 0 にもしない。

レポートの追加フィールド:

- `pesq_policy`: `official_include_negative_one` または `research_exclude_na`
- `pesq_failures` / `pesq_scored`
- `audio_lengths[]`: `model_input`, `clean`, `audio`, `pesq_reference`, `pesq_estimate`, `time_overlap`

ライブラリ欠如のとき `pesq_failures` は null である。失敗を 0 件とは数えない。

## 切り詰めと全長の数値

CPU、スレッド 1。`torch.manual_seed(1234)` で公式 `MPNet`（dense 64、block 4）を作り、同じ state_dict を `build_model("mp_senet")` に載せる。そのあと `manual_seed(7)`、`randn(2, 1600)` を 2 回、続けて `randn(1, 1650)`。`n_fft=400`、`hop=100`、`win=400`、`compress=0.3`。

| 比較 | 最大絶対誤差 |
|---|---|
| 長さ 1600。公式 `mag_pha` + 公式 Generator と、audio-db STFT + audio-db Generator | 0.0 |
| 長さ 1650。同じ組み合わせ | 0.0 |
| 同じ audio-db Generator。全長 1650 と先頭 1600 の magnitude | 1.5627447366714478 |

フレーム数はどちらも `(1, 201, 17)`。監査の `{:.3e}` は `1.563e+00` で、今回の値をその桁にすると同じである。期待値として 1.563 をテストに固定していない。

## PESQ の失敗件数

807 は失敗件数ではない。

読み取りだけ。スナップショット `data/db/snapshots/36ab4590b113d0aa00884c44041eaa97d65085cee57bcc9c9e06135741fc76d7`。valid の case set `0d2c0497e22da34fd91b55def8bf99ff95c126af6710f68f2d5896f6684eead0`。`case.frames`、`sample_rate` 16000、824 件。

| 量 | 件数 |
|---|---|
| hop 100 の倍数ではない | 807 |
| hop 100 の倍数 | 17 |
| 旧 `align_validation_length` が短すぎるとして拒否する | 0 |

最短 19801、最長 156302。ファイルは変更していない。

PESQ を呼んだ結果:

- 824 件すべての長さについて、220 Hz・振幅 0.1 の float32 正弦を全長と hop 切り詰め（差は最大 99 サンプル）で `pesq(..., on_error=RETURN_VALUES)` した。失敗 0、負の戻り 0、例外 0。23.29 秒。これは合成正弦の長さ差であり、実波形でもモデル出力でもない。
- 無音。1600 と 1650 は `BufferTooShortError`（戻り値 -6）。4096、8000、16000 は `NoUtterancesError`（戻り値 -7）。公式経路はどちらも -1 にして平均に残す。
- 長さ 1650 の検証レポートが PESQ に渡す配列は、参照 1650 と iSTFT 出力 1600 である。この長さは 0.25 秒未満なのでバッファ不足で -1 になる。hop 非倍数そのものでは -1 にならない。

824 件の実 VoiceBank 波形と、824 件の Generator 出力に対する PESQ 失敗件数は測っていない。

## テスト

```text
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venvs/se/bin/python -m unittest \
  tests.training.test_mp_validation_length \
  tests.training.test_mp_senet_validation_full_length \
  tests.training.test_mp_senet_official_pesq_mean \
  tests.training.test_mp_senet_train -v
```

33 件、25.865 秒、OK。`test_mp_senet_train.py` は編集していない。既存の短い検証アサーションは通った。

確認したこと:

- hop 倍数は入力長のまま
- 1650 のモデル入力は 1650。1600 ではない。PESQ の推定波形長は 1600
- 250（hop 100、n_fft 400）と 20（hop 8、n_fft 32）は、旧関数が拒否しても全長を生成に入れる
- 800、850、1600、1650、301、12345
- 更新経路は 1650 で `TrainingError` のまま
- `[4.0, -1.0]` の公式平均は 1.5。研究側は `na` を外し、4.0 のまま
- 無音と、ライブラリの -6、例外は公式経路で -1
- ライブラリ欠如は `unavailable`

## 触ったファイル

- `src/training/se/mp_senet.py`
- `tests/training/test_mp_senet_validation_full_length.py`（新規）
- `tests/training/test_mp_senet_official_pesq_mean.py`（新規）
- この文書

## 親が残しているもの

この作業では編集していない。

- `.agents/DECISIONS.md` の 24 は、検証を hop 倍数へ切るとまだ書いてある。実装は変わった。
- `src/training/se/preview.py` も検証と同じく全長を Generator に入れる。表示長は iSTFT 後の重なりだけに揃える。
- `validation_metrics` は更新経路と同じく長さ不一致を拒否する。Trainer の検証は `validation_report` を使う。
- epoch 要約は `_combine_metric` が `pesq` の `value` だけを平均する。`pesq_failures` と `audio_lengths` は発話レポートに残り、要約にはコピーされない。1 件ずつ検証する限り、平均自体は -1 を含む。
