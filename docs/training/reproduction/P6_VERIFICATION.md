# P6 途中記録（2026-11-06）

MS-SNSD の混合式と、1 クリップの書き出しまで。全量生成、DB 登録、スナップショット、Trainer 接続はしていない。

## 公式から確認した生成

ESPnet `0d136c5` の `egs2/ms_snsd/enh1/local/ms_snsd_noisyspeech_synthesizer.py` と、コーパスの `audiolib.py`:

- 素材は `clean_train` 23,075、`noise_train` 128、`clean_test` 1,100、`noise_test` 51。すべて直下の wav。手元の件数はこれと一致する。
- train 30 時間、test 1 時間。この時間は 5 SNR のサンプル数を足したもの。サンプリングレートは 16 kHz。`sampling_rate` ではリサンプルしない。
- 名目 SNR は `linspace(0, 40, 5)`。`snr_mixer` の平方根のため、測った SNR は名目の半分。
- 各ファイルを −25 dBFS RMS にしてから混ぜる。
- 10 秒より短い素材は 0.2 秒の無音を挟んで連結する。10 秒より長い素材は切らない。雑音は clean の長さまで先頭から切る。ランダムな開始位置は無い。
- 同じクリップを 5 SNR すべてに使う。clean は SNR ごとに上書きされる。
- `np.random` は seed されない。`glob` の順も固定されない。公式の実ファイル列は再現できない（D-07）。
- 公式シェルは `rm -r data/` を使う。audio-db は消さない。

## 実装とテスト

`src/reproduction/build/ms_snsd.py`。実行時に公式コードは import しない。テストだけが `data/corpora/MS-SNSD/audiolib.py` の `snr_mixer` を呼び、同じ入力で配列が一致した。測った SNR は名目の半分。

1 クリップ（5 SNR）を一時ディレクトリへ書き、SHA-256 が再実行で一致した。マニフェストを消して壊した wav は、その shard だけ作り直して同じハッシュに戻った。定義が違う `build_id` は拒否する。`reproduction.duckdb` と `data/staging/reproduction/` の既存物には書いていない。

空き容量は `/gs/bs` で約 17 TB。全量の目安は 3 ストリームで約 10 GB。所要時間は未測定なので、30 時間の生成は保留する。

## 判定

P6 全体は未完了。混合式と小さい Build の書き出しだけができた。
