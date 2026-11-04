# PESQ の長さ依存と対応（2026-11-04）

対象は `dev` `e602d6ef47f67b622fbf2312538f258ce718fd5e`、VoiceBank-DEMAND test の `p232_002`、case ID `6624900d190d50054f21e570d3505c3f325ba26ff2566cb2b51c6d2baad4f371`。Whisper Large v3 と公式 TF-GridNet MS-SNSD の重みを使用した4発話評価で、enhanced PESQ のみ `pesq_utterance_split_discontinuity` になった。一次記録は `data/results/training/pesq-investigation-20261104/`（以下 `R`）と `data/results/se-evaluation/whisper-vb4-paired-final-20261104/`。DB・元の評価と重みは変更していない。

## 波形と原因の切り分け

同じ設定で再推論し、enhanced は 32-bit float WAV に保存した。3 波形とも 48 kHz、130,330 samples、NaN / Inf は 0。clean / noisy / enhanced の peak は 0.5000 / 0.5294 / 1.2649、RMS は 0.0902 / 0.0935 / 0.2288。enhanced は ±1 を超えるが、保存形式は float でクリップしておらず、PESQ に渡す値もクリップしていない。20 ms ごとの RMS が -60 dBFS 未満の区間は3波形とも0%。16 kHz 変換後、clean との相互相関絶対値が最大となる lag は noisy / enhanced とも 0 samples（±0.5 秒の探索、正規化相関 0.9650 / 0.9895）。これは細かな時間変動や全帯域の完全な整列を証明するものではないが、全体的なずれ、長さ不一致、非有限値は認めなかった。詳細は `R/diagnosis.json`。

`pesq==0.0.4` の wideband 実装へ同じ波形の接頭辞を渡した調査値は次のとおり。切り詰めは調査だけに使用し、採点を改善するためには使わない。元の48 kHz 波形は 16 kHz に変換後 43,444 samples である。

| 16 kHz での長さ | noisy | enhanced |
|---:|---:|---:|
| 43,391（既存ガードが比較する長さ） | 3.061132 | 3.227279 |
| 43,392（64 samples の境界） | 3.061099 | 3.226623 |
| 43,400 | 3.061085 | 3.203382 |
| 43,440 | 3.060167 | 3.098071 |
| 43,444（全文） | 3.060075 | 3.095389 |

PESQ のネイティブ関数 `pesq_psychoacoustic_model` 入口を `gdb` で観測すると、43,391 / 43,392 / 43,444 samples の全てで分割発話数は1、推定 delay は0。境界直前・直後の enhanced 差は約0.00066で、分割数や推定 delay の急変は観測しなかった。既存ガードの比較は隣接長ではなく、全文と直前の bin の最後（今回53 samples 差）であり、その間の PESQ 変動は約0.132。発話分割の異常を原因と断定できず、既存 reason 名はこの事例の機構を正確に表さない。score の長さ依存が PESQ 内部のどの計算に由来するか、全文の raw score が品質指標として妥当かは未確定である。したがって、より高い境界値への差し替えや、閾値を調整して全文値を無条件に採用する修正はしない。標準実装の全文 raw 値を記録し、保守的な `na` を維持する。PESQ の参照実装を含む勧告については [ITU-T P.862](https://www.itu.int/rec/t-rec-p.862) を参照。

再現時は `R/config.yaml` の run_id を新規に変え、GPU 使用前に空きを確認する。`diagnose.py` は保存済みの `whisper-vb4-pesq-investigation-20261104` の音声を読む。

```bash
CUDA_VISIBLE_DEVICES=<空きGPU> OMP_NUM_THREADS=1 PYTHONPATH=. \
  .venvs/asr/bin/python -m src.evaluation.cli --config data/results/training/pesq-investigation-20261104/config.yaml --run-id <新規run_id>
OMP_NUM_THREADS=1 PYTHONPATH=. .venvs/asr/bin/python data/results/training/pesq-investigation-20261104/diagnose.py
# gdbscore.py の引数を 43391、43392、43444 に変え、入口で err_info を表示
OMP_NUM_THREADS=1 gdb -batch -ex 'set breakpoint pending on' -ex 'break pesq_psychoacoustic_model' -ex run -ex 'p err_info->Nutterances' -ex 'p err_info->Utt_Delay[0]@err_info->Nutterances' --args .venvs/asr/bin/python data/results/training/pesq-investigation-20261104/gdbscore.py 43391
```

## 修正と再評価

`summary.json` に `paired_signal_metrics` を追加し、同じ record の noisy / enhanced がともに `ok`・有限の case を採用する。指標別に case ID、除外理由、両平均、enhanced − noisy の平均を出す。各条件の従来の単独平均は維持する。全文スコアも異常検出も変更しない。case ID は両条件の音声が同一発話である証跡として保存する。

前の run `whisper-vb4-lab-20261104` と修正後 `whisper-vb4-paired-final-20261104` の比較。PESQ の paired 比較は前の results.jsonl に新集計を適用して求めた。いずれも条件別 WER は `wer-norm-v1` micro の 0/54、STOI / SI-SDR は4件。PESQ は共通の3件である。

| | PESQ noisy / enhanced（3件） | STOI noisy / enhanced（4件） | SI-SDR noisy / enhanced（4件、dB） |
|---|---|---|---|
| 修正前 | 2.354362 / 3.072025 | 0.929970 / 0.945617 | 8.843637 / 17.393583 |
| 修正後 | 2.354362 / 3.072013 | 0.929970 / 0.945617 | 8.843637 / 17.393583 |

enhanced PESQ の差約 0.000012 は別 GPU 推論での差を含む。従来の noisy 単独平均は4件で 2.530790、enhanced は3件なので、これらの差を同一集合の改善量として扱わない。4件全ての PESQ は得られていない。全件で比較するには、全文 raw の信頼性を独立に立証するか、事前に別指標・評価手順を定める必要がある。修正前後の集計とケース一覧は `R/before-after.json`、実行ログは `R/paired-final.log`。回帰試験は `R/test-evaluation-final.log`（90件、1件 skip）、`R/test-metrics.log`、`R/test-paired.log`。保存済み旧 discontinuity 音声が無いテストは skip であり、モックを用いたガードのテストは成功した。
