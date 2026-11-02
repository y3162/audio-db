# TSUBAME 初回 DB 構築（2026-11-02）

ユーザーがデータ転送と完全性確認の完了を確認し、今回に限って staging への先行保存を省略するよう指示した。12 コーパスを候補 DB に索引してから、評価セットを登録し、検証後に `data/staging/resource-current` へ書き出した。本番 `data/db/resource.duckdb` は既存ファイルがないことを確認して新規作成した。この例外は今後の staging 先行ルールを変更しない。

初回 CPU ジョブは途中で停止したが、完成済み 4 コーパスを含む候補を `data/db/resource-partial-20261102.duckdb` に保持し、別コピーで残りを処理した。索引は 1,218,957 件で、読み取り不能ファイルは 0 件。

| コーパス | 件数 |
|---|---:|
| CHiME3 | 122,277 |
| DEMAND | 288 |
| DNS5 | 124,058 |
| LibriSpeech | 292,367 |
| LibriTTS | 375,086 |
| OpenSLR26 | 120,000 |
| OpenSLR28 | 61,260 |
| URGENT2024 | 5,000 |
| VCTK | 44,085 |
| VoiceBank-DEMAND | 70,942 |
| WHAM48 | 1,578 |
| musan | 2,016 |

登録済み評価セットは `libri_demand_snr0` 2 件、`libri_testclean_full_smoke` 8 件、`chime3_et05_real_ch5`・`chime3_et05_real_ch0`・`chime3_et05_simu_ch5` 各 1,320 件、`chime3_tr05_org` 7,138 件、`voicebank_demand_test` 824 件。合計 7 セット、11,932 ケース。サンプル音声が無い例示 Manifest は skip した。

DB と staging のコーパス全行について、件数と DuckDB の全列ハッシュの bit XOR が一致した。ケース・セットの ID と内容ハッシュ、およびセット項目を双方向の `EXCEPT ALL` で比較し、差はなかった。`libri_demand_snr0`、`voicebank_demand_test`、`chime3_et05_real_ch5` の各先頭ケースをレシピから再生し、有限波形を確認した。これは全ケースの再生確認や研究室側 DB との完全一致を意味しない。
