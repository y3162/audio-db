# 設計判断

変更履歴ではなく、決定と理由の記録。状態は 2 つに分ける。

- **実装済み**: コードとテストにある。
- **採用決定済み・未実装**: 方針は決まったが、コード・スキーマには無い。実装済みとして扱わず、現状の記述（`PROJECT.md`、`DATA.md`、`docs/`）も書き換えない。実装したら「実装済み」へ移す。

## 実装済み

1. **評価の入力は DuckDB の名前付きセットだけ。Manifest は登録時にしか読まない。** 同じ名前なら入力・順序・混合条件が同じになり、Manifest や元ファイルの変更に評価が左右されない。
2. **`make db-init` と `make db-ingest` を分ける。`db-init` は `corpora.audio_files` に触れず、前後の指紋で確認する。** 評価セットの登録で、コーパス由来のテーブルを壊さない。
3. **合成混合は `rms_v1`。条件を case に保存して再生成する。** 区間全体の平均二乗電力の比で、VoiceBank-DEMAND・URGENT の手順でも、ITU-T P.56 でも、VAD でもない。seed は SHA-256 で作り（Python の `hash()` は使わない）、雑音の対応はセットの seed と path で決める（実験 seed では変わらない）。SE を変えても同じ入力で比べるため。
4. **WER は micro 集計と `wer-norm-v1`。出せない値は `na` と理由で残し、0 にしない。** 部分音声と全文の組は無効にする。不正な WER を作らないため。
5. **CHiME3 の正解は公式注釈の `prompt`。CH0・real・simu・`tr05_org` を clean 参照にしない。** 時間整合した clean 参照が無いのに PESQ / STOI を出さないため。
6. **SE と ASR は独立モジュール。評価は共通 API だけを呼ぶ。** ASR は使う系列のバックエンドだけを import し、環境も lab と分ける（NeMo の依存が合わない）。
7. **推論 API は推論専用。** `transcribe()` と `enhance()` に学習や勾配の経路を持たせない。
8. **独立 SE 実装は、公式コードに実行時依存しない。一致は最終波形の最大絶対誤差で判定する。** 7 モデルとも判定済み（`docs/se/VALIDATION.md`）。未検証の範囲もそこに残してある。
9. **ASR は、ASR 用に fine-tune されたチェックポイントを、そのデコーダのまま使う。** 自己教師ありだけのチェックポイントは拒否する。Parakeet は TDT / CTC / RNN-T をモデル ID のデコーダのまま使い、Whisper は `transcribe` だけにする。系列やデコーダの取り違えを防ぐため。
10. **結果は run 単位のディレクトリに JSON / JSONL で保存し、`results.jsonl` から再集計する。** 書き込みは原子的で、再開は成功レコードだけを再利用する。DuckDB への登録は、そのファイルを `register_run` で後から載せる。
11. **DB はレシピ JSON を管理し、音声生成は Python が解釈する。** 操作は順序付きの `additive_noise`、`rir`、`peak_normalize`（いずれも version 1）。未知の操作と version は拒否する。詳細は `docs/db/RECIPE.md`。
12. **`case_id` は正規化レシピの SHA-256 である。** キー順はソートし、浮動小数は小数 6 桁へ丸める。`case_id` 自身はハッシュに入れない。同じレシピは再構築しても同じ ID になる。
13. **登録データは `data/staging/` に保存してから DuckDB へ入れる。** 参照は `case_id` と `path` で持ち、rowid や採番に依存しない。同じステージングから作った DB は、登録順が違っても内容が一致する。
14. **旧 `eval.cases` の固定列を実行時に読まない。** 一度きりの変換は `src.data.migrate_legacy` で、ステージングと検証用 DB を作る。本番 `resource.duckdb` は、その検証のあとレシピスキーマへ置き換えた。旧ファイルは `data/db/backups/resource-legacy-20261031.duckdb` に残す。`migrate_legacy` は既存のステージングと再構築 DB を上書きしない。
15. **SE の学習は `src/training/` に置き、`enhance()` と `transcribe()` には入れない。** 最初のモデルは MP-SENet。Generator は推論実装をそのまま使い、損失と Metric Discriminator は学習側にある。公式ツリーは実行時に import しない。
16. **Global batch は厳密に保つ。** `micro_batch_size_per_gpu` は上限で、その範囲の最大の micro batch と整数の蓄積回数で global batch を作る。作れない構成はエラーにする。Global step は Optimizer 更新の回数である。MP-SENet の ExponentialLR は、公式と同じく、そのエポックの使える micro-batch を終えたときだけ進む。蓄積回数では進まない。
17. **学習の乱数は用途ごとに分ける。** 初期化、dropout、サンプル順、切り出しは独立である。切り出しと順序はサンプル ID と seed から再現し、消費順には依存させない。`deterministic: true` では決定的アルゴリズムを有効にし、決定的に計算できない CUDA 逆伝播は非決定的カーネルへ戻さず、同じ順伝播値の別実装にする。
18. **学習データは名前付きセットのレシピだけを使う。** 教師が定義できないケースと、LibriSpeech test-clean・CHiME3 et05 を含む評価用セットは拒否する。ケースの識別はレシピハッシュで、採番 ID ではない。本番 `resource.duckdb` は評価セットの本番ファイルなので、学習からは名前で拒否する。
19. **7 モデルの学習タスクは `src/training/se/` に分け、共通 Trainer に損失を書かない。** MP-SENet は公式ステップとの数値比較まで行った。AdamW 更新後の重み差は、初回ステップの閉形式で説明できる。TF-GridNet の学習損失は ESPnet の `mr_l1_tfd` である。SEMamba++ の参照学習ソースにはライセンスが無いので、Discriminator は CQT ではなく独立実装のままである。MossFormerGAN の Discriminator に公開の Conformer は無い。DeepFilterNet3 の学習損失は出荷 `config.ini` の multi-resolution スペクトルと local SNR で、mask と SI-SDR は 0 である。エポック内の cosine はステップ数がデータに依存するため、タスクは YAML の scheduler を使う。UniPASE と StuPASE の公式 DeWavLM 教師は WavLM-Large であり、そのファイルが無い段階は縮小 WavLM の MSE でしか実行しない。StuPASE の CFM 学習は `forward` の流速 MSE で、推論の ODE サンプルとは別である。段階学習は段階ごとに Optimizer を分ける。
