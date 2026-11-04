# 学習ジョブ継続の実装と検証（2026-11-04）

## 判定

**本学習は未投入、qsub での連鎖は未検証、GO ではない。** ユーザーの指示により、以下の短時間検証ジョブは待機中に取り消した。新たな qsub は投入しない。対話ノードの GPU での検証は qsub の検証には代わらない。2 モデルの並列投入可否、割り当てられる MIG、qacct の反映時間と `-hold_jid` の実効動作は未確認。

作業時 HEAD は `8f6b592228b1b303ed2768ccd0910cf6a12e62cb`、開始時の作業ツリーはクリーンだった。最初に作った qsub テスト 9258525→9258527→9258528→9258530 は、終了状態だけで前ジョブの実際の終了コードを確認できない問題を見つけたため、実行前に取り消した。`qacct` 確認を追加し、正常系 9258554→9258555→9258556→9258557、失敗系 9258559→9258560、破損 checkpoint 9258563 を登録した後、ユーザーから qsub を投入しない指示を受けて、すべて待機中に取り消した。開始済みの検証 qsub はない。

## 実装

`src.training.se.job_chain` は空の新規実験ディレクトリへ設定と source / commit / DB の指紋を記録し、明示した個数まで `-hold_jid` で連結する。投入後の実行は排他 lock を取り、前状態、qacct の正常終了、設定・source・DB・環境、checkpoint SHA-256、進捗を照合する。失敗・不明・破損の場合は自動再試行や巻き戻しをせず FAILED とする。完了済みの場合は追加更新をしない。qacct の記録が 120 秒以内に見つからない場合も失敗する。実験ごとに `experiment.json`、`submissions.jsonl`、`job-NNN.json` と checkpoint、metrics を残す。`submissions.jsonl` の PENDING は投入時点の記録であり、現在のキュー状態は qstat を参照する。部分投入時は既投入 ID をこのファイルに残す。

Trainer の managed 停止は、内部時刻に到達後、gradient accumulation を含む次の optimizer 更新を完了してから checkpoint を保存し、CHECKPOINTED を返す。epoch 末であれば validation と scheduler 更新も終える。保存時は一時ファイルを flush・fsync し、CPU への読み戻しで形式・進行位置・必須キーを確認してから正式名へ切り替える。SEMamba++ の task state には generator、2 discriminator、2 optimizer、2 scheduler が入る。Python / NumPy / CPU / CUDA の RNG、epoch、消費済み micro-batch と dataset identity、best 関連状態も保存する。勾配は更新境界でのみ保存するため、部分的な蓄積勾配は保存しない。

既存 Trainer の再開時には metrics の checkpoint より後を切り詰める。managed 継続では直前成功ジョブの checkpoint だけを受け付け、自動巻き戻しをしない。手動復旧で以前の checkpoint を使う場合はログを退避し、巻き戻しを明示的に記録すること。

MIG では `torch.cuda.mem_get_info` の割り当てインスタンスの使用量を確認し、親 GPU の別 MIG の使用量とは分ける。1024 MiB 未満の基準は維持し、`allow_busy_gpu: false` を変更していない。qsub が設定する CUDA 可視性での判定は未確認。

## 検証

CPU の chunk plan、logical epoch、managed stop / 再開・入力順、failed predecessor、破損、lock、qacct 異常、qsub コマンド生成、MIG 空き判定、既存 mock 再開の 12 テストが成功。ESPnet の公式 iterator 比較 2 テストは `typeguard` がなく import に失敗した。公式 iterator の再検証成功とは扱わない。

既存の対話ジョブ 115720 の H100 MIG 4g.47gb、CUDA fused、VoiceBank、公式幅、micro 4 × 蓄積 2 の短いテスト `data/results/training/job-chain-local-20261104-152728/` では、別プロセスで 2 更新ずつ 6 更新、最終状態 COMPLETED と次のプロセスの追加更新なしを確認した。6 更新の 48 case ID は独立に生成した epoch の先頭順と一致した。続く `data/results/training/job-chain-deadline-20261104-153520/` は 1 秒の内部目標を指定し、最初の更新境界で CHECKPOINTED、別プロセスで step 3 まで再開した。これは実時間経過による停止と再開のスモークで、18.5 時間運転と qsub の保証ではない。

前者のプロセス時間は 31.83、18.93、18.58、4.93 秒、後者は 15.89、17.81 秒。今回加算 107.97 秒、以前の 16,174.17 秒と合わせて 16,282.14 秒、上限 25,200 秒まで 8,917.86 秒。これらは予約時間やポイントではない。

## 残作業

qsub 投入を許可された別の機会に、短時間ジョブ A/B と失敗・破損・完了済み・二重起動を実際に検証する。GPU 型、`CUDA_VISIBLE_DEVICES`、MIG 空き判定、qacct と hold の遅延、ログ・終了状態も確認する。データ順と更新の重複欠落を照合する。時間制限前の保存・復元時間を実測し、十分な余裕を確定する。

TF-GridNet の通常実行では `RecipeDataset.waveform_lengths()` が初回に全発話を読み込む。前回の短時間プローブは長さキャッシュを事前に入れていたため、この初期処理を含む時間は今回測れていない。内部時刻の確認は更新後なので、初期読み込み中には停止できない。3 時間の予約に対する余裕も含めて実測する必要がある。

ログインノード専用の `t4-user-info` は現在の計算ノード上では使えず、SSH は既知ホスト鍵がないため接続できなかった。残ポイントと 2 ジョブの同時実行可否は未確認。公式 chunk TF-GridNet は固定 5,000 更新 checkpoint を使わず、新しい実験として始める。本学習については、ジョブ数、h_rt、GPU、ポイント上限、保存先、停止・再開方法とコマンドを確定・提示し、別途明示承認を得る。現在のユーザー指示は qsub を投入しないことである。
