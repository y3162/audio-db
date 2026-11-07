# P10-C Preflight Agent E: Checkpoint・中断再開・長時間運用

調査日: 2026-11-07。対象は `src/training/core/trainer.py` の状態機械と、その場で呼ばれる checkpoint・ログ・Scheduler。過去の検証文書は根拠にしていない。確度 `CONFIRMED` は、このセッションで一時ディレクトリの CPU 実行が同じ事実を出した項目だけに使う。ソースを読んで実行していないものは `CODE`、実測していない耐久性は `UNKNOWN`。

結論: 公開済みの `step_*.pt` は 1 回の `os.replace` で出る単一 pickle である。その中には、その時点までに実行した Optimizer 更新、`step_batch`、epoch 終端なら plateau と `chunk_history` が入る。Epoch 終端で「Optimizer だけが入って Validation と Scheduler と best 選択が pickle から欠ける」ファイルは残らない。欠けるのは pickle の外（`checkpoint_io`、`data_state.json`、`generator.pt`、sidecar、`status.json`）である。Epoch 途中の公開 checkpoint は、まだ実行していない Validation・plateau・best を含まない。逆向きの窓があり、Validation の成果物とログだけが新しく、学習 checkpoint は前の epoch のまま残る。TF-Locoformer の chunk 経路は `checkpoint_interval: 0` で、Validation の batch 位置を保存しない。Validation の途中でプロセスが消えると、当該 epoch の Optimizer 更新を全部やり直してから Validation を先頭からやり直す。

## 状態遷移

`status.json` を書くのは rank 0 だけ。`fit` は `complete.json` の有無だけを見て再開を拒む。`status.json` と `interrupted.json` は再開条件ではない。

| 状態 | 書く時点 | `complete.json` | `interrupted.json` | 再開 |
|---|---|---|---|---|
| `RUNNING` | `fit` 開始時。定期 checkpoint では更新しない | 触らない | あれば消す | `complete.json` が無ければ、渡した `step_*.pt` から可能 |
| `CHECKPOINTED` | 正常 return。`SIGTERM`、`deadline`、`segment_steps` が、現在の Optimizer 更新とその epoch 終端処理のあと | 書かない | あれば消す | その `step_*.pt` から可能 |
| `INTERRUPTED` | `verification_max_updates` 到達。chunk 経路が `max_epochs` の前に `max_steps` へ到達 | 書かない | 同じ内容を書く | 同じ verification 上限、または `global_step >= max_steps` のときは拒否 |
| `COMPLETED` | 非 chunk はループを抜けたとき。chunk は early stop、`max_epochs` 完了、またはそのどちらでもなく `max_steps` にも達していないとき | 先に書く | あれば消す | `complete.json` があるので拒否 |
| `FAILED` | `Exception` で `stop_reason` が未設定のとき。書き込みに失敗しても例外は残す | 書かない | あれば消す | `complete.json` が無ければ直前の完成 `step_*.pt` から可能 |

`KeyboardInterrupt`、`SystemExit`、`SIGKILL` は `except Exception` に入らない。`status.json` は最後に書けた値のまま残る。通常は開始時の `RUNNING`。

chunk と非 chunk で `max_steps` の終端が違う。

- 非 chunk（`_run_epochs` / `_run_data_epochs`）は、先に `stop_reason` が無ければ `COMPLETED` と `complete.json`。既存テスト `test_official_max_steps_stays_completed` がこの拒否を確認した。
- chunk（`_close_chunk_run`）は、`max_epochs` 未完了の `max_steps` 到達を `INTERRUPTED` にする。epoch 終端で Validation まで終わっていても同じ。`global_step >= max_steps` の再開は `fit` が拒否する。`strict_resume` は `max_steps` の変更を設定不一致にする。`verification_max_updates` だけは `strict_resume` の比較から外してある。

根拠: `trainer.py` の `fit`（185–239、267–273行）、`_write_status`（623–636行）、`_publish_stop`（638–644行）、`_mark_complete`（604–612行）、`_close_chunk_run`（646–657行）、`run.py` の `strict_resume`（201–208行）。

## 保存順序

chunk の 1 epoch（Optimizer 更新 2 回、Validation batch 2、`checkpoint_interval: 0`）で、実 Trainer の呼び出し順は次だった。モデルは `Linear(4, 1)` と `WarmupReduceLROnPlateau`。ループは TF-Locoformer と同じ `_run_chunk_epochs`。

1. `metrics.jsonl` に `run`。`status.json` を `RUNNING`。
2. Optimizer 更新。TF-Locoformer と同じ位置で `sched_g.step_batch()`（`tf_locoformer.py` 117–119行）。
3. epoch 途中の更新は `train` 行だけ。`step_*.pt` は無い（interval 0、epoch 未完了）。
4. epoch 最終更新のあと、Validation の全 batch。batch ごとの checkpoint は無い（`_validate_chunks` 964行の内包表記）。
5. `{epoch}epoch.pt`、`valid.loss.best.pt` と JSON、`valid.si_snr.best.pt` と JSON。`keep_nbest` の外にある `*epoch.pt` を削除。履歴が `keep_nbest` 未満なら `valid.loss.ave_1best.pt` を symlink（`_retain_chunk_models` 1015–1074行）。
6. `metrics.jsonl` に `validation`。
7. `step_schedulers`。Warmup plateau はここで `step_validation`（`optim.py` 173–180行）。重みは変えない。
8. `metrics.jsonl` に `train`。`checkpoint` 欄は、このあと書く予定のパス。
9. TensorBoard を使うときだけ `flush`（今回のプローブは無効）。
10. `step_XXXXXXXX.pt` を一時ファイルへ書き、fsync、読み戻し検証、`os.replace`（`checkpoint.py` 37–58行、`verify_training=True`）。中身は更新後の重み、AdamW、`step_batch` と plateau のあと、`chunk_history`、RNG、`data_state`。
11. `step_*.data_provenance.json`。
12. `checkpoint_io` 行。
13. `data_state.json`。
14. `generator.pt`。
15. ランが止まるときだけ `complete.json` または `interrupted.json` と、終端の `status.json`。

公開された epoch 終端 pickle の実測: `global_step=2`、`step_num=2`、`last_epoch=1`、`chunk_history` に epoch 1。`best_loss` は chunk 経路では更新されない（`_consider_best` は非 chunk 用）。

非 chunk の `_run_epochs`（342–370行）と `_run_data_epochs`（450–487行）も、Validation、Scheduler、`train` ログ、`_save` の順は同じ。違いは保存の条件である。chunk は epoch 完了ごとに rank 0 が保存する。非 chunk は `checkpoint_interval`、`max_steps`、`verification_max_updates`、停止要求、`logical_steps_per_epoch`、`batch_schedule` のときだけ保存する。`validation_interval` を見るのは非 chunk と TrainingData 経路だけ（349–354行、457–462行）。chunk ループはこの値を見ない。出荷 YAML の TF-Locoformer はすべて `checkpoint_interval: 0` かつ `validation_interval: 0` で、chunk 経路なので epoch 終端だけが定期保存になる。

`step_*.pt` を消す処理は無い。epoch ファイルだけを `keep_nbest` の和集合で削る。和集合は valid loss の上位と、有限な `si_snr` の上位である（1023–1026行）。

## 中断窓

「実行済みのものが pickle に入るか」と「別ファイルが先に或は後に残るか」を分けて書く。

| 窓 | きっかけ | `step_*.pt` | Validation / plateau / best | ログと副作用 | 実測 |
|---|---|---|---|---|---|
| epoch 終端の replace 成功 | 正常完了 | その更新の Optimizer、`step_batch`、plateau、`chunk_history`、RNG、data 位置 | pickle に含まれる。`{epoch}epoch.pt` と best は replace より前に既にある | `train` と `validation` は replace より前。`checkpoint_io` と `generator.pt` と `data_state.json` は後 | `CONFIRMED`（順序プローブ） |
| replace 成功の直後に停止 | `KeyboardInterrupt` を `save_atomic` の return 直後に注入 | 上記の完全な pickle がある | pickle に含まれる。epoch ファイルもある | `checkpoint_io` 無し。`generator.pt` 無し。`data_state.json` 無し。step の sidecar 無し。ディレクトリの `data_provenance.json` はある。`status` は `RUNNING` | `CONFIRMED` |
| `_save` に入る前に停止 | epoch 終端の Validation と Scheduler のあと | その step のファイルは無い | `validation` 行と best / epoch ファイルはある。Scheduler はメモリだけで、前の pickle は plateau 前 | `train` 行の `checkpoint` 欄が存在しないパスを指す。`status` は `RUNNING` | `CONFIRMED` |
| Validation の batch ループ中に停止 | epoch 2 の最初の Validation batch で `KeyboardInterrupt` | 前 epoch の `step_00000002.pt` だけ | 当該 epoch の epoch ファイルは無い。前 epoch の best はそのまま | 当該 epoch の `train` が 1 行ある（step 3）。`validation` 行は前 epoch だけ。`status` は `RUNNING` | `CONFIRMED` |
| `_retain_chunk_models` の return 直後に停止 | epoch 2 が loss でも `si_snr` でも上位 | 前 epoch の step だけ。`chunk_history` は epoch 1 だけ。`last_epoch` は 1 | `2epoch.pt` と best と symlink は epoch 2。`1epoch.pt` は削除済み。`1epoch.data_provenance.json` は残る | `validation` 行はまだ無い（retain のあと、ログの前）。`status` は `RUNNING` | `CONFIRMED` |
| その状態から、やり直した epoch の loss が悪い | 上のディレクトリを再開 | 再開自体は完了し `step_00000004.pt` まで書く | `valid.loss.ave_1best.pt` が欠落した `1epoch.pt` を指す。`valid.loss.ave.pt` も切れる。epoch 重みファイルは両方無い | 学習の `COMPLETED` は付く。選択用の平均リンクは切れたまま | `CONFIRMED` |
| epoch 途中の `SIGTERM` | 更新 1 の完了時にフラグ | その更新の pickle。`step_num=1`、`last_epoch=0`、`chunk_history=[]` | Validation も plateau も best も未実行 | `train` と `checkpoint_io` がある。`status` は `CHECKPOINTED`。`complete.json` は無い | `CONFIRMED` |
| Validation の最初の batch で `SIGTERM` | フラグだけで例外にしない | epoch 終端の完全な pickle | Validation 2 batch と plateau が終わってから保存 | `status` は `CHECKPOINTED`。当該 epoch はやり直さない | `CONFIRMED` |
| 蓄積の 1 micro-batch 目の backward 後 | 非 chunk、accumulation 2 | 直前の完成 step だけ。当該 Optimizer 更新は未保存 | その step では未実行 | `status` は再開完了後に上書きされる。中断時点の step は `step_00000001.pt` だけ | `CONFIRMED`（同一プロセスで連続実行と重みが一致） |
| `step_*.pt` の 2 回目の `save_atomic` が `ENOSPC` | epoch 終端の Validation のあと | `step_00000001.pt` だけ | `1epoch.pt` と best と `validation` 行と step 2 の `train` 行がある | `status` は `FAILED`。`complete.json` は無い | `CONFIRMED` |
| `torch.save` の return 直後に `SIGKILL` | replace 前 | 以前のバイト列 `PREVIOUS` のまま | 当該保存は無い | `.killed.pt.*.tmp` が残る。`finally` の unlink は走らない | `CONFIRMED`（return code -9） |
| ファイルが pickle でない | `load_checkpoint` | 読み戻しを `TrainingError: unreadable` で拒否 | — | 検証失敗時は replace しない。既存テスト 3 件も成功 | `CONFIRMED` |
| `metrics.jsonl` の最終行が途中 | `truncate_after` | checkpoint 側は無関係 | — | `JSONDecodeError`。ファイルは途中行のまま | `CONFIRMED` |
| chunk が epoch 途中で `max_steps=2` | 発話長 24、chunk 8 で更新 5 のうち 2 | その 2 更新。`last_epoch=0`、history 空、Validation 0 | 未実行 | `INTERRUPTED` と `interrupted.json`。同じ `max_steps` の再開は拒否。`max_steps` を 8 にした比較は strict の条件で拒否。strict 無しの 5 は保存済み 2 更新をやり直さず残り 3 更新と Validation 5 batch | `CONFIRMED` |
| `verification_max_updates=1` | chunk、epoch 途中 | step 1。history 空、`last_epoch=0` | 未実行 | 同じ上限は拒否。上限 3 では続きの更新だけ | `CONFIRMED` |
| Dataset identity の `size` を変える | `CHECKPOINTED` のあと | 保存済み step は残る | — | `checkpoint dataset identity does not match this run` | `CONFIRMED` |
| `complete.json` がある | 1 epoch 完了のあと | 最終 step はある | — | `training is already complete` で再開拒否 | `CONFIRMED` |

`data_state.json` を再開は読まない。位置は pickle の `data_state` と `micro_batches_consumed`。chunk で `check_resume` が無い Dataset は `_check_chunk_state` のあと return し、`micro_batches_consumed` と `update_in_epoch` の一致を見ていない（1355–1384行）。通常の `_save` は両方を同じ cursor から書くので、1 個の pickle の中では割れない。

RNG は Validation の `finally` で戻してから Scheduler と `_save` に入る（940–941行、1157行）。checkpoint の RNG は Validation 中の消費を含まない、という順序はコードとイベント順で一致する。RNG バイト列の比較はこのセッションではしていない（`CODE`）。

## Validation 中断時の再開

TF-Locoformer は `TFLocoformerTask` を chunk ループへ渡す（`run.py` 44–45行、118–135行）。Validation は epoch の最後の Optimizer 更新が戻ったあと、Scheduler より前に 1 回だけ走る。途中 batch のカーソル、完了 batch 数、部分集計は checkpoint に無い。

`checkpoint_interval: 0` のとき、最後に公開される学習 checkpoint は前の epoch 終端（または、それより前に `SIGTERM` / segment / verification / `max_steps` が作った epoch 途中の step）である。Validation のループ中にプロセスが消えると、再開はそのファイルから始まる。

このセッションのミニ chunk（1 epoch あたり Optimizer 更新 2、Validation batch 2）では次だった。

- 停止時: メモリ上の更新は 4、Validation batch は 3。ディスクの学習 checkpoint は step 2 だけ。`status` は `RUNNING`。`2epoch.pt` は無い。
- 再開: Optimizer 更新を 2 回やり直した。これは epoch 2 の全部である。Validation batch も 2 回、つまり先頭から全部である。残りの 1 batch だけにはならなかった。
- 同じプロセスの連続実行と、重み・`chunk_history`・Scheduler が一致した。完了済みの epoch 1 はやり直していない。

したがって、出荷設定の TF-Locoformer で Validation の途中に `SIGKILL` されると、やり直すのは Validation だけではなく、直前の epoch checkpoint 以降の当該 epoch の学習更新全体と、Validation イテレータの先頭からの全 batch である。Validation だけを再開する分岐は無い。前の epoch までの Optimizer 更新、plateau、`chunk_history` は、その epoch の `step_*.pt` に入っているのでやり直さない。

`SIGTERM` は別である。ハンドラはフラグを立てるだけで、確認は Validation と Scheduler と保存のあと（567–568行、681–684行）。ミニ実行では Validation の最初の batch でフラグを立てても batch は 2 のまま終わり、plateau も入った `CHECKPOINTED` になった。再開は次の epoch からで、その epoch の学習はやり直さなかった。依頼文の 1 回約 5.61 時間はここでは測っていない。コード上、その時間が 1 回の Validation 呼び出しの長さなら、`SIGTERM` はその呼び出しが終わるまで保存しない。猶予のあと `SIGKILL` になると、上の「epoch 全体の学習と Validation の最初から」になる。

Validation の対象は `train=False` で `num_samples=None` の全件イテレータ（686–700行）。出荷 YAML に `validation_max_updates` は無い。研究 YAML `tf_locoformer_voicebank.yaml` の valid は 872 件、DNS の受理 recipe の valid は 36,000 件というデータ側の件数であり、Trainer はどちらの件数も途中保存しない。`num_samples_per_epoch` は学習側の発話数で、Validation の長さではない。

TF-Locoformer の `validation_report` は `si_snr` を返し `si_sdr` は返さない（124–147行）。`validation_case` 行は `si_sdr` がある報告だけなので、このモデルの Validation は case 行を積まない。

## Issue

共通フィールド: ID、タイトル、種別、重大度、確度、経路、場所、条件、ディスクに残るもの、再開、根拠、残る不確実さ。

### P10P-E-001

- タイトル: epoch 終端では Validation 成果物が学習 checkpoint より先に公開される
- 種別: 欠陥（複数ファイルの途中状態）
- 重大度: 高
- 確度: `CONFIRMED`
- 経路: chunk。非 chunk で epoch 終端に Validation と保存が重なるときも同じ順序（`CODE`）
- 場所: `trainer.py` `_validate_chunks` 977–990行、`_retain_chunk_models` 1029–1074行、`_run_chunk_epochs` 559–588行
- 条件: epoch の Optimizer 更新が終わり、Validation または best 選択がファイルを書いたあと、`step_*.pt` の `os.replace` 前にプロセスが消える。`ENOSPC` も同じ位置で起きる
- ディスクに残るもの: 新しい `{epoch}epoch.pt`、best、平均 symlink、古い epoch 重みの削除、`validation` 行、場合によっては当該 step の `train` 行。学習 checkpoint は前の epoch。`ENOSPC` のときは `status=FAILED`。シグナルで `Exception` にならない停止は `RUNNING`
- 再開: 前の `step_*.pt` から当該 epoch の学習と Validation をやり直す。消した epoch ファイルと、先に書いた best は戻さない
- 根拠: 順序プローブ、retain 直後の停止、`ENOSPC`、`_save` 前の停止
- 残る不確実さ: 実 TF-Locoformer のファイルサイズと 5.61 時間の壁時計は未測定

### P10P-E-002

- タイトル: Validation 途中の消失は、当該 epoch の学習ごとやり直しになる
- 種別: 運用リスク（途中 Validation checkpoint は存在しない、という動作）
- 重大度: 高
- 確度: `CONFIRMED`
- 経路: chunk。TF-Locoformer の出荷 YAML は `checkpoint_interval: 0`
- 場所: `_run_chunk_epochs` 549–588行、`_validate_chunks` 946–995行、`configs/training/tf_locoformer*.yaml`
- 条件: Validation の batch ループ中の `SIGKILL`、または `Exception` にならない停止。`SIGTERM` はこの窓に入らず、Validation 完走後に `CHECKPOINTED` になる
- ディスクに残るもの: 直前に公開された `step_*.pt`。進行中 Validation の batch 番号は無い
- 再開: その step の次の Optimizer 更新から。interval 0 なら epoch の先頭から。Validation は先頭から。完了済み epoch はやり直さない
- 根拠: 更新 4・Validation batch 3 で停止し、再開は更新 2 と Validation batch 2。連続実行と一致。`SIGTERM` は batch 2 まで完走し、再開の更新は次 epoch の 2 だけ
- 残る不確実さ: 公式規模の 1 epoch の Optimizer 更新数と学習時間は、このセッションでは数えていない

### P10P-E-003

- タイトル: best 選択の削除と symlink が、学習 checkpoint より先に確定する
- 種別: 欠陥
- 重大度: 高
- 確度: `CONFIRMED`
- 経路: chunk の `_retain_chunk_models`
- 場所: 1022–1074行、`average.py` `replace_symlink` 21–25行
- 条件: 新しい epoch が loss と `si_snr` の両方で `keep_nbest` の外に前の epoch を押し出した直後に停止する。再開後の同じ epoch の loss が、残っている history 上の古い epoch より悪い
- ディスクに残るもの: 停止直後は新しい epoch 重みと best。古い `*epoch.pt` は無い。sidecar の `*epoch.data_provenance.json` は残る。再開が完了すると、平均リンクが存在しない `1epoch.pt` を指し、epoch 重みファイルが両方無くなる。学習は `COMPLETED` になり得る
- 再開: 学習状態は前の step から復元できる。選択用ファイルの穴は復元しない
- 根拠: `keep_nbest=1` のミニ実行。停止前のファイル一覧に `1epoch.pt` が無く、再開後の `valid.loss.ave_1best.pt` の `exists` は false
- 残る不確実さ: `valid.loss.best.pt` の中身が殺された実行の重みのままかは、バイト比較をしていない。書く条件は「今の epoch が best 行のときだけ」（1037–1044行、`CODE`）

### P10P-E-004

- タイトル: `step_*.pt` の replace 後も、推論用ファイルと一部のログと status は別更新である
- 種別: 欠陥（学習再開の pickle は揃っている）
- 重大度: 中
- 確度: `CONFIRMED`
- 経路: 共通の `_save`（1178–1197行）と、停止時だけの `_publish_stop`
- 条件: replace 成功後、sidecar、`checkpoint_io`、`data_state.json`、`generator.pt` の前に停止する。定期保存では `status.json` を書かない
- ディスクに残るもの: 完全な学習 pickle（plateau と `chunk_history` を含む）。`generator.pt` と `data_state.json` と step sidecar と `checkpoint_io` が無い。`status` は `RUNNING`。ディレクトリの `data_provenance.json` はある
- 再開: sidecar が無くてもディレクトリの来歴があれば `assert_checkpoint_provenance` は通る、という分岐は `CODE`（`provenance.py` 181–184行）。この穴のファイルでの再開実行はしていない
- 根拠: `step_00000002.pt` の `last_epoch=1` と history、同時に `generator.pt` 不在
- 残る不確実さ: ディレクトリ fsync が無く、replace 後の電源断で新しいディレクトリエントリが残るかは `UNKNOWN`

### P10P-E-005

- タイトル: chunk の `max_steps` 到達は、良い epoch checkpoint でも再開口が狭い
- 種別: 欠陥
- 重大度: 中
- 確度: `CONFIRMED`
- 経路: chunk。非 chunk は同じ上限で `COMPLETED` になり、`complete.json` が再開を拒む（既存テスト、`CONFIRMED`）
- 場所: `_close_chunk_run` 646–657行、`fit` 206–207行、`run.py` 204–208行、`job_chain.py` 164–165行
- 条件: `max_epochs` の前に `global_step` が `max_steps` に達する。epoch 途中でも epoch 終端でも `INTERRUPTED`。`job_chain` は `CHECKPOINTED` と `COMPLETED` 以外を失敗にする（`CODE`）
- ディスクに残るもの: その step の pickle と `interrupted.json`。`complete.json` は無い
- 再開: 同じ `max_steps` は `checkpoint global_step is already at max_steps`。上限を上げると strict 比較が `checkpoint configuration does not match this experiment`。strict 無しなら保存済み更新はやり直さない
- 根拠: 更新 5 の epoch を step 2 で停止。続けて max_steps 5 にした非 strict 再開は追加更新 3 と Validation 5
- 残る不確実さ: `job_chain` の qsub 実ジョブはこのセッションでは動かしていない

### P10P-E-006

- タイトル: `metrics.jsonl` の途中行が再開の切り詰めを止める
- 種別: 欠陥
- 重大度: 中
- 確度: `CONFIRMED`（関数）。呼び出し位置は `CODE`
- 場所: `metrics_log.py` `append_record` 24–27行、`truncate_after` 30–45行。`fit` は `try` の外、214行で呼ぶ
- 条件: 追記の途中でプロセスが消え、最終行が JSON として閉じない
- ディスクに残るもの: 途中行を含む `metrics.jsonl`。完成済み `step_*.pt` はそのまま
- 再開: `JSONDecodeError`。`FAILED` は書かない。ファイルも直さない
- 根拠: 途中行に対する `truncate_after` が `Unterminated string`。完全な未来行（step 99）は checkpoint の step 以下へ切り詰めた
- 残る不確実さ: 実ランの追記途中を `SIGKILL` では再現していない。使ったのは同じ関数への直接入力

### P10P-E-007

- タイトル: 保存中の `SIGKILL` は前の checkpoint を残し、一時ファイルを残す
- 種別: 動作の確認。前のファイルが守られる点は意図どおり。一時ファイルは残る
- 重大度: 低
- 確度: `CONFIRMED`
- 場所: `checkpoint.py` 41–58行。`finally` は `SIGKILL` では走らない
- 条件: `torch.save` が戻ったあと、`os.replace` の前
- ディスクに残るもの: 置換前のファイル。`.{name}.*.tmp`
- 再開: 完成した前の `step_*.pt` を指定すれば可能。一時ファイルは再開元にならない。名前が `.` で始まるので、空ディレクトリ判定の除外には入る（1507–1513行、`CODE`）
- 根拠: 子プロセス return code -9。対象バイト列は `PREVIOUS` のまま。tmp が残った。壊れたバイト列は `unreadable`
- 残る不確実さ: 検証読み戻し中の `SIGKILL` は別タイミングとしては未実施。replace 前であることは同じ構造

### P10P-E-008

- タイトル: epoch 途中の公開 checkpoint は、その epoch の Validation と plateau と best を含まない
- 種別: 設計どおりの記録。クラッシュで欠落したわけではない
- 重大度: 情報
- 確度: `CONFIRMED`
- 経路: chunk の `SIGTERM`、`verification_max_updates`、`max_steps`。非 chunk で interval が epoch 終端と一致しないときも、保存 step で Validation が走っていなければ同じ
- 場所: `train_update` 内の `step_batch` と、epoch 完了時だけの `step_schedulers`（`optim.py` 6–8行、89–111行）
- 条件: epoch の最終更新より前に保存する
- ディスクに残るもの: その Optimizer 更新と `step_batch` 後の Scheduler。`last_epoch` は前の epoch のまま。`chunk_history` は増えない
- 再開: 次の更新から。Validation は epoch を終えたときだけ、先頭から 1 回
- 根拠: `SIGTERM` で更新 1、Validation 0、`step_num=1`、`last_epoch=0`、history 空、`status=CHECKPOINTED`。再開は残り更新だけを足して連続実行と一致。verification 上限 1 も history 空
- 残る不確実さ: 無し（このミニ設定の範囲）

### P10P-E-009

- タイトル: 非 chunk の ExponentialLR は、Validation が走らない epoch 終端でも進む
- 種別: 設計どおり。MP-SENet の `step_schedulers` は validation 引数を捨てて `scheduler.step()` する（`mp_senet.py` 86–90行）
- 重大度: 情報
- 確度: `CONFIRMED`（既存 `test_epoch_scheduler_and_resume_do_not_follow_the_validation_interval` をこのセッションで実行）
- 経路: 非 chunk。TF-Locoformer の chunk 経路では epoch 終端に必ず Validation してから plateau する
- 条件: `validation_interval` が epoch 長の約数ではない。epoch 終端の保存（verification や interval）が同じ step にある
- ディスクに残るもの: Scheduler は進んだ状態。その step の Validation 記録は無い
- 再開: 既存テストは連続実行と重み・学習率が一致
- 根拠: 当該テストが `scheduler_steps=1`、`validations=0` で通過
- 残る不確実さ: そのテストは pickle の `sched_*` を直接比較していない。再開後の重み一致はテストが確認している

### P10P-E-010

- タイトル: legacy chunk の再開検査は学習 Dataset だけで、Validation Dataset は別物になり得る
- 種別: 欠陥の可能性
- 重大度: 中
- 確度: `CODE`
- 経路: `RecipeDataset` を chunk に渡す研究設定（`tf_locoformer_voicebank.yaml`）。DNS on-demand は train identity に snapshot 全体の hash が入る（`snapshot_reader.py` 56–58行、`dns_ondemand.py` 79–106行）ので、valid の case 集合が変われば hash が変わる
- 場所: `dataset.py` `identity` 105–124行、`_check_resume` 1365–1366行、`_chunk_state` 862行
- 条件: 同じ train identity のまま validation セットだけを差し替える
- ディスクに残るもの: 古い Validation で作った `chunk_history` と best
- 再開: train identity が一致すれば拒否しない。次の epoch 終端で新しい Validation を history に足す
- 根拠: ソース。差し替え実験はしていない
- 残る不確実さ: VoiceBank の実ファイルで identity がどこまで固定されるかは未実行

### P10P-E-011

- タイトル: DDP の他 rank 死亡は、保存と status を同じ原子操作にしていない
- 種別: 未実測のリスク
- 重大度: 中
- 確度: `CODE`
- 場所: `distributed.py` `barrier` 89–91行。保存は barrier の前、rank 0 だけ（`_run_aligned_chunk_epochs` 783–796行）。`barrier` に timeout 引数は無い
- 条件: rank 0 が `step_*.pt` を replace したあと、他 rank が collective または barrier で死ぬ
- ディスクに残るもの: rank 0 の pickle は残り得る。例外が `Exception` なら `FAILED`。タイムアウトまで待つ実装にはなっていない
- 再開: 全員が同じファイルを、DDP wrap の前に `load_state_dict` する（`fit` 190–194行のあと 231行）。world size は chunk state の全フィールド比較で拒否する（865–873行）
- 根拠: ソース。死亡 rank の実行はハングし得るのでしていない
- 残る不確実さ: NCCL / gloo のプロセスグループ既定 timeout と、共有 FS 以外で rank 0 のファイルが見えるかは未確認

## 長時間運用でコード上確定していること

- epoch 境界: chunk はここでのみ定期保存する（interval 0）。保存は Validation と plateau のあと。
- checkpoint 境界: `step_*.pt` は step ごとに別名。上書きしない。削除しない。150 epoch なら学習 pickle が 150 個残る。各ファイルはモデル、AdamW、Scheduler、`chunk_history` 全体を持つ。history は epoch ごとに 1 要素増える。
- validation 境界: 1 epoch に 1 回、全 batch のあとで集計と best。途中境界は無い。
- 強制終了: `SIGTERM` は現在の更新に加え、その更新が epoch 終端なら Validation 完走後。`SIGKILL` は replace 前なら前の step、replace 後なら新しい pickle と未完了の副作用。
- 部分ファイル: 学習 pickle の replace 前は前のファイル。tmp は残り得る。JSONL の途中行は再開を止める。`write_json` は replace だが fsync は無い（`metrics_log.py` 17–21行、`CODE`）。
- ディスク不足: 前の `step_*.pt` は残り、Validation 成果物は先に残り、`FAILED`。
- DataLoader 例外: chunk の読み失敗は `fit` の `Exception` で `FAILED`。既存 `test_injected_failure_is_recorded_and_does_not_close_the_run` が、更新中の例外で `FAILED` かつ `complete.json` 無しを確認した。prefetch は位置に数えない、は `CODE`（`_load_group` は更新が消費した cursor だけを返す）。
- DDP: 上記 P10P-E-011。仮想 world は 1 プロセスの逐次平均で、勾配 all-reduce ではない（749–754行、`CODE`）。
- ログ: 更新ごとに `train` 1 行、保存ごとに `checkpoint_io` 1 行、epoch ごとに `validation` 1 行。TF-Locoformer は case 行を足さない。ローテーションは無い。
- TensorBoard: 再開時 `purge_step=resume_step+1`（`tensorboard.py` 45行、`CODE`）。今回は再実行していない。JSONL の `truncate_after` とは別実装。
- 数百 epoch 後: 状態フィールド自体に epoch 数の上限は無い。増えるのは history とファイル数。このセッションでは 2 epoch まで。

MossFormerGAN は Validation の推論 export のあと、`step_schedulers` が最良重みへ戻してから `_save` する（`mossformergan.py` 135–172行、`CODE`）。TF-Locoformer の `step_schedulers` は学習率だけで、この巻き戻しは無い。

## 未監査

- 実 GPU、実 TF-Locoformer、DNS 36,000 件または VoiceBank 872 件の Validation 時間。5.61 時間は依頼の推定であり、ここの測定ではない。
- 1 epoch の Optimizer 更新数と、Validation に加えてやり直される学習時間。
- DDP rank の実死亡、NCCL timeout、共有していない checkpoint ディレクトリ。
- replace 後の電源断、ディレクトリ fsync、NFS。
- `metrics.jsonl` 追記中の実 `SIGKILL`。関数には途中行を渡した。
- prefetch スレッドだけが落ちる場合。
- MossFormerGAN の巻き戻し窓の実行。
- legacy chunk で validation セットだけを差し替えた再開。
- TensorBoard `purge_step` の再実行。
- `job_chain` の qsub。ソース上、`INTERRUPTED` は次のジョブへ渡さない。
- 数百 epoch のディスク使用量の実測。
- P10P-E-004 の「sidecar 無し、ディレクトリ来歴あり」からの実際の `fit` 再開。

## 実行したテスト

すべて CPU。既存の checkpoint ディレクトリ、DB、`data/results`、ソース、YAML、テスト期待値は変更していない。GPU は使っていない。

1. `/net/spring/work/seki/exp/audio-db/.venvs/se/bin/python /tmp/p10e_ckpt_probe.py`
   - 約 13 秒。一時ディレクトリだけ。実 Trainer、実 `save_atomic`、実 `WarmupReduceLROnPlateau`。モデルは `Linear(4, 1)`。波形生成にプロセスごとに変わる `hash()` を使ったので、プロセスをまたいだ重みの一致は主張しない。同一プロセスの連続実行と再開は一致した。
   - 成功: 保存順序、epoch 途中 `SIGTERM`、Validation 中の `SIGTERM`、Validation 中の `KeyboardInterrupt`、retain 直後、削除後の再開、replace 直後、ログ先行、JSONL 切り詰め、途中行、破損と `SIGKILL`、verification 上限、identity と `complete.json`、蓄積の途中、`ENOSPC`。
   - 最初の `max_steps` 実験は再開側の `max_epochs` を取り違えて失敗した。修正後の再実行が P10P-E-005 の数値である。
   - `SIGTERM` 直後の `status.json` は、再開で上書きされる前に別実行で読んだ。`CHECKPOINTED`、`complete.json` 無し。
2. `python -m unittest tests.training.test_checkpoint_atomic tests.training.test_run_status -v`
   - 9 件、4.649 秒、成功。原子的置換の失敗が前のファイルを残すこと、途中バイト列の拒否、verification 上限の再開、`complete.json` の拒否、更新中例外の `FAILED`、Validation interval と epoch Scheduler の分離。

長時間になるため走らせていないもの: `tests/training` 一式、2 rank の DDP、公式規模の chunk 計画、1 回の DNS Validation。公式規模を 1 epoch だけ動かしても、Validation は依頼推定で約 5.61 時間であり、その途中停止の再開はさらに当該 epoch の学習時間と同じだけかかる。ミニ実行がすでに制御フローを数えているので、同じ質問のための長時間実行は不要だった。

再現の要点: 一時ディレクトリに chunk Dataset（長さ 12、chunk 8、batch 1、`max_updates` 無しで更新 2、Validation batch 2）を置き、`Trainer._run_chunk_epochs` を `checkpoint_interval=0` で回す。停止は `KeyboardInterrupt` または `signal.SIGTERM` のフラグ。比較相手は同じプロセスの無停止実行。`max_steps` の途中停止だけ長さ 24（更新 5）を使う。
