# P10-C Preflight Phase 2 — Agent G 相互レビュー

Phase 1 の独立報告 A–G を突き合わせた。各報告の `CONFIRMED` は正解にしていない。このレビューがコードを読み直したのは、HIGH または BLOCKER、および依頼で指定された突き合わせ点だけである。新しい推測は CONFIRMED BUG にしていない。

本番コード、YAML、DB、Snapshot は変更していない。GPU は使っていない。学習プロセスは止めていない。追加の読み取りは、`reproduction.duckdb` の不在確認、公式 TF-GridNet YAML の `git show`、スナップショット `mp_senet.vb28.ondemand.official@1` の `case.frames` 集計だけである。音声の再描画、PESQ の再呼び出し、他エージェントの CPU プローブの再実行はしていない。

「過去テストが PASS した」ことだけを根拠に、危険とも安全とも判定していない。テストが公式比較を見ていない、という各報告の説明は、欠陥の証拠にしていない。

## 矛盾表

同じ現象に対して、深刻度または BLOCKER 判定が割れているもの。機構がコード上一致していても、判定が違えば載せている。

| 現象 | 報告と判定 | このレビューで残る事実 |
|---|---|---|
| MP-SENet の検証 PESQ | A-002 は HIGH / BLOCKER（生成前の hop 切り詰め）。A-003 は HIGH / NON-BLOCKER（失敗を平均から外す）。G-001 は高で、切り詰めと「公式は -1 を平均に入れる」を一つの Issue にしている | 切り詰めは `validation_report` が `align_validation_length` のあとで生成する。公式 `train.py` は全長を STFT する。失敗の集約も違う。公式 `eval_pesq` は例外を -1 にして `np.mean` し、現行は `ok` だけを平均する。「長さが hop の倍数でないと公式 PESQ が必ず -1」は、A の pesq 0.0.4 実測（16050 と 16000 で例外にならない）と両立しない。G はその Python で `pesq` を呼んでいない |
| `valid.si_snr` の符号と best ファイル | G-003 は高（`valid.si_snr.best` を結果に使う場合）。B-002 は中 / NON-BLOCKER | 公式 criterion の文字列は `valid / si_snr / max`。`SISNRLoss` は `name` が無いとき統計名を `si_snr_loss` にする。公式 Trainer は `reporter.has` が偽の criterion から best を作らない。現行は `si_snr` を最大化する。TF-GridNet は正の SI-SDR を符号反転して入れ、TF-Locoformer は正の dB を入れる。`valid.loss` の min と loss の n-best は、この差の外にある |
| MossFormerGAN の crop | A-011 は HIGH / NON-BLOCKER。G-004 は高（検証の 2 秒が変わり、半減と早期停止の epoch が分かれる） | 公式は `__getitem__` ごとの `random.randint`。現行は `mix_seed("crop", ...)` で epoch 内 1 か所。protocol の検証 crop は `per_epoch_random`、窓は 32000。開始位置の実 WAV 比較はどちらもしていない。停止 epoch が実際にずれた測定は無い |
| MossFormerGAN の BatchNorm | A-008 は HIGH / BLOCKER（Generator の `BatchNorm1d` も含む）。D-002 は高。BLOCKER という語は無い。Generator は LayerNorm で running stats が無い、と書く。G はこの差を Issue にしていない | 公式は Generator と Discriminator を `SyncBatchNorm.convert_sync_batchnorm` してから DDP する。現行は SyncBatchNorm を呼ばず、Discriminator は `broadcast_buffers=False` で、micro-batch 前に rank 0 の buffer を 1 回配る。検証 forward は rank 0 だけ。`conv_module.py` の `BatchNorm1d` は `Conformer_ConvModule` の中だけで、`MossFormerGAN_SE_16K` からは構築されない。生きているのは Discriminator の `BatchNorm2d` 6 個。1 プロセスでは DDP しない |
| validation の 21 batch | B-001 は中 / NON-BLOCKER の Issue。D は公式と同じ prefix なので欠陥ではない。G の H6 棄却は、batch が world で割り切れない余りであり、この 21 ではない。F は時刻表に 2908 / 2887 / 破棄 21 と書く | 公式 `validate_one_epoch` は短い rank の終了で打ち切る。現行 `plan_chunk_epoch` は `updates = min(raw_counts)` の prefix だけを残す。短い発話やキャッシュ端とは別、という B と D の切り分けはコードの形と一致する。整数 21 はこのレビューでは再計算していない |
| `reproduction.duckdb` の不在 | C-001 は高 / BLOCKER。A は DB を開いていない。F は学習が DB を開かないと書く。G は snapshot を開いていない | 2026-11-07 のこの作業ディレクトリに `data/db/reproduction.duckdb` と `data/staging/reproduction` は無い。`open_training_data` はスナップショットだけを開き、DuckDB を開かない。`doctor` は不在を `NOT_TESTED`、`required=False` にする。スナップショット 6 件の自己ハッシュ一致は C の測定であり、このレビューでは再計算していない |
| スモーク上限が `COMPLETED` になる経路 | B-005 は高 / BLOCKER。無修正の p10b は `verification_max_updates: 2` で `INTERRUPTED` であり、本格 YAML が無い。F-001 は高 / BLOCKER（本格 YAML を作るとき、または p10b を編集して使うとき）。`verification_max_updates` 以外の cap が残ると `COMPLETED` と `complete.json` になると書く。MP-SENet / MossFormerGAN の official YAML 単体は NON-BLOCKER | 無修正 p10b は `_at_verification_limit` で `INTERRUPTED` になり、`complete.json` は書かない。`verification_max_updates` を外し、`chunk_iterator.max_updates` や `validation_max_updates`、`data.max_updates_per_epoch`、`data.validation_max_batches` を残すと、ループは公式の終了条件で `COMPLETED` になる。`_close_chunk_run` の最後の else が無条件に `COMPLETED` を書く枝は、F が書くとおり現行の while からは届かない |
| MP-SENet の再開と ExponentialLR | A-004 は HIGH / BLOCKER。公式は `last_epoch` で scheduler を作り直し、lr がもう一度 0.99 倍され、その epoch の先頭からやり直す。`best_pesq` は 0 に戻る。G の H9 は棄却。現行の `state_dict` 再開が途中から学習率を変える証拠は無い、と書く | 公式 `train.py` は checkpoint の `epoch` を `ExponentialLR(..., last_epoch=)` に渡し、`range(max(0, last_epoch), ...)` で回し、`best_pesq = 0` で始める。現行は `sched_g.load_state_dict` と保存済み `best_pesq` を使う。A の lr 数値は A の CPU 測定で、このレビューは再実行していない |
| 検証中の消失 | E-002 は高。当該 epoch の学習と検証を先頭からやり直す。F-008 は MEDIUM / NON-BLOCKER。途中 checkpoint を新設しない制約の中の運用リスク | chunk 経路は検証と scheduler のあとで `step_*.pt` を書く。`checkpoint_interval: 0` のとき、検証ループ中にプロセスが消えると、直前 epoch の pickle から当該 epoch をやり直す、という順序はソースにある。捨てる壁時計は E と F の外挿で、このレビューの測定ではない |
| 2 GPU YAML を 1 プロセスで起動 | A は、蓄積 4 になるという YAML の注意が `batch.py` と一致する、と書き、Issue にしていない。F-003 は HIGH / BLOCKER | `plan_batches(4, 1, 1)` は micro 1、accumulation 4。再現データの経路はこの組を拒否しない。chunk モデルは accumulation が 1 でないと `TrainingError` になる |

5000 対 5001 は、この表の割れには入らない。A-001 と G-002 はどちらも中で、A は NON-BLOCKER である。F は YAML の間隔 5000 が公式の設定値と一致すると書き、インクリメントの前後は見ていない。F の一致は、A と G の 1 更新差を否定しない。

## 指定した突き合わせ

### MP-SENet の 5000 対 5001

公式ループは更新のあと、`steps += 1` の前に `steps % validation_interval == 0 and steps != 0` を見る。`steps` の初期値は 0 なので、ログ上の 5000 は 5001 回目の optimizer 更新のあとである。現行 `_run_data_epochs` は更新後に `global_step += 1` し、`global_step % validation_interval == 0` で検証する。5000 回目の更新で保存する。checkpoint も同じカウンタである。

ExponentialLR は公式も現行も epoch の終わりだけである。この 1 更新差は epoch 境界のずれではない。BLOCKER に上げる材料は、このレビューでは増えていない。

### PESQ

二つに分ける。

生成前の切り詰めは残る。公式検証は `split=False` の全長を `mag_pha_stft` に入れ、`mag_pha_istft` に長さ引数は無い。現行 `MPSENetTask.validation_report` は、長さが hop の倍数でなければ先に切り、その波形で `_generate` する。MP-SENet の on-demand protocol の検証 crop は `full_length` なので、この切り詰めはデータ計画の窓とは別である。

スナップショット `36ab4590…` の `case.frames` は sample rate 16000 で、valid 824 件のうち 807 件が 100 の倍数ではない。倍数は 17 件。短すぎて `align_validation_length` が拒否する件数は 0。train 11572 件では 11032 件が倍数ではない。C は、レシピ文書の再サンプル長 `(source_frames + 1) // 3` が case の `frames` と 12396 件一致した、と報告している。このレビューは WAV を再描画していない。検証がこの `frames` を波形長として使うなら、A-002 の分岐は valid の大半に入る。G が「16 kHz WAV が無く未計測」とした件数は、case の `frames` から数えると未計測ではない。

「短い iSTFT を公式 PESQ に渡すと例外になり -1 になる」は維持しない。G の 297/300 はゼロ波形の iSTFT 長の差であり、`pesq` の呼び出しではない。A は pesq 0.0.4 で、長さ 16050 と 16000 がどちらも `4.643888473510742` になり、例外にならなかったと報告している。無音は別で、A は raw が `NoUtterancesError`、現行 `metric_targets` が `None`、`score_pesq` が `na` だったと報告している。例外を -1 として全件平均する公式と、`na` を平均から外す現行の差は、その失敗が起きたときに残る。VoiceBank で `score_pesq` の 0.1 境界が何件 `na` になるかは、A も G も未計測のままである。

現行 `best_pesq` の初期値は `None` で、最初の有限 PESQ を保存できる。公式の初期値は 0 で、平均が 0 以下なら `g_best` を書かない。通常の正の PESQ だけなら、この初期値差は best を分けない。-1 が平均を 0 以下に引くときに効く。

### `valid.si_snr` の符号

公式 MS-SNSD YAML の `best_model_criterion` は `valid / si_snr / max` と `valid / loss / min`。`si_snr` criterion の `conf` は `eps` だけで、`wrapper_conf.weight` は 0 である。DNS の保存設定も同じ二つの criterion を持つ、と B が書き、TF-Locoformer の tuning YAML も同じ文字列である。

`SISNRLoss.__init__` は `name is None` のとき `_name = "si_snr_loss"` とする。`FixedOrderSolver` は `stats[self.criterion.name]` に損失を書く。公式 Trainer は `reporter.has(phase, key)` のときだけ `{phase}.{key}.best.pth` を作る。キー `si_snr` はこの構築では出てこない。

現行 `_retain_chunk_models` は `(-row["si_snr"], row["epoch"])` で最大化する。TF-GridNet の `validation_report` は `score_si_sdr` の正の値を負にして `si_snr` に入れるので、最大化は SI-SDR が低い epoch を選ぶ。TF-Locoformer は `si_sdr_loss` の符号を戻した正の dB を最大化する。loss の 5 本は loss の順位のままである。`configs/training/tf_gridnet.yaml` のコメントは、この最大化がピン留め Trainer と同じだと書いている。ピン留め Trainer はそのキーを見ない。

公式の推論指定が `valid.loss.ave_5best` である、という B のファイル観察は、このレビューではディレクトリを開き直していない。loss 側の選択規則が公式の `valid.loss` min と同じだ、という範囲はコードと一致する。

### MossFormerGAN の BN と crop

crop の乱数源の差は上の表のとおり残る。2 GPU の発話順は別で、D と G は `rank_mod_fixed_v1` が公式 `DistributedSampler` と一致したと書き、A も 2 GPU の本数をその sampler 側に置いている。1 プロセスの公式は `DataLoader(shuffle=True)` であり、現行 world 1 の `rank_mod_fixed_v1` とは別、という D-003 も、G の H10 の留保と一致する。

BN は Discriminator に残る。公式 `solver.py` は分散時に両方を SyncBatchNorm にして `find_unused_parameters=True` で DDP する。`broadcast_buffers` は渡さないので既定 True である。現行 `distribute` は SyncBatchNorm を呼ばない。検証は公式も現行も Generator だけ `eval()` で、Discriminator は `train()` のままである。現行の検証 forward は rank 0 だけなので、running stats を動かす検証バッチの母集団が公式の全 rank とは違う。D の gloo は、broadcast 後に rank 0 だけ追加 forward すると buffer が再び分かれる、という縮小実験である。SyncBatchNorm との出力差は、D が CPU DDP に拒否されて NOT_TESTED のままである。

`find_unused_parameters` の差は D-009 が SUSPECTED のままである。未使用パラメータの有無は数えていない。

### 21 batch

機構は公式の打ち切りと現行の `min` prefix で揃っている。B は、これを「valid 1835 発話の全サンプル平均」と呼ばない、という記録上の Issue にしている。D と G は、公式と同じ破棄を現行固有の欠陥にしていない。BLOCKER にしている報告は無い。DNS 36,000 件は長さが揃い破棄 0、という B と F の記述も、この 21 とは別の集合である。整数 21、2908、2887 は再計算していない。

### `reproduction.duckdb` の不在

ファイルは無い。学習の再現経路は、その不在では止まらない。止まらないことは、DB 行とスナップショットが同一であることの証明ではない。C が言うとおり、親 protocol と `deviation_class` がスナップショットに無く、SEMamba 以外の protocol 式は再計算できない。Dataset ID の再計算一致は C の結果であり、ここでは繰り返していない。

`reproduction-p8.duckdb` も無い。`reproduction-p9.duckdb` はある。DeepFilterNet3 のスナップショットは、C の一覧とこのレビューの snapshot 名の両方で無い。

### スモーク上限が `COMPLETED` になる経路

無修正の p10b 4 ファイルは `verification_max_updates: 2` を持つ。この上限は `global_step >= limit` かつ `global_step < max_steps` のとき `INTERRUPTED` にし、`complete.json` を書かない。B-005 の「そのままでは本格にならない」は、この経路では成り立つ。

`COMPLETED` になるのは、そのキーを外したあとに別の短縮が残る場合である。

- `chunk_iterator.max_updates` は、その本数で `epoch_done` になる。`max_epochs` まで繰り返すと `_official_epoch_complete` が立ち、`global_step` が公式予算より小さくても `COMPLETED` になる。p10b の TF は `max_updates: 2` と `validation_max_updates: 1` を同時に持つ。
- `validation_max_updates` だけを残すと、学習の chunk 数は全件のままで、検証は各 rank の prefix になる。`max_epochs` まで終わると `COMPLETED` になり、`global_step` は公式予算に見える。JSONL には `validation_scope` が残る。`complete.json` には無い。
- 再現データの `max_updates_per_epoch` は、`logical_steps_per_epoch` と同時には使えない。`steps_per_epoch` が無いので、cap の終わりが epoch の終わりになり、scheduler が進む。`max_steps` に達すると `stop_reason` が空のまま抜け、`COMPLETED` になる。
- `validation_max_batches` は学習更新を切らない。best はその prefix の平均で更新され、`max_steps` で `COMPLETED` になる。

CLI の `execute` は `strict_resume` を渡さない。`_check_resume` は `chunk_iterator` 全体を比較するので、chunk の `max_updates` と `validation_max_updates` を消した設定への再開は拒否する。`data.max_updates_per_epoch`、`data.validation_max_batches`、`verification_max_updates` はこの比較に入らない。非 chunk のスモーク checkpoint から、その cap を消した設定へ CLI 再開する経路は残る。job chain は `logging` 以外の config を自分で比較するので、`verification_max_updates` の差をそこで拒否する。`strict_resume` 側の無視リストには `verification_max_updates` が入っているが、job chain の先行比較の方が先に見る。

本格の `mp_senet_vb28_official.yaml` と MossFormerGAN の official YAML には、F が列挙したスモークキーは無い。このファイル単体を、p10b の cap と同じ BLOCKER にはしない。

## 降格すべき主張

1. G-001 の「hop 非倍数の発話は公式平均では -1」は、CONFIRMED の因果から外す。根拠は iSTFT 長の件数と、`pesq` を呼んでいないことによる推論である。A の pesq 0.0.4 は、50 サンプルの長さ差だけでは例外にしていない。
2. その降格のあと、PESQ 失敗の平均差（A-003）を BLOCKER に上げない。無音で -1 と `na` に分かれることは A の実測とコードの両方にある。valid で何件起きるかは未計測である。807/824 は切り詰めの件数であり、PESQ 例外の件数ではない。
3. A-004 の BLOCKER は、現行再開が学習率を壊す欠陥としては過大である。現行は scheduler の `state_dict` を戻す。余分な 0.99 倍は、公式が `last_epoch` で scheduler を作り直す側にある。G の H9 棄却は、現行が公式より早く減衰する、という仮説の棄却であり、公式側の作り直しを否定していない。公式の再開バグまで写す実験でなければ、この差は BLOCKER ではない。A の対応方針自身が、正しく続けるなら現行のままだと書いている。
4. A-008 のうち、Generator の `BatchNorm1d` が 2 GPU の数値を変える、という読みは外す。そのクラスは `Conformer_ConvModule` にあり、SE 16 kHz の Generator 構築からは呼ばれない。公式の `SyncBatchNorm` は `_BatchNorm` 以外を変換しない。Discriminator の `BatchNorm2d` は残る。
5. C-001 を、学習開始の BLOCKER から外す。開始経路はスナップショットを読む。残すのは、登録 DB との同一性と、SEMamba 以外の protocol 式が未検証であることである。
6. C-002 を、P10-C 全体の開始 BLOCKER から外す。DFN3 の DB とスナップショットが無いことは再確認した。F は DeepFilterNet3 と SEMamba++ を P10-C の開始対象にしていない。公式オンライン混合をこのディレクトリから始める主張は、データが無いので成立しない。
7. G-003 を、無条件の高 BLOCKER として扱わない。公式が残す選択は `valid.loss` で、現行の loss n-best はその規則である。`valid.si_snr.best` を公式成果物として使うときだけ、GridNet は SI-SDR の悪い側、Locoformer は公式が書かない SI-SDR 最大側になる。B の NON-BLOCKER は、この条件と両立する。
8. G-004 の「半減、巻き戻し、停止の epoch が分かれる」を、測定済みの結果として読まない。分かれているのは検証に使う 2 秒の乱数源である。epoch 番号のずれは、その帰結としてまだ反証していない。
9. B-005 を「無修正 p10b が `COMPLETED` になる」と読まない。無修正は `INTERRUPTED` である。F-001 を「今の p10b を起動すると黙って公式完了になる」と読まない。黙って完了に見えるのは、`verification_max_updates` を外し、他の cap を残したときである。
10. `_close_chunk_run` の else が、現行ループの通常終了で `COMPLETED` を書く、という読みはしない。F はその引数に while から届かないと書いている。生きている偽完了は、cap 付きの epoch 完了と、非 chunk の `max_steps` 完了である。

## 維持すべき BLOCKER

条件付きのものも、条件ごと残す。

- A-002。MP-SENet の本格検証は、hop 非倍数を切ってから生成する。公式は全長をモデルに入れる。valid の `case.frames` では 824 件中 807 件がこの分岐に入る。A が測った、同じフレーム数でも magnitude が最大絶対誤差 1.563 になる、という 1650 対 1600 の forward は再実行していない。分岐が valid の例外的な数件だけ、という読みは snapshot の件数と合わない。
- A-006。公式 `train.py` は `network_wrapper` を 2 回作り、1 回目の Generator と 2 回目の Discriminator を残す。現行は同じ seed の範囲で 1 回ずつ作る。checksum の差 `601.67` は A の測定で、ここでは再測していない。構造の差は fresh start の Discriminator 初期値に残るので、再開専用の A-004 とは別に維持する。
- A-007 と D-004。公式の学習は PESQ が `None` の batch を backward せず次へ進む。現行 `train_update` は `metric_targets` が `None` なら `TrainingError` を送る。無音で `None` になることは A が測っている。`train_update` 全体の実行は A がしていない。検証側が 0 を分母に残すことは、公式の検証平均と同じであり、この BLOCKER の中身ではない。
- A-008 と D-002 の Discriminator 側。2 GPU の本格比較では、SyncBatchNorm と、検証を全 rank で回すことが公式側にある。現行は局所 batch stats と rank 0 だけの検証更新である。1 プロセスの `mossformergan_vb28_official.yaml` ではこの差は出ない。SyncBN の出力差そのものは未測定のままである。
- B-005 と F-001。TF-GridNet と TF-Locoformer の公式規模 YAML は、スモーク cap の無いファイルとしてはリポジトリに無い。無修正 p10b は本格完了にならない。cap を一部残して本格設定として起動すると、`complete.json` が公式完了に見える経路がある。official の MP-SENet / MossFormerGAN YAML にはその cap が無い。
- F-003 のうち、TF-GridNet の world 4 と、MossFormerGAN 2 GPU YAML の 1 プロセス。どちらも既存の chunk 検査または再現データ経路が、公式の rank 数と違う計画を拒否しない。TF-Locoformer は micro 上限 1 のため、chunk 検査が world 4 以外の accumulation を落とす。YAML に world size は無い。
- F-002。p10b の `checkpoint_interval: 2` を本格の更新数まで使うと、F が測った step ファイルの byte から step 数が桁で増える。本格ファイルを p10b のコピーにして interval を戻さない場合に限る。保存ループ自体は F が走らせていない。

## まだ反証していない仮説

反証できなかったので、捨ててもいない。CONFIRMED BUG にもしていない。

- VoiceBank の実波形で、`score_pesq` の 0.1 境界が何件 `na` になるか。A の正弦とランダム 8 本では発火していない。G は `pesq` を呼んでいない。
- 公式 iSTFT が入力より短いとき、pesq 0.0.4 が長さ差以外の理由で例外になるか。A の 16050 対 16000 は例外ではなかった。valid 807 件での例外率は未計測である。
- crop の開始位置が公式 `random.randint` と数値でどれだけ違うか、その差で MossFormerGAN の半減 epoch が実際に変わるか。
- SyncBatchNorm と現行の局所 BatchNorm の出力差。D の CPU DDP は SyncBN を拒否した。2 GPU の実勾配 norm が 10 を超え、D-001 の clip 順の差が実際の更新を動かすかも未計測である。コード上、非最終 micro は `no_sync` のあと局所勾配を clip し、最終 micro の backward のあとに再度 clip する。
- D-009。`find_unused_parameters` が False のまま未使用パラメータがあると、2 GPU の backward が待ち得る。被覆は数えていない。
- D-005。rank 局所の例外や SIGTERM で、他 rank が collective 上に残る。意図的には再現していない。
- A-012。検証が公式の dropout 乱数を進め、現行は RNG を戻す。乱数列の比較はしていない。
- E-001 と E-003。検証成果物と epoch ファイルの削除が `step_*.pt` の `os.replace` より先に公開される、という順序はソースにある。再開後に平均リンクが欠ける、という穴は E のミニ実行であり、ここでは再実行していない。
- E-002 の壁時計。DNS 検証の 5.61 時間は接頭辞速度の外挿で、F も全 126000 chunk は測っていない。
- C の「登録 DB とスナップショットが同一」と、SEMamba 以外の protocol 式。DB が無いので、このレビューでも反証できない。
- B-003。現行 36000 件が公開スクリプトの `cv_synthetic` の件数と uttid 順に合う、という C と B の観察は一致する。保存 config の `cv_synthetic_small` と同一かは、その実ファイルが無く未解決のままである。
- B-004。MS-SNSD のクリップ境界への移動が 0 件かどうか。snapshot に移動件数は無い。
- 公式 `Stft` と現行の矩形 STFT の長波形での勾配差。G も B も数値比較していない。
- F-004 の再開を、cap 付き checkpoint から実際に走らせて `COMPLETED` まで到達すること。比較キーの欠落はコードにある。実行は F がしていない。
- F-007。検証が学習より長い、という計画は接頭辞 2048 chunk の外挿である。全件速度が接頭辞の半分でも学習を上回る、という F の算術は、全件測定にはなっていない。

## 読んだ実行経路

公式 `src/se/official/mp-senet/train.py` のカウンタと PESQ 呼び出し、`models/model.py` の `pesq_score`、`dataset.py` の全長 STFT。現行 `src/training/se/mp_senet.py` の `validation_report` と `align_validation_length`、`src/evaluation/metrics/pesq.py`、`src/training/core/trainer.py` の `_run_data_epochs`、`_consider_best`、`_retain_chunk_models`、`_close_chunk_run`、`_at_verification_limit`、`_check_resume`、`accumulation_scope`。TF は `tf_gridnet.py` の `_negative_si_sdr`、`tf_locoformer.py` の `_positive_si_sdr`、公式 `SISNRLoss` と `FixedOrderSolver` と Trainer の `reporter.has`、MS-SNSD YAML の criterion。MossFormer は公式 `solver.py` の SyncBatchNorm と clip、現行 `mossformergan.py` の `distribute` と `train_update`、`conv_module.py` の未使用 `Conformer_ConvModule`。データは `open_training_data`、`planner._crop`、protocol の検証 crop、`job_chain` の config 比較、`batch.plan_batches`。
