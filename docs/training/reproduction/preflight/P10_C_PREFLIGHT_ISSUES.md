# P10-C Preflight Issue

相互レビュー後の統合一覧である。生の Issue は 57 件（A 15、B 7、C 2、D 9、E 11、F 9、G 4）。同じ現象は 1 行にまとめた。確度 `CONFIRMED` は、コード上の直接証拠または CPU 再現があるものに限る。レビューで割れた識別子は `DISPUTED` とし、確定 BUG の件数に入れない。

深刻度はレビュー後の値である。

## 件数

| 深刻度 | 統合後の主題 | うち P10-C の開始を止めるもの |
|---|---|---|
| CRITICAL | 0 | 0 |
| HIGH | 14 | 8（条件付きを含む） |
| MEDIUM | 16 | 0 |
| LOW | 6 | 0 |

## HIGH（開始を止める）

| ID | 対象 | 種別 | 確度 | 結論 | P10-C |
|---|---|---|---|---|---|
| P10P-A-002 | MP-SENet | OFFICIAL_DIFF | CONFIRMED | 検証は生成前に hop 倍数へ切る。公式は全長をモデルへ入れる。valid の `case.frames` は 824 件中 807 件が 100 の倍数ではない。長さ 1650 を 1600 に切った CPU forward の最大絶対誤差は 1.563 | BLOCKER |
| P10P-A-006 | MossFormerGAN | BUG | CONFIRMED（checksum は DISPUTED） | 現行は Discriminator を 1 回構築する。公式 `train.py` は `network_wrapper` を複数回作り、学習に残す初期値が違う。A の checksum `7729.488281` は、A-D が残存重みではないと読んだ。2 回目の Discriminator と 4 回目の初期化は、同じ構築の呼び方として DISPUTED | BLOCKER |
| P10P-A-007 / P10P-D-004 | MossFormerGAN | BUG | CONFIRMED | 学習中の PESQ 失敗で現行は `TrainingError`。公式はその batch を更新せず次へ進む。検証平均が 0 を残すことは、この停止とは別 | BLOCKER |
| P10P-A-008 / P10P-D-002 | MossFormerGAN 2 GPU | OFFICIAL_DIFF | CONFIRMED | 公式は SyncBatchNorm。現行の Discriminator は局所 BatchNorm で、検証の running stats を更新するのは rank 0 だけ。その buffer が checkpoint に残る。Generator に BatchNorm は無い。1 GPU では公式も現行も SyncBatchNorm にしない | BLOCKER（2 GPU のみ） |
| P10P-B-005 / P10P-F-001 | TF-GridNet / TF-Locoformer | BUG | CONFIRMED | 公式規模で上限の無い本格 YAML は無い。無修正の p10b は `verification_max_updates: 2` で `INTERRUPTED` になり、`complete.json` は出ない。`verification_max_updates` を外して他の cap を残すと、`COMPLETED` と `complete.json` になる。MP-SENet と MossFormerGAN の official YAML にその cap は無い | BLOCKER（TF の本格 YAML を作るとき、p10b を編集して使うとき） |
| P10P-F-002 | p10b | RISK | CONFIRMED | 4 ファイルとも `checkpoint_interval: 2`。TF-Locoformer の本格更新数までこの間隔で保存すると、計測した 1 ファイル約 180 MB から step ファイルが約 172 TiB 規模になる | BLOCKER（p10b を本格へコピーするとき） |
| P10P-F-003 | TF-GridNet / MossFormerGAN | RISK | CONFIRMED | YAML に world size が無い。TF-GridNet は world 2 と world 4 の両方が蓄積 1 で chunk 検査を通る。MossFormerGAN の 2 GPU YAML を 1 プロセスで読むと蓄積 4 のまま再現経路を通る。TF-Locoformer は world 4 以外の蓄積を落とす | BLOCKER（GridNet と MossFormer 2 GPU YAML） |
| P10P-F-007 | TF-Locoformer | PERFORMANCE | CONFIRMED | 保存済み 11 step から学習は 384.4 時間。validation は先頭 2,048 chunk の 0.160 秒/chunk から、126,000 chunk × 150 epoch で 841 時間。全件は未計測。検証方式はこの監査で変えていない | BLOCKER（案 A の 150 epoch を始める判断） |

## HIGH（開始は止めない）

| ID | 対象 | 種別 | 確度 | 結論 | P10-C |
|---|---|---|---|---|---|
| P10P-A-003 / P10P-G-001 の平均差 | MP-SENet | OFFICIAL_DIFF | CONFIRMED | 公式は PESQ 失敗を -1 のまま平均する。現行は失敗と境界不一致を平均から外す。hop 非倍数だから公式が必ず -1 になる、という因果は降格。pesq 0.0.4 は 50 サンプル差だけでは例外にしていない。807/824 は切り詰め件数であり、例外件数ではない | NON-BLOCKER |
| P10P-A-011 / P10P-G-004 | MossFormerGAN | OFFICIAL_DIFF | CONFIRMED | 公式の crop は `random.randint`。現行は `mix_seed`。検証の 2 秒の乱数源が違う。半減 epoch が実際にずれた測定は無い | NON-BLOCKER |
| P10P-B-003 | TF-Locoformer | UNKNOWN | CONFIRMED（未知であること） | 保存 config は `cv_synthetic_small`。件数と選択規則は公開ソースに無い。現行 36,000 は sorted uttid の末尾 1/10 であり、その名前ではない | NON-BLOCKER（集合は変えない） |
| P10P-B-006 | TF-Locoformer | OFFICIAL_DIFF | CONFIRMED | 現行は標準偏差 `1e-6` 未満で停止する。公式は割る | NON-BLOCKER |
| P10P-D-001 | MossFormerGAN 2 GPU | OFFICIAL_DIFF | CONFIRMED | 蓄積 2 では、平均する前の局所勾配に clip する。公式は micro-batch ごとに平均してから clip する。実ネットワークの勾配 norm が 10 を超えるかは未計測。toy の最大差は 0.018 | NON-BLOCKER |
| P10P-E-001 / P10P-E-003 | 共通 | RISK | CONFIRMED | epoch 終端では validation 成果物と best リンクの更新が、`step_*.pt` の `os.replace` より先に残ることがある。公開済み pickle の中では Optimizer、Scheduler、best 状態は揃う | NON-BLOCKER |
| P10P-F-004 | 共通 | RISK | CONFIRMED | CLI は `strict_resume` を渡さず、cap キーを再開比較から外す。cap 付き checkpoint から cap 無しへ進み、その後 `COMPLETED` になり得る。再開の実行はしていない | BLOCKER（再開を本格手順に含めるとき）。新規開始だけなら NON-BLOCKER |
| P10P-F-006 | 共通 | RISK | CONFIRMED | `complete.json` は status、step、パスだけを書く。early stop も `COMPLETED`。一次記録は JSONL ヘッダ | BLOCKER（完了の自動判定を作るとき） |

## MEDIUM

| ID | 対象 | 種別 | 確度 | 結論 | P10-C |
|---|---|---|---|---|---|
| P10P-A-001 / P10P-G-002 | MP-SENet | BUG | CONFIRMED | 公式の 5,000 回目の保存は 5001 番目の更新後。現行の `step_00005000` は 5000 番目。ExponentialLR の epoch 境界は一致 | NON-BLOCKER |
| P10P-A-004 | MP-SENet | OFFICIAL_DIFF | CONFIRMED | 公式の再開は ExponentialLR を `last_epoch` で作り直し、学習率がもう一度 0.99 倍され、epoch 先頭からやり直す。現行は `state_dict` を戻す。現行再開が学習率を壊す欠陥としては降格 | NON-BLOCKER |
| P10P-A-010 | MP-SENet / MossFormerGAN | OFFICIAL_DIFF | CONFIRMED | cuDNN benchmark / deterministic が公式の 2 行と違う | NON-BLOCKER |
| P10P-A-012 | MP-SENet | OFFICIAL_DIFF | CONFIRMED | 現行は検証後に乱数を戻す。公式の検証 dropout は学習の乱数を進めたままにする。乱数列の比較は未実施 | NON-BLOCKER |
| P10P-A-013 | MP-SENet 研究 YAML | OFFICIAL_DIFF | CONFIRMED | 研究用 YAML は検証間隔と前処理が公式と違う | NON-BLOCKER |
| P10P-A-015 | MossFormerGAN | OFFICIAL_DIFF | CONFIRMED | 公式の学習率半減は checkpoint の epoch 番号を巻き戻す。現行の epoch は単調 | NON-BLOCKER |
| P10P-B-001 | TF-GridNet | RISK | CONFIRMED | validation の raw batch は 2,887 と 2,908。公式 `iterator_stop` と同じく両方 2,887 まで実行し、21 batch を捨てる。破棄 42 chunk のうち 6 発話は損失に入らない。全件検証ではない。現行固有ではない | NON-BLOCKER |
| P10P-B-002 / P10P-G-003 | TF-GridNet / TF-Locoformer | OFFICIAL_DIFF | CONFIRMED | 保存 config の `valid/si_snr/max` は stats に無い。実キーは `si_snr_loss`（負の SI-SDR）。公式は `valid.si_snr.best` を書かない。現行 GridNet の `valid.si_snr.best.pt` は SI-SDR が低い側を選ぶ。`valid.loss` の n-best は公式と同じ規則 | NON-BLOCKER（そのファイルを公式成果物にしないとき） |
| P10P-B-004 | TF-GridNet | OFFICIAL_DIFF | CONFIRMED | 公式シェルは `count*8/10`。現行はクリップを割らないよう端を動かす。登録集合で何件動いたかは snapshot に無い | NON-BLOCKER |
| P10P-B-007 | TF-GridNet | UNKNOWN | CONFIRMED | README の結果 hash `239f516` の tree に MS-SNSD recipe は無い | NON-BLOCKER |
| P10P-C-001 | 共通 | DATA_GAP | CONFIRMED | `reproduction.duckdb` と `data/staging/reproduction` はこの作業木に無い。学習は snapshot だけを開く。開始の BLOCKER からは降格。DB と snapshot の同一性は未証明 | NON-BLOCKER |
| P10P-D-003 | MossFormerGAN | OFFICIAL_DIFF | CONFIRMED | world 1 は公式の `DataLoader(shuffle=True)` ではなく、replicas=1 の Sampler 順 | NON-BLOCKER |
| P10P-D-006 | DeepFilterNet3 | OFFICIAL_DIFF | CONFIRMED | BN 統計を更新するのは rank 0 だけ。P10-C の 4 モデルの開始条件ではない | NON-BLOCKER |
| P10P-E-002 / P10P-F-008 | TF chunk | RISK | CONFIRMED | `checkpoint_interval: 0` で validation 中にプロセスが消えると、当該 epoch の学習からやり直す。`SIGTERM` は検証を完走してから `CHECKPOINTED`。参照関数は DISPUTED。新しい途中 checkpoint は作らない | NON-BLOCKER |
| P10P-E-004 / P10P-E-005 / P10P-E-006 / P10P-E-010 | 共通 | RISK | CONFIRMED | replace 後も `generator.pt` と status は別更新。同じ `max_steps` の再開は `fit` が拒否する。上限を上げた strict 無しの再開は継続できる。`metrics.jsonl` の途中行が巻き戻しを止める。legacy chunk は validation Dataset の identity を再開時に見ていない | NON-BLOCKER |
| P10P-F-005 / P10P-F-009 | 共通 | RISK | CONFIRMED | git dirty は provenance に無い。sidecar は容量見積の 25.1 GiB に入っていない | NON-BLOCKER |

## LOW

| ID | 対象 | 種別 | 確度 | 結論 | P10-C |
|---|---|---|---|---|---|
| P10P-A-005 | MP-SENet | OFFICIAL_DIFF | CONFIRMED | MaxPool の同値タイだけ勾配が分かれる | NON-BLOCKER |
| P10P-A-009 | MossFormerGAN | RISK | CONFIRMED | Plateau は公式どおり、悪化した損失が履歴最小を上書きする。同値は非改善。5 回で G の学習率だけ半分、10 回で停止 | NON-BLOCKER（公式の実効動作） |
| P10P-A-014 | 共通 | DATA_GAP | CONFIRMED | 定数波形の `audio_norm` は公式が NaN、現行が例外 | NON-BLOCKER |
| P10P-D-007 / P10P-D-008 | 共通 | RISK | CONFIRMED | 学習ログは rank 0 の loss。`unavailable` が他 rank のスコアを消す経路は、現行 TF の検証では未発火 | NON-BLOCKER |
| P10P-D-009 | MossFormerGAN | RISK | SUSPECTED | `find_unused_parameters=False`。試した小入力では勾配なしパラメータは 0。実学習形状は未計測 | NON-BLOCKER |
| P10P-E-007 / P10P-E-008 / P10P-E-009 | 共通 | RISK | CONFIRMED | 保存中の SIGKILL は前の checkpoint を残す。epoch 途中の checkpoint が未実行の validation を含まないのは未実行の記録。非 chunk の ExponentialLR は検証が無くても epoch 終端で進む | NON-BLOCKER |

## 4 モデルの外

| ID | 対象 | 種別 | 確度 | 結論 | P10-C |
|---|---|---|---|---|---|
| P10P-C-002 | DeepFilterNet3 | DATA_GAP | CONFIRMED | `reproduction-p8.duckdb` とオンライン snapshot が無い。開始できない。SEMamba++ の ready な Dataset は代替素材の claim のまま | 4 モデルの開始条件ではない。DFN3 オンラインだけ BLOCKER |
| P10P-D-005 | DDP | RISK | SUSPECTED | 片 rank の例外や SIGTERM で、他 rank が collective 上に残る。実測は無い | NON-BLOCKER |

## 降格の記録

次は、独立報告では開始を止めると読めたが、レビュー後は止める材料から外した。

- hop 非倍数なら公式 PESQ が必ず -1 になる、という因果
- A-004 を、現行再開が学習率を壊す欠陥として扱うこと
- Generator の `BatchNorm1d` が 2 GPU の数値を変える、という読み。そのクラスは SE 16 kHz の構築から呼ばれない
- C-001 を学習開始の BLOCKER にすること
- C-002 を 4 モデル全体の BLOCKER にすること
- G-003 を、`valid.loss` の n-best まで公式と違うと読むこと
- G-004 の半減 epoch が既に測定済みだと読むこと
- 無修正 p10b が `complete.json` を書くと読むこと
- `_close_chunk_run` の else が、現行ループの通常終了で `COMPLETED` を書くと読むこと
