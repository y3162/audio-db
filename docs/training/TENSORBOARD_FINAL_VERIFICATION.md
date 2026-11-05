# TensorBoard 最終検証（2026-11-05）

対象は `feature/tensorboard` の `a08c9f841f792defd4c845a2fe79d41ce4d9be90` に、作業ツリーの修正を加えたもの。導入直前の親は `c3adede161ee1a4b89b7b368773b4d45cc4d9275`。親のテストファイルは対象 commit と同一。turing の Xeon Silver 4214、CPU、Python 3.12.13、PyTorch 2.6.0+cu124、TensorBoard 2.20.0、NumPy 2.5.4、OMP/MKL 各 1 thread を使用した。試験データと生成物は新規一時ディレクトリで作成した。

前回の `/tmp/test-training.log` では以下の 3 件が、同じテストが完了させた run 内の `complete.json` により再開を拒否されていた。各ケースを単独および以下の順序で実行したところ、修正前の対象と親 commit の両方で同じ `TrainingError: training is already complete` が再現した。単独でも失敗するため、テスト間の成果物共有ではない。親の `test_mp_senet_train` 全体でも 17 件中この 3 件が失敗した。

1. `tests.training.test_mp_senet_train.StepTests.test_prefetch_and_load_workers_match_the_serial_run`
2. `tests.training.test_mp_senet_train.StepTests.test_two_runs_and_a_resumed_run_match`
3. `tests.training.test_mp_senet_train.ValidationTests.test_short_audio_metrics_are_na_and_best_checkpoint_loads`

各テストは `TemporaryDirectory` を使うが、その中で自ら完了した `fast-stop`、`interrupted`、`ckpt` を再利用していた。テストを管理停止に変更後、同モジュール 17 件は PASS。

次のコマンドは、カレントディレクトリを repository root とする。ローカル環境では site-packages の `tests` がリポジトリの名前空間を隠すため、隔離された shim を一時領域に置いた。親ではその作業ツリーを `cd` 先とし、同じ Python と shim の作成手順を使う。

```bash
D=$(mktemp -d)
mkdir -p "$D/tests"
printf '__path__ = ["%s/tests"]\n' "$PWD" > "$D/tests/__init__.py"
export PYTHONPATH="$D:$PWD" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
.venvs/official/bin/python -m unittest tests.training.test_mp_senet_train -v
.venvs/official/bin/python -m unittest tests.training.test_tensorboard_resume -v
.venvs/official/bin/python -m unittest tests.training.reference.tensorboard_equivalence -v
.venvs/official/bin/python -m unittest tests.training.test_tensorboard_scalars tests.training.test_tensorboard_integration tests.training.test_mock_trainer tests.training.test_job_chain tests.training.test_protocol tests.training.test_logical_epoch -v
```

最終作業ツリーで順に 17 件 PASS（13.261 秒）、3 件 PASS（11.133 秒）、1 件 PASS・7 条件（139.801 秒）、29 件 PASS（216.073 秒）。各実行の skip は 0。親の 17 件は 11.751 秒で上記 3 件が失敗した。所要時間は同時に動いていた CPU 試験の影響を含む。

ON/OFF 比較は CPU・seed 7・同じ合成データ、global/micro batch 1、2 更新、step 2 で validation。初期 state、各更新後と validation 後の task state および Python / NumPy / CPU PyTorch RNG は 7 条件とも厳密一致した。event 側の音声 3 種と画像 1 種も確認した。Mock のプレビュー失敗は、Mock task に MP-SENet の `spec` がないことによる意図的な例で、記録後も task state と RNG は一致した。MP-SENet の公式損失は短い合成波形で PESQ ターゲットを再現しない。SEMamba++ は reference scan。小規模実行は学習品質・公式実装一致・GPU 決定性を意味しない。

中断試験は例外を注入して writer の close を実行する。checkpoint 公開後、JSONL 書き込み後、event flush 後の各境界で、完成 checkpoint から再開して step の欠落・重複がないことを確認した。通常継続と巻き戻しは validation を毎 step 実行し、JSONL と EventAccumulator を照合、連続学習との task state と CPU RNG も比較した。再生成コマンドは別ディレクトリの Scalar を検査し、既存先を拒否することと JSONL・checkpoint のハッシュ不変を確認した。強制終了時の部分 JSON 行や OS・ストレージ障害は検証していない。
