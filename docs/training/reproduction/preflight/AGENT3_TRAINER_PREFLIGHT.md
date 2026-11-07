# Agent 3: 共有トレーナー、保持、再開、preflight

この文書は実行条件の実装記録である。preflight の PASS は公式再現の成功ではない。

## チェックポイント保持

`checkpoint_retention` が YAML に無い実験は、これまでどおり step ファイルを消さない。p10b のスモーク YAML はこのキーを持たない。

キーがある実行だけ、その `checkpoint_dir` の中で、その実行が `save_atomic` の成功後に作ったファイルを削る。対象は `step_*.pt` と `Nepoch.pt` だけである。残すものは次である。

- 最新の `keep_latest` 個（本格 YAML では 3）の `step_*.pt`
- `keep_best: true` のとき、best にした step ファイル。最新 3 より古くても残す
- `keep_nbest` の平均に使う epoch ファイル。TF-Locoformer の 5 件はここ
- 平均ファイル `valid.loss.ave_*best.pt` と、最終 checkpoint（`complete.json` / `status.json` が指すファイル）
- ディレクトリ内シンボリックリンクの実体。リンク自体はここでは消さない

所有ファイルは `retention_owned.json` に記録する。このリストに無いファイル、別ディレクトリのファイル、リンクの実体は消さない。既存の `data/` 以下を一括削除しない。`save_atomic` は一時ファイルへ書いてから `os.replace` する。置換が失敗したときは、直前の checkpoint が残る。削るのは置換が成功したあとだけである。

chunk の `*epoch.pt` を n-best 以外から外す既存の動作は、保持キーが無いテストでもそのままである。

## 検証の中断

検証の途中バッチからは再開しない。Optimizer 更新のあと、検証・scheduler・best の前に、`validation_pending` 付きの再開用 checkpoint を書く。この時点で対象更新の Optimizer は進んでいる。

次の起動で `validation_pending` があるときは、その更新をやり直さない。検証を先頭バッチからやり直し、scheduler と best を一度だけ進めてから、フラグの無い checkpoint で置き換える。scheduler のあと、置換の前にプロセスが死んでも、ディスク上はまだ `validation_pending` なので、次は検証のやり直しと scheduler 一回のままである。

フラグの無い古い checkpoint は、これまでどおり再開する。`global_step` が `max_steps` に達した checkpoint でも、フラグがあるときだけ検証を終えてから止める。検証上限に達しているときは、その検証を終えたあと追加の更新をしない。

`metrics.jsonl` の validation 行だけでは best を動かさない。再開時、その step の validation 行は捨ててから書き直す。確定した行には `validation_committed: true` を付ける。

## 再開で違ってよいキー

CLI の `--resume` は設定比較を有効にする。違ってよいのは `logging` だけである。`checkpoint_dir` も比較し、違うディレクトリは拒否する。

拒否する差には次が含まれる。

- Dataset の identity と snapshot（`dataset.identity()`）
- model、world size、batch 計画、global batch、蓄積、optimizer、scheduler、loss、seed
- `validation_interval`、`best_checkpoint_start_epoch`、`chunk_iterator.keep_nbest`
- `verification_max_updates`、`data.max_updates_per_epoch`、`data.validation_max_batches`、`chunk_iterator.max_updates`、`chunk_iterator.validation_max_updates`

上限付き checkpoint を、上限の無い本格設定へ黙って続けない。比較はモデル構築とスナップショットを開く前に行う。Trainer を直接呼ぶ既存テストは、同一設定の再開を拒否しない。`verification_max_updates` だけを外して Trainer を直接続ける既存テストは、CLI の比較とは別である。

## world size

`required_world_size` があるとき、`execute` はプロセス起動のあと、`build_task` とスナップショットを開く前に、`process.world_size` と比べる。global batch が組めても、プロセス数が違えば拒否する。キーが無い YAML は従来どおり起動できる。

## MossFormer 側の契約

`train_update` が `update_applied: 0.0` を返したときは、`global_step` を増やさず、scheduler も検証も行わない。データカーソルは進んだまま戻さない。`pesq_skip` を記録し、`task.last_skip` があれば入れる。キーが無い戻り値は、これまでどおり更新済みとして扱う。rank 間で値が違うときは `TrainingError` にする。

`validation_requires_all_ranks` が真のときは、全 rank が同じ回数 `validation_report` を呼ぶ。回数が違うときは forward の前に拒否する。集計と best は rank 0 だけが行い、他 rank は forward の結果を捨てる。フラグが無いときは、これまでどおり rank 0 だけが検証する。

## YAML

| ファイル | 内容 |
|---|---|
| `configs/training/reproduction/tf_gridnet_ms_snsd_full.yaml` | MS-SNSD。world 2、global 4、micro 2、蓄積 1。`max_epochs` 35、`max_steps` 553521。`checkpoint_interval` 0、`validation_interval` 0（chunk は epoch 末に検証し、epoch 末に保存する）。`keep_nbest` 5。claim は `official_algorithm_fixed_seed`。出力は `data/results/training/reproduction/tf_gridnet.ms_snsd.espnet_recipe-full` |
| `configs/training/reproduction/tf_locoformer_dns_research_valid_full.yaml` | Dataset は `tf_locoformer.dns.accept_recipe.research_valid_fixed@1`。world 4、global 4、micro 1。150 epoch、`max_steps` 2100000。`keep_nbest` 5。claim は `research_checkpoint_selection`。公式の `cv_synthetic_small` でも、36000 件検証の公式再現でもない。出力は `data/results/training/reproduction/tf_locoformer.dns.accept_recipe.research_valid_fixed-full` |
| `mp_senet_vb28_official.yaml` | `required_world_size: 2`、`experiment_profile: full`、`experiment_claim: official_faithful`。数値の学習ハイパラは変えていない |
| `mp_senet_vb28_safe.yaml` | world 2、`experiment_claim: leakage_safe`、profile full |
| `mossformergan_vb28_official.yaml` | world 1、`official_faithful`、profile full |
| `mossformergan_vb28_official_2gpu.yaml` | world 2、`official_faithful`、profile full |
| `mossformergan_vb28_safe.yaml` | ファイルが 1 プロセスと書いているので world 1、`leakage_safe`、profile full |

p10b のスモーク YAML は変えていない。本格 YAML には `verification_max_updates`、`max_updates_per_epoch`、`validation_max_batches`、`chunk_iterator.max_updates`、`validation_max_updates` を置いていない。

`max_steps` 2000000 は公式の更新数ではない。実行計画の 35 epoch 合計 553521 を GridNet に使った。Locoformer は 14000 更新 × 150 epoch の 2100000 である。

## preflight が拒否すること

入口は `python -m src.training.se.cli --config X --preflight` である。モデルは作らず、Optimizer も進めない。`--preflight` が無い CLI は、これまでどおり学習する。`--resume` がある学習は、上の設定比較を行う。

結果は `preflight.json` と `preflight.md` に書く。PASS の文は「実行条件の確認であり、公式再現の成功ではない」。

拒否（status FAIL、終了コード非 0）は次である。

- full profile に上のスモーク上限が残っている
- full profile に `required_world_size` が無い
- `--world-size` が要求 world size と違う
- global batch を `plan_batches` で組めない、または `expected_global_batch` と違う
- `--expect-dataset` または `--expect-snapshot` と違う
- `selection_uses_test` が真なのに `allow_test_selection` が偽
- `official_faithful` または `official_algorithm_fixed_seed` と `research_valid_fixed` の混在、およびその逆（`research_checkpoint_selection` なのにその Dataset ではない）
- `--resume` の設定が、logging 以外で一致しない
- 出力ディレクトリが存在して空でなく、`--resume` が無い。中身は消さない
- 公式 claim なのに素材が blocked、または source access が UNKNOWN
- 保持が best または `keep_nbest` の入力を守れない（`keep_best: false`、`protect_nbest: false`）
- 見積もりより空きディスクが少ない。見積もりは「最新 3 + best + n-best + 平均 + 最終」× 1 ファイルのバイト数。既定は GridNet 31,071,895、Locoformer 179,910,164、MP-SENet 29,689,087、MossFormer 136,454,907。`checkpoint_retention.bytes_per_checkpoint` で上書きできる。バイト数が不明な full profile は UNKNOWN とし、PASS にしない

スナップショットは `dataset.json`、`manifest.json`、case set の件数だけを読む。波形は描かない。開けないときは traceback ではなく `data gap` で FAIL する。

この環境で GridNet の full YAML は PASS だった。読んだ snapshot hash は `a69a477af70bc43dbcabdafbd5c97ec8a26aac4debd51158079545de1e4f0757`、dataset id は `13c6058881ef86fe7506f705b1bd5a5bc68b313181410d8fdaaee20c33ef3fc6`、件数は train 7345 / valid 1835 / test 305、world 2 の蓄積は 1、ディスク見積もりは 341,790,845 バイトで空きは足りた。これは実行条件の確認であり、公式再現の成功ではない。

Locoformer の `research_valid_fixed@1` はスナップショットが 0 件で、`data gap` の FAIL だった。Dataset は登録していない。

## テスト

CPU、`.venvs/se/bin/python -m unittest`。

- `tests/training/test_checkpoint_retention.py`
- `tests/training/test_preflight_full.py`
- `tests/training/test_resume_identity.py`
- `tests/training/test_validation_interrupt.py`

既存の `test_checkpoint_atomic.py`、`test_ave_nbest.py`、`test_run_status.py`、`test_logical_epoch.py` もこの変更のあと通した。実学習と GPU 1, 2, 6, 7 は使っていない。
