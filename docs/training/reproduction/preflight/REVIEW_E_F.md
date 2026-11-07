# P10-C Preflight Phase 2: Agent E × Agent F

調査日: 2026-11-07。対象は終了状態と本格学習設定。読んだのは `AGENT_E_CHECKPOINT_RESUME.md`、`AGENT_F_CONFIG_OPERATIONS.md`、`src/training/core/trainer.py`、`src/training/se/run.py`、`src/training/se/cli.py`、`src/training/se/job_chain.py`、`src/training/se/config.py`、`src/training/core/batch.py`、`configs/training/` の YAML。YAML と `src/` は変更していない。学習は起動していない。GPU は使っていない。学習プロセスは止めていない。

## 総評

`verification_max_updates` 以外の短縮が公式の終了まで残ると `COMPLETED` になる、という規則は両者とも `Trainer._close_chunk_run` と、非 chunk の `stop_reason is None` 分岐を指している。chunk が `max_epochs` の前に `max_steps` へ達する終了は、同じ `_close_chunk_run` の `INTERRUPTED` である。同じ `max_steps` の再開拒否は `fit` の `global_step >= max_steps` であり、無条件の「再開できない」はどちらの文書の結論でもない。

Validation 途中の消失で当該 epoch をやり直す、という結論は interval 0 では同型である。E が実測したのは単一プロセスの `_run_chunk_epochs` / `_validate_chunks`、F が指すのは 4 rank が通る `_run_aligned_chunk_epochs` で未実行なので、コード箇所は DISPUTED。p10b の `checkpoint_interval: 2` と、本格 5 YAML にスモークキーが無いことは、実ファイルと F が一致する。E はその 2 点を p10b に対して述べていない。TF-GridNet と TF-Locoformer の本格再現 YAML は `configs/training` に無い。

## 主張の照合

| 主張 | 判定 | 両者が指すコード |
|---|---|---|
| `verification_max_updates` 以外の短縮が残ると、公式終了で `COMPLETED` | 一致 | `_at_verification_limit`（618–621 行）。非 chunk は `_run_epochs` 393–395 行、`_run_data_epochs` 511–513 行。chunk は `_close_chunk_run` 646–657 行。cap の適用は F が `_run_data_epochs` 412 行、`_run_aligned_chunk_epochs` 732 行、`_validate_training_data` 1271 行、`_validate_aligned_chunks` 894 行。単一プロセス側の同じ cap は 533–535 行と `_validate_chunks` 957 行 |
| その else（公式 epoch でも `max_steps` でも halt でもない `COMPLETED`）が現行ループから起きるか | DISPUTED（到達性だけ） | 同じ else、655–656 行。while の出口は epoch 上限、step 上限、halt だけ |
| chunk の `max_steps` 到達は `INTERRUPTED` | 一致 | `_close_chunk_run` 651–654 行。`max_epochs` 完了や early stop が先なら 649–650 行で `COMPLETED` |
| その状態のまま再開できない | 無条件なら不採用。条件付きなら一致 | 同じ上限は `fit` 206–207 行。strict で上限を変えると `run.py` 204–208 行。CLI は `strict_resume` を渡さない（`cli.py` 26 行、`run.py` 74 行の既定は偽）。`_check_resume` は `max_steps` を比較しない（1355–1405 行） |
| Validation 途中の消失は当該 epoch のやり直し | 結論は interval 0 で一致。参照関数は DISPUTED | E: `_run_chunk_epochs` 549–588 行、`_validate_chunks` 946–995 行（実測）。F: `_run_aligned_chunk_epochs` 766–796 行（未実行）。保存が検証と scheduler の後、という順は両方にある |
| p10b の `checkpoint_interval: 2` | F と実ファイルが一致。E は p10b をこの値で述べていない | 4 ファイルの YAML。保存条件は `_should_save` 1139–1144 行に、chunk の epoch 末保存（574 行、781 行）が加わる |
| 本格 YAML にスモークキーが無い | F と実ファイルが一致。E は本格 5 件を棚卸ししていない | `configs/training/reproduction/` の official 3 件と safe 2 件。キーを受け取るのは `config.py` |

## 終了状態

`verification_max_updates` だけが、`global_step` が上限以上かつ `max_steps` 未満のとき `stop_reason = INTERRUPTED` になる。`limit >= max_steps` はロード時に拒否される。E の状態表と F の「このキーだけは完了にしない」は、この関数を指している。

それ以外の短縮は `stop_reason` を立てない。非 chunk はループを抜けるとき `stop_reason` が空なら `COMPLETED` と `complete.json` を書く。`data.max_updates_per_epoch` は epoch の更新数を短くし、`data.validation_max_batches` は rank 0 の検証を接頭辞にする。chunk の `max_updates` は `usable` を短くし、最後の epoch で `_official_epoch_complete` を真にする。`validation_max_updates` は検証 batch の接頭辞だけを変える。いずれも `max_epochs` の完了、early stop、または非 chunk の `max_steps` 到達で `COMPLETED` になる。F の「経路がある」はこの流れである。E はキー名を列挙していないが、`INTERRUPTED` の原因を verification 上限と、chunk の epoch 未完了 `max_steps` に限っているので、同じ規則である。

共有の例外も同じ分岐である。chunk で `halt` も `_official_epoch_complete` も `epoch >= max_epochs` も無く、`global_step >= max_steps` のときは `INTERRUPTED` と `interrupted.json` で、`complete.json` は無い。epoch 終端の Validation が終わっていても、公式 epoch 完了より前ならこちらに入る。TF-Locoformer の `max_steps: 2100000` が 150 epoch の最終更新と同時に達する場合は、`_official_epoch_complete` が先に真になり `COMPLETED` になる。`INTERRUPTED` になるのは、その完了より前に上限へ達したときである。

`_close_chunk_run` の最後の else は、上のどれでもない呼び出しを `COMPLETED` にする。E はこれを生きた終了状態として表に載せている。F は、現行の while（epoch 上限、step 上限、halt）からはこの引数にならないと書いている。while の出口はその 3 つだけなので、到達しないという F がコードと一致する。分岐の効果は同じ行であり、到達性だけが DISPUTED である。運用上の完了判定は、この else を使わない。

`_managed`（`deadline` または `segment_steps`）のとき、chunk の epoch 未完了 `max_steps` は `INTERRUPTED` を書かず `TrainingError` を送出する。`fit` の `except Exception` が `stop_reason` 未設定なら `FAILED` を書く。両者ともこの分岐を終了表に載せていない。`job_chain` は `deadline` を渡すので `_managed` が真になる。E が引用する `job_chain.py` 164–165 行は、戻ってきた `stop_reason` が `CHECKPOINTED` と `COMPLETED` 以外なら失敗にする、という戻り後の検査である。

## 再開

同じ `max_steps` で `global_step` が既にその値以上なら、`fit` が `checkpoint global_step is already at max_steps` で拒否する。strict の比較から外れるのは `logging` と `verification_max_updates` だけなので、`max_steps` を変えた strict 再開は `checkpoint configuration does not match this experiment` になる。E の P10P-E-005 と F の strict の説明は、この 2 箇所を指している。

CLI の `--resume` は `strict_resume` を渡さない。`_check_resume` は `max_steps`、`verification_max_updates`、`data` の cap、scheduler、`checkpoint_interval` を比較しない。chunk 節全体は比較する。したがって、保存 step より大きい `max_steps` へ strict 無しで進む再開は、保存済み更新をやり直さずに続く。E はこの継続をミニ chunk で確認している。F は同じ穴を、verification 上限を外した checkpoint から CLI で進める、と書いている。無条件の「再開できない」にはしない。

chunk 節の `max_updates` や `validation_max_updates` を消すと、CLI でも `_check_resume` が chunk 節不一致で拒否する。静かに外せるのは `verification_max_updates` と、`_check_resume` が見ない `data` の cap である。F の文はこの区別を含んでいる。

`job_chain` の設定比較が外すのは `logging` だけなので、`verification_max_updates` の差は拒否する。再現データの `database` は `config.py` の reproduction 経路で `None` になり、`initialize` は `data.database` のパスを hash する。この 2 点は F の記述とコードが一致し、E の「`INTERRUPTED` は chain の成功状態ではない」と矛盾しない。

## Validation 途中のやり直し

chunk の 1 更新は、epoch が終わったときだけ検証し、そのあと scheduler を進め、そのあと `step_*.pt` を書く。`checkpoint_interval: 0` の定期保存は epoch 末だけなので、検証の batch ループ中にプロセスが消えると、当該 epoch の終端 pickle は無い。再開は直前に公開された `step_*.pt` からで、当該 epoch の Optimizer 更新をやり直し、検証は先頭からである。完了済み epoch の pickle はやり直さない。

この順序は 2 つの関数にある。

- 単一プロセスは `_run_chunk_epochs` が `_validate_chunks` を呼ぶ。E のプローブはこの経路で、更新 4・検証 batch 3 の停止から、更新 2 と検証 batch 2 のやり直しを確認している。
- `world_size != 1` または `virtual_world_size` は、`_run_chunk_epochs` の入口で `_run_aligned_chunk_epochs` に入る。検証は `_validate_aligned_chunks`。TF-Locoformer の 4 rank はこちらである。F はこちらをコード順だけで指し、プロセス停止は起こしていない。

結論の骨格は interval 0 で一致する。4 rank の本線を E の実測関数と読むことはできないので、参照箇所は DISPUTED とする。aligned 側の停止実験は、このレビューでもしていない。

`SIGTERM` はフラグだけで、確認は検証と scheduler のあと、保存の前に読み、保存してから `CHECKPOINTED` になる。E は単一プロセスで、検証の最初の batch にフラグを立てても検証が完走し、当該 epoch をやり直さないことを確認している。aligned 側もフラグを読む位置は検証呼び出しのあとである。F は「プロセスが死ぬ」と書き、`SIGTERM` と `SIGKILL` を分けていない。`SIGTERM` までやり直しに含める読みは、E および両方のループと合わない。

`checkpoint_interval: 2` の p10b では、epoch 末より前の 2 の倍数 step も `_should_save` が書く。検証中の消失で失うのは、検証のあとで書く終端 step であり、epoch 内に既にある step からはそこへ戻る。E はやり直しの起点を interval 0 に限っている。F の「前 epoch 末から」は、epoch 内 checkpoint が無い本格（interval 0、または epoch 長以上）に当てはまる。p10b の interval 2 を残したファイルには、その文をそのまま当てはめない。

## p10b の interval と本格 YAML

`configs/training/reproduction/p10b/` の 4 ファイルは、すべて `checkpoint_interval: 2` である。

| ファイル | checkpoint_interval | validation_interval | スモークキー |
|---|---:|---:|---|
| `p10b/mp_senet.yaml` | 2 | 2 | `verification_max_updates: 2`、`max_updates_per_epoch: 2`、`validation_max_batches: 2` |
| `p10b/mossformergan.yaml` | 2 | 2 | 同上 |
| `p10b/tf_gridnet.yaml` | 2 | 0 | `verification_max_updates: 2`、`chunk_iterator.max_updates: 2`、`validation_max_updates: 1` |
| `p10b/tf_locoformer.yaml` | 2 | 0 | 同上 |

F の表と一致する。E は p10b の interval を述べていない。E の「TF-Locoformer の出荷 YAML はすべて `checkpoint_interval: 0`」の引用は `configs/training/tf_locoformer*.yaml` で、この glob は p10b を含まない。トップレベルの `tf_locoformer.yaml`、`tf_locoformer_dns_ondemand.yaml`、`tf_locoformer_dns_accept_recipe.yaml`、`tf_locoformer_voicebank.yaml` は interval 0 である。p10b をその文に含めると実ファイルと割れる。文書の引用範囲では、E が 2 を否定しているわけではない。

本格 5 件、`mp_senet_vb28_official.yaml`、`mossformergan_vb28_official.yaml`、`mossformergan_vb28_official_2gpu.yaml`、`mp_senet_vb28_safe.yaml`、`mossformergan_vb28_safe.yaml` には、次のキーは無い。

- `verification_max_updates`
- `data.max_updates_per_epoch`
- `data.validation_max_batches`
- `chunk_iterator.max_updates`
- `chunk_iterator.validation_max_updates`

`configs/training` 全体でこれらのキーがあるのは p10b の 4 ファイルだけである。研究 YAML と `trial100/` にも無い。F の「本格 5 件にスモークキーは無い」は維持する。

F の「DNS / MS-SNSD の短い YAML 4 件は `num_samples_per_epoch: 1`」は、`tf_gridnet_ms_snsd_ondemand.yaml` が 4 なので、その 1 件だけ外れる。このファイルは `max_steps: 1` のスモークであり、本格 YAML ではない。

## TF の本格 YAML

`configs/training/reproduction/` 直下に、TF-GridNet または TF-Locoformer の official / safe に相当するファイルは無い。あるのは p10b のスモーク 2 件だけである。F の「本格再現 YAML は無い」は維持する。

公式 Dataset 名の所在は次のとおり。

- `tf_gridnet.ms_snsd.espnet_recipe@1` は `reproduction/p10b/tf_gridnet.yaml` だけ。`max_updates: 2`、`validation_max_updates: 1`、`verification_max_updates: 2`、`checkpoint_interval: 2` が付いている。
- `tf_locoformer.dns.accept_recipe@1` は p10b に加え、`configs/training/tf_locoformer_dns_accept_recipe.yaml` も指す。後者は `max_steps: 2`、`num_samples_per_epoch: 1`、`max_epochs: 1` の CPU スモークである。F の「公式 Dataset を指す reproduction ファイルは p10b だけ」は、この 1 件を数え落としている。本格ファイルが無い、という結論は変わらない。

研究用としてリポジトリにあるのは本格再現ではない。

- `tf_gridnet.yaml` と `tf_gridnet_chunk_seed17.yaml` は VoiceBank の legacy DB、`micro_batch_size_per_gpu: 4`、`checkpoint_interval: 0`。実行計画の公式は micro 上限 2 の 2 rank と `espnet_recipe@1` である。
- `tf_locoformer_voicebank.yaml` は VoiceBank、chunk 長 32,000、`num_samples_per_epoch: 10700`、micro 4。ファイル自身が DNS 再現ではなく、このファイルだけでは 150 epoch を始めないと書いている。
- `tf_gridnet_fixed5000_seed17.yaml` は `max_steps: 175000` の別スケジュールである。

## BLOCKER

| ID | 判定 | 理由 |
|---|---|---|
| P10P-F-001 | 維持。TF の本格 YAML を作るとき、および p10b を編集して使うときは BLOCKER。現行の MP-SENet / MossFormerGAN official と safe は、ファイルに当該キーが無いので F の NON-BLOCKER のまま | 短縮キーが `INTERRUPTED` にならない経路は `_close_chunk_run` と非 chunk の完了代入で一致。本格 5 件にキーは無い |
| P10P-F-002 | BLOCKER を維持 | p10b 4 件の `checkpoint_interval: 2` は実ファイルにある。`_should_save` は正の interval でその倍数ごとに保存する |
| P10P-F-003 | BLOCKER を維持 | `plan_batches` は micro を上限として蓄積を決める。TF-Locoformer（global 4、micro 上限 1）で蓄積 1 になるのは world 4。TF-GridNet（micro 上限 2）は world 2 と world 4 の両方が蓄積 1。MossFormerGAN の 2 GPU YAML を world 1 で読むと蓄積 4 で、chunk 以外はこの蓄積を拒否しない。E はこの起動幅を扱っていない |
| P10P-F-004 | 再開を本格手順に含めるときの BLOCKER を維持 | strict の無視リストと、CLI が strict を渡さないことは E と F が同じ `run.py` 204–208 行と CLI に帰着する |
| P10P-F-006 | 完了を `complete.json` だけで自動判定するときの BLOCKER を維持 | `_mark_complete` が書くのは `status`、`global_step`、`checkpoint`。early stop も `max_epochs` 完了も `COMPLETED`。E の状態表と同じ |
| P10P-F-007 | BLOCKER を維持 | 案 A の検証時間をこのレビューでは再集計していない。E は 5.61 時間を未測定と書き、F の式を否定していない。降格する反証は無い |
| P10P-F-008 | NON-BLOCKER のまま | 現象（interval 0 で検証中に消えると当該 epoch をやり直す）は維持する。E の重大度は高、F は新しい途中 checkpoint を足さないので NON-BLOCKER。参照関数の DISPUTED は、このラベルを BLOCKER に上げる根拠にはしない。長い検証を始める判断は F-007 に残る |
| P10P-F-005、P10P-F-009 | この照合の範囲では降格しない | provenance と sidecar 容量は、終了状態の 5 主張と衝突していない |

E の P10P-E-001 から E-011 は BLOCKER ラベルを付けていない。このレビューで新たに BLOCKER へは上げない。E-002 のやり直しは F-008 と同じ現象で、上の NON-BLOCKER を維持する。E-005 の再開口の狭さは、同じ `max_steps` と strict に限ればコードと一致し、無条件の再開不能にはしない。
