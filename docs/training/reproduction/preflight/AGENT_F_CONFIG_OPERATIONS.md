# P10-C Preflight Agent F: 実験設定・計算性能・成果物

調査日: 2026-11-07。作業ツリーは `3942d27ad612aca934b695971d7897954e2fd155`、`git status --porcelain` は 0 行。`configs/training/` の YAML 27 件は `load_training_config` で読んだ（学習は起動していない）。完了判定は `Trainer._close_chunk_run` を `/tmp` で実行した。時間は `data/results/training/p10b2/20261107-p10b2/` の `metrics.jsonl` と `tf_locoformer/valid_prefix2048.json` から再計算した。GPU ジョブは出していない。

## 結論

本格再現として置いてある YAML に、次のスモーク制限キーは残っていない。

- `verification_max_updates`
- `data.max_updates_per_epoch`
- `data.validation_max_batches`
- `chunk_iterator.max_updates`
- `chunk_iterator.validation_max_updates`

対象は `configs/training/reproduction/mp_senet_vb28_official.yaml`、`mossformergan_vb28_official.yaml`、`mossformergan_vb28_official_2gpu.yaml`、および同ディレクトリの `*_safe.yaml` 2 件。`max_steps` はどの YAML にも必須で、これら 5 件の値は公式または safe 分割の全予算である。

TF-GridNet と TF-Locoformer の本格再現 YAML は無い。再現パイプラインで公式 Dataset を指すファイルは `configs/training/reproduction/p10b/` のスモークだけである。そこには上のキーが残っている。そのまま起動すると `verification_max_updates: 2` で `INTERRUPTED` になり、`complete.json` は出ない。`verification_max_updates` だけを外し、epoch 短縮か検証短縮を残すと、例外なく `COMPLETED` と `complete.json` になる経路がある。P10-C 用 YAML は今回作っていない。

## YAML 一覧

役割は、公式規模の再現、公式 Dataset の別分割、VoiceBank 等の研究長時間、短い試行、に分けた。27 件すべて `load_training_config` は成功した。`checkpoint_dir` の文字列は 27 件で重複しない。

| YAML | 役割 | モデル | max_steps | epoch 予算 | global / micro | scheduler | ckpt / val 間隔 | best 開始 | 出力 |
|---|---|---|---|---:|---|---|---|---:|---|
| `reproduction/mp_senet_vb28_official.yaml` | 本格再現 | MP-SENet | 1,157,200 | 400 × 2,893 | 4 / 4 | exponential 0.99 | 5,000 / 5,000 | 40 | `reproduction/mp_senet.vb28.ondemand.official` |
| `reproduction/mossformergan_vb28_official.yaml` | 本格再現 | MossFormerGAN | 700,106 | 121 × 5,786 | 2 / 1 | none（タスク内 plateau） | 5,786 / 5,786 | 40 | `...mossformergan.vb28.ondemand.official` |
| `reproduction/mossformergan_vb28_official_2gpu.yaml` | 本格再現 | MossFormerGAN | 350,053 | 121 × 2,893 | 4 / 1 | none（タスク内 plateau） | 2,893 / 2,893 | 40 | `...official-2gpu` |
| `reproduction/mp_senet_vb28_safe.yaml` | 別分割の全予算 | MP-SENet | 1,070,000 | 400 × 2,675 | 4 / 4 | exponential 0.99 | 2,675 / 2,675 | 40 | `...mp_senet.vb28.ondemand.safe` |
| `reproduction/mossformergan_vb28_safe.yaml` | 別分割の全予算 | MossFormerGAN | 647,350 | 121 × 5,350 | 2 / 1 | none | 5,350 / 5,350 | 40 | `...mossformergan.vb28.ondemand.safe` |
| `reproduction/p10b/mp_senet.yaml` | スモーク | MP-SENet | 1,157,200 | 上限は公式値。実 epoch は 2 更新 | 4 / 4 | exponential 0.99 | 2 / 2 | 0 | `p10b/placeholder/mp_senet` |
| `reproduction/p10b/mossformergan.yaml` | スモーク | MossFormerGAN | 350,053 | 実 epoch は 2 更新 | 4 / 1 | none | 2 / 2 | 40 | `p10b/placeholder/mossformergan` |
| `reproduction/p10b/tf_gridnet.yaml` | スモーク | TF-GridNet | 2,000,000 | `max_epochs` 35。実更新は 2 | 4 / 2 | plateau 0.7 / 1 | 2 / 0 | 0 | `p10b/placeholder/tf_gridnet` |
| `reproduction/p10b/tf_locoformer.yaml` | スモーク | TF-Locoformer | 2,100,000 | `max_epochs` 150。実更新は 2 | 4 / 1 | warmup 4,000 + plateau 0.5 / 3 | 2 / 0 | 0 | `p10b/placeholder/tf_locoformer` |
| `mp_senet.yaml` | 研究 | MP-SENet | 1,070,000 | 400 × 2,675 | 4 / 4 | exponential 0.99 | 2,675 / 2,675 | 40 | `voicebank-seed42/mp_senet` |
| `mossformergan.yaml` | 研究 | MossFormerGAN | 642,000 | 120 × 5,350 | 2 / 1 | none | 5,350 / 5,350 | 40 | `voicebank-seed42/mossformergan` |
| `semambapp.yaml` | 研究 | SEMamba++ | 133,700 | 100 × 1,337 | 8 / 4 | exponential 0.99 | 1,337 / 1,337 | 40（保存しない） | `voicebank-seed42/semambapp` |
| `deepfilternet3.yaml` | 研究 | DeepFilterNet3 | 22,711 | schedule 120 epoch | 64 / 64、limit 64 | deepfilter_cosine | 0 / 0 | 0 | `voicebank-seed42/deepfilternet3` |
| `tf_gridnet.yaml` | 研究 | TF-GridNet | 2,000,000 | chunk 35 | 4 / 4 | plateau 0.7 / 1 | 0 / 0 | 0 | `voicebank-seed42/tf_gridnet` |
| `tf_gridnet_chunk_seed17.yaml` | 研究 | TF-GridNet | 2,000,000 | chunk 35 | 4 / 4 | plateau 0.7 / 1 | 0 / 0 | 0 | `tf-gridnet-chunk-seed17` |
| `tf_gridnet_fixed5000_seed17.yaml` | 研究 | TF-GridNet | 175,000 | `logical_steps_per_epoch` 5,000 | 4 / 4 | plateau 0.7 / 1 | 5,000 / 5,000 | 0 | `tf-gridnet-fixed-5000-seed17` |
| `tf_locoformer_voicebank.yaml` | 研究 | TF-Locoformer | 2,000,000 | chunk 150、chunk 長 32,000 | 4 / 4 | warmup 4,000 + plateau | 0 / 0 | 0 | `voicebank-seed42/tf_locoformer` |
| `trial100/*.yaml` 5 件 | 試行 | 各モデル | 100 | なし、または cosine の表だけ 120 | モデル依存 | モデル依存 | 50 / 0 | 40 | `trial100-20261104/<model>` |
| `tf_locoformer.yaml` | スモーク | TF-Locoformer | 2 | chunk なし | 1 / 1 | none | 0 / 0 | 0 | `training/tf-locoformer` |
| `tf_locoformer_dns_ondemand.yaml` | スモーク | TF-Locoformer | 14 | chunk `max_epochs` 2、`num_samples_per_epoch` 1 | 1 / 1 | warmup + plateau | 0 / 0 | 0 | `se-repro/p7-tflocoformer-ondemand` |
| `tf_locoformer_dns_accept_recipe.yaml` | スモーク | TF-Locoformer | 2 | chunk 1 epoch、`num_samples_per_epoch` 1。Dataset は accept_recipe@1 | 1 / 1 | warmup + plateau | 0 / 0 | 0 | `se-repro/p10a1-tflocoformer-accept-recipe-smoke` |
| `tf_gridnet_ms_snsd_smoke.yaml` | スモーク | TF-GridNet | 20 | chunk 1 epoch、`num_samples_per_epoch` 1、`virtual_world_size` 2 | 4 / 2 | plateau | 1 / 0 | 0 | `se-repro/p6-tfgridnet-smoke` |
| `tf_gridnet_ms_snsd_ondemand.yaml` | スモーク | TF-GridNet | 1 | chunk 1 epoch、`num_samples_per_epoch` 1 | 4 / 2 | plateau | 1 / 0 | 0 | `se-repro/p6-tfgridnet-ondemand` |

DeepFilterNet3 の `batch_schedule` は `[[0,16],[2,24],[5,32],[10,64],[20,128],[40,9999]]`、`early_stopping_patience` は 25、cosine の `steps_per_epoch` は 668、`epochs` は 120。P10-C の実行計画は DeepFilterNet3 と SEMamba++ を開始対象にしていない。

TensorBoard は、本格再現 5 件と `trial100/` と MossFormer / MP-SENet の一部スモーク以外では、研究 YAML は `logging.tensorboard.enabled: true`（`scalar_interval` 省略時は 10）。本格再現 5 件は `logging` 自体が無く、TensorBoard は出ない。`metrics.jsonl` は rank 0 が常に書く。

## スモーク制限の有無

キーはローダが既知として受け取る。未知キーは拒否する。`/tmp` で公式 MP-SENet YAML に `not_a_field` を足すと `unknown training config fields`、`verification_max_updates` を `max_steps` と同じにすると `must be less than max_steps`。0 は `_int` の下限 1 で拒否する。正の小さい値は拒否しない。

| YAML | verification_max_updates | max_updates_per_epoch | validation_max_batches | chunk.max_updates | chunk.validation_max_updates |
|---|---|---|---|---|---|
| 本格再現 3 件と safe 2 件 | 無 | 無 | 無 | chunk 節なし | chunk 節なし |
| 研究 8 件（上表） | 無 | 無 | 無 | 無（chunk が無いファイルは節なし） | 無 |
| `trial100/` 5 件 | 無 | 無 | 無 | 節なし | 節なし |
| DNS / MS-SNSD の短い YAML 4 件 | 無 | 無 | 無 | 無。代わりに `num_samples_per_epoch: 1` と小さい `max_steps` | 無 |
| `p10b/mp_senet.yaml` | 2 | 2 | 2 | 節なし | 節なし |
| `p10b/mossformergan.yaml` | 2 | 2 | 2 | 節なし | 節なし |
| `p10b/tf_gridnet.yaml` | 2 | 無 | 無 | 2 | 1 |
| `p10b/tf_locoformer.yaml` | 2 | 無 | 無 | 2 | 1 |

`max_steps` は全ファイルにある。本格 5 件では全予算そのもので、追加の短縮キーは無い。試行と短い YAML では `max_steps` 自体が短い。

### コードが許可する終了

`verification_max_updates` だけは完了にしない。`_at_verification_limit` は `global_step >= limit` かつ `global_step < max_steps` のとき真。その場で `stop_reason = INTERRUPTED` になり、`interrupted.json` を書き、`complete.json` は書かない。`limit >= max_steps` はロード時に拒否する。上限ちょうどの step ではこの停止は偽になる（`/tmp` で確認）。CLI は `stop_reason` を見ないので、プロセスの終了コードは 0 である。再開時、保存済み step が同じ上限以上なら `checkpoint global_step is already at verification_max_updates` で拒否する。

それ以外の制限は、付いたままループが公式の終了条件を満たすと `COMPLETED` になる。

| 残したキー | 経路 | 終了 |
|---|---|---|
| `data.max_updates_per_epoch` | `_run_data_epochs` が epoch の更新数を `min(計画, cap)` にする。epoch 末で scheduler が進む。`global_step` が `max_steps` に達すると `stop_reason` が空のまま抜け、`COMPLETED` | 公式の step 数で `complete.json`。epoch 長と ExponentialLR / plateau の回数は公式と違う。JSONL の run ヘッダに cap が残る。`complete.json` には無い |
| `data.validation_max_batches` | 学習更新は全件。検証は rank 0 の公式順の接頭辞。`validation_scope` を JSONL に書く。best はその平均で更新する | `max_steps` で `COMPLETED`。`complete.json` は接頭辞を書かない |
| `chunk_iterator.max_updates` | その本数で `epoch_done`。scheduler、early stop、nbest が進む。`max_epochs` の最後で `_official_epoch_complete = True` | `global_step` が公式より小さくても `COMPLETED`。`/tmp` で flag 真・step 300・`max_steps` 2,100,000 を渡すと `complete.json` が書かれた |
| `chunk_iterator.validation_max_updates` | 学習の chunk 数は全件。検証は rank ごとの接頭辞。early stop と `valid.loss` の nbest はその損失 | `max_epochs` で `COMPLETED`。`global_step` は公式予算に見える |
| 非 chunk の小さい `max_steps` | `_run_epochs` / `_run_data_epochs` は上限到達で `COMPLETED` | `trial100` の 100 step はこの形。chunk 経路は、`max_epochs` 未達の上限到達を `INTERRUPTED` にする（`/tmp` で `complete.json` は無く `interrupted.json` があった） |
| chunk の `num_samples_per_epoch: 1` | 6 キーには無い。epoch が短くても `max_epochs` まで終わると公式 epoch 完了 | `tf_locoformer_dns_accept_recipe.yaml` は accept_recipe@1 を指す。現行の `max_steps: 2` は 1 発話 14 chunk より小さく、先に `INTERRUPTED`。`max_steps` だけ上げると 1 発話 epoch の `COMPLETED` になる |

`_close_chunk_run` の最後の分岐は、公式 epoch でも `max_steps` でも halt でもない呼び出しを `COMPLETED` にする。`/tmp` でその引数を渡すと `complete.json` が書かれた。現行の while 条件（epoch 上限、step 上限、halt）からはこの引数にならない。

early stop も halt として `COMPLETED` になる。`complete.json` の中身は `status`、`global_step`、`checkpoint` だけである。150 epoch を走破した完了と、patience で止まった完了は、このファイルだけでは区別できない。

p10b を無修正で起動した場合は、step 2 で `verification_max_updates` が先に立ち、`INTERRUPTED` である。黙って公式完了に見えるのは、このキーを外したあとに他の制限が残った場合である。

`strict_resume` が比較から外すのは `logging` と `verification_max_updates` だけである。epoch cap と chunk cap の差は strict では拒否する。CLI の `--resume` は `strict_resume` が偽で、`_check_resume` は `max_steps`、`verification_max_updates`、`max_updates_per_epoch`、`validation_max_batches`、scheduler、`checkpoint_interval` を比較しない。chunk 節全体は比較する。したがって CLI 再開は、検証上限で止めた checkpoint から上限キーを消した設定へ進み、その後 `max_steps` で `COMPLETED` にできる。job chain は `logging` 以外を比較するので `verification_max_updates` の差を拒否するが、`initialize` は `data.database` のハッシュを取る。再現パイプラインは `database: None` なので、この経路は P10-C の再現 YAML を初期化できない。

## 公式設定との差

比較は YAML と、それを解釈する `config.py`、`run.py`、`trainer.py`、`batch.py`、`chunking.py`。上流リポジトリのファイルは今回開き直していない。

本格再現 3 件は、バッチ、optimizer、scheduler 名、epoch 予算、`max_steps`、checkpoint 間隔、best 開始が YAML に書かれた公式値と、ローダが実行する値で一致する。追加の 6 キーは無い。差としてコードが固定しているものは次である。

- MP-SENet の検証と checkpoint は 5,000 更新ごと、ExponentialLR は epoch 末（2,893）だけ。5,000 の倍数と 2,893 の倍数は 1,157,200 の範囲で一致しない。epoch 末の scheduler には、その step の検証結果は渡らない。exponential は検証値を使わない。
- `micro_batch_size_per_gpu` は上限である。MP-SENet は world 1 で micro 4、world 2 で micro 2、world 4 でも micro 1 として通る。YAML に world size は無い。
- MossFormerGAN の `scheduler.name: none` は、plateau をタスクが epoch 末に持つ、という意味と一致する。`validation_interval` が epoch 長と一致するときだけ、毎 epoch の生成器損失が plateau に入る。
- `mossformergan_vb28_official_2gpu.yaml` を 1 プロセスで起動すると、`plan_batches` は global 4、micro 1、accumulation 4 を返し、再現データ経路はそれを拒否しない。コメントが禁じている起動が、設定エラーにならない。2 プロセスでは accumulation 2 で、YAML の意図と一致する。
- safe 2 件はスモークキーは無い。Dataset 名、`allow_test_selection: false`、更新数が official と違う。公式再現の設定ではない。
- SEMamba++ の研究 YAML は `best_checkpoint_start_epoch` を書いても `_consider_best` が `semambapp` で何も保存しない。
- chunk モデルは `validation_interval` を見ない。epoch 末に必ず検証する。`best_checkpoint_start_epoch` も `_retain_chunk_models` は見ない。p10b の `0` と研究 YAML の `0` は、chunk の nbest には効かない。
- TF-GridNet の p10b は、公式値（global 4、micro 上限 2、Adam 0.001、clip 5、chunk 48,000、shift 0.5、`num_samples_per_epoch` 5,000、35 epoch、plateau 0.7 / patience 1、early stop 5、`keep_nbest` 5）にスモーク制限と `checkpoint_interval: 2` を足したもので、本格ファイルではない。`max_steps: 2000000` は 35 epoch の更新数ではない。
- TF-Locoformer の p10b も同様で、公式値（global 4、micro 上限 1、chunk 64,000、rank あたり 1,000 発話、150 epoch、warmup 4,000、factor 0.5、patience 3、early stop 10、`keep_nbest` 5、AdamW 0.001）に `max_updates: 2`、`validation_max_updates: 1`、`verification_max_updates: 2`、`checkpoint_interval: 2` が乗っている。
- chunk の accumulation が 1 でないと `TrainingError`。TF-Locoformer の micro 上限 1 は world 4 だけが accumulation 1 になる（`plan_batches` を実行した。world 1 は accumulation 4、world 2 は 2、world 8 は global が割れず拒否）。TF-GridNet の micro 上限 2 は world 2（micro 2、accumulation 1）と world 4（micro 1、accumulation 1）の両方が chunk 検査を通る。world 4 は公式の 2 rank ではないが、設定は拒否しない。
- 検証の chunk 数は長さから決まる。`chunk_length` 64,000、shift 0.5 では幅 64,000、個数 `(length - 64000) // 32000 + 1`。長さ 480,000 なら 14。`valid_prefix2048.json` は 4 rank とも 9,000 発話、計画 126,000 batch、破棄 0 と記録している。9,000 × 14 = 126,000。学習側は rank あたり 1,000 発話 × 14 = 14,000 更新、150 epoch で 2,100,000。これは `max_steps: 2100000` と一致する。発話長の全件再読はしていない。

p10b2 の計測に使った実行時設定は、今の p10b YAML と違う。JSONL ヘッダでは TF-Locoformer が `verification_max_updates: 12`、`checkpoint_interval: 1000000000`、`max_updates` と `validation_max_updates` は無い。MP-SENet と TF-GridNet の中断ランは上限 16、MossFormerGAN は 12。いずれも epoch cap は無く、`complete.json` は無い。ステップ時間はこの計測のものである。

## 時間と容量

### TF-Locoformer の検証と学習

学習時間の元は `tf_locoformer/steps12/metrics.jsonl` の train 12 行。`AUDIO_DB_STEP_SYNC` 済みのホスト計測として保存されている。step 1 の更新は 1.535 秒、待ちは無い。step 2 から 12 の 11 個は、更新の中央値 0.374 秒、待ちの平均 0.272 秒、更新と待ちの和の平均 0.659021 秒、中央値 0.667 秒、最小 0.556 秒、最大 0.822 秒。2,100,000 更新に平均を掛けると 384.43 時間。中央値なら 389.0 時間。観測した最大の和を全 step に使うと 479.4 時間。11 step の標本であり、全 2,100,000 step の再計測ではない。

検証時間の元は `tf_locoformer/valid_prefix2048.json`。全 126,000 chunk の実行ではない。4 rank が各 2,048 chunk を処理し、計画作成だけが 126,000 batch 分（8.59 秒から 9.64 秒）である。再構築と forward を足した 1 chunk は、rank 0 が 2,048 chunk で 0.160184 秒、512 で 0.15799 秒、128 で 0.16010 秒。他 rank の 2,048 chunk は 0.1564 秒から 0.1579 秒。最も遅い rank 0 の 126,000 倍は 20,183 秒、5.606 時間。150 epoch で 841.0 時間。文書の 0.16018359375 は、rank 0 の再構築と forward の和を 328.056 秒に丸めた値を 2,048 で割ったもので、今回の未丸め値と 1e-7 秒以内で一致する。計画と all-reduce を 1 回あたり約 10.3 秒足しても、150 epoch で 0.43 時間であり、時間の桁は動かない。all-reduce は検証ループの後に 1 回で、テンソルは和と件数なので、chunk 数に比例させない。

学習 384.4 時間と検証 841 時間は、この二つのファイルから再計算できる。検証が長い、という関係は、接頭辞の速度が全件にそのまま使えるときの話である。検証は `shuffle: false` の先頭 2,048 / 126,000（約 1.6%、約 146 発話）である。残りが接頭辞の半分の速度でも、平均は約 0.081 秒/chunk で、学習と並ぶ境の 0.073 秒をまだ上回る。観測した接頭辞内のばらつき（0.156 から 0.169）では逆転しない。全件をこのセッションでは測っていない。

実行計画の足し算は、コード上の待ち方と一致する。chunk epoch は学習のあと、checkpoint 保存の前に検証する。壁時計は学習と検証の和になる。early stop（patience 10）で短くなることはある。その場合も `complete.json` は `COMPLETED` である。余裕 1.3 を学習に掛けると 500 時間、1.5 を検証 841 時間に掛けると 1,262 時間、合計約 1,762 時間。この係数は計測ではなく倍率である。案 A のまま 150 epoch を始めると、検証が学習より長い、という計画の判断は、上の再計算の範囲では変わらない。検証の選び方はこの報告では変えていない。現行 p10b YAML は cap 付きのままで、案 A の全件検証にはならない。cap を外したときに現行コードが毎 epoch 全件を検証する。

検証の途中でプロセスが死ぬと、その epoch の checkpoint はまだ無い。chunk 経路の保存は検証と scheduler の後である。再開は前 epoch 末から学習をやり直し、検証も最初からである。1 epoch の学習は 14,000 × 0.659 秒で約 2.56 時間、検証が 5.61 時間なら、検証中の停止は約 8 時間を捨てる。途中 checkpoint は無い。

### 容量

実ファイルと JSONL の `checkpoint_io.bytes` は次のとおり。TF-Locoformer は `torch.load`（CPU）で要素も数えた。

| モデル | 実測の step ファイル | 重み側 | 本格で残す個数 | step ファイルの合計 |
|---|---:|---:|---:|---:|
| TF-Locoformer | 179,910,164 byte（step 12） | `generator.pt` 59,967,018 byte。float32 の separator は 59,894,024 byte（14,973,506 要素）。Adam 状態は 119,787,160 byte | epoch 末の step が 150。`checkpoint_interval: 0` でも epoch 末は保存する。interval 2 だと 2 step ごと | 150 × 179,910,164 = 26,986,524,600 byte（25.13 GiB） |
| TF-GridNet | 31,071,895 byte（step 16） | `generator.pt` 10,318,606 byte | epoch 末 35 | 1,087,516,325 byte（1.01 GiB） |
| MP-SENet | 29,689,087 byte（step 16） | 同じ step ファイルに含む | 5,000 の倍数 231 個に、1,157,200 は倍数ではないので末尾が 1 個増える | 232 × 29,689,087 = 6.42 GiB。231 個だけなら 6.39 GiB |
| MossFormerGAN 2 GPU | 136,454,907 byte（step 12） | 同じ step ファイルに含む | 2,893 × 121 = 350,053 なので 121 個、末尾の追加は無い | 16,511,043,747 byte（15.38 GiB） |

TF-Locoformer の `data_provenance.json` は 43,273,749 byte。`materials` が 130,950 件ある。`write_provenance_sidecar` は checkpoint ごとに同じバイト列を `<stem>.data_provenance.json` へコピーする。step 150 個で sidecar だけで 6.05 GiB。`*epoch.pt` の prune は `.pt` だけを消し、sidecar は消さない。150 epoch すべてで epoch ファイルを書いたあと 5 個だけ残すと、sidecar が最大もう 6.05 GiB 残る。`keep_nbest` の重み 5 個は約 0.30 GiB。計画の 25.1 GiB は step ファイルだけで、sidecar を含まない。

`checkpoint_interval: 2` を p10b のまま 2,100,000 更新に使うと、step ファイルは 1,050,000 個で約 172 TiB、sidecar が約 41 TiB 乗る。計測ランが interval を 1,000,000,000 にしていたのは、この保存を避けるためと読める。本格ファイルを p10b から作るときに interval を戻さないと、完了前にディスクが埋まる。

`metrics.jsonl` の train 行は TF-Locoformer で平均 492 byte。2,100,000 行で約 0.96 GiB。chunk 検証は発話ごとの行を書かず、epoch あたり 1 行である。

他モデルの学習時間も同じ JSONL から再計算した。検証の全件秒数は、TF-GridNet だけ `validation_full.json` を読み直した。MP-SENet と MossFormerGAN の検証秒数は今回の生ファイル再集計に含めていない。

| モデル | 定常 step の和の平均 | 更新数 | 学習時間 | 検証 |
|---|---:|---:|---:|---|
| MP-SENet | 0.53412 秒（15 step、step 1 を除く） | 1,157,200 | 171.7 時間 | 今回は未再集計 |
| MossFormerGAN | 1.08889 秒（11 step） | 350,053 | 105.9 時間 | 今回は未再集計 |
| TF-GridNet | 0.39098 秒（15 step） | 計画 553,521 | 60.1 時間 | 全件。遅い rank の再構築 300.80 秒 + forward 270.73 秒 + all-reduce 20.55 秒 = 592.1 秒。35 回で 5.76 時間。rank 1 の raw batch は 2,908、採用は 2,887、破棄 21 |
| TF-Locoformer | 0.65902 秒（11 step） | 2,100,000 | 384.4 時間 | 接頭辞からの外挿。上のとおり 841 時間 |

## 成果物の対応

`metrics.jsonl` が一次記録である。rank 0 だけが書く。

- 先頭は `type: run`。`config` 全体、`run_id`、`provenance_sha256`、dataset の `provenance()` が入る。スモークキーもここには残る。
- 毎更新が `type: train`。損失、更新後の学習率、batch、`update_seconds`、`load_wait_seconds`、`case_ids`。
- 検証は `type: validation`。chunk と再現データの cap があるとき `validation_scope` が付く。`na` や `unavailable` も JSON のまま残る。
- checkpoint 保存のあと `type: checkpoint_io`（秒、byte、ファイル名）。
- 再開は、checkpoint の `global_step` より大きい行を切る。ヘッダは残る。

TensorBoard は `logging.tensorboard.enabled` のときだけ開く。本格再現 5 件は無効なので、対応する event ファイルは無い。有効なとき `write_scalars` は `status == ok` の有限値だけを書く。train のスカラーは `scalar_interval` ごと、検証スカラーはその step が間隔外でも書く。`validation_scope`、`case_ids`、checkpoint パス、`run_id`、dataset id、スモークキーは event に入らない。`run/settings` は seed、loss、batch、deterministic、モデル名、optimizer、scheduler のスカラー、legacy の set 名、world size、`git rev-parse HEAD`、Python 版だけである。dirty、`max_steps`、chunk、dataset id は無い。JSONL から `tensorboard.rebuild` するとスカラーは再現できる。性能スカラーと `run/settings` は JSONL に無いので再現できない。

実験 ID は `build_provenance` の `run_id` で、SHA-256 の入力は format、pipeline、model 名、dataset id、snapshot hash、`config_hash`、profile、claim、`selection_uses_test`、seed である。`config_hash` の入力は model 名、seed、loss、world size、ddp モード、dataset identity、任意の `virtual_world_size` だけである。`max_steps`、スモークキー、scheduler、checkpoint 間隔は `run_id` を変えない。同じ Dataset と seed のスモークと本格で `run_id` が一致する。計測ランの `run_id` は `ccda88e187dc549fedb8f959df6df0f901acae3519f8d47b4d97184bd47e28af`。学習は DB を開かない。登録時に同じ `run_id` で列が違うと `repro.runs` は拒否する。

Dataset identity は残る。provenance には `dataset_id`、`dataset_name`、`dataset_version`、`snapshot_hash`、`protocol_id`、`profile`、`claim_label`、`selection_uses_test`、`dataset_identity`（計測ランでは `case_ids_sha256` と `length_sha256` を含む 20 キー）、`sets`、`materials`、`data_sha256` がある。checkpoint の `dataset` と `data_state` にも identity がある。再開は identity と provenance SHA-256 が違うと拒否する。

Git の dirty は残らない。`git_metadata` は commit と dirty の両方を返すが、`build_provenance` が書くのは `git_commit` だけである。checkpoint の `environment` は `capture_environment` で、Git を含まない。計測ランの provenance は `git_commit: 7979d931a9696f7a2b323309c2f68313a7248e49` で、dirty フィールドは無い。今日の作業ツリーは porcelain 0 なので、今の HEAD を書けば dirty ではない状態と一致する。学習中に汚しても provenance は更新しない。

## 自動検査の提案

実装はしていない。開始前に設定ファイルと起動時の world size だけを見る。

1. 本格として起動する YAML に `verification_max_updates`、`data.max_updates_per_epoch`、`data.validation_max_batches`、`chunk_iterator.max_updates`、`chunk_iterator.validation_max_updates` が一つでもあれば拒否する。
2. `num_samples_per_epoch` が公式の rank あたり本数（TF-Locoformer 1,000、TF-GridNet 5,000）と違えば拒否する。`max_epochs` が 150 と 35 以外なら拒否する。
3. `max_steps` が、データ計画の 1 epoch 更新数 × epoch 数と一致しなければ拒否する。TF-Locoformer は 2,100,000。安全上限として大きくするファイルは、完了条件が `max_steps` ではなく `max_epochs` であることを別フラグで明示しない限り拒否する。
4. `checkpoint_interval` が epoch の更新数より小さい本格 YAML を拒否する。保存数 × 実測 byte ×（1 + provenance sidecar）が空き容量を超えたら拒否する。p10b の interval 2 はこの検査で落ちる。
5. 起動プロセス数が、TF-Locoformer は 4、TF-GridNet と MossFormerGAN 2 GPU は 2、MP-SENet は指定した 1 または 2、と一致しなければ拒否する。`plan_batches` の accumulation が chunk モデルで 1 でない場合は既存コードが拒否する。TF-GridNet の world 4 は既存コードが通すので、この検査が必要である。
6. `checkpoint_dir` が空で、`p10b`、`20261107-p10b-lab`、`20261107-p10b1`、`20261107-p10b2` と違うこと。`complete.json` があるディレクトリは既存どおり再開しない。
7. Dataset 名、dataset id、snapshot hash が実行計画の固定値と一致すること。`allow_test_selection` は MP-SENet と MossFormerGAN の official だけ真、TF は偽。
8. 起動時の `git status --porcelain` が空であること。dirty は provenance に残らない。commit を実験メモに書く。
9. 終了後、`complete.json` だけで公式完了としない。JSONL ヘッダにスモークキーが無いこと、`validation_scope` が無いこと、`global_step` が予算と一致すること、chunk なら epoch が `max_epochs` に達しているか early stop を別記録していること。終了コード 0 は `INTERRUPTED` でも付く。
10. CLI 再開を本格に使うなら、`strict_resume` の無視リストから `verification_max_updates` を外し、`data` の cap も比較する、というコード変更が要る。今回は変更していない。検査だけなら、再開元 checkpoint 内の `config` と現行 YAML の差を、`logging` を除く全キーで拒否すればよい。

## Issue

| ID | 発見者 | 対象 | 種別 | 深刻度 | 確度 | 場所 | 根拠 | 再現手順 | 影響 | 対応方針 | P10-C影響 | 検証状態 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P10P-F-001 | Agent F | 共通 Trainer | BUG | HIGH | CONFIRMED | `src/training/se/config.py` の既知キー。`trainer.py` `_run_data_epochs`（412 行付近）、`_run_aligned_chunk_epochs`（732 行付近）、`_close_chunk_run`（646 行付近）、`_validate_training_data`（1271 行付近）、`_validate_aligned_chunks`（894 行付近） | 6 キーのうち `verification_max_updates` だけが `INTERRUPTED`。他は epoch 完了または `max_steps` 到達で `COMPLETED` と `complete.json` を書く。本格 5 YAML にキーは無い。p10b にはある | 公式 YAML は `load_training_config` 済み。`/tmp` で `_close_chunk_run` に公式 flag 真・step 300・`max_steps` 2,100,000 を渡し `complete.json` を確認。上限未達の `max_steps` は `interrupted.json` だけだった | cap を残した本格起動が、例外なく公式完了に見える | 開始前検査 1。本格 YAML にキーを書かない。`complete.json` に cap と epoch 完了理由を足す変更は別作業 | BLOCKER（TF の本格 YAML を作るとき、および p10b を編集して使うとき）。現行の MP-SENet / MossFormerGAN official ファイル単体はキーが無く NON-BLOCKER | `_close_chunk_run` は PASS。学習ループ全体は NOT_TESTED |
| P10P-F-002 | Agent F | p10b YAML | RISK | HIGH | CONFIRMED | `configs/training/reproduction/p10b/*.yaml` の `checkpoint_interval: 2`。TF-Locoformer は加えて `max_updates: 2` と `validation_max_updates: 1`。MP-SENet は `validation_interval: 2` と `best_checkpoint_start_epoch: 0` | 実行計画が外すと書いたのは 3 種類の cap である。interval 2 は残る。2,100,000 更新なら step ファイル約 172 TiB と sidecar 約 41 TiB。計測ランのヘッダは interval 1,000,000,000 で、現行ファイルと違う。TF の本格 YAML はリポジトリに無い | `load_training_config` で 4 ファイルを読んだ。byte は `steps12` の `checkpoint_io` 179,910,164 | 本格の前にディスクが埋まる。MP-SENet は 2 step ごとの全件検証と epoch 0 からの PESQ best になる | 本格ファイルを p10b のコピーにしない。interval は 0 か epoch 長以上。検査 4 | BLOCKER | PASS（設定のロードと byte の実測）。保存ループは NOT_TESTED |
| P10P-F-003 | Agent F | 起動時 world size | RISK | HIGH | CONFIRMED | `src/training/core/batch.py` `plan_batches`。`trainer.py` の chunk accumulation 検査。YAML に world size フィールドは無い | TF-Locoformer は world 4 だけ accumulation 1。TF-GridNet は world 2 と world 4 の両方が accumulation 1 で chunk 検査を通る。MossFormerGAN 2 GPU YAML の 1 プロセスは accumulation 4 のまま再現データ経路を通る | `plan_batches(4, 2, 4)` と `plan_batches(4, 1, 1)` を実行 | 公式と違う rank 分割のまま `COMPLETED` になり得る | 検査 5。設定に要求 world size を足して拒否する変更は別作業 | BLOCKER（TF-GridNet と MossFormerGAN 2 GPU）。TF-Locoformer は現行検査が world 4 以外を落とす | PASS（`plan_batches`）。分散起動は NOT_TESTED |
| P10P-F-004 | Agent F | 再開 | RISK | HIGH | CONFIRMED | `src/training/se/run.py` 204 行。`trainer.py` `_check_resume`。CLI `cli.py` は `strict_resume` を渡さない | strict は `verification_max_updates` を無視する。CLI は data cap も比較しない。cap 付き checkpoint から cap 無し設定へ進み、その後 `COMPLETED` になり得る。job chain は再現 YAML を初期化できない | コード経路。再開そのものは実行していない | 短い epoch で scheduler が進んだあとの再開が、公式予算の完了になる | 検査 10。無視リストから上限キーを外す | BLOCKER（再開を本格の手順に含めるとき） | NOT_TESTED |
| P10P-F-005 | Agent F | provenance | RISK | MEDIUM | CONFIRMED | `src/training/provenance.py` `build_provenance`。`src/evaluation/envinfo.py` `git_metadata` | dirty は計算されるが文書に無い。checkpoint の environment にも無い。計測 provenance の Git キーは `git_commit` だけ | 計測の `data_provenance.json` を読んだ。今日の porcelain は 0 | 汚れた木の本格結果が、commit だけが一致した綺麗な実行に見える | 検査 8。`git_dirty` を provenance に書く変更は別作業 | NON-BLOCKER（起動時に porcelain 0 を確認する場合） | PASS |
| P10P-F-006 | Agent F | 追跡 | RISK | MEDIUM | CONFIRMED | `provenance.py` の `config_hash` と `run_id`。`trainer.py` `_mark_complete`。`tensorboard.py` `write_run_info` | `run_id` はスモークキーと `max_steps` を含まない。`complete.json` は status と step とパスだけ。TensorBoard の text も cap を含まない。JSONL ヘッダには config 全体がある。early stop も `COMPLETED` | 計測 provenance と `interrupted.json` を読んだ。`_close_chunk_run` の halt 真を `/tmp` で実行し `COMPLETED` を確認 | 公式完了の判定を `complete.json` か `run_id` だけにすると、短い検証や早期停止を通す | 検査 9。一次記録は JSONL ヘッダと `validation_scope` の不在 | BLOCKER（完了の自動判定を作るとき） | PASS（文書と `/tmp` の status）。DB 登録は NOT_TESTED |
| P10P-F-007 | Agent F | TF-Locoformer 時間 | PERFORMANCE | HIGH | CONFIRMED | 検証は `trainer.py` の epoch 末。計測は `valid_prefix2048.json` と `steps12/metrics.jsonl` | 11 step の平均 0.659 秒 × 2,100,000 = 384.4 時間。最遅 rank の 0.160184 秒/chunk × 126,000 × 150 = 841.0 時間。全件検証は未計測。接頭辞は計画 126,000 の 2,048。実行計画の「検証が学習より長い」は、この外挿の範囲で維持される。選び方は変えていない | 生 JSON から再計算。GPU の再計測はしていない | 案 A の 150 epoch は検証が本体より長い。cap を外した現行コードは毎 epoch 全件を検証する | 開始前に案を決める、という既存の開始条件を維持する。全件を測り直すなら接頭辞ではない区間を含める | BLOCKER（案 A で 150 epoch を始める判断）。設定ファイルの不存在とは別 | PASS（保存済み計測の算術）。全件検証は NOT_TESTED |
| P10P-F-008 | Agent F | 検証中の停止 | RISK | MEDIUM | CONFIRMED | `trainer.py` `_run_aligned_chunk_epochs`。検証のあとで `_save` | 検証中に死ぬと、その epoch の step checkpoint が無い。再開は前 epoch から学習と検証をやり直す。案 A では約 2.6 時間 + 5.6 時間が単位 | コード順。プロセス停止は起こしていない | 長い検証の失敗が 1 epoch 分を捨てる | 新しい途中 checkpoint は作らない、という制約のまま、運用で epoch 末の成功を確認する | NON-BLOCKER（仕組みの追加は不要）。時間予算には捨て分を見ていない | NOT_TESTED |
| P10P-F-009 | Agent F | ディスク | RISK | MEDIUM | CONFIRMED | `provenance.py` `write_provenance_sidecar`。`trainer.py` `_retain_chunk_models` の `*epoch.pt` unlink | sidecar は provenance 全バイトの複製。TF-Locoformer は 43,273,749 byte。epoch ファイルの削除は sidecar を消さない | ファイルサイズを読んだ。prune は実行していない | 25.1 GiB に sidecar 最大約 12 GiB が乗る。interval 2 の爆発は P10P-F-002 | 容量見積に sidecar を入れる。prune 時に sidecar を消す変更は別作業 | NON-BLOCKER（interval を epoch 末だけにし、空きが数十 GiB ある場合） | PASS（サイズ）。削除漏れの実行は NOT_TESTED |

## 未監査

- 学習プロセス、GPU、Dataset、Snapshot、DB、checkpoint の書き込みはしていない。`_close_chunk_run` と `plan_batches` と `load_training_config` だけを実行した。
- 360,000 件の長さは読み直していない。126,000 batch は保存済み計画 JSON と、長さ 480,000 の chunk 式である。
- 2,048 chunk の速度を再計測していない。全 126,000 chunk の検証時間は外挿のままである。
- MP-SENet と MossFormerGAN の全件検証の秒数は、今回の生ファイルから再集計していない。
- 上流の ESPnet / 論文リポジトリは開き直していない。公式値は現行 YAML とトレーナーの解釈に限る。
- DeepFilterNet3 と SEMamba++ の公式 Dataset が blocked であることは実行計画の記述を確認しただけで、データ側は見ていない。
- `repro.runs` への登録、TensorBoard event の再生成、分散プロセスの起動はしていない。
- 他エージェントの結論と、このファイル以外の `preflight/` は読んでいない。
