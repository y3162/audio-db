# P10-C Preflight Agent G（Red Team）

Phase 1 の独立調査。過去の GO 判定、`docs/training/reproduction/P10_*.md`、P3–P9 検証、`.agents/DECISIONS.md` の結論は前提にしていない。公式 checkout の HEAD は次のとおり。

| 公式ツリー | `git rev-parse HEAD` | 指定 |
|---|---|---|
| `src/se/official/mp-senet` | `89932cfe90d1dacb8e170e4a331d762462c21792` | 一致 |
| `src/se/official/clearervoice-studio` | `6b3774dc79c46ae8bed2a4fa5f706f0ac8c75c61` | 一致 |
| `src/se/official/tf-locoformer` | `7a615460d347ff7334a13dbb831d16280da72cdc` | 一致 |
| `src/se/official/espnet` | `0d136c5456d403cb2c0a4c13e76e8136e32e9b7a` | 一致 |
| ESPnet `90eed8e53498e7af682bc6ff39d9067ae440d6a4` | `git cat-file -t` は `commit` | オブジェクトあり |

本番コード、YAML、テスト、DB、Dataset、Snapshot、checkpoint は変更していない。GPU は使っていない。PID 614674 / 614675 / 4109908 には触れていない。

CONFIRMED はコード上の対応、または CPU 上の小再現があるものだけ。反証できなかった仮説は Issue に数えない。SUSPECTED は 0 件。

## 反証仮説一覧

| ID | 仮説 | 判定 |
|---|---|---|
| H1 | MP-SENet の best 選択 PESQ が、公式 `pesq_score` の平均と別物である | 採用 → P10P-G-001 |
| H2 | MP-SENet の checkpoint / validation が、公式より 1 optimizer 更新早い | 採用 → P10P-G-002 |
| H3 | 公式 `best_model_criterion` の `si_snr` は reporter に無く、こちらの `valid.si_snr.best` は公式が選ばない重みである | 採用 → P10P-G-003 |
| H4 | 発話数は公式どおりでも、Crop が公式 `random.randint` ではなく、MossFormerGAN の検証母集団と MP-SENet の学習窓が変わる | 採用 → P10P-G-004 |
| H5 | On-demand の正規化と Trainer の正規化が二重になる | 棄却 |
| H6 | Chunk iterator が DDP で一部 Chunk を落とす、または `num_samples_per_epoch` の単位が rank あたりではない | 棄却 |
| H7 | ExponentialLR / WarmupReduceLROnPlateau の更新時点が公式と違う | 棄却 |
| H8 | 同じ seed の Data Plan が worker 数や再開で変わる | 棄却 |
| H9 | 再開でこちらの ExponentialLR が公式より早く減衰する | 棄却 |
| H10 | MossFormerGAN の 2 GPU 順が公式 `DistributedSampler` と違う | 棄却 |
| H11 | `mr_l1_tfd` の reduction が公式と違い、勾配スケールが batch 倍になる | 棄却 |
| H12 | DDP が local mean を world size 倍し忘れる | 棄却 |
| H13 | TF-Locoformer の空 `architecture` が DNS YAML の既定と違う | 棄却 |
| H14 | DeepFilterNet3 / SEMamba++ の blocked 公式 Dataset が本格完了になる | 棄却 |
| H15 | 早期停止の epoch 差が公式 `patience` より 1 つずれる | 棄却 |

## Issue

### P10P-G-001

- 状態: CONFIRMED
- 確度: コード対応と CPU 再現
- 重大度: 高
- 対象: MP-SENet 本格設定 `configs/training/reproduction/mp_senet_vb28_official.yaml`
- 仮説: Validation は終了するが、best checkpoint に使う PESQ が公式 `pesq_score` ではない。
- 観測:
  - 公式 `models/model.py` の `eval_pesq` は `pesq` の例外を `-1` にし、`pesq_score` は成功も `-1` も含めて `np.mean` する。`best_pesq` の初期値は `0` で、`epoch >= 40` かつこの平均がそれを超えたときだけ `g_best` を書く。
  - 公式 validation は全文を `center=True` の STFT / iSTFT に通し、iSTFT の長さ引数は無い。CPU で hop 100、n_fft 400 を確認すると、長さ 1000 から 1299 のうち 297 本で iSTFT 長が入力長より短い。短い推定と元の clean を `pesq` に渡すと例外になり、公式は `-1` として平均に入れる。
  - 現行 `validation_report` は先に `align_validation_length` で hop 倍数まで切り、その短い波形で生成してから `score_pesq` を呼ぶ。`score_pesq` は境界長の再スコアと元スコアの差が 0.1 を超えると `na` にする。`_combine_scores` は `ok` だけを平均し、`-1` は入れない。
- 証拠: `src/se/official/mp-senet/models/model.py` の `pesq_score` / `eval_pesq`、`src/se/official/mp-senet/train.py` の validation ループ、`src/training/se/mp_senet.py` の `align_validation_length` と `validation_report`、`src/evaluation/metrics/pesq.py` の `score_pesq`、`src/training/core/trainer.py` の `_consider_best`。
- 再現: `/tmp` ではなくこの調査の CPU プロセス。ゼロ波形 300 本で iSTFT 長の不一致は 297。長さが hop の倍数でない発話は、公式平均では `-1`、現行 best では切り詰めた PESQ か除外である。
- 実験への影響: epoch 40 以降の `best_generator.pt` が、公式 `g_best` と別の更新を選ぶ。公式平均が初期値 0 を超えなければ公式は `g_best` を書かない。現行は正の集約 PESQ で書く。824 件のうち何件が hop 非倍数かは、この作業ツリーに 16 kHz WAV が無く未計測。
- テストが PASS し得る理由: テストが `score_pesq` 同士、または `_consider_best` が集約 PESQ を見ることを確認しても、公式 `np.mean`（失敗は `-1`）との一致は見ていない。
- 関連ファイル: `src/se/official/mp-senet/train.py`、`src/se/official/mp-senet/models/model.py`、`src/training/se/mp_senet.py`、`src/evaluation/metrics/pesq.py`、`src/training/core/trainer.py`

### P10P-G-002

- 状態: CONFIRMED
- 確度: コード対応と CPU のカウンタ再現
- 重大度: 中
- 対象: MP-SENet 本格設定。interval は 5000。1 epoch は 11572 / 4 = 2893 更新。
- 仮説: Checkpoint は保存できるが、公式が書く重みは 1 更新後の重みである。
- 観測:
  - 公式は optimizer step のあと、`steps += 1` の前に `steps % interval == 0 and steps != 0` を見る。カウンタが 5000 の反復は、その更新を含めて 5001 回目の更新後に検証し、`g_00005000` を書く。
  - 現行は更新後に `global_step += 1` し、`global_step % 5000 == 0` で検証・保存する。5000 回目の更新で `step_00005000` を書く。
  - ExponentialLR は両方とも epoch 末だけである。CPU 再現では scheduler は 2893、5786、8679、11572。validation / checkpoint は公式が「カウンタ 5000、完了更新 5001、epoch 1」、現行が「global_step 5000、epoch 1」。
- 証拠: `src/se/official/mp-senet/train.py` 200–275 行、`src/training/core/trainer.py` の `_run_data_epochs` と `_should_save`。
- 再現: 上記カウンタを Python で 4 epoch 分実行。scheduler 境界は一致し、保存時点だけ 1 更新ずれる。
- 実験への影響: 5000 更新ごとの重みと、その時点の PESQ が公式の 5001 更新時点と違う。epoch 番号と学習率の切替回数は一致する。1 更新差であり、epoch 境界のずれではない。P10P-G-001 と重なると、best の候補そのものが別の反復になる。
- テストが PASS し得る理由: テストが `global_step % checkpoint_interval == 0` を現行ループに対して確認すると、公式のインクリメント前判定には届かない。
- 関連ファイル: `src/se/official/mp-senet/train.py`、`src/training/core/trainer.py`、`configs/training/reproduction/mp_senet_vb28_official.yaml`

### P10P-G-003

- 状態: CONFIRMED
- 確度: 公式ソースの対応
- 重大度: 高（`valid.si_snr.best` を結果として使う場合）。`valid.loss.ave_*` の平均集合はこの Issue では崩していない。
- 対象: TF-GridNet（ESPnet `0d136c5` の `egs2/ms_snsd/enh1/conf/tuning/train_enh_tfgridnet.yaml`）と TF-Locoformer（`src/se/official/tf-locoformer/egs2/dns_ins20/enh1/conf/tuning/train_enh_tflocoformer.yaml`、学習参照は ESPnet `90eed8e` の同じ損失名）。
- 仮説: Validation loss の n-best は公式の `valid.loss` に沿うが、もう一つの best ファイルは公式が作らないキーで、GridNet では SI-SDR の悪い側を選ぶ。
- 観測:
  - 両 YAML の `best_model_criterion` は `valid / si_snr / max` と `valid / loss / min`。
  - `SISNRLoss` は `name` 引数が無いとき統計名を `si_snr_loss` にする。YAML の `conf` は `eps` だけで、`enh.py` は `SISNRLoss(**criterion_conf)` を呼ぶ。reporter に登録されるキーは `si_snr_loss`。
  - `Trainer` は `reporter.has(phase, key)` が偽の criterion を best、n-best 保持、平均のすべてから外す。`valid.si_snr` は作られない。`valid.loss` の min と `valid.loss.ave_5best` は残る。早期停止の既定は `valid / loss / min` で、こちらは loss で止める。
  - 現行 `_retain_chunk_models` は loss の小さい方に加え、`si_snr` を最大化して `valid.si_snr.best.pt` を書く。TF-GridNet の `validation_report` は公式損失と同じく SI-SDR の符号を反転して `si_snr` に入れる。最大化は SI-SDR が低い epoch を選ぶ。TF-Locoformer は正の SI-SDR を入れるので、最大化は SI-SDR が高い epoch を選ぶ。どちらも公式の `valid.si_snr.best.pth` ではない。
  - loss の n-best は `kept` に残る。`valid.loss.ave_{keep}best` は loss の列だけを平均する。この平均ファイルが公式 `valid.loss` 平均と違う、という証拠は無い。
- 証拠: ESPnet `0d136c5` の `espnet2/enh/loss/criterions/time_domain.py`（`SISNRLoss`）、`espnet2/tasks/enh.py` の criterion 構築、`espnet2/train/trainer.py` の `reporter.has`、`espnet2/main_funcs/average_nbest_models.py`。現行は `src/training/se/tf_gridnet.py`、`src/training/se/tf_locoformer.py`、`src/training/core/trainer.py` の `_retain_chunk_models`。
- 再現: 公式ツリーを実行せず、ピン留めソースの名前解決で確認した。`si_snr` という統計キーは criterion 構築経路に出てこない。
- 実験への影響: 公式が残すモデルは loss の best と loss の 5-best 平均である。現行が追加する `valid.si_snr.best.pt` を本格結果にすると、GridNet は SI-SDR の悪い epoch、Locoformer は公式が出力しない SI-SDR 最大 epoch になる。`configs/training/tf_gridnet.yaml` のコメントは、この最大化がピン留め Trainer と同じだと書いている。ピン留め Trainer はそのキーを見ない。
- テストが PASS し得る理由: テストが「`si_snr` を最大化する」「loss の 5 本を平均する」を現行実装に対して確認すると、公式 reporter に `si_snr` が無いことは落ちない。
- 関連ファイル: `src/training/core/trainer.py`、`src/training/se/tf_gridnet.py`、`src/training/se/tf_locoformer.py`、`configs/training/tf_gridnet.yaml`、`configs/training/reproduction/p10b/tf_gridnet.yaml`、`configs/training/reproduction/p10b/tf_locoformer.yaml`

### P10P-G-004

- 状態: CONFIRMED
- 確度: 公式ローダと現行 planner の対応
- 重大度: 高（MossFormerGAN の学習率半減と早期停止）。MP-SENet は学習窓の同一性。
- 対象: MossFormerGAN 本格設定と MP-SENet 本格設定。
- 仮説: Dataset 件数と、条件が整ったときの発話順は公式に合う。学習に使う波形窓と、MossFormerGAN の検証に使う 2 秒は公式と違う。
- 観測:
  - 公式 MossFormerGAN `Wave_Processor.process` は毎 `__getitem__` で `random.randint(0, len_wav - segment_length)` する。train も validation も同じで、2 秒（`max_length: 2`）である。検証 loss が 5 回連続で直前より下がらないと generator の学習率を半分にし、10 回で止める。
  - 現行 Crop は `mix_seed("crop", seed, epoch, case_id) % (length - window + 1)` で、発話ごとに epoch 内 1 か所である。`CROP_NOTE` は公式 worker の `random.randint` 列を再現しないと書いている。検証は `per_epoch_random`。
  - 公式 MP-SENet の 1 GPU 順は `random.seed(1234)` の 1 回 shuffle で、現行 `fixed_single_shuffle_v1` とこの Python では一致した（先頭 `[10144, 3955, 4816, 11369, 987]`）。窓の開始だけが `random.randint` と `mix_seed` で分かれる。
  - 発話順の方は、11572 が 1、2、4 で割り切れる。MossFormerGAN の `rank_mod_fixed_v1` は公式 sampler と size 10、11、11572、world 1、2、4、epoch 0 と 3 で一致した。MP-SENet の DDP 順は、割り切れる N では pad と drop が同じである。
- 証拠: `src/se/official/clearervoice-studio/train/speech_enhancement/dataloader/dataloader.py`、`solver.py` の `train`、`src/se/official/mp-senet/dataset.py`、`src/reproduction/pipeline/planner.py` の `_crop`、`src/reproduction/pipeline/training_data.py` の `CROP_NOTE`。
- 再現: 順の一致は CPU で `rank_order` と公式 sampler の移植、および `random.seed(1234)` と `fixed_single_shuffle_v1`。窓の不一致は両実装の乱数源が違うことそのもの。実 WAV での開始位置の数値比較はしていない。
- 実験への影響: MossFormerGAN は検証 loss が別の 2 秒集合なので、半減、巻き戻し、停止の epoch が公式と分かれる。選ばれる `last_best` の重みが変わる。MP-SENet は同じ発話順でも、epoch ごとに見る 32000 サンプルが公式 DataLoader と違う。件数 11572 と 824 は、この差を消さない。
- テストが PASS し得る理由: 件数、`rank_mod_fixed_v1`、`fixed_single_shuffle_v1`、loss 式のテストは、公式 worker 乱数との窓一致を要求していない。
- 関連ファイル: `src/reproduction/pipeline/planner.py`、`src/reproduction/pipeline/order.py`、`src/reproduction/pipeline/training_data.py`、`src/training/se/mossformergan.py`、`configs/training/reproduction/mossformergan_vb28_official.yaml`、`configs/data/protocols/mossformergan_se_16k.fixed_pair.ondemand.yaml`

## 棄却した仮説

### H5 正規化の二重適用

- MP-SENet 公式 `dataset.py` は `librosa.load(..., sr=16000)` のあと、noisy RMS で clean と noisy を一度だけスケールする。現行 `noisy_rms_unit_v1` もレンダラで一度である。`MPSENetTask._waveforms` は再スケールしない。
- MossFormerGAN 公式は `audio_norm` のあと、solver が crop に対して noisy エネルギー係数 `c` を clean と noisy の両方に掛ける。現行は `independent_audio_norm_v1` のあと `_generator_terms` が同じ `c` を掛ける。二段目は公式の二段目である。公開 scp 名は 16 kHz で、公式 `audioread` は sample rate が既に 16000 なら resample しない。48 kHz からの変換器の差は protocol の `vb28-ondemand-resample` に書かれており、Trainer がレンダラ後にもう一度 `audio_norm` する形ではない。
- TF-GridNet `ms-snsd` は `normalize_variance` を loss に掛けない。公式 YAML の `normalize_variance: true` はコメントアウトである。TF-Locoformer はネットワーク側の `torch.std`（既定の不偏）と、loss 側 `normalize_variance=False` である。ESPnet `90eed8e` の `espnet_model.py` も `torch.std` で、loss の既定は False。

### H6 Chunk の欠落とサンプル数の単位

- ESPnet `0d136c5` の `build_chunk_iter_factory` は、先に `batches[rank::world_size]` し、そのあと YAML の `num_iters_per_epoch` を rank あたりの `num_samples_per_epoch` として渡す。cache は `num_cache_chunks // world_size`。短い rank で止まる、というコメントがある。
- 現行 `plan_chunk_epoch` は同じ shard、同じ rank あたりサンプル数、`max(num_cache_chunks // world, per_rank_batch)`、`min` の batch 数で切る。
- `select_source_ids` を、`SequenceIterFactory` の `num_iters_per_epoch` 分岐（N より小さい場合と大きい場合）の移植と、17 発話、num 5 / 17 / 23、epoch 1、2、3、6 で比較し、一致した。
- 余り batch を低い rank に渡す公式分岐は、現行が拒否する。公開レシピは global batch 4 に対して 2 rank と 4 rank で割り切れる。この拒否は本格経路の欠落ではない。
- `90eed8e` と `0d136c5` の `chunk_iter_factory.py` の差は、短い発話を残すオプションと `chunk_max_abs_length` である。DNS 既定は短い発話を捨て、chunk 長は 64000 の単一候補である。その経路の `choice`、`randint`、shift は同じ式である。

### H7 Scheduler の更新時点

- MP-SENet の `ExponentialLR.step` は公式も現行も epoch 末だけ。CPU カウンタで 2893 の倍数が一致した。P10P-G-002 は保存時点であり、学習率の更新時点ではない。
- TF-Locoformer の公式 `WarmupReduceLROnPlateau.step` は、metrics 無しのとき `step_num += 1` し、`step_num <= warmup_steps` のあいだ `lr = base * warmup_steps**-1 * step_num` である。warmup 中の `min(step**-0.5, step * warmup**-1.5)` はこの式に簡約される。現行 `step_batch` は同じ式で、optimizer step のあとである。公式 trainer も `optimizer.step` のあと `AbsBatchStepScheduler.step` である。最初の更新は構築時の学習率のまま、という対応は取れている。
- epoch 末の plateau は `valid / loss`。現行は generator loss を broadcast してから `step_validation` する。patience の比較 `num_bad_epochs > patience` は公式クラスと同じである。

### H8 Data Plan が実行条件で変わる

- 現行 Crop と順は `mix_seed` か policy の純関数で、`GroupLoader.map` は `ThreadPoolExecutor.map` の順序を保つ。worker は乱数を引かない。
- 再開位置は `data_state` の epoch と update で、plan hash が変わると拒否する。
- world size で順が変わるのは公式も同じである。MP-SENet は 1 process が `fixed_single_shuffle_v1`、2 process 以上が `torch_distributed_sampler_v1`。11572 は 2 と 4 で割り切れるので、公式 DistributedSampler の pad と現行の tail drop は同じ集合になる。

### H9 再開で学習率が変わる

- 現行は scheduler の `state_dict` を保存する。CPU で `ExponentialLR` を epoch step 前に保存し、別 optimizer へ load すると lr は 0.0005 のまま、次の `step` で両方 0.000495 になった。
- 公式は scheduler を保存せず、checkpoint の epoch 番号を `last_epoch` に渡す。中断 epoch の再開は公式側の別問題で、現行の state_dict 再開が途中から学習率を変える証拠は無い。

### H10 MossFormerGAN の 2 GPU 順

- `rank_order("rank_mod_fixed_v1", ...)` は、公式 `DistributedSampler.__iter__` の移植と一致した（上記の size / world / epoch）。
- 1 process の公式は `DataLoader(shuffle=True)` で、`rank_mod_fixed_v1` ではない。本格の 2 GPU スクリプト `train.sh` は `n_gpu=2` である。1 process 用 YAML は DDP sampler の world 1（epoch 毎の全件 shuffle、seed は epoch）であり、公式 1 GPU の `RandomSampler` ではない。公開経路が 2 GPU なので、本格 GO を崩す主欠陥にはしていない。1 process YAML を公式 1 GPU 実行とみなすなら、順は未再現のままである。

### H11 `mr_l1_tfd` の reduction

- 公式 `MultiResL1SpecLoss` は reduction `sum` で `(Batch,)` を返し、`FixedOrderSolver` が `loss.mean()` する。現行 `multi_resolution_l1` は発話ごとの sum を batch で mean する。batch 内のスケールは同じ形である。STFT 実装（公式 `Stft` と現行の reflect pad + `center=False`）の数値一致は、この調査では実行していない。式の対応だけでは「勾配が誤っている」とはしない。

### H12 DDP の勾配平均

- 公式は local loss を global weight で割ったあと `world_size` を掛け、DDP の平均と打ち消して global mean の勾配にする。
- 現行 TF-GridNet / TF-Locoformer は local mean を backward し、DDP が rank 間で平均する。batch サイズが rank 間で同じとき、global mean の勾配になる。公開レシピはその batch 分割である。
- MP-SENet の generator は forward を `no_sync` にし、backward 後に SUM して world size で割る。これも mean である。実行中の rank 状態は見ていない。

### H13 TF-Locoformer の空 architecture

- DNS YAML の separator は n_layers 6、emb_dim 128、n_heads 4、ffn 384/384、conv kernel 4、shift 1、dropout 0、eps 1e-5、n_fft 256、hop 128、`normalize_variance: true`。
- 現行 `_DNS_SEPARATOR` と `TFLocoformer` の既定はこれと一致する。公式コンストラクタの `attention_dim=128`、`pos_enc="rope"` も既定のままである。空 dict はこの既定を上書きしない。

### H14 DeepFilterNet3 と SEMamba++ の偽 GO

- `dfn3.online.official` と `semambapp.gsr.official` は `claim_label: official_reproduction` かつ `declared_status: blocked`。snapshot は `declared_status == ready` を要求し、`ReproductionTrainingData` も ready 以外を拒否する。blocked の公式素材では学習が始まらない。
- 代替 Dataset の claim は `protocol_conformant_substitute_materials` である。
- `configs/training/deepfilternet3.yaml` と `configs/training/semambapp.yaml` は VoiceBank の別経路で、コメントが公式オンライン混合ではないと書いている。`complete.json` は status、global_step、checkpoint だけで claim を公式とは書かない。provenance の claim を公式に書き換える経路は見当たらない。

### H15 早期停止の off-by-one

- 公式 `check_early_stopping` は `epoch - best_epoch > patience`。loss の同値は、挿入順に対する安定ソートで早い epoch を残す。
- 現行は `(loss, epoch)` の最小を best とし、`espnet_epoch - best_epoch > patience` で止める。patience 5（GridNet）と 10（Locoformer）の比較方向は公式と同じである。

## 未検証

- VoiceBank 824 件のうち、長さが hop 100 の倍数でない件数。16 kHz WAV がこの作業ツリーに無く、P10P-G-001 の平均が実際に負になるかは未計測。
- `score_pesq` の 0.1 境界が、実在の検証発話でどれだけ `na` になるか。使った Python に `pesq` は入っていない。
- 公式 `Stft(window=None, center=True)` と現行 rectangular STFT の、長波形での勾配の数値一致。
- 複数 rank を起動したときの DDP 勾配と BatchNorm buffer。CPU の計画比較だけである。
- MS-SNSD / DNS の snapshot 並びが、公式 shape ファイルのバイト列と一致するか。コードは `speech_mix_shape` 順を渡すと書いており、snapshot は開いていない。
- 公式 1 GPU MossFormerGAN の `RandomSampler` と、1 process YAML の `rank_mod_fixed_v1` の差が、採用する本格経路に含まれるか。`train.sh` の公開値は 2 GPU である。

## 読んだファイル

公式:

- `src/se/official/mp-senet/train.py`、`config.json`、`dataset.py`
- `src/se/official/mp-senet/models/model.py`、`models/discriminator.py`
- `src/se/official/clearervoice-studio/train/speech_enhancement/train.py`、`train.sh`、`solver.py`
- `src/se/official/clearervoice-studio/train/speech_enhancement/dataloader/dataloader.py`
- `src/se/official/clearervoice-studio/train/speech_enhancement/losses/loss.py`
- `src/se/official/clearervoice-studio/train/speech_enhancement/config/train/MossFormerGAN_SE_16K.yaml`
- `src/se/official/tf-locoformer/egs2/dns_ins20/enh1/conf/tuning/train_enh_tflocoformer.yaml`
- `src/se/official/tf-locoformer/espnet2/enh/separator/tflocoformer_separator.py`（コンストラクタ既定）
- ESPnet `0d136c5`: `egs2/ms_snsd/enh1/conf/tuning/train_enh_tfgridnet.yaml`、`espnet2/iterators/chunk_iter_factory.py`、`espnet2/iterators/sequence_iter_factory.py`、`espnet2/tasks/abs_task.py`、`espnet2/tasks/enh.py`、`espnet2/train/trainer.py`、`espnet2/train/reporter.py`、`espnet2/main_funcs/average_nbest_models.py`、`espnet2/enh/loss/criterions/time_domain.py`、`espnet2/enh/loss/wrappers/fixed_order.py`、`espnet2/enh/espnet_model.py`、`espnet2/schedulers/warmup_reducelronplateau.py`
- ESPnet `90eed8e` と `0d136c5` の `chunk_iter_factory.py` の diff、`90eed8e` の `espnet_model.py` の variance 正規化

現行:

- `configs/training/reproduction/mp_senet_vb28_official.yaml`
- `configs/training/reproduction/mossformergan_vb28_official.yaml`
- `configs/training/reproduction/p10b/tf_gridnet.yaml`
- `configs/training/reproduction/p10b/tf_locoformer.yaml`
- `configs/training/tf_gridnet.yaml`
- `configs/training/deepfilternet3.yaml`
- `configs/training/semambapp.yaml`
- `configs/data/protocols/mp_senet.fixed_pair.ondemand.yaml`
- `configs/data/protocols/mossformergan_se_16k.fixed_pair.ondemand.yaml`
- `configs/data/datasets/mp_senet.vb28.ondemand.official.yaml`
- `configs/data/datasets/dfn3.online.yaml`
- `configs/data/datasets/semambapp.online.yaml`
- `src/training/core/trainer.py`
- `src/training/core/checkpoint.py`
- `src/training/core/average.py`
- `src/training/core/distributed.py`
- `src/training/se/config.py`
- `src/training/se/optim.py`
- `src/training/se/losses.py`
- `src/training/se/mp_senet.py`
- `src/training/se/mossformergan.py`
- `src/training/se/tf_gridnet.py`
- `src/training/se/tf_locoformer.py`
- `src/training/se/spectral.py`
- `src/training/se/run.py`
- `src/training/data/sampler.py`
- `src/training/data/chunk_plan.py`
- `src/training/provenance.py`
- `src/reproduction/pipeline/training_data.py`
- `src/reproduction/pipeline/order.py`
- `src/reproduction/pipeline/planner.py`
- `src/reproduction/pipeline/renderer.py`
- `src/reproduction/pipeline/transform.py`
- `src/reproduction/pipeline/chunking.py`
- `src/reproduction/pipeline/loader.py`
- `src/reproduction/pipeline/store.py`
- `src/reproduction/ids.py`
- `src/se/models/tf_locoformer/model.py`
- `src/evaluation/metrics/pesq.py`
- `src/evaluation/metrics/sisdr.py`
