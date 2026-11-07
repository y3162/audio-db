# P10-C Preflight Agent B: 公式実装監査（TF-GridNet / TF-Locoformer）

発見者: Agent B。Phase 1 の独立調査。他エージェントの結論、`docs/training/reproduction/preflight/` の他ファイル、過去の P10 監査文書、`.agents/DECISIONS.md` は根拠にしていない。根拠は公式コード、現行コード、実 YAML、スナップショットの長さ、CPU 上の小テスト。

調査日の audio-db HEAD は `3942d27`。GPU は使っていない。PID 614674、614675、4109908 は止めていない。`src/`、YAML、テスト期待値、Dataset、Snapshot、DuckDB、checkpoint は変更していない。

## 実測 commit

| 対象 | 期待 | `git -C` の HEAD | 判定 |
|---|---|---|---|
| `src/se/official/tf-locoformer` | `7a615460d347ff7334a13dbb831d16280da72cdc`（v1.1.0） | 同じ。`git describe` は `v1.1.0`。detached HEAD、作業木の差分表示なし | ピン一致 |
| `src/se/official/espnet` | `0d136c5456d403cb2c0a4c13e76e8136e32e9b7a` | 同じ。detached HEAD | ピン一致 |
| TF-Locoformer が README で checkout する ESPnet | `90eed8e53498e7af682bc6ff39d9067ae440d6a4` | オブジェクトは存在する。`merge-base --is-ancestor` の終了コード 0 で、現 checkout の祖先 | 現在の作業木ではない |

ESPnet の作業木は sparse checkout（`/*`、`!/*/`、`/espnet2/`、cone）である。`egs2/` は作業木に無い。`espnet2/` の学習コードは作業木にある。

`90eed8e` の学習コードで、作業木の blob と比較できたもの:

- `espnet2/schedulers/warmup_reducelronplateau.py` の blob は `90eed8e` と HEAD で同一（`0c1ef80da9fbb2cc12b7b728ef5594cfbed57eaa`）。`git diff 90eed8e HEAD -- 当該ファイル` は空。
- `tests/training/test_tf_locoformer_scheduler.py` は、作業木のそのファイルを import する。import の前に `git rev-parse 90eed8e:espnet2/schedulers/warmup_reducelronplateau.py` と `git hash-object` が違うと `AssertionError` になる。ファイルが無いときだけ Skip する。この checkout ではファイルがあり、blob は一致した。テストが「90eed8e のファイルを import する」という記述は、この作業木では成立する。
- chunk iterator の rank 分配（`batch_size % world_size` の余りを低い rank に足す、`batches[rank::world_size]`、`num_cache_chunks // world_size`）は `90eed8e` の `build_chunk_iter_factory` と HEAD で同じ式だった。
- `validate_one_epoch` の `iterator_stop` は `90eed8e` にもある（`espnet2/train/trainer.py` の all_reduce）。

`90eed8e` に無く、HEAD にあるもの:

- `egs2/ms_snsd/` 一式。TF-GridNet の公開 MS-SNSD recipe は pin `0d136c5` の対象であり、`90eed8e` の対象ではない。追加 commit は `3515fe270`（Add MS-SNSD enh1 recipe）。この YAML の履歴はこの 1 commit だけだった。

ローカル blob が無く、ピンの commit URL から読んだ公式スクリプト（作業木には書いていない）:

- `0d136c5:egs2/ms_snsd/enh1/local/ms_snsd_data_prep.sh`
- `0d136c5:egs2/TEMPLATE/enh1/enh.sh`
- `90eed8e` と `0d136c5` の `egs2/dns_ins20/enh1/local/dns_data_prep.sh`（2 ファイルは `cmp` で一致）
- `90eed8e:egs2/dns_ins20/enh1/local/data.sh`
- `90eed8e:egs2/dns_ins20/enh1/README.md`

`GIT_NO_LAZY_FETCH=1` では上記シェルの blob は欠ける。YAML `egs2/ms_snsd/enh1/conf/tuning/train_enh_tfgridnet.yaml` はローカルの `git show HEAD:` で読めた。`conf/train.yaml` はその tuning YAML への symlink である。

README が結果の git hash として書く `239f5166a520566b132c359fb01ce38972aeeefc` は pin の祖先だが、その commit の tree に `egs2/ms_snsd/enh1/` は無い。README の数値表を、この YAML を含む commit の実行結果としては結び付けられない。

## 監査済み

TF-GridNet（公開 MS-SNSD SE recipe。論文実験ではない）:

- 初期化 `xavier_uniform`（`espnet2/torch_utils/initialize.py` の dim>1 への `xavier_uniform_`、1 次元 `.bias` の 0、Embedding / LayerNorm / GroupNorm の `reset_parameters`）。`90eed8e` と HEAD でこの枝の式は同じ。`chainer` 枝は HEAD で消えている。現行 `apply_espnet_initialization` は同じ順序。
- separator の Hann STFT。`n_fft=512`、`win_length=512`、`hop=256`、`center=True`。公式は `STFTEncoder` / `STFTDecoder` 経由の `torch.stft`。現行 `TFGridNet.forward` は `torch.stft` / `torch.istft(..., length=)`。入力を `torch.std` で割ってから STFT し、出力に戻す。
- `mr_l1_tfd`。クラス選択名であり、stats のキーではない。損失の STFT は矩形窓、hop は窓の半分、窓 256/512/768/1024、`time_domain_weight=0.5`、`reduction=sum`、`normalize_variance` 既定 False。
- Chunk iterator、`num_iters_per_epoch`（chunk では rank あたりの元発話数）、rank 分配、短い発話とキャッシュ端の破棄、`iterator_stop`、validation の重み付き平均、ReduceLROnPlateau、checkpoint 選択、shape file 順。
- 本学習に近い YAML は `configs/training/reproduction/p10b/tf_gridnet.yaml`（dataset `tf_gridnet.ms_snsd.espnet_recipe@1`）と、VoiceBank 用の `configs/training/tf_gridnet.yaml`。

TF-Locoformer（DNS Medium）:

- 保存 config `egs2/dns_ins20/enh1/exp/enh_train_enh_tflocoformer_raw/config.yaml` と tuning YAML、現行 `TFLocoformer` / `TFLocoformerSeparator` のコンストラクタ引数。
- STFT（Hann、256 / 128、`center=True`）と、`model_conf.normalize_variance: true` の `torch.std`。
- Chunk iterator と rank 別 plan。保存 config は `dist_world_size: 4`、`batch_size: 4`、`num_iters_per_epoch: 1000`、`chunk_length: 64000`、`num_cache_chunks: 1024`。
- Warmup 4,000、plateau、`valid.loss`、`valid.si_snr`、`ave_5best`、early stopping、checkpoint 平均、DDP validation。
- 現行 validation 36,000 件が sorted uttid の末尾 1/10 であること。`cv_synthetic_small` の件数と選択規則は公開ソースから追加で分からなかった。

## 未監査

- `GridNetBlock` と `TFLocoformerBlock` の forward を、同じ重みで数値比較していない。比べたのは recipe YAML とコンストラクタ引数。
- 公式 `Stft` クラスは `.venvs/se` に `torch_complex` が無く import できなかった。`torch.stft` の kwargs は読んだ。ISTFT の `pad2` と現行 `length=` のサンプル差は測っていない。
- `iterator_stop` の `all_reduce` を 2 process では実行していない。分岐は `trainer.py` の for/else を読んだ。
- DNS の 126,000 batch を公式 `ChunkIterFactory` で再実行していない。件数は `plan_chunk_epoch`。同じ関数は MS-SNSD の実長と合成列で公式クラスと一致した。
- README の HF 数値を出した実行ファイル一式は、引用 hash `239f516` の tree に recipe が無いため比べられない。
- 公式 DNS 混合のプロセス順依存と、現行の seeded draw の波形ビット一致は見ていない。
- `90eed8e` の `dns_data_prep.sh` 以外の未取得 blob（ローカルに無く、今回 URL で取っていないファイル）は UNKNOWN のままにした。

## テストと数値

CPU、`.venvs/se/bin/python`、PyTorch `2.6.0+cu124`、`CUDA_VISIBLE_DEVICES=""`、`torch.cuda.is_available()` は False。`OMP_NUM_THREADS=1`。

1. 公式 `ChunkIterFactory` と `plan_chunk_epoch`。合成長（40 本、1 本は chunk 未満、world 2、chunk 8、global batch 4、cache 8、shuffle false）で raw batch 数は `(97, 99)`。公式クラスと span（utt, start, stop）は両 rank とも不一致 0。短い発話は公式が warning を出して捨て、plan も捨てる。
2. MS-SNSD の登録済み valid 1,835 件（snapshot `13c6058881ef86fe7506f705b1bd5a5bc68b313181410d8fdaaee20c33ef3fc6`、utterance 順と `frames`）で、chunk 48,000、shift 0.5、world 2、global batch 4、`num_cache_chunks` 1024、shuffle false、`num_samples=None`。
   - `plan_chunk_epoch` の raw batch 数は `(2887, 2908)`。保持は両 rank 2,887。破棄 batch は `(0, 21)`。キャッシュ端の余り chunk は `(0, 0)`。per-rank batch 2、cache 512。
   - 公式 `ChunkIterFactory` は rank 0 が 2,887、rank 1 が 2,908。span 不一致は 0。
   - shuffle false のとき epoch と seed を変えても plan は同じだった。
   - 発話長の最小は 160,016、最大は 313,910。48,000 以下は 0 件。短い発話の破棄は 0。
   - サンプル合計 345,066,605。shuffle false なので各発話の開始は 0。チャンクに入らない末尾は合計 22,866,605 サンプル。重なりがあるので、チャンク内サンプル数の合計は発話長の合計より大きい。
   - rank 0 は 918 発話、raw chunk 5,774、保持後も 918 発話。rank 1 は 917 発話、raw chunk 5,816。破棄 21 batch は 42 chunk。その中だけに現れる発話は 6、保持側にも現れる発話は 1。rank 1 の損失に残るユニーク発話は 911。
3. DNS accept snapshot `5469f0416329a8661df2cc91cd5bd9a149d6ea223c72e69ebdd5696bea6bdfbe` の valid 36,000 件は、長さがすべて 480,000。world 4、chunk 64,000、global batch 4、cache 1024、shuffle false の plan は 4 rank とも raw 126,000、破棄 0、端 chunk 0、per-rank batch 1、cache 256。各 rank 9,000 発話。`(480000-64000)//32000+1 = 14` chunk/発話、`9000*14 = 126000`。末尾はちょうど 480,000 で、この長さでは未カバーの端サンプルは 0。公式クラスでの再実行はしていない。
4. `SequenceIterFactory` と `select_source_ids`。発話数より `num_samples` が大きい wrap を含む 4 ケース（4 発話から 10、7 発話から 5 と 20）で ID 列は一致。
5. 単一 chunk 長での `RandomState.choice(lengths, 1)` と `choice(lengths, size=1).item()` のあと `randint`。seed 0, 1, 17, 42 で同じ。`90eed8e` と HEAD の単一長の乱数消費は、この API 差では変わらなかった。複数長の抽選は試していない。両 recipe は単一長。
6. Warmup。公式クラスと現行 `WarmupReduceLROnPlateau`。初期 lr は両方 `0.001`。batch step を 4,000 回したあと `step_num=4000`、lr `0.001`。4,001 回目も lr `0.001`（warmup を超えた batch step は lr を書かない）。等しい metric `2.0` を 4 回入れると `best=2.0`、`num_bad_epochs=3`、lr はまだ `0.001`。両実装のこれらの値は一致。半減は `num_bad_epochs > patience` のときで、この 4 回では `3 > 3` にならない。5 回目は実行していない。
7. `torch.optim.Adam` の既定 `eps` はこの環境で `1e-8`。公式 YAML の `eps: 1.0e-08` と同じ値になる。現行 optimizer は `eps` を渡していない。
8. `SISNRLoss(eps=1e-7)` の clamp 換算はコード式 `-log10(eps/(1-eps))*10` で約 `70.0`。クラス自体は `torch_complex` が無く実行していない。`name` の既定 `si_snr_loss` はソースの代入で確認した。

## 一致範囲

公式 MS-SNSD YAML（HEAD）と、2 rank で動かすときの現行 plan:

- `max_epoch` 35、`patience` 5（early stopping は `valid` / `loss` / `min`）、`keep_nbest_models` 5。
- Adam、lr `1e-3`、weight decay 0、grad clip 既定 5、AMP false。
- `iterator_type: chunk`、chunk 48,000、shift 既定 0.5、`num_cache_chunks` 既定 1024、`num_iters_per_epoch` 5000。5000 は shard の後に各 rank の factory へ渡る元発話数。optimizer step 数ではない。train は rank あたり約 3,673 と 3,672 なので、5000 は shard 内で wrap する。wrap の式は上の小テストと一致した。
- `run.sh` の `ngpu=2` と batch 4 なら per-rank batch は 2。4 は 2 で割り切れるので、余りを低い rank に足す公式分岐は発火しない。現行 planner はその余りを拒否するが、この recipe では同じ 2。
- cache は `max(1024 // 2, 2) = 512`。公式は `// world` を渡したあと `ChunkIterFactory` が `max(..., batch_size)` する。同じ。
- validation は `num_iters_per_epoch=None`、shuffle は `train=False`。全 valid 発話が shard される。短い発話は `length <= chunk` で捨てる。この valid では 0 件。
- shape の基底順は `UnsortedBatchSampler(batch_size=1)` が読む最初の shape file。chunk 経路は `batch_type: folded` を使わない。公式 `make_set` は uttid で `sort -k1,1` する。登録済み valid の utterance key はソート済みだった。
- 損失の学習項は `mr_l1_tfd` の重み 1。`si_snr` の wrapper 重みは 0。`valid.loss` は重み付き和なので、この設定では mr_l1 の batch 平均になる。現行の `multi_resolution_l1` も発話内 sum、batch mean、矩形 STFT、同じ窓。
- plateau は `reducelronplateau`、mode min、factor 0.7、patience 1。現行 `plateau` は `ReduceLROnPlateau` に threshold `1e-4` rel を明示する。PyTorch 既定と同じ。epoch 末に validation の generator loss を渡す。公式の `val_scheduler_criterion` は `valid` / `loss`。
- early stopping は `epoch - best_epoch > patience`。現行も同じ不等号。patience 5 は「悪い epoch が 5 個」ではなく、最良 epoch からの差が 5 を超えたとき。公式 reporter と同じ。
- `valid.loss` の n-best 平均は float を n で割り、整数は和のまま。履歴が n 未満のときは平均せず `ave_1best` を最良 epoch へ向ける。現行もそうする。拡張子は公式 `.pth`、現行 `.pt`。
- xavier_uniform の手順は上のとおり。p10b YAML は `init: xavier_uniform` を付ける。重み列の一致は、同じモジュール順で同じ乱数を消費するときだけであり、forward 比較はしていない。

公式 DNS 保存 config と現行 p10b TF-Locoformer:

- Medium の separator 引数は保存 config と現行 `_DNS_SEPARATOR` で一致する。`n_layers=6`、`emb_dim=128`、`norm_type=rmsgroupnorm`、`num_groups=4`、`tf_order=ft`、`n_heads=4`、`flash_attention=false`、FFN は swiglu conv1d が 2 段で hidden 384、kernel 4、shift 1、dropout 0、eps `1e-5`。保存 config が書かない `attention_dim=128` と `pos_enc=rope` は公式クラスの既定で、現行もその既定を使う。公式ソースの `norm_type` 既定文字列は `rmsgrouporm` だが、recipe は `rmsgroupnorm` を渡す。
- `num_spk=1`、STFT 256/128、Hann、center true、`normalize_variance: true` は `torch.std`（unbiased 既定）。`90eed8e` と HEAD のモデル側も `torch.std`。`always_forward_in_48k` は HEAD で増えたが既定 False。
- world 4、global batch 4 なら per-rank batch 1。公式 README の学習例は `--ngpu 4`。保存 config の `dist_world_size` は 4。
- `num_iters_per_epoch` 1000 は rank あたりの元発話数。warmup は線形 `base * step / 4000`。docstring の `step**-0.5` は実装と違う。最初の optimizer 更新は構築時の lr `1e-3`。scheduler は optimizer の後。現行 `step_batch` も optimizer の後。上の数値は一致した。
- plateau は mode min、factor 0.5、patience 3、criterion `valid` / `loss`。early stopping も `valid` / `loss` / `min`、patience 10。
- 推論の公式指定は README の `valid.loss.ave_5best.pth`。exp ディレクトリにある重みファイルはそれ（約 60 MB）と画像 3 枚。画像名は `loss.png`、`l1_timedomain+magspec_loss.png`、`si_snr_loss.png`。`si_snr.png` は無い。
- 登録済み DNS valid 36,000 は train 324,000 と合わせて 360,000。各側は uttid ソートで、train の最後の文字列は valid の最初より小さい。`dns_data_prep.sh` の `num*9/10` の頭と、その補集合という件数と順序に合う。これはスクリプト上の `cv_synthetic` であり、`cv_synthetic_small` ではない。
- この 36,000 件は長さが揃っているので、公式と同じ chunk 式なら 4 rank の batch 数は等しく、`iterator_stop` は batch を捨てない。

DDP の損失平均。公式は step ごとに `recursive_average` で重み付き平均し、DDP の勾配平均を打ち消すため loss に world size を掛ける。現行は rank 内の batch 平均をそのまま backward し、DDP が勾配を平均する。per-rank の件数と batch サイズが等しいとき、勾配は同じ平均の勾配になる。all_reduce の数値比較はしていない。validation は、公式が step ごとの重み付き平均を reporter に積む。現行は batch サイズを重みに総和し、最後に割る。同じ batch 集合なら同じ重み付き平均になる。

## Issue

### P10P-B-001

- ID: P10P-B-001
- 発見者: Agent B
- 対象: TF-GridNet MS-SNSD validation の母集団
- 種別: RISK
- 深刻度: 中
- 確度: CONFIRMED
- 場所: `src/se/official/espnet/espnet2/train/trainer.py:839`、`src/se/official/espnet/espnet2/tasks/abs_task.py:2165`、`src/se/official/espnet/espnet2/iterators/chunk_iter_factory.py:188`、`src/reproduction/pipeline/chunking.py:207`
- 根拠: 登録済み valid 1,835 件の長さで、公式 `ChunkIterFactory` の batch 数は 2,887 と 2,908、span は `plan_chunk_epoch` と不一致 0。公式 `validate_one_epoch` は、ある rank の iterator が尽きた for/else で `iterator_stop=1` を all_reduce し、他 rank は次 batch を取り出してから処理前に break する。処理数は最短の 2,887。長い rank の 21 batch は損失に入らない。これは現行だけの仕様ではない。公式コメントも「最少の反復で打ち切り、超えたサンプルは捨てる」と書く。短い発話の破棄は 0、キャッシュ端の余り batch も 0。捨てられるのは iterator_stop の 42 chunk。rank 1 では 6 発話が破棄 batch にしか無く、1 発話は保持と破棄にまたがる。加えて shuffle false の端サンプル 22,866,605 はどの chunk にも入らない。
- 再現手順: snapshot の valid を utterance 順と `frames` で取り、world 2、chunk 48000、shift 0.5、global batch 4、cache 1024、shuffle false、`num_samples=None` で公式 `ChunkIterFactory` と `plan_chunk_epoch` を比べる。CPU、1 スレッドで約 3 秒。
- 影響: 発話候補は 1,835 件すべて。損失に効くのは、各 rank の chunk 列の先頭 2,887 batch。rank 0 の 918 発話はすべて入る。rank 1 は 917 発話のうち 911。6 発話は validation 損失に寄与しない。サンプル単位でも、各発話の末尾は入らない。この平均を「valid 1,835 発話の全サンプルの平均」や「全 chunk の平均」と呼ぶことはできない。公式 procedure としての validation ではある。DNS の登録済み 36,000 件は長さ 480,000 で rank 数が揃い、同じ規則でも batch 破棄は 0 だった。GridNet の不揃いは発話長の差から来る。
- 対応方針: 母集団の説明を、iterator に入った発話と、損失に残った batch に分ける。研究上の規則は変えない。21 batch を戻す、末尾サンプルを足す、発話を間引く、は評価規則の変更になる。計算を減らすなら、同じ保持集合（各 rank の先頭 2,887 batch）を分割して評価し、`batch平均 * batch内chunk数` の和と chunk 数の和を最後に 1 回割る。公式 reporter の重み付き平均、現行 `_reduce_weighted_reports` と、同じ集合なら同じ値になる。これは提案であり、採用ではない。
- P10-C影響: NON-BLOCKER
- 検証状態: batch 数と span は公式クラスで確認。21 batch の破棄は trainer の for/else を読んだ。2 process の all_reduce は未実行。

### P10P-B-002

- ID: P10P-B-002
- 発見者: Agent B
- 対象: `valid/si_snr/max` と checkpoint 名
- 種別: OFFICIAL_DIFF
- 深刻度: 中
- 確度: CONFIRMED
- 場所: `src/se/official/espnet/espnet2/enh/loss/criterions/time_domain.py:255`、`src/se/official/espnet/espnet2/enh/loss/wrappers/fixed_order.py:41`、`src/se/official/espnet/espnet2/enh/loss/criterions/time_domain.py:450`、`src/se/official/espnet/espnet2/tasks/enh.py:526`、`src/se/official/espnet/espnet2/train/trainer.py:445`、`src/training/se/tf_gridnet.py:174`、`src/training/se/tf_locoformer.py:168`、`src/training/core/trainer.py:1025`
- 根拠: 保存 config と MS-SNSD YAML の `best_model_criterion` は `valid/si_snr/max` と `valid/loss/min`。criterion は `get_class(ctr["name"])(**ctr["conf"])` で、`conf` に `name` は無い。`SISNRLoss` は `name is None` のとき `_name = "si_snr_loss"`。`forward` は `fast_bss_eval.si_sdr_loss` で、戻りは負の SI-SDR。`eps: 1e-7` は clamp_db 約 70 に換算される。`FixedOrderSolver` は `stats[criterion.name]` にその損失を書く。`MultiResL1SpecLoss.name` はコンストラクタ名を無視して `l1_timedomain+magspec_loss` を返す。trainer は `reporter.has(phase, key)` のときだけ `{phase}.{key}.best.pth` を作る。`si_snr` が stats に無ければ `valid.si_snr.best.pth` は作られない。`valid.loss` はある。early stopping と plateau は `valid.loss`。平均も `reporter.has` が真の criterion だけ。公開 exp の画像は `loss.png`、`l1_timedomain+magspec_loss.png`、`si_snr_loss.png`。重みは `valid.loss.ave_5best.pth`。README の推論モデルもそれ。現行は `valid.si_snr.best.pt` を、履歴の `si_snr` を大きい順に選んで書く。TF-GridNet は `score_si_sdr` の正の dB を符号反転して入れるので、最大化は SI-SDR が低い側を選ぶ。TF-Locoformer は `si_sdr_loss` の符号を戻した正の dB を最大化するので、SI-SDR が高い側を選ぶ。どちらも公式が書かないファイルである。拡張子は `.pt`。現行の SI-SDR は `fast_bss_eval` ではなく、clamp も公式の約 70 dB とは違う。学習損失の重み 0 なので、mr_l1 の勾配には入らない。
- 再現手順: 保存 config の criterions と、上の 3 画像のファイル名を見る。`SISNRLoss.__init__` と `FixedOrderSolver.forward` と trainer の `reporter.has` を読む。現行は `tf_gridnet.py` の `_negative_si_sdr` と `tf_locoformer.py` の `_positive_si_sdr`、trainer の `(-row["si_snr"], row["epoch"])` を読む。
- 影響: 公式のモデル選択と早期停止と plateau と `ave_5best` は `valid.loss`。現行の `valid.loss.best` と `valid.loss.ave_5best` も generator loss の最小で、この部分は同じ規則。`valid.si_snr.best` を公式の選択結果として使うと、公式にそのファイルが無く、GridNet と Locoformer で符号が逆で、数値も `fast_bss_eval` ではない。n-best の保持集合は loss と si_snr の和集合なので、loss の 5 本に加えて si_snr 側の epoch ファイルが残る。loss の平均に使う 5 本自体は loss 順位のまま。
- 対応方針: 公開 recipe の推論に使うファイルは `valid.loss` の最良と `ave_5best` とする。`valid.si_snr.best` は公式成果物として扱わない。符号を公式 stats に合わせるか、キーが無い公式挙動に合わせるかは、この監査では決めない。ファイル名を `.pth` に変えることも、この監査ではしない。
- P10-C影響: NON-BLOCKER
- 検証状態: キー名はソースと公開 exp の画像名。clamp の 70 dB は式の計算。`fast_bss_eval` との数値差は未計測。`torch_complex` が無く損失クラスは import していない。

### P10P-B-003

- ID: P10P-B-003
- 発見者: Agent B
- 対象: `cv_synthetic_small`
- 種別: UNKNOWN
- 深刻度: 高
- 確度: CONFIRMED
- 場所: 保存 config `valid_data_path_and_name_and_type` の `dump/raw/cv_synthetic_small/wav.scp`。公開スクリプト `egs2/dns_ins20/enh1/local/dns_data_prep.sh:67`。現行 snapshot の valid 36,000
- 根拠: `90eed8e` と `0d136c5` の `dns_data_prep.sh` は同一で、uttid を `sort -k1,1` し、`num*9/10` を `tr_synthetic`、残りを `cv_synthetic` にする。`cv_synthetic_small` という文字列は、このスクリプト、`local/data.sh`、TEMPLATE `enh.sh`、TF-Locoformer 作業木の DNS recipe（保存 config の 2 行を除く）に無かった。`90eed8e` の DNS README が書く validation 名は `enhanced_cv_synthetic`。保存 config は `cv_synthetic_small` を指す。件数、seed、抽出規則は付いていない。現行 36,000 は 360,000 の sorted uttid 末尾で、スクリプトの `cv_synthetic` の件数と順序には合う。それを `cv_synthetic_small` と同一とは言えない。
- 再現手順: ピン URL の `dns_data_prep.sh` を `rg cv_synthetic` する。保存 config の valid パスと、snapshot の train 324,000 / valid 36,000 を見る。
- 影響: 公開 checkpoint が学習時に見た validation 集合の中身は、この調査では決まらない。現行の 36,000 は別の名前の公開分割規則とは一致する。ここを取り違えると、公開 run と同じ validation で選んだモデルだと主張することになる。
- 対応方針: 未知のままにする。36,000 件の規則は変えない。計算を減らす提案は、この 36,000 から作る chunk 集合を分割し、重み付き和を最後に割ること。集合の抽出や `iterator_stop` の有無を変えない。この長さでは `iterator_stop` の破棄は 0 なので、分割の前後で平均が同じなら、全 126,000 batch × 4 rank を一度に持たなくてよい。提案であり、採用ではない。
- P10-C影響: NON-BLOCKER
- 検証状態: 公開スクリプトと保存 config と snapshot 件数は確認。`cv_synthetic_small` の実ファイルは手元に無い。

### P10P-B-004

- ID: P10P-B-004
- 発見者: Agent B
- 対象: MS-SNSD の train/valid カット
- 種別: OFFICIAL_DIFF
- 深刻度: 中
- 確度: CONFIRMED
- 場所: 公式 `ms_snsd_data_prep.sh:82`（`find | sort`、`num * 8 / 10`、`head` / `tail`）。現行 `src/reproduction/build/ms_snsd.py:416` の `_clip_atomic_cut`。snapshot `dataset.json` の deviation `clip_atomic_split`
- 根拠: 公式シェルはソートした noisy 一覧を `count*8/10` で切る。クリップ境界へ動かさない。その後 TEMPLATE `enh.sh` が `min < utt2num_samples < max` で残す。`run.sh` は `max_wav_duration=31`、template の min は 0.1 秒。16 kHz では 1600 と 496000 の開区間。現行の duration 条件は同じ開区間。現行はカットの前に、同じクリップの 5 SNR が両側に割れるとき、近いクリップ端へ動かす。登録済み集合は train 7,345、valid 1,835、test 305。valid は uttid ソートで、train との境界クリップは `_clnsp66` と `_clnsp670` で違う。移動した件数は snapshot に無い。duration で落ちたファイルも snapshot に無いので、公式シェルと同じ集合かはここからは出ない。
- 再現手順: ピンの `ms_snsd_data_prep.sh` の `train_num` と、現行 `_clip_atomic_cut` を読む。snapshot の valid/train 境界の utterance key を比べる。
- 影響: カットがクリップ内なら、公式は同じ clean が train と valid に分かれる。現行はそれを避けるので、発話集合が公式 `cv_ms_snsd` と違う。違うなら chunk 数 2,887 / 2,908 は現行集合の数であり、公式が生成した wav 一覧の数ではない。移動が 0 なら、duration フィルタ後の集合は公式と同じになり得る。そのどちらかは UNKNOWN。
- 対応方針: 集合を公式シェルに戻すかどうかは、この監査では決めない。P10-C の学習対象が登録済み snapshot なら、その 1,835 の長さで上の iterator 結果は再現する。公式 dump との集合一致は主張しない。
- P10-C影響: NON-BLOCKER
- 検証状態: シェルと現行関数と snapshot の境界は確認。移動件数は未記録。

### P10P-B-005

- ID: P10P-B-005
- 発見者: Agent B
- 対象: MS-SNSD / DNS の本学習 YAML
- 種別: RISK
- 深刻度: 高
- 確度: CONFIRMED
- 場所: `configs/training/reproduction/p10b/tf_gridnet.yaml:14`、`configs/training/reproduction/p10b/tf_locoformer.yaml:16`、`configs/training/tf_gridnet.yaml:15`
- 根拠: 公式規模の dataset を指す YAML は p10b の 2 ファイル。両方とも `verification_max_updates: 2`、`chunk_iterator.max_updates: 2`、`validation_max_updates: 1`。trainer はこの上限で `interrupted` になり、`complete` にはしない。validation は `iterator_stop` 後の先頭 1 batch だけ。MS-SNSD の 2,887 batch でも、DNS の 126,000 batch でもない。`configs/training/tf_gridnet.yaml` は epoch 35 と patience 5 を持つが、データは VoiceBank、`init` が無く、`micro_batch_size_per_gpu: 4` なので 1 process では per-rank batch 4 になる。公式の 2 GPU・per-rank batch 2 ではない。上限の無い MS-SNSD 本学習 YAML は `configs/training` に無かった。
- 再現手順: 上記 YAML の `verification_max_updates` と `validation_max_updates` を見る。`trainer.py` の `_at_verification_limit` と `_validate_aligned_chunks` の `limit` を見る。
- 影響: p10b YAML のままでは公式 epoch 予算も、上で測った validation 集合も実行されない。VoiceBank の `tf_gridnet.yaml` は別実験で、xavier も rank 分割も公式 MS-SNSD と違う。
- 対応方針: 本学習用に、smoke 上限だけを外した YAML が要る。この監査では YAML を変えていない。外すのは `verification_max_updates`、`max_updates`、`validation_max_updates`。epoch、chunk、損失、データ集合は変えない、という提案であり、採用ではない。
- P10-C影響: BLOCKER
- 検証状態: YAML と trainer の停止条件を読んだ。ジョブは投げていない。

### P10P-B-006

- ID: P10P-B-006
- 発見者: Agent B
- 対象: TF-Locoformer の分散 0 付近
- 種別: RISK
- 深刻度: 低
- 確度: CONFIRMED
- 場所: `src/se/models/tf_locoformer/model.py:58`、`src/se/official/espnet/espnet2/enh/espnet_model.py:311`
- 根拠: 公式は `torch.std` で割る。現行は std が `1e-6` 未満の chunk で `ValueError` を上げ、学習を止める。通常の DNS 混合がこの床を割ることは、この調査では見ていない。
- 再現手順: `TFLocoformer.forward` の比較と、公式 `normalize_variance` 分岐を読む。
- 影響: ほぼ無音の chunk で、公式は非有限勾配を skip し得る。現行はその step で run が落ちる。36,000 件の計画長 480,000 では未発生。
- 対応方針: 規則は変えない。無音 chunk が出てから扱う。
- P10-C影響: NON-BLOCKER
- 検証状態: コード比較のみ。実音声での std 分布は未計測。

### P10P-B-007

- ID: P10P-B-007
- 発見者: Agent B
- 対象: README の結果 hash と recipe の所在
- 種別: UNKNOWN
- 深刻度: 中
- 確度: CONFIRMED
- 場所: pin の `egs2/ms_snsd/enh1/README.md` の Git hash `239f5166a520566b132c359fb01ce38972aeeefc`。recipe 追加は `3515fe270`
- 根拠: `239f516` は pin の祖先で、その tree に `egs2/ms_snsd/enh1` は無い。tuning YAML は `3515fe270` の 1 commit だけで pin に入っている。README の結果表の config 名は `train_enh_tfgrid.yaml` で、pin のファイル名 `train_enh_tfgridnet.yaml` とも違う。
- 再現手順: `GIT_NO_LAZY_FETCH=1 git ls-tree 239f516 egs2/ms_snsd/enh1/conf` と、`git log HEAD -- egs2/ms_snsd/enh1/conf/tuning/train_enh_tfgridnet.yaml`。
- 影響: README の STOI / SI_SNR を、pin の YAML を実行した結果としては使えない。pin の YAML 自体の学習規則は、このファイルの前半で読んだ内容が対象。
- 対応方針: 数値表を再現目標にしない。学習するなら pin の YAML を対象にする、と分けて書く。この監査では目標を変えていない。
- P10-C影響: NON-BLOCKER
- 検証状態: tree の有無は確認。`239f516` 時点の別ブランチや未 commit の実験ファイルは見ていない。

## 反証して問題にならなかった仮説

- 「validation の 2,887 / 2,908 と 21 batch 破棄は現行 planner 固有」。公式 `ChunkIterFactory` が同じ 2,887 と 2,908 を返し、公式 trainer が最短で止まる。現行 plan の span も一致した。
- 「21 batch は短い発話かキャッシュ端の破棄」。短い発話は 0、端 chunk も 0。42 chunk は長い rank の末尾 batch。
- 「`num_iters_per_epoch: 5000` は全体の発話数、または optimizer step」。shard 後の各 rank に同じ数が渡る。chunk iterator では元発話数。wrap の小テストは公式 `SequenceIterFactory` と一致した。
- 「`batch_type: folded` が chunk の順序を変える」。chunk factory は `UnsortedBatchSampler(batch_size=1)` で shape file 順を読む。
- 「余り batch の rank 差が、この recipe の batch サイズを変える」。batch 4 と world 2、batch 4 と world 4 は割り切れる。公式の `rank < batch % world` は偽。
- 「`90eed8e` の scheduler が作業木に無く、テストは別実装同士」。blob は HEAD と同一で、テストは作業木のそのファイルを import する。
- 「warmup は docstring の `step**-0.5`」。実装は `base * step / warmup_steps`。4,000 step 後の lr は `1e-3`。公式クラスと現行は一致した。
- 「`best_model_criterion` の `si_snr` が `valid.si_snr.best.pth` を作る」。stats のキーは `si_snr_loss`。`reporter.has` が偽ならファイルは作られない。公開画像名が `si_snr_loss` だった。
- 「`90eed8e` と HEAD の chunk rank 分配が DNS と MS-SNSD で違う」。`build_chunk_iter_factory` の余りと stride は同じ式。短い発話を残す旗は HEAD で増えたが既定 True で、`90eed8e` の常時破棄と同じ。両 recipe は単一 chunk 長で、choice の乱数消費も試した 4 seed では同じだった。
- 「DNS validation も GridNet と同じ rank 破棄がある」。登録済み 36,000 件は長さがすべて 480,000 で、plan の 4 rank は 126,000 で破棄 0。
- 「モデル側 `normalize_variance` の式が `90eed8e` と HEAD で変わった」。変わったのは損失側の `std` から `sqrt(var+eps)` で、両 recipe の損失は `normalize_variance` を有効にしていない。モデル側は `torch.std` のまま。
- 「plateau の patience 3 は 3 回の非改善で半減する」。実測では等しい metric 4 回のあと `num_bad_epochs=3` で lr はまだ `1e-3`。条件は `>`。公式クラスと現行はそこまで一致した。
