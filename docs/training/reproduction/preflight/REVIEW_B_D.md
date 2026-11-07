# P10-C Preflight Phase 2 相互レビュー（Agent B × Agent D）

対象は Chunk Iterator と rank 分配。読んだ報告は `AGENT_B_OFFICIAL_TF_MODELS.md` と `AGENT_D_DDP_VALIDATION.md` だけである。HIGH、21 batch 破棄、`iterator_stop`、loss 分母、world 1/2/4 の一致は、引用先の現行コードと公式コードを読み直し、snapshot は読み取りだけで再計算した。

GPU は使っていない。本番コード、YAML、DB、snapshot は変更していない。学習プロセスは止めていない。公式 `ChunkIterFactory` の再実行は、MS-SNSD valid・world 2・shuffle false の ID 列だけである。2 process の `all_reduce` は実行していない。

このレビューで「snapshot」と書いたディレクトリ名は dataset id である。manifest の `snapshot_hash` は別で、MS-SNSD が `a69a477af70bc43dbcabdafbd5c97ec8a26aac4debd51158079545de1e4f0757`、DNS が `3343a55efb4c57d9ffef9179fb2f39c3e0186d36f587a0f4ad9f0aa14e9926f5` である。B が書いたハッシュはディレクトリ名と一致し、中身はそれぞれ `tf_gridnet.ms_snsd.espnet_recipe@1` と `tf_locoformer.dns.accept_recipe@1` だった。

## 判定

二つの報告が反対の結論を述べている項目は無い。したがって DISPUTED に上げる事実は無い。

機構として一致するのは、「最短 rank の prefix だけを処理し、それより後の batch は loss の分子にも分母にも入らない」ことである。21 batch、発話 6 件、MS-SNSD 1,835 件が全件 validation ではないこと、は B の測定である。D は同じ機構を一般形で書いており、これらの件数は書いていない。件数まで両報告の合意として扱わない。

DNS の 36,000 と 126,000 は矛盾しない。36,000 は validation 発話数、126,000 は rank あたりの validation batch 数である。D の「126 比較」は別物で、合成ケースの比較回数である。

BLOCKER は B-005 だけである。現行の本学習 YAML は smoke 上限付きで、測った validation 集合も公式 epoch 予算も実行しない。21 batch の母集団差は公式 procedure と同じで、NON-BLOCKER のままである。

## 三つの主張は両報告で同じか

| 主張 | B | D | このレビュー |
|---|---|---|---|
| 21 batch は公式 `iterator_stop` で、現実装固有ではない | 明示。valid 1,835、world 2、raw `(2887, 2908)`、保持 2,887、破棄 `(0, 21)`。公式 factory と span 不一致 0。2 process の `all_reduce` は未実行 | 件数 21 は無い。機構として、現行は `min(raw_batch_counts)` の prefix で、公式 trainer は最短 rank の終了後を処理しない、と書く。合成例は raw 14 と 26 | 機構はコードで同じ。件数は現行 `plan_chunk_epoch` と公式 factory の ID 列で再計算し、B と一致した。D の合意には含めない |
| 捨てた chunk は loss 分母に入らない | 「21 batch は損失に入らない」。分母という語は使っていない | 短い発話、cache 端、`updates` 以降は `train_update` にも aligned validation にも渡らず、分子にも分母にも入らない | aligned chunk 経路ではそのとおり。公式 reporter も未処理 batch を `register` しない |
| 全件 validation ではない | 明示。1,835 発話の全サンプル平均でも、全 chunk の平均でもない。発話 6 件は寄与しない。端サンプル 22,866,605 はどの chunk にも入らない | chunk 経路では prefix だけが分母に入ると書く。1,835 や発話 6 件は無い。別経路のレシピ validation は「rank 0 が全件」 | chunk の MS-SNSD world 2 については B が正しく、再計算でも全件ではない。D の「全件」はレシピ経路の別 validator で、chunk の 1,835 には使えない |

## 一致

公式 `validate_one_epoch` は、loop の先頭で `iterator_stop` を `all_reduce` し、正なら処理前に `break` する。iterator を尽きさせた rank だけが `for/else` で flag を 1 にする。処理数は最短 iterator の長さで、長い rank の残りは `reporter.register` されない。`90eed8e` の `espnet2/train/trainer.py` にも同じ 10 行の `iterator_stop` がある。現行 `plan_chunk_epoch` は `updates = min(raw_counts)` とし、`rank_batches` をその prefix にする。`_validate_aligned_chunks` と `_run_aligned_chunk_epochs` が渡すのはこの prefix だけである。

公式の weight は `speech_ref[0].shape[0]`（batch 次元）で、reporter の `WeightedAverage` は `sum(value * weight) / sum(weight)` である。未登録の batch は分母に入らない。現行の aligned validation は batch 平均の値に `len(case_id)` を掛ける。chunk 1 本が batch の 1 行なら、batch サイズが 2 でも重みの意味は同じである。D の「chunk 1 本が 1 サンプルなら」は、batch サイズが 1 のときだけ、という意味には読めない。

MS-SNSD valid、world 2、chunk 48,000、shift 0.5、global batch 4、cache 1024、shuffle false、`num_samples=None` を、case set の position 順と `frames` で再計算した。

- 発話 1,835。長さの最小 160,016、最大 313,910、合計 345,066,605。48,000 以下は 0。
- 現行 plan の raw batch は `(2887, 2908)`、破棄 `(0, 21)`、cache 端 chunk `(0, 0)`、選択発話 `(918, 917)`。
- 公式 `ChunkIterFactory` に同じ shard と cache 512 を渡した ID 列は、両 rank とも現行 plan と不一致 0。件数も 2,887 と 2,908。
- したがって 21 は、短い発話の破棄でも cache 端でもなく、長い rank の末尾 21 batch である。21 batch × per-rank batch 2 = 42 chunk。

DNS valid 36,000 は `frames` がすべて 480,000。world 4、chunk 64,000、global batch 4、cache 1024、shuffle false の plan は 4 rank とも raw 126,000、破棄 0、端 chunk 0、選択 9,000。`(480000-64000)//32000+1 = 14`、`9000*14 = 126000`。公式 factory はこの 36,000 件では再実行していない。B も同じ留保を書いている。

rank 分配の式も一致する。shard は `batches[rank::world_size]`、cache は `num_cache_chunks // world_size` のあと `max(..., batch_size)`、`global_batch % world_size != 0` のとき公式は低い rank に 1 を足し、現行 planner は拒否する。公開設定の global 4 は world 1、2、4 で割り切れる。`num_samples_per_epoch` は shard 後の各 rank に同じ回数だけ渡る元発話数で、optimizer step 数ではない。

p10b の 2 YAML は `verification_max_updates: 2`、`chunk_iterator.max_updates: 2`、`validation_max_updates: 1` である。trainer はこの上限で `INTERRUPTED` になり、`complete.json` は書かない。validation は `iterator_stop` 後の prefix をさらに 1 batch だけ切る。D も、p10b の smoke だけがこの上限 1 で、切った分は分母に入らない、と書く。

## DISPUTED

該当なし。

合意にしてはいけない範囲は次のとおり。

- 21 batch と発話 6 件を、D も確認した件数として書かない。D が反証したのは「捨てた chunk が分母に入る」という一般仮説で、根拠の数値例は合成の 14 と 26 である。
- D の world 1/2/4、不一致 0、126 比較を、MS-SNSD や DNS の検証集合が world size をまたいで同じだ、と読まない。126 は rank ごとの比較回数で、`(1+2+4) rank × num_samples 3 通り × shuffle 2 × epoch 3 = 126` と数えると合成グリッドの件数になる。リポジトリ内の `tests/training/test_chunk_official.py` はより小さいグリッドで、D の `/tmp` スクリプトはこのレビューでは再実行していない。
- D のレシピ経路「rank 0 が全件」を、TF-GridNet / TF-Locoformer の chunk validation に当てはめない。chunk は `_validate_aligned_chunks` が全 rank の prefix を重み付きで集約する。
- DNS の 126,000 を、1 epoch の学習更新数として書かない。学習の `num_samples_per_epoch: 1000` を同じ長さで plan すると、4 rank とも raw 14,000、破棄 0 だった。14,000 は 1,000 発話 × 14 chunk で、validation の 126,000 とは別である。

## 発話 6 件が損失に入らない根拠

これは validation、world 2、shuffle false、`num_samples=None` の計画についての主張である。学習 epoch の主張ではない。

rank の shard は重ならない。rank 0 の 918 発話は、保持した 2,887 batch のすべてに現れる。rank 1 の 917 発話のうち、破棄 21 batch にしか現れないものが 6、保持と破棄の両方に現れるものが 1 である。保持側に残る rank 1 のユニーク発話は 911。6 件は rank 0 の保持 batch にも無いので、どの rank の `validation_report` にも渡らない。weight は処理した batch の `len(case_id)` だけなので、この 6 件の chunk は分子にも分母にも入らない。

再計算した utterance key と、破棄側の chunk 数:

- `noisy999_SNRdb_30.0_clnsp999` が 7
- `noisy99_SNRdb_0.0_clnsp99` が 7
- `noisy99_SNRdb_20.0_clnsp99` が 7
- `noisy99_SNRdb_40.0_clnsp99` が 7
- `noisy9_SNRdb_10.0_clnsp9` が 5
- `noisy9_SNRdb_30.0_clnsp9` が 5

合計 38 chunk。またがる `noisy999_SNRdb_10.0_clnsp999` は保持 3、破棄 4 である。38+4=42 で、21 batch × 2 chunk と一致する。またがる 1 件は、保持された 3 chunk の分だけ損失に入る。「7 件がまるごと入らない」ではない。

B の報告本文には key の一覧が無く、再現手順も batch 数と span の比較までで、6 件の数え方は手順に分かれていない。このレビューの再計算で件数と内訳は再現した。D はこの 6 件を述べていない。

これとは別に、shuffle false の未カバー端は発話ごとに `(length - 48000) % 24000` で、合計 22,866,605 サンプルだった。これは chunk になる前の余りで、42 chunk とは別である。D の破棄三分類（短い発話、cache 端、iterator の尾）には、この発話内の未カバー端は入っていない。損失に入らない、という帰結は chunk にならないことから従うが、D がこの 22,866,605 を確認したことにはならない。

## DNS の 36,000 と 126,000

矛盾しない。単位が違う。

- 36,000 は valid 発話数。train 324,000 と合わせて 360,000 で、末尾 1/10 である。uttid は両側ともソート済みで、train の最後の文字列は valid の最初より小さい。
- 長さは train も valid もすべて 480,000。chunk 64,000、shift 32,000 なので発話あたり 14 chunk。末尾の未カバーサンプルは 0。
- world 4 なので rank あたり 9,000 発話。per-rank batch は 1 なので、validation batch は rank あたり 126,000。4 rank で 504,000。`36000 * 14 = 504000` と一致する。
- B の「全 126,000 batch × 4 rank」は、この 504,000 回の batch 評価を一度に持たない、という意味で、126,000 を全体の batch 数とした書き方ではない。
- B-005 の「DNS の 126,000 batch」は、`validation_max_updates: 1` が実行しない validation 側の rank あたり batch 数である。学習の 14,000 更新とは別の文脈である。
- この長さと batch 1 では cache 256 でも端 chunk は構造上 0 で、rank 間の長さが同じなので `iterator_stop` の破棄も 0 である。公式 class での再実行は、B もこのレビューもしていない。

## world 1/2/4

D の「一致」は、試した合成入力で `plan_chunk_epoch` と公式 `ChunkIterFactory` の batch 列が world 1、2、4 のそれぞれで一致した、という意味である。同じ valid 集合の損失が world size を変えても同じ、という意味ではない。B も後者は主張していない。

同じ MS-SNSD valid 1,835 を global batch 4、cache 1024、shuffle false で plan すると、chunk の総数は 11,590 で、world ごとに捨てるものが変わる。

| world | raw batch | 保持 batch | 破棄 batch | cache 端 chunk |
|---|---|---|---|---|
| 1 | 2,897 | 2,897 | 0 | 2 |
| 2 | 2,887 と 2,908 | 両方 2,887 | 0 と 21 | 0 と 0 |
| 4 | 2,890、2,906、2,884、2,910 | 全 rank 2,884 | 6、22、0、26 | すべて 0 |

world 1 は発話 1,835 をすべて shard に載せるが、batch サイズ 4 の端 2 chunk は batch にならない。world 2 が落とす 42 chunk、world 4 が落とす 54 chunk（破棄 batch の和）とは別の集合である。rank 間の保持 batch 数が揃うのは `min` の定義で、評価に残る発話集合が world 間で同じだとは言えない。

D が「kept chunk 数は rank 間で同じ」と書いたのは、一つの world の中の話として読める。その読みなら、上の表とも公式の最短打ち切りとも矛盾しない。

## 根拠が弱い点

- 21 batch が損失に入らないこと自体は、trainer の `for/else` を読めば処理前の `break` で足りる。2 process の `all_reduce` が同じ collective 回数で完了することは、B も D も実行していない。死活の確認は未実施のままである。件数と ID 列は、分散を起動せずに再計算できた。
- D の 126 比較、合成の raw 14 と 26、gloo の数値は `/tmp` のスクリプトにあり、このリポジトリからは再実行していない。機構はソースと、このレビューの MS-SNSD ID 比較で足りる。126 という回数そのものは、このレビューでは未確認である。
- 「同じ batch 集合なら同じ重み付き平均」は、公式が batch 平均を batch サイズで重み付けし、現行も batch 平均を `len(case_id)` で重み付ける、という式の対応である。同じ波形で loss の float を比べてはいない。
- B の「短い発話を残す旗は既定 True」は旗の名前が逆である。HEAD の旗は `chunk_discard_short_samples` で既定 True、つまり破棄する。`90eed8e` にはこの旗が無く、chunk 長が無いサンプルは常に `continue` で捨てる。挙動の結論（既定は常時破棄と同じ）は正しく、呼称だけが違う。
- B-005 の「公式規模の dataset を指す YAML は p10b の 2 ファイル」は、DNS については狭い。`configs/training/tf_locoformer_dns_accept_recipe.yaml` も `tf_locoformer.dns.accept_recipe@1` を指す。ただしこちらは `max_steps: 2`、global batch 1、`num_samples_per_epoch: 1` の smoke で、126,000 batch の validation も 150 epoch も実行しない。BLOCKER の結論は残る。
- B-003 の `cv_synthetic_small` について、このレビューは `dns_data_prep.sh` を読み直していない。作業木の sparse checkout には `egs2/` が無い。snapshot の `dataset.json` は、同じ名前を UNKNOWN とする deviation をすでに持っている。未知のまま、という結論を覆す材料は無い。公開 checkpoint が見た集合の中身が分かったわけでもない。

## 見落とし

- 21 と 6 は validation の world 2 に限る。同じ valid の world 1 は端 2 chunk、world 4 は破棄 batch `(6, 22, 0, 26)` で、6 件のリストはそのままでは成り立たない。どちらの報告も、この world 差を MS-SNSD の実長では書いていない。
- world 1 の単プロセス chunk 経路は `_validate_chunks` と `_combine_metric` で、batch 平均を batch 数で割る。重みは `len(case_id)` ではない。chunk iterator は端数 batch を出さないので、batch サイズが定数のときは重み付き平均と同じ値になる。D が TF の検証集約を `_reduce_weighted_reports` と書いたのは、aligned 経路（実 DDP と virtual world）には合う。`configs/training/tf_gridnet.yaml` の 1 process はこちらではない。その YAML は VoiceBank で、B が書くとおり公式 MS-SNSD の 2 GPU・per-rank batch 2 でもない。二つの記述は、対象ファイルを混ぜなければ矛盾しない。
- 学習ログの train loss は、D-007 のとおり rank 0 の局所平均である。B が一致すると書いたのは、per-rank batch が等しいときの勾配と、validation の重み付き平均である。ログの train loss まで公式の全体平均だと読むと、D と衝突する。B の本文はその読みをしていない。

## HIGH と BLOCKER

### P10P-B-005（深刻度 高、B は BLOCKER）

判定は BLOCKER のまま。`configs/training/reproduction/p10b/tf_gridnet.yaml` と `tf_locoformer.yaml` の三つの上限は、上に書いた値で実在する。`_at_verification_limit` は `global_step` がその上限以上で `max_steps` 未満のとき真になり、chunk loop は `INTERRUPTED` で止まる。`validation_max_updates` は prefix をさらに切る。上限の無い MS-SNSD `espnet_recipe` の本学習 YAML は `configs/training` に無かった。DNS のもう一つの YAML も smoke である。iterator の研究規則を変える問題ではなく、この YAML のままでは公式予算の学習が始まらない、という BLOCKER である。

### P10P-B-003（深刻度 高、B は NON-BLOCKER）

判定は NON-BLOCKER のまま。未知であること自体は、保存 config が `cv_synthetic_small` を指し、登録済み 36,000 件は sorted uttid の末尾 1/10 である、というところまでこのレビューでも見た。それを公開 run の validation 集合と同一だと主張できない、という影響は残る。集合を変えない、という B の対応方針を BLOCKER に上げる材料は無い。

### P10P-B-001（深刻度 中、参考）

NON-BLOCKER のまま。21 batch と発話 6 件は公式の最短打ち切りで再現する。戻すと評価規則の変更になる、という B の線でよい。

### P10P-D-001（深刻度 高）

MossFormerGAN の clip 順は、引用どおりコードにある。公式 `solver.py` は micro-batch の backward のあと `clip_grad_norm_` する。現行 `train_update` は非最終 micro を `no_sync` にした局所勾配に clip し、最終 micro の allreduce 後にも clip する。`mossformergan_vb28_official_2gpu.yaml` は global 4、micro 1、2 process なので accumulation 2、`grad_clip` 10 である。clip が恒等なら差は出ない。実ネットワークの grad norm が 10 を超えるかは、D が書くとおり NOT_TESTED で、このレビューも測っていない。TF の chunk は accumulation 1 で、この差は出ない。TF の BLOCKER にはしない。MossFormer の 2 GPU を公式と数値一致したと主張することは、この差が残る限りできない。

### P10P-D-002（深刻度 高）

`_validate_training_data` は rank 0 だけが `validation_report` を呼ぶ。`eval_mode` と `validation_report` は Discriminator を `train()` のままにする。公式が SyncBatchNorm にする、という差はソース上ある。CPU では SyncBatchNorm の出力差が測れない、という D の留保を変えない。chunk の TF 検証とは別経路である。

### P10P-D-004（深刻度 高）

現行の学習は PESQ が取れないと `TrainingError` で止まる。公式の学習は `loss_gen is None` の batch を backward しない。公式の検証平均は `total_loss / (i+1)` で、失敗 batch は分子に足さず分母には残る。現行の `validation_report` は値 0、status ok で平均に残す。検証分母の扱いは公式と同じ向きで、学習の中断は違う。実音声の PESQ 失敗はこのレビューでも起こしていない。

### P10P-D-005（深刻度 高、D は SUSPECTED）

`fit` の `except` は他 rank への通知 collective を出さずに再送出する。`install_training_stop` は rank 0 だけが呼ぶ。待ちが起きる、という実行確認はしていない。D の SUSPECTED を CONFIRMED には上げない。正常 step の chunk 分配とは別の問題である。

## このレビューがしていないこと

NCCL、実 GPU、実モデルの 1 step、2 process の `iterator_stop`、DNS 36,000 件に対する公式 `ChunkIterFactory`、D の `/tmp` スクリプトの再実行、はしていない。
