# DB の配置と構築記録

評価用 DB の操作は [セットアップ](SETUP.md)、再現用 DB の仕様は [再現用 DB](REPRODUCTION_DB_DESIGN.md)、コーパスの内訳は [データセット](DATASETS.md)。ここは、どのファイルに何があるかと、日付つきの測定・構築記録だけを置く。

2026-11-12 の確認は、ファイルの一覧と `stat`、および `tests.suite_guard` の読み取り専用マウントの中で `duckdb.connect(..., read_only=True)` したクエリである。時間（時）の再計算と、コーパス音声の全件ハッシュはしていない。

## 2026-11-12 夜: DEMAND48・Babble・SSN を加えた再構築

同日の再構築後、評価用の本番は `data/db/resource.duckdb`（191,901,696 バイト、更新時刻 2026-11-12 18:27:03 +0900）。候補 `data/db/resource-next.duckdb` は同じバイト数・同じ更新時刻のまま残している。往復先 `data/db/resource-roundtrip-20261112.duckdb` は 187,183,104 バイト。再構築元の staging は `data/staging/resource-20261112/`。置換前のファイルは `data/db/backups/resource-before-demand48-20261112.duckdb`（203,698,176 バイト、SHA-256 `3998f5ae86b3d1b7a3809bdff20f2564a8e90859b0e237f2191150db9afacb6a`）。退避用 staging `data/staging/resource-before-demand48-20261112/` は新 DB の入力ではない。

索引は 17 コーパス、1,375,691 行。読めないファイルの記録 `data/staging/ingest/unreadable.jsonl` は無い。置換前の 14 コーパスの件数は下の節と同じで、追加は DEMAND48 208、Babble 1、SSN 1。DEMAND は 288 のまま。48 kHz は SCAFE の 16 ファイルだけで、ほかの 17 シーンは 16 kHz の 16 ファイル。DEMAND48 は 13 シーン、すべて 48,000 Hz、1 チャンネル、14,400,192 frames。無いシーンは DWASHING、NFIELD、NPARK、NRIVER、OHALLWAY。Babble と SSN の `subset` は null。

| コーパス | 件数 |
|---|---:|
| Babble | 1 |
| CHiME3 | 122,277 |
| DEMAND | 288 |
| DEMAND48 | 208 |
| DNS2020 | 132,150 |
| DNS5 | 124,058 |
| LibriSpeech | 292,367 |
| LibriTTS | 375,086 |
| MS-SNSD | 24,374 |
| OpenSLR26 | 120,000 |
| OpenSLR28 | 61,260 |
| SSN | 1 |
| URGENT2024 | 5,000 |
| VCTK | 44,085 |
| VoiceBank-DEMAND | 70,942 |
| WHAM48 | 1,578 |
| musan | 2,016 |
| 合計 | 1,375,691 |

評価セットは次の 5 つ。`eval.set_items` は 11,922。`eval.cases` も 11,922。`eval.runs` と `eval.results` は 0。

| 名前 | 件数 |
|---|---:|
| `chime3_et05_real_ch5` | 1,320 |
| `chime3_et05_real_ch0` | 1,320 |
| `chime3_et05_simu_ch5` | 1,320 |
| `chime3_tr05_org` | 7,138 |
| `voicebank_demand_test` | 824 |
| 合計 | 11,922 |

置換前は `eval.cases` 11,932、`eval.sets` 7、`eval.set_items` 11,932、`eval.runs` 0、`eval.results` 0。新 DB に入れていないセットは `libri_demand_snr0` 2 件と `libri_testclean_full_smoke` 8 件。この 2 セットと、置換前の runs / results は backup と `data/staging/resource-before-demand48-20261112/` に残る。

候補、往復先、置換後の本番を、`src.data.recipe.json_hash` で 6 テーブルの全列について比較した。NULL は None、JSON 列は `json.loads` した値、浮動小数は `_normalize`、行の順は `path`、`case_id`、`name`、`(set_name, position)`、`run_id`、`record_id`。3 ファイルとも次のハッシュで一致した。`eval.runs` と `eval.results` はどちらも 0 行なので、空リストのハッシュになっている。

| テーブル | 論理ハッシュ |
|---|---|
| `corpora.audio_files` | `40a31d298d7c31b27b032c09c10306d884451cfcd17935a713b807f266847708` |
| `eval.cases` | `0956712fedd9794efdee1316e145e779cdac5999abb976ce7dcc92d232fc50fa` |
| `eval.sets` | `2ef81dd1d6f6bdc8e2fb1ba016622255546bfd35927d1ebd3be68641562468f7` |
| `eval.set_items` | `91d70102a50d65d45b0446aa631d01d20a7ce07e493ae7d829dd6e34f63d4584` |
| `eval.runs` | `4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945` |
| `eval.results` | `4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945` |

実体は `/net/spring/work/seki/database/`。リンクは `data/corpora/DEMAND` → `DEMAND`、`DEMAND48` → `DEMAND48`、`Babble` → `Babble`、`SSN` → `SSN`。ファイルごとの SHA-256 は `data/results/demand48-handoff-20261112/demand48-raw.json`（DEMAND48 の WAV 208、`ch01_mismatches` は空）。代表値は次のとおり。

| パス | SHA-256 |
|---|---|
| `Babble/babble.wav` | `8c0115bb4cbbc604994b951c11142b6f203b0ed132736a4279652d619f6ad68e` |
| `SSN/ssn.wav` | `edda17d2ce54dfe73b6f8dfc864d0994050d9071ac04181cfc95f87ddabe0b53` |
| `_provenance/babble_ssn.zip` | `d4a07ca68605bdc172d9d5c05327d3a21f23cdd8e52f64891bd0e83dea72d64e` |

Babble は 48,000 Hz、2,880,000 frames、1 ch、5,760,044 バイト。SSN は 48,000 Hz、1,440,000 frames、1 ch、2,880,044 バイト。DEMAND48 の ZIP は `_provenance/demand48/` にあり、コーパスの外なので索引されない。

| ZIP | SHA-256 |
|---|---|
| `DKITCHEN_48k.zip` | `4197f1a02dbb36c4c655a11d14cc4c5dca50eae52a4abaa2c61bfca9de6be17f` |
| `DLIVING_48k.zip` | `7e2817fffa8e4b3a10e4772e25d103e7242a8b5309f81def1bc21b3ebc1f0262` |
| `OMEETING_48k.zip` | `74eb791ec87bd43d31818f334a6831d599907e86c0df1154901dd8cf1079d9f9` |
| `OOFFICE_48k.zip` | `5e0e50f3ab7750c795b71eaf3b001a9fe1d6e5e15a95c6b036d3eefaa5ec6eaf` |
| `PCAFETER_48k.zip` | `ea1defbc29d564d4cdbf8b34191783ebf8debba47cdd3a82c6026608b9788942` |
| `PRESTO_48k.zip` | `5f4acec91143a6cfa6a10cfdfb0908a7748f6511f99128d6224e9f787a06218f` |
| `PSTATION_48k.zip` | `86d7412f779aac85c5eaa92b0664c93217ed6cbed17882b59318e59c831888c5` |
| `SCAFE_48k.zip` | `600f3b5b93af2b949f96e4d3fa488d678d4b14c381030a84bfcbb05d7b2d236d` |
| `SPSQUARE_48k.zip` | `b82897ae23355d28c352527da431a353c880716999c98c6a7a4244cfcf9246c7` |
| `STRAFFIC_48k.zip` | `6e4d400d76f151b1387b143d10854ce526e3f107465244f8791c811286066e98` |
| `TBUS_48k.zip` | `b46499812baf8d7e2735c0dc1c33f5e54e53ff3159ff9607032aa6fd490f9f96` |
| `TCAR_48k.zip` | `8ff68f65c8af88764a6bae44ce35747f592929753b03319e68211f9ef58c55d6` |
| `TMETRO_48k.zip` | `2d9be4da51098491be1ed67474dabfa7f5ed254cae921cca1dd06585bddfe883` |

SNR シフトレシピの正本は `/net/spring/work/seki/database/recipes/voicebank-snr-shift-v1/`。実行時の場所と SHA-256 は `configs/data/voicebank_snr_shift.yaml`。`recipes.jsonl` は `836ca5f1566404b016087332a764467d36ed0c237f6bcee2736e6003b6f0bd23`、`common.json` は `9a290d8b149293fc59663822425efe20414ff7d39d7cb08b773b873a806493b0`。発話は 12,396。このレシピは `eval.cases` に入れていない。

## 2026-11-12 に在ったファイル

`data/db/` の DuckDB と、そのバイト数、更新時刻（+0900）:

| パス | バイト数 | 更新時刻 | 中身の要約 |
|---|---:|---|---|
| `data/db/resource.duckdb` | 203,698,176 | 2026-11-05 00:24:27 | `corpora` と `eval`。索引 1,375,481 行、評価 7 セット |
| `data/db/reproduction.duckdb` | 460,337,152 | 2026-11-08 12:21:06 | `repro`。Dataset は 2 件 |
| `data/db/reproduction-vb-ondemand.duckdb` | 34,615,296 | 2026-11-08 19:36:23 | VoiceBank オンデマンド 4 Dataset |
| `data/db/reproduction-p8.duckdb` | 11,808,768 | 2026-11-08 19:37:42 | DeepFilterNet3 の 3 Dataset |
| `data/db/reproduction-p9.duckdb` | 11,808,768 | 2026-11-07 16:48:30 | SEMamba++ の 3 Dataset |
| `data/db/voicebank.duckdb` | 10,760,192 | 2026-11-04 21:08:12 | `eval` のみ。28 話者の train / valid |
| `data/db/semamba-voicebank-test-20261101-v2.duckdb` | 2,895,872 | 2026-11-07 22:13 | 公式 test 824 件の別ファイル |
| `data/db/backups/resource-before-dns-mssnsd-20261104.duckdb` | 180,105,216 | 2026-11-04 16:24:14 | 再索引前。12 コーパス |

上の表の `resource.duckdb` は、同日夜の再構築より前のファイルである。再構築後のバイト数、件数、ハッシュは「DEMAND48・Babble・SSN を加えた再構築」にある。

`data/db/.reproduction.lock` は 0 バイトのファイルとして在った。ロックを掴んでいるプロセスまでは見ていない。`data/db/.gitkeep` も在る。

`data/db/snapshots/` は 11 ディレクトリ。中身は後の表。`data/staging/` は `reproduction`、`reproduction-p8`、`reproduction-p9`、`reproduction-vb-ondemand`、`voicebank-current`、`semamba-voicebank-test-20261101-v2`。

`data/corpora/` は 14 本のシンボリックリンクで、実体は `/net/spring/work/seki/database/` 配下。その置き場には、カタログにもリンクにも無い `Arni-RIR`、`VCTK-0.92`、`VCTK-RVA`、`FSD50K`、`SpeechCommands`、`LibriLM`、`VoxCelebMeta`、`cmudict`、`_ingest` という名前のディレクトリも在った。確認したのは名前だけで、中身と、SEMamba++ の公式素材（VCTK 0.92、Arni RIR）として使えるかは見ていない。`data/results/` と `data/checkpoints/` はディレクトリとして在った。中身はここには書かない。

## 2026-11-12 に無かったパス

次は、これまでの文書がパスとして挙げているが、この作業ツリーには無かった。削除したのか、別の場所にあるのかは未確認。

- `data/db/backups/resource-legacy-20261031.duckdb` と `resource-legacy-20261031.sha256`
- `data/db/rebuild-a.duckdb`、`data/db/rebuild-b.duckdb`
- `data/db/training.duckdb`、`data/staging/training/`
- `data/db/resource-partial-20261102.duckdb`
- `data/staging/resource-current`、`data/staging/resource-full`、`data/staging/voicebank`、`data/staging/` 直下の `metadata.json` と parquet
- `data/staging/reproduction-p71`
- `data/db/snapshots/f75b3e7a4dbab8a68a6594487b3d605e949d96ef85fb8667b34417ea12101a39`
- `data/derived/`、`data/generated/`
- `data/staging/ingest/unreadable.jsonl`

`data/generated/` は、保存済みの事前生成 WAV の置き場として記録されている。学習の必須データではない。残す対象として書かれていたのは、`tf_gridnet.ms_snsd.smoke@1` が参照する `ms_snsd_synth/<build_id の先頭 16 桁>/` の PCM_16 だけである。公式規模の途中 WAV は登録前に削除した、とある。オンデマンド Dataset はここへ混合 WAV を書かない。2026-11-12 にはディレクトリ自体が無かった。

`data/derived/` は派生素材の置き場として記録されている。VoiceBank-DEMAND の 16 kHz PCM_16 のパスは `data/derived/VoiceBank-DEMAND/soxr_hq_16000/`。2026-11-12 にはこのディレクトリは無かった。同じ変換条件の正常なファイルは再利用し、内容が違うファイルは上書きしない、と `.agents/DATA.md` に書いてある。現行の `prepare/run.py` はこのパスを、比較のときファイルがあれば読むだけで、書く処理はこの確認では見つかっていない。上書き規則を実装している関数も、この確認では特定していない。

`e4495fa6ac0048c6346cc65bc66fd71114fd1523f95230537e799610c8092024` はディレクトリ名ではない。研究用 Dataset のスナップショットハッシュで、ディレクトリは dataset id `e4a53415122aba24055f4ba7eef6367a7e41d062485d74de3d7ce94c07d7bdd0` である。

## 評価用 `resource.duckdb`

この節の件数は、2026-11-12 夜の再構築より前のファイルである。

スキーマは `corpora` と `eval`。`eval.cases` の列は `case_id`、`kind`、`dataset`、`split`、`sample_rate`、`transcript`、`speaker_id`、`recipe_json`、`content_hash`、`metadata`。`eval.runs` と `eval.results` は 0 行。`eval.cases` は 11,932 行。

コーパス件数:

| コーパス | 件数 |
|---|---:|
| CHiME3 | 122,277 |
| DEMAND | 288 |
| DNS2020 | 132,150 |
| DNS5 | 124,058 |
| LibriSpeech | 292,367 |
| LibriTTS | 375,086 |
| MS-SNSD | 24,374 |
| OpenSLR26 | 120,000 |
| OpenSLR28 | 61,260 |
| URGENT2024 | 5,000 |
| VCTK | 44,085 |
| VoiceBank-DEMAND | 70,942 |
| WHAM48 | 1,578 |
| musan | 2,016 |
| 合計 | 1,375,481 |

MS-SNSD の `subset` は `clean_train` 23,075、`clean_test` 1,100、`noise_train` 128、`noise_test` 51、`hitapps` 20。VoiceBank-DEMAND はすべて 48 kHz・1 チャンネルで、`clean_trainset_28spk_wav` と `noisy_trainset_28spk_wav` が各 11,572、`clean_testset_wav` と `noisy_testset_wav` が各 824、`clean_trainset_56spk_wav` と `noisy_trainset_56spk_wav` が各 23,075。16 kHz の行は、この索引には無かった。

評価セット:

| 名前 | 件数 |
|---|---:|
| `libri_demand_snr0` | 2 |
| `libri_testclean_full_smoke` | 8 |
| `chime3_et05_real_ch5` | 1,320 |
| `chime3_et05_real_ch0` | 1,320 |
| `chime3_et05_simu_ch5` | 1,320 |
| `chime3_tr05_org` | 7,138 |
| `voicebank_demand_test` | 824 |
| 合計 | 11,932 |

再構築元の staging は、記録上 `data/staging/resource-current` である（`data/staging/` 直下、`resource-full`、`voicebank`、`training` は過去の再構築元）。2026-11-12 のこの作業ツリーには `resource-current` が無い。したがって、この作業ツリーには `resource.duckdb` を staging から作り直す元が無い。別ホストに在るかは未確認。

## バックアップ

`resource-before-dns-mssnsd-20261104.duckdb` の SHA-256 は `3d43bd61135c5f4de73daa7bec7e488912bf46f64f4a2b1b6afa926f30c05c45`。2026-11-12 に `sha256sum` で同じ値だった。同じディレクトリに `.sha256` ファイルは無い。中身は `corpora` と `eval` で、コーパスは上の 14 から DNS2020 と MS-SNSD を除いた 12。評価セット 7 つと件数は、上の `resource.duckdb` と同じだった。

切り替え前の固定列ファイルとして記録されているのは `data/db/backups/resource-legacy-20261031.duckdb`、SHA-256 `f3fdcd89b34cf44eda9853e71278806c1ecc4042f03173821805be054b61d156`。戻し方は [セットアップ](SETUP.md)。2026-11-12 にはこのファイルも、同じディレクトリの `resource-legacy-20261031.sha256` も無かった。ハッシュは再計算していない。

`rebuild-a.duckdb` と `rebuild-b.duckdb` は、切り替え前に本番と同じ内容で作った再構築コピー、と記録されている。2026-11-12 にはどちらも無かった。内容の一致は再確認していない。

## 学習セット用の別ファイル

`voicebank.duckdb` は `eval` だけを持つ。`voicebank_28_train` が 10,700、`voicebank_28_valid` が 872、`eval.cases` は 11,572。再構築元として在ったのは `data/staging/voicebank-current/`（`metadata.json` と `evaluation/` の parquet）。`corpora/audio_files.parquet` はこの staging には無かった。

`training.duckdb` と `data/staging/training/` は 2026-11-12 に無かった。登録コードが書く先は、いまもその二つのパスである（`src/data/training_sets.py`）。セットの中身は [学習スモークセット](TRAINING_SETS.md)。

`semamba-voicebank-test-20261101-v2.duckdb` は、`tests/numerics/reference/prepare_voicebank_test.py` が別ファイルへ書く公式 test である。セット `voicebank_demand_official_test` が 824。ケースは `paired`、`VoiceBank-DEMAND`、`test`、列の `sample_rate` は 16,000。レシピは操作が空で `output.sample_rate` が 16000。索引上の VoiceBank-DEMAND は 48 kHz なので、ファイル自体のレートとこの列が一致するかは未確認。staging は `data/staging/semamba-voicebank-test-20261101-v2/evaluation/`。

## 再現用 DB

4 ファイルともスキーマは `repro` だけ。`schema_version` は `1`、`ready` は `1`、`column_fingerprint` はどれも `ed292bcefae95a0596b15c633e0b2fca515bc6289429cc045296ef00808c418c`。`repro.runs` はどれも 0 行。

### `reproduction.duckdb`

`schema_info` には上の 3 キーに加え、次がある。

| key | value |
|---|---|
| `database_origin` | `reconstructed_from_verified_snapshot` |
| `original_database_identity` | `unverified` |
| `reconstructed_parent_dataset_id` | `5469f0416329a8661df2cc91cd5bd9a149d6ea223c72e69ebdd5696bea6bdfbe` |
| `reconstructed_parent_snapshot_hash` | `3343a55efb4c57d9ffef9179fb2f39c3e0186d36f587a0f4ad9f0aa14e9926f5` |

行数は assets 130,950、cases 360,150、datasets 2、protocols 4、builds 1、audits 7。

| name | claim | status | profile | selection_uses_test | dataset_id |
|---|---|---|---|---|---|
| `tf_locoformer.dns.accept_recipe` | `espnet_public_se_recipe_reproduction` | ready | `official_faithful` | false | `5469f0416329a8661df2cc91cd5bd9a149d6ea223c72e69ebdd5696bea6bdfbe` |
| `tf_locoformer.dns.accept_recipe.research_valid_fixed` | `research_checkpoint_selection` | ready | `official_faithful` | false | `e4a53415122aba24055f4ba7eef6367a7e41d062485d74de3d7ce94c07d7bdd0` |

分割の件数は、accept が train 324,000 / valid 36,000 / test 150、研究用が train 324,000 / test 150 / valid 1,024。ケースの protocol は、train と accept の valid 36,000 と研究用 valid 1,024 が `dns.tflocoformer.ondemand.accept`、両方の test 150 が `dns.tflocoformer.ondemand`。Dataset 行が指す protocol は、accept が `dns.tflocoformer.ondemand.accept`（`algorithm_change`）、研究用が `tf_locoformer.dns.accept_recipe.research_valid_fixed.protocol`（`research_improvement`）。ほかに `dns.tflocoformer.formula`（`official_effective`、親なし）と `dns.tflocoformer.ondemand`（`algorithm_change`）がある。

`material_origin` はどちらも `derived_from_official`。

staging の `data/staging/reproduction/INDEX.json` は、この 2 Dataset と build `0c34775cb2067416b719600bc626e9762b4b59dea04f5816b527942f89076e91`、catalog `d982440c72ce44c999e1ab4c056114df49f5da1203810118d1d00207c41f38e4` を含む。`builds/.../build.json` の generator は `dns2020_synth`、`generator_version` は 1、`seed` は 17、`shard_size` は 1000。`params_json` は `num_files` 360,000、`clean_count` 65,348、`noise_count` 65,302、`excluded_non_16k_noise` 0、`renderer` `dns_pcm16_accept_v2`、`total_hours` 3000.0、`audio_seconds` 30.0、SNR の下限 -5 上限 15、目標レベルの下限 -35 上限 -15。`research_valid_fixed_v1/registration_result.json` の `published_list_sha256` は `73e8d44a3c2196af874f8cddb7a1cc9a6b1745a220d6f1b88fda7a9a502f7ad4`。同じ JSON の `sha256_sample` は 130,950 件中 32 件を見て不一致 0 で、全ファイルの照合ではない。親グラフの登録件数は assets 130,950、audits 4、datasets 1 で、研究用の追加が audits 3 と datasets 1。合計は上の DB 行数と合う。

### `reproduction-vb-ondemand.duckdb`

assets 24,792、cases 12,396、datasets 4、protocols 4、audits 12。ケース 12,396 件の protocol はすべて `mp_senet.fixed_pair.ondemand`。Dataset 行の protocol はモデルごとで、MP-SENet が `mp_senet.fixed_pair.ondemand`、MossFormerGAN が `mossformergan_se_16k.fixed_pair.ondemand`。親 protocol `mp_senet.fixed_pair` と `mossformergan_se_16k.fixed_pair` もある。4 つとも `deviation_class` は `official_effective`。

| name | claim | selection_uses_test | train | valid | test | dataset_id |
|---|---|---|---:|---:|---:|---|
| `mp_senet.vb28.ondemand.official` | `official_reproduction` | true | 11,572 | 824 | 824 | `36ab4590b113d0aa00884c44041eaa97d65085cee57bcc9c9e06135741fc76d7` |
| `mp_senet.vb28.ondemand.safe` | `leakage_safe_comparison` | false | 10,700 | 872 | 824 | `54ee0fc9b6b4675c685991ecc9f090e51a3dc8fba0da3cd21231f542c88232da` |
| `mossformergan.vb28.ondemand.official` | `official_reproduction` | true | 11,572 | 824 | 824 | `153096c312f8027bba93fa635224bff8681170841bbe40164d189f923d16aebe` |
| `mossformergan.vb28.ondemand.safe` | `leakage_safe_comparison` | false | 10,700 | 872 | 824 | `5748381b83c289f58485062ab62f099f48899ec98456d85a8e487bf09cb0d8b6` |

4 つとも ready、`material_origin` は `derived_from_official`。保存されている `leak_check` は、official 2 件が `PASS`、safe 2 件が `UNKNOWN`。`resource.duckdb` を渡したときの FAIL は、この日には再実行していない。

再構築元は `data/staging/reproduction-vb-ondemand/`。`reproduction.duckdb` にはこの 4 件は入っていない。`make reproduction-prepare` の既定の DB は `data/db/reproduction.duckdb`、staging は `data/staging/reproduction` である（`src/experiments/data/paths.py`）。この vb-ondemand のパスを既定にする Make ターゲットは無い。どの引数でこのファイルへ書いたかは未確認。

### `reproduction-p8.duckdb`

assets 412、cases 88、datasets 3、protocols 2、audits 9。protocol は `dfn3.online`（`official_effective`、親なし）と `dfn3.online.audio-db`（`algorithm_change`、親あり）。

| name | claim | status | dataset_id |
|---|---|---|---|
| `dfn3.online.official` | `official_reproduction` | blocked | `be8e03d6f583c14c5667e6a86a09f6d3cae26b510c5a1aa872bed106ee8f8f60` |
| `dfn3.online.substitute` | `protocol_conformant_substitute_materials` | ready | `ee6223405c19bef4e8f39c9609b830273c422d7ed32b17fbaad0b0554ce1328b` |
| `dfn3.online.smoke` | `protocol_conformant_substitute_materials` | ready | `283030adc4a81a90d833338fbd2d9e8677ec387b0cd2b7f7d222efb3aa0340e8` |

official にバインドは無い。ready な 2 件の `material_origin` は `substitute`、`selection_uses_test` は false。登録は `make reproduction-register-p8`（`register_p8()`）。既定の DB と staging は `data/db/reproduction-p8.duckdb` と `data/staging/reproduction-p8/`。`reproduction.duckdb` と `resource.duckdb` には入れない。

### `reproduction-p9.duckdb`

assets 60、cases 12、datasets 3、protocols 2、audits 9。protocol は `semambapp.gsr`（`official_effective`、親なし）と `semambapp.gsr.audio-db`（`algorithm_change`、親あり）。`minimal_launch_repair` の protocol 行は、このファイルには無かった。

| name | claim | status | dataset_id |
|---|---|---|---|
| `semambapp.gsr.official` | `official_reproduction` | blocked | `63a0e6785851996743c1ed040ddfbabb811c00098f821d09e4c57ea36a1f995f` |
| `semambapp.gsr.substitute` | `protocol_conformant_substitute_materials` | ready | `4acf30da286a36cb8c6edcdf1091456c979fd839d7898ecc7528467e4b04d76e` |
| `semambapp.gsr.smoke` | `protocol_conformant_substitute_materials` | ready | `d3bc5dbea5b7a19c8495a0d2ce5d7d53267c96d6f82dc049c193fd14a04980eb` |

official にバインドは無い。ready な 2 件の `material_origin` は `substitute`、`selection_uses_test` は false。現行の登録は `make reproduction-register-p9`（`register_p9()`、`reproduction-register-p8` と同じ形）。既定の DB、staging、スナップショット先は `data/db/reproduction-p9.duckdb`、`data/staging/reproduction-p9/`、`data/db/snapshots/`。`reproduction.duckdb` と `resource.duckdb` には入れない。

以前は `tests/data/reproduction/test_p9_semamba.py` が、この本番パスとスナップショット `semambapp.gsr.smoke@1` / `semambapp.gsr.substitute@1` を副作用として書いていた。いまのテストは一時ディレクトリを使う。上のファイルが、その変更より前の副作用で残っているのか、`make reproduction-register-p9` の結果なのかは、この確認では区別していない。

## スナップショット

ディレクトリ名は dataset id。`manifest.json` の `snapshot_hash` は次のとおり。blocked の official 2 件にスナップショットは無い。

| dataset_id | name | snapshot_hash | allow_unknown_leak_check |
|---|---|---|---|
| `13c6058881ef86fe7506f705b1bd5a5bc68b313181410d8fdaaee20c33ef3fc6` | `tf_gridnet.ms_snsd.espnet_recipe` @1 | `a69a477af70bc43dbcabdafbd5c97ec8a26aac4debd51158079545de1e4f0757` | false |
| `153096c312f8027bba93fa635224bff8681170841bbe40164d189f923d16aebe` | `mossformergan.vb28.ondemand.official` @1 | `ba0c77c94babbbc2c3e70b149554f8a922a68cf6fbe71ccf833927dc71dff555` | false |
| `36ab4590b113d0aa00884c44041eaa97d65085cee57bcc9c9e06135741fc76d7` | `mp_senet.vb28.ondemand.official` @1 | `265f256b938a89746dcaa62cd44daa2c6bcca3bce4cc1b1719eb2170717d33d5` | false |
| `54ee0fc9b6b4675c685991ecc9f090e51a3dc8fba0da3cd21231f542c88232da` | `mp_senet.vb28.ondemand.safe` @1 | `4528f73a317140cbe05e89d6d30e3a9d2058c28ad00002457551ceb2935d5c3a` | true |
| `5748381b83c289f58485062ab62f099f48899ec98456d85a8e487bf09cb0d8b6` | `mossformergan.vb28.ondemand.safe` @1 | `369bffdd754b096a16a59d5a3cac655a2f4ff851a15dd04c90800d03eccd7bbf` | true |
| `283030adc4a81a90d833338fbd2d9e8677ec387b0cd2b7f7d222efb3aa0340e8` | `dfn3.online.smoke` @1 | `dec95878ea5854a54656a7a547511cd95809bb383166af94eb402ef41bcbeca6` | false |
| `ee6223405c19bef4e8f39c9609b830273c422d7ed32b17fbaad0b0554ce1328b` | `dfn3.online.substitute` @1 | `861992f6a520823bbe8fc75faf9ffa32740881c89b65eca21bc1160a1de7401b` | false |
| `4acf30da286a36cb8c6edcdf1091456c979fd839d7898ecc7528467e4b04d76e` | `semambapp.gsr.substitute` @1 | `1b9be5fe72e9522516aff29f9ad60f21a3fe4f2dbfe3912a08c42173f3f66d37` | false |
| `d3bc5dbea5b7a19c8495a0d2ce5d7d53267c96d6f82dc049c193fd14a04980eb` | `semambapp.gsr.smoke` @1 | `dd57fd637dbc095fc77a082bd9f8a3e6c0e2a52ec00df89690f2717cdfb2802f` | false |
| `5469f0416329a8661df2cc91cd5bd9a149d6ea223c72e69ebdd5696bea6bdfbe` | `tf_locoformer.dns.accept_recipe` @1 | `3343a55efb4c57d9ffef9179fb2f39c3e0186d36f587a0f4ad9f0aa14e9926f5` | false |
| `e4a53415122aba24055f4ba7eef6367a7e41d062485d74de3d7ce94c07d7bdd0` | `tf_locoformer.dns.accept_recipe.research_valid_fixed` @1 | `e4495fa6ac0048c6346cc65bc66fd71114fd1523f95230537e799610c8092024` | false |

`tf_gridnet.ms_snsd.espnet_recipe` は、上の 4 つの DuckDB の `repro.datasets` には無かった。スナップショットだけが在る。`docs.jsonl` は 9,485 行。記録上の train 7,345 / valid 1,835 / test 305 の合計と一致する。分割ごとの件数はこのファイルでは数えていない。claim は `espnet_public_se_recipe_reproduction`、`selection_uses_test` は false。

accept の `manifest.json` の provenance には、`generator_code_sha256` `6dd072d518ed8242fa8a8247ebe71c184a3aa4daa0825bc3b5ca4f8e3417a145`、`porcelain_sha256` `ebde4f7a6e45848d946e07f13ca473ba1d712519d3dd2bd9f436e17c24c6e2a5`、`worktree_dirty` true がある。研究用スナップショットの provenance には `database_origin` `reconstructed_from_verified_snapshot`、`original_database_identity` `unverified`、親 dataset id と親 snapshot hash（上の accept と同じ値）がある。

VoiceBank の official 2 件のスナップショットは、再構築のとき既存ディレクトリとハッシュが一致したので置き換えていない、と記録されている。safe 2 件はこの再構築で `data/db/snapshots/` に追加した、と記録されている。2026-11-12 のハッシュは上の表であり、その「一致したので置き換えなかった」時点のハッシュとの比較はしていない。

## 構築記録

### 2026-11-02 TSUBAME 初回

ユーザーがデータ転送と完全性確認の完了を確認し、この回に限って staging への先行保存を省略するよう指示した。12 コーパスを候補 DB に索引してから評価セットを登録し、検証後に `data/staging/resource-current` へ書き出した。本番 `data/db/resource.duckdb` は、既存ファイルが無いことを確認して新規作成した。この例外は、その後の staging 先行の規則を変えない。研究室側 DB のコピーや、研究室側との同一性の確認ではない。

初回 CPU ジョブは途中で停止した。完成済み 4 コーパスを含む候補を `data/db/resource-partial-20261102.duckdb` に残し、別コピーで残りを処理した。索引は 1,218,957 件、読み取り不能ファイルは 0 件。

| コーパス | 件数 |
|---|---:|
| CHiME3 | 122,277 |
| DEMAND | 288 |
| DNS5 | 124,058 |
| LibriSpeech | 292,367 |
| LibriTTS | 375,086 |
| OpenSLR26 | 120,000 |
| OpenSLR28 | 61,260 |
| URGENT2024 | 5,000 |
| VCTK | 44,085 |
| VoiceBank-DEMAND | 70,942 |
| WHAM48 | 1,578 |
| musan | 2,016 |

登録済み評価セットは `libri_demand_snr0` 2 件、`libri_testclean_full_smoke` 8 件、`chime3_et05_real_ch5`・`chime3_et05_real_ch0`・`chime3_et05_simu_ch5` 各 1,320 件、`chime3_tr05_org` 7,138 件、`voicebank_demand_test` 824 件。合計 7 セット、11,932 ケース。サンプル音声が無い例示 Manifest は skip した。

DB と staging のコーパス全行について、件数と DuckDB の全列ハッシュの bit XOR が一致した。ケース・セットの ID と内容ハッシュ、およびセット項目を双方向の `EXCEPT ALL` で比較し、差はなかった。`libri_demand_snr0`、`voicebank_demand_test`、`chime3_et05_real_ch5` の各先頭ケースをレシピから再生し、有限波形を確認した。これは全ケースの再生確認や、研究室側 DB との完全一致を意味しない。

この件数は、そのとき取得した 12 コーパスの記録である。2026-11-12 の `resource.duckdb` は 14 コーパス、1,375,481 行で、`resource-current` と `resource-partial-20261102.duckdb` は無い。

### 2026-11-04 再索引

SHINODA-LAB の `resource.duckdb` を、14 コーパスを最初から含む索引として作り直した。退避は `data/db/backups/resource-before-dns-mssnsd-20261104.duckdb`（SHA-256 は上で 2026-11-12 に再計算した値）。再索引は 1,375,481 行、読めないファイルは 0。DNS2020 は 132,150、MS-SNSD は 24,374（内訳は上の 2026-11-12 の件数と同じ）。その後の `make db-init` は既存の評価セットを増やさず、`corpora.audio_files` の指紋は変わらなかった。

2026-11-12 の `resource.duckdb` のコーパス件数と評価セット件数は、この記録と一致した。指紋そのものは再計算していない。

### 2026-11-06 02:48 の観測

その時刻に更新された、と記録された `resource.duckdb` は 189,280,256 バイトで、`read_only=True` で開くとテーブルは `corpora.audio_files` だけだった。`eval` スキーマは無かった。誰がいつ更新したかは、その記録の時点で未確認で、その作業はファイルに触れていない。

同じ観測の件数と時間。時間は 2026-11-12 には再計算していない。件数のうち、コーパス合計、VoiceBank の subset、MS-SNSD の subset は 2026-11-12 の上の表と一致した。DNS2020 の「188 件は 44.1 / 48 / 96 kHz」、DNS5 のチャンネル内訳、VCTK の話者数、WHAM48、OpenSLR の内訳は、2026-11-12 には見ていない。

| コーパス / subset | レート | ch | 件数 | 時間 |
|---|---|---|---:|---|
| VoiceBank-DEMAND `clean_trainset_28spk_wav` / `noisy_trainset_28spk_wav` | 48 kHz | 1 | 各 11,572 | 各 9.39 h |
| VoiceBank-DEMAND `clean_testset_wav` / `noisy_testset_wav` | 48 kHz | 1 | 各 824 | 各 0.58 h |
| VoiceBank-DEMAND `*_trainset_56spk_wav` | 48 kHz | 1 | 各 23,075 | 各 19.03 h |
| MS-SNSD `clean_train` / `noise_train` / `clean_test` / `noise_test`（ほかに `hitapps` 20） | 16 kHz | 1 | 23,075 / 128 / 1,100 / 51 | 19.03 / 3.29 / 3.64 / 0.66 h |
| DNS2020 `datasets` | 16 kHz（188 件は 44.1 / 48 / 96 kHz） | 1 | 132,150 | 747.5 h |
| DNS5 `datasets_fullband` | 48 kHz | 1 / 2 / 16 / 30 | 124,058 | 193.9 h |
| VCTK `wav48` | 48 kHz | 1 | 44,085（話者 108） | 43.9 h |
| WHAM48 `high_res_wham` | 48 kHz | 2 | 1,578 | 78.5 h |
| OpenSLR26 `sim_rir_16k` / OpenSLR28 `RIRS_NOISES` | 16 kHz（OpenSLR26 には `sim_rir_8k` 60,000 件が 8 kHz で別にある） | 1（RIRS_NOISES の 379 件は 2〜30 ch） | 60,000 / 61,260 | 19.44 / 25.5 h |

同じ観測は、VoiceBank-DEMAND の索引が 48 kHz のみであること、`data/staging/` が空であること、`data/db/` に `resource.duckdb` 以外が無いことを書いている。2026-11-12 の索引も VoiceBank は 48 kHz のみだった。staging と `data/db/` の他ファイルは、上のとおり在る。ファイルサイズも、2026-11-12 の 203,698,176 バイトとは違う。更新時刻は 2026-11-05 00:24:27 で、観測時刻より前である。どのコピーが入れ替わったかは未確認。

同じ日の見積りとして、TF-Locoformer の公式規模を WAV にすると 30 秒 × 360,000 本 × 3 ストリーム、1 ストリームは 3000 h × 16,000 Hz × 2 バイト = 345.6 GB、3 ストリームで約 1.04 TB、とある。ファイルシステム `/gs/bs` の空きは、その日 17 TB（54 TB 中 38 TB 使用、共有）。2026-11-12 の空きは見ていない。現行の登録は混合 WAV を書かない。

### 2026-11-08 の再構成

この作業木の `reproduction.duckdb` は、元ファイルの復旧ではない。親 snapshot `3343a55efb4c57d9ffef9179fb2f39c3e0186d36f587a0f4ad9f0aa14e9926f5` から再構成したデータベースで、`schema_info.database_origin` は `reconstructed_from_verified_snapshot`、元ファイルとの同一性は未検証である。載っているのは `tf_locoformer.dns.accept_recipe@1` と `tf_locoformer.dns.accept_recipe.research_valid_fixed@1` だけ、と記録されている。VoiceBank、MS-SNSD、P7 の recipe / smoke、学習 run は入っていない、とある。研究用 Dataset の id は `e4a53415122aba24055f4ba7eef6367a7e41d062485d74de3d7ce94c07d7bdd0`、snapshot は `e4495fa6ac0048c6346cc65bc66fd71114fd1523f95230537e799610c8092024`。train 324,000 と test 150 は親と同じ case set、valid は 1,024 件である。

2026-11-12 に開いた `reproduction.duckdb` は、この記述と一致した。

これより前の記述は、同じ `reproduction.duckdb` に次が入っている、としていた。2026-11-08 の再構成のあと、この一覧は `reproduction.duckdb` の中身ではなくなった。2026-11-12 にファイルとして残っているかを併記する。

| 記録上の Dataset | 記録上の内容 | 2026-11-12 |
|---|---|---|
| VoiceBank 履歴 4（`mp_senet.vb28.official@1`、`mp_senet.vb28.safe@1`、`mossformergan.vb28.official@1`、`mossformergan.vb28.safe@1`） | 48 kHz の 28 話者 train と公式 test から soxr HQ で作った 16 kHz PCM_16。同じ 16 kHz ペアの `case_id` はモデル間で共有し、`cases.protocol_id` は `mp_senet.fixed_pair` | 4 つの DuckDB にもスナップショット一覧にも無い。`data/derived/VoiceBank-DEMAND/soxr_hq_16000/` も無い |
| オンデマンド 4 | 48 kHz から学習時に 16 kHz。件数は上の vb 表 | `reproduction-vb-ondemand.duckdb` とスナップショットにある。`reproduction.duckdb` には無い |
| `tf_gridnet.ms_snsd.smoke@1` | 保存済み WAV の履歴。`claim_label` は `official_recipe_scaled_down`。検証は生成した train の末尾であり test ではない | DuckDB にもスナップショット一覧にも無い |
| `tf_gridnet.ms_snsd.ondemand@1` | 小規模。条件から再構築 | 同上 |
| `tf_gridnet.ms_snsd.espnet_recipe@1` | 公式規模。train 7,345 / valid 1,835 / test 305 | スナップショット `13c60588…` のみ。`docs.jsonl` は 9,485 行 |
| `tf_locoformer.dns.recipe@1` | 固定条件 360,000。train 324,000 / valid 36,000 / test 150。混合 WAV は無い | この名前の Dataset は 2026-11-12 のファイルには無い |
| `tf_locoformer.dns.smoke@1` | 同じ素材プールの 8 件。train 7 / valid 1。同じ test 150 | 同上 |
| `tf_locoformer.dns.accept_recipe@1` | P7.1 の受理規則で確定した 360,000。train 324,000 / valid 36,000 / test 150。非 16 kHz 雑音 188 件を含む全雑音。混合 WAV は無い。`cv_synthetic_small` ではない | `reproduction.duckdb` とスナップショット `5469f041…` にある。188 件は 2026-11-12 には数え直していない。build.json の `excluded_non_16k_noise` は 0 |
| `tf_locoformer.dns.accept_smoke@1` | train 230 / valid 26 / test 150。本番 DB には無く、スナップショット `f75b3e7a4dbab8a68a6594487b3d605e949d96ef85fb8667b34417ea12101a39` と `data/staging/reproduction-p71` だけ | そのスナップショットも staging も 2026-11-12 には無い |

P7 の recipe と smoke は長さ計画のまま、と記録されている。オンデマンド描画は各ファイル全体を −25 dBFS にしてから区間を切る。既存の case 文書と snapshot は区間だけを持ち、この正規化では ID を変えていない、とある。どの snapshot を指すかは、この文だけでは特定できない。

`research_checkpoint_selection` は研究用検証集合の claim で、公式再現の claim とは別である。DNS の受理規則による 360,000 件は未登録、という文は、上の再構成より前の記述である。2026-11-12 の `reproduction.duckdb` には accept の 360,000 件が入っている。

### VoiceBank オンデマンドの leak と P8

再構築元は `data/staging/reproduction-vb-ondemand/` と `data/db/reproduction-vb-ondemand.duckdb`。`reproduction.duckdb` には入れていない。現行の `resource.duckdb` を渡すと、safe の `leak_check` は `eval.cases` と test 素材の重複で FAIL する。resource DB を開かない再構築の `leak_check` は UNKNOWN であり、PASS ではない。2026-11-12 に保存されていた監査は、上のとおり official が PASS、safe が UNKNOWN。FAIL の再実行はしていない。

DeepFilterNet3 のオンライン混合は `reproduction-p8` に置く。ready な代替は `dfn3.online.substitute@1` と `dfn3.online.smoke@1`。公式素材の `dfn3.online.official@1` は `blocked` で、バインドもスナップショットも無い。分布の正本は `DFN3_ONLINE_V1`。混合 WAV は作らない。`official_reproduction` は ready なら全素材 SHA-256 が要る。blocked または unknown でバインドが空なら、欠けているものを `blocking_json` に書いて登録できる。

SEMamba++ のオンライン劣化は `reproduction-p9` に置く。ready な代替は `semambapp.gsr.substitute@1` と `semambapp.gsr.smoke@1`。公式素材の `semambapp.gsr.official@1` は `blocked`。分布の正本は `SEMAMBA_GSR_V1`（HEAD `config.yaml`、16 kHz、窓 48,000）。劣化 WAV は作らない。代替素材を公式データセットの再現と呼ばない。`kaiser_best` と `kaiser_fast` は `resampy==0.4.3` を使う。この pin は `requirements/se.in` にあり、`.venvs/se` に入っている、と記録されている。2026-11-12 にその venv のインストール状態は見ていない。`.venvs/asr` のパスは足さない。

## 未確認

- 2026-11-06 の表の「時間」、DNS2020 の非 16 kHz 188 件、DNS5 のチャンネル内訳、VCTK の話者 108、WHAM48 と OpenSLR の内訳
- 固定列バックアップの SHA-256（ファイルが無い）
- `rebuild-a.duckdb` / `rebuild-b.duckdb` が本番と同じ内容だったこと
- safe の `leak_check` を、いまの `resource.duckdb` を渡して再実行した結果
- `semamba-voicebank-test-20261101-v2` の波形が 16 kHz か 48 kHz か
- `resampy==0.4.3` が `.venvs/se` に入っていること
- 無くなっていたパスが、削除されたのか別ホストに残っているのか
- `tf_gridnet.ms_snsd.espnet_recipe` の train / valid / test の内訳（行数の合計だけ一致）
- `reproduction-vb-ondemand.duckdb` を、どの引数で既定パス以外へ書いたか
