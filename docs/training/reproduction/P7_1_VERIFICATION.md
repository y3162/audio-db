# P7.1 検証（2026-11-06）

P7 のオンデマンド方式を残し、公式 DNS の棄却規則と、16 kHz でない雑音の描画時リサンプルを別 Protocol として足した。P7 の `tf_locoformer.dns.recipe@1` と `tf_locoformer.dns.smoke@1` の意味は変えていない。全量 360,000 件の再登録はしていない。`cv_synthetic_small` は UNKNOWN のままである。

公式のピンは次のとおり。

| 対象 | commit |
|---|---|
| DNS-Challenge `interspeech2020/master` | `70f19285c36cca4df2338f9248775ddc50980c6b` |
| ESPnet | `90eed8e53498e7af682bc6ff39d9067ae440d6a4` |
| TF-Locoformer | `7a615460d347ff7334a13dbb831d16280da72cdc` |

## 総合

判定は CONDITIONAL GO。受理、非 16 kHz のリサンプル、同じ Case の再構築、chunk 計画、小さい CPU Trainer の再開、`ave_1best` のリンク、研究 YAML と 2 step スモークの分離は通った。公式の検証部分集合 `cv_synthetic_small` は特定していない。公式 librosa の版は不明で、`kaiser_best` との波形差は測っていない。3000 h の受理 Dataset は未登録である。P8 を、公式の検証集合を再現した前提で始めない。

## 公式の採否

出典はローカルの DNS ツリー（上の commit）と、ピンした TF-Locoformer の README、ピンした ESPnet の `average_nbest_models.py`。`dns_data_prep.sh` と `dns_create_mixture.sh` は、この作業の sparse checkout（`espnet2` のみ）には無い。

| 処理 | 公式 | ファイル |
|---|---|---|
| 元素材の clip | `any(abs(audio) > 0.99)`。等号は棄却しない | `data/corpora/DNS2020/audiolib.py` `is_clipped`（17–18 行） |
| 切り出し中の clip | その片を捨て、`tries_left` を 1 減らし、既に連結した接頭辞は残す。`MAXTRIES = 50` | `noisyspeech_synthesizer_multiprocessing.py` `build_audio`（67–104 行） |
| clean の activity | 閾値が 0 なら測らない。否则 `percactive > threshold`。DNS の cfg は 0.6 | 同ファイル `gen_audio`（122–137 行）。`noisyspeech_synthesizer.cfg` 45 行 |
| noise の activity | cfg は 0.0 なので測らない。clip の検査は残る | 同 cfg 46 行。`gen_audio` 130–131 行 |
| activity の中身 | −25 dBFS、50 ms、シグモイド `a=-1, b=0.2`、`alpha_att=0.8`、`alpha_rel=0.05`、平滑後 0.13 | `audiolib.py` `activitydetector`（224–258 行） |
| 混合後の clip | clean、noise、noisy のいずれかが clip なら両方を作り直す。外側は `while True` | `main_gen`（153–187 行） |
| SNR とレベル | `randint(snr_lower, snr_upper)` と、`snr_mixer` 内の `randint(target_level_lower, target_level_upper)`。上端は排他 | `main_gen` 172 行。`audiolib.py` `snr_mixer` 137 行 |
| TF-Locoformer の境界 | README は `total_hours=3000`、`snr_lower=-5`、`snr_upper=15` だけを書き換える。activity とレベルは DNS cfg の 0.6 / 0.0 と −35 / −15 | `src/se/official/tf-locoformer/README.md` 209–218 行 |
| リサンプル | ファイル全体を切る前に `librosa.resample(input_audio, fs_input, fs_output)` | `build_audio` 75–77 行 |
| 共有カウンタ | `multiprocessing.Value` をロックして 1 足し、`value % n`。最初の添字は 1 | `build_audio` 71–73 行 |
| 5 件未満の平均 | `n` を 1 に強制し、`{epoch}epoch.pth` への `ave_1best` リンクと、そこへの `ave` リンクを書く | `src/se/official/espnet/espnet2/main_funcs/average_nbest_models.py` 54–68 行、118–123 行 |
| `cv_synthetic_small` | 保存 config が `dump/raw/cv_synthetic_small/{wav,spk1}.scp` を指す。README は検証の縮小を勧めるが、件数も seed もスクリプトも書かない | `.../exp/enh_train_enh_tflocoformer_raw/config.yaml` 111–116 行。README 209–218 行 |

`cv_synthetic`（36,000 件）は、以前の監査が引用した ESPnet `dns_data_prep.sh` の sorted uttid 末尾 1/10 である。この checkout ではそのスクリプトを再読できない。36,000 件も、今回の 26 件も、`cv_synthetic_small` と呼ばない。

## 実装

新しい規則キーは `retry_policy` と `renderer`。無い文書は P7 の `length_plan_no_sample_retry` / `dns_pcm16_v1` として読む。受理は `deterministic_case_stream_v1` と `dns_pcm16_accept_v2` の対だけを許す。アルゴリズムのレジストリ版は 1 のままである。

受理の計画は案件ごとに `Generator(PCG64(mix_seed("dns", seed, gen_index)))` を一つ作る。読み出しの開始は `gen_index % 件数` なので、16 kHz で最初の候補が通るとき、切り出し・SNR・レベルは P7 の `plan_case` と一致する（合成データで確認）。clip した片は公式と同じく捨て、乱数とカーソルは進む。clean の activity が 0.6 以下、または混合後の三者のいずれかが clip なら、その候補全体を捨てて同じストリームで引き直す。noise の閾値 0 では activity を測らない。外側は 50 回で `ReproductionError` にし、最後の不適合候補は保存しない。公式の外側ループは無制限なので、この上限は `algorithm_change` である。

描画は保存した区間を読むだけである。再探索も波形キャッシュも無い。`source_rate` が 16 kHz で無い片だけ、ファイル全体を `librosa.resample(..., orig_sr=, target_sr=16000, res_type="soxr_hq")` してから切る。16 kHz は従来の `soundfile` 区間読みである。この環境の librosa 1.0.0 は公式の位置引数 `resample(y, fs_in, fs_out)` を受け付けない。公式が使った librosa の版は `requirements.txt` に無く UNKNOWN。

checkpoint の `identity()` は、規則に `renderer` か `retry_policy` があるときだけそれを入れる。P7 のスナップショットの identity は増えない。文書の方針が Dataset の規則と違うときは拒否する。

`valid.loss` が 5 件未満のときは平均ファイルを書かない。`valid.loss.ave_1best.pt` は損失最小（同値なら早い epoch）の `{epoch}epoch.pt` へのシンボリックリンク、`valid.loss.ave.pt` は `ave_1best` へのリンクである。5 件以上では従来どおり `valid.loss.ave_5best.pt` を書き、`ave.pt` をそこへ向ける。`ave_1best` は消さない。拡張子は `.pt`（公式は `.pth`）。非有限の `valid.loss` は順位に入れない。公式の `sort_epochs_and_values` は NaN を除かない。整数テンソルはこちらが `is_floating_point` で平均し、公式は `str(dtype).startswith("torch.int")` のとき加算する。uint では食い違う。この差は残している。

`configs/training/tf_locoformer.yaml` は 2 step の CPU スモークに戻した。研究用の VoiceBank 設定は `configs/training/tf_locoformer_voicebank.yaml`（seed 42、TensorBoard あり）。`configs/training/tf_locoformer_dns_ondemand.yaml` は P7 の smoke Dataset のままである。

## 実測

設定は `configs/data/builds/dns_tflocoformer_accept_smoke.yaml`。clean 48、16 kHz noise 48、名前順で最初の非 16 kHz noise 4（48 kHz が 2、96 kHz が 2。いずれも 10 秒）、`num_files` 256、30 秒、seed 17。44.1 kHz はこの 4 件の名前順には入らない。本番の `reproduction.duckdb` と `data/staging/reproduction` は更新していない。登録先は `data/staging/reproduction-p71` の別 DB と、新しいスナップショットだけである。`resource.duckdb` は読み取り専用で、更新時刻は 2026-11-06 02:48 のままである。`reproduction.duckdb` の更新時刻は 17:49 のままである。

| 項目 | 値 |
|---|---|
| Dataset | `tf_locoformer.dns.accept_smoke@1` |
| dataset_id | `f75b3e7a4dbab8a68a6594487b3d605e949d96ef85fb8667b34417ea12101a39` |
| snapshot_hash | `db52950f22d0b5f9f947a341d1afdf1c2b38f30da375d3fe505e384f5816ec8a` |
| 親 protocol | `cb81f473d53ef80ee535fd7b06ec3ca9b57f9ab5e0146e8f89635d79e5e592e1`（本番 DB の `dns.tflocoformer.formula` と一致） |
| 子 protocol | `cadda632ebd08a9066e5b7a81851920762cc2a5fa5bddc21f8488bf1a1630116` |
| 分割 | train 230 / valid 26 / test 150。`selection_uses_test` は false |
| claim | `official_recipe_scaled_down` |
| WAV | 0 |
| 計画 | 41.3 秒。登録全体は約 184 秒 |
| 再測 | 256 件で 183.9 秒。最大 RSS 454 MB |
| 計画中に観測した RSS | 約 726 MB |

スナップショットの `manifest.json` が記録する `git_commit` は `7997bcb73d9c1dc4e314ad6871568a0cf24c25a2` である。これは登録時点の HEAD で、未コミットの P7.1 差分は含まない。

計画時の棄却（256 件、失敗 0、外側の最大試行 5）:

| 外側のやり直し回数 | 件数 |
|---|---|
| 0 | 123 |
| 1 | 82 |
| 2 | 22 |
| 3 | 14 |
| 4 | 9 |
| 5 | 6 |

activity の棄却イベントは 230、noise の clip スキップは 216、clean の clip スキップは 0、混合後 clip の棄却は 4、短い信号の棄却は 0。非 16 kHz の noise 片は 75。受理後の再測では、clean activity の最小は 0.6017（平均 0.658）、0.6 以下は 0、元素材 clip は 0、混合後 clip は 0、長さが 480,000 でないものは 0。SNR の絶対誤差は最大 3.36e-4 dB、平均 1.43e-5 dB。保存した SNR は −5…14、目標レベルは −35…−16 dBFS。

PCM_16 は `soundfile` の往復である。train と valid から 8 件を再構築し、`wave * 32768` と最近傍整数の差の最大は 0 だった。別プロセスで同じ Case を 2 回描画し、長さ 480,000 と先頭 4 サンプルが一致した。

librosa 1.0.0、soxr 1.1.0。10 秒の 44.1 / 48 / 96 kHz を `soxr_hq` で 16 kHz にすると、いずれも 160,000 サンプルだった。`res_type="kaiser_best"` は `resampy` が無く実行できない。公式版との波形差は未測定。

chunk 計画は train 230 件、chunk 64,000、seed 17。world 1 / 2 / 4 の plan hash は互いに違い、同じ world では再計算と一致した。更新数はいずれも 3,220。shape ファイルは無く、公式 `collect_stats` の順は UNKNOWN のままである。音声は文書から再構築するので、world size は PCM を変えない。

## Trainer

合成の受理スナップショットに、TF-Locoformer の小さい構成（`n_layers=1`、`emb_dim=16`、`n_fft=64`）を CPU で 2 step 通した。1 step で epoch が終わり validation が記録され、`data_provenance.json` に `dns_pcm16_accept_v2` が入る。同じ checkpoint を `data.virtual_world_size: 2` で再開すると `TrainingError` で拒否する。方針を合わせた再開の 2 step 目の入力波形は、中断しない実行と一致した。Medium の DNS 設定でも GPU でもない。

`tests/training/test_ave_nbest.py` は、履歴 4 件で `ave_5best.pt` が無く `ave_1best.pt` が最良 epoch を指すことと、1 件から 6 件までの遷移を確認する。

## 回帰

`tests/reproduction` は 73 件、162 秒、OK。P7 の `test_register_snapshot_rebuilds_the_same_pcm` を含む。`tests/training` は 227 件中 204 件が OK、20 件が SKIP、3 件が FAIL である。FAIL は `data/db/voicebank.duckdb` が無いこと、`configs/training/mp_senet.yaml` の `deterministic: false`、`configs/training/trial100/` が空であることで、今回の差分にはこれらのファイルが無い。`test_ave_nbest` と `test_tf_locoformer_protocol` は OK である。

## 残る差

- 案件間でカウンタを共有しない。公式の multiprocessing ダンプとのビット一致は主張しない。
- 外側の再試行は 50 回で止める。公式は無制限。
- librosa の版と `res_type` は公式が不明。こちらは librosa 1.0.0 の `soxr_hq`。
- `cv_synthetic_small` は UNKNOWN。
- 3000 h へ同じ受理規則を載せる再登録はしていない。本番 DB への結合は、全グラフの読み込みになり、P7 の登録は 36 分だった。追加の承認が要る。
- 評価重みの拡張子は `.pt`。`si_snr` の best は作らない（stats のキーは `si_snr_loss`）。これは P7 と同じである。
