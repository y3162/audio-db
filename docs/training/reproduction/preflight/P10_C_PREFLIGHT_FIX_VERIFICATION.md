# P10-C Preflight 修正の検証

日付: 2026-11-07。本格学習は開始していない。公式 epoch 数の学習は実行していない。Preflight の PASS は公式再現の成功ではない。

## 修正内容

- MP-SENet の `validation_report` は、hop 倍数へ切る前に Generator へ入れない。学習更新は、iSTFT 長が入力長と違うとき拒否する。検証だけ `allow_official_istft_length` を許す。TensorBoard のプレビューも同じ全長入力である。
- 公式損失の検証 PESQ は、例外とライブラリの負値を -1 として平均に残す。再構成損失は欠損を平均から外す。ライブラリ欠測は `unavailable` のままである。
- MossFormerGAN の初期化は、同一 seed ブロックで G1 を残し、D1 G2 D2 を捨て、G1 を device へ移してから G3 D3 G4 を捨て、D4 を残す。checksum `7729.488281` は期待値にしていない。
- 学習中の PESQ 欠測と非有限損失は、その optimizer 群を backward も step もしない。`update_applied` が 0 のとき Trainer は `global_step` を増やさない。DDP は全 rank で同じ可否にする。公式の単一 GPU が蓄積窓を後続 batch へ伸ばす動作は、固定群をまとめて捨てる差として残す。
- world size が 2 以上のとき Discriminator だけを SyncBatchNorm にする。world size 1 は通常の BatchNorm である。CUDA では `broadcast_buffers=False` で DDP する。CPU の torch 2.6 は SyncBatchNorm の DDP と CPU forward を拒否する。検証は `validation_requires_all_ranks` のとき全 rank が `validation_report` を同じ回数呼ぶ。回数が違うときは forward の前に拒否する。
- 本格 YAML を新設した。p10b は `checkpoint_interval: 2` のままである。`required_world_size` が違うときは `build_task` の前に拒否する。
- `checkpoint_retention` がある実行だけ、所有する `step_*.pt` と `Nepoch.pt` を直近 3 件、best、n-best の入力に絞る。`validation_pending` がある再開は、完了済みの Optimizer 更新をやり直さず、検証を先頭からやり直す。
- TF-Locoformer の研究用 1,024 件は git 上の定義としてある。`reproduction.duckdb` がこの作業木に無いので、行の追記と snapshot hash は発行していない。

## 公式数値比較

| 比較 | 結果 | 範囲 |
|---|---|---|
| MP-SENet、同じ重み、長さ 1600 と 1650 の magnitude | 公式経路との最大絶対誤差 0 | CPU。`test_mp_senet_validation_full_length` の出力 |
| 長さ 1650 を 1600 に切った入力と、全長 1650 | magnitude 最大絶対誤差 1.5627447366714478 | CPU。監査の 1.563 と同じ桁 |
| 公式 PESQ 平均 `[4.0, -1.0]` | 1.5。研究用集約は `na` を落とす | CPU |
| 合成正弦を 824 長とその hop 切り詰め | PESQ 失敗 0 | 実 VoiceBank 824 件の PESQ ではない |
| MossFormerGAN の残した G1 と D4 | 公式 state_dict の最大絶対差 0 と、担当テストが報告 | CPU。このセッションでは重い初期化テストを再実行していない |
| Discriminator を SyncBatchNorm にして DDP、`broadcast_buffers=False` | rank 0 と rank 1 が完了。出力は有限で shape `(2, 1)`。`bn1.running_mean[0]` は両 rank とも `-0.023856936022639275` | GPU。物理 GPU 1 と 2。入力 `(2, 1, 201, 19)`。完全な `validation_report` ではない |

807 は hop 100 の非倍数の件数である。PESQ 失敗件数ではない。実検証波形 824 件に対する PESQ 失敗件数は NOT_TESTED。

## CPU 回帰

次は成功した。

```text
.venvs/se/bin/python -m unittest \
  tests.training.test_preflight_full \
  tests.training.test_checkpoint_retention \
  tests.training.test_resume_identity \
  tests.training.test_validation_interrupt \
  tests.reproduction.test_research_valid_fixed \
  tests.training.test_checkpoint_atomic \
  tests.training.test_ave_nbest \
  tests.training.test_run_status -q
```

47 件、7.827 秒、OK。

MP-SENet の検証・PESQ・既存学習テスト 15 件は 624 秒で OK。先に SIGTERM になったのは同じスイートの重複である。

MossFormer の初期化、PESQ スキップ、plateau、Discriminator、SyncBN の代替 collective は、担当の再実行 4 件が 350 秒で OK であり、その前の一括実行で plateau と Discriminator と初期化が OK だった。最初の一括実行は、CPU で SyncBatchNorm を DDP に包んだ 3 件が失敗した。包まない修正のあと、その 3 件は OK である。

`tests/reproduction` の全発見は 129 件、1,226 秒、ERROR 4、SKIP 2。失敗は `reproduction.duckdb`、`reproduction-p8.duckdb`、`dfn3.online.smoke@1` の snapshot が作業木に無いためである。今回の修正が落とすアサーションではない。

## GPU 検証

| 項目 | 状態 |
|---|---|
| 物理 GPU 1 と 2 で、実 Discriminator の SyncBatchNorm forward が両 rank で終わる | 実行した。上の数値 |
| 公式 2 GPU 学習との損失一致 | NOT_TESTED |
| Trainer の検証ループを GPU で回す | NOT_TESTED |
| MP-SENet の全長検証を GPU で回す | NOT_TESTED |
| 勾配 clip が norm 10 を超える実 step | NOT_TESTED |

GPU 1 と 2 の負荷テストは、この forward の前後で止めて、直後に同じコマンドで戻した。戻したあとの使用メモリは各 32,848 MiB、利用率 100% だった。

## Dataset 監査

研究用 case id は 1,024 件、重複なし、辞書順。SHA-256 は `73e8d44a3c2196af874f8cddb7a1cc9a6b1745a220d6f1b88fda7a9a502f7ad4`。親 snapshot の記録したサイズと mtime は一致した。`resource.duckdb` のサイズと mtime も記録と一致した。`reproduction.duckdb` は無い。

リークは、選択 case id と親 train / test の交差が 0。案件文書の clean SHA-256、話者、録音 ID は UNKNOWN。雑音ファイル名は train と共有し、選択 1,024 件のすべてが共有する。`leakage_safe` ではない。

## Checkpoint 監査

保持テストは、保存失敗で前のファイルが残ること、best が最新 3 件より古くても残ること、平均の参照元を消さないこと、を CPU で見ている。既存の `data/checkpoints` と既存 run は削除していない。

`validation_pending` の再開テストは、偽の Optimizer と scheduler が二重に進まないことを見ている。検証の途中 batch からの再開機構は無い。

## Preflight

`src.training.se.cli --preflight` は学習を始めない。GridNet の本格 YAML は、担当の報告では snapshot `a69a477af70bc43dbcabdafbd5c97ec8a26aac4debd51158079545de1e4f0757`、train 7,345 / valid 1,835 / test 305 で実行条件 PASS だった。このセッションではその CLI を再実行していない。Locoformer の研究用 YAML は snapshot が無く FAIL である。拒否テスト（スモーク cap、world size、claim の混用、resume の dataset 不一致）は `test_preflight_full` に含まれ、上の 47 件の中で OK だった。
