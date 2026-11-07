# P10-C Preflight 対応案

研究上の validation 方式は、この監査では決めていない。`cv_synthetic_small` を現行の 36,000 件へ改名しない。21 batch の破棄を、公式と違う欠陥として埋めない。

## 開始前に必須

1. **P10P-A-002。** MP-SENet の検証入力を、生成前に hop 倍数へ切らない。公式と同じく全長をモデルへ入れ、PESQ の欠損は平均の外に出す規則と分けて記録する。824 件中 807 件がこの分岐に入る。
2. **P10P-A-006。** MossFormerGAN の初期 Discriminator を、公式 `train.py` が学習に残す構築回数に合わせる。checksum `7729.488281` は残存重みとして使わない。CPU で公式の構築順を再実行し、残る重みの checksum を取り直してから合わせる。2 回目を残すだけでは、A-D の読みでは公式の初期値に届かない。
3. **P10P-A-007 / P10P-D-004。** 学習中の PESQ 失敗は `TrainingError` にしない。公式と同じく、その batch は更新せず次へ進む。検証が 0 を分母に残す経路は変えない。
4. **P10P-A-008 / P10P-D-002。** 2 GPU の本格比較では、Discriminator だけ SyncBatchNorm にする。Generator には BatchNorm が無い。1 GPU の official YAML ではこの項目を開始条件にしない。
5. **P10P-B-005 / P10P-F-001 / P10P-F-002 / P10P-F-003。** TF-GridNet と TF-Locoformer の本格 YAML を、p10b のコピー以外として新しく作る。`verification_max_updates`、`max_updates_per_epoch`、`validation_max_batches`、`chunk_iterator.max_updates`、`validation_max_updates` を置かない。`checkpoint_interval` は 2 にしない。要求 world size を設定に書き、違うプロセス数は開始前に拒否する。TF-GridNet は world 2 と world 4 の両方が蓄積 1 で通る。
6. **P10P-F-007。** TF-Locoformer の 150 epoch 全件検証を始める前に、検証集合の方針を決める。36,000 件を毎 epoch 見る案は、接頭辞外挿で検証 841 時間が学習 384 時間を上回る。集合の定義を変える修正は、この決定の前にしない。

## 開始前に推奨

- **P10P-F-004。** 本格を中断再開するなら、cap キーを再開比較に含める。新規開始だけで再開しないなら、開始の必須にはしない。
- **P10P-A-003。** PESQ 失敗を平均から外す差を、公式平均（-1 を残す）と結果表で分ける。例外件数は未計測なので、807 件を例外件数として書かない。
- **P10P-D-001。** 2 GPU の MossFormerGAN で、勾配 norm が 10 を超える step があるかを短い GPU 計測で見る。超えるなら clip を平均の後へ移す。未計測のまま、公式と数値一致したと書かない。
- **P10P-A-011 / P10P-G-004。** 検証 crop の開始位置を、公式の `random.randint` と比較する。半減 epoch がずれるかは、その後の測定まで未確認として残す。
- 開始前検査を、学習ループの外に置く。本格 YAML にスモーク cap が無いこと、`checkpoint_interval` が更新間隔 2 でないこと、world size が要求と一致すること、`complete.json` を完了の唯一の証拠にしないこと。

## 本格と並行してよい

- **P10P-F-005。** provenance に `git_dirty` を書く。
- **P10P-F-006 / P10P-F-009。** `complete.json` に終了理由を足す。容量見積に sidecar を入れる。
- **P10P-A-001。** 保存時点の 1 更新差を、結果表の注記にする。epoch 境界の学習率は一致している。
- **P10P-E-001 / P10P-E-003。** epoch 終端の成果物が pickle より先に残る窓を、運用手順に書く。途中 checkpoint の新設はしない。
- **P10P-B-001。** 21 batch の破棄は公式と同じ手順である、と検証記録に残す。6 発話が損失に入らないことを、全件検証と呼ばない。

## 公式情報が無く、保留する

- **P10P-B-003。** `cv_synthetic_small` の件数と選択規則。公開 checkout の欠ける blob を取ってもなお不明なら、UNKNOWN のままにする。
- **P10P-B-007。** README の結果 hash と MS-SNSD recipe の対応。
- **P10P-C-001。** 登録 DB と snapshot の同一性。DB ファイルが作業木に無いので、今は証明できない。学習開始は snapshot だけで足りる。
- 公式 iSTFT 実装との長波形の勾配差。`torch_complex` が無く、公式 `Stft` を import できていない。

## 研究上の判断が必要

- MP-SENet と MossFormerGAN の公式 protocol は、validation 824 件と test 824 件が同一である。`selection_uses_test` は true。公式再現の選択結果と、リークを避けた研究評価を、同じ数値として報告しない。safe 分割（p286 / p287 の 872 件）は別 Dataset として残す。
- TF-Locoformer の検証を 36,000 件のままにするか、時間を削るために別集合にするか。削る場合も、評価規則を変えたことを別 Protocol にする。`cv_synthetic_small` と呼ばない。
- TF-GridNet の 21 batch 破棄を残す。これは公式 trainer の最短 rank 打ち切りと同じである。
- `valid.si_snr.best.pt` を公式の選択結果として使わない。公式が残す選択は `valid.loss` の n-best である。GridNet の `si_snr` 最大化は、SI-SDR が低い側を選ぶ。

## 次の修正順

1. A-006 の構築順を CPU で再同定し、Discriminator の初期値を合わせる。checksum の DISPUTED を先に閉じる。
2. A-007 / D-004 の学習停止を、batch スキップへ変える。
3. A-002 の検証切り詰めを外し、PESQ 欠損の集計と分離する。
4. 2 GPU だけ Discriminator を SyncBatchNorm にする。1 GPU YAML は対象外。
5. TF の本格 YAML を新規に書き、cap と `checkpoint_interval: 2` と world size 不一致を開始前検査で拒否する。
6. TF-Locoformer の検証方針を決めるまで、150 epoch の全件検証は投入しない。
7. 21 batch、5000 対 5001、DuckDB 不在、公式再開の scheduler 再構築は、修正の必須列に入れない。記録と、公式成果物の呼び分けだけ行う。
