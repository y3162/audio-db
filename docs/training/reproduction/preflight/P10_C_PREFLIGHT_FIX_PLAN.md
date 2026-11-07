# P10-C Preflight 修正計画

本格学習は開始しない。p10b のスモーク YAML は上書きしない。既存 Dataset `tf_locoformer.dns.accept_recipe@1` と、その snapshot は変更しない。

## 担当

| Agent | 範囲 | 編集してよい主なファイル |
|---|---|---|
| 1 | MP-SENet の検証入力と PESQ 集約 | `src/training/se/mp_senet.py` と MP-SENet のテスト |
| 2 | MossFormerGAN の初期化、PESQ スキップ、SyncBatchNorm | `src/training/se/mossformergan.py` と MossFormer のテスト |
| 3 | Trainer、Checkpoint 保持、再開照合、本格 YAML、Preflight | `src/training/core/trainer.py`、`config.py`、`run.py`、新しい YAML、`preflight.py` |
| 4 | TF-Locoformer の研究用固定 Validation | 新しい selector、追記登録、研究用 Dataset |
| 5 | 独立検証 | 本番コードは編集しない |

同じファイルを同時に編集しない。Agent 2 は Trainer を編集せず、`update_applied` と `validation_requires_all_ranks` を返す。Agent 3 がその契約を Trainer に接続する。

## 対象 Issue と方針

### P10P-A-002 MP-SENet の hop 切り詰め

公式 `train.py` は検証波形の全長を STFT して Generator に入れる。現行 `validation_report` は hop 倍数へ切ってから入れている。824 件中 807 件がこの分岐に入る。

修正: 検証でも全長を Generator へ渡す。学習更新の長さ不一致拒否は維持する。center iSTFT が短く返す差は、公式と同じく波形側の長さ差として扱う。切り詰め入力と全長入力の magnitude 最大絶対誤差を測定して残す。807 件は切り詰め件数であり、PESQ 失敗件数ではない。

### P10P-A-003 PESQ 集約

公式 `eval_pesq` は例外を `-1` にして `np.mean` に残す。現行の研究用集約は `na` を平均から外す。

修正: `loss: official` の検証平均だけ公式の集約にする。再構成損失の研究用集約は別名のまま残す。失敗件数は別カウンタで測る。未計測の全件件数を、切り詰め件数で代用しない。

### P10P-A-006 Discriminator 初期化

公式は `network_wrapper` を 2 回作り、各 wrapper が `MossFormerGAN_SE_16K` を 2 回作る。乱数順は G1 D1 G2 D2 G3 D3 G4 D4 で、学習に残るのは G1 と D4 である。checksum `7729.488281` は残存重みの正解にしない。

修正: 同じ seed の初期化ブロックで 8 構築し、G1 と D4 を残す。公式コードを CPU で実行し、残存重みを比較する。

### P10P-A-007 / P10P-D-004 PESQ 失敗

公式は該当 batch を backward も optimizer step もせず次へ進む。現行は `TrainingError` で run を止める。

修正: 失敗 batch は更新しない。DDP では skip フラグを all-reduce し、どれか 1 rank が失敗したら全 rank が backward しない。同じ batch は再試行しない。スキップ理由、位置、件数をログへ残す。`update_applied == 0` のとき Trainer は `global_step` を進めない。これにより再開時の Optimizer 二重更新を避ける。

公式の単一 GPU は、失敗 micro-batch を蓄積回数に数えず、後続の成功 batch で窓を埋める。audio-db の `train_update` は固定グループを受け取る。グループ全体を skip するのは、DDP の collective と再開カーソルを壊さないための意図した差である。

### P10P-A-008 / P10P-D-002 SyncBatchNorm

公式の 2 GPU は Discriminator を `SyncBatchNorm` にしてから DDP する。Generator に生きた BatchNorm は無い。world size 1 は通常 BatchNorm のまま。

修正: world size が 2 以上のとき Discriminator だけ変換する。検証で rank 0 だけが SyncBatchNorm の forward を実行しない。`validation_requires_all_ranks` のとき全 rank が同じ回数 forward する。回数が違うときは forward の前に拒否する。`broadcast_buffers=False` を維持する場合は、3 回 forward の逆伝播を守る最小差として記録する。GPU 上の数値一致は、この作業では測らない。

### P10P-B-005 / P10P-F-001 / P10P-F-002 / P10P-F-003 本格 YAML

p10b はスモークのまま残す。新しい本格 YAML を置く。

- TF-GridNet: world size 2、`checkpoint_interval` は 2 ではない、スモーク cap なし、claim は「公式生成アルゴリズムに準拠し、乱数系列のみ再現可能な方式へ変更した」。ビット一致や統計的同等の実証とは書かない。
- TF-Locoformer: world size 4、研究用 Validation を参照、`keep_nbest: 5`、スモーク cap なし。公式 `cv_synthetic_small` とは呼ばない。
- `required_world_size` が実プロセス数と違うときは、モデル構築と snapshot の大量読み込みと Optimizer 更新より前に拒否する。global batch の偶然の一致では代用しない。

### P10P-F-004 再開と比較

cap 付き checkpoint から cap 無しの本格設定へ、無警告では進まない。許可する差は logging だけを allowlist にする。Dataset、snapshot、protocol、model、world size、global batch、蓄積、optimizer、scheduler、選択規則、validation 集合、validation 頻度、実験設定 hash は一致を要求する。

### P10P-F-007 と研究用 Validation（D1, D2）

既存 36,000 件は残す。新しい固定集合は別 Dataset `tf_locoformer.dns.accept_recipe.research_valid_fixed@1` とする。設計件数は 1,024。選択は seed `20261107` の `research_valid_fixed_v1`。先頭 1,024 件は使わない。毎 epoch、`valid.loss` で選択する。最終 test は選択に使わない。`selection_uses_test=false`。完全な公式再現とは呼ばない。

独立性は case id、clean SHA-256、録音、話者、noise、test を分けて記録する。UNKNOWN は PASS にしない。noise 共有があるときは件数と研究評価への影響を書く。独立性を確認できない範囲は `leakage_safe` と呼ばない。

### Checkpoint 保持（D3）と検証中断（D4）

新しい run から、直近 3 件の再開可能 checkpoint、best、n-best 平均に必要な checkpoint、平均済み checkpoint、最終 checkpoint を守る。`ave_5best` の参照 5 件は削除しない。symlink の参照先と、他 run のファイルは削除しない。保存は一時ファイル、検証、`os.replace` の順を維持し、失敗時は直前の正常 checkpoint を残す。

検証の途中再開機構は作らない。中断した検証は最初からやり直す。Optimizer 更新のあと、scheduler と best の前に `validation_pending` の再開 checkpoint を残す。未完了の検証を完了扱いにしない。scheduler と best は完了 checkpoint が replace されてから確定したとみなす。途中で落ちた場合は pending のまま検証を最初からやり直し、scheduler は 1 回だけ進める。

### D5 公式忠実と leakage-safe

MP-SENet と MossFormerGAN の official YAML は `official_faithful` と `selection_uses_test=true`。safe YAML は `leakage_safe`。claim と Dataset の組み合わせが違う Preflight は拒否する。結果と checkpoint ディレクトリは混ぜない。

### D6 TF-GridNet の乱数

本格 YAML の注記は、公式生成アルゴリズムに準拠し、乱数系列だけを再現可能な固定 `mix_seed` に変えた、とする。公式 WAV とのビット一致、統計的同等の実証、同一 seed による公式学習の完全再現は書かない。

## 依存

1. Agent 2 の `update_applied` と `validation_requires_all_ranks` を Agent 3 が Trainer に接続する。
2. Agent 4 の Dataset ID を Agent 3 の TF-Locoformer 本格 YAML が参照する。登録が BLOCKED のとき、YAML の ID は予定のままにし、Preflight は Dataset 不在を FAIL にする。本格学習は始めない。
3. Agent 5 は上記が揃ってから、公式コードとテストを独立に見る。

## テスト

CPU で、PROMPT の 15 項目に対応する単体テストを追加する。GPU でしか確認できない数値は NOT_TESTED とし、PASS と書かない。

## この計画の時点でまだ行っていないこと

実装、公式重みの再測定、回帰の実行、独立検証、GO 判定は、この文書の後の作業である。この文書は修正仕様であり、検証済みの結果ではない。
