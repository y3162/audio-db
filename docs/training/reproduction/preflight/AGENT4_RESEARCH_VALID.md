# TF-Locoformer 研究用固定検証（Agent 4）

日付: 2026-11-07。学習は開始していない。GPU は使っていない。混合 WAV と波形キャッシュは作っていない。

## 結論

- Dataset id は `tf_locoformer.dns.accept_recipe.research_valid_fixed@1`。登録状態は **BLOCKED**。snapshot hash は発行していない。
- 親は `tf_locoformer.dns.accept_recipe@1`。manifest の snapshot hash は `3343a55efb4c57d9ffef9179fb2f39c3e0186d36f587a0f4ad9f0aa14e9926f5`。検証 36,000 件から 1024 件を選んだ。
- リーク判定は case id の train / test が PASS、案件文書の clean SHA-256 が UNKNOWN、話者が UNKNOWN、録音 ID が UNKNOWN、雑音ファイル名が OVERLAP。`leakage_safe` ではない。
- 150 epoch の検証時間は式で外挿すると 23.9232 時間。公開されている全件検証 841 時間に対する外挿であり、再計測ではない。

## Dataset

| 項目 | 値 |
|---|---|
| dataset id | `tf_locoformer.dns.accept_recipe.research_valid_fixed@1` |
| claim | `research_checkpoint_selection` |
| profile | `official_faithful`。スキーマ上の非 leakage 値であり、公式再現という意味ではない |
| selection_uses_test | false |
| 登録 | BLOCKED |
| snapshot hash | BLOCKED。null のまま。作っていない |
| 親 dataset | `tf_locoformer.dns.accept_recipe@1` |
| 親 dataset_id | `5469f0416329a8661df2cc91cd5bd9a149d6ea223c72e69ebdd5696bea6bdfbe` |
| 親 protocol_id | `cadda632ebd08a9066e5b7a81851920762cc2a5fa5bddc21f8488bf1a1630116` |
| 親 snapshot hash | `3343a55efb4c57d9ffef9179fb2f39c3e0186d36f587a0f4ad9f0aa14e9926f5` |
| 親の件数 | train 324000 / valid 36000 / test 150 |
| この定義の valid | 1024 |
| train / test の case set | 親と同じ id を指す想定。新しいスナップショットは書いていない |
| case_id_list_sha256 | `73e8d44a3c2196af874f8cddb7a1cc9a6b1745a220d6f1b88fda7a9a502f7ad4` |

未登録の内容ハッシュは、登録できた場合に `dataset_id_for` が返す値である。スナップショットハッシュではない。

- protocol 内容 id: `1267fe5f4aff1e37ab1e4c008ebfbbc57bf98676104870dd731be17167f14f0e`
- valid case set 内容 id: `9a5a369bee9556178c4f86248229b18ad2d664446e4bff05f5414836aefd30eb`
- dataset 内容 id: `8bd167b5a650baefb05ab24189d2c540c3fc3f9eb3a19900161e4a75e6d29782`

## 選択規則

アルゴリズム名は `research_valid_fixed_v1`。seed は整数 `20261107`。候補数は 1024。母集団は親の validation 36,000 件だけである。先頭 1024 件は取らない。

```text
research_valid_fixed_v1

Population: the validation case_id values of tf_locoformer.dns.accept_recipe@1.
Input order is ignored. A duplicate case_id is an error.
Case ids that also appear in the parent train or test split are removed before
ranking. If that removal leaves fewer than 1024 ids, the published set is the
entire remainder and the count is documented as smaller. Otherwise the count
stays 1024. When the validation split is disjoint from train and test, the
ranked population is exactly the sorted validation case ids.

Rank: SHA-256 digest of the UTF-8 bytes of "{seed}\n{case_id}".
The seed is the non-negative decimal integer 20261107, with no sign and no
leading zeros. The digest is the raw 32 bytes, ordered lexicographically.
Ties break by case_id ascending. Take the first 1024 surviving case ids.
Publish those ids sorted lexicographically. Do not publish them in rank order.

case_id_list_sha256 is the hex SHA-256 of the UTF-8 bytes of
"\n".join(published_case_ids) + "\n".

Call the function on the full population. Hashing each worker's shard and
uniting the lowest ids is a different set. The random module and NumPy
Generator are not used.
```

公開リストは辞書順である。SHA-256 は `73e8d44a3c2196af874f8cddb7a1cc9a6b1745a220d6f1b88fda7a9a502f7ad4`。
同じ整列済み case id を渡す限り、プロセス数には依存しない。worker ごとの部分母集団から下位を集める操作は、この規則と違う集合になる。

train / test と重なった validation case id は 0 件だった。件数は 1024 のままである。

## 登録が BLOCKED である理由

- この作業木に data/db/reproduction.duckdb が無い。空の DB は作っていない。親の行が無いファイルへは追記できない。
- data/staging/reproduction/INDEX.json が無い。register_staging は INDEX の無いディレクトリを読めない。書いた JSON は登録グラフではない。
- claim_label research_checkpoint_selection は CLAIM_LABELS と repro.datasets の CHECK に無い。スキーマは変えていない。既存の公式再現ラベルへは付け替えていない。
- 話者、案件文書の clean SHA-256、録音 ID のいずれかが UNKNOWN であるか、雑音ファイルが train と共有する。leakage_safe としては登録しない。
- スナップショットは export していない。snapshot hash は作っていない。

`register_staging` は呼んでいない。`reproduction.duckdb` は作っていない。`INDEX.json` も作っていない。staging に書いた JSON は定義の写しであり、登録グラフではない。

## リーク

PASS は、その検査の入力が揃っていて重なりが 0 のときだけ使う。欠測を PASS にはしない。

| 検査 | 状態 | 内容 |
|---|---|---|
| 選択 case id と親 train の交差 | PASS | 交差 0 |
| 選択 case id と親 test の交差 | PASS | 交差 0 |
| 案件文書の clean SHA-256 | UNKNOWN | 文書にある件数 parent valid 0 / selected 0 |
| speaker_id | UNKNOWN | valid present 0 / absent 36000、selected present 0 / absent 1024 |
| recording id | UNKNOWN | 案件文書に recording_id が無い |
| 雑音ファイル名の train との共有 | OVERLAP | 共有ファイル名 3000、共有する選択案件 1024 |

leakage_safe: false。上の検査がすべて PASS ではないため、この名前では呼ばない。

### ファイル名と、案件文書に無い SHA-256

選択 1024 件のうち 1024 件が、train と少なくとも 1 つの雑音ファイル名を共有する。共有する雑音ファイル名は 3000 件である。DNS の受理計画は train と validation が同じ雑音プールから切り出されるので、この共有は想定内である。共有があるため、この検証集合は学習データと音源が独立ではない。checkpoint の選択にだけ使い、独立な汎化評価とは呼ばない。

選択 1024 件のうち 0 件が、train と少なくとも 1 つの clean ファイル名を共有する。共有する clean ファイル名は 0 件である。案件文書に clean の SHA-256 は無いので、その検査は UNKNOWN のままである。ファイル名の一致を SHA-256 の PASS にはしない。

親 validation 36,000 件と train の clean ファイル名交差は 2 件、雑音ファイル名交差は 43823 件である。選択 1024 件の clean 交差が 0 なのは、その交差を除く規則ではなく、ハッシュ抽出の結果である。

案件の speaker_id は無く、文書に recording_id も無い。ファイル名に reader という綴りがあっても、話者 ID としては読まない。話者と録音 ID は UNKNOWN であり、PASS ではない。

| 追加の計数 | 値 |
|---|---|
| train と共有する clean ファイル名 | 0 |
| clean ファイル名を train と共有する選択案件 | 0 / 1024 |
| 共有 clean ファイル名の sha256（名前リスト） | なし（共有 0） |
| スナップショット asset へファイル名で結んだ clean SHA の共有 | NONE / 共有 0 |
| train と共有する雑音ファイル名 | 3000 |
| 雑音ファイル名を train と共有する選択案件 | 1024 / 1024 |
| 共有雑音ファイル名の sha256（名前リスト） | `3798a00b0add2e45fcf7faf230e7a5794b60cd9cceeef28915ed917bdcebf33c` |
| スナップショット asset へ結んだ雑音 SHA の共有 | OVERLAP / 共有 3000 |
| 未解決のファイル名 | clean 0 / noise 0 |
| 親 validation 全体と train の clean ファイル名共有 | 2 |
| 親 validation 全体と train の雑音ファイル名共有 | 43823 |

案件文書の clean SHA-256 が UNKNOWN であることは、asset 表への結合結果では覆らない。結合はファイル名が指す元素材の計数であり、文書内ハッシュの PASS ではない。

親 Dataset の DB `leak_check` は、このホストに DB が無いので再実行していない。受理レシピの検証記録では asset 交差が 0 で話者は WARN だった。生成案件は asset 列を持たず、素材集合は train スロットにだけ付く。そのため DB の asset 交差 0 は、案件文書の中の clean / noise ファイル再利用が 0 であることを意味しない。ここでの計数は案件文書のファイル名である。

## 分布

比較相手は親の validation 36,000 件である。train だけとの比較ではない。割合は件数からの表示用丸め（小数 6 桁）である。

| 項目 | parent valid | selected |
|---|---:|---:|
| 案件数 | 36000 | 1024 |
| ユニーク clean ファイル名 | 5174 | 919 |
| ユニーク noise ファイル名 | 43835 | 3001 |
| RIR フィールドがある文書 | 0 | 0 |
| clean SHA-256 が文書にある案件 | 0 | 0 |
| speaker_id がある案件 | 0 | 0 |

### SNR（dB）

最大絶対割合差 0.012115。parent 合計 36000、選択合計 1024。

| 値 | parent valid | selected | parent 割合 | selected 割合 | 差 |
|---:|---:|---:|---:|---:|---:|
| -5 | 1766 | 57 | 0.049056 | 0.055664 | 0.006609 |
| -4 | 1796 | 46 | 0.049889 | 0.044922 | -0.004967 |
| -3 | 1836 | 42 | 0.051000 | 0.041016 | -0.009984 |
| -2 | 1806 | 60 | 0.050167 | 0.058594 | 0.008427 |
| -1 | 1798 | 49 | 0.049944 | 0.047852 | -0.002093 |
| 0 | 1778 | 53 | 0.049389 | 0.051758 | 0.002369 |
| 1 | 1844 | 50 | 0.051222 | 0.048828 | -0.002394 |
| 2 | 1915 | 59 | 0.053194 | 0.057617 | 0.004423 |
| 3 | 1776 | 53 | 0.049333 | 0.051758 | 0.002424 |
| 4 | 1817 | 42 | 0.050472 | 0.041016 | -0.009457 |
| 5 | 1820 | 44 | 0.050556 | 0.042969 | -0.007587 |
| 6 | 1849 | 65 | 0.051361 | 0.063477 | 0.012115 |
| 7 | 1799 | 45 | 0.049972 | 0.043945 | -0.006027 |
| 8 | 1760 | 49 | 0.048889 | 0.047852 | -0.001037 |
| 9 | 1756 | 46 | 0.048778 | 0.044922 | -0.003856 |
| 10 | 1828 | 59 | 0.050778 | 0.057617 | 0.006839 |
| 11 | 1816 | 55 | 0.050444 | 0.053711 | 0.003266 |
| 12 | 1797 | 47 | 0.049917 | 0.045898 | -0.004018 |
| 13 | 1769 | 54 | 0.049139 | 0.052734 | 0.003595 |
| 14 | 1674 | 49 | 0.046500 | 0.047852 | 0.001352 |

### 目標レベル（dBFS）

最大絶対割合差 0.010312。parent 合計 36000、選択合計 1024。

| 値 | parent valid | selected | parent 割合 | selected 割合 | 差 |
|---:|---:|---:|---:|---:|---:|
| -35 | 1891 | 54 | 0.052528 | 0.052734 | 0.000207 |
| -34 | 1831 | 47 | 0.050861 | 0.045898 | -0.004963 |
| -33 | 1877 | 58 | 0.052139 | 0.056641 | 0.004502 |
| -32 | 1866 | 51 | 0.051833 | 0.049805 | -0.002029 |
| -31 | 1862 | 59 | 0.051722 | 0.057617 | 0.005895 |
| -30 | 1913 | 64 | 0.053139 | 0.062500 | 0.009361 |
| -29 | 1824 | 44 | 0.050667 | 0.042969 | -0.007698 |
| -28 | 1814 | 51 | 0.050389 | 0.049805 | -0.000584 |
| -27 | 1925 | 53 | 0.053472 | 0.051758 | -0.001714 |
| -26 | 1852 | 46 | 0.051444 | 0.044922 | -0.006523 |
| -25 | 1864 | 60 | 0.051778 | 0.058594 | 0.006816 |
| -24 | 1874 | 61 | 0.052056 | 0.059570 | 0.007515 |
| -23 | 1831 | 48 | 0.050861 | 0.046875 | -0.003986 |
| -22 | 1802 | 51 | 0.050056 | 0.049805 | -0.000251 |
| -21 | 1845 | 56 | 0.051250 | 0.054688 | 0.003438 |
| -20 | 1672 | 37 | 0.046444 | 0.036133 | -0.010312 |
| -19 | 1721 | 52 | 0.047806 | 0.050781 | 0.002976 |
| -18 | 1653 | 41 | 0.045917 | 0.040039 | -0.005878 |
| -17 | 1560 | 51 | 0.043333 | 0.049805 | 0.006471 |
| -16 | 1523 | 40 | 0.042306 | 0.039063 | -0.003243 |

### clean の source_rate

最大絶対割合差 0.000000。parent 合計 36000、選択合計 1024。

| 値 | parent valid | selected | parent 割合 | selected 割合 | 差 |
|---:|---:|---:|---:|---:|---:|
| 16000 | 36000 | 1024 | 1.000000 | 1.000000 | 0.000000 |

### noise の source_rate

最大絶対割合差 0.001082。parent 合計 109831、選択合計 3133。

| 値 | parent valid | selected | parent 割合 | selected 割合 | 差 |
|---:|---:|---:|---:|---:|---:|
| 16000 | 109486 | 3126 | 0.996859 | 0.997766 | 0.000907 |
| 44100 | 189 | 2 | 0.001721 | 0.000638 | -0.001082 |
| 48000 | 145 | 5 | 0.001320 | 0.001596 | 0.000276 |
| 96000 | 11 | 0 | 0.000100 | 0.000000 | -0.000100 |

### clean 断片数

最大絶対割合差 0.000000。parent 合計 36000、選択合計 1024。

| 値 | parent valid | selected | parent 割合 | selected 割合 | 差 |
|---:|---:|---:|---:|---:|---:|
| 1 | 36000 | 1024 | 1.000000 | 1.000000 | 0.000000 |

### noise 断片数

最大絶対割合差 0.009298。parent 合計 36000、選択合計 1024。

| 値 | parent valid | selected | parent 割合 | selected 割合 | 差 |
|---:|---:|---:|---:|---:|---:|
| 5 | 33698 | 949 | 0.936056 | 0.926758 | -0.009298 |
| 6 | 474 | 14 | 0.013167 | 0.013672 | 0.000505 |
| 7 | 1825 | 61 | 0.050694 | 0.059570 | 0.008876 |
| 9 | 3 | 0 | 0.000083 | 0.000000 | -0.000083 |

### RIR

案件文書に RIR フィールド（rir, rir_id, reverb, rt60）は無かった。parent valid の該当文書は 0、選択集合は 0。状態は UNKNOWN であり、PASS ではない。DNS のこの受理計画は RIR を混合しない、という既存の説明を、この欠測の PASS には使わない。

## 時間見積

再計測ではない。P10-C の公開値を、発話数に比例させて外挿した。1 chunk 0.1602 秒、1 発話 14 chunk、4 rank が並列で、1 回の壁時計は rank あたり chunk 数 × 0.1602 秒である。36,000 発話では rank あたり 126,000 chunk になり、公開の 5.61 時間 / 150 epoch 841 時間と同じ式である。式どおりに計算すると 1 回 5.607 時間、150 epoch で 841.05 時間になる。841 時間は公開文書の丸めである。

| 項目 | 1024 発話 | 親の 36000 発話 |
|---|---:|---:|
| 発話数 | 1024 | 36000 |
| chunk 数 | 14336 | 504000 |
| rank あたり chunk | 3584 | 126000 |
| 検証 1 回（時間） | 0.159488 | 5.607 |
| 検証 150 epoch（時間） | 23.9232 | 841.05 |

学習時間は 384.4 時間のままである。
Checkpoint は epoch ごとに 1 回書き、その後に剪定する想定である。書き込み時間は 2.29 秒 × 150 = 343.5 秒（0.0954167 時間）。150 ファイルは残さない。保持は最新 3、best、nbest 5、平均の役割で、役割が重ならないときの上限は 10 ファイル、1799101640 バイトである。平均ファイルを epoch 書き込みとは別に足す計測は無いので、足していない。

合計は学習 384.4 + 検証 23.9232 + checkpoint 書き込み 343.5 秒で、408.4186 時間。
公開の全件検証 841 時間に対し、この検証 150 epoch は 23.9232 時間である。同じ式の全件側は 841.05 時間で、差は検証時間の短縮分である。
P10-C の余裕（学習 1.3、検証 1.5）を同じ外挿に掛けると、合計は約 535.7002 時間になる。checkpoint 書き込みには係数を掛けていない。これも外挿である。

## 主張していないこと

- 公式再現であるとは主張しない。`cv_synthetic_small` でもない。公式の共有カウンタとのビット一致も主張しない。
- `leakage_safe` とは主張しない。話者、録音 ID、案件文書の clean SHA-256 は UNKNOWN である。雑音ファイル名は train と共有する。選択 1024 件の clean ファイル名と train の交差は 0、親 validation 全体では 2 である。文書内 SHA-256 が無いので、clean の独立は PASS にしない。
- 親の 36,000 件 validation を置き換えたとは主張しない。親の protocol id と snapshot hash は変えていない。
- 新しい snapshot hash は無い。未登録の内容 id をスナップショットの所在として扱わない。
- 150 epoch は走らせていない。時間は公開計測からの外挿である。
- この集合での WER、PESQ、STOI、early stopping の実際の epoch 数は測っていない。

## 変更していないもの

- `data/db/resource.duckdb`
- `data/db/reproduction.duckdb`（この作業木には無かった。作っていない）
- 親スナップショットディレクトリのファイル
- `data/corpora`、`data/derived`、`data/checkpoints`、`data/results`
- 受理レシピの build yaml
