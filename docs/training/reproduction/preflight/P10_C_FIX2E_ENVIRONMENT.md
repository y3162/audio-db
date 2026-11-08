# P10-C Fix.2-E 環境

確認したホストは `tensor` だけである。TSUBAME 4.0 と、このホスト以外の shinoda_lab は NOT_TESTED である。

## 開始時

| 項目 | 値 |
|---|---|
| HEAD | `cc6ae84a0f017da01e7f23c96061a0359df6cff4` |
| ブランチ | `feature/reproduction` |
| 作業木 | クリーン |
| OS | Ubuntu 20.04.6 LTS、Linux 5.4.0-216-generic |
| glibc | 2.31（`ldd (Ubuntu GLIBC 2.31-0ubuntu9.18)`） |
| 仮想環境 | `.venvs/se` |
| Python | 3.12.13 |
| PyTorch | 2.6.0+cu124 |
| DuckDB | 2.0.0 |
| rustc / cargo | 1.76.0 |
| `DF_RFFT_LIBRARY` | 未設定 |

## DeepFilterNet3

| ファイル | このホスト |
|---|---|
| `src/se/models/deepfilternet3/libdf_rfft.so` | `GLIBC_2.33` と `GLIBC_2.34` を要求し、読めない。置き換えていない |
| `build/df-rfft/release/libdf_rfft.so` | `ldd` 成功。要求は `GLIBC_2.28` まで。ABI 1。数値試験の RFFT と一致 |

`pytest tests/training tests/reproduction` は、追跡ファイルが読めないとき上のビルドを `DF_RFFT_LIBRARY` に入れる。パスはリポジトリ相対で、ホストの絶対パスは書いていない。ビルドが無いホストでは `make df-rfft` を先に実行する。Cargo は `src/se/models/deepfilternet3/rfft_kernel` の lock を使う。システム glibc は変えていない。

`unittest` は `conftest.py` を読まない。同じビルドを使うときは次を付ける。

```bash
export DF_RFFT_LIBRARY="$PWD/build/df-rfft/release/libdf_rfft.so"
```

glibc が追跡ファイルを読めるホストでは、pytest はそのファイルのままである。

## P8

```bash
make reproduction-register-p8
```

`data/db/reproduction-p8.duckdb` と `data/staging/reproduction-p8/` を作り、ready な 2 Dataset の snapshot を `data/db/snapshots/` に書く。公式 `dfn3.online.official@1` は `blocked` のままである。`reproduction.duckdb` と `resource.duckdb` は開かない。同じ入力での再実行は挿入 0 で、snapshot hash は次のままである。

| Dataset | dataset_id | snapshot hash |
|---|---|---|
| `dfn3.online.smoke@1` | `283030adc4a81a90d833338fbd2d9e8677ec387b0cd2b7f7d222efb3aa0340e8` | `dec95878ea5854a54656a7a547511cd95809bb383166af94eb402ef41bcbeca6` |
| `dfn3.online.substitute@1` | `ee6223405c19bef4e8f39c9609b830273c422d7ed32b17fbaad0b0554ce1328b` | `861992f6a520823bbe8fc75faf9ffa32740881c89b65eca21bc1160a1de7401b` |
| `dfn3.online.official@1` | `be8e03d6f583c14c5667e6a86a09f6d3cae26b510c5a1aa872bed106ee8f8f60` | 無し。`blocked` |

この hash は [P8 検証](../P8_VERIFICATION.md) の記録と一致する。削除されていた DB ファイルそのもののバイト復元ではない。

## P4 と VoiceBank

現行 `resource.duckdb` には `eval` がある。P4 の実スナップショット試験は、その有無ではなく試験前後の不変を見る。

`make reproduction-prepare` は、この `resource.duckdb` を渡すと safe の `leak_check` が `eval_sets` で FAIL する。重なるのは test 素材の SHA-256 である。製品コードは変えていない。

今回の safe snapshot は、resource DB を開かずに登録した。再構築元は `data/db/reproduction-vb-ondemand.duckdb` と `data/staging/reproduction-vb-ondemand/` である。`reproduction.duckdb` には入れていない。leak_check は UNKNOWN で、PASS ではない。

| Dataset | snapshot hash | 既存ディレクトリ |
|---|---|---|
| `mp_senet.vb28.ondemand.official@1` | `265f256b938a89746dcaa62cd44daa2c6bcca3bce4cc1b1719eb2170717d33d5` | 一致したので置き換えていない |
| `mossformergan.vb28.ondemand.official@1` | `ba0c77c94babbbc2c3e70b149554f8a922a68cf6fbe71ccf833927dc71dff555` | 一致したので置き換えていない |
| `mp_senet.vb28.ondemand.safe@1` | `4528f73a317140cbe05e89d6d30e3a9d2058c28ad00002457551ceb2935d5c3a` | 新規 `54ee0fc9b6b4675c685991ecc9f090e51a3dc8fba0da3cd21231f542c88232da` |
| `mossformergan.vb28.ondemand.safe@1` | `369bffdd754b096a16a59d5a3cac655a2f4ff851a15dd04c90800d03eccd7bbf` | 新規 `5748381b83c289f58485062ab62f099f48899ec98456d85a8e487bf09cb0d8b6` |

履歴の 16 kHz Dataset の snapshot は、この作業では復元していない。

## 移植

新しいホストでは、`.venvs/se` を作ったあと `make df-rfft` と `make reproduction-register-p8` を実行する。VoiceBank の safe snapshot が無いときは、resource DB を開かない再構築を `data/db/reproduction-vb-ondemand.duckdb` へ行い、既存 ID と hash が違う snapshot は置き換えない。glibc が追跡 `.so` を読めるなら、`make df-rfft` の出力を試験が選ばないことがある。その場合も追跡ファイルを新しいビルドで上書きしない。
