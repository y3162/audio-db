# モデルの構造

構造と内部ハイパーパラメータは実験 YAML に書く。Python は計算と構築を持つ。新しいローダや、モデルごとの別スキーマは足さない。既存の `role: module`、`class`、`kwargs`、`architecture` を使う。

変えてはいけない条件は [.agents/ARCHITECTURE.md](../../.agents/ARCHITECTURE.md)。損失の係数は [損失](LOSS.md)。

## 方針

設定は 4 つに分ける。

| 分類 | 扱い |
|---|---|
| A. 設定可能な構造 | YAML に明示する。研究用の実験はここを変える |
| B. 導出値 | 他の値から Python が計算する。YAML に複製しない |
| C. アルゴリズム定数 | そのモデルの式の一部。YAML に出さない |
| D. 実行時のスイッチ | 重みの形を変えない。構造の変更とは扱わない |

省略したキーは、そのモデルの公開コンストラクタ既定値で埋める。YAML に書いた値を Python の既定値で上書きしない。実験モジュールの未知の kwargs、型の違う値、割れないチャネルとヘッドの組は構築時に拒否する。黙って公式値へ戻さない。`src.se` の MP-SENet 構築は、公開 JSON に混ざる optimizer などの非構造キーを読まず、構造キーだけを使う。

`config_sha256` はコンポーネントの class、kwargs、inject、prepare を含む。構造を変えた YAML は別の実験であり、変える前の checkpoint から再開できない。重みの shape が違う state_dict は strict load で失敗する。一部だけ読み込まない。

## YAML の置き場

モデルは `components` の `role: module` である。フラットな kwargs を取るものと、`kwargs.architecture` を取るものがある。

| モデル | コンポーネント | 構造の場所 |
|---|---|---|
| MP-SENet | `generator` と `metric_discriminator`、または波形アダプタ | `kwargs` |
| MossFormerGAN | `generator` と `discriminator` | `kwargs` |
| SEMamba++ | `generator`、`mrd`、`cqt` | `kwargs` |
| DeepFilterNet3 | `model` | `kwargs.architecture` |
| TF-GridNet | `model` | `kwargs.architecture` |
| TF-Locoformer | `model` | `kwargs.architecture` |

公式忠実条件は `configs/reproduction/experiments/official/`。1 step の宣言は `tests/fixtures/configs/experiments/`。SEMamba++、DeepFilterNet3、TF-Locoformer の公式学習用 YAML は、公式データが閉じているか研究用レシピであるため、`official/` には置いていない。数値の正本は下の表と、各クラスの公開定数である。

Generator と Discriminator を別コンポーネントに持つモデルは、構造の辞書が一致しないと Trainer が構築を拒否する。`disc_dim` のように Discriminator だけが使う値も、両方の辞書に書いて一致させる。ネットワークへ渡す前に MP-SENet は `disc_dim` を、SEMamba++ は MRD と CQT のキーを落とす。

## 公式値

研究用に変えてよい。`official_faithful` の MP-SENet と MossFormerGAN は、この表と違う構造を構築時に拒否する。比較相手はモデル自身の重みではなく、コードに置いた公開定数である。`leakage_safe` は損失係数だけを公式に固定し、構造は固定しない。`research` はどちらも固定しない。

### MP-SENet

| 名前 | 公式値 | 制約 |
|---|---|---|
| `dense_channel` | 64 | 正の整数。`n_heads` で割り切れる |
| `num_tsconformers` | 4 | 正の整数 |
| `n_heads` | 4 | 正の整数 |
| `dense_depth` | 4 | 正の整数。Encoder と両 Decoder の DenseBlock |
| `ts_dropout` | 0.0 | 0 以上 1 以下。TS-Conformer の attention dropout |
| `n_fft` | 400 | 正の整数。`win_size` 以上 |
| `hop_size` | 100 | 正の整数 |
| `win_size` | 400 | 正の整数 |
| `compress_factor` | 0.3 | 正の数 |
| `beta` | 2.0 | 正の数 |
| `sampling_rate` | 16000 | 16000 固定 |
| `disc_dim` | 16 | 正の整数。MetricDiscriminator の幅。MPNet には渡さない |

`disc_dim` は波形アダプタには無い。

### MossFormerGAN

| 名前 | 公式値 | 制約 |
|---|---|---|
| `variant` | `se_16k` | この値だけ |
| `num_channel` | 64 | 正の整数。`n_head` で割り切れる |
| `n_layers` | 6 | 正の整数。SyncANetBlock の数 |
| `n_head` | 4 | 正の整数 |
| `approx_qk_dim` | 512 | 正の整数 |
| `dense_depth` | 4 | 正の整数 |
| `dropout` | 0.1 | 0 以上 1 以下。FFConvM と MossFormer |
| `query_key_dim` | 128 | 正の整数 |
| `expansion_factor` | 4 | 正の偶数。YAML の整数 4 は 4.0 として扱う |
| `n_fft` | 400 | `win_len` と同じ。学習経路は 400 以外を拒否する |
| `win_len` | 400 | 学習経路は 400 固定 |
| `hop` | 100 | 学習経路は 100 固定 |
| `sampling_rate` | 16000 | 16000 固定 |

### SEMamba++

| 名前 | 公式値 | 制約 |
|---|---|---|
| `hid_feature` | 48 | 正の整数 |
| `num_tfmamba` | 4 | 正の整数 |
| `d_state` | 16 | 正の整数 |
| `d_conv` | 4 | 正の整数 |
| `expand` | 2 | 正の整数 |
| `norm_epsilon` | 1e-5 | 正の数 |
| `input_channel` | 2 | 2 固定 |
| `output_channel` | 1 | 1 固定 |
| `n_fft` | 400 | 400 固定 |
| `hop` | 100 | 正の整数。`enhance()` の hop は 100 のまま |
| `win` | 400 | `n_fft` と同じ |
| `sampling_rate` | 16000 | 16000 固定 |
| `scan_backend` | `reference` | `reference` または `fused`。構造ではない |
| `mrd_channels` | 32 | 正の整数 |
| `mrd_resolutions` | `(1024, 120, 600)`, `(2048, 240, 1200)`, `(512, 50, 240)` | 空でない `(n_fft, hop, win)` の列 |
| `cqt_channels` | 128 | 正の整数 |
| `cqt_hops` | `512, 256, 256` | `cqt_octaves`、`cqt_bins_per_octave` と同じ長さ |
| `cqt_octaves` | `8, 8, 8` | 同上 |
| `cqt_bins_per_octave` | `12, 24, 36` | 同上 |

`mrd_*` と `cqt_*` は 3 コンポーネントすべての kwargs に書く。生成器の `build_model` には渡さない。

### DeepFilterNet3

`architecture` に公開 `DFN3Params` を書く。空の辞書は同じ既定値になる。公式の 1 step 宣言は既定値を明示している。

| 名前 | 公式値 | 制約 |
|---|---|---|
| `sr` | 48000 | 構築は他の値も受ける。公式データ手順は 48000 以外を拒否する |
| `fft_size` | 960 | モデル、DFStream、損失が同じ params を使う |
| `hop_size` | 480 | 同上 |
| `nb_erb` | 32 | 8 の倍数 |
| `nb_df` | 96 | |
| `norm_tau` | 1.0 | |
| `lsnr_max` | 35 | |
| `lsnr_min` | -15 | |
| `min_nb_freqs` | 2 | |
| `df_order` | 5 | |
| `df_lookahead` | 2 | |
| `conv_lookahead` | 2 | |
| `conv_ch` | 64 | |
| `emb_hidden_dim` | 256 | |
| `emb_num_layers` | 3 | 2 以上 |
| `enc_lin_groups` | 32 | |
| `lin_groups` | 16 | |
| `df_hidden_dim` | 256 | |
| `df_num_layers` | 2 | |
| `df_gru_skip` | `groupedlinear` | |
| `df_pathway_kernel_size_t` | 5 | |
| `enc_concat` | false | |
| `conv_kernel` | `[1, 3]` | |
| `convt_kernel` | `[1, 3]` | |
| `conv_kernel_inp` | `[3, 3]` | |
| `df_n_iter` | 1 | 1 以外は拒否 |
| `mask_pf` | false | |
| `pf_beta` | 0.02 | |

### TF-GridNet（MS-SNSD）

`variant` は `ms-snsd`。`init` は初期化名で、構造ではない。

| 名前 | 公式値 | 制約 |
|---|---|---|
| `n_fft` | 512 | 正の整数 |
| `stride` | 256 | 正の整数 |
| `n_layers` | 4 | 正の整数 |
| `lstm_hidden_units` | 128 | 正の整数 |
| `attn_n_head` | 4 | `emb_dim` を割り切る |
| `attn_approx_qk_dim` | 512 | 正の整数 |
| `emb_dim` | 32 | 正の整数 |
| `emb_ks` | 4 | 正の整数 |
| `emb_hs` | 4 | 正の整数 |
| `activation` | `prelu` | 空でない文字列 |
| `eps` | 1e-5 | 正の数 |

### TF-Locoformer（DNS）

`variant` は `dns`。`num_spk` は 1 だけ。

| 名前 | 公式値 | 制約 |
|---|---|---|
| `n_fft` | 256 | 正の整数 |
| `hop_length` | 128 | 正の整数 |
| `normalize_variance` | true | ネットワークの入力正規化。損失の正規化とは別 |
| `n_layers` | 6 | 正の整数 |
| `emb_dim` | 128 | `num_groups` で割り切れる |
| `norm_type` | `rmsgroupnorm` | |
| `num_groups` | 4 | 正の整数 |
| `tf_order` | `ft` | |
| `n_heads` | 4 | `attention_dim` を割り切る |
| `flash_attention` | false | |
| `attention_dim` | 128 | 正の整数 |
| `pos_enc` | `rope` | |
| `ffn_type` | `swiglu_conv1d` を 2 回 | 要素はすべて `swiglu_conv1d` |
| `ffn_hidden_dim` | `384, 384` | `ffn_type` と同じ長さ |
| `conv1d_kernel` | 4 | |
| `conv1d_shift` | 1 | |
| `dropout` | 0.0 | |
| `eps` | 1e-5 | 正の数 |

## Python に残すもの

| 項目 | 分類 | 理由 |
|---|---|---|
| MP-SENet の FFN GRU 隠れ次元 `2 * d_model` | B | チャネルから一意 |
| MP-SENet の双方向 GRU、FFN dropout 0、kernel `(2, 3)`、dilation `2**i` | C | TS-Conformer の式。`ts_dropout` は attention だけ |
| MP-SENet の Encoder 入力 2、Decoder 出力 1、sub-pixel `r=2`、InstanceNorm、PReLU | C | 複素スペクトル入出力の定義 |
| MetricDiscriminator の畳み込み形状、dropout 0.3、入力チャネル 2、sigmoid beta 1 | C | 公開 Discriminator の式。幅だけ `disc_dim` |
| MossFormerGAN の STFT を学習経路で 400 / 100 に固定 | C | `build()` が公式窓を上書きする |
| MossFormerGAN Discriminator の NISQA / AdaptCNN / Conformer 配置 | C | プール段数と conformer 幅 384 が結合しており、独立な幅にできない |
| SEMamba++ の周波数幅 100 / 50 / 25 | C | `n_fft` 400 に結び付いたリテラル |
| SEMamba++ の `scan_backend` | D | fused と reference は同じ重み |
| DeepFilterNet3 の `emb_gru_skip`、`emb_gru_skip_enc`、`lsnr_dropout` | C | 公開 ini にあるが DfNet は読まない。既定値以外は拒否する |
| TF-GridNet / TF-Locoformer の双方向 LSTM、3x3 畳み込み、RoPE theta、分散の下限、Macaron の 1/2 | C | ブロック内部の式 |

## 研究用の構造

1. 公式 YAML を別ファイルにコピーする。`experiment_claim` は `research` にする。
2. 上の表の値を変える。Generator と Discriminator があるモデルは、同じ構造をすべてのモジュールに書く。
3. 新しいディレクトリで `initialize` する。変える前の checkpoint を resume しない。
4. 構築できたあと、パラメータ数か `state_dict` の shape が公式構造と違うことを確認する。

`official_faithful` のまま幅や層数だけを変えると、MP-SENet と MossFormerGAN は構築時に拒否する。DeepFilterNet3 の `dfn3.online.official@1` と SEMamba++ の `semambapp.gsr.official@1` は、構造とは別に閉じたままである。

## 公式 checkpoint

公式値を明示した構築は、キーを省略して公開既定値で構築した state_dict と一致する。初期化の乱数順は変えない。GAN のコンストラクタが捨てる一時モジュールも残す。

公式の学習済みファイルを、構造を変えたモデルへ strict load しない。shape が違うキーで失敗する。公式値のモデルへの strict load は、ファイルのキーが公開構造と一致するときだけ成功する。

既に学習した checkpoint は、そのときの `config_sha256` を持つ。構造キーを足した YAML はハッシュが変わるので、その checkpoint の resume は拒否する。続きは、変える前の YAML と checkpoint の組で行う。新しい構造は新しい実験である。

## 確かめ方

`tests/experiments/test_model_architecture.py` が、公式値と省略時の state_dict、研究用の層数・幅・Discriminator 解像度、未知キー、割れないヘッド、`config_sha256` の変化、小さいネットの forward と backward を見る。公式初期化の回帰は、各 `tests/experiments/test_<model>.py` と `tests/numerics/` が既存の許容差で行う。CPU の成功は GPU の成功ではない。
