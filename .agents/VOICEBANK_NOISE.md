# VoiceBank の雑音と SNR シフト

件数、ハッシュ、ファイルの配置は `docs/db/INVENTORY.md`。実行時の場所と SHA-256 は `configs/data/voicebank_snr_shift.yaml`。詳細な中間 JSONL と報告書は、データセットルートの `investigations/voicebank-20261112/` にあり、ここへは複製しない。

## コーパス

- DEMAND、DEMAND48、Babble、SSN は別コーパスである。既存 DEMAND の実体と `data/corpora/DEMAND` のリンク先は変えない。
- DEMAND48 は、取得済みの 48 kHz ZIP にある 13 シーンだけである。DWASHING、NFIELD、NPARK、NRIVER、OHALLWAY は無い。シーン索引は DEMAND と同じパーサーで、ディレクトリだけが違う。
- Babble と SSN は、それぞれ直下に 1 ファイルの汎用索引である。`subset` は null。由来の ZIP はデータセットルートの `_provenance/` にあり、索引しない。

## レシピ

- 再生の正本は `recipes/voicebank-snr-shift-v1/`。`depends.json` が同じディレクトリのレシピ、パス一覧、共通定義、件数のパスと SHA-256 を指定する。
- アルゴリズムは `vctk-aligned-p56-peak-shift-v1`。式は v1 の `common.json` にあり、`shifted = g * clean + shifted_noise_gain * noise[start:end]`、参照は `g * clean`、`snr_shift_db` は -15 である。
- これは全区間 RMS の `additive_noise` バージョン 1 ではない。推定した区間とゲインであり、公式生成器の出力と同一だと扱わない。`eval.cases` に入れない。シフト後の WAV をこのレシピの登録のために作らない。
- パスはリポジトリルートからの相対パスである。clean と noisy は `data/corpora/VoiceBank-DEMAND`、VCTK は `data/corpora/VCTK`、雑音は `data/corpora/DEMAND48`、`data/corpora/Babble`、`data/corpora/SSN` である。`source_vctk` は名前に `path` が無いが、VCTK のファイルパスである。
- DEMAND の行は、切り出し位置とゲインの一部を JSON の数値ではなく十進文字列で持つ。整数または有限な数として読めるものだけを使い、null や空を 0 にしない。
- リポジトリ直下に `research_*` を置かない。調査当時のファイル名は investigation の中に残っている。

## 技術的知見

再生は、レシピに保存された `g`、`beta`、`noise_start_48k`、`noise_end_exclusive_48k` だけを使う。P.56 や seed から区間とゲインを計算し直さない。

P.56 は `scripts/data/voicebank/asl_P56_fast.py` の method B である。`asl_P56.py` は別実装である。ゲイン推定に使った入力は、配布 clean に相関整合し、そのピークへ合わせた VCTK の float である。配布 PCM16 を直接入れた値とは別である。investigation の `sv-p56.c` は STL 系の C コピーで、Python との差は活動カウントの +1 と、補間で境界を更新する点にある。DEMAND 10,070 件のうち 122 件で差が 0.05 dB を超え、最大は約 0.115 dB である。これは ITU-T P.56 への適合宣言ではない。

`beta` は、その P.56 の活動音声レベル、切り出した雑音の RMS、公式ログ SNR からの推定である。`shifted_noise_gain = g * beta * 10^(15/20)`。`target_log_snr_db` は公式ログ SNR から 15 を引いたラベルであり、成分 SNR の差が -15 dB であることを指す。全区間 RMS の比がその値になる、という意味ではない。

段階的ピーク抑制は、混合の絶対ピークが 1 を超えるあいだ 1.1、1.2、1.3、1.4 で順に割る。4 段で 1 以下にならなかった候補は採用しない。`g != 1` は 55 件である。シフト後の出力は IEEE float で、再ピーク抑制もクリップも PCM16 量子化もしない。ピークは 1 を超え得る。

`noisy - clean = (g - 1) * clean + g * beta * noise + 量子化誤差` である。`g != 1` のとき、単純な差分は雑音そのものではない。DEMAND 10,070 件では、推定した PCM と公式 noisy のサンプル差が -1、0、+1 LSB に収まった。発話単位でビットが完全に一致した件数は 0 である。1 LSB の単一原因は未特定である。

公式生成器のコードは確認していない。切り出し位置は公式 noisy を使った探索であり、独立したブラインド試験ではない。競合位置の探索は局所であり、全位置の網羅ではない。このレシピを公式混合の再生成と呼ばない。

## 再実行

リポジトリルートで実行する。データセットルートは `--dataset-root` または `AUDIO_DB_DATASET_ROOT` で渡す。Git 管理するスクリプトは `scripts/data/voicebank/`。

`verify.py` は v1 と `depends.json` に指定された共通定義を読み、正本には書かない。`residual_check.py` は既存の出力を拒否する。`search.py` と `assemble.py` は既存の正本と investigation を保護する。`generate_one.py` は既存の出力を上書きしない。
