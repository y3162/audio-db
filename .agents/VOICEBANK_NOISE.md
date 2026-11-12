# VoiceBank の雑音と SNR シフト

件数、ハッシュ、ファイルの配置は `docs/db/INVENTORY.md`。実行時の場所と SHA-256 は `configs/data/voicebank_snr_shift.yaml`。ここには、その設定を誤って使わないための条件だけを書く。

## コーパス

- DEMAND、DEMAND48、Babble、SSN は別コーパスである。既存 DEMAND の実体と `data/corpora/DEMAND` のリンク先は変えない。
- DEMAND48 は、取得済みの 48 kHz ZIP にある 13 シーンだけである。DWASHING、NFIELD、NPARK、NRIVER、OHALLWAY は無い。シーン索引は DEMAND と同じパーサーで、ディレクトリだけが違う。
- Babble と SSN は、それぞれ直下に 1 ファイルの汎用索引である。`subset` は null。由来の ZIP はデータセットルートの `_provenance/` にあり、索引しない。

## レシピ

- アルゴリズムは `vctk-aligned-p56-peak-shift-v1`。式は共有ストレージの `common.json` にあり、`shifted = g * clean + shifted_noise_gain * noise[start:end]`、参照は `g * clean`、`snr_shift_db` は -15 である。
- これは全区間 RMS の `additive_noise` バージョン 1 ではない。推定した区間とゲインであり、公式生成器の出力と同一だと扱わない。`eval.cases` に入れない。シフト後の WAV をこのレシピの登録のために作らない。
- 正本の JSONL はデータセットルートの `recipes/voicebank-snr-shift-v1/`。パスはリポジトリルートからの相対パスのままである。共有ストレージのコピーだけでは参照先を持たない。
- 解決先は、Git 管理外の `research_voicebank_noise_20261112/sources_48k/` と `research_voicebank_snr_shift_20261112/` の babble・ssn、および `data/corpora/VoiceBank-DEMAND` と `data/corpora/VCTK` である。`source_vctk` は名前に `path` が無いが、VCTK のファイルパスである。
- DEMAND の行は、切り出し位置とゲインの一部を JSON の数値ではなく十進文字列で持つ。整数または有限な数として読めるものだけを使い、null や空を 0 にしない。
- 研究ディレクトリはディスク上の調査原資料である。Git には入れず、消さない。

## 再実行

リポジトリルートで実行する。Git 管理するスクリプトは `scripts/data/voicebank/`。P.56 method B は `asl_P56_fast.py`（numba）で、`asl_P56.py` はその別実装である。`search.py`、`assemble.py`、`verify.py`、`residual_check.py` は上記の研究ディレクトリにある先行結果と音声を読む。`generate_one.py` は 1 発話の候補を、既存ファイルを上書きせずに書く。

同じ処理の元スクリプトは各 `research_voicebank_*/` に残している。Git 管理側を変えても、そちらは調査当時のファイルとして残す。
