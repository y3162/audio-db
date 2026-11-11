# コーパス

再現実験の Dataset と Protocol は `configs/data/datasets/` と `configs/data/protocols/` にある。学習が読むのはスナップショットで、選び方は [学習](../experiments/RUN.md)。

標準リンクと索引対象は `src/data/corpora_catalog.py` の 14 コーパス。索引は学習に使うかどうかを理由に音声を除外せず、元の sample rate・フレーム数・チャンネル数を保存する。列の意味と各コーパスの対応は [audio_files](tables/AUDIO_FILES.md)、既存コーパスの収録条件と内訳は [データセット](DATASETS.md)。TSUBAME 初回構築の件数と、その後の再索引は [配置と記録](INVENTORY.md) にある。初回の件数は、そのとき取得した 12 コーパスの記録であり、現在の 14 コーパス DB の全件ではない。

## リンクと索引

`python -m src.data.link_corpora --dataset-root <dir>` または `AUDIO_DB_DATASET_ROOT` でコーパスの親ディレクトリを指定する。引数は環境変数より優先される。個別の `--corpus-path NAME=PATH` または `AUDIO_DB_CORPUS_<SOURCE>` も指定でき、個別指定は親ディレクトリより優先される。全選択対象に個別パスを指定する場合や `--allow-partial` の場合は親ディレクトリを省略できる。既定では不足・競合を事前検査し、不足があればリンクを作らず、競合する既存リンクを変更しない。`--dry-run` は作成しない。

`make db-ingest` は既存の `corpora.audio_files` を削除して作り直す。ユーザーの明示指示と DB の退避なしに実行しない。セット登録と staging の手順は [DB セットアップ](SETUP.md)。

## データセットごとの索引と用途

下表の「ソース」は親ディレクトリ直下の名前、「リンク」は `data/corpora/` 以下の名前。索引されたことと、名前付き評価セットが登録済みであることは別である。`configs/db/init.yaml` のセットは入力がないと skip され得る。

| ソース | リンク | 内容・索引上の注意 |
|---|---|---|
| CHiME3 | `CHiME3` | 実録・模擬の雑音下音声。`data/audio/16kHz/` の track・条件・マイク等を索引し、評価時の transcript は公式 JSON 注釈の `prompt` から登録する。CH0 は clean ではなく、real・simu と元 WSJ は時間整合ペアとみなさない。`tr05_org` は clean の元素材である。 |
| DEMAND | `DEMAND` | 環境雑音。シーン名、シーン分類、チャンネルを索引する。混合条件は登録するレシピに残す。 |
| DNS5 | `DNS5` | DNS Challenge 5 の fullband 音声・雑音素材。汎用索引を使う。このリポジトリでは単一の公式混合評価セットを定義していない。 |
| DNS2020 | `DNS2020` | DNS Challenge Interspeech 2020 の clean・noise・synthetic test・blind test。汎用索引を使う。`subset` はリンク直下の `datasets` であり、clean / noise / test の区別は下位ディレクトリに残る。blind test に clean 参照は無い。 |
| MS-SNSD | `MS-SNSD` | 16 kHz の clean と noise。汎用索引を使う。`clean_train`、`clean_test`、`noise_train`、`noise_test` が `subset` になる。混合済み noisy は同梱されていない。 |
| LibriSpeech | `LibriSpeech` | 英語読み上げ音声。subset・話者・章・発話と章単位の transcript を索引する。`test-clean` は評価用で、学習・検証には使わない。 |
| LibriTTS | `LibriTTS` | 音声合成向けの英語読み上げ音声。subset・話者・章・区間と章単位の transcript を索引する。LibriSpeech と原音源を共有するため、分割間の重複に注意する。 |
| OpenSLR26 | `OpenSLR26` | シミュレート RIR。汎用索引を使う。RIR としての利用条件は生成レシピに明示する。 |
| OpenSLR28 | `OpenSLR28` | RIR と雑音。汎用索引を使うため、パスを確認して RIR と雑音を選択する。 |
| URGENT2024 | `URGENT2024` | チャレンジ音声。汎用索引を使う。blind test に clean 参照はなく、当該セットの参照付き評価セットは作っていない。 |
| VCTK | `VCTK` | 多話者の読み上げ音声。`wav48/<speaker>/` の音声、話者情報と、存在する `txt/<speaker>/` の transcript を索引する。現在リンクされているデータの README は version 0.80 を示す一方、カタログの注記は 0.92 なので、別環境では実体とパーサーの期待する配置を確認する。 |
| VoiceBank+DEMAND | `VoiceBank-DEMAND` | clean/noisy の対応ペア。汎用索引に加え、ファイル名から話者・発話を取り、対応 txt があれば transcript を保存する。SNR は推定しない。分割は下記。 |
| WHAM48 | `WHAM48` | 48 kHz 雑音。汎用索引を使う。雑音単体を音声混合の評価セットとはみなさない。 |
| musan | `musan` | music / noise / speech の素材。汎用索引を使い、ディレクトリを確認して用途を選択する。 |

2026-11-04 の再索引の件数、退避ファイル、指紋については [配置と記録](INVENTORY.md) の「2026-11-04 再索引」。

専用パーサーを持たない DNS5、DNS2020、MS-SNSD、OpenSLR26、OpenSLR28、URGENT2024、VoiceBank-DEMAND、WHAM48、musan は `.wav` / `.flac` を再帰走査する。`_archives`、`__MACOSX`、`.git` は除外する。`subset` はリンク直下の最初のディレクトリ名であり、ファイルが直下なら null。公式 train / validation / test を自動判定する列ではない。読み取れない音声は除外し、`data/staging/ingest/unreadable.jsonl` に理由を記録する。2026-11-12 にはこのファイルは無かった（[配置と記録](INVENTORY.md)）。VoiceBank 以外の汎用索引では話者や transcript を推定しない。

公式 transcript の原文と WER 正規化は別である。LibriSpeech、LibriTTS、VCTK、VoiceBank-DEMAND は対応テキストを読み取れる場合に原文を索引する。CHiME3 の transcript は `corpora.audio_files` ではなく上記の公式注釈から評価セットに登録する。

## VoiceBank-DEMAND の分割

公式 test は `clean_testset_wav` と `noisy_testset_wav` の 824 ペア。公式 28 話者 train の 11,572 ペアのうち、p286 と p287 の 872 ペアを validation、残り 10,700 ペアを train とする。test 話者とは重ならない。test のセット定義は `configs/db/init.yaml`。学習用 DB と staging の場所は [配置と記録](INVENTORY.md)。28 話者と 56 話者の train 素材は別に索引されるが、上記の学習分割は 28 話者版に基づく。
