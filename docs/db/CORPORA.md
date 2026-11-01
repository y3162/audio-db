# コーパス

索引の列は [tables/AUDIO_FILES.md](tables/AUDIO_FILES.md)。標準のリンクと登録方針は `src/data/corpora_catalog.py` の 12 コーパスである。索引は学習に使うかどうかでファイルを落とさない。sample rate はファイルの値のまま保存する。公式 transcript がある音声は原文を保存し、WER 正規化は評価時の別処理である。

親ディレクトリは `python -m src.data.link_corpora --dataset-root <dir>` で指定する。未指定時は環境変数 `AUDIO_DB_DATASET_ROOT`、それも無いときは `/net/spring/work/seki/database`。

| ディレクトリ | リンク名 | 役割 |
|---|---|---|
| CHiME3 | CHiME3 | 評価用の雑音音声。CH0 は clean ではない |
| DEMAND | DEMAND | 環境雑音 |
| DNS5 | DNS5 | SE 用の fullband 素材。公式の混合評価セットは作っていない |
| LibriSpeech | LibriSpeech | クリーン音声と transcript |
| LibriTTS | LibriTTS | クリーン音声と transcript |
| OpenSLR26 | OpenSLR26 | シミュレート RIR |
| OpenSLR28 | OpenSLR28 | RIR と雑音 |
| URGENT2024 | URGENT2024 | チャレンジ音声。blind test に clean 参照は無いので評価セットは作っていない |
| VCTK | VCTK | クリーン音声 |
| VoiceBank+DEMAND | VoiceBank-DEMAND | ファイル名で対応する clean/noisy |
| WHAM48 | WHAM48 | 48 kHz 雑音。単体では混合評価セットにしない |
| musan | musan | music / noise / speech |

VoiceBank-DEMAND の公式 test は `clean_testset_wav` と `noisy_testset_wav`（824 件）。28 話者 train は 11,572 件で、p286 と p287 の 872 件を validation、残り 10,700 件を train にする。test 話者とは重ならない。
