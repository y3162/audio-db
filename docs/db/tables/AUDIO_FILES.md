# `corpora.audio_files` の列

標準 14 コーパスの対応。○ はパーサーが値を設定し得る列で、すべての行に値が入る保証ではない。話者 metadata、transcript、パス由来の情報などは元データやファイル名により null になる。— は該当パーサーが設定しない列。パス列は `data/corpora/` からの相対パスである。索引方法と評価セットとの違いは [コーパス](../CORPORA.md)。

「汎用 8」は DNS5、DNS2020、MS-SNSD、OpenSLR26、OpenSLR28、URGENT2024、WHAM48、musan。VoiceBank-DEMAND は同じ汎用パーサーを使うが、話者・発話と対応する transcript も読み取るため別列とした。汎用索引の `subset` はリンク直下のディレクトリ名であり、公式分割を意味するとは限らない。CHiME3 の transcript はこの表ではなく公式 JSON 注釈から評価セットへ登録する。

| カラム | CHiME3 | DEMAND | LibriSpeech | LibriTTS | VCTK | VoiceBank-DEMAND | 汎用 8 |
|---|---|---|---|---|---|---|---|
| `path` | ○ | ○ | ○ | ○ | ○ | ○ | ○ |
| `corpus` | ○ | ○ | ○ | ○ | ○ | ○ | ○ |
| `subset` | ○ | ○ | ○ | ○ | ○ | ○ | ○ |
| `relative_dir` | ○ | ○ | ○ | ○ | ○ | ○ | ○ |
| `file_name` | ○ | ○ | ○ | ○ | ○ | ○ | ○ |
| `extension` | ○ | ○ | ○ | ○ | ○ | ○ | ○ |
| `sample_rate` | ○ | ○ | ○ | ○ | ○ | ○ | ○ |
| `frames` | ○ | ○ | ○ | ○ | ○ | ○ | ○ |
| `channels` | ○ | ○ | ○ | ○ | ○ | ○ | ○ |
| `speaker_id` | — | — | ○ | ○ | ○ | ○ | — |
| `chapter_id` | — | — | ○ | ○ | — | — | — |
| `utterance_id` | — | — | ○ | ○ | ○ | ○ | — |
| `segment_start_id` | — | — | — | ○ | — | — | — |
| `segment_end_id` | — | — | — | ○ | — | — | — |
| `speaker_name` | — | — | ○ | ○ | — | — | — |
| `speaker_gender` | — | — | ○ | ○ | ○ | — | — |
| `chapter_title` | — | — | ○ | ○ | — | — | — |
| `book_title` | — | — | ○ | ○ | — | — | — |
| `chapter_duration_min` | — | — | ○ | ○ | — | — | — |
| `transcript_path` | — | — | ○ | ○ | — | — | — |
| `utterance_index` | — | — | — | — | ○ | — | — |
| `speaker_age` | — | — | — | — | ○ | — | — |
| `speaker_accent` | — | — | — | — | ○ | — | — |
| `speaker_region` | — | — | — | — | ○ | — | — |
| `transcript_txt_path` | — | — | — | — | ○ | — | — |
| `transcript` | — | — | ○ | ○ | ○ | ○ | — |
| `track_type` | ○ | — | — | — | — | — | — |
| `condition_label` | ○ | — | — | — | — | — | — |
| `eval_split` | ○ | — | — | — | — | — | — |
| `noise_type` | ○ | — | — | — | — | — | — |
| `reverberation` | ○ | — | — | — | — | — | — |
| `chime_speaker_label` | ○ | — | — | — | — | — | — |
| `wsj_utterance_id` | ○ | — | — | — | — | — | — |
| `mic_channel` | ○ | — | — | — | — | — | — |
| `scene_code` | — | ○ | — | — | — | — | — |
| `scene_class` | — | ○ | — | — | — | — | — |
| `channel_label` | — | ○ | — | — | — | — | — |
