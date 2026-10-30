def create_corpora_schema(con, *, recreate: bool = False) -> None:
    con.execute("CREATE SCHEMA IF NOT EXISTS corpora;")
    if recreate:
        con.execute("DROP TABLE IF EXISTS corpora.audio_files;")

    con.execute("""
        CREATE TABLE IF NOT EXISTS corpora.audio_files (
            -- 共通
            path VARCHAR NOT NULL PRIMARY KEY,
            corpus VARCHAR NOT NULL,
            subset VARCHAR,
            relative_dir VARCHAR,
            file_name VARCHAR NOT NULL,
            extension VARCHAR,
            sample_rate INTEGER NOT NULL,
            frames BIGINT NOT NULL,
            channels INTEGER NOT NULL,

            -- LibriSpeech / LibriTTS
            -- path: {subset}/{speaker_id}/{chapter_id}/...
            -- flac/wav: {speaker_id}-{chapter_id}-{utterance_id}
            -- SPEAKERS.txt / speakers.tsv, CHAPTERS.txt, *.trans.txt
            speaker_id VARCHAR,
            chapter_id VARCHAR,
            utterance_id VARCHAR,
            segment_start_id VARCHAR,
            segment_end_id VARCHAR,
            speaker_name VARCHAR,
            speaker_gender VARCHAR,
            chapter_title VARCHAR,
            book_title VARCHAR,
            chapter_duration_min DOUBLE,
            transcript_path VARCHAR,

            -- VCTK
            -- path: wav48/{speaker_id}/{speaker_id}_{utterance_id}.wav
            -- speaker-info.txt, txt/{speaker_id}/...
            utterance_index VARCHAR,
            speaker_age INTEGER,
            speaker_accent VARCHAR,
            speaker_region VARCHAR,
            transcript_txt_path VARCHAR,
            transcript VARCHAR,

            -- CHiME3
            -- path: data/audio/16kHz/{track_type}/{condition}/...
            -- 例: F04_22GC0110_CAF.wav, BGD_..._CAF.CH2.wav
            track_type VARCHAR,
            condition_label VARCHAR,
            eval_split VARCHAR,
            noise_type VARCHAR,
            reverberation VARCHAR,
            chime_speaker_label VARCHAR,
            wsj_utterance_id VARCHAR,
            mic_channel VARCHAR,

            -- DEMAND
            -- path: {scene_code}/ch{NN}.wav（SCAFE, OOFFICE など）
            scene_code VARCHAR,
            scene_class VARCHAR,
            channel_label VARCHAR
        )
    """)
