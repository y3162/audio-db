# audio-db

## 概要

音声強調の研究基盤である。複数の SE モデルを共通の手続きで学習し、固定した ASR の WER（`wer-norm-v1`、micro）で比べる。PESQ / STOI は補助指標で、ASR の重みは更新しない。Dataset の再現条件と Experiment の中断再開を保持する。

## 目次

手順

- [セットアップ・環境構築](docs/envs/SETUP.md)（[TSUBAME 4.0](docs/envs/TSUBAME.md)、[SHINODA-LAB](docs/envs/SHINODA_LAB.md)、[ASR 用の環境](docs/envs/ASR.md)）
- [データベースの操作](docs/db/SETUP.md)
- [学習・実験の実行と再開](docs/experiments/RUN.md)
- [テスト](docs/TESTING.md)
- [コマンド・チートシート](docs/CHEATSHEET.md)

仕様

- [Checkpoint（Latest、Archive、Selector、再開、初期化）](docs/experiments/CHECKPOINT.md)
- [Launcher（終了状態、停止、障害）](docs/experiments/LAUNCHER.md)
- [評価パイプラインと WER の完全性](docs/evaluation/PIPELINE.md)
- [公式指標での評価](docs/evaluation/OFFICIAL_EVALUATION.md)（仕様のみ。未実装）
- [レシピ JSON](docs/db/RECIPE.md)、[再現用 DB](docs/db/REPRODUCTION_DB_DESIGN.md)、[audio_files](docs/db/tables/AUDIO_FILES.md)
- [コーパスの索引](docs/db/CORPORA.md)、[コーパスの内訳](docs/db/DATASETS.md)、[学習スモークセット](docs/db/TRAINING_SETS.md)
- [音声強調 API](docs/se/ARCHITECTURE.md)、[音声強調モデル](docs/se/MODELS.md)、[音声認識モデル](docs/asr/MODELS.md)

記録

- [DB の配置と構築記録](docs/db/INVENTORY.md)
- [公式出力との数値比較](docs/se/VALIDATION.md)
- [PESQ の長さ依存の調査](docs/evaluation/PESQ_LENGTH_INVESTIGATION.md)
- [2026-11-04 のクリーン再構築](docs/envs/CLEAN_REBUILD.md)

エージェント向けの不変条件は [AGENTS.md](AGENTS.md) と `.agents/` にある。
