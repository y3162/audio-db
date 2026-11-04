# ドキュメント案内

概要は [README](../README.md)、全エージェント共通の制約は [AGENTS.md](../AGENTS.md)。実装上の前提と決定は [.agents/](../.agents/PROJECT.md) にある。詳細・数値は次の文書を正本とする。

## 学習・実験

| 知りたいこと | 参照先 |
|---|---|
| 5 モデルの公式学習との差、TF-GridNet の chunk iterator、SEMamba++ の蓄積 | [公式学習設定の監査](training/OFFICIAL_TRAINING_AUDIT.md) |
| 学習・checkpoint 保存と再開、データ制約、コマンド | [MP-SENet と共通 Trainer](training/MP_SENET.md)、[その他のモデル](training/OTHER_MODELS.md) |
| A100 のメモリ・更新時間と条件 | [A100 学習コスト](training/GPU_COST.md) |
| H100 の更新時間、I/O、高速化、予算 | [性能](training/PERFORMANCE.md)、[GPU 予算](training/GPU_BUDGET.md) |
| 比較した条件・数値・制限・結果の保存先、未解決事項 | [学習検証](training/VERIFICATION.md)、[短い再開と Whisper 評価](training/RESIDUAL_20261104.md) |
| ジョブ継続の実装・未検証範囲 | [ジョブ継続の検証](training/JOB_CONTINUATION_AUDIT.md) |
| 実験案と評価順序 | [実験計画](training/EXPERIMENT_PLAN.md)（案と実装済み事項を区別） |

## データと評価

- [コーパスと VoiceBank の分割](db/CORPORA.md)、[対応データセット](db/DATASETS.md)、[音声ファイル表](db/tables/AUDIO_FILES.md)
- [DB セットアップ](db/SETUP.md)、[音声生成レシピ](db/RECIPE.md)、[学習スモークセット](db/TRAINING_SETS.md)、[TSUBAME 初回構築の記録](db/TSUBAME_INITIAL.md)
- [評価パイプライン](evaluation/PIPELINE.md)、[PESQ の長さ依存と欠損](evaluation/PESQ_LENGTH_INVESTIGATION.md)

## モデル・環境

- SE: [設計](se/ARCHITECTURE.md)、[モデル・出典](se/MODELS.md)、[実装状況](se/IMPLEMENTATION_STATUS.md)、[推論検証](se/VALIDATION.md)
- ASR: [モデルと検証状況](asr/MODELS.md)
- 環境: [共通セットアップ](envs/SETUP.md)、[SHINODA-LAB](envs/SHINODA_LAB.md)、[TSUBAME](envs/TSUBAME.md)、[ASR](envs/ASR.md)、[クリーン再構築](envs/CLEAN_REBUILD.md)
