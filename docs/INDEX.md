# ドキュメント案内

概要は [README](../README.md)、全エージェント共通の制約は [AGENTS.md](../AGENTS.md)。実装上の前提と決定は [.agents/](../.agents/PROJECT.md) にある。詳細・数値は次の文書を正本とする。

## 学習・実験

| 知りたいこと | 参照先 |
|---|---|
| 5 モデルの公式学習との差、TF-GridNet の chunk iterator、SEMamba++ の蓄積 | [公式学習設定の監査](training/OFFICIAL_TRAINING_AUDIT.md) |
| 学習・checkpoint 保存と再開、データ制約、コマンド | [MP-SENet と共通 Trainer](training/MP_SENET.md)、[その他のモデル](training/OTHER_MODELS.md)、[TF-Locoformer](training/TF_LOCOFORMER.md) |
| TensorBoard の設定・比較・再生成 | [TensorBoard](training/TENSORBOARD.md) |
| A100 のメモリ・更新時間と条件 | [A100 学習コスト](training/GPU_COST.md) |
| H100 の更新時間、I/O、高速化、予算 | [性能](training/PERFORMANCE.md)、[GPU 予算](training/GPU_BUDGET.md) |
| 比較した条件・数値・制限・結果の保存先、未解決事項 | [学習検証](training/VERIFICATION.md)、[短い再開と Whisper 評価](training/RESIDUAL_20261104.md) |
| ジョブ継続の実装・未検証範囲 | [ジョブ継続の検証](training/JOB_CONTINUATION_AUDIT.md) |
| 実験案と評価順序 | [実験計画](training/EXPERIMENT_PLAN.md)（案と実装済み事項を区別） |
| Experiment Framework（Phase 1、CPU checkpoint の Phase 2-A、CPU / Gloo DDP の Phase 2-B、CUDA / NCCL / AMP FP16 の Phase 2-C、固定ペア Data System の Phase 3-B、chunk Data Pipeline の Phase 3-C、オンライン Data Pipeline の Phase 3-D、CPU / Gloo world size 2 Data System の Phase 3-E、平坦 Sampler の動的重みの Phase 3-F、MP-SENet Generator の Waveform L1 接続の Phase 4-A、TF-GridNet MS-SNSD の Phase 4-B、TF-Locoformer DNS の Phase 4-C、MP-SENet 公式 GAN の Phase 4-D、MossFormerGAN 公式 GAN の Phase 4-E を実装。Phase 4-F は 4 モデルの横断検証。DeepFilterNet3、SEMamba++、FSDP、sharded checkpoint、AMP BF16 は未実装） | [構成](training/experiment-framework/ARCHITECTURE.md)、[API](training/experiment-framework/PUBLIC_API.md)、[ライフサイクル](training/experiment-framework/COMPONENT_LIFECYCLE.md)、[状態](training/experiment-framework/STATE_AND_CHECKPOINT.md)、[Phase 2-A](training/experiment-framework/PHASE_2A_CHECKPOINT.md)、[Phase 2-B](training/experiment-framework/PHASE_2B_DDP.md)、[Phase 2-C](training/experiment-framework/PHASE_2C_CUDA_AMP.md)、[Phase 3-B](training/experiment-framework/PHASE_3B_FIXED_PAIR.md)、[Phase 3-C](training/experiment-framework/PHASE_3C_CHUNK.md)、[Phase 3-D](training/experiment-framework/PHASE_3D_ONLINE.md)、[Phase 3-E](training/experiment-framework/PHASE_3E_DISTRIBUTED_DATA.md)、[Phase 3-F](training/experiment-framework/PHASE_3F_WEIGHTED_SAMPLING.md)、[Phase 4-A](training/experiment-framework/PHASE_4A_MP_SENET_CONNECTION.md)、[Phase 4-B](training/experiment-framework/PHASE_4B_TF_GRIDNET_TRAINING.md)、[Phase 4-C](training/experiment-framework/PHASE_4C_TF_LOCOFORMER_TRAINING.md)、[Phase 4-D](training/experiment-framework/PHASE_4D_MP_SENET_GAN.md)、[Phase 4-E](training/experiment-framework/PHASE_4E_MOSSFORMERGAN_TRAINING.md)、[Phase 4-F](training/experiment-framework/PHASE_4F_INTEGRATION.md)、[Data 調査](training/experiment-framework/PHASE_3A_DATA_AUDIT.md)、[Data 構成](training/experiment-framework/DATA_ARCHITECTURE.md)、[Data API](training/experiment-framework/DATA_PUBLIC_API.md)、[Data 移行](training/experiment-framework/DATA_MIGRATION_PLAN.md)、[Data 受け入れ](training/experiment-framework/DATA_ACCEPTANCE_CRITERIA.md)、[移行](training/experiment-framework/MIGRATION_PLAN.md)、[受け入れ](training/experiment-framework/ACCEPTANCE_CRITERIA.md) |
| 6 モデルの公式再現: 再現対象、採用した方針 D-01〜D-09 | [TARGETS](training/reproduction/TARGETS.md)、[OVERVIEW](training/reproduction/OVERVIEW.md) |
| 公式再現のデータ設計（P1〜P4 実装済み。P5 以降は未実装） | [Pipeline 設計](training/reproduction/DATA_PIPELINE_DESIGN.md)、[実装計画](training/reproduction/IMPLEMENTATION_PLAN.md)、[P4 検証](training/reproduction/P4_VERIFICATION.md) |

## データと評価

- [再現用 DB の設計](db/REPRODUCTION_DB_DESIGN.md)（P1 のスキーマ、P2 の 4 Dataset、P4 の run 登録は実装済み）
- [コーパスと VoiceBank の分割](db/CORPORA.md)、[対応データセット](db/DATASETS.md)、[音声ファイル表](db/tables/AUDIO_FILES.md)
- [DB セットアップ](db/SETUP.md)、[音声生成レシピ](db/RECIPE.md)、[学習スモークセット](db/TRAINING_SETS.md)、[TSUBAME 初回構築の記録](db/TSUBAME_INITIAL.md)
- [評価パイプライン](evaluation/PIPELINE.md)、[公式評価の仕様（未実装）](evaluation/OFFICIAL_EVALUATION.md)、[PESQ の長さ依存と欠損](evaluation/PESQ_LENGTH_INVESTIGATION.md)

## モデル・環境

- SE: [設計](se/ARCHITECTURE.md)、[モデル・出典](se/MODELS.md)、[実装状況](se/IMPLEMENTATION_STATUS.md)、[推論検証](se/VALIDATION.md)
- ASR: [モデルと検証状況](asr/MODELS.md)
- 環境: [共通セットアップ](envs/SETUP.md)、[SHINODA-LAB](envs/SHINODA_LAB.md)、[TSUBAME](envs/TSUBAME.md)、[ASR](envs/ASR.md)、[クリーン再構築](envs/CLEAN_REBUILD.md)
