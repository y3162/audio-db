# 学習停止ポリシー

`components` に `role: stopping` を 1 つ書くと、Trainer が検証境界で発行した指標を独立した停止ポリシーに渡す。省略した実験の動作は従来どおり。保存と保持の方針は別の [Selector](CHECKPOINT.md) が決める。

```yaml
components:
  early_stop:
    role: stopping
    class: src.experiments.stopping.PatienceStopPolicy
    kwargs:
      metric: validation.loss
      mode: min
      patience: 10
      min_delta: 0.001
      start_after: 5
```

`metric` は `source.name`。`MetricEvent` は source と logical step、名前空間を付けない指標名と値のマップ、任意の event ID と期待指標名の集合を持つ。期待指標名を明示したイベントは、集合にない指標にとって対象外である。省略時は同じ source の監視指標があることを期待する。source は `train`、`train_eval`、`validation` を区別する。現在の 6 公式 Trainer が配送するのは validation のみ。共通 `StandardTrainer` は `train.mean_rank_loss` を配送する。これは各 rank の更新時損失の等重み平均で、サンプル単位の全体平均ではない。モデル固有の GAN 損失を同じ名前にまとめない。train_eval の Evaluator は未実装。各 Trainer が配送しない source は構築時に拒否する。Selector には従来どおり `loss` など名前空間なしの名前を渡す。Trainer と評価器ごとに提供する名前は異なる。検証を実行しなかった境界は観測に数えない。検証を実行したイベントで対象指標が欠けているか非有限なら失敗する。

`mode` は `min` または `max`。`start_after` は監視対象の有効な観測値を初めに何回読み飛ばすかであり、step や epoch ではない。最初の監視対象値を best とする。以後の改善は min では `best - value > min_delta`、max ではその逆であり、同値と閾値ちょうどは改善ではない。連続した非改善が正の整数 `patience` に到達すると停止する。対象外 source は数えない。

`StopPolicy` は [Checkpoint](CHECKPOINT.md) のクラス ID を具象クラス自身に宣言し、`observe(MetricEvent) -> StopDecision`、`state_dict()`、`load_state_dict()` を実装する。評価計算、モデル、Optimizer、保存方針は扱わない。`TrainingControl` は全 rank でイベントを配送し、判定と状態を比較する。Trainer は検証・scheduler・公式選択の完了後に配送し、その後 Selector に境界を渡す。停止時には次の更新に進まず、final を offer して終了する。最終状態が Latest に保存されない Selector では停止を失敗として扱う。現在、制御層を明示的にサポートする Trainer だけに設定でき、複数ポリシーの合成は未実装。

状態にはポリシーの best、best の logical step、読み飛ばしを含む有効観測回数、連続非改善回数、停止状態と、制御層の直近イベントを含む。checkpoint の通常の状態検査と復元に参加し、設定も実験同一性に入る。再開でポリシーや設定を変えること、旧 checkpoint に新規ポリシーを加えることはできない。`boundary_pending` からの再開では未完了の境界を処理し、完了済みの境界は再観測しない。

共通ポリシーと公式 Trainer 固有の停止のどちらかが成立すると終了する。共通ポリシーが成立した場合は `early_stopping` と記録し、Launcher は全 rank の正常終了と最終保存を確認して `COMPLETED` とする。`complete.json` と `status.json` に停止理由とポリシー状態・直近イベントが入る。既存の公式の停止、scheduler、推論用重みの選択は置き換えない。`official_faithful` の Trainer claim または `official_reproduction` の dataset claim との併用は拒否する。検証素材の選択来歴、`selection_uses_test` と不完全な来歴の扱いは変えない。
