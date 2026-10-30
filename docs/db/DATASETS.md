## CHiME3

#### 概要

> 実環境の雑音下における音声認識を目的としたコーパス。
> [Wall Street Journal](https://www.wsj.com/) の英文を、4 環境で読み上げた音声と、同じ環境の雑音を重畳した模擬音声からなる。

- [ ] 収録情報

> 16 kHz・モノラル・WAV。ファイル数は 122,277。
> パスは `data/audio/16kHz/{enhanced|isolated}/{split}_{noise}_{real|simu}/`。
> `enhanced` は発話あたり 1 ファイル。`isolated` はマイクごとで、real は CH0–CH6、simu は CH1–CH6。

- [ ] データセット

> 表は `enhanced` のファイル数。環境は bus、caf、ped、str。

|分割|real|simu|話者|
|:-|:-:|:-:|:-:|
|tr05|1,600|7,138|87|
|dt05|1,640|1,640|4|
|et05|1,320|1,320|4|
|合計|4,560|10,098|95|

> 話者は実収録が各分割 4 名、模擬の学習だけ 83 名。開発と評価の real / simu は同一話者。
> 開発・評価は同じ発話を 4 環境に分けている。ユニークな発話は開発が各 410、評価が各 330。

---

## DEMAND

#### 概要

> 実環境の背景雑音を、16 チャンネルで収録したコーパス。
> 音声発話は含まず、雑音付与や雑音抑圧の評価に使われる。

- [ ] 収録情報

> 18 シーン × 16 チャンネル（`ch01`–`ch16`）。各ファイルはモノラル WAV で、長さは 300 秒。
> サンプリング周波数は `SCAFE` のみ 48 kHz、それ以外は 16 kHz。量子化は 16 bit。

- [ ] データセット

|区分|シーン|ファイル|
|:-|:-:|:-:|
|domestic|DKITCHEN, DLIVING, DWASHING|各 16|
|nature|NFIELD, NPARK, NRIVER|各 16|
|office|OHALLWAY, OMEETING, OOFFICE|各 16|
|public|PCAFETER, PRESTO, PSTATION|各 16|
|street|SCAFE, SPSQUARE, STRAFFIC|各 16|
|transport|TBUS, TCAR, TMETRO|各 16|
|合計|18 シーン|288|

---

## LibriSpeech

#### 概要

> [LibriVox](https://librivox.org/) のパブリックドメイン音読を元にした、英語の読み上げ音声コーパス。
> 自動音声認識の学習と評価を目的とする。話者は分割のあいだで重複しない。

- [ ] 収録情報

> 16 kHz・モノラル・FLAC。ファイル数は 292,367。
> パスは `{subset}/{speaker_id}/{chapter_id}/{speaker_id}-{chapter_id}-{utterance_id}.flac`。
> `clean` は比較的明瞭で米国英語寄りの音声、`other` は認識が難しいと自動判定された音声。

- [ ] データセット

|分割|発話|話者|章|
|:-|:-:|:-:|:-:|
|train-clean-100|28,539|251|585|
|train-clean-360|104,014|921|2,097|
|train-other-500|148,688|1,166|2,784|
|dev-clean|2,703|40|97|
|dev-other|2,864|33|91|
|test-clean|2,620|40|87|
|test-other|2,939|33|90|
|合計|292,367|2,484|5,831|

> 話者は女性 1,201、男性 1,283。

---

## LibriTTS

#### 概要

> LibriSpeech と同じ原音源から、音声合成向けに作り直した英語の読み上げコーパス。
> 文の境界で分割し、サンプリング周波数を 24 kHz にそろえている。大文字と句読点を残した書き起こしを含む。

- [ ] 収録情報

> 24 kHz・モノラル・WAV。ファイル数は 375,086。
> パスは `{subset}/{speaker_id}/{chapter_id}/{speaker_id}_{chapter_id}_{start}_{end}.wav`。
> `clean` は WADA-SNR が 20 dB 以上、`other` は 0 dB 以上の発話だけを残している。
> 話者は LibriSpeech と同じ分割に入るが、雑音などの除外で人数と発話数は一致しない。

- [ ] データセット

> 時間はコーパス付属 README の集計。発話数と話者数はデータベースの件数で、話者数は README と一致する。

|分割|時間|発話|話者|
|:-|:-:|:-:|:-:|
|train-clean-100|53.78|33,236|247|
|train-clean-360|191.29|116,500|904|
|train-other-500|310.08|205,044|1,160|
|dev-clean|8.97|5,736|40|
|dev-other|6.43|4,613|33|
|test-clean|8.56|4,837|39|
|test-other|6.69|5,120|33|
|合計|585.80|375,086|2,456|

> 話者は女性 1,185、男性 1,271。

---

## VCTK

#### 概要

> CSTR VCTK Corpus（version 0.80）。英語の多話者読み上げ音声。
> 話者適応型の音声合成を目的に、新聞（Herald Glasgow）の文章に加え、全話者共通の Rainbow Passage と elicitation paragraph を収録している。

- [ ] 収録情報

> エジンバラ大学の半無響室で、DPA 4035 により 96 kHz・24 bit で収録し、48 kHz・16 bit のモノラル WAV に変換されている。
> パスは `wav48/{speaker_id}/{speaker_id}_{utterance_index}.wav`。
> 話者 108 名、発話 44,085。話者あたり 231–503 発話（中央値 417）。発話番号は 503 種。

- [ ] 話者

> `speaker-info.txt` は 109 名。このツリーでは `p315` の音声がなく、`p280`（410 発話）は `speaker-info.txt` にない。
> 情報が揃う 107 名は、女性 61、男性 46。年齢は 18–38 歳（平均 22.7）。

|アクセント|話者|
|:-|:-:|
|English|33|
|American|21|
|Scottish|19|
|Irish|9|
|Canadian|8|
|NorthernIrish|6|
|SouthAfrican|4|
|Indian|3|
|Australian English|2|
|Welsh|1|
|NewZealand English|1|
