# 音楽パフォーマンスへの応用のための脳波を用いた画像生成のプロトタイピング

Prototyping of EEG generative image system for musical performance (Data-Driven Art 2020)

2020-01-26
Atsuya Kobayashi Keio University SFC, t16366ak@sfc.keio.ac.jp

## 1. 背景

脳活動を読み取る技術の、音楽パフォーマンスへの応用について

### 1.1 EEG (brain wave) and AI (machine learning, deep neural network)

非侵襲の脳情報を用いたヒトの認知のデコーディングに関する研究は多くあり、運動想起のデコーディング [6] や音想起のデコーディングに関する研究 [4] では脳波信号に対して行列変換による特徴量抽出を行ったのち、機械学習による分類を行なっている。また、深層学習を用いて言語情報をデコーディングする [1] 研究など、機械学習・深層学習技術と脳情報計測による情報抽出に関する研究は多く、実用化も勧められている。

### 1.2 EEG (brain wave) and performance

音楽家へのニューロフィードバックにより低周波成分を変調させパフォーマンスの質を向上する研究 [3] など、パフォーマンスを向上させるための神経科学的調査にEEGが用いられる例が多く、アート・音楽パフォーマンス作品自体への応用事例としては、EEGsynth [5] があり、このプロジェクトでは脳波を用いた音の合成やビジュアルエフェクトを可能としている。

## 2. Idea and System Overview

脳波によりパフォーマンスの質を向上させる・脳波を用いてパフォーマンスに変調をかけるといったアプローチは多く存在するが、パフォーマンス中に脳波情報からダイレクトに作品を作成するといったアプローチは少ない。そこで、 [2] をもとに、音楽パフォーマンス中にパフォーマーが思い描いたアルバムジャケットを生成し、映像表現・映像効果へ応用するといった手法を提案し、プロトタイピングを行う。

### 2.1 EEG Data Acquisition and Classification

簡易ワイヤレス脳波計である [Open BCI](https://openbci.com/)によって脳波の測定を行う。10クラスのサンプル画像データ用意し、画像視聴時の脳活動を計測する。計測された脳波の周波数特性を特徴量として、10クラス分類タスクをPyTorchで定義したMLPで行う。

### 2.2 Genarative Model and Dataset

archive.orgが提供しているAlbum Cover Dataset ( [One Million Audio Cover Images for Research](https://archive.org/details/audio-covers)) を用いる。生成モデルはエンコーダとデコーダに畳み込みニューラルネットワークを用いた変分オートエンコーダモデルをPyTorchで定義し、上記のデータセットで学習する。32次元の潜在変数$z$から分布のパラメータをサンプリングし、アルバム画像を生成する。

### 2.3. Web based Visualization

今回はシステムのテストを行うための環境構築も兼ねて、Webベースのリアルタイム可視化環境を構築した。node.js (express) を用いてwebアプリケーションサーバーを立ち上げ、Web socket経由での通信を行う。クライアントサイドではOSC (Open Sound Control) 経由でOpenBCIで計測した脳波情報をサーバーへ送信し、サーバ側でその脳波情報から画像を生成し、クライアントサイドへ再度送信する。

## 3. Discussion

- 脳波からの生成結果に対する被験者の評価が必要
- 脳波情報の限界
- パフォーマンス中の脳波中のアーティファクトの除去

## 4. Conclusion

今回のプロトタイピングでは、OpenBCIからの周波数情報のOSC送信とその特徴量に基づくアルバム画像の生成、生成画像の送受信と表示までを実装した。今後は本システムを用いて、被験者による生成画像の評価を行うことで、生成精度の向上を目指す。

## 5. References

1. Hale, John, Chris Dyer, Adhiguna Kuncoro, and Jonathan Brennan. 2018. “Finding Syntax in Human Encephalography with Beam Search.” In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). Association for Computational Linguistics. https://doi.org/10.18653/v1/p18-1254.
1. Rashkov, Grigory, Anatoly Bobe, Dmitry Fastovets, and Maria Komarova. 2019. “Natural Image Reconstruction from Brain Waves: A Novel Visual BCI System with Native Feedback.” Cold Spring Harbor Laboratory. https://doi.org/10.1101/787101.
1. Gruzelier, John. 2011. “Enhancing Imaginative Expression in the Performing Arts with EEG-Neurofeedback.” In Musical ImaginationsMultidisciplinary Perspectives on Creativity, Performance and Perception, 332–50. Oxford University Press. https://doi.org/10.1093/acprof:oso/9780199568086.003.0021.
1. Sakamoto, Shu, Atsuya Kobayashi, Karin Matsushita, Risa Shimizu, and Atsushi Aoyama. 2019. “Decoding Relative Pitch Imagery Using Functional Connectivity: An Electroencephalographic Study.” In 2019 IEEE 1st Global Conference on Life Sciences and Technologies (LifeTech). IEEE. https://doi.org/10.1109/lifetech.2019.8884007.
1. “The EEGsynth,” The EEGsynth.  [On line]. Available: http://www.eegsynth.org/.  [Ac cessed: 01-Feb-2020]
1. Li, M.-A., Wang, Y.-F., Jia, S.-M., Sun, Y.-J., & Yang, J.-F. (2019). Decoding of motor imagery EEG based on brain source estimation. Neurocomputing, 339, 182–193. https://doi.org/10.1016/j.neucom.2019.02.006
