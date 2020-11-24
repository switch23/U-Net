# U-Net
PyTorchを用いてセグメンテーションアルゴリズムU-Netを実装する
U-Netの元論文
U-Net: Convolutional Networks for Biomedical Image Segmentation 
https://arxiv.org/abs/1505.04597

## データセット
下記の上皮細胞データセットを利用する
標準細胞染色画像とマスク画像が与えられており, 疾患領域にアノテーションが付与されている
本プログラムを実行する際には下記サイトより epi.tgz ファイルをダウンロードし, ./data/ ディレクトリ下で解凍してください
EPITHELIUM SEGMENTATION
http://www.andrewjanowczyk.com/use-case-2-epithelium-segmentation/

## 実行方法
Python3.7系で動作確認済み
必要なライブラリ
- PyTorch
- PIL

epi.sh ファイル内では Anaconda 環境を設定しているが, 必要なければ適宜修正してください
使用するGPUサーバーやGPUなどを設定した後に
```
sh epi.sh
```
または
```
qsub epi.sh
```
を実行する

## 結果
![AccuracyとLossの変化](https://github.com/switch23/U-Net/result/accuracy_loss.pdf)
![セグメンテーション結果](https://github.com/switch23/U-Net/result/segmentation_image.pdf)