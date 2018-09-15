# テキスト分類

RCNN (Reccurent CNN) でテキスト分類を行う

- amazonレビューのコメントとレーティングの関係を学習する
- コメントからレーティングを推定する

実装

- Python 3 + TesorFlow + Keras

参考文献

- Recurrent Convolutional Neural Networks for Text Classification
  Siwei Lai, Liheng Xu, Kang Liu, Jun Zhao, Chinese Academy of Sciences, China
  AAAI. 2015.

- [PDF](https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/view/9745/9552)

# amazonコメント

- [Amazon Multi Language Reviews Scraper](https://github.com/philipperemy/amazon-reviews-scraper) を使用して収集した
- gitには添付してません

```
# comments.json
[
    {
        "body": "ファン必見の本です",
        "product_id": "xxxxxxxx",
        "rating": "5",
        "title": "ファン必見"
    },
    ...
]
```

# コード

手順

1. amazonコメントの加工
2. RCNNモデルの学習

## textclass/consts.py

- 定数まとめ

## textclass/ml_util.py

- 補助

## textclass/make_review_data.py

- コメントデータを実験用データに変換する

## textclass/textlass_rcnn_fasttext.py

- RCNNでテキスト分類
- 訓練

## fasttext

- hogehoge.vec / hogehoge.bin 
- gitには添付してません

