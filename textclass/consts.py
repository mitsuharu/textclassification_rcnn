# コメントファイル
SOURCE_JSONS = "../source/*.json"

# コメントデータを実験用にまとめた
EXP_FILE = "./data/exp_data.pickle"

# fastTextのvecファイル
FASTTEXT_FILE = "../fasttext/fasttext_data.vec"

# 学習時のlogファイル
CB_TENSORBOARD = "./log"

# 学習時の重み保存ファイル
CB_CHECKPOINT = "./checkpoint"

# 学習時の精度を記録
CB_HISTORY = "./data/history.csv"

# 分類するクラス数
NUM_TEXT_CLASSES = 5

# 入力するトークン数
MAX_INPUT_TOKEN_COUNT = 100
