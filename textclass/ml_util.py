# -*- coding: utf-8 -*-

import pickle
import string
import MeCab
import keras
import shutil
from keras.layers import Embedding
from keras.callbacks import *
from sklearn.model_selection import train_test_split
import consts

# デフォルトの保存用ファイルパス
default_exp_file_path = consts.EXP_FILE


# モデルデータのクラス
class ExpData:

    # 初期化
    def __init__(self, file_path=None):
        self.x = []
        self.y = []
        self.x_embed = None
        self.y_embed = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

        self.x_train_right = None
        self.x_train_left = None
        self.x_test_right = None
        self.x_test_left = None

        self.file_path = file_path
        self.max_class_count = 10
        self.max_embed_count = 1000

        if file_path is not None:
            temp_self = ExpData.load(file_path=file_path)
            if temp_self is not None:
                self.x = temp_self.x
                self.y = temp_self.y
                self.x_embed = temp_self.x_embed
                self.y_embed = temp_self.y_embed
                self.x_train = temp_self.x_train
                self.x_test = temp_self.x_test
                self.y_train = temp_self.y_train
                self.y_test = temp_self.y_test
                self.file_path = temp_self.file_path
                self.max_class_count = temp_self.max_class_count
                self.max_embed_count = temp_self.max_embed_count
                self.x_train_right = temp_self.x_train_right
                self.x_train_left = temp_self.x_train_left
                self.x_test_right = temp_self.x_test_right
                self.x_test_left = temp_self.x_test_left

    # 訓練とテストデータを(x, y)から分割して生成する
    def make_train_and_test_data(self):
        temp_x = self.x
        if self.x_embed is not None:
            temp_x = self.x_embed
        temp_y = self.y
        if self.y_embed is not None:
            temp_y = self.y_embed
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(temp_x,
                                                                                temp_y,
                                                                                test_size=0.2,
                                                                                random_state=1)

    # データの埋め込み準備を行う
    def embed_data(self, embedding):
        print("embed_data")

        def nparray_regulated_length(words, max_count):
            nrl_words = words[:]
            nrl_is_regulated = False
            if len(words) > max_count:
                nrl_words = words[:max_count]
                nrl_is_regulated = True
            nrl_arr = np.array([nrl_words])
            return nrl_arr, nrl_is_regulated

        # 文字可変なので配列は大きくとって0で埋める
        self.x_embed = None
        for text in self.x:
            tokens = embedding.text_to_word_index_array(text=text)
            token_array, is_regulated = nparray_regulated_length(words=tokens, max_count=self.max_embed_count)
            if self.x_embed is None:
                self.x_embed = np.zeros(shape=(1, self.max_embed_count), dtype='int32')
                self.x_embed[0, :token_array.size] = token_array
            else:
                temp_x_train = np.zeros(shape=(1, self.max_embed_count), dtype='int32')
                temp_x_train[0, :token_array.size] = token_array
                self.x_embed = np.append(self.x_embed, temp_x_train, axis=0)

        self.y_embed = keras.utils.to_categorical(self.y, num_classes=self.max_class_count)

        self.make_train_and_test_data()
        self.x_train_left, self.x_train_right = embedding.left_and_right_shifted_index_nparray(self.x_train)
        self.x_test_left, self.x_test_right = embedding.left_and_right_shifted_index_nparray(self.x_test)

        self.save()

        return True

    # 書き出し
    def save(self, file_path=None):
        if file_path is None:
            if self.file_path is None:
                self.file_path = default_exp_file_path
        else:
            self.file_path = file_path

        try:
            os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        except Exception as e:
            print("error:", e)

        with open(self.file_path, mode='wb') as f:
            pickle.dump(self, f)
        return self.file_path

    # 読み込み
    @classmethod
    def load(cls, file_path=default_exp_file_path):
        result = None
        if file_path is not None and os.path.exists(file_path):
            try:
                with open(file_path, mode='rb') as f:
                    result = pickle.load(f)
            except Exception as e:
                print("error: ", str(e))
        # result.__class__ = ModelData
        # print(result.category_names)
        return result


# 自作の埋め込み層
class EmbeddingData:
    def __init__(self, fasttext_vec_file=None, is_default_fasttext=False):
        self.layer = None
        self.word_index = None
        self.max_token_index = 0

        if is_default_fasttext:
            self.load_fasttext_vec(model_name=self.default_fasttext_vec_file())
        elif fasttext_vec_file is not None:
            self.load_fasttext_vec(model_name=fasttext_vec_file)

    @staticmethod
    def default_fasttext_vec_file():
        file_name = None
        if file_name is None:
            file_name = consts.FASTTEXT_FILE
        if not os.path.exists(file_name):
            file_name = consts.FASTTEXT_FILE
        return file_name

    def load_fasttext_vec(self, model_name=None):
        file_name = model_name
        if model_name is None:
            file_name = self.default_fasttext_vec_file()
        if not os.path.exists(file_name):
            print("error: model file is none.")
            return False
        else:
            print("model_name:", file_name)

        with open(file_name, 'r') as f:
            data = f.readlines()

        word_vectors = {}
        samples, dim = data[0].split()

        for line in data[1:]:
            word, vec = line.split(' ', 1)
            word_vectors[word] = np.array([
                float(i) for i in vec.split()
            ], dtype='float32')

        weights_fasttext = np.zeros(shape=(int(samples) + 1, int(dim)), dtype='float32')
        self.word_index = list(word_vectors.keys())

        for ix in range(len(self.word_index)):
            word = self.word_index[ix]
            vec = word_vectors[word]
            for j in range(int(dim)):
                weights_fasttext[ix][j] = vec[j]

        self.max_token_index = len(self.word_index) + 1
        self.layer = Embedding(
            input_dim=self.max_token_index,
            output_dim=int(dim),
            weights=[weights_fasttext],
            trainable=False
        )

        # print("input_dim:", self.layer.input_dim)
        # print("output_dim:", self.layer.output_dim)

        return True

    # 1行の文字列から、単語ごとの token_index を格納したPython配列を生成する
    def text_to_word_index_array(self, text):

        # mecabで形態素解析を行う
        mt = MeCab.Tagger("-Owakati")
        wakati_text = mt.parse(text)

        # 空白区切りから配列に変換する
        temp_text = wakati_text.strip().lower().translate(
            str.maketrans({key: " {0} ".format(key) for key in string.punctuation}))
        tokens = temp_text.split()

        # 文字を token_index に変換する
        temp = [self.word_index.index(token) if token in self.word_index else (len(self.word_index) - 1) for token in tokens]
        return temp

    # 訓練またはテストデータから、RCNN（双方向LSTM）用の前後にシフトしたデータを生成する
    def left_and_right_shifted_index_nparray(self, index_nparray):
        shape = index_nparray.shape
        max_index_array = [self.max_token_index] * shape[0]
        added_array = np.array([max_index_array]).reshape(shape[0], 1)

        left_arr = index_nparray.copy()
        left_arr = np.delete(left_arr, shape[1] - 1, axis=1)
        left_arr = np.hstack((added_array, left_arr))

        right_arr = index_nparray.copy()
        right_arr = np.delete(right_arr, 0, axis=1)
        right_arr = np.hstack((right_arr, added_array))

        return left_arr, right_arr


class KerasUtil:

    @classmethod
    def export_history_to_csv(cls, history, csv_file=consts.CB_HISTORY):

        if not isinstance(history, keras.callbacks.History):
            print("error: export_history_to_csv/ history is not keras.callbacks.History")
            return

        epoch = history.epoch
        hist = history.history

        records_str = "epoch, acc, loss, val_acc, val_loss\n"
        for i in range(len(epoch)):
            records_str += "{}, {}, {}, {}, {}\n".format(i, hist["acc"][i], hist["loss"][i], hist["val_acc"][i], hist["val_loss"][i])

        with open(csv_file, "w", encoding="utf-8") as file:
            file.write(records_str)

    @classmethod
    def callback_tensorboard(cls, output_path="./log", model=None, feed={}):
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        os.makedirs(output_path, exist_ok=True)
        cb = [keras.callbacks.TensorBoard(log_dir=output_path, histogram_freq=1, write_graph=True)]
        return cb

    @classmethod
    def callback_checkpoint(cls, output_path="./checkpoint", period=1):

        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        os.makedirs(output_path, exist_ok=True)

        hdf5_name = 'weights.epoch{epoch:02d}-loss{loss:.2f}-acc{acc:.2f}-val_loss{val_loss:.2f}-val_acc{val_acc:.2f}.hdf5'
        cp_file_path = os.path.join(output_path, hdf5_name)
        cp_cb = keras.callbacks.ModelCheckpoint(cp_file_path,
                                                monitor='val_loss',
                                                verbose=0,
                                                save_best_only=False,
                                                save_weights_only=False,
                                                mode='auto',
                                                period=period)
        return [cp_cb]

    @classmethod
    def callbacks_early_stopping(cls, has_val_loss=True):
        temp_cbs = []

        cb0 = keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.01, patience=10, verbose=1, mode="auto")
        temp_cbs += [cb0]

        cb1 = keras.callbacks.EarlyStopping(monitor='acc', min_delta=0.001, patience=10, verbose=1, mode="auto")
        temp_cbs += [cb1]

        if has_val_loss:
            cb101 = EarlyStoppingByLossVal(monitor='val_loss', value=0.10, verbose=1)
            temp_cbs += [cb101]
            cb102 = EarlyStoppingByValue(monitor="acc", value=0.999, patience=5)
            temp_cbs += [cb102]
            cb103 = EarlyStoppingByValue(monitor="loss", value=0.001, patience=5)
            temp_cbs += [cb103]

        return temp_cbs

    @classmethod
    def callbacks_hyperdash(cls, exp):
        cb = Hyperdash(exp=exp)
        return [cb]


class EarlyStoppingByLossVal(Callback):
    """
    valueより小さくなったら、終わり
    """

    def __init__(self, monitor='val_loss', value=0.10, verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)

        if current < self.value:
            if self.verbose > 0:
                print("Epoch %05d: early stopping THR" % epoch)
            self.model.stop_training = True


class EarlyStoppingByValue(Callback):
    """
    valueより小さくなったら、終わり
    mode: one of {auto, min, max}. I
    """

    def __init__(self, monitor='val_loss', value=0.10, patience=5, mode="auto", verbose=1):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.patience = patience
        self.value = value
        self.verbose = verbose
        self.mode = mode
        self.count = 0

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn("EarlyStoppingByValue/Early stopping requires %s available!" % self.monitor, RuntimeWarning)

        is_less = True
        if self.mode == "auto":
            if "loss" in self.monitor:
                is_less = True
            if "acc" in self.monitor:
                is_less = False

        condition = (current >= self.value)
        if is_less:
            condition = (current <= self.value)

        if condition:
            self.count += 1
            if self.count > self.patience:
                if self.verbose > 0:
                    print("EarlyStoppingByValue/{}, {}, {}".format(self.monitor, self.value, self.mode))
                    print("EarlyStoppingByValue/Epoch %05d: early stopping THR" % epoch)
                self.model.stop_training = True
        else:
            self.count = 0


class Hyperdash(Callback):
    """
    from hyperdash import Experiment
    """

    def __init__(self, exp, verbose=1):
        super(Hyperdash, self).__init__()
        self.exp = exp
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs=None):
        acc = logs.get("acc")
        loss = logs.get("loss")
        val_acc = logs.get('val_acc')
        val_loss = logs.get('val_loss')
        if acc is not None:
            self.exp.metric("acc", acc)
        if loss is not None:
            self.exp.metric("loss", loss)
        if val_acc is not None:
            self.exp.metric("val_acc", val_acc)
        if val_loss is not None:
            self.exp.metric("val_loss", val_loss)
