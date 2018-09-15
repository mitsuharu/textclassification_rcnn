# -*- coding: utf-8 -*-

import os
import keras.backend as K
from keras.layers import Dense, LSTM
from keras.layers import Input, Lambda, TimeDistributed
from keras.layers.merge import concatenate
from keras.models import Model
import ml_util as mu
import consts


def model_rcnn(embedding=None):
    print("model_rcnn_regular")

    if embedding is None:
        embedding = mu.EmbeddingData(is_default_fasttext=True)
    if embedding.layer is None:
        return None

    hidden_dim_1 = 200
    hidden_dim_2 = 100

    document = Input(shape=(None,), dtype="int32")
    left_context = Input(shape=(None,), dtype="int32")
    right_context = Input(shape=(None,), dtype="int32")

    doc_embedding = embedding.layer(document)
    l_embedding = embedding.layer(left_context)
    r_embedding = embedding.layer(right_context)

    forward = LSTM(hidden_dim_1, return_sequences=True)(l_embedding)
    backward = LSTM(hidden_dim_1, return_sequences=True, go_backwards=True)(r_embedding)
    together = concatenate([forward, doc_embedding, backward], axis=2)

    semantic = TimeDistributed(Dense(hidden_dim_2, activation="tanh"))(together)

    pool_rnn = Lambda(lambda x: K.max(x, axis=1), output_shape=(hidden_dim_2,))(semantic)

    output = Dense(consts.NUM_TEXT_CLASSES, input_dim=hidden_dim_2, activation="softmax")(pool_rnn)

    model = Model(inputs=[document, left_context, right_context], outputs=output)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    return model


def train_model(source_file: str, save_path: str =None):
    print("train_model")

    # 埋め込み層
    embedding = mu.EmbeddingData(is_default_fasttext=True)
    if embedding.layer is None:
        print("error: embedding is not found")
        return None

    asset_data = mu.ExpData(file_path=source_file)

    model = model_rcnn(embedding=embedding)

    # コールバック
    callbacks = []
    callbacks += mu.KerasUtil.callback_tensorboard(output_path=consts.CB_TENSORBOARD)
    callbacks += mu.KerasUtil.callback_checkpoint(output_path=consts.CB_CHECKPOINT)
    callbacks += mu.KerasUtil.callbacks_early_stopping()

    # 学習回数
    epoch_count = 100
    # 100

    # 学習
    history = model.fit([asset_data.x_train, asset_data.x_train_left, asset_data.x_train_right],
                        asset_data.y_train,
                        epochs=epoch_count,
                        callbacks=callbacks,
                        validation_split=0.1,
                        verbose=1)
    loss = history.history["loss"][0]
    print("loss:", loss)
    print()

    mu.KerasUtil.export_history_to_csv(history=history)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        model.save(save_path)


def main():
    print("main")

    method_h5_file_path = "./data/model.h5"
    exp_file = consts.EXP_FILE
    train_model(source_file=exp_file, save_path=method_h5_file_path)


if __name__ == '__main__':
    main()
