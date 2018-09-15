# -*- coding: utf-8 -*-

import os
import json
import glob
import ml_util as mu
import consts


def load_review_jsons(embedding=None, save_file=None):

    def import_dict(temp_json_dict):
        temp_comment = temp_json_dict["body"]
        temp_rate_str = temp_json_dict["rating"]
        temp_rate_int = int(temp_rate_str)
        temp_rating = (temp_rate_int - 1)
        return temp_comment, temp_rating

    data = mu.ExpData()
    data.x = []
    data.y = []
    data.max_class_count = 5
    for file_name in sorted(glob.glob(consts.SOURCE_JSONS)):
        print("file_name:", file_name)
        with open(file_name, "r") as file:
            json_data = json.load(file)
            for json_dict in json_data:
                comment, rating = import_dict(json_dict)
                data.x += [comment]
                data.y += [rating]

    temp_embedding = embedding
    if embedding is None:
        temp_embedding = mu.EmbeddingData(is_default_fasttext=True)

    data.embed_data(embedding=temp_embedding)

    os.makedirs(os.path.dirname(save_file), exist_ok=True)
    data.save(save_file)

    return data


def main():
    print("main")

    load_review_jsons(save_file=consts.EXP_FILE)


if __name__ == '__main__':
    main()