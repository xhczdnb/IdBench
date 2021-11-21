#预训练词嵌入模型
#

import argparse
import numpy
from os.path import basename
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
import csv


parser = argparse.ArgumentParser()


parser.add_argument(
    '--target', help=("target file path"),required=True
)

embeddings=["trained_embeddings/ft_cbow.vec", "trained_embeddings/ft_sg.vec"]

paths=['small_pair_wise.csv','medium_pair_wise.csv','large_pair_wise.csv']

# 加载模型
# param path:词嵌入模型文件路径
def load_embedding(path):
    if path.endswith(".vec"):
        return KeyedVectors.load_word2vec_format(path, encoding='utf8')
    elif path.endswith(".model"):
        return Word2Vec.load(path)
    elif path.endswith(".bin"):
        return KeyedVectors.load_word2vec_format(path, binary=True, encoding='utf8')
    else:
        raise Exception(f"Unsupported kind of embedding: {path}")

# 计算cosin相似度
def cos_sim(x, y):
    temp = x / numpy.linalg.norm(x, ord=2)
    temp2 = y / numpy.linalg.norm(y, ord=2)
    return round(numpy.dot(temp, temp2), 2)


def look_up_word(embedding, word, wrap=False):
    if wrap:
        word = f"\"ID:{word}\""
    return embedding[word]

# 得到所有标识符对
# param
# path: 已经预处理好的含标识符对的文件路径
def get_all_pairs(path):
    result =[]
    with open(path, 'rt') as f:
        cr = csv.reader(f)
        for row in cr:
            id=[]
            id.append(row[0])
            id.append(row[1])
            result.append(id)
    result.pop(0)
    return result

# 测试用demo
def demo():
    embedding = embeddings[0]
    embedding_load_1 = load_embedding(embedding)
    wrap = basename(embedding) in [
        "ft_cbow.vec", "ft_sg.vec", "w2v_cbow.model", "w2v_sg.model","path_based.bin"]

    for w1, w2 in [["response", "alert"], ["ln", "ilen"], ["tasks", "todos"]]:
        v1 = look_up_word(embedding_load_1, w1, wrap)
        v2 = look_up_word(embedding_load_1, w2, wrap)

        sim = cos_sim(v1, v2)

        print(
            f"According to {basename(embedding)}, identifiers {w1} and {w2} have similarity {sim}")


if __name__ == "__main__":
    demo()

    # 获取参数
    args = parser.parse_args()
    target=args.target


    print('id1,id2,FT-cbow,FT-SG,w2v-SG,w2v-cbow')
    with open(target, "w", encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['id1','id2','FT-cbow','FT-SG','w2v-SG','w2v-cbow'])
    csvfile.close()
    embedding_load = []
    for embedding_path in embeddings:
        embedding = load_embedding(embedding_path)
        embedding_load.append(embedding)
    for path in paths:
        pairs = get_all_pairs(path)
        for w1, w2 in pairs:
            info=[w1,w2]
            for i in range(len(embedding_load)):
                wrap = basename(embeddings[i]) in [
                    "ft_cbow.vec", "ft_sg.vec", "w2v_cbow.model", "w2v_sg.model"]

                v1 = look_up_word(embedding_load[i], w1, wrap)
                v2 = look_up_word(embedding_load[i], w2, wrap)

                sim = cos_sim(v1, v2)
                info.append(sim)
                # print(
                #     f"According to {basename(embedding[i])}, identifiers {w1} and {w2} have similarity {sim}")
            print(info[0],info[1],info[2],info[3])
            with open(target, "a+", encoding='utf-8', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(info)
            csvfile.close()


