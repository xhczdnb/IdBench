
#计算语义分析与标准的相关性

from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import re
import enchant
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import numpy as np


kinds = ["relatedness", "similarity", "contextual_similarity"]
sizes = ["small", "medium", "large"]
approaches = ["FT-cbow", "FT-SG", "w2v-SG",
              "w2v-cbow", "Path-based", "LV", "NW"]
new_approaches = []  # filled automatically based on given .csv files

dict = enchant.Dict("en_US")


parser = argparse.ArgumentParser()
parser.add_argument(
    '--small', help="Pairwise scores in small dataset", required=True)
parser.add_argument(
    '--medium', help="Pairwise scores in medium dataset", required=True)
parser.add_argument(
    '--large', help="Pairwise scores in large dataset", required=True)




# 逐类逐数据集计算相关性
# param size_to_kind_to_pairs：按数据集大小(size)，计算种类(kind)规格化后的标识符对
# size:"small","medium","large"
# kind:"similarity","relatedness","contextual_similarity"
#
def plot_correlations_all(size_to_kind_to_pairs):
    plt.rcParams.update({'font.size': 17})

    for kind in kinds:
        label_to_approach = {
            "LV": "LV",
            "NW": "NW",
            "FT-cbow": "FT-cbow",
            "FT-SG": "FT-SG",
            "w2v-cbow": "w2v-cbow",
            "w2v-SG": "w2v-SG",
            "Path-\nbased": "Path-based"
        }
        for a in new_approaches:
            label_to_approach[a] = a


        small_ys = compute_correlations(
            size_to_kind_to_pairs["small"][kind], kind, label_to_approach)
        medium_ys = compute_correlations(
            size_to_kind_to_pairs["medium"][kind], kind, label_to_approach)
        large_ys = compute_correlations(
            size_to_kind_to_pairs["large"][kind], kind, label_to_approach)

        plot_correlations(large_ys, medium_ys, small_ys,
                          f"correlations_{kind}.pdf",
                          labels=label_to_approach.keys())


# 计算相关性
# param
# pairs:按数据集大小(size)，计算种类(kind)规格化后的标识符对
# kind:计算的具体种类
# label_to_approach：语义表示种类
def compute_correlations(pairs, kind, label_to_approach):
    correlations = []
    for _, approach in label_to_approach.items():
        c = spearmanr(pairs[kind], pairs[approach]).correlation
        correlations.append(c)
    return correlations

# 可视化
def plot_correlations(ys_large, ys_medium, ys_small, out_file, labels):
    x = np.arange(len(labels))  # the label locations
    width = 0.2  # the width of the bars

    fig, ax = plt.subplots()
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)
    ax.bar(x - width, ys_large, width, label='Large benchm.')
    ax.bar(x, ys_medium, width, label='Medium benchm.')
    ax.bar(x + width, ys_small, width, label='Small benchm.')

    ax.set_ylim([0.0, 0.85])
    ax.set_yticks(np.arange(0, 0.9, step=0.2))
    ax.set_xlabel('Similarity functions')
    ax.set_ylabel('Correlation with benchmark')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
               ncol=2, mode="expand", borderaxespad=0.)

    fig.tight_layout()

    plt.savefig(out_file, format="pdf")



# 读取并格式化数据
def read_and_clean_pairs(args):
    size_to_additional_embeddings = {}
    size_to_kind_to_pairs = {}
    for size in sizes:
        size_to_kind_to_pairs[size] = {}
        pairs = pd.read_csv(getattr(args, size), dtype=object)
        print(pairs)

        new_column_headers = list(pairs.columns[12:])
        size_to_additional_embeddings[size] = new_column_headers

        # 忽略NAN
        def get_row_filter(kind):
            def row_filter(r):
                if r[kind] == "NAN":
                    return False
                for approach in approaches:
                    if r[approach] == "NAN":
                        return False
                return True
            return row_filter

        for kind in kinds:
            # 筛选每一行
            filtered_pairs = pairs[pairs.apply(get_row_filter(kind), axis=1)]
            size_to_kind_to_pairs[size][kind] = filtered_pairs


    return size_to_kind_to_pairs


if __name__ == "__main__":
    args = parser.parse_args()
    size_to_kind_to_pairs = read_and_clean_pairs(args)

    for size in sizes:
        for kind in kinds:
            pairs = size_to_kind_to_pairs[size][kind]

    plot_correlations_all(size_to_kind_to_pairs)
