# 谱面训练模块简介

上传的 `train_model.py` 通过 `datasets/` 目录下存放的训练集和根目录下 `combined_index.csv` 作为数据集的索引和信息指示，其中 `datasets/` 目录下存放的 `json` 文件数据集均应经过 `serializer` 工具的序列化处理和 `inference` 模块中进一步的特征提取处理。而 `combined_index.csv` 中至少应当包含 `file_name` 和 `combined_diff` 两列，模型训练中 `combined_diff` 采用了官标定数 : 水鱼拟合定数 = 2 : 1 的权重，您可以自行调整。

原训练集关系到谱面版权问题，此处不做提供，为适应您的需求，您应当自己对持有的谱面文件做批处理，并适当对应调整 `train_model` 中读取文件的部分。
