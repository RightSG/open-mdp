<div align="center">

# open-mdp

_✨ 娱乐性maimai谱面定数预测 ✨_

</div>

快速尝试 > [https://right-sg.icu/mdp-web](https://right-sg.icu/mdp-web)

# 简介

open-mdp 或称 maimai-diff-predictor

旨在使用一个已经过训练 LSTM 模型来分析基于 Simai 语法的 maidata.txt 谱面文件，并据此推测出该谱面的难度定数。

当前版本的项目于 2024/8/24 上传，模型完成于 2024/8/17。目前先开源了模型文件和推理模块，在进一步完善后训练模块也会一同放出。

目前模型较为稚嫩，预测定数仅供参考，如果有准确度要求请务必辅以人工检查。

# 结构

`inference` 目录下存放了模型的推理模块，可以使用相应程序和模型完成定数推测。

`serializer` 目录下存放了谱面序列化预处理模块，其中 Process 脚本基于 MajdataEdit 项目。

# 使用

当前项目可能还未针对易于使用的场景进行优化。如果需要适应不同的使用需求，请查阅并修改对应目录下的源代码，以便构建出更符合实际需求的程序。

~~把大象装进冰箱需要几步？~~

1. 进入 `serializer\` 目录，运行 `SimaiSerializerFromMajdataEdit.exe <谱面文件路径> <谱面等级(1-6)>`
2. 移动 `serializer\temps\` 下的 json 文件到 `inference\`， 重命名为 `pre-rawchart.json`
3. 进入 `inference\` 目录，运行 `python main.py`

要应对大量谱面处理的情况，您有必要需要修改源代码，对应的函数均做了一定程度的封装，因此相关的过程并不繁琐。