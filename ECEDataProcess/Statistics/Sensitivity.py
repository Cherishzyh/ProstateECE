# 分割模型（参考周五段毅汇报的那篇综述，中间有一块描述怎么计算）：
# 模型输入有两个：模型分割的ROI（二值化的np.array 2D/3D）和模型真实ROI（二值化的np.array 2D/3D）。

# 如果是多标签分类，每个做单独统计
# 模型输出有：敏感性（label中标记为1的部分）、特异性（label中标记为0的部分）、


import torch
import torch.nn as nn


# 敏感性：在患有癌症的所有人中，诊断正确的人有多少？ 真阳性人数/（真阳性人数+假阴性人数）*100%。正确判断病人的率；
# 特异性：在未患癌症的所有人中，诊断正确的人有多少？ 真阴性人数/（真阴性人数+假阳性人数））*100%。正确判断非病人的率

# TP    predict 和 label 同时为1
# TN    predict 和 label 同时为0
# FN    predict 0 label 1
# FP    predict 1 label 0





# if __name__ == '__main__':
