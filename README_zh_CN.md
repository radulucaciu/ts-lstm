![](https://img.shields.io/badge/scikit--learn-0.20-brightgreen) ![](https://img.shields.io/badge/tensorflow-2-orange) ![](https://img.shields.io/badge/plotly-5.12-blue)
![](https://img.shields.io/badge/pandas-1.3-green)

[英文文档](./README.md)
## 介绍
`lymboy-lstm`包含了几种常用的LSTM模型用于时间序列预测。目前仅支持单变量时间序列预测。
目前内置的模型有：`LSTM` `BiLSTM` `CNN_LSTM`  `CNN_BiLSTM`
其他的模型正在研究中...（CNN_BiLSTM_Attention, Encoder-Decoder Model, 多元时间预测支持）尽请期待~

## 打包方法
```shell
python ./setup.py sdist
pip install dist/lymboy-lstm-[latest-version].tar.gz
# 上传到pypi
# pip install twine
# twine upload dist/*
```


## 安装方法
```shell
pip install lymboy-lstm
```

## 如何使用
### 以LSTM模型预测电力消耗为例
+ 导包
```python
import pandas as pd
import numpy as np
from lstm import LSTM
from lstm.util import plot
```
+ 读取数据
```python
file = './dataset/power/power_consumption_A.csv'
df = pd.read_csv(file, index_col=0)
sequence = df.load
```
+ 建模和预测
```python
# 用过去96次的数据预测未来未来10次的数据
model = LSTM(n_steps=96, n_output=10)
# 将序列数据处理为模型输入，指定测试集比例为20%
model.createXY(sequence, test_size=0.2)
model.fit(epochs=500, verbose=True)
print(model.score()) 
```
![lstm-predict-96to10](https://itbird.oss-cn-beijing.aliyuncs.com/img/2023/03/02/lstm-predict-96to10.png)

```python
plot(model.y_hat[:,0], model.y_test[:,0])
```
![lstm-predict-96to10-plot](https://itbird.oss-cn-beijing.aliyuncs.com/img/2023/03/02/lstm-predict-96to10-plot.png)

### CNN_BiLSTM预测变压器油温
+ 导包
```python
import pandas as pd
import numpy as np
from lstm import LSTM, BiLSTM, CNN_BiLSTM
from lstm.util import plot
```
+ 读取数据
```python
file = './dataset/ETT/ETTh1.csv'
df = pd.read_csv(file, index_col=0)
sequence = df.OT
```
+ 建模和预测
```python
model = CNN_BiLSTM(n_steps=96, n_output=24, n_seq=6)
model.createXY(sequence)
model.fit(epochs=500, verbose=True)
print(model.score())
```

![cnnbilstm-predict-96to24-plot](https://itbird.oss-cn-beijing.aliyuncs.com/img/2023/03/02/cnnbilstm-predict-96to24-plot.png)

## 参数说明

+ n_steps: 训练步长，表示历史数据的步长，int
+ n_output：预测输出长度，int
+ n_seq：子序列，int (注意，n_seq应能被n_steps整除，最小为1)

其他参数与`tensorflow`一致


## 错误反馈
alayama@163.com