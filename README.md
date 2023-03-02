![](https://img.shields.io/badge/scikit--learn-0.20-brightgreen) ![](https://img.shields.io/badge/tensorflow-2-orange) ![](https://img.shields.io/badge/plotly-5.12-blue)
![](https://img.shields.io/badge/pandas-1.3-green)

[中文文档](./README_zh_CN.md)

## Introduce
`lymboy-lstm` contains several commonly used LSTM models for time series forecasting. Currently only univariate time series forecasting is supported.
Currently built-in models are: `LSTM` `BiLSTM` `CNN_LSTM` `CNN_BiLSTM`
Other models are under study... (CNN_BiLSTM_Attention, Encoder-Decoder Model, Multivariate Time Prediction Support) Please look forward to it~

## Packaging method

```shell
python ./setup.py sdist bdist_wheel
pip install dist/lymboy-lstm-[latest-version].tar.gz
# Upload to pypi
# pip install twine
# twine upload dist/*
```

## How to install?

```shell
pip install lymboy-lstm
```

## How to use?

### Taking LSTM model to predict power consumption as an example

+ Import lib
```python
import pandas as pd
import numpy as np
from lstm import LSTM
from lstm.util import plot
```
+ Read dataset
```python
file = './dataset/power/power_consumption_A.csv'
df = pd.read_csv(file, index_col=0)
sequence = df.load
```
+ Modeling
```python
# Use the data of the past 96 times to predict the data of the next 10 times in the future
model = LSTM(n_steps=96, n_output=10)
# Process sequence data as model input, specifying a test set ratio of 20%
model.createXY(sequence, test_size=0.2)
model.fit(epochs=500, verbose=True)
print(model.score()) 
```
![lstm-predict-96to10](https://itbird.oss-cn-beijing.aliyuncs.com/img/2023/03/02/lstm-predict-96to10.png)

```python
plot(model.y_hat[:,0], model.y_test[:,0])
```
![lstm-predict-96to10-plot](https://itbird.oss-cn-beijing.aliyuncs.com/img/2023/03/02/lstm-predict-96to10-plot.png)

### CNN_BiLSTM predicts transformer oil temperature

+ Import lib
```python
import pandas as pd
import numpy as np
from lstm import LSTM, BiLSTM, CNN_BiLSTM
from lstm.util import plot
```
+ Read dataset
```python
file = './dataset/ETT/ETTh1.csv'
df = pd.read_csv(file, index_col=0)
sequence = df.OT
```
+ Modeling
```python
model = CNN_BiLSTM(n_steps=96, n_output=24, n_seq=6)
model.createXY(sequence)
model.fit(epochs=500, verbose=True)
print(model.score())
```

![cnnbilstm-predict-96to24-plot](https://itbird.oss-cn-beijing.aliyuncs.com/img/2023/03/02/cnnbilstm-predict-96to24-plot.png)

## Parameter Description

+ n_steps: training step size, representing the step size of historical data, int
+ n_output: predicted output length, int
+ n_seq: subsequence, int (note that n_seq should be divisible by n_steps, the minimum is 1)

Other parameters are consistent with `tensorflow`


## Error feedback

alayama@163.com