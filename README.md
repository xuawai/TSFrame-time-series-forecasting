<h4 align="center">
    <p>
        <b>简体中文</b> |
        <a href="https://github.com/xuawai/TSFrame-time-series-forecasting/blob/master/README_en.md">English</a> 
    <p>
</h4>

## 项目背景
TSFrame是一个单指标时间序列数据预测项目，使用统一接口整合了多种时序分解方法与时序预测方法，保留核心参数，降低使用成本。

项目在Python 3.6的环境下运行。

##### 特点:
* 支持的时间序列分解方法包括：STL、RobustSTL、X11
* 支持的时间序列预测方法包括：Arima、Holt、Prophet
* 在无分解模式下，直接对时序数据进行预测。
* 在分解模式下，首先将时序数据分解为趋势项、季节项与残差项；接着，使用指定模型对每一项分别进行预测；最后，聚合各项预测结果，最后时序数据的最终预测结果。

* 支持in-sample & out-of-sample预测
* 轻松拓展至更多模型

## 快速上手

```shell
python main.py
```

## 用法

参考[Config.py](Config.py) 文件中的注释，根据时序预测任务的实际需求，灵活配置模型及参数。

## 示例

以下实验结果来自 [Config.py](Config.py)文件提供的默认配置：

##### 分解模式 (趋势项建模：Holt,；季节项建模：Prophet, 残差项建模：Arima)

* 加性分解

  ![decompose](./img/decompose.png)

* 趋势项预测

  ![trend](./img/trend.png)

* 季节项预测

  ![season](./img/season.png)

* 残差项预测

  ![residual](./img/residual.png)

* 最终预测结果

  ![total](./img/total.png)

##### 无分解模式 (使用Arima建模)

![no_decompose_total](./img/no_decompose_total.png)
