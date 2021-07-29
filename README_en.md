<h4 align="center">
    <p>
        <b>English</b> |
        <a href="https://github.com/xuawai/TSFrame-time-series-forecasting/blob/master/README.md">简体中文</a> 
    <p>
</h4>

## Background
TSFrame is a project used for univariate time series forecasting. 
Python 3.6 is required.

##### Features:
* support STL/RobustSTL/X11 for time series decomposition
* support Arima/Holt/Prophet for time series forecasting
* no-decompose mode: predict the time series data directly
* decompose mode:
  1. decompose the data into three components: trend, seasonal and residual
  2. predict each component separately with specified model
  3. aggregate the results as the final prediction
* in-sample & out-of-sample predictions
* easy to extend for more models

## Quick Start

```shell
python main.py
```

## Usage

Refer to [Config.py](Config.py) for the docs on how to change the default parameters for your own time series forecasting task.

## Examples

Use the default parameters provided in [Config.py](Config.py)

##### Decompose Mode (Holt for trend, Prophet for seasonal, Arima for residual)

* Decompose result 

  ![decompose](./img/decompose.png)

* trend prediction

  ![trend](./img/trend.png)

* seasonal prediction

  ![season](./img/season.png)

* residual prediction

  ![residual](./img/residual.png)

* final prediction

  ![total](./img/total.png)

##### No-decompose Mode (use Arima)

![no_decompose_total](./img/no_decompose_total.png)
