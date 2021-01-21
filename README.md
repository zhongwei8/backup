# Human Activity Recognition Time Series Classification

## 文件目录

```
|---data // 训练好的模型文件，数据处理工具及数据标注工具。
    |---model // 模型文件
        |--- har_cnn-0615.pb   // tensorflow pb 模型文件，从hdf5文件转换来。
        |--- weights.best-0615.hdf5  // keras训练得到的原始模型文件
        |--- har_cnn-0615.yml  // mace部署的配置文件
        |--- har_cnn_micro.tar.gz  // 模型专程mace micro的代码包
    |--- tools
        |--- inter.py  // 插值工具
        |--- label_tool.py  // 标注工具
        |--- utils.py  // 公用函数
|---src
    |---c
        |---har_model  // har模型预测c源码，输入三轴加速度，输出6类概率值
            |--- CMakeLists.txt // 编译文件
            |--- har_model.c     // har模型预测c代码
            |--- har_model.h     // har模型预测头文件
            |--- har_model_python.cc  // har模型pybind接口代码
            |--- micro_engine_c_interface.h   // har模型mace micro头文件
        |---har_detector // har上层处理源代码，输入6类概率值，投票输出预测结果
            |--- har_detector.c  // c代码
            |--- har_detector.h  // 头文件
            |--- har_detector_func.h  // 工具函数头文件
            |--- har_detector_python.cc  // pybind接口文件
            |--- CMakeLists.txt  // 编译文件
    |---py
        |---tools
        |---har_det.py
        |---har_det_c.py
        |---har_model.py
        |---har_model_c.py
|---test
    |---cunit // 单元测试
    |---eval //算法测试代码

```


## Download the dataset
```sh
./data/download_dataset.sh
```

## Trainning
Train the CNN model:
```sh
python tools/train.py --type cnn \
                      --dir dir \
		              --test Test \
                      --kfold 5 \
                      --epochs 100 \
                      --batch 128 \
                      --save_best=True
```

Monitoring training progress using tensorboard:
```sh
tensorboard --logdir=logs
```

## Evaluation
```sh
python3 scripts/evaluate.py configs/har-model-evaluation.yml
```
