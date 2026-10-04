# detectron2-tools

Cópia dos scripts de exemplo/utilitários do [Detectron2](https://github.com/facebookresearch/detectron2) (cabeçalho "Copyright (c) Facebook, Inc. and its affiliates"), mantida aqui para treinar, avaliar e inspecionar os modelos do projeto. Os scripts **não** têm código específico do Juiz de Bocha: em especial, nenhum deles registra os datasets `juiz_de_bocha_train`/`juiz_de_bocha_val` (isso é feito em `../train/register_datasets.py` e `../train_custom_dataset/register_datasets.py`). Para o treinamento do projeto, veja [`../train/README.MD`](../train/README.MD).

Pré-requisito: Detectron2 e PyTorch instalados (ver [README da raiz](../README.MD)).

This directory contains a few example scripts that demonstrate features of detectron2.


* `train_net.py`

An example training script that's made to train builtin models of detectron2.

For usage, see [GETTING_STARTED.md](https://github.com/facebookresearch/detectron2/blob/main/GETTING_STARTED.md) (o arquivo `GETTING_STARTED.md` não existe neste repositório).

* `plain_train_net.py`

Similar to `train_net.py`, but implements a training loop instead of using `Trainer`.
This script includes fewer features but it may be more friendly to hackers.

* `lazyconfig_train_net.py`

Training script that uses the new "LazyConfig" python config files (espera configs Python, como o `configs/common/train.py` do Detectron2, que não está neste repositório).

* `lightning_train_net.py`

Training with PyTorch Lightning (o próprio cabeçalho do script o descreve como beta).

* `benchmark.py`

Benchmark the training speed, inference speed or data loading speed of a given config.

Usage:
```
python benchmark.py --config-file config.yaml --task train/eval/data [optional DDP flags]
```

* `analyze_model.py`

Analyze FLOPs, parameters, activations of a detectron2 model.  See its `--help` for usage.

* `visualize_json_results.py`

Visualize the json instance detection/segmentation results dumped by `COCOEvalutor` or `LVISEvaluator`

Usage:
```
python visualize_json_results.py --input x.json --output dir/ --dataset coco_2017_val
```
If not using a builtin dataset, you'll need your own script or modify this script.

* `visualize_data.py`

Visualize ground truth raw annotations or training data (after preprocessing/augmentations).

Usage:
```
python visualize_data.py --config-file config.yaml --source annotation/dataloader --output-dir dir/ [--show]
```

NOTE: the script does not stop by itself when using `--source dataloader` because a training
dataloader is usually infinite.

* `convert-torchvision-to-d2.py`

Converte pesos ResNet do torchvision (`.pth`) para o formato `.pkl` do Detectron2. Uso (do cabeçalho do script): `./convert-torchvision-to-d2.py r50.pth r50.pkl`.

* `deploy/`

Exportação de modelos (TorchScript etc.) e exemplo em C++. Ver [`deploy/README.md`](deploy/README.md).
