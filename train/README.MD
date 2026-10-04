# Treinamento (`train/`, `train_custom_dataset/` e `models/`)

Este README cobre as três pastas envolvidas no treinamento do modelo do Juiz de Bocha com Detectron2 (Mask R-CNN, uma única classe: `sports ball`).

Pré-requisitos: ambiente com PyTorch + Detectron2 instalados (ver [README da raiz](../README.MD)), de preferência com GPU; os scripts abaixo foram escritos para rodar **a partir da raiz do repositório** (os caminhos `models/...`, `datasets/...` e `./train/training/` são relativos a ela). Datasets e pesos não são versionados; podem ser baixados do Google Drive com `gdrive_download.py` (ver README da raiz).

## `train/` — treino inicial a partir do COCO

Arquivos:

| Arquivo | Função |
|---|---|
| `prepare_datasets.py` | Lê `datasets/coco/annotations/instances_{train,val}2017.json` e gera `juiz_de_bocha_{train,val}.json` na mesma pasta, mantendo só a categoria COCO `sports ball` (reindexada para `id=1`); no conjunto `val` remove imagens sem bola. |
| `register_datasets.py` | Função `juiz_de_bocha()` que registra `juiz_de_bocha_train` e `juiz_de_bocha_val` (formato COCO). Edite `annotations_dir`, `train_images_dir` e `val_images_dir` no topo do arquivo (padrão `datasets/coco/annotations`, `datasets/coco/train2017`, `datasets/coco/val2017`). |
| `train.py` | Treino com `DefaultTrainer` + `COCOEvaluator`. Usa `models/mask_rcnn_juiz_de_bocha/mask_rcnn_juiz_de_bocha.yaml`, inicia dos pesos COCO Mask R-CNN R-50 FPN 3x do model zoo, `IMS_PER_BATCH=2`, `BASE_LR=0.005`, `MAX_ITER=15000`, `BATCH_SIZE_PER_IMAGE=128`, `NUM_WORKERS=8`; saída em `./train/training/` (ignorada pelo git); `resume_or_load(resume=True)` retoma o último checkpoint se existir. |
| `plot.py` | Lê `train/training/metrics.json` e plota todas as métricas com matplotlib. |

Uso:

```bash
# 1) (opcional) gerar os JSON de sports ball a partir do COCO completo
python train/prepare_datasets.py

# 2) treinar
python train/train.py

# 3) ver as curvas de treino
python train/plot.py
```

`train.py` faz `import register_datasets`, o que funciona ao executá-lo como `python train/train.py` (o diretório do script entra no `sys.path`).

> O README anterior sugeria `python detectron2-tools/train_net.py --config-file detectron2-configs/mask_rcnn_juiz_de_bocha/mask_rcnn_juiz_de_bocha.yaml --num-gpus 1`. Esse caminho `detectron2-configs/` não existe no repositório (as configs ficam em `models/`) e o `train_net.py` não registra os datasets `juiz_de_bocha_*`, então o comando não funciona sem adaptação. Use `train/train.py`.

## `train_custom_dataset/` — dataset próprio e re-treino

Pipeline numerado, escrito para fotos reais enviadas pelos usuários (pasta `datasets/juizdebocha-raw/`, que não está no repositório):

| Script | Função |
|---|---|
| `1_test_results.py` | Roda `_process_image`/`_to_gif_bytes` do `cloud-run-function/main.py` (importado via `importlib`) em todas as fotos de `<base_dir>/raw` e grava um GIF por foto em `<base_dir>/output`. O `base_dir` está fixo como um caminho absoluto da máquina do autor original (`/media/viniciusas/...`); ajuste antes de usar. O docstring final lista, por categoria, os nomes de arquivo de casos problemáticos (bolim estranho, sujeira, bola não identificada etc.). |
| `2_generate_labels_files.py` | Auto-rotulação: usa um `Recognizer` (config `cloud-run-function/models/model/config.yaml`, pesos `models/mask_rcnn_juiz_de_bocha_custom/weights_first_train.pth`, confiança 0.7, GPU 0) para predizer máscaras em `./datasets/juizdebocha-raw/dataset/data` e escreve `juiz_de_bocha_train.json` / `juiz_de_bocha_val.json` (COCO, split 80/20, `random.seed(2)`) em `./datasets/juizdebocha-raw/dataset`. Todas as instâncias preditas viram a categoria `sports ball` (id 1). |
| `3_train.py` | Fine-tuning a partir de `models/mask_rcnn_juiz_de_bocha_custom/weights_first_train.pth` com `models/mask_rcnn_juiz_de_bocha_custom/mask_rcnn_juiz_de_bocha.yaml`: `IMS_PER_BATCH=3`, `BASE_LR=0.0025`, `MAX_ITER=1000`, com augmentations (brilho, flip, crop 640x640, rotação, contraste, iluminação, saturação). Saída em `./train_custom_dataset/training/` (ignorada pelo git). |
| `4_cases_validation.py` | Troca o `_rec` do `cloud-run-function/main.py` por um `Recognizer` com `models/mask_rcnn_juiz_de_bocha_custom/mask_rcnn_juiz_de_bocha.yaml` + `train_custom_dataset/training/model_final.pth` (confiança 0.82) e gera GIFs dos casos de erro listados no dicionário `error_cases` (a maioria está comentada; só os grupos `leaves` e `before launch` estão ativos). Usa caminhos relativos com `../`, ou seja, foi escrito para rodar de um subdiretório. |
| `fix_deleted_dataset.py` | Copia de `raw/` para `dataset/data/` (em `./datasets/juizdebocha-raw`, via `cp`) as fotos que não estão nas listas de erro nem em `removed/`. |
| `register_datasets.py` | Função `juiz_de_bocha_custom()` que registra `juiz_de_bocha_train`/`juiz_de_bocha_val` apontando para `dataset_dir` (padrão `datasets/juizdebocha-raw/dataset`, JSON de anotação + subpasta `data/`). |

Ordem de uso pretendida: 1 (inspecionar resultados) → 2 (gerar rótulos) → 3 (treinar) → 4 (validar casos de erro).

Observações de execução (não testadas): `2_generate_labels_files.py` faz `from recognizer import Recognizer` e `3_train.py` faz `from train_custom_dataset import register_datasets`, então a raiz do repositório precisa estar no `PYTHONPATH` (ex.: `PYTHONPATH=. python train_custom_dataset/3_train.py`). O `1_test_results.py` e o `4_cases_validation.py` importam `cloud-run-function.main`, que exige `recognizer.py` e `models/model/{config.yaml,weights.pkl}` dentro de `cloud-run-function/` (ver [README dessa pasta](../cloud-run-function/README.MD)).

## `models/` — configurações do Detectron2

```
models/
├── Base-RCNN-FPN.yaml                    # base: GeneralizedRCNN + ResNet-FPN, datasets coco_2017_*
├── mask_rcnn_R_50_FPN_3x/                # Mask R-CNN R-50 FPN 3x (+ metrics.json de treino)
├── mask_rcnn_X_101_32x8d_FPN_3x/         # Mask R-CNN X-101 32x8d FPN 3x (+ metrics.json, run-demo.sh)
├── mask_rcnn_juiz_de_bocha/              # config do projeto: R-50, NUM_CLASSES=1, datasets juiz_de_bocha_*
└── mask_rcnn_juiz_de_bocha_custom/       # idem, sem MODEL.WEIGHTS; *.pth ignorados pelo git
```

- Todos herdam de `Base-RCNN-FPN.yaml` via `_BASE_`.
- `mask_rcnn_juiz_de_bocha/` é a config usada por `train/train.py`; `mask_rcnn_juiz_de_bocha_custom/` é a usada por `train_custom_dataset/` (espera `weights_first_train.pth` nessa pasta).
- `mask_rcnn_X_101_32x8d_FPN_3x/run-demo.sh` executa `detectron2-test/demo.py` com essa config, a imagem `datasets/Jogos de bocha/IMG-20220108-WA0044.jpg` e os pesos `model_final_2d9806.pkl` da própria pasta (arquivo não versionado).
- O modelo usado pela API (`cloud-run-function/`) é copiado manualmente para `cloud-run-function/models/model/` (config + pesos); o README original dessa pasta usa a config X-101.
- Arquivos `*.pkl` e `*.pth` não estão no repositório.
