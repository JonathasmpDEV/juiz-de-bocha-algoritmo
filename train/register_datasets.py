# from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances

# ==============================================================================
# INSTRUÇÕES PARA USAR DATASETS DO GOOGLE DRIVE (OU OUTRO LOCAL):
# 1. Baixe/prepare seus datasets no formato COCO.
#    Se estiver usando o Google Drive e os scripts deste projeto:
#    a. Configure 'gdrive_config.json'.
#    b. Execute 'python gdrive_auth.py' (se necessário).
#    c. Execute 'python gdrive_download.py --type datasets --output_path ./datasets_gdrive'
# 2. Certifique-se que a estrutura dentro de './datasets_gdrive/' (ou seu caminho) seja:
#    ./datasets_gdrive/
#       annotations/  <-- Contendo juiz_de_bocha_train.json, juiz_de_bocha_val.json
#       train2017/    <-- Contendo imagens de treino (ou o nome da sua pasta de imagens de treino)
#       val2017/      <-- Contendo imagens de validação (ou o nome da sua pasta de imagens de validação)
# 3. ATUALIZE AS VARIÁVEIS 'annotations_dir', 'train_images_dir', e 'val_images_dir' ABAIXO.
# ==============================================================================
# Diretórios base para os datasets. MODIFIQUE CONFORME NECESSÁRIO.
annotations_dir = 'datasets/coco/annotations'  # <--- MODIFIQUE AQUI
train_images_dir = 'datasets/coco/train2017'    # <--- MODIFIQUE AQUI
val_images_dir = 'datasets/coco/val2017'        # <--- MODIFIQUE AQUI

classes = ["sports ball"]


def juiz_de_bocha():
    register_coco_instances(
        name="juiz_de_bocha_train",
        metadata=dict(thing_classes=classes),
        json_file=f'{annotations_dir}/juiz_de_bocha_train.json',
        image_root=train_images_dir,
    )

    register_coco_instances(
        name="juiz_de_bocha_val",
        metadata=dict(thing_classes=classes),
        json_file=f'{annotations_dir}/juiz_de_bocha_val.json',
        image_root=val_images_dir,
    )
    # DatasetCatalog.register("juiz_de_bocha_train", juiz_de_bocha)
    # MetadataCatalog.get("juiz_de_bocha_train").thing_classes = classes
