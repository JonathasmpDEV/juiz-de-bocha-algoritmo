# from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances

# ==============================================================================
# INSTRUÇÕES PARA USAR DATASETS DO GOOGLE DRIVE:
# 1. Configure 'gdrive_config.json' com a URL da sua pasta no Google Drive.
# 2. Execute 'python gdrive_auth.py' para autenticar (se necessário).
# 3. Execute 'python gdrive_download.py --type datasets --output_path ./datasets_gdrive'
#    (ou o caminho que preferir). Isso baixará a subpasta 'datasets'
#    do seu Google Drive para './datasets_gdrive/'.
# 4. Certifique-se que a estrutura dentro de './datasets_gdrive/' seja:
#    ./datasets_gdrive/
#       juiz_de_bocha_train.json
#       juiz_de_bocha_val.json
#       data/
#           imagem1.jpg
#           imagem2.jpg
#           ...
# 5. ATUALIZE A VARIÁVEL 'dataset_dir' ABAIXO para apontar para este diretório.
#    Por exemplo: dataset_dir = './datasets_gdrive'
# ==============================================================================
# Diretório base onde os datasets (baixados do Google Drive ou locais) estão localizados.
# Este diretório deve conter os arquivos JSON de anotação e uma subpasta 'data/' com as imagens.
dataset_dir = 'datasets/juizdebocha-raw/dataset'  # <--- MODIFIQUE AQUI PARA SEU CAMINHO DE DATASET
classes = ["sports ball"]


def juiz_de_bocha_custom():
    register_coco_instances(
        name="juiz_de_bocha_train",
        metadata=dict(thing_classes=classes),
        json_file=f'{dataset_dir}/juiz_de_bocha_train.json',
        image_root=f'{dataset_dir}/data',
    )

    register_coco_instances(
        name="juiz_de_bocha_val",
        metadata=dict(thing_classes=classes),
        json_file=f'{dataset_dir}/juiz_de_bocha_val.json',
        image_root=f'{dataset_dir}/data',
    )
    # DatasetCatalog.register("juiz_de_bocha_train", juiz_de_bocha)
    # MetadataCatalog.get("juiz_de_bocha_train").thing_classes = classes
