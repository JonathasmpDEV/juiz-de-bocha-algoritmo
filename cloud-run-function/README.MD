# cloud-run-function — API do reconhecedor

API Flask (servida por gunicorn) que recebe a foto de uma jogada de bocha no corpo da requisição, detecta as bolas com o `Recognizer` (`../recognizer.py`, Detectron2/Mask R-CNN em CPU, limiar de confiança 0.82), identifica o bolim (menor bola na região central, margem de 10% da imagem) e a bola vencedora (a mais próxima do bolim), e devolve o resultado. É o serviço publicado no Google Cloud Run (serviço `recognizer`, região `southamerica-east1`, projeto `juizdebocha`).

## Conteúdo da pasta

| Arquivo | Descrição |
|---|---|
| `main.py` | App Flask (`app`) com os endpoints, desenho do resultado (círculo verde na vencedora, cruz/círculo no bolim), geração de GIF/miniatura e gravação dos arquivos. |
| `Dockerfile` | Imagem baseada em `dkimg/opencv:4.5.3-ubuntu`; instala `python3.9-dev`, `ffmpeg`, PyTorch 1.10.2 (CPU), Detectron2 (CPU, torch 1.10) e `requirements.txt`; inicia com `gunicorn --workers 1 --threads 8 --timeout 0 main:app` na porta `$PORT`. |
| `requirements.txt` | Dependências fixadas da API (Flask 2.0.2, gunicorn 20.1.0, opencv-python, scikit-image, ExifRead, google-api-python-client etc.). O PyTorch/Detectron2 são instalados à parte (Dockerfile / passo de instalação abaixo). `google-cloud-storage` está listado, mas o `main.py` atual não o importa (a saída remota usa Google Drive). |
| `app_settings.json` | Configuração lida por `main.py` (ver abaixo). |
| `.dockerignore`, `.gcloudignore`, `.gitignore` | `.gcloudignore` inclui o `.gitignore`; o `.gitignore` local ignora `venv/`. O `.gitignore` da raiz ignora `cloud-run-function/recognizer.py` e `cloud-run-function/models/`. |

## Preparar os arquivos (não versionados)

O `main.py` faz `from recognizer import Recognizer` e carrega `models/model/config.yaml` e `models/model/weights.pkl` (caminhos relativos ao diretório de execução, `/app` no container). Nenhum dos dois está no git, então copie-os antes de rodar/construir:

```bash
# a partir de cloud-run-function/
cp ../recognizer.py recognizer.py
mkdir -p models/model
cp '../models/mask_rcnn_X_101_32x8d_FPN_3x/mask_rcnn_X_101_32x8d_FPN_3x.yaml' models/model/config.yaml
cp '../models/mask_rcnn_X_101_32x8d_FPN_3x/model_final_2d9806.pkl' models/model/weights.pkl   # pesos .pkl não versionados; obtenha-os (ex.: Google Drive, ver README da raiz)
```

(O README anterior usava `model/` em vez de `models/model/` e um `ln -s` para o `recognizer.py`; o código atual espera `models/model/...`. Como o `Dockerfile` faz `COPY . ./`, prefira copiar o arquivo em vez de usar link simbólico.)

## Instalação local

```bash
virtualenv -p python3 venv
source venv/bin/activate

pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.10/index.html
pip install torch==1.10.2+cpu torchvision==0.11.3+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html
pip install -r requirements.txt
```

## Executar

```bash
python main.py          # Flask em modo debug, 0.0.0.0:$PORT (padrão 8080)
```

## Configuração (`app_settings.json`)

| Chave | Uso |
|---|---|
| `output_storage_type` | `local` (padrão) grava em disco; `gdrive` envia para o Google Drive. |
| `local_storage_path_generated_files` / `local_storage_path_raw_files` | Pastas de saída no modo local (padrão `/app/output/generated_files` e `/app/output/raw_files`). |
| `google_drive_upload_folder_id` | ID da pasta do Drive (no repositório está o placeholder `COLE_O_ID_DA_PASTA_DO_GOOGLE_DRIVE_AQUI`). |
| `service_account_key_path` | Caminho do JSON da conta de serviço (padrão `gdrive_service_account.json`; **não versionar**). |
| `allow_raw_upload_to_gdrive` | Se `true`, também envia a foto original ao Drive em `/image` e `/url`. |
| `generated_files_parent_folder_name` | Nome da subpasta criada no Drive (`JuizDeBochaUploadsApp`). |
| `default_thumbnail_size` | Tamanho da miniatura, `[300, 300]`. |

## Endpoints

Todos são `POST` e esperam os **bytes da imagem** no corpo (não é multipart). Parâmetro opcional `userID` (query) cria subpasta `user_<id>` na saída; `with-thumbnail` (query, presença basta) inclui miniatura.

| Rota | Resposta |
|---|---|
| `/image` | GIF animado (anexo). Com `with-thumbnail`: JSON `{thumbnail, gif}` em base64. |
| `/url` | Referência (caminho local ou `webViewLink` do Drive) do GIF gerado. Com `with-thumbnail`: JSON `{thumbnail, animated}`. |
| `/coordinates` | JSON `{url, smallest, winner, balls}` (contrato original): `url` é a referência do arquivo salvo (Drive/local); `smallest` é o bolim, `winner` a bola vencedora e `balls` as demais; cada item traz `center` (x, y normalizados 0–1) e `ellipse` (width, height normalizados). |

```bash
curl -X POST --data-binary "@foto.jpg" "http://localhost:8080/coordinates"
curl -X POST --data-binary "@foto.jpg" "http://localhost:8080/image?with-thumbnail" 
```

## Docker

```bash
# build
docker build --tag test .
# run
docker run -e PORT=8080 --publish 8080:8080 test
```

Também é possível subir pelo `docker-compose.yml` da raiz (serviço `app_recognizer`, porta 8000), que monta `models_from_gdrive/config.yaml` e `models_from_gdrive/weights.pkl` em `/app/models/model/`.

## Deploy

```bash
gcloud run deploy recognizer \
  --region southamerica-east1 \
  --project juizdebocha \
  --source .

#or 

#gcloud builds submit \
#  --region southamerica-east1 \
#  --project juizdebocha \
#  --tag southamerica-east1-docker.pkg.dev/juizdebocha/cloud-run-source-deploy/recognizer \
#  .
#
#gcloud run deploy recognizer \
#  --region southamerica-east1 \
#  --project juizdebocha \
#  --image southamerica-east1-docker.pkg.dev/juizdebocha/cloud-run-source-deploy/recognizer:latest
```

Antes do deploy, `recognizer.py` e `models/model/` precisam existir nesta pasta (ver "Preparar os arquivos"). O `.gcloudignore` desta pasta só inclui o `.gitignore` local (`venv/`), então esses arquivos, mesmo ignorados pelo `.gitignore` da raiz, ainda são enviados ao build.

## Logs

```bash
gcloud functions logs read --limit 50 --project juizdebocha
gcloud functions logs read --limit 30 \
  --project juizdebocha \
  --region southamerica-east1 \
  process_game_image
```

(Comandos mantidos do README original: usam `gcloud functions`, e o nome `process_game_image` não aparece no código atual, onde o serviço é o `recognizer` no Cloud Run.)
