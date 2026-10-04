# Juiz de Bocha Eletrônico — algoritmo

Projeto de visão computacional que analisa fotos de uma jogada de bocha: detecta as bolas (classe `sports ball`) com um modelo **Mask R-CNN (Detectron2)**, identifica o bolim (a menor bola na região central da imagem), calcula qual bola está mais próxima dele (bola vencedora) e devolve o resultado como GIF, miniatura ou coordenadas normalizadas via API HTTP (Flask + gunicorn, empacotada em Docker para o Google Cloud Run).

A documentação completa (em Docsify) está em [`DOCUMENTACAO.md`](DOCUMENTACAO.md). Este README é um guia rápido e aponta para os READMEs de cada pasta.

## Estrutura do repositório

| Pasta / arquivo | O que é | README |
|---|---|---|
| `recognizer.py` | Classe `Recognizer`: carrega config/pesos do Detectron2, lê imagens (corrigindo orientação EXIF) e faz a predição (`predict_data`, `predict_instances`, `predict_img`, `test`). Roda em CPU, ou em GPU se `gpu_id` for informado. | — |
| `cloud-run-function/` | API Flask do reconhecedor (endpoints `/image`, `/url`, `/coordinates`), `Dockerfile` e `app_settings.json`. É o serviço publicado no Cloud Run. | [README](cloud-run-function/README.MD) |
| `train/` | Treinamento inicial do modelo a partir de um subconjunto "sports ball" do COCO. | [README](train/README.MD) |
| `train_custom_dataset/` | Pipeline numerado (1 a 4) para gerar o dataset próprio (auto-rotulação), treinar e validar casos de erro. | [README](train/README.MD) (seção própria) |
| `models/` | Configs YAML do Detectron2 (base, R-50, X-101 e as do Juiz de Bocha). Pesos (`*.pkl`, `*.pth`) não são versionados. | [README](train/README.MD) (seção "Modelos") |
| `detectron2-tools/` | Cópia dos scripts utilitários do Detectron2 (treino, benchmark, análise, visualização, exportação). | [README](detectron2-tools/README.md), [deploy](detectron2-tools/deploy/README.md) |
| `detectron2-test/` | `demo.py` e `predictor.py` do demo oficial do Detectron2 (usados por `models/mask_rcnn_X_101_32x8d_FPN_3x/run-demo.sh`). | — |
| `cloud-run-hello-world/` | Exemplo mínimo Flask para testar deploy no Cloud Run. Ver nota de obsolescência abaixo. | [README](cloud-run-hello-world/README.MD) |
| `test-cloudrun/` | Clientes de teste em Dart para a API. Ver nota abaixo. | — |
| `old-tests/` | Scripts de teste antigos. Ver nota abaixo. | — |
| `docs-server/` + `docker-compose.yml` + `index.html` + `_sidebar.md` + `DOCUMENTACAO.md` | Documentação servida com Docsify + Nginx (serviço `docsify_docs`, porta 8080). | — |
| `gdrive_auth.py`, `gdrive_download.py`, `gdrive_config.json`, `config_gdrive.md` | Download de datasets/modelos do Google Drive (ver abaixo). | — |
| `test-recognizer.py` | Script manual: carrega o `Recognizer` com o modelo R-50 FPN 3x, mede tempo de startup/predição e abre uma janela OpenCV com o resultado. | — |
| `data.json` | Exemplo de saída de `Recognizer.predict_data` (centro, caixa etc. de cada instância). | — |
| `datasets` | Link simbólico versionado cujo alvo é um caminho local da máquina do autor original (`/media/viniciusas/Elements/Datasets/JuizDeBocha`). Os datasets em si não estão no repositório. | — |

### Pastas legadas / possivelmente obsoletas (fatos verificáveis)

- `cloud-run-hello-world/`: é um "hello world" Flask (`GET /` responde `Hello <NAME>!`). Nenhum outro arquivo do repositório o referencia; último commit em 2022-03. Serve apenas como exemplo de deploy.
- `old-tests/`: `test-detectron.py` é idêntico a `detectron2-test/demo.py`; `test-main.py` importa `main` e chama `main.process_image` / `main.rec`, que não existem no `cloud-run-function/main.py` atual (lá as funções são `_process_image` e `_rec`). Último commit em 2022-05 ("Removidos arquivos antigos").
- `test-cloudrun/`: clientes Dart (`test-file.dart`, `test-url.dart`, `process_image.dart`, `processar_jogo_de_bocha.dart`) que apontam para `http://localhost:8080` ou para uma URL do Cloud Run. Contém `.packages` e `.dart_tool/` versionados (formato antigo de pub). Último commit em 2022-10. Alguns clientes enviam parâmetros (acelerômetro, giroscópio etc.) que o `main.py` atual não lê.

## Pré-requisitos

- Python 3.8+ (o `Dockerfile` da API usa `python3.9`).
- Para o ambiente completo (treinamento/GPU): PyTorch 1.10.1 + CUDA 11.3 e Detectron2 0.6 compatível. Para a API em CPU: PyTorch 1.10.2 CPU. Veja `requirements.txt` (raiz) e `cloud-run-function/requirements.txt`.
- Docker / Docker Compose (para servir a documentação e/ou a API localmente).
- Google Cloud SDK (`gcloud`) para o deploy no Cloud Run.

## Instalação

Baseado em:
- Detectron2: https://detectron2.readthedocs.io/en/latest/tutorials/install.html
- PyTorch: https://pytorch.org/get-started/locally/

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# PyTorch (CUDA 11.3)
pip3 install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

# Detectron2 (CUDA 11.3 / torch 1.10)
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html

# demais dependências (o requirements.txt também fixa torch/detectron2 +cu113)
pip install -r requirements.txt
```

Para a variante CPU (usada pela API), veja o [README de `cloud-run-function/`](cloud-run-function/README.MD).

## Uso rápido

```bash
# Testar o reconhecedor localmente (precisa dos pesos e de uma imagem; edite os caminhos no script)
python test-recognizer.py

# Servir a documentação (e a API) com Docker Compose
docker-compose up -d        # docs: http://localhost:8080 | API: http://localhost:8000
```

Observações sobre o `docker-compose.yml`:
- `docsify_docs` (Nginx, porta 8080) monta `DOCUMENTACAO.md`, `index.html` e `_sidebar.md` em modo leitura.
- `app_recognizer` (porta 8000, `PORT=8000`) monta `./cloud-run-function` em `/app` e espera os arquivos `./models_from_gdrive/config.yaml`, `./models_from_gdrive/weights.pkl` e `./gdrive_service_account.json`, além de `cloud-run-function/recognizer.py` (que é ignorado pelo git; ver README da pasta).

## Dados e modelos no Google Drive

Os datasets e os pesos não ficam no git. Os scripts da raiz baixam-nos de uma pasta do Google Drive:

1. Preencha `gdrive_config.json` (chaves `google_drive_folder_url`, `local_datasets_path`, `local_models_path`; os valores atuais são placeholders). A página `config_gdrive.md` (servida pelo Docsify) gera esse JSON. A pasta raiz do Drive deve conter subpastas `datasets` e `models`.
2. Baixe as credenciais OAuth do Google Cloud Console e salve como `gdrive_credentials.json` na raiz; rode `python gdrive_auth.py` (escopo somente leitura; gera `gdrive_token.json`).
3. Baixe: `python gdrive_download.py --type datasets` ou `--type models` (opcional `--output_path <dir>`; padrões `./datasets_from_gdrive` e `./models_from_gdrive` quando a chave não existe no JSON).

Nunca versione `gdrive_credentials.json`, `gdrive_token.json` ou a chave de conta de serviço (`gdrive_service_account.json`). Nenhum deles está no repositório; os `.gitignore` atuais **não** os listam, então tome cuidado ao fazer `git add`.

## Deploy

Ver [`cloud-run-function/README.MD`](cloud-run-function/README.MD) (`gcloud run deploy recognizer --source .`, região `southamerica-east1`, projeto `juizdebocha`).

## Treinamento

Ver [`train/README.MD`](train/README.MD).
