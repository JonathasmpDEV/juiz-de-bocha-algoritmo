# cloud-run-hello-world

Exemplo mínimo de serviço para o Google Cloud Run: uma app Flask (`helloworld/main.py`) com a rota `GET /` que responde `Hello <NAME>!` (variável de ambiente `NAME`, padrão `World`). O `helloworld/Dockerfile` usa `python:3.10-slim`, instala `Flask==2.0.2` e `gunicorn==20.1.0` e inicia com `gunicorn --workers 1 --threads 8 --timeout 0 main:app`.

> **Status:** pasta de exemplo/legada. Nenhum outro arquivo do repositório a utiliza e o último commit é de 2022-03. O serviço real do projeto está em [`../cloud-run-function`](../cloud-run-function/README.MD).

## Estrutura

```
cloud-run-hello-world/
└── helloworld/
    ├── Dockerfile
    ├── main.py
    ├── requirements.txt
    └── .dockerignore
```

## Executar localmente

```bash
cd helloworld
pip install -r requirements.txt
python main.py          # http://localhost:8080 (ou a porta em $PORT)
```

## Deploy

Execute a partir de `cloud-run-hello-world/helloworld/` (onde estão o `Dockerfile`, o `main.py` e o `requirements.txt`):

```bash
gcloud run deploy \
  --region southamerica-east1 \
  --project juizdebocha \
  --source . \
  helloworld
```
