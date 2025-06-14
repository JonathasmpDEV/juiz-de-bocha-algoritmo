# Documentação do Projeto Juiz de Bocha Eletrônico

Bem-vindo à documentação oficial do projeto Juiz de Bocha Eletrônico.
Este documento visa fornecer uma visão completa da arquitetura, implementação e uso do sistema.

## Sumário
- [1. Visão Geral do Projeto](#1-visão-geral-do-projeto)
- [2. Configuração do Ambiente de Desenvolvimento](#2-configuração-do-ambiente-de-desenvolvimento)
- [3. O Reconhecedor (`recognizer.py`)](#3-o-reconhecedor-recognizerpy)
- [4. Treinamento de Modelos](#4-treinamento-de-modelos)
- [5. Servindo o Modelo com Google Cloud Run (`cloud-run-function/`)](#5-servindo-o-modelo-com-google-cloud-run-cloud-run-function)
- [6. Estrutura de Diretórios do Projeto](#6-estrutura-de-diretórios-do-projeto)
- [7. Servindo a Documentação Web com Docker](#7-servindo-a-documentação-web-com-docker)


## 1. Visão Geral do Projeto

O projeto **Juiz de Bocha Eletrônico** tem como objetivo principal automatizar a análise de jogadas de bocha a partir de imagens. Utilizando visão computacional, o sistema é capaz de identificar as bolas de bocha e o bolim (bola alvo) em uma imagem, determinar suas posições e, potencialmente, auxiliar na contagem de pontos e na identificação da bola vencedora.

### Tecnologias Utilizadas

O projeto é construído sobre as seguintes tecnologias principais:

*   **Python:** Linguagem de programação principal para desenvolvimento.
*   **Detectron2:** Uma plataforma de detecção de objetos de última geração da Facebook AI Research (FAIR), construída sobre o PyTorch. É utilizada para treinar modelos customizados de detecção e segmentação das bolas de bocha.
*   **PyTorch:** Biblioteca de aprendizado de máquina de código aberto.
*   **OpenCV:** Biblioteca de visão computacional utilizada para manipulação de imagens.
*   **Google Cloud Run:** Plataforma para executar containers stateless via HTTP. Utilizada para hospedar a API de reconhecimento de imagens como um serviço escalável.
*   **Docker:** Plataforma para desenvolver, enviar e executar aplicativos em containers.

### Arquitetura de Alto Nível

A arquitetura do sistema pode ser simplificada da seguinte forma:

```mermaid
graph TD
    A[Usuário/Aplicação Cliente] -- Upload de Imagem (HTTP Request) --> B(Google Cloud Run: API Endpoint);
    B -- Imagem --> C{`cloud-run-function/main.py`};
    C -- Processar Imagem --> D[`recognizer.py: Recognizer`];
    D -- Carrega Modelo e Pesos --> E[Modelo Detectron2 (.pkl, .yaml)];
    D -- Realiza Predição --> F[Resultado da Detecção (JSON)];
    C -- Retorna Resultado --> B;
    B -- Resposta HTTP (JSON) --> A;
```

Este diagrama ilustra o fluxo básico:
1.  O usuário ou uma aplicação cliente envia uma imagem para a API hospedada no Google Cloud Run.
2.  A função principal (`main.py`) no Cloud Run recebe a imagem.
3.  A classe `Recognizer` (`recognizer.py`) é instanciada, carregando o modelo Detectron2 treinado.
4.  O `Recognizer` processa a imagem usando o modelo para detectar as bolas.
5.  As informações sobre as bolas detectadas (posições, classes, etc.) são formatadas em JSON.
6.  A API retorna o resultado JSON para o cliente.

## 2. Configuração do Ambiente de Desenvolvimento

Esta seção descreve os passos para configurar o ambiente de desenvolvimento local para trabalhar com o projeto Juiz de Bocha Eletrônico.

### Pré-requisitos

*   **Python:** Recomenda-se a versão 3.8 ou superior. Você pode baixar Python em [python.org](https://www.python.org/).
*   **pip:** O gerenciador de pacotes do Python, geralmente instalado automaticamente com o Python.
*   **Git:** Para clonar o repositório.

### Passos para Configuração

Após clonar o repositório e navegar para o diretório do projeto, siga estes passos:

1.  **Criar e Ativar um Ambiente Virtual (Recomendado):**
    É uma boa prática usar ambientes virtuais para isolar as dependências do projeto.
    ```bash
    python -m venv venv
    ```
    Para ativar o ambiente virtual:
    *   No Linux/macOS:
        ```bash
        source venv/bin/activate
        ```
    *   No Windows:
        ```bash
        venv\Scripts\activate
        ```
    Você saberá que o ambiente virtual está ativo pelo prefixo `(venv)` no seu terminal.

2.  **Instalar PyTorch:**
    O projeto utiliza PyTorch. Recomenda-se instalar uma versão compatível com CUDA se você tiver uma GPU NVIDIA para acelerar o treinamento e a inferência. Caso contrário, a versão CPU pode ser usada.

    *   **Com Suporte a CUDA (exemplo para CUDA 11.3 e PyTorch 1.10.1, ajuste conforme sua GPU e o `README.MD`):**
        ```bash
        pip3 install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
        ```
    *   **Versão CPU (conforme `cloud-run-function/README.MD`):**
        ```bash
        pip install torch==1.10.2+cpu torchvision==0.11.3+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html
        ```
    Consulte o `README.MD` principal e o `cloud-run-function/README.MD` para as versões exatas e os links de download do PyTorch, pois podem variar dependendo da configuração (CPU/GPU) e atualizações do Detectron2.

3.  **Instalar Detectron2:**
    Após o PyTorch, instale o Detectron2.
    *   **Com Suporte a CUDA (exemplo para CUDA 11.3 e PyTorch 1.10, ajuste conforme sua GPU e o `README.MD`):**
        ```bash
        python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html
        ```
    *   **Versão CPU (conforme `cloud-run-function/README.MD`):**
        ```bash
        pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.10/index.html
        ```

4.  **Instalar Outras Dependências:**
    O projeto utiliza dois arquivos `requirements.txt` principais:
    *   `requirements.txt` (na raiz): Contém as dependências para desenvolvimento geral, treinamento de modelos e execução de scripts locais. Instale com:
        ```bash
        pip install -r requirements.txt
        ```
    *   `cloud-run-function/requirements.txt`: Contém as dependências específicas para o ambiente de execução da função em Google Cloud Run. Este pode ser um subconjunto do principal e pode especificar versões de CPU de bibliotecas para otimizar a imagem Docker. Se estiver trabalhando diretamente com a função em nuvem ou construindo sua imagem, instale-o no ambiente apropriado:
        ```bash
        pip install -r cloud-run-function/requirements.txt
        ```

5.  **Verificação (Opcional):**
    Para verificar se o Detectron2 foi instalado corretamente, você pode tentar importar a biblioteca em um interpretador Python:
    ```python
    import detectron2
    print(detectron2.__version__)
    ```

Com esses passos, seu ambiente de desenvolvimento deve estar pronto para executar e modificar o código do projeto.

## 3. O Reconhecedor (`recognizer.py`)

O coração do sistema de detecção de objetos é a classe `Recognizer`, definida no arquivo `recognizer.py`. Esta classe encapsula toda a lógica necessária para carregar um modelo Detectron2 treinado, processar uma imagem de entrada e retornar informações estruturadas sobre os objetos detectados.

### Finalidade

A classe `Recognizer` é projetada para:

*   Abstrair os detalhes de baixo nível da inferência com Detectron2.
*   Fornecer uma interface simples para realizar predições em imagens.
*   Lidar com o pré-processamento de imagens, incluindo a correção de orientação EXIF.
*   Formatar os resultados da detecção em uma estrutura de dados Python fácil de usar.

### Parâmetros de Configuração no `__init__`

Ao instanciar a classe `Recognizer`, os seguintes parâmetros são necessários:

*   `config_file (str)`: Caminho para o arquivo de configuração YAML do modelo Detectron2 (ex: `models/mask_rcnn_R_50_FPN_3x/mask_rcnn_R_50_FPN_3x.yaml`). Este arquivo define a arquitetura do modelo, os datasets de treinamento/teste, e outros hiperparâmetros.
*   `weights_file (str)`: Caminho para o arquivo de pesos do modelo treinado (ex: `models/mask_rcnn_R_50_FPN_3x/model_final_2d9806.pkl`).
*   `confidence_threshold (float, opcional)`: Um valor entre 0 e 1 que define o limiar de confiança mínimo para que uma detecção seja considerada válida. Detecções com pontuação (score) abaixo deste limiar são descartadas. O padrão é `0.5`.
*   `gpu_id (int, opcional)`: O ID da GPU a ser utilizada para a inferência. Se `None` (padrão), a inferência será realizada na CPU.

```python
# Exemplo de instanciação
from recognizer import Recognizer

recognizer = Recognizer(
    config_file='models/mask_rcnn_juiz_de_bocha/mask_rcnn_juiz_de_bocha.yaml',
    weights_file='models/mask_rcnn_juiz_de_bocha/model_final.pkl',
    confidence_threshold=0.7
)
```

### Principais Métodos

A seguir, uma descrição dos métodos mais importantes da classe:

*   **`__init__(self, config_file: str, weights_file: str, confidence_threshold: float = 0.5, gpu_id=None)`**
    *   Construtor da classe.
    *   Inicializa a configuração do Detectron2 (`cfg`) a partir do `config_file`.
    *   Define o caminho dos pesos (`cfg.MODEL.WEIGHTS`).
    *   Ajusta o limiar de confiança para diferentes partes do modelo (RetinaNet, ROI Heads, Panoptic FPN).
    *   Configura o dispositivo de processamento (`cfg.MODEL.DEVICE`) para CPU ou GPU.
    *   Cria uma instância do `DefaultPredictor` do Detectron2, que é o objeto responsável por realizar as predições.
    *   Carrega os metadados do dataset (usado para obter nomes de classes).
    *   Define `self._class_names` a partir dos metadados (`metadata.get("thing_classes")`). Estes são os nomes das categorias de objetos que o modelo foi treinado para detectar (ex: 'bola_vermelha', 'bolim'). Inclui um fallback para `['sports ball']` caso os `thing_classes` não sejam encontrados nos metadados, útil para modelos genéricos ou como um padrão de segurança.

*   **`read_file(file_path_or_bytesio) -> np.ndarray`** (Método Estático)
    *   Lê um arquivo de imagem. O argumento `file` pode ser uma string com o caminho para o arquivo ou um objeto `io.BytesIO` (útil para ler imagens de uploads web, por exemplo).
    *   Utiliza o método privado `__read_img_and_correct_exif_orientation` para carregar a imagem e corrigir sua orientação com base nos dados EXIF.
    *   Converte a imagem para o formato RGB (Detectron2 espera BGR por padrão, mas a conversão ocorre internamente no `predict_instances`).
    *   Retorna a imagem como um array NumPy.

*   **`__read_img_and_correct_exif_orientation(file) -> Image`** (Método Estático Privado)
    *   Lê os metadados EXIF da imagem para identificar a tag "Image Orientation".
    *   Se a tag existir e indicar uma orientação diferente da padrão, a imagem (objeto PIL `Image`) é transposta (rotacionada/invertida) apropriadamente para a orientação correta.
    *   Retorna o objeto `Image` corrigido.

*   **`predict_instances(self, img: np.ndarray) -> detectron2.structures.Instances`**
    *   Recebe uma imagem como um array NumPy no formato RGB.
    *   Converte a imagem de RGB para BGR, pois o `DefaultPredictor` do Detectron2 espera imagens nesse formato.
    *   Passa a imagem BGR para o `self.__predictor` para obter as predições.
    *   As predições são retornadas como um objeto `Instances` do Detectron2, que contém informações detalhadas sobre cada objeto detectado (caixas delimitadoras, máscaras de segmentação, pontuações, classes).
    *   As instâncias são movidas para a CPU (`self.__cpu_device`) para facilitar o processamento subsequente.

*   **`predict_data(self, img: np.ndarray) -> list[dict]`**
    *   Este é o método principal para obter dados de detecção formatados.
    *   Chama `predict_instances(img)` para obter as detecções brutas.
    *   Extrai as seguintes informações para cada instância detectada:
        *   `pred_boxes`: Caixas delimitadoras dos objetos.
        *   `centers`: Coordenadas (x, y) do centro de cada caixa delimitadora.
        *   `scores`: Pontuação de confiança da detecção.
        *   `pred_classes`: Índice da classe predita.
        *   `pred_masks`: Máscaras de segmentação para cada objeto.
    *   Monta uma lista de dicionários, onde cada dicionário representa um objeto detectado e contém:
        *   `center`: Dicionário com `x` e `y` do centro.
        *   `box`: Dicionário com `x1`, `y1`, `x2`, `y2` (coordenadas do canto superior esquerdo e inferior direito da caixa).
        *   `mask`: Array NumPy da máscara de segmentação do objeto (valores booleanos ou 0/1).
        *   `class`: Nome da classe do objeto (ex: 'bola_vermelha', 'bola_azul', 'bolim').
        *   `score`: Pontuação de confiança da detecção.
        *   `area`: A área da máscara de segmentação em pixels.
    *   Retorna a lista de dicionários.

*   **`predict_img(self, img: np.ndarray) -> tuple[Instances, np.ndarray]`**
    *   Chama `predict_instances(img)` para obter as detecções.
    *   Utiliza a classe `Visualizer` do Detectron2 para desenhar as caixas delimitadoras, máscaras e rótulos sobre a imagem original.
    *   Retorna uma tupla contendo as `Instances` brutas e a imagem com as visualizações desenhadas (como um array NumPy).

*   **`test(self, img: np.ndarray)`**
    *   Um método utilitário para testar rapidamente o reconhecedor em uma imagem.
    *   Chama `predict_img(img)` para obter as instâncias e a imagem visualizada.
    *   Imprime o número de instâncias detectadas.
    *   Exibe a imagem com as detecções em uma janela OpenCV chamada "prediction".
    *   Aguarda o usuário pressionar a tecla ESC para fechar a janela.

### Fluxo de Predição Típico

1.  Carregar a imagem usando `Recognizer.read_file()`.
2.  Passar a imagem (array NumPy) para o método `recognizer.predict_data()` para obter a lista de objetos detectados com seus atributos.
3.  Se necessário, passar a imagem para `recognizer.predict_img()` para obter uma visualização das detecções.

Este componente é fundamental para a funcionalidade do Juiz de Bocha Eletrônico, permitindo a interpretação das imagens do jogo.

## 4. Treinamento de Modelos

Esta seção aborda o processo de treinamento de modelos de detecção de objetos customizados para o Juiz de Bocha Eletrônico, utilizando Detectron2. O treinamento permite que o sistema aprenda a identificar objetos específicos (bolas de bocha de diferentes cores, bolim) em novas imagens.

Os principais scripts e configurações para treinamento residem nos diretórios `train/`, `models/`, e `train_custom_dataset/`. As ferramentas em `detectron2-tools/` também são relevantes para este processo.

### Processo Geral de Treinamento

O treinamento de um modelo com Detectron2 geralmente envolve os seguintes passos:

1.  **Preparação do Dataset:**
    *   Coletar e anotar imagens. As anotações especificam as bounding boxes (caixas delimitadoras) e, opcionalmente, máscaras de segmentação para cada objeto de interesse em cada imagem.
    *   Converter as anotações para um formato que o Detectron2 entenda (comumente o formato COCO JSON). O diretório `train_custom_dataset/` contém scripts que auxiliam nessa tarefa (ex: `2_generate_labels_files.py`), sugerindo um fluxo de trabalho que pode envolver ferramentas como LabelImg ou LabelMe para criar as anotações e depois convertê-las.
    *   Registrar o dataset customizado no Detectron2 para que ele possa ser carregado durante o treinamento.

2.  **Configuração do Modelo:**
    *   Escolher uma arquitetura de modelo base (ex: Mask R-CNN com ResNet-50 FPN) a partir do "model zoo" do Detectron2 ou de um modelo previamente treinado.
    *   Criar um arquivo de configuração YAML (ex: `models/mask_rcnn_juiz_de_bocha/mask_rcnn_juiz_de_bocha.yaml`). Este arquivo define:
        *   Arquitetura do modelo.
        *   Caminhos para os pesos pré-treinados (se aplicável).
        *   Nomes dos datasets de treino e teste registrados.
        *   Hiperparâmetros de treinamento como taxa de aprendizado, número de iterações, tamanho do batch, etc.
        *   Configurações específicas do solver e do data loader.

3.  **Execução do Treinamento:**
    *   Utilizar um script de treinamento, como `train/train.py`, que carrega a configuração, o dataset, e inicia o loop de treinamento.

4.  **Avaliação e Iteração:**
    *   Monitorar as métricas de treinamento (loss, mAP - Mean Average Precision) para avaliar o desempenho do modelo.
    *   Ajustar hiperparâmetros, aumentar o dataset, ou modificar a arquitetura do modelo e treinar novamente até atingir a performance desejada.
    *   Os modelos treinados (arquivos `.pkl`) e as métricas são geralmente salvos no diretório de saída especificado na configuração (ex: `train/training/`).

### Script Principal de Treinamento: `train/train.py`

O script `train/train.py` é o ponto de entrada para iniciar o processo de treinamento. Suas principais responsabilidades incluem:

*   **Registro de Datasets:**
    ```python
    import register_datasets
    register_datasets.juiz_de_bocha()
    ```
    Este trecho chama funções do arquivo `train/register_datasets.py` (ou `train_custom_dataset/register_datasets.py`, dependendo da configuração exata do projeto) que utilizam as funções `register_coco_instances` ou similares do Detectron2 para informar à biblioteca onde encontrar as imagens e os arquivos de anotação JSON para os datasets de treino e validação.

*   **Carregamento da Configuração:**
    ```python
    cfg = get_cfg()
    cfg.merge_from_file('models/mask_rcnn_juiz_de_bocha/mask_rcnn_juiz_de_bocha.yaml')
    ```
    Um objeto de configuração padrão do Detectron2 é criado e, em seguida, mesclado com as configurações específicas do projeto definidas no arquivo YAML.

*   **Definição de Hiperparâmetros:**
    O script ajusta vários hiperparâmetros diretamente no objeto `cfg`:
    ```python
    cfg.DATALOADER.NUM_WORKERS = 8  # Número de threads para carregar dados
    cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"  # Ponto de partida (pesos pré-treinados do COCO)
    cfg.SOLVER.IMS_PER_BATCH = 2  # Imagens por batch
    cfg.SOLVER.BASE_LR = 0.005  # Taxa de aprendizado inicial
    cfg.SOLVER.MAX_ITER = 15000  # Número máximo de iterações de treinamento
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128 # Número de ROIs por imagem durante o treino
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  # IMPORTANTE: Descomente e ajuste este valor para o número exato de classes do seu dataset!
    cfg.OUTPUT_DIR = './train/training/' # Diretório para salvar os modelos e logs
    ```
    É crucial ajustar `MODEL.ROI_HEADS.NUM_CLASSES` para o número correto de categorias de objetos que seu modelo precisa detectar (ex: bola vermelha, bola azul, bolim = 3 classes).

*   **Trainer e Avaliação:**
    ```python
    class Trainer(DefaultTrainer):
        @classmethod
        def build_evaluator(cls, cfg, dataset_name, output_folder=None):
            return DatasetEvaluators([
                COCOEvaluator(dataset_name, output_dir=output_folder)
            ])

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=True) # Tenta resumir o treinamento se interrompido, senão carrega os pesos
    trainer.train()
    ```
    Uma classe `Trainer` customizada é definida para herdar de `DefaultTrainer` do Detectron2. Isso permite sobrescrever o método `build_evaluator` para configurar o `COCOEvaluator`, que calcula métricas de performance (como mAP) no dataset de validação durante o treinamento. O treinamento é então iniciado com `trainer.train()`.

### Arquivos de Configuração de Modelo (`models/`)

O diretório `models/` armazena os arquivos de configuração YAML para diferentes modelos ou experimentos. Por exemplo:

*   `models/Base-RCNN-FPN.yaml`: Pode conter configurações base comuns.
*   `models/mask_rcnn_juiz_de_bocha/mask_rcnn_juiz_de_bocha.yaml`: Configuração específica para o modelo treinado para o Juiz de Bocha.

Estes arquivos YAML são cruciais, pois definem a arquitetura do modelo, os datasets a serem usados, e todos os hiperparâmetros de treinamento.

### Ferramentas em `detectron2-tools/`

O diretório `detectron2-tools/` contém scripts fornecidos pelo Detectron2 que são úteis:

*   `analyze_model.py`: Para analisar FLOPs, parâmetros e ativações de um modelo.
*   `visualize_data.py`: Para visualizar as anotações do dataset ou os dados após o pré-processamento/aumentations.
*   `visualize_json_results.py`: Para visualizar os resultados de detecção salvos em formato JSON.
*   Consulte `detectron2-tools/README.md` para mais detalhes sobre cada ferramenta.

### Preparação de Datasets (`train_custom_dataset/`)

O diretório `train_custom_dataset/` parece conter scripts específicos para o fluxo de trabalho de criação e preparação de datasets para o projeto Juiz de Bocha:

*   `1_test_results.py`: Provavelmente para testar um modelo treinado em um conjunto de imagens.
*   `2_generate_labels_files.py`: Crucial para converter anotações (possivelmente de um formato como XML do LabelImg) para o formato JSON COCO esperado pelo Detectron2.
*   `3_train.py`: Pode ser uma versão alternativa ou mais específica do script de treinamento.
*   `4_cases_validation.py`: Para validar o modelo em casos específicos ou um conjunto de validação.
*   `register_datasets.py`: Script para registrar os datasets customizados.

A documentação detalhada de cada um desses scripts exigiria uma análise mais aprofundada do seu código, mas eles indicam um pipeline estabelecido para lidar com dados de treinamento customizados.

Treinar modelos de deep learning é um processo iterativo. Espere experimentar diferentes configurações, aumentar seus dados e analisar os resultados cuidadosamente para alcançar o melhor desempenho para a tarefa de Juiz de Bocha.

## 5. Servindo o Modelo com Google Cloud Run (`cloud-run-function/`)

Uma vez que um modelo de detecção de bocha satisfatório é treinado, ele pode ser implantado como um serviço web para que aplicações externas possam enviar imagens e receber os resultados da detecção. O diretório `cloud-run-function/` contém o necessário para empacotar o `Recognizer` em uma função do Google Cloud Run.

### Visão Geral

A função em nuvem expõe um endpoint HTTP que aceita uploads de imagens. Internamente, ela utiliza a classe `Recognizer` (descrita anteriormente) para processar a imagem e retorna os resultados da detecção em formato JSON.

O `cloud-run-function/README.MD` é a principal referência para os comandos de instalação e deploy.

### Arquivo Principal: `cloud-run-function/main.py`

O script `main.py` dentro de `cloud-run-function/` é o ponto de entrada da aplicação Flask (ou FastAPI, dependendo da implementação exata, o `README.MD` sugere Flask com `python main.py`) que serve o modelo.

**Principais Funcionalidades do `main.py` (esperado):**

*   **Inicialização do `Recognizer`:** No início da aplicação, uma instância do `Recognizer` é criada, carregando o arquivo de configuração (`model/config.yaml`) e os pesos do modelo (`model/weights.pkl`) especificados. Note que, conforme o `cloud-run-function/README.MD`, links simbólicos ou cópias desses arquivos são criados dentro do diretório `cloud-run-function/model/` durante a configuração.
*   **Definição de Rota HTTP:** Define uma rota (ex: `/predict` ou `/`) que aceita requisições HTTP POST.
*   **Processamento da Requisição:**
    *   Recebe o arquivo de imagem da requisição (geralmente como `multipart/form-data`).
    *   Lê a imagem usando `Recognizer.read_file()`.
    *   Chama `recognizer.predict_data()` para obter as detecções.
    *   Retorna os resultados como uma resposta JSON.
*   **Tratamento de Erros:** Idealmente, inclui tratamento para casos como arquivos inválidos, erros de predição, etc.

*(Nota: Uma análise direta do `cloud-run-function/main.py` seria necessária para detalhar a implementação exata, mas o fluxo acima é o padrão para tais serviços.)*

### Configuração e Arquivos

*   **`Dockerfile`**: Define como a imagem Docker para o serviço Cloud Run deve ser construída. Ele especifica a imagem base do Python, copia os arquivos necessários (código da aplicação, `requirements.txt`, modelo), instala as dependências e define o comando para iniciar a aplicação.
*   **`requirements.txt`**: Lista as dependências Python específicas para a função em nuvem (pode incluir Flask/FastAPI, gunicorn, além das bibliotecas de visão computacional como opencv-python-headless, e versões CPU do PyTorch e Detectron2 para otimizar o tamanho da imagem Docker e custos, se a GPU não for usada no Cloud Run).
*   **`.dockerignore`**: Especifica arquivos e diretórios que não devem ser incluídos na imagem Docker (ex: `venv/`, `__pycache__/`).
*   **`model/` (dentro de `cloud-run-function/`)**: Conforme o `README.MD`, este diretório é populado com o `config.yaml` e `weights.pkl` do modelo a ser usado pela função.

### Instalação Local (para Teste)

O `cloud-run-function/README.MD` fornece os seguintes passos para rodar a função localmente (geralmente para teste antes do deploy):

1.  **Criar links simbólicos ou copiar arquivos do modelo e `recognizer.py`:**
    ```bash
    # Navegue para o diretório cloud-run-function/
    cd cloud-run-function

    ln -s ../recognizer.py .  # ou cp ../recognizer.py .
    mkdir -p model
    cp '../models/mask_rcnn_X_101_32x8d_FPN_3x/mask_rcnn_X_101_32x8d_FPN_3x.yaml' model/config.yaml
    cp '../models/mask_rcnn_X_101_32x8d_FPN_3x/model_final_2d9806.pkl' model/weights.pkl
    # Adapte os caminhos acima para o modelo específico que deseja testar
    ```

2.  **Configurar ambiente virtual e instalar dependências (versão CPU):**
    ```bash
    virtualenv -p python3 venv
    source venv/bin/activate
    pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.10/index.html
    pip install torch==1.10.2+cpu torchvision==0.11.3+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html
    pip install -r requirements.txt
    ```

3.  **Iniciar a aplicação:**
    ```bash
    python main.py
    ```
    A aplicação Flask/FastAPI deve iniciar, geralmente na porta `8080`.

### Deploy no Google Cloud Run

O Google Cloud Run permite executar containers stateless que são invocados via requisições HTTP.

**Passos para Deploy (usando gcloud CLI):**

1.  **Navegue até o diretório `cloud-run-function/`:**
    ```bash
    cd cloud-run-function
    ```

2.  **Comando de Deploy (Source-based):**
    Este comando constrói a imagem a partir do código fonte e faz o deploy.
    ```bash
    gcloud run deploy recognizer       --region southamerica-east1       --project juizdebocha       --source .       --allow-unauthenticated # Adicione se desejar permitir acesso público
    ```
    *   `recognizer`: Nome do serviço no Cloud Run.
    *   `--region`: Região do Google Cloud onde o serviço será implantado (ex: `southamerica-east1`).
    *   `--project`: ID do seu projeto no Google Cloud.
    *   `--source .`: Indica que o código fonte está no diretório atual.
    *   `--allow-unauthenticated`: Permite que o serviço seja invocado sem autenticação. Remova se o acesso precisar ser restrito.

3.  **Alternativa: Build com Cloud Build e Deploy da Imagem:**
    Você pode primeiro construir a imagem Docker usando o Google Cloud Build e depois fazer o deploy da imagem a partir do Google Container Registry (gcr.io) ou Artifact Registry.
    ```bash
    # Submeter o build para o Cloud Build
    gcloud builds submit       --region southamerica-east1       --project juizdebocha       --tag southamerica-east1-docker.pkg.dev/juizdebocha/cloud-run-source-deploy/recognizer       .

    # Fazer o deploy da imagem construída
    gcloud run deploy recognizer       --region southamerica-east1       --project juizdebocha       --image southamerica-east1-docker.pkg.dev/juizdebocha/cloud-run-source-deploy/recognizer:latest       --allow-unauthenticated
    ```
    Substitua `southamerica-east1-docker.pkg.dev/juizdebocha/cloud-run-source-deploy/recognizer` pelo caminho correto do seu Artifact Registry, se diferente.

### Uso com Docker Localmente

O `README.MD` também fornece comandos para construir e rodar a imagem Docker localmente:

1.  **Construir a Imagem Docker:**
    (Certifique-se de estar no diretório `cloud-run-function/`)
    ```bash
    docker build --tag test .
    ```

2.  **Rodar o Container Docker:**
    ```bash
    docker run -e PORT=8080 --publish 8080:8080 test
    ```
    Isso expõe a porta 8080 do container na porta 8080 da sua máquina local.

### Visualização de Logs

Para monitorar a função e diagnosticar problemas, você pode visualizar os logs. Para Cloud Run, os logs são acessíveis através do Google Cloud Console na seção de Logging ou diretamente na página do serviço Cloud Run.

Para logs específicos do Cloud Run:
```bash
gcloud logging read "resource.type="cloud_run_revision" resource.labels.service_name="recognizer" resource.labels.location="southamerica-east1"" --project=juizdebocha --limit=50
```

Esta configuração permite que o modelo de detecção de bocha seja acessado como um serviço robusto e escalável.

## 6. Estrutura de Diretórios do Projeto

A seguir, uma descrição dos principais diretórios encontrados na raiz do projeto e suas finalidades:

*   **`.idea/`**: Diretório de configuração específico do IntelliJ IDEA ou outras IDEs da JetBrains. Geralmente não é versionado ou é incluído no `.gitignore`.
*   **`cloud-run-function/`**: Contém todo o código e configuração para empacotar e implantar o serviço de reconhecimento de imagens como uma função no Google Cloud Run. Inclui:
    *   `main.py`: Ponto de entrada da aplicação Flask/FastAPI que serve o modelo.
    *   `Dockerfile`: Instruções para construir a imagem Docker do serviço.
    *   `requirements.txt`: Dependências Python específicas para a função em nuvem.
    *   `README.MD`: Instruções de configuração, build e deploy para esta função.
    *   `model/`: Subdiretório (criado durante a configuração) para armazenar o arquivo de configuração (`.yaml`) e os pesos (`.pkl`) do modelo que será usado pela função.

*   **`detectron2-test/`**: Parece conter scripts de teste ou demonstração para o Detectron2 (ex: `demo.py`, `predictor.py`).
    *(Nota: O conteúdo exato e a relevância para o fluxo principal do projeto podem precisar de análise adicional se houver funcionalidades críticas aqui.)*

*   **`detectron2-tools/`**: Scripts e ferramentas fornecidos como parte da biblioteca Detectron2. São úteis para:
    *   Treinamento (`train_net.py`, `plain_train_net.py`).
    *   Benchmark de performance (`benchmark.py`).
    *   Análise de modelos (`analyze_model.py`).
    *   Visualização de dados e resultados (`visualize_data.py`, `visualize_json_results.py`).
    *   Deploy (contém um subdiretório `deploy/` com exemplos para exportar modelos).

*   **`models/`**: Armazena os arquivos de configuração de modelos (`.yaml`) e, potencialmente, os pesos de modelos treinados (`.pkl`, embora os pesos finais possam estar em diretórios de saída de treinamento). Organizado em subdiretórios por tipo ou versão de modelo. Exemplos:
    *   `Base-RCNN-FPN.yaml`: Configuração base.
    *   `mask_rcnn_R_50_FPN_3x/`: Configurações e métricas para um modelo específico.
    *   `mask_rcnn_juiz_de_bocha/`: Configurações para o modelo customizado do Juiz de Bocha.

*   **`old-tests/`**: Provavelmente contém scripts de teste legados ou experimentais.

*   **`test-cloudrun/`**: Contém scripts em Dart para testar a função implantada no Cloud Run, enviando imagens e processando as respostas.

*   **`train/`**: Contém os scripts principais e configurações para o treinamento de modelos Detectron2.
    *   `train.py`: Script principal para iniciar e gerenciar o processo de treinamento.
    *   `register_datasets.py`: Script para registrar datasets customizados no Detectron2.
    *   `plot.py`: Provavelmente para plotar métricas de treinamento ou resultados.
    *   `prepare_datasets.py`: Scripts para auxiliar na preparação de datasets.
    *   `training/`: Subdiretório (criado durante o treinamento) onde os modelos treinados, logs e outras saídas do treinamento são salvos.

*   **`train_custom_dataset/`**: Diretório dedicado aos scripts e processos específicos para a criação, anotação e formatação do dataset customizado para o projeto Juiz de Bocha. Inclui:
    *   Scripts para gerar arquivos de rótulos no formato COCO (`2_generate_labels_files.py`).
    *   Scripts de treinamento (`3_train.py`) e validação (`1_test_results.py`, `4_cases_validation.py`) possivelmente adaptados para o dataset customizado.
    *   `register_datasets.py`: Versão específica para os datasets customizados.

**Arquivos Relevantes na Raiz:**

*   **`recognizer.py`**: Script Python central que define a classe `Recognizer`, responsável por carregar modelos e realizar predições.
*   **`requirements.txt`**: Lista as dependências Python globais do projeto.
*   **`README.MD`**: Documentação inicial do projeto, geralmente com instruções de instalação.
*   **`DOCUMENTACAO.md`**: Este arquivo de documentação detalhada.
*   **`data.json`**: Não explorado, mas o nome sugere que pode conter dados de exemplo, configurações ou metadados.
*   **`test-recognizer.py`**: Script para testar a funcionalidade do `recognizer.py`.

Esta estrutura visa organizar os diferentes aspectos do projeto, desde a preparação de dados e treinamento de modelos até o deploy da aplicação em nuvem.

## 7. Servindo a Documentação Web com Docker

Para facilitar a visualização e o deploy da documentação web (gerada com Docsify.js), foi configurado um ambiente Docker com Nginx. Você pode servir a documentação localmente usando Docker Compose.

### Pré-requisitos

*   Docker instalado (consulte [docs.docker.com/get-docker/](https://docs.docker.com/get-docker/))
*   Docker Compose instalado (geralmente vem com o Docker Desktop, ou consulte [docs.docker.com/compose/install/](https://docs.docker.com/compose/install/))

### Passos para Execução

1.  **Navegue até a Raiz do Projeto:**
    Certifique-se de que você está no diretório raiz do projeto onde o arquivo `docker-compose.yml` está localizado.

2.  **Construir a Imagem Docker (se ainda não foi construída):**
    Este comando constrói a imagem Docker conforme definido no `docs-server/Dockerfile`.
    ```bash
    docker-compose build
    ```

3.  **Iniciar o Serviço de Documentação:**
    Este comando inicia o container Nginx em modo detached (`-d`), servindo a documentação.
    ```bash
    docker-compose up -d
    ```
    A documentação estará acessível no seu navegador em: `http://localhost:8080` (ou a porta que você configurou no `docker-compose.yml`).

    *Nota sobre Desenvolvimento:* O `docker-compose.yml` está configurado com volumes que montam os arquivos `DOCUMENTACAO.md`, `index.html` e `_sidebar.md` diretamente no container. Isso significa que se você editar esses arquivos no seu host, as alterações serão refletidas imediatamente ao recarregar a página no navegador, sem precisar reconstruir a imagem Docker (`docker-compose build`). Para um deploy em "produção", você pode comentar ou remover a seção `volumes` no `docker-compose.yml` para usar os arquivos que foram copiados para a imagem durante o build.

4.  **Visualizar Logs (Opcional):**
    Se precisar verificar os logs do servidor Nginx dentro do container:
    ```bash
    docker-compose logs docsify_docs
    ```
    Para seguir os logs em tempo real:
    ```bash
    docker-compose logs -f docsify_docs
    ```

5.  **Parar o Serviço de Documentação:**
    Para parar e remover os containers definidos no `docker-compose.yml`:
    ```bash
    docker-compose down
    ```

Com esses passos, você pode facilmente servir e visualizar a documentação do projeto em um ambiente containerizado.
