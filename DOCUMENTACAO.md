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
- [8. Gerenciamento de Dados com Google Drive](#8-gerenciamento-de-dados-com-google-drive)
- [9. Manual de Implantação com Docker](#9-manual-de-implantação-com-docker)
- [10. Manual de Uso da API com Flutter](#10-manual-de-uso-da-api-com-flutter)


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

### Executando a Aplicação Principal e a Documentação

O arquivo `docker-compose.yml` foi configurado para gerenciar tanto o servidor da documentação quanto a aplicação principal de reconhecimento de bocha.

Ao executar `docker-compose up -d`, ambos os serviços serão iniciados:

*   **Serviço de Documentação (`docsify_docs`):**
    *   Acessível em: `http://localhost:8080` (ou a porta configurada no `docker-compose.yml`).
    *   Serve a documentação interativa do projeto.

*   **Serviço da Aplicação Principal (`app_recognizer`):**
    *   Acessível em: `http://localhost:8000` (ou a porta configurada no `docker-compose.yml`).
    *   Expõe a API para reconhecimento de imagens de bocha. Os endpoints disponíveis são:
        *   `POST /image`: Envia uma imagem e recebe um GIF animado como resposta.
        *   `POST /url`: Envia uma imagem e recebe URLs para a imagem processada (GIF animado e thumbnail).
        *   `POST /coordinates`: Envia uma imagem e recebe as coordenadas das bolas detectadas e a URL da imagem original.

#### Testando a API da Aplicação Principal

Para testar os endpoints da API da aplicação (`app_recognizer`), você pode usar:

*   **Scripts em `test-cloudrun/`**: Este diretório contém scripts Dart (ex: `test-file.dart`, `test-url.dart`) que podem ser adaptados para enviar requisições para `http://localhost:8000` em vez do endpoint na nuvem.
*   **Ferramentas como Postman ou Insomnia**: Configure uma requisição POST para um dos endpoints acima, enviando uma imagem no corpo da requisição (como `form-data` ou `binary`).

**Exemplo de comando `curl` (para o endpoint `/url`, requer uma imagem `test.jpg` no mesmo diretório):**
Este é um exemplo mais complexo devido à necessidade de enviar um arquivo. Ferramentas como Postman são geralmente mais fáceis para isso.
```bash
# Este comando é um exemplo e pode precisar de ajustes.
# O endpoint /url espera o corpo da requisição como os bytes da imagem.
curl -X POST --data-binary "@test.jpg" -H "Content-Type: image/jpeg" http://localhost:8000/url
```
Lembre-se que a aplicação pode tentar interagir com o Google Cloud Storage. Se as credenciais não estiverem configuradas localmente, algumas funcionalidades de upload podem não funcionar completamente, mas o processamento da imagem ainda deve ocorrer.

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

## 8. Gerenciamento de Dados com Google Drive

Esta seção descreve como integrar o Google Drive para gerenciar datasets de treinamento e arquivos de modelo para este projeto. Isso permite um fluxo de trabalho mais flexível, onde os dados podem ser armazenados centralmente no Google Drive e baixados para o ambiente local conforme necessário.

### Visão Geral do Fluxo com Google Drive

1.  **Configuração Única**:
    *   Você cria um arquivo `gdrive_credentials.json` a partir do Google Cloud Console.
    *   Você cria um arquivo `gdrive_config.json` na raiz do projeto, especificando a URL da sua pasta principal no Google Drive que conterá os subdiretórios `datasets` e `models`.
    *   Você executa o script `gdrive_auth.py` uma vez para autorizar o acesso da aplicação ao seu Google Drive. Isso criará um `gdrive_token.json`.

2.  **Download de Datasets para Treinamento**:
    *   Antes de treinar, você executa `python gdrive_download.py --type datasets` para baixar a subpasta `datasets` do seu Google Drive para um diretório local (ex: `./datasets_gdrive`).
    *   Você ajusta os scripts `register_datasets.py` para apontar para este diretório local.

3.  **Download de Modelos para Inferência**:
    *   Antes de executar a aplicação `app_recognizer` (via `docker-compose up`), você executa `python gdrive_download.py --type models` para baixar a subpasta `models` do seu Google Drive para um diretório local (ex: `./models_from_gdrive`).
    *   Você garante que os arquivos baixados nesta pasta sejam nomeados `config.yaml` e `weights.pkl` para que o `docker-compose.yml` possa montá-los corretamente para a aplicação.

### Configuração da Conta de Serviço para Uploads da Aplicação

Para que a aplicação `app_recognizer` (rodando via Docker) possa salvar arquivos automaticamente no Google Drive, você precisará configurar uma Conta de Serviço do Google Cloud. Esta é diferente da autenticação OAuth 2.0 usada pelos scripts `gdrive_auth.py` e `gdrive_download.py` (que é para seu acesso pessoal).

1.  **Acesse o Google Cloud Console:**
    *   Vá para [console.cloud.google.com](https://console.cloud.google.com/) e selecione o projeto Google Cloud associado às suas credenciais OAuth (ou um projeto apropriado).

2.  **Crie uma Conta de Serviço:**
    *   No menu de navegação, vá para "IAM e Admin" > "Contas de serviço".
    *   Clique em "+ CRIAR CONTA DE SERVIÇO".
    *   **Nome da conta de serviço:** Dê um nome descritivo (ex: `juiz-bocha-gdrive-uploader`). O ID da conta de serviço será gerado automaticamente.
    *   **Descrição:** Opcional, mas útil (ex: "Conta de serviço para a API Juiz de Bocha fazer upload de arquivos para o Google Drive").
    *   Clique em "CRIAR E CONTINUAR".
    *   **Conceder acesso ao projeto (Opcional para esta finalidade):** Para esta conta de serviço, o papel principal é acessar o Google Drive, o que é feito compartilhando a pasta do Drive diretamente com ela, não necessariamente por papéis do IAM no projeto GCP, a menos que ela precise interagir com outras APIs do GCP. Você pode pular a atribuição de papéis do IAM no projeto por enquanto, ou conceder um papel mínimo se souber que será necessário para outras interações. Clique em "CONTINUAR".
    *   **Conceder aos usuários acesso a esta conta de serviço (Opcional):** Você pode pular esta etapa. Clique em "CONCLUÍDO".

3.  **Gere uma Chave para a Conta de Serviço:**
    *   Na lista de contas de serviço, encontre a que você acabou de criar.
    *   Clique no nome da conta de serviço para abrir seus detalhes.
    *   Vá para a aba "CHAVES".
    *   Clique em "ADICIONAR CHAVE" > "Criar nova chave".
    *   **Tipo de chave:** Escolha **JSON**.
    *   Clique em "CRIAR". Um arquivo JSON contendo a chave da conta de serviço será baixado automaticamente pelo seu navegador.

4.  **Guarde e Renomeie a Chave JSON:**
    *   Este arquivo JSON é muito importante e sensível, pois permite o acesso como a conta de serviço.
    *   **Renomeie o arquivo baixado para `gdrive_service_account.json`**.
    *   Coloque este arquivo na **raiz do seu projeto** Juiz de Bocha Eletrônico.
    *   **IMPORTANTE:** Adicione `gdrive_service_account.json` ao seu arquivo `.gitignore` principal para evitar que esta chave secreta seja enviada para o seu repositório Git.
        ```
        # .gitignore
        # ... outras entradas ...
        gdrive_service_account.json
        gdrive_token.json
        gdrive_credentials.json
        ```

5.  **Compartilhe a Pasta do Google Drive com a Conta de Serviço:**
    *   Abra o Google Drive no seu navegador.
    *   Navegue até a pasta que você especificou em `cloud-run-function/app_settings.json` (no campo `google_drive_upload_folder_id`). Se esta pasta ainda não existir, crie-a.
        *   *Lembre-se: O `google_drive_upload_folder_id` no `app_settings.json` é o ID da pasta, não a URL completa.* Você pode obter o ID da URL da pasta (a string longa após `folders/`).
    *   Clique com o botão direito na pasta e selecione "Compartilhar" ou "Gerenciar acesso".
    *   No campo "Adicionar pessoas e grupos", cole o **endereço de e-mail da Conta de Serviço** que você criou. Você pode encontrar este e-mail nos detalhes da conta de serviço no Google Cloud Console (geralmente algo como `nome-da-conta@seu-projeto-id.iam.gserviceaccount.com`).
    *   Conceda à conta de serviço o papel de **Editor** (ou "Colaborador", dependendo da interface do Drive). Isso permitirá que a aplicação crie arquivos e pastas dentro desta pasta compartilhada.
    *   Desmarque a opção "Notificar pessoas" (opcional) e clique em "Compartilhar" ou "Salvar".

Com esses passos, a aplicação `app_recognizer`, quando executada com o `docker-compose.yml` configurado para montar o `gdrive_service_account.json`, poderá autenticar-se no Google Drive e fazer upload de arquivos para a pasta especificada.

### Configuração Inicial Detalhada

1.  **Criar `gdrive_credentials.json` (Credenciais OAuth 2.0):**
    *   Acesse o [Google Cloud Console](https://console.cloud.google.com/).
    *   Selecione ou crie um projeto.
    *   Navegue para "APIs e Serviços" > "Credenciais".
    *   Clique em "Criar Credenciais" > "ID do cliente OAuth".
    *   Se solicitado, configure a "Tela de consentimento OAuth":
        *   **Tipo de usuário**: "Externo" (geralmente).
        *   Preencha o nome do aplicativo, e-mail de suporte e informações de contato do desenvolvedor.
        *   **Escopos**: Não adicione escopos aqui; o script `gdrive_auth.py` os solicitará.
        *   **Usuários de teste**: Adicione seu próprio endereço de e-mail do Google.
    *   Após configurar a tela de consentimento, volte para "Criar Credenciais" > "ID do cliente OAuth".
    *   Escolha o **Tipo de aplicativo**: "Aplicativo para computador" ou "Aplicativo de desktop".
    *   Dê um nome (ex: "JuizDeBochaGdriveClient").
    *   Clique em "Criar". Uma janela mostrará seu ID de cliente e Chave de cliente.
    *   **Baixe o JSON**: Clique no botão de download (geralmente um ícone de seta para baixo) ao lado do ID do cliente OAuth que você acabou de criar na lista de "IDs de cliente OAuth 2.0".
    *   Renomeie o arquivo baixado para `gdrive_credentials.json` e coloque-o na raiz do seu projeto.
    *   **Importante**: Este arquivo contém segredos. Adicione `gdrive_credentials.json` ao seu arquivo `.gitignore` global para evitar enviá-lo para o repositório Git.

2.  **Criar e Configurar `gdrive_config.json`:**
    *   Na raiz do projeto, crie um arquivo chamado `gdrive_config.json`.
    *   Cole o seguinte conteúdo e modifique a URL:
        ```json
        {
          "google_drive_folder_url": "COLE_A_URL_DA_SUA_PASTA_RAIZ_NO_GOOGLE_DRIVE_AQUI",
          "local_datasets_path": "./datasets_gdrive",
          "local_models_path": "./models_from_gdrive"
        }
        ```
    *   Substitua `"COLE_A_URL_DA_SUA_PASTA_RAIZ_NO_GOOGLE_DRIVE_AQUI"` pela URL da pasta no Google Drive que você usará. Dentro desta pasta, você deve criar subpastas chamadas `datasets` e `models`.
    *   `local_datasets_path` e `local_models_path` são os diretórios locais padrão onde os scripts baixarão os respectivos conteúdos.

3.  **Executar Autenticação Inicial (`gdrive_auth.py`):**
    *   Abra um terminal na raiz do projeto.
    *   Execute o script de autenticação:
        ```bash
        python gdrive_auth.py
        ```
    *   Este script tentará abrir uma janela no seu navegador para você autorizar o acesso ao Google Drive. Siga as instruções.
    *   Após a autorização, um arquivo `gdrive_token.json` será criado na raiz do projeto. Este arquivo armazena seus tokens de acesso e deve ser mantido em segurança (adicione-o ao `.gitignore` também).

### Baixando Datasets para Treinamento

1.  **Certifique-se de que a Configuração Inicial foi concluída.**
2.  **Estrutura no Google Drive:** Dentro da sua pasta raiz configurada em `gdrive_config.json`, crie uma subpasta chamada `datasets`. Dentro desta subpasta `datasets`, organize seus arquivos de dataset no formato COCO, por exemplo:
    ```
    Sua_Pasta_Raiz_GDrive/
    └── datasets/
        ├── juiz_de_bocha_train.json  (arquivo de anotações de treino)
        ├── juiz_de_bocha_val.json    (arquivo de anotações de validação)
        └── data/                     (pasta contendo todas as imagens .jpg)
            ├── img1.jpg
            ├── img2.jpg
            └── ...
    ```
3.  **Execute o Script de Download:**
    No terminal, na raiz do projeto, execute:
    ```bash
    python gdrive_download.py --type datasets
    ```
    Isso baixará o conteúdo da pasta `datasets` do seu Google Drive para o diretório especificado por `local_datasets_path` em `gdrive_config.json` (padrão: `./datasets_gdrive`).

4.  **Ajuste os Scripts de Registro de Dataset:**
    *   Abra o script `train_custom_dataset/register_datasets.py`.
    *   Modifique a variável `dataset_dir` para apontar para o caminho onde você baixou os datasets. Por exemplo, se baixou para `./datasets_gdrive`, e a estrutura interna é como a descrita acima, `dataset_dir` deve ser `'./datasets_gdrive'`.
    *   Se estiver usando `train/register_datasets.py`, ajuste as variáveis `annotations_dir`, `train_images_dir`, `val_images_dir` de forma similar.

5.  **Prossiga com o Treinamento:**
    Agora você pode executar seus scripts de treinamento (ex: `train_custom_dataset/3_train.py`), e eles usarão os datasets baixados.

### Baixando Modelos para Inferência (`app_recognizer`)

1.  **Certifique-se de que a Configuração Inicial foi concluída.**
2.  **Estrutura no Google Drive:** Dentro da sua pasta raiz configurada em `gdrive_config.json`, crie uma subpasta chamada `models`. Dentro desta subpasta `models`, coloque seus arquivos de modelo:
    ```
    Sua_Pasta_Raiz_GDrive/
    └── models/
        ├── config.yaml  (Seu arquivo de configuração .yaml do modelo)
        └── weights.pkl  (Seu arquivo de pesos .pkl do modelo)
    ```
    *Nota: Certifique-se de que os nomes dos arquivos são exatamente `config.yaml` e `weights.pkl` para que o `docker-compose.yml` os monte corretamente.*

3.  **Execute o Script de Download:**
    No terminal, na raiz do projeto, execute:
    ```bash
    python gdrive_download.py --type models
    ```
    Isso baixará o conteúdo da pasta `models` do seu Google Drive para o diretório especificado por `local_models_path` em `gdrive_config.json` (padrão: `./models_from_gdrive`).

4.  **Execute a Aplicação com Docker Compose:**
    ```bash
    docker-compose up -d app_recognizer
    ```
    O `docker-compose.yml` já está configurado para montar os arquivos `config.yaml` e `weights.pkl` de `./models_from_gdrive/` para dentro do container `app_recognizer`, nos locais onde a aplicação espera encontrá-los.

Lembre-se de adicionar `gdrive_credentials.json` e `gdrive_token.json` ao seu arquivo `.gitignore` para não versionar informações sensíveis.

## 9. Manual de Implantação com Docker

Este manual fornece um guia passo a passo para configurar e executar o projeto Juiz de Bocha Eletrônico (tanto a API de reconhecimento quanto a documentação web) em seu ambiente local usando Docker e Docker Compose.

### Introdução

Utilizamos Docker para empacotar a aplicação e suas dependências, garantindo um ambiente de execução consistente. O Docker Compose é usado para orquestrar os múltiplos serviços (aplicação e documentação).

### Pré-requisitos

Antes de começar, certifique-se de que você tem os seguintes softwares instalados em sua máquina:

1.  **Git:** Para clonar o repositório do projeto.
    *   [Download Git](https://git-scm.com/downloads)
2.  **Docker e Docker Compose:**
    *   **Windows/Mac:** Recomendamos instalar o [Docker Desktop](https://www.docker.com/products/docker-desktop/). Ele já inclui o Docker Compose.
    *   **Linux:**
        *   Instale o Docker Engine seguindo as instruções para sua distribuição em [docs.docker.com/engine/install/](https://docs.docker.com/engine/install/).
        *   Instale o Docker Compose seguindo as instruções em [docs.docker.com/compose/install/](https://docs.docker.com/compose/install/).
3.  **Python (versão 3.8 ou superior):** Necessário se você planeja usar a integração com Google Drive para gerenciar datasets e modelos, pois os scripts de autenticação e download (`gdrive_auth.py`, `gdrive_download.py`) são em Python.
    *   [Download Python](https://www.python.org/downloads/)
    *   Certifique-se de que o Python e o pip estão no PATH do seu sistema.

### Passo 1: Obter o Código do Projeto

Clone o repositório do projeto para sua máquina local usando Git:
```bash
git clone <URL_DO_REPOSITORIO_DO_PROJETO>
cd <NOME_DO_DIRETORIO_DO_PROJETO>
```
*Substitua `<URL_DO_REPOSITORIO_DO_PROJETO>` pela URL correta do repositório e `<NOME_DO_DIRETORIO_DO_PROJETO>` pelo nome da pasta que será criada.*

### Passo 2: Configuração de Modelos para Inferência (API `app_recognizer`)

A API de reconhecimento (`app_recognizer`) precisa de um arquivo de configuração (`config.yaml`) e um arquivo de pesos do modelo (`weights.pkl`). Você tem duas opções para fornecê-los:

#### Opção A: Usando a Integração com Google Drive (Recomendado e Flexível)

Esta abordagem permite que você gerencie seus modelos em uma pasta no Google Drive e os baixe quando necessário.

1.  **Siga as instruções de Configuração Inicial do Google Drive:** Detalhadas na seção "[8. Gerenciamento de Dados com Google Drive](#8-gerenciamento-de-dados-com-google-drive)" desta documentação. Isso inclui:
    *   Obter seu arquivo `gdrive_credentials.json` do Google Cloud Console.
    *   Criar e preencher o arquivo `gdrive_config.json` na raiz do projeto (você pode usar a página "[Configurar Google Drive (JSON)](config_gdrive.md)" na documentação web para ajudar a gerar este JSON).
    *   Executar `python gdrive_auth.py` para autorizar o acesso.

2.  **Prepare seus Modelos no Google Drive:**
    *   Na pasta do Google Drive que você configurou em `gdrive_config.json` (campo `google_drive_folder_url`), crie uma subpasta chamada `models`.
    *   Dentro desta subpasta `models` no Google Drive, coloque os arquivos do modelo de inferência que você deseja usar. **É crucial que eles sejam nomeados exatamente `config.yaml` e `weights.pkl`** para que a configuração do Docker Compose funcione corretamente.

3.  **Baixe os Modelos do Google Drive:**
    No terminal, na raiz do projeto, execute:
    ```bash
    python gdrive_download.py --type models
    ```
    Este comando baixará os arquivos da pasta `models` do seu Google Drive para o diretório local especificado em `gdrive_config.json` (padrão: `./models_from_gdrive/`). Verifique se os arquivos `config.yaml` e `weights.pkl` estão presentes em `./models_from_gdrive/` após o download.

#### Opção B: Configuração Manual dos Modelos

Se você não deseja usar a integração com Google Drive ou prefere gerenciar os modelos localmente de forma manual:

1.  Crie um diretório chamado `models_from_gdrive` na raiz do projeto (o nome é para manter a consistência com a configuração do Docker Compose).
    ```bash
    mkdir models_from_gdrive
    ```
2.  Copie o arquivo de configuração do seu modelo (formato `.yaml`) para dentro de `models_from_gdrive/` e **renomeie-o para `config.yaml`**.
3.  Copie o arquivo de pesos do seu modelo (formato `.pkl`) para dentro de `models_from_gdrive/` e **renomeie-o para `weights.pkl`**.

Após este passo, você deve ter a seguinte estrutura (seja via GDrive ou manual):
```
<raiz_do_projeto>/
└── models_from_gdrive/
    ├── config.yaml
    └── weights.pkl
```

*(Nota sobre Datasets para Treinamento: Se você também planeja treinar modelos, o processo de obtenção de datasets via Google Drive ou manualmente é similar, usando `python gdrive_download.py --type datasets` ou colocando-os em um diretório local e ajustando os scripts `register_datasets.py` conforme detalhado na seção "8. Gerenciamento de Dados com Google Drive".)*

### Passo 3: Construir e Executar os Serviços Docker

Com os arquivos de modelo no lugar (`./models_from_gdrive/`), você pode construir as imagens Docker e iniciar os serviços (API da aplicação e documentação web).

1.  **Construir as Imagens Docker:**
    Este comando lê o `docker-compose.yml` e constrói as imagens para todos os serviços definidos nele (`app_recognizer` e `docsify_docs`). Pode levar alguns minutos na primeira vez, pois baixa as imagens base e instala dependências.
    ```bash
    docker-compose build
    ```

2.  **Iniciar os Serviços:**
    Este comando inicia os containers em segundo plano (`-d` de "detached mode").
    ```bash
    docker-compose up -d
    ```
    Ambos os serviços (API e documentação) serão iniciados.

### Passo 4: Acessar os Serviços

Após os containers iniciarem com sucesso (pode levar alguns segundos):

1.  **Documentação Web:**
    *   Abra seu navegador e acesse: `http://localhost:8080`
    *   Você verá a página da documentação do projeto, servida pelo Docsify.

2.  **API da Aplicação de Reconhecimento:**
    *   A API estará escutando em: `http://localhost:8000`
    *   Você pode interagir com ela usando ferramentas como Postman, Insomnia, ou scripts (veja Passo 5).

### Passo 5: Testar a API da Aplicação

Consulte a seção "Testando a API da Aplicação Principal" dentro da "[7. Servindo a Documentação Web com Docker](#7-servindo-a-documentação-web-com-docker)" para exemplos de como enviar requisições para os endpoints (`/image`, `/url`, `/coordinates`).

Lembre-se que a aplicação, por padrão, tentará fazer upload de algumas imagens para o Google Cloud Storage. Se você não configurou credenciais do Google Cloud para o ambiente Docker local, essas partes específicas podem falhar, mas o processamento da imagem e a resposta principal da API (detecções, GIF local) ainda devem funcionar.

### Passo 6: Gerenciando os Serviços Docker

*   **Visualizar Logs:** Para ver os logs de um serviço específico (útil para depuração):
    ```bash
    docker-compose logs app_recognizer
    docker-compose logs docsify_docs
    ```
    Para seguir os logs em tempo real, adicione a flag `-f`:
    ```bash
    docker-compose logs -f app_recognizer
    ```

*   **Verificar Status dos Containers:**
    ```bash
    docker-compose ps
    ```

*   **Parar os Serviços:** Para parar e remover os containers definidos no `docker-compose.yml`:
    ```bash
    docker-compose down
    ```
    Se quiser apenas parar sem remover:
    ```bash
    docker-compose stop
    ```

### Considerações sobre Hardware, Desempenho e Plataformas

*   **Uso de CPU para Inferência no Docker:** A configuração Docker atual para a API `app_recognizer` utiliza versões CPU do PyTorch e Detectron2. Isso garante que a aplicação funcione em uma ampla variedade de máquinas, mesmo aquelas sem uma GPU NVIDIA dedicada. No entanto, a inferência (processamento de imagens) será mais lenta do que seria em uma GPU.
*   **Desempenho da CPU:** A velocidade do processamento de imagens na CPU dependerá do poder de processamento da CPU da sua máquina host.
*   **Treinamento de Modelos com GPU:** Este manual foca na *implantação* da API e documentação via Docker. O processo de *treinamento* de novos modelos de IA não é executado dentro destes containers Docker. Se você deseja treinar modelos usando uma GPU NVIDIA, você precisará configurar um ambiente Python local separado com as versões GPU do PyTorch, Detectron2 e drivers CUDA apropriados.
*   **Compatibilidade entre Windows e Linux:**
    *   **Docker Desktop no Windows:** Recomenda-se usar o backend WSL 2 (Windows Subsystem for Linux 2) para Docker Desktop, pois oferece melhor performance e compatibilidade com containers Linux.
    *   **Comandos:** Os comandos `docker-compose` são os mesmos em terminais Linux, PowerShell no Windows, ou terminais dentro do WSL2.
    *   **Caminhos de Arquivo:** O Docker Compose e o Docker Desktop (com WSL2) geralmente lidam bem com a tradução de caminhos de arquivo entre o sistema operacional host (Windows) e os containers Linux. Os caminhos relativos usados no `docker-compose.yml` (ex: `./models_from_gdrive/`) devem funcionar corretamente.

Seguindo este manual, você deverá ser capaz de implantar e executar todo o sistema Juiz de Bocha Eletrônico em seu ambiente local.

## 10. Manual de Uso da API com Flutter

Este manual descreve como interagir com a API de reconhecimento de bocha a partir de uma aplicação Flutter. Serão fornecidos exemplos de código Dart para enviar imagens aos endpoints da API e processar as respostas.

### Pré-requisitos e Configuração Inicial

1.  **Ambiente Flutter Configurado:** Certifique-se de que seu ambiente de desenvolvimento Flutter está funcionando.
2.  **Adicionar Dependências ao `pubspec.yaml`:**
    Adicione os seguintes pacotes ao seu arquivo `pubspec.yaml`:
    ```yaml
    dependencies:
      flutter:
        sdk: flutter
      http: ^0.13.6 # Ou a versão mais recente
      image_picker: ^0.8.9 # Ou a versão mais recente, para selecionar imagens
      # provider: ^6.0.0 # Opcional, para gerenciamento de estado se for construir uma UI complexa
      # path_provider: ^2.0.0 # Opcional, para lidar com caminhos de arquivo
    ```
    Depois de adicionar, execute `flutter pub get` no seu terminal.

3.  **Configurar Permissões (para `image_picker`):**
    *   **iOS:** Adicione as chaves necessárias ao seu arquivo `Info.plist` (em `ios/Runner/Info.plist`) para acesso à galeria e câmera. Exemplo:
        ```xml
        <key>NSPhotoLibraryUsageDescription</key>
        <string>Este aplicativo precisa de acesso à sua galeria para selecionar imagens de bocha.</string>
        <key>NSCameraUsageDescription</key>
        <string>Este aplicativo precisa de acesso à sua câmera para capturar imagens de bocha.</string>
        <key>NSMicrophoneUsageDescription</key>
        <string>Este aplicativo não precisa de acesso ao microfone (mas o image_picker pode pedir).</string>
        ```
    *   **Android:** Nenhuma configuração extra é geralmente necessária para o `image_picker` funcionar com API level mais recente, mas se enfrentar problemas, consulte a documentação do `image_picker`. Certifique-se de que seu `minSdkVersion` em `android/app/build.gradle` é 21 ou superior.

4.  **URL Base da API:**
    Defina a URL base da sua API. Se estiver rodando o Docker localmente:
    ```dart
    // No seu código Dart, por exemplo, em um arquivo de constantes ou configurações:
    const String baseUrl = "http://localhost:8000";
    // Se estiver testando de um emulador Android, localhost pode não funcionar.
    // Use "http://10.0.2.2:8000" para o emulador Android padrão se a API estiver rodando na sua máquina host.
    // Se testando de um dispositivo físico na mesma rede, use o IP local da sua máquina host
    // (ex: "http://192.168.1.10:8000").
    ```

### Função Auxiliar para Selecionar Imagem (Usando `image_picker`)

Aqui está uma função simples que você pode usar para permitir que o usuário selecione uma imagem da galeria:
```dart
import 'dart:io';
import 'package:image_picker/image_picker.dart';

Future<File?> pickImageFromGallery() async {
  final ImagePicker picker = ImagePicker();
  final XFile? pickedFile = await picker.pickImage(source: ImageSource.gallery);

  if (pickedFile != null) {
    return File(pickedFile.path);
  }
  return null;
}

// Para usar a câmera:
// final XFile? pickedFile = await picker.pickImage(source: ImageSource.camera);
```

### Função Auxiliar para Enviar Imagem (Requisição POST Multipart)

Esta função genérica pode ser usada para enviar um arquivo de imagem para qualquer um dos endpoints.
```dart
import 'dart:convert';
import 'dart:io';
import 'package:http/http.dart' as http;
import 'package:http_parser/http_parser.dart'; // Para MediaType

Future<http.Response?> uploadImage(File imageFile, String endpointUrl, {Map<String, String>? queryParams}) async {
  try {
    var uri = Uri.parse(endpointUrl);
    if (queryParams != null && queryParams.isNotEmpty) {
      uri = uri.replace(queryParameters: queryParams);
    }

    var request = http.MultipartRequest('POST', uri);

    // Adicionar o arquivo de imagem
    request.files.add(
      await http.MultipartFile.fromPath(
        'file', // Nome do campo esperado pelo backend (ajuste se necessário, mas a API atual lê o stream diretamente)
        imageFile.path,
        contentType: MediaType('image', imageFile.path.split('.').last), // Ex: image/jpeg, image/png
      ),
    );

    print("Enviando imagem para: ${uri.toString()}");
    var streamedResponse = await request.send();
    var response = await http.Response.fromStream(streamedResponse);

    if (response.statusCode == 200) {
      print("Upload da imagem bem-sucedido para $endpointUrl.");
      return response;
    } else {
      print("Falha no upload da imagem para $endpointUrl. Status: ${response.statusCode}, Corpo: ${response.body}");
      return response; // Retorna a resposta mesmo em caso de erro para análise
    }
  } catch (e) {
    print("Erro ao enviar imagem para $endpointUrl: $e");
    return null;
  }
}
```
*Nota: A API atual lê diretamente `request.stream.read()`, então o nome do campo 'file' no `MultipartFile.fromPath` não é estritamente usado por ela, mas é uma boa prática incluí-lo.*

### Consumindo o Endpoint `POST /image`

Este endpoint retorna um GIF (binário ou Base64) da imagem processada.

**1. Enviando a Requisição e Processando a Resposta:**

```dart
// Assumindo que 'selectedImageFile' é um objeto File obtido do image_picker
// e 'baseUrl' está definido como "http://localhost:8000"

Future<void> callApiImageEndpoint(File selectedImageFile, bool withThumbnail) async {
  String endpoint = "$baseUrl/image";
  Map<String, String> queryParams = {};
  if (withThumbnail) {
    queryParams['with-thumbnail'] = 'true';
  }

  http.Response? response = await uploadImage(selectedImageFile, endpoint, queryParams: queryParams);

  if (response != null && response.statusCode == 200) {
    if (withThumbnail) {
      // Resposta é JSON com 'gif' e 'thumbnail' em Base64
      try {
        var decodedResponse = jsonDecode(response.body);
        String? gifBase64 = decodedResponse['gif'];
        String? thumbnailBase64 = decodedResponse['thumbnail'];

        if (gifBase64 != null) {
          // Use gifBytes para exibir a imagem GIF (veja abaixo)
          final gifBytes = base64Decode(gifBase64);
          print("GIF (Base64) recebido e decodificado (${gifBytes.lengthInBytes} bytes)");
          // setState(() { _gifImageData = gifBytes; }); // Exemplo em StatefulWidget
        }
        if (thumbnailBase64 != null) {
          final thumbnailBytes = base64Decode(thumbnailBase64);
          print("Thumbnail (Base64) recebido e decodificado (${thumbnailBytes.lengthInBytes} bytes)");
          // setState(() { _thumbnailImageData = thumbnailBytes; }); // Exemplo
        }
      } catch (e) {
        print("Erro ao decodificar JSON da resposta /image: $e");
      }
    } else {
      // Resposta é o GIF binário diretamente
      final gifBytes = response.bodyBytes;
      print("GIF (binário) recebido diretamente (${gifBytes.lengthInBytes} bytes)");
      // setState(() { _gifImageData = gifBytes; }); // Exemplo em StatefulWidget
    }
  } else {
    print("Falha ao chamar /image. Status: ${response?.statusCode}, Corpo: ${response?.body}");
  }
}
```

**2. Exibindo a Imagem GIF (Bytes) em um Widget Flutter:**

```dart
// Em um StatefulWidget, você teria uma variável de estado, por exemplo:
// Uint8List? _gifImageData;

// No método build():
// if (_gifImageData != null)
//   Image.memory(_gifImageData!)
// else
//   Text("Nenhuma imagem GIF para exibir"),
```

### Consumindo o Endpoint `POST /url`

Este endpoint retorna URLs (para Google Cloud Storage) da imagem processada.

```dart
// Assumindo que 'selectedImageFile' é um objeto File
// e 'baseUrl' está definido

Future<void> callApiUrlEndpoint(File selectedImageFile, bool withThumbnail) async {
  String endpoint = "$baseUrl/url";
  Map<String, String> queryParams = {};
  if (withThumbnail) {
    queryParams['with-thumbnail'] = 'true';
  }

  http.Response? response = await uploadImage(selectedImageFile, endpoint, queryParams: queryParams);

  if (response != null && response.statusCode == 200) {
    if (withThumbnail) {
      // Resposta é JSON com 'animated' (URL do GIF) e 'thumbnail' (URL do Thumbnail)
      try {
        var decodedResponse = jsonDecode(response.body);
        String? animatedGifUrl = decodedResponse['animated'];
        String? thumbnailUrl = decodedResponse['thumbnail'];

        if (animatedGifUrl != null) {
          print("URL do GIF Animado: $animatedGifUrl");
          // setState(() { _animatedGifUrl = animatedGifUrl; }); // Exemplo
        }
        if (thumbnailUrl != null) {
          print("URL do Thumbnail: $thumbnailUrl");
          // setState(() { _thumbnailUrl = thumbnailUrl; }); // Exemplo
        }
      } catch (e) {
        print("Erro ao decodificar JSON da resposta /url: $e");
      }
    } else {
      // Resposta é a string da URL do GIF diretamente
      String animatedGifUrl = response.body;
      print("URL do GIF Animado (direta): $animatedGifUrl");
      // setState(() { _animatedGifUrl = animatedGifUrl; }); // Exemplo
    }
  } else {
    print("Falha ao chamar /url. Status: ${response?.statusCode}, Corpo: ${response?.body}");
  }
}
```

**Exibindo Imagens a partir de URLs em Flutter:**
```dart
// Em um StatefulWidget:
// String? _animatedGifUrl;
// String? _thumbnailUrl;

// No método build():
// if (_animatedGifUrl != null)
//   Image.network(_animatedGifUrl!) // Para o GIF
// if (_thumbnailUrl != null)
//   Image.network(_thumbnailUrl!)   // Para o Thumbnail
```

### Consumindo o Endpoint `POST /coordinates`

Este endpoint retorna coordenadas JSON das bolas detectadas e a URL da imagem original no GCS.

```dart
// Assumindo que 'selectedImageFile' é um objeto File
// e 'baseUrl' está definido

// Defina classes Dart para melhor tipagem da resposta (opcional, mas recomendado)
class BallCoordinate {
  final Point center;
  final Ellipse ellipse;
  BallCoordinate({required this.center, required this.ellipse});

  factory BallCoordinate.fromJson(Map<String, dynamic> json) {
    return BallCoordinate(
      center: Point.fromJson(json['center']),
      ellipse: Ellipse.fromJson(json['ellipse']),
    );
  }
}

class Point {
  final double x;
  final double y;
  Point({required this.x, required this.y});

  factory Point.fromJson(Map<String, dynamic> json) {
    return Point(x: (json['x'] as num).toDouble(), y: (json['y'] as num).toDouble());
  }
}

class Ellipse {
  final double width;
  final double height;
  Ellipse({required this.width, required this.height});

  factory Ellipse.fromJson(Map<String, dynamic> json) {
    return Ellipse(width: (json['width'] as num).toDouble(), height: (json['height'] as num).toDouble());
  }
}

class CoordinatesResponse {
  final String url;
  final BallCoordinate smallest;
  final BallCoordinate winner;
  final List<BallCoordinate> balls;

  CoordinatesResponse({
    required this.url,
    required this.smallest,
    required this.winner,
    required this.balls,
  });

  factory CoordinatesResponse.fromJson(Map<String, dynamic> json) {
    var ballsList = json['balls'] as List;
    List<BallCoordinate> parsedBalls = ballsList.map((i) => BallCoordinate.fromJson(i)).toList();
    return CoordinatesResponse(
      url: json['url'],
      smallest: BallCoordinate.fromJson(json['smallest']),
      winner: BallCoordinate.fromJson(json['winner']),
      balls: parsedBalls,
    );
  }
}

Future<CoordinatesResponse?> callApiCoordinatesEndpoint(File selectedImageFile) async {
  String endpoint = "$baseUrl/coordinates";

  http.Response? response = await uploadImage(selectedImageFile, endpoint);

  if (response != null && response.statusCode == 200) {
    try {
      var decodedResponse = jsonDecode(response.body);
      CoordinatesResponse result = CoordinatesResponse.fromJson(decodedResponse);

      print("Imagem Original URL (GCS): ${result.url}");
      print("Bolim (Smallest): Centro X: ${result.smallest.center.x.toStringAsFixed(3)}}, Y: ${result.smallest.center.y.toStringAsFixed(3)}}");
      print("Vencedora (Winner): Centro X: ${result.winner.center.x.toStringAsFixed(3)}}, Y: ${result.winner.center.y.toStringAsFixed(3)}}");
      result.balls.asMap().forEach((index, ball) {
        print("Outra Bola $index: Centro X: ${ball.center.x.toStringAsFixed(3)}}, Y: ${ball.center.y.toStringAsFixed(3)}}");
      });
      return result;
      // Você pode então usar 'result' para desenhar sobre a imagem original (obtida de result.url)
      // ou para qualquer outra lógica de visualização.
    } catch (e) {
      print("Erro ao decodificar JSON da resposta /coordinates: $e");
      return null;
    }
  } else {
    print("Falha ao chamar /coordinates. Status: ${response?.statusCode}, Corpo: ${response?.body}");
    return null;
  }
}
```

**Ideias para Visualização das Coordenadas:**
Para desenhar as coordenadas sobre a imagem original (cuja URL é retornada), você pode:
1.  Carregar a imagem da `result.url` usando `Image.network`.
2.  Usar um `CustomPaint` widget em Flutter.
3.  No `CustomPainter`, desenhar a imagem carregada e, em seguida, desenhar elipses ou círculos sobre ela usando as coordenadas normalizadas (multiplicando-as pelas dimensões reais da imagem no canvas).

### Tratamento de Erros e Considerações

*   **Status Codes:** Sempre verifique o `response.statusCode`. Um status `200 OK` geralmente indica sucesso. Outros códigos (4xx, 5xx) indicam erros.
*   **Exceções de Rede:** Envolva suas chamadas de API em blocos `try-catch` para lidar com problemas de conectividade (ex: `SocketException`).
*   **Timeouts:** O pacote `http` tem um timeout padrão. Você pode configurá-lo usando `.timeout()` no Future da requisição se precisar de mais controle.
*   **UI Responsiva:** Faça chamadas de API de forma assíncrona (usando `async/await`) para não bloquear a thread de UI. Use indicadores de carregamento (ex: `CircularProgressIndicator`) enquanto a API processa.
*   **Segurança da URL Base:** Em um aplicativo de produção, evite "hardcodar" URLs. Use variáveis de ambiente ou um sistema de configuração.
*   **Interação com GCS:** Lembre-se que os endpoints `/url` e `/coordinates` dependem do upload para o Google Cloud Storage. Se as credenciais não estiverem configuradas no backend Docker (o que geralmente não é o caso para um setup local simples), essas partes podem falhar, e as URLs retornadas podem não ser válidas ou os endpoints podem retornar erros.

Este manual deve fornecer uma base sólida para consumir a API de reconhecimento de bocha a partir de uma aplicação Flutter.
