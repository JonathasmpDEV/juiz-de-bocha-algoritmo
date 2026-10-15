// Gera JuizDeBocha_Experimentos.ipynb — fila de experimentos + relatórios para benchmark
const fs = require('fs');
const out = process.argv[2];

const md = (s) => ({ cell_type: 'markdown', metadata: {}, source: lines(s) });
const code = (s) => ({ cell_type: 'code', metadata: {}, execution_count: null, outputs: [], source: lines(s) });
function lines(s) {
  const arr = s.replace(/^\n/, '').replace(/\s+$/, '').split('\n');
  return arr.map((l, i) => (i < arr.length - 1 ? l + '\n' : l));
}

const cells = [
md(`
# Juiz de Bocha — fila de experimentos (benchmark)

Roda vários treinos **com as mesmas fotos e as mesmas marcações** e grava um relatório completo de cada um no Drive, em \`00-Juiz de Bocha/experimentos/<experimento>__<gpu>/\`. Os relatórios são importados pelo app local de benchmark (banco SQLite + exportação CSV).

**Como usar (padrão, \`FONTE_FILA = "lista"\`, sem integração):** abra no **Chrome** (a janela de permissão do Drive precisa abrir), escolha a GPU em *Ambiente de execução → Alterar tipo*, ajuste \`GRUPO\` na célula 2 e use **Executar tudo**. Cada experimento grava \`00-Juiz de Bocha/experimentos/<nome>__<GPU>/relatorio.json\`; esses arquivos vão para o app de benchmark.

*Opcional — controlado pelo app (\`FONTE_FILA = "app"\`):* exige a chave do Firebase no secret \`FIREBASE_ADMIN_JSON\`; o notebook passa a pegar os treinos criados no app.

A fila roda sozinha. Se o Colab cair, rode de novo: experimentos com relatório pronto são pulados e o que estava no meio continua do último checkpoint.

**Modo reativo (\`ADAPTAR_A_GPU\`, ligado por padrão):** o notebook lê a memória e a geração da GPU e ajusta sozinho o tamanho do lote pela memória da placa (15 na T4, 22 na L4, 24 na A100/G4; menor para modelos maiores e imagens inteiras), a taxa de aprendizado e as iterações (regra linear: o modelo vê o mesmo número de imagens em qualquer GPU), a precisão mista (AMP) e o TF32. Também reage ao **modelo** (R101, Cascade e X101 usam lotes menores porque ocupam mais memória) e a **falta de memória**: se a GPU estourar, o treino recomeça sozinho com metade do lote em vez de derrubar a fila. Os valores usados ficam no relatório; para comparar velocidade entre GPUs use **imagens/s**. Desligue para reproduzir exatamente os hiperparâmetros originais.

**Correção das caixas:** no treino original a rotação aleatória deixava as caixas maiores que as bolas (AP75 de caixa ≈ 0 nos treinos A100 e G4, enquanto o de máscara ficou ≈ 97). Aqui as caixas são recalculadas a partir das máscaras depois de cada transformação (\`recompute_boxes=True\`).

**Base congelada:** marcações revisadas e pesos da etapa 1 do treino da A100 (\`00-Juiz de Bocha/treinoA100\`). Assim a única diferença entre experimentos é a variável testada.

| Variável | Experimentos |
|---|---|
| GPU | \`base_r50_it1000\` + \`bench_etapa1_r50\` rodados em cada GPU (grupo \`gpu\`) |
| Iterações | 1000 · 3000 · 5000 na etapa 2 |
| Taxa de aprendizado | 0.00125 · 0.0025 · 0.005 |
| Modelo | R50 (atual) × R101 × Cascade R50 × X101 — cada um treina a própria etapa 1 (~15000 iterações) uma vez só |
| Aumentos de dados | com × sem |

**O que cada relatório coleta:** ambiente (GPU, memória, driver, CUDA, versões), hiperparâmetros e config completa, tempos de cada fase, segundos por iteração (média, mediana, p90, tempo de dados), pico de memória da GPU, curvas de todas as métricas de treino, métricas COCO (bbox e máscara, 12 números cada), avaliação do juiz por foto em 5 limiares de confiança (decidiu?, acertou a vencedora e o balim em relação às marcações revisadas?, distâncias, margem, tempo de inferência), tamanho e nº de parâmetros do modelo.
`),

md(`## 1. Ambiente`),
code(`
import time
T_SESSAO = time.time()
!nvidia-smi -L || echo "Sem GPU — ative em Ambiente de execução → Alterar tipo"
try:
    import detectron2
except ImportError:
    !pip install -q exifread
    !pip install -q --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git'
    import detectron2
!pip install -q firebase-admin

import os, sys, json, random, shutil, platform, subprocess, statistics, hashlib, datetime, math
from google.colab import drive
drive.mount("/content/drive")

REPO_DIR = "/content/juiz-de-bocha-algoritmo"
if not os.path.isdir(REPO_DIR):
    !git clone -q https://github.com/JonathasmpDEV/juiz-de-bocha-algoritmo.git {REPO_DIR}
sys.path.insert(0, REPO_DIR)
T_SETUP_S = time.time() - T_SESSAO
print(f"Ambiente pronto em {T_SETUP_S:.0f} s")
`),

md(`## 2. Configuração e fila`),
code(`
BASE = "/content/drive/MyDrive/00-Juiz de Bocha"
DATASET_BASE = f"{BASE}/treinoA100"  # @param {type:"string"}
FOTOS_BOCHA = "/content/drive/MyDrive/BACKUP PENDRIVE 256/CursoSegImgPython/bocha"  # @param {type:"string"}
EXPERIMENTOS_DIR = f"{BASE}/experimentos"
GRUPO = "todos"  # @param ["todos", "gpu", "iteracoes", "lr", "modelo", "aumentos"]
SALVAR_MODELO = True  # @param {type:"boolean"}
# SALVAR_MODELO: guarda o model_final.pth (~350 MB) de cada experimento no Drive, para testar no notebook de teste

FILA = [
    # nome, grupos, modelo, etapa1 ("reusar" | "treinar" | "benchmark"), it1, it2, lr2, aumentos
    dict(nome="base_r50_it1000",         grupos=["gpu", "iteracoes", "lr", "modelo", "aumentos"], modelo="R50",  etapa1="reusar",    it1=0,     it2=1000, lr2=0.0025,  aumentos=True),
    dict(nome="bench_etapa1_r50",        grupos=["gpu"],       modelo="R50",  etapa1="benchmark", it1=1000,  it2=0,    lr2=0.0025,  aumentos=True),
    dict(nome="r50_it3000",              grupos=["iteracoes"], modelo="R50",  etapa1="reusar",    it1=0,     it2=3000, lr2=0.0025,  aumentos=True),
    dict(nome="r50_it5000",              grupos=["iteracoes"], modelo="R50",  etapa1="reusar",    it1=0,     it2=5000, lr2=0.0025,  aumentos=True),
    dict(nome="r50_lr0.00125_it1000",    grupos=["lr"],        modelo="R50",  etapa1="reusar",    it1=0,     it2=1000, lr2=0.00125, aumentos=True),
    dict(nome="r50_lr0.005_it1000",      grupos=["lr"],        modelo="R50",  etapa1="reusar",    it1=0,     it2=1000, lr2=0.005,   aumentos=True),
    dict(nome="r50_sem_aumentos_it1000", grupos=["aumentos"],  modelo="R50",  etapa1="reusar",    it1=0,     it2=1000, lr2=0.0025,  aumentos=False),
    # modelos: etapa1="auto" treina a etapa 1 (COCO) do modelo só na primeira vez e depois reaproveita
    dict(nome="r101_it1000",             grupos=["modelo"],    modelo="R101", etapa1="auto",      it1=15000, it2=1000, lr2=0.0025,  aumentos=True),
    dict(nome="cascade_r50_it1000",      grupos=["modelo"],    modelo="CASCADE_R50", etapa1="auto", it1=15000, it2=1000, lr2=0.0025, aumentos=True),
    dict(nome="x101_it1000",             grupos=["modelo"],    modelo="X101", etapa1="auto",      it1=15000, it2=1000, lr2=0.0025,  aumentos=True),
]
LIMIARES = [0.5, 0.6, 0.7, 0.82, 0.9]

import torch
GPU_NOME = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
def gpu_tag(nome):
    n = nome.upper()
    for chave, tag in (("A100", "A100"), ("H100", "H100"), ("L4", "L4"), ("T4", "T4"), ("V100", "V100"),
                       ("RTX PRO 6000", "G4"), ("BLACKWELL", "G4"), ("G4", "G4"), ("CPU", "CPU")):
        if chave in n:
            return tag
    return "".join(c for c in n if c.isalnum())[:12]
GPU = gpu_tag(GPU_NOME)

# --- modo reativo: adapta lote, LR, iterações, AMP e TF32 à GPU escolhida no Colab ---
ADAPTAR_A_GPU = True  # @param {type:"boolean"}
def perfil_gpu():
    if not torch.cuda.is_available():
        return dict(adaptado=False, mem_gb=0.0, cc=0.0, lote1=2, lote2=3, amp=False, tf32=False)
    p = torch.cuda.get_device_properties(0)
    mem, cc = p.total_memory / 2**30, float(f"{p.major}.{p.minor}")
    if not ADAPTAR_A_GPU:
        return dict(adaptado=False, mem_gb=mem, cc=cc, lote1=2, lote2=3, amp=False, tf32=False)
    # medido na L4 (R50, AMP, recorte 640): ~0,6 GB por imagem → reservamos 1 GB/imagem
    # → T4 15 · L4 22 · A100/H100/G4 24 (teto). Se mesmo assim faltar memória, o treino reduz o lote sozinho.
    lote = max(2, min(24, int(mem / 1.0)))
    return dict(adaptado=True, mem_gb=mem, cc=cc, lote1=lote, lote2=lote, amp=cc >= 7.0, tf32=cc >= 8.0)
PERFIL = perfil_gpu()
torch.backends.cuda.matmul.allow_tf32 = PERFIL["tf32"]
torch.backends.cudnn.allow_tf32 = PERFIL["tf32"]

# memória relativa de cada modelo (R50 = 1): modelos maiores recebem lote proporcionalmente menor
PESO_MODELO = {"R50": 1.0, "R101": 1.3, "CASCADE_R50": 1.4, "X101": 2.0}

def lote_para(modelo, etapa, aumentos=True):
    """Lote reativo: memória da GPU ÷ peso do modelo ÷ tamanho da imagem.
    Etapa 1 (COCO) e treino sem aumentos usam a imagem inteira (~2,5x os pixels do recorte 640)."""
    base = PERFIL["lote1"] if etapa == 1 else PERFIL["lote2"]
    if PERFIL["adaptado"] and (etapa == 1 or not aumentos):
        base = base / 2.5
    if not PERFIL["adaptado"]:
        return base
    return max(1, int(base / PESO_MODELO.get(modelo, 1.0)))

def ajustar(lote_ref, lote, lr, iteracoes):
    """Regra linear: LR proporcional ao lote e iterações inversas, para ver o mesmo nº de imagens."""
    return lr * lote / lote_ref, max(1, round(iteracoes * lote_ref / lote))

fila = [e for e in FILA if GRUPO == "todos" or GRUPO in e["grupos"]]
os.makedirs(EXPERIMENTOS_DIR, exist_ok=True)
print(f"GPU: {GPU_NOME} → tag {GPU} | {PERFIL['mem_gb']:.0f} GB | compute {PERFIL['cc']}")
print(f"Perfil {'reativo' if PERFIL['adaptado'] else 'original'}: lote etapa1={PERFIL['lote1']} · lote etapa2={PERFIL['lote2']} · AMP={PERFIL['amp']} · TF32={PERFIL['tf32']}")
for e in fila:
    pronto = os.path.exists(f"{EXPERIMENTOS_DIR}/{e['nome']}__{GPU}/relatorio.json")
    print(f"  {'✔ pronto ' if pronto else '… na fila'}  {e['nome']}__{GPU}")
`),

md(`
## 2b. Conexão com o app (fila no Firestore)
Com \`FONTE_FILA = "app"\` este notebook vira o **executor** do app de benchmark: pega os treinos que você criou no app
(coleção \`benchmark_treinos\` do Firestore do projeto **juizdebocha**) marcados para esta GPU ou para "qualquer",
informa o progresso ao vivo, obedece ao botão **cancelar** e devolve o relatório (Firebase Storage em \`benchmark/relatorios/\`).

**Uma vez só:** em 🔑 *Secrets* (barra lateral do Colab) crie \`FIREBASE_ADMIN_JSON\` com o **conteúdo** da chave de conta de
serviço do Firebase (a mesma que o app usa) e ative o acesso do notebook. Com \`FONTE_FILA = "lista"\` roda a lista \`FILA\` acima, sem app.`),
code(`
FONTE_FILA = "lista"  # @param ["lista", "app"]
ESPERAR_NOVOS_MIN = 10  # @param {type:"integer"}
# ESPERAR_NOVOS_MIN: quando a fila esvaziar, continua esperando novos treinos por esse tempo antes de encerrar

FS = BUCKET = None
if FONTE_FILA == "app":
    import firebase_admin
    from firebase_admin import credentials, firestore, storage
    from google.colab import userdata
    if not firebase_admin._apps:
        firebase_admin.initialize_app(credentials.Certificate(json.loads(userdata.get("FIREBASE_ADMIN_JSON"))),
                                      {"storageBucket": "juizdebocha.appspot.com"})
    FS, BUCKET = firestore.client(), storage.bucket()
    FILA_FS = FS.collection("benchmark_treinos")
    pendentes = [d for d in FILA_FS.where("status", "==", "na_fila").stream()
                 if (d.to_dict().get("gpu_desejada") or "qualquer") in (GPU, "qualquer")]
    print(f"Conectado ao Firestore. {len(pendentes)} treino(s) na fila para {GPU} (ou qualquer GPU).")
`),

md(`## 3. Dados congelados (fotos + marcações da A100) e dataset COCO para a etapa 1`),
code(`
import numpy as np
from tqdm.auto import tqdm
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, MetadataCatalog

DATA_DIR = "/content/bocha/data"
os.makedirs(DATA_DIR, exist_ok=True)
for nome in sorted(os.listdir(FOTOS_BOCHA)):
    if nome.startswith("._") or not nome.lower().endswith((".jpg", ".jpeg", ".png")):
        continue
    if not os.path.exists(f"{DATA_DIR}/{nome}"):
        shutil.copy(f"{FOTOS_BOCHA}/{nome}", f"{DATA_DIR}/{nome}")

JSON_TRAIN = f"{DATASET_BASE}/juiz_de_bocha_train.json"
JSON_VAL = f"{DATASET_BASE}/juiz_de_bocha_val.json"
ETAPA1_R50 = f"{DATASET_BASE}/etapa1_coco_sports_ball/model_final.pth"

def sha1(caminho):
    with open(caminho, "rb") as f:
        return hashlib.sha1(f.read()).hexdigest()[:12]

with open(JSON_TRAIN) as f: _tr = json.load(f)
with open(JSON_VAL) as f: VAL = json.load(f)
DATASET_INFO = dict(
    base=DATASET_BASE, fotos_pasta=FOTOS_BOCHA,
    train_imagens=len(_tr["images"]), train_bolas=len(_tr["annotations"]),
    val_imagens=len(VAL["images"]), val_bolas=len(VAL["annotations"]),
    train_sha1=sha1(JSON_TRAIN), val_sha1=sha1(JSON_VAL),
)
del _tr
print(DATASET_INFO)

def registrar(nome, json_file, image_root):
    if nome in DatasetCatalog.list():
        DatasetCatalog.remove(nome)
        MetadataCatalog.remove(nome)
    register_coco_instances(nome, {"thing_classes": ["sports ball"]}, json_file, image_root)

def registrar_bocha():
    registrar("juiz_de_bocha_train", JSON_TRAIN, DATA_DIR)
    registrar("juiz_de_bocha_val", JSON_VAL, DATA_DIR)

# --- COCO "sports ball" (só é preciso para etapa1 = treinar/benchmark) ---
COCO_DIR = "/content/coco"
def preparar_coco():
    import zipfile, urllib.request
    from concurrent.futures import ThreadPoolExecutor
    os.makedirs(f"{COCO_DIR}/annotations", exist_ok=True)
    if not os.path.exists(f"{COCO_DIR}/annotations/instances_train2017.json"):
        urllib.request.urlretrieve("http://images.cocodataset.org/annotations/annotations_trainval2017.zip", "/content/ann.zip")
        with zipfile.ZipFile("/content/ann.zip") as z:
            for n in ("annotations/instances_train2017.json", "annotations/instances_val2017.json"):
                z.extract(n, COCO_DIR)
        os.remove("/content/ann.zip")
    for fase in ("train", "val"):
        destino_json = f"{COCO_DIR}/annotations/juiz_de_bocha_{fase}.json"
        if not os.path.exists(destino_json):
            with open(f"{COCO_DIR}/annotations/instances_{fase}2017.json") as f:
                coco = json.load(f)
            cat = next(c for c in coco["categories"] if c["name"] == "sports ball")
            anns = [{**a, "category_id": 1} for a in coco["annotations"] if a["category_id"] == cat["id"]]
            com_bola = {a["image_id"] for a in anns}
            dados = dict(images=[im for im in coco["images"] if im["id"] in com_bola], annotations=anns,
                         categories=[{**cat, "id": 1}])
            with open(destino_json, "w") as f:
                json.dump(dados, f)
        with open(destino_json) as f:
            imagens = json.load(f)["images"]
        os.makedirs(f"{COCO_DIR}/{fase}2017", exist_ok=True)
        def baixar(nome):
            d = f"{COCO_DIR}/{fase}2017/{nome}"
            if not os.path.exists(d):
                urllib.request.urlretrieve(f"http://images.cocodataset.org/{fase}2017/{nome}", d)
        with ThreadPoolExecutor(32) as ex:
            list(tqdm(ex.map(baixar, [im["file_name"] for im in imagens]), total=len(imagens), desc=f"coco {fase}"))
    registrar("juiz_de_bocha_train", f"{COCO_DIR}/annotations/juiz_de_bocha_train.json", f"{COCO_DIR}/train2017")
    registrar("juiz_de_bocha_val", f"{COCO_DIR}/annotations/juiz_de_bocha_val.json", f"{COCO_DIR}/val2017")
`),

md(`## 4. Treino, avaliação e coleta`),
code(`
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, hooks
from detectron2.evaluation import COCOEvaluator, DatasetEvaluators
from detectron2.data import DatasetMapper, build_detection_train_loader
from detectron2.data import transforms as T

from detectron2 import model_zoo
from detectron2.engine import HookBase

# modelos disponíveis (todos do Model Zoo do Detectron2, pré-treinados no COCO)
MODELOS_ZOO = {
    "R50": "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",       # o atual do projeto
    "R101": "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml",
    "X101": "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml",
    "CASCADE_R50": "Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml",           # caixas mais precisas (3 estágios)
}
PESOS_COCO = {m: model_zoo.get_checkpoint_url(c) for m, c in MODELOS_ZOO.items()}

class Cancelado(Exception):
    pass

class Progresso(HookBase):
    """Avisa o app a cada N iterações; se o app pedir cancelamento, interrompe o treino."""
    def __init__(self, a_cada=20):
        self.a_cada = a_cada

    def after_step(self):
        it, total = self.trainer.iter + 1, self.trainer.max_iter
        if Trainer.ao_progresso and (it % self.a_cada == 0 or it == total):
            hist = self.trainer.storage.histories().get("total_loss")
            if Trainer.ao_progresso(it, total, hist.median(20) if hist else None):
                raise Cancelado()

class Trainer(DefaultTrainer):
    usar_aumentos = True
    ao_progresso = None  # função (iteração, total, loss) -> True para cancelar

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return DatasetEvaluators([COCOEvaluator(dataset_name, output_dir=output_folder or cfg.OUTPUT_DIR)])

    @classmethod
    def build_train_loader(cls, cfg):
        if not cls.usar_aumentos:
            return super().build_train_loader(cfg)
        # recompute_boxes: depois de girar/recortar, a caixa é recalculada a partir da máscara.
        # Sem isso (treino original) a rotação deixa as caixas "folgadas" e o AP75 de caixa cai para ~0.
        return build_detection_train_loader(cfg, mapper=DatasetMapper(cfg, is_train=True, recompute_boxes=True, augmentations=[
            T.RandomBrightness(0.5, 1.5),
            T.RandomFlip(prob=0.5),
            T.RandomCrop("absolute_range", (640, 640)),
            T.RandomRotation(angle=[-90, 90], expand=True),
            T.RandomContrast(intensity_min=0.2, intensity_max=1.8),
            T.RandomLighting(scale=1.0),
            T.RandomSaturation(intensity_min=0.2, intensity_max=1.8),
        ]))

    def build_hooks(self):
        hs = super().build_hooks()
        for i, h in enumerate(hs):
            if isinstance(h, hooks.PeriodicCheckpointer):
                hs[i] = hooks.PeriodicCheckpointer(self.checkpointer, self.cfg.SOLVER.CHECKPOINT_PERIOD, max_to_keep=1)
        hs.append(Progresso())
        return hs

def montar_cfg(modelo, etapa):
    cfg = get_cfg()
    if modelo == "R50":  # configs do próprio repositório (iguais ao treino original)
        sufixo = "_custom" if etapa == 2 else ""
        cfg.merge_from_file(f"{REPO_DIR}/models/mask_rcnn_juiz_de_bocha{sufixo}/mask_rcnn_juiz_de_bocha.yaml")
    else:
        cfg.merge_from_file(model_zoo.get_config_file(MODELOS_ZOO[modelo]))
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        cfg.INPUT.MIN_SIZE_TRAIN = (800,)
        cfg.SOLVER.STEPS = (60000, 80000)
    cfg.DATASETS.TRAIN = ("juiz_de_bocha_train",)
    cfg.DATASETS.TEST = ("juiz_de_bocha_val",)
    cfg.DATALOADER.NUM_WORKERS = min(8, os.cpu_count() or 2)
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    return cfg

def ler_metricas(caminho):
    linhas = []
    if os.path.exists(caminho):
        with open(caminho) as f:
            for l in f:
                try:
                    linhas.append(json.loads(l))
                except json.JSONDecodeError:
                    pass
    return linhas

def resumo_velocidade(linhas, lote):
    tempos = [l["time"] for l in linhas if "time" in l]
    dados = [l["data_time"] for l in linhas if "data_time" in l]
    if not tempos:
        return {}
    tempos_s = sorted(tempos)
    mediana = statistics.median(tempos)
    return dict(
        lote=lote, imagens_por_s=lote / mediana if mediana else None,
        s_iter_media=statistics.mean(tempos), s_iter_mediana=mediana,
        s_iter_p90=tempos_s[int(0.9 * (len(tempos_s) - 1))], s_iter_min=tempos_s[0], s_iter_max=tempos_s[-1],
        data_time_media=statistics.mean(dados) if dados else None, amostras=len(tempos),
    )

def treinar(cfg, saida, aumentos=True):
    cfg.SOLVER.AMP.ENABLED = PERFIL["amp"]
    os.makedirs(saida, exist_ok=True)
    cfg.OUTPUT_DIR = saida
    Trainer.usar_aumentos = aumentos
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t = time.time()
    oom = 0
    while True:  # reativo: se faltar memória na GPU, recomeça com metade do lote (mesmo nº de imagens vistas)
        trainer = Trainer(cfg)
        trainer.resume_or_load(resume=True)
        it_inicial = trainer.start_iter
        try:
            trainer.train()
            break
        except torch.cuda.OutOfMemoryError:
            lote = cfg.SOLVER.IMS_PER_BATCH
            del trainer
            torch.cuda.empty_cache()
            if lote <= 1:
                raise
            novo = max(1, lote // 2)
            print(f"⚠ sem memória na GPU com lote {lote}: recomeçando com lote {novo}")
            cfg.SOLVER.BASE_LR, cfg.SOLVER.MAX_ITER = ajustar(lote, novo, cfg.SOLVER.BASE_LR, cfg.SOLVER.MAX_ITER)
            cfg.SOLVER.WARMUP_ITERS = min(cfg.SOLVER.MAX_ITER, ajustar(lote, novo, 1, cfg.SOLVER.WARMUP_ITERS)[1])
            cfg.SOLVER.IMS_PER_BATCH = novo
            for f in os.listdir(saida):  # checkpoints com outro lote não servem para retomar
                if f.endswith(".pth") or f in ("last_checkpoint", "metrics.json"):
                    os.remove(f"{saida}/{f}")
            oom += 1
    tempo = time.time() - t
    mem = torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else None
    t = time.time()
    coco = Trainer.test(cfg, trainer.model)
    tempo_aval = time.time() - t
    n_params = sum(p.numel() for p in trainer.model.parameters())
    del trainer
    torch.cuda.empty_cache()
    linhas = ler_metricas(f"{saida}/metrics.json")
    return dict(
        tempo_s=tempo, tempo_avaliacao_coco_s=tempo_aval, iteracao_inicial=it_inicial, retomado=it_inicial > 0,
        memoria_pico_mb=mem, parametros=n_params, velocidade=resumo_velocidade(linhas, cfg.SOLVER.IMS_PER_BATCH),
        lote=cfg.SOLVER.IMS_PER_BATCH, lr_efetivo=cfg.SOLVER.BASE_LR, iteracoes_efetivas=cfg.SOLVER.MAX_ITER,
        amp=cfg.SOLVER.AMP.ENABLED, reducoes_de_lote_por_memoria=oom,
        coco={k: {m: (None if (isinstance(v, float) and math.isnan(v)) else v) for m, v in d.items()} for k, d in coco.items()},
        curvas=[{k: v for k, v in l.items() if isinstance(v, (int, float))} for l in linhas if "iteration" in l],
    )
`),
code(`
# --- avaliação do juiz por foto, comparando com as marcações revisadas (referência) ---
from recognizer import Recognizer

def processar_bolas(img_shape, instancias):
    instancias = [i for i in instancias if i["class"] == "sports ball"]
    bolas, balim, vencedora = [], None, None
    if len(instancias) >= 2:
        margem = 0.10
        x_min, x_max = img_shape[1] * margem, img_shape[1] * (1 - margem)
        y_min, y_max = img_shape[0] * margem, img_shape[0] * (1 - margem)
        no_meio, nas_bordas = [], []
        for inst in instancias:
            c = inst["center"]
            (no_meio if x_min <= c["x"] <= x_max and y_min <= c["y"] <= y_max else nas_bordas).append(inst)
        if no_meio:
            no_meio.sort(key=lambda i: i["area"])
            while True:
                balim = no_meio.pop(0)
                if not no_meio:
                    break
                if math.fabs(no_meio[-1]["area"] / balim["area"] - no_meio[0]["area"] / balim["area"]) <= 1.5:
                    break
            nas_bordas = [b for b in nas_bordas if b["area"] >= balim["area"] / 2]
            bolas = no_meio + nas_bordas
            for b in bolas:
                b["distance"] = math.hypot(b["center"]["x"] - balim["center"]["x"], b["center"]["y"] - balim["center"]["y"])
            bolas.sort(key=lambda b: b["distance"])
            vencedora = bolas[0] if bolas else None
    return bolas, vencedora, balim

def referencia_por_imagem():
    por_img = {}
    for a in VAL["annotations"]:
        x, y, w, h = a["bbox"]
        por_img.setdefault(a["image_id"], []).append(dict(
            center={"x": x + w / 2, "y": y + h / 2}, box={"x1": x, "y1": y, "x2": x + w, "y2": y + h},
            area=a["area"], score=1.0, **{"class": "sports ball"}))
    ref = {}
    for im in VAL["images"]:
        bolas, venc, balim = processar_bolas((im["height"], im["width"]), por_img.get(im["id"], []))
        ref[im["file_name"]] = dict(n=len(por_img.get(im["id"], [])), bolas=bolas, vencedora=venc, balim=balim)
    return ref

def dentro(inst, alvo):
    if inst is None or alvo is None:
        return None
    b = alvo["box"]
    return bool(b["x1"] <= inst["center"]["x"] <= b["x2"] and b["y1"] <= inst["center"]["y"] <= b["y2"])

def avaliar_juiz(config_yaml, pesos):
    rec = Recognizer(config_file=config_yaml, weights_file=pesos, confidence_threshold=min(LIMIARES),
                     gpu_id=0 if torch.cuda.is_available() else None)
    ref = referencia_por_imagem()
    nomes = [im["file_name"] for im in VAL["images"]]
    img0 = Recognizer.read_file(f"{DATA_DIR}/{nomes[0]}")
    for _ in range(3):  # aquecimento da GPU
        rec.predict_data(img0)
    por_foto, t0 = [], time.time()
    for nome in tqdm(nomes, desc="juiz"):
        img = Recognizer.read_file(f"{DATA_DIR}/{nome}")
        if torch.cuda.is_available(): torch.cuda.synchronize()
        t = time.time()
        todas = rec.predict_data(img)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        ms = (time.time() - t) * 1000
        r = ref[nome]
        for lim in LIMIARES:
            dets = [dict(i) for i in todas if i["score"] >= lim]
            bolas, venc, balim = processar_bolas(img.shape, dets)
            scores = [d["score"] for d in dets]
            por_foto.append(dict(
                foto=nome, limiar=lim, largura=img.shape[1], altura=img.shape[0], inferencia_ms=ms,
                n_detectadas=len(dets), decidiu=balim is not None, n_bolas=len(bolas),
                dist_vencedora=venc["distance"] if venc else None,
                dist_segunda=bolas[1]["distance"] if len(bolas) > 1 else None,
                margem_px=(bolas[1]["distance"] - venc["distance"]) if len(bolas) > 1 else None,
                balim_diametro_px=float((balim["box"]["x2"] - balim["box"]["x1"] + balim["box"]["y2"] - balim["box"]["y1"]) / 2) if balim else None,
                score_min=min(scores) if scores else None, score_medio=float(np.mean(scores)) if scores else None,
                ref_n_bolas=r["n"], ref_decidiu=r["balim"] is not None,
                acerto_balim=dentro(balim, r["balim"]), acerto_vencedora=dentro(venc, r["vencedora"]),
            ))
    tempo_total = time.time() - t0
    resumo = []
    for lim in LIMIARES:
        linhas = [p for p in por_foto if p["limiar"] == lim]
        com_ref = [p for p in linhas if p["ref_decidiu"]]
        avaliaveis = [p for p in com_ref if p["acerto_vencedora"] is not None]
        ms = sorted(p["inferencia_ms"] for p in linhas)
        resumo.append(dict(
            limiar=lim, fotos=len(linhas),
            taxa_decisao=sum(p["decidiu"] for p in linhas) / len(linhas),
            acuracia_vencedora=(sum(p["acerto_vencedora"] for p in avaliaveis) / len(com_ref)) if com_ref else None,
            acuracia_balim=(sum(bool(p["acerto_balim"]) for p in com_ref) / len(com_ref)) if com_ref else None,
            fotos_com_referencia=len(com_ref),
            erro_contagem_medio=float(np.mean([abs(p["n_detectadas"] - p["ref_n_bolas"]) for p in linhas])),
            bolas_media=float(np.mean([p["n_bolas"] for p in linhas])),
            inferencia_ms_media=float(np.mean(ms)), inferencia_ms_p50=ms[len(ms) // 2], inferencia_ms_p90=ms[int(0.9 * (len(ms) - 1))],
        ))
    del rec
    torch.cuda.empty_cache()
    return dict(tempo_s=tempo_total, por_limiar=resumo, por_foto=por_foto)
`),
code(`
def ambiente():
    def cmd(c):
        try:
            return subprocess.run(c, shell=True, capture_output=True, text=True, timeout=30).stdout.strip()
        except Exception:
            return None
    smi = cmd("nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader")
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / 2**30
    except Exception:
        ram_gb = None
    return dict(
        gpu=GPU, gpu_nome=GPU_NOME, nvidia_smi=smi, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
        torch=torch.__version__, detectron2=detectron2.__version__, python=platform.python_version(),
        cpus=os.cpu_count(), ram_gb=ram_gb, sistema=platform.platform(),
        repo_commit=cmd(f"git -C {REPO_DIR} rev-parse --short HEAD"), setup_s=T_SETUP_S,
    )

def rodar(exp, exp_id=None, progresso=None):
    """Roda um experimento e grava o relatório. progresso(etapa, it, total, loss) -> True cancela."""
    exp_id = exp_id or f"{exp['nome']}__{GPU}"
    pasta = f"{EXPERIMENTOS_DIR}/{exp_id}"
    if os.path.exists(f"{pasta}/relatorio.json"):
        print(f"✔ {exp_id}: já tem relatório, pulando")
        with open(f"{pasta}/relatorio.json") as f:
            return json.load(f)
    os.makedirs(pasta, exist_ok=True)
    def avisar(etapa):
        Trainer.ao_progresso = (lambda it, total, loss: progresso(etapa, it, total, loss)) if progresso else None
    if exp["etapa1"] == "auto":  # reaproveita a etapa 1 já treinada deste modelo; senão treina
        ja_tem = exp["modelo"] == "R50" or os.path.exists(f"{EXPERIMENTOS_DIR}/_etapa1_{exp['modelo']}/model_final.pth")
        exp = {**exp, "etapa1": "reusar" if ja_tem else "treinar"}
    print(f"\\n▶ {exp_id}")
    rel = dict(id=exp_id, nome=exp["nome"], grupos=exp["grupos"], gpu=GPU,
               parametros={k: v for k, v in exp.items() if k not in ("nome", "grupos")},
               inicio=datetime.datetime.now().isoformat(timespec="seconds"),
               ambiente=ambiente(), perfil_gpu=PERFIL, dataset=DATASET_INFO, etapa1=None, etapa2=None, juiz=None)
    t_exp = time.time()

    # etapa 1
    if exp["etapa1"] == "reusar":
        pesos1 = ETAPA1_R50 if exp["modelo"] == "R50" else f"{EXPERIMENTOS_DIR}/_etapa1_{exp['modelo']}/model_final.pth"
        rel["etapa1"] = dict(modo="reusar", pesos=pesos1)
    else:
        preparar_coco()
        cfg1 = montar_cfg(exp["modelo"], 1)
        cfg1.MODEL.WEIGHTS = PESOS_COCO[exp["modelo"]]
        lote1 = lote_para(exp["modelo"], 1)
        lr1, it1 = ajustar(2, lote1, 0.005, exp["it1"])
        cfg1.SOLVER.IMS_PER_BATCH = lote1
        cfg1.SOLVER.BASE_LR = lr1
        cfg1.SOLVER.MAX_ITER = it1
        cfg1.SOLVER.CHECKPOINT_PERIOD = max(100, it1 // 15)
        cfg1.SOLVER.WARMUP_ITERS = min(it1, ajustar(2, lote1, 1, cfg1.SOLVER.WARMUP_ITERS)[1])
        saida1 = f"{pasta}/etapa1" if exp["etapa1"] == "benchmark" else f"{EXPERIMENTOS_DIR}/_etapa1_{exp['modelo']}"
        avisar("etapa1")
        r1 = treinar(cfg1, saida1)
        pesos1 = f"{saida1}/model_final.pth"
        rel["etapa1"] = dict(modo=exp["etapa1"], pesos=pesos1, iteracoes=exp["it1"], **r1)
        if exp["etapa1"] == "benchmark":  # só mede velocidade; não guarda o modelo
            for f in os.listdir(saida1):
                if f.endswith(".pth"):
                    os.remove(f"{saida1}/{f}")

    # etapa 2
    if exp["it2"] > 0:
        registrar_bocha()
        cfg2 = montar_cfg(exp["modelo"], 2)
        cfg2.MODEL.WEIGHTS = pesos1
        lote2 = lote_para(exp["modelo"], 2, exp["aumentos"])
        lr2, it2 = ajustar(3, lote2, exp["lr2"], exp["it2"])
        cfg2.SOLVER.IMS_PER_BATCH = lote2
        cfg2.SOLVER.BASE_LR = lr2
        cfg2.SOLVER.MAX_ITER = it2
        cfg2.SOLVER.CHECKPOINT_PERIOD = max(50, it2 // 4)
        cfg2.SOLVER.WARMUP_ITERS = min(it2, ajustar(3, lote2, 1, cfg2.SOLVER.WARMUP_ITERS)[1])  # aquecimento encolhe junto
        saida2 = f"{pasta}/etapa2"
        avisar("etapa2")
        r2 = treinar(cfg2, saida2, aumentos=exp["aumentos"])
        avisar(None)
        if progresso:
            progresso("juiz", 0, 1, None)
        cfg_export = cfg2.clone()
        cfg_export.MODEL.WEIGHTS = "models/model/model_final.pth"
        with open(f"{pasta}/config.yaml", "w") as f:
            f.write(cfg_export.dump())
        pesos2 = f"{saida2}/model_final.pth"
        rel["etapa2"] = dict(iteracoes=exp["it2"], lr=exp["lr2"], aumentos=exp["aumentos"],
                             modelo_mb=os.path.getsize(pesos2) / 2**20, **r2)
        rel["juiz"] = avaliar_juiz(f"{pasta}/config.yaml", pesos2)
        if SALVAR_MODELO:
            shutil.copy(pesos2, f"{pasta}/model_final.pth")
        for f in os.listdir(saida2):
            if f.endswith(".pth"):
                os.remove(f"{saida2}/{f}")

    rel["fim"] = datetime.datetime.now().isoformat(timespec="seconds")
    rel["tempo_total_s"] = time.time() - t_exp
    # CSVs para leitura humana + JSON completo para o app
    import pandas as pd
    if rel["juiz"]:
        pd.DataFrame(rel["juiz"]["por_foto"]).to_csv(f"{pasta}/por_foto.csv", index=False)
        pd.DataFrame(rel["juiz"]["por_limiar"]).to_csv(f"{pasta}/por_limiar.csv", index=False)
    for etapa in ("etapa1", "etapa2"):
        if rel[etapa] and rel[etapa].get("curvas"):
            pd.DataFrame(rel[etapa]["curvas"]).to_csv(f"{pasta}/curvas_{etapa}.csv", index=False)
    with open(f"{pasta}/relatorio.json", "w") as f:
        json.dump(rel, f, ensure_ascii=False, default=str)
    resumo = rel["juiz"]["por_limiar"][LIMIARES.index(0.82)] if rel["juiz"] else {}
    print(f"✔ {exp_id} em {rel['tempo_total_s'] / 60:.1f} min"
          + (f" | decide {resumo['taxa_decisao']:.0%} | acerta vencedora {resumo['acuracia_vencedora'] or 0:.0%} (limiar 0.82)" if resumo else ""))
    return rel
`),

md(`## 5. Rodar a fila`),
code(`
import traceback

def resumo_do(rel):
    j = next((l for l in (rel.get("juiz") or {}).get("por_limiar", []) if abs(l["limiar"] - 0.82) < 1e-6), {})
    e1, e2 = rel.get("etapa1") or {}, rel.get("etapa2") or {}
    return dict(acerta_vencedora=j.get("acuracia_vencedora"), decide=j.get("taxa_decisao"),
                ap_segm=(e2.get("coco") or {}).get("segm", {}).get("AP"), ap_bbox=(e2.get("coco") or {}).get("bbox", {}).get("AP"),
                img_s_etapa1=(e1.get("velocidade") or {}).get("imagens_por_s"),
                img_s_etapa2=(e2.get("velocidade") or {}).get("imagens_por_s"), tempo_total_s=rel.get("tempo_total_s"))

def pegar_proximo():
    """Reserva (transação) o treino mais antigo da fila para esta GPU; retoma também treinos órfãos desta GPU."""
    agora = datetime.datetime.now(datetime.timezone.utc)
    candidatos = []
    for d in FILA_FS.where("status", "in", ["na_fila", "rodando"]).stream():
        j = d.to_dict()
        if (j.get("gpu_desejada") or "qualquer") not in (GPU, "qualquer"):
            continue
        batida = j.get("batimento")
        orfao = j["status"] == "rodando" and (batida is None or (agora - batida).total_seconds() > 600)
        if j["status"] == "na_fila" or orfao:
            candidatos.append((j.get("criado_em") or agora, d.reference))
    for _, ref in sorted(candidatos, key=lambda c: c[0]):
        @firestore.transactional
        def reservar(tx, ref=ref):
            snap = ref.get(transaction=tx).to_dict()
            if snap["status"] not in ("na_fila", "rodando"):
                return None
            tx.update(ref, {"status": "rodando", "iniciado_em": firestore.SERVER_TIMESTAMP,
                            "batimento": firestore.SERVER_TIMESTAMP, "gpu": GPU, "gpu_nome": GPU_NOME,
                            "perfil_gpu": PERFIL, "tentativas": firestore.Increment(1), "erro": None})
            return snap
        job = reservar(FS.transaction())
        if job:
            return ref, job
    return None, None

def executar_do_app():
    ultimo_trabalho = time.time()
    while True:
        ref, job = pegar_proximo()
        if not ref:
            if time.time() - ultimo_trabalho > ESPERAR_NOVOS_MIN * 60:
                print("Fila vazia — encerrando."); break
            time.sleep(30); continue
        exp = {**job["parametros"], "nome": job["nome"], "grupos": ["app"]}
        print(f"\\n▶ {ref.id}: {job['nome']} ({exp})")
        ultimo_aviso = [0.0]
        def progresso(etapa, it, total, loss):
            if time.time() - ultimo_aviso[0] < 15 and it not in (0, total):
                return False
            ultimo_aviso[0] = time.time()
            ref.update({"progresso": {"etapa": etapa, "iteracao": it, "total": total, "loss": loss},
                        "batimento": firestore.SERVER_TIMESTAMP})
            return (ref.get(["cancelar"]).to_dict() or {}).get("cancelar", False)
        try:
            rel = rodar(exp, exp_id=ref.id, progresso=progresso)
            rel.update(id=ref.id, job_id=ref.id)
            blob = BUCKET.blob(f"benchmark/relatorios/{ref.id}/relatorio.json")
            blob.upload_from_string(json.dumps(rel, ensure_ascii=False, default=str), content_type="application/json")
            ref.update({"status": "concluido", "concluido_em": firestore.SERVER_TIMESTAMP, "relatorio": blob.name,
                        "resumo": resumo_do(rel), "progresso": {"etapa": "fim", "iteracao": 1, "total": 1}})
        except Cancelado:
            ref.update({"status": "cancelado", "concluido_em": firestore.SERVER_TIMESTAMP})
            print("✘ cancelado pelo app")
        except Exception as e:
            ref.update({"status": "erro", "erro": traceback.format_exc()[-3000:], "concluido_em": firestore.SERVER_TIMESTAMP})
            print("✘ erro:", e)
        finally:
            Trainer.ao_progresso = None
        ultimo_trabalho = time.time()

if FONTE_FILA == "app":
    executar_do_app()
else:
    for exp in fila:
        rodar(exp)
    print(f"\\nFila concluída na {GPU}. Relatórios em {EXPERIMENTOS_DIR}")
`),

md(`
## 6. (Opcional) Relatório dos treinos que já existem
Gera relatórios para os modelos treinados antes da fila — \`treinoA100\` e \`treino\` (G4) — lendo as métricas gravadas no treino e avaliando os dois **na mesma validação congelada**. Rode uma vez só.`),
code(`
IMPORTAR = {  # nome -> (pasta do treino, GPU em que foi treinado)
    "pipeline_original": (f"{BASE}/treinoA100", "A100"),
    "pipeline_original_g4": (f"{BASE}/treino", "G4"),
}
for nome, (pasta_treino, gpu_treino) in IMPORTAR.items():
    exp_id = f"{nome}__{gpu_treino}"
    pasta = f"{EXPERIMENTOS_DIR}/{exp_id}"
    pesos = f"{pasta_treino}/modelo_final/model_final.pth"
    if os.path.exists(f"{pasta}/relatorio.json") or not os.path.exists(pesos):
        print(("✔ já importado: " if os.path.exists(f"{pasta}/relatorio.json") else "✘ sem modelo: ") + exp_id)
        continue
    os.makedirs(pasta, exist_ok=True)
    registrar_bocha()
    cfg = get_cfg(); cfg.merge_from_file(f"{pasta_treino}/modelo_final/config.yaml"); cfg.MODEL.WEIGHTS = pesos
    from detectron2.modeling import build_model
    from detectron2.checkpoint import DetectionCheckpointer
    modelo = build_model(cfg); DetectionCheckpointer(modelo).load(pesos)
    t = time.time(); coco = Trainer.test(cfg, modelo); t_coco = time.time() - t
    n_params = sum(p.numel() for p in modelo.parameters()); del modelo
    rel = dict(id=exp_id, nome=nome, grupos=["importado"], gpu=gpu_treino,
               parametros=dict(modelo="R50", etapa1="treinar", it1=15000, it2=1000, lr2=0.0025, aumentos=True),
               inicio=None, ambiente=dict(ambiente(), gpu=gpu_treino, gpu_avaliacao=GPU), dataset=DATASET_INFO,
               perfil_gpu=dict(adaptado=False, lote1=2, lote2=3, amp=False, tf32=False),
               observacao="Treino feito antes da fila; métricas COCO e juiz medidas na validação congelada da A100.")
    for etapa, sub in (("etapa1", "etapa1_coco_sports_ball"), ("etapa2", "etapa2_bocha")):
        linhas = ler_metricas(f"{pasta_treino}/{sub}/metrics.json")
        lote = 2 if etapa == "etapa1" else 3
        rel[etapa] = dict(modo="treinar", iteracoes=max([l["iteration"] for l in linhas if "time" in l] or [-1]) + 1,
                          lote=lote, amp=False, velocidade=resumo_velocidade(linhas, lote),
                          curvas=[{k: v for k, v in l.items() if isinstance(v, (int, float))} for l in linhas if "iteration" in l])
    rel["etapa2"].update(lr=0.0025, aumentos=True, parametros=n_params, modelo_mb=os.path.getsize(pesos) / 2**20,
                         tempo_avaliacao_coco_s=t_coco,
                         coco={k: {m: (None if (isinstance(v, float) and math.isnan(v)) else v) for m, v in d.items()} for k, d in coco.items()})
    rel["juiz"] = avaliar_juiz(f"{pasta_treino}/modelo_final/config.yaml", pesos)
    rel["fim"] = datetime.datetime.now().isoformat(timespec="seconds")
    import pandas as pd
    pd.DataFrame(rel["juiz"]["por_foto"]).to_csv(f"{pasta}/por_foto.csv", index=False)
    pd.DataFrame(rel["juiz"]["por_limiar"]).to_csv(f"{pasta}/por_limiar.csv", index=False)
    with open(f"{pasta}/relatorio.json", "w") as f:
        json.dump(rel, f, ensure_ascii=False, default=str)
    print("✔ importado:", exp_id)
`),
];

const nb = {
  cells,
  metadata: {
    accelerator: 'GPU',
    colab: { provenance: [], gpuType: 'A100' },
    kernelspec: { display_name: 'Python 3', name: 'python3' },
    language_info: { name: 'python' },
  },
  nbformat: 4,
  nbformat_minor: 0,
};
fs.writeFileSync(out, JSON.stringify(nb, null, 1) + '\n', 'utf8');
console.log('ok', out, cells.length, 'células');
