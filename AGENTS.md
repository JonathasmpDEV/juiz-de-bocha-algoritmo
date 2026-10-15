# Instruções para agentes (Claude, Codex, Gemini, Copilot…)

Projeto **Juiz de Bocha**: reconhece as bolas numa foto de bocha (Detectron2 / Mask R-CNN), acha o **balim**
(a menor bola) e aponta a **bola vencedora** (a mais próxima dele). Responda ao usuário em **português do Brasil**.

## Mapa do repositório

| Pasta / arquivo | O que é |
|---|---|
| `recognizer.py`, `cloud-run-function/main.py` | Reconhecedor e lógica do juiz em produção (`__process_balls`: classe *sports ball*, margem de 10% na borda, menor bola viável = balim, vencedora = mais próxima). Confiança de produção **0.82**. |
| `models/*/…yaml` | Configs Detectron2 do modelo (R50-FPN). |
| `train/`, `train_custom_dataset/` | Pipeline de treino original (scripts locais). |
| `colab/` | **Notebooks do Colab** — é aqui que se treina e testa (ver abaixo). |
| `colab/geradores/build_experimentos.js` | Gerador opcional do notebook de experimentos (`node build_experimentos.js saida.ipynb`). Pode editar o `.ipynb` direto; se usar o gerador, mantenha os dois iguais. |
| `benchmark-app/` | App local (Streamlit + SQLite) que compara os treinos e mostra os envios dos usuários. |
| `DOCUMENTACAO.md`, `README.MD` | Documentação original do projeto. |

## Notebooks (`colab/`)

Abra direto do GitHub no Colab: `https://colab.research.google.com/github/JonathasmpDEV/juiz-de-bocha-algoritmo/blob/main/colab/<arquivo>.ipynb`.
Todos montam o Google Drive do usuário e gravam em `MyDrive/00-Juiz de Bocha/`.

| Notebook | Para quê |
|---|---|
| `JuizDeBocha_Treino.ipynb` | Pipeline original completo: COCO 2017 *sports ball* → rotula as fotos de bocha → revisão → ajuste fino → validação a 0.82. Grava em `00-Juiz de Bocha/treino` (**cuidado: rodar de novo sobrescreve o treino G4 que está lá**; o da A100 está em `treinoA100`). |
| `JuizDeBocha_Teste.ipynb` | Teste visual (Gradio) dos modelos treinados: A100 × G4, aba de comparação e galeria. |
| `JuizDeBocha_Experimentos.ipynb` | **Benchmark.** Roda a `FILA` de experimentos (GPU, iterações, LR, modelo R50/R101/X101/Cascade R50, aumentos) e grava `00-Juiz de Bocha/experimentos/<nome>__<GPU>/relatorio.json`. Padrão `FONTE_FILA = "lista"` (sem integração). |
| `JuizDeBocha_Colab.ipynb` | Primeiro notebook: roda o reconhecedor com pesos COCO, sem treino. |

Detalhes que já foram decididos e testados no notebook de experimentos — não desfaça sem motivo:
- **Reativo à GPU**: lote = memória da GPU ÷ 1 GB (teto 24; T4 15, L4 22, A100/G4 24), dividido pelo peso do modelo
  (`PESO_MODELO`: R50 1.0, R101 1.3, Cascade 1.4, X101 2.0) e por 2,5 na etapa 1 / sem aumentos. LR e iterações seguem a
  regra linear (`ajustar`), warmup também escala. AMP se compute capability ≥ 7, TF32 se ≥ 8.
- **OOM**: o treino captura `torch.cuda.OutOfMemoryError`, divide o lote por 2, reescala e recomeça (`reducoes_de_lote_por_memoria` no relatório).
- `DatasetMapper(..., recompute_boxes=True)`: sem isso a `RandomRotation` afrouxa as caixas e o AP75 de caixa vai a ~0.
- A avaliação do juiz compara, foto a foto, com os rótulos **congelados da A100** (`treinoA100`), nos limiares 0.5/0.6/0.7/0.82/0.9.

## App de benchmark (`benchmark-app/`)

- Rodar: `iniciar.bat` (Windows, dois cliques), `iniciar.ps1` (PowerShell), `bash iniciar.sh` (Linux),
  `iniciar.command` (macOS). Todos usam o [`uv`](https://docs.astral.sh/uv/) e abrem http://localhost:8501.
  Equivalente manual: `cd benchmark-app && uv run streamlit run app.py`.
- Primeira vez num clone novo: **Importar → Importar da pasta relatorios/** recria o banco `dados/benchmark.db`
  a partir de `relatorios/*/relatorio.json` (versionados).
- Novos relatórios do Drive: copie a pasta `experimentos/<nome>__<GPU>/` para `relatorios/` ou use a página Importar.
  `importar_download.py` importa o JSON baixado pelo conector do Google Drive (`{content: base64}`);
  `importar_metrics.py` gera relatório a partir de `metrics.json` de treinos antigos.
- Duas áreas: **🧪 Testes de modelos** (controle, visão geral, comparar, GPUs e custo, por foto, curvas, importar, exportar CSV/Excel/ZIP/SQLite)
  e **📷 Envios dos usuários** (cópia local de `users` e `Photos` do Firestore, separada por UID).
- Firebase (opcional; só para *Controle de treinos* e *Envios*): chave do Firebase Admin do projeto `juizdebocha`
  em `benchmark-app/credenciais/firebase-admin.json`; no Colab, o conteúdo vai no secret `FIREBASE_ADMIN_JSON`.
  Fila de treinos no Firestore: coleção `benchmark_treinos`. Storage: `juizdebocha.appspot.com`.
- Configuração opcional em `benchmark-app/.env` (modelo em `.env.example`): `BENCH_DB`, `BENCH_RELATORIOS`, `BENCH_FIREBASE_CRED`.
- Verificar sintaxe sem instalar Python: `uv run --no-project python -m py_compile benchmark-app/*.py`.

## Regras

- **Repositório público.** Nunca versione chaves, tokens, `firebase-admin.json`, `.env`, `dados/` (banco local) nem pesos `.pth`
  (ficam no Drive, ~350 MB cada). Localização dos usuários é dado pessoal: o mapa dos envios fica desligado por padrão.
- Não use o arquivo multipartes RAR/ZIP do Drive — não serve. As fotos válidas estão em
  `MyDrive/BACKUP PENDRIVE 256/CursoSegImgPython/bocha` (125 fotos: 100 treino / 25 validação).
- Recomendações de modelo: sempre **comparável ou melhor** que o R50 atual, nunca pior (empate geral é aceitável).
- O usuário roda o Colab manualmente e vê os resultados no app ("sem integrar"); o modo `FONTE_FILA = "app"` existe mas é opcional.
- Windows: o terminal do usuário é PowerShell 5.1 (`&&` não funciona; use `;`). Mensagens de commit em português, no estilo `tipo: descrição`.

## Estado atual (15/10/2026)

Resultados (validação, 25 fotos; "acerta" = vencedora correta a 0.82):

| Experimento | GPU | Lote | img/s | AP caixa | AP máscara | Decide | Acerta | Tempo |
|---|---|---|---|---|---|---|---|---|
| `pipeline_original` | A100 | 3 | 22.6 | 32.6 | 90.5 | 96% | 88% | — |
| `pipeline_original_g4` | G4 | 3 | 42.5 | 21.1 | 90.1 | 96% | 96% | — |
| `base_r50_it1000` | L4 | 4 | 15.4 | 86.9 | 93.5 | 96% | 92% | 4.0 min |
| `base_r50_it1000` | G4 | 12 | 61.4 | 84.5 | 92.8 | 96% | 96% | 1.2 min |

- G4 (RTX PRO 6000 Blackwell) foi ~1,8× mais rápida que a A100 com o mesmo lote.
- O AP de caixa baixo dos `pipeline_original` é o problema da `RandomRotation` (corrigido com `recompute_boxes=True`).
- No Drive também existem `bench_etapa1_r50__L4` e `bench_etapa1_r50__G4` (só etapa 1/COCO, sem avaliação do juiz) — ainda não estão em `relatorios/`.

Próximos passos:
1. Rodar o resto da `FILA` (iterações 3000/5000, LR, sem aumentos, R101, Cascade R50, X101) com a versão atual do notebook
   (lote generoso + retry de OOM — ainda não testada numa GPU real) e importar os relatórios.
2. Comparar os modelos novos com o R50 pelo critério acima e escolher o modelo de produção.
3. No Drive, o `JuizDeBocha_Experimentos.ipynb` antigo (`1WWeGs_MDszu7Wtvv8XLRhKrtimqDY6EJ`) está desatualizado; use o do GitHub.
4. Arquivos criados no Drive em 15/10/2026 fora desta sessão (`JuizDeBocha_Treino_Aprimorado.ipynb`, `main.py`,
   `recognizer.py` e a pasta `app/` com `Dockerfile`) **não estão versionados aqui** — confirme com o usuário antes de trazê-los.
