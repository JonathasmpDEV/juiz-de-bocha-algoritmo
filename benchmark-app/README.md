# Benchmark de treinos — Juiz de Bocha

App local (Streamlit + SQLite) que junta os relatórios gerados pelo notebook
`colab/JuizDeBocha_Experimentos.ipynb` (os resultados vão para o Drive em `00-Juiz de Bocha/experimentos/`) e permite comparar e exportar tudo.

## Duas áreas

- **🧪 Testes de modelos** — controle dos treinos (fila no Firestore `benchmark_treinos`), comparação e exportação.
- **📷 Envios dos usuários** — cópia local das coleções `users` e `Photos` do app (Firebase Auth + Firestore),
  separada por UID, com galeria, mapa opcional e CSV por usuário.

Ambas precisam da chave do Firebase Admin do projeto `juizdebocha` em `credenciais/firebase-admin.json`
(Console → Configurações → Contas de serviço → Gerar nova chave privada). A pasta `credenciais/` está no `.gitignore`.
No Colab, a mesma chave vai no secret `FIREBASE_ADMIN_JSON`.

## Controle de treinos

1. No app, **Controle de treinos → Novo treino** (modelo R50/R101/X101/Cascade R50, iterações, LR, aumentos,
   repetições e GPU) ou **Fila padrão do benchmark**.
2. Abra `JuizDeBocha_Experimentos.ipynb` no Colab **com a GPU pedida** e use *Executar tudo*
   (`FONTE_FILA = "app"`). O Colab não permite que o app ligue a sessão ou escolha a GPU.
3. O progresso aparece no app a cada ~15 s; *Cancelar* interrompe; *Rodar novamente* clona o treino
   (podendo trocar a GPU). Ao concluir, o relatório volta pelo Firebase Storage e entra no banco sozinho.

## Fluxo manual (sem Firebase)

1. **Colab:** abra [`JuizDeBocha_Experimentos.ipynb`](https://colab.research.google.com/github/JonathasmpDEV/juiz-de-bocha-algoritmo/blob/main/colab/JuizDeBocha_Experimentos.ipynb), escolha a GPU e rode *Executar tudo*.
   Cada experimento grava `00-Juiz de Bocha/experimentos/<nome>__<gpu>/relatorio.json`.
2. **Trazer os relatórios:** copie as pastas de `experimentos/` para `relatorios/` aqui
   (só o `relatorio.json` é necessário), ou envie os arquivos pela página **Importar**.
3. **App:** abra com o lançador do seu sistema (todos usam o `uv` e abrem em http://localhost:8501):

   | Sistema | Como abrir |
   |---|---|
   | Windows | dois cliques em `iniciar.bat` — ou, no PowerShell, `& ".\iniciar.ps1"` |
   | Linux | `bash iniciar.sh` |
   | macOS | dois cliques em `iniciar.command` (na primeira vez: `chmod +x iniciar.command`) |

   Depois vá em
   **Importar → Importar da pasta relatorios/**. Reimportar o mesmo experimento substitui a versão anterior.

Num clone novo, **Importar → Importar da pasta relatorios/** recria o banco a partir dos relatórios versionados.
Configuração opcional: copie `.env.example` para `.env`.

## Páginas

| Página | O que mostra |
|---|---|
| Visão geral | Ranking por acerto da vencedora, decisão, AP, velocidade, tempo e memória |
| Comparar experimentos | Acerto, AP COCO, acerto/decisão por limiar e todos os parâmetros lado a lado |
| GPUs e custo | Imagens/s por GPU, inferência, custo estimado (unidades Colab/hora editáveis) |
| Por foto | Mapa foto × experimento (✔ acertou · ✘ errou · – sem decisão) e fotos mais difíceis |
| Curvas de treino | Qualquer métrica gravada no treino, por iteração ou por imagens vistas |
| Exportar | CSV por tabela, ZIP com tudo, planilha Excel e o próprio banco SQLite |

## Banco (`dados/benchmark.db`)

`experimentos` (uma linha por experimento + JSON original completo), `coco`, `juiz_limiar`,
`por_foto`, `curvas` (formato longo: experimento, etapa, iteração, métrica, valor),
`ambiente` (GPU, versões, perfil reativo, dataset, parâmetros) e `gpu_custos`.

`gerar_exemplo.py` cria relatórios **fictícios** só para testar o app; não importe esses no banco real.
