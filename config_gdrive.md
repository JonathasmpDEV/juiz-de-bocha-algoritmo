# Configurar Google Drive

Esta página ajuda você a gerar o conteúdo para o seu arquivo `gdrive_config.json`.

1.  **Obtenha a URL da Pasta Raiz do Google Drive:**
    *   No Google Drive, navegue até a pasta que você deseja usar como raiz para este projeto.
    *   Dentro desta pasta, você deverá criar subpastas chamadas `datasets` (para seus dados de treinamento) e `models` (para seus arquivos de modelo `config.yaml` e `weights.pkl`).
    *   Copie a URL da barra de endereços do seu navegador. Ela deve ser algo como `https://drive.google.com/drive/folders/SEU_FOLDER_ID_AQUI`.

2.  **Cole a URL abaixo:**
    <div class="config-gdrive-input">
      <label for="gdriveUrl">URL da Pasta Raiz do Google Drive:</label>
      <input type="text" id="gdriveUrl" name="gdriveUrl" style="width: 80%; padding: 8px;" placeholder="https://drive.google.com/drive/folders/...">
    </div>

3.  **Defina os Caminhos Locais (Opcional):**
    Estes são os diretórios locais onde os dados serão baixados. Os padrões são geralmente adequados.
    <div class="config-gdrive-input">
      <label for="localDatasetsPath">Caminho Local para Datasets:</label>
      <input type="text" id="localDatasetsPath" name="localDatasetsPath" style="width: 80%; padding: 8px;" value="./datasets_gdrive">
    </div>
    <div class="config-gdrive-input">
      <label for="localModelsPath">Caminho Local para Modelos:</label>
      <input type="text" id="localModelsPath" name="localModelsPath" style="width: 80%; padding: 8px;" value="./models_from_gdrive">
    </div>

4.  **Gere o Conteúdo do `gdrive_config.json`:**
    <button onclick="generateGdriveConfig()">Gerar JSON</button>
    <pre id="gdriveConfigOutput" style="background-color: #f5f5f5; padding: 10px; border: 1px solid #ccc; white-space: pre-wrap; word-wrap: break-word;"></pre>

5.  **Crie/Atualize seu `gdrive_config.json`:**
    *   Copie o JSON gerado acima.
    *   Crie um arquivo chamado `gdrive_config.json` na raiz do seu projeto (se ainda não existir).
    *   Cole o JSON copiado neste arquivo e salve-o.

**Lembre-se:** Você também precisará do `gdrive_credentials.json` e de executar `python gdrive_auth.py` conforme descrito na seção principal da documentação do Google Drive.
