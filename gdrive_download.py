import os
import io
import json
import re
import argparse
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2.credentials import Credentials # Usado para carregar o token.json
# Importar a função de autenticação do gdrive_auth.py pode ser útil,
# mas para manter o script independente, vamos assumir que token.json já existe
# ou que o usuário o gerou. Se não, ele falhará, o que é esperado.

CONFIG_PATH = 'gdrive_config.json'
TOKEN_PATH = 'gdrive_token.json'

def get_gdrive_service():
    """Cria e retorna um objeto de serviço da API do Google Drive autenticado."""
    if not os.path.exists(TOKEN_PATH):
        print(f"ERRO: Arquivo de token '{TOKEN_PATH}' não encontrado.")
        print("Por favor, execute o script 'gdrive_auth.py' primeiro para autenticar.")
        return None
    try:
        creds = Credentials.from_authorized_user_file(TOKEN_PATH)
        if not creds or not creds.valid:
            # Idealmente, aqui também tentaria um refresh se creds.expired and creds.refresh_token
            # Mas para simplificar, vamos pedir para re-autenticar via gdrive_auth.py
            print("Credenciais inválidas ou expiradas. Por favor, execute 'gdrive_auth.py' para re-autenticar.")
            return None
        service = build('drive', 'v3', credentials=creds, static_discovery=False) # static_discovery=False para evitar problemas em alguns ambientes
        return service
    except Exception as e:
        print(f"Erro ao criar serviço do Google Drive: {e}")
        return None

def extract_folder_id_from_url(url):
    """Extrai o ID da pasta de uma URL do Google Drive."""
    match = re.search(r'/folders/([a-zA-Z0-9_-]+)', url)
    if match:
        return match.group(1)
    match = re.search(r'id=([a-zA-Z0-9_-]+)', url) # Para URLs com ?id=
    if match:
        return match.group(1)
    print(f"Não foi possível extrair o Folder ID da URL: {url}")
    return None

def download_file(service, file_id, file_name, local_folder_path):
    """Baixa um único arquivo do Google Drive."""
    try:
        request = service.files().get_media(fileId=file_id)
        os.makedirs(local_folder_path, exist_ok=True)
        file_path = os.path.join(local_folder_path, file_name)

        fh = io.FileIO(file_path, 'wb')
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        print(f"Baixando '{file_name}' para '{file_path}'...", end='', flush=True)
        while done is False:
            status, done = downloader.next_chunk()
            if status:
                print(f"{int(status.progress() * 100)}%...", end='', flush=True)
        print(" Completo.")
        return True
    except Exception as e:
        print(f"Erro ao baixar o arquivo '{file_name}' (ID: {file_id}): {e}")
        return False

def list_and_download_folder_contents(service, folder_id, local_base_path, current_relative_path=""):
    """Lista e baixa o conteúdo de uma pasta do Google Drive recursivamente."""
    query = f"'{folder_id}' in parents and trashed=false"
    try:
        results = service.files().list(
            q=query,
            pageSize=1000, # Max 1000, pode precisar de paginação para pastas muito grandes
            fields="nextPageToken, files(id, name, mimeType)"
        ).execute()
        items = results.get('files', [])

        if not items:
            # print(f"Nenhum arquivo encontrado na pasta com ID: {folder_id} (Caminho relativo: {current_relative_path if current_relative_path else '.'})")
            return

        for item in items:
            item_name = item['name']
            item_id = item['id']
            item_mime_type = item['mimeType']

            local_current_folder_path = os.path.join(local_base_path, current_relative_path)

            if item_mime_type == 'application/vnd.google-apps.folder':
                print(f"Entrando na subpasta: '{item_name}'")
                new_relative_path = os.path.join(current_relative_path, item_name)
                list_and_download_folder_contents(service, item_id, local_base_path, new_relative_path)
            else:
                # Não é uma pasta, então é um arquivo para baixar
                download_file(service, item_id, item_name, local_current_folder_path)
    except Exception as e:
        print(f"Erro ao listar/baixar conteúdo da pasta ID {folder_id}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Baixa arquivos/pastas do Google Drive.")
    parser.add_argument('--type', type=str, choices=['datasets', 'models'], required=True,
                        help="Tipo de conteúdo a baixar: 'datasets' ou 'models'.")
    # Opcional: permitir que o caminho de saída seja especificado via CLI,
    # caso contrário, usa o do gdrive_config.json
    parser.add_argument('--output_path', type=str,
                        help="Caminho local para onde baixar os arquivos. Sobrepõe o gdrive_config.json se fornecido.")

    args = parser.parse_args()

    # Carregar configuração
    if not os.path.exists(CONFIG_PATH):
        print(f"ERRO: Arquivo de configuração '{CONFIG_PATH}' não encontrado.")
        return
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)

    gdrive_url = config.get('google_drive_folder_url')
    if not gdrive_url or gdrive_url == "COLE_A_URL_DA_PASTA_RAIZ_DO_GOOGLE_DRIVE_AQUI":
        print(f"ERRO: 'google_drive_folder_url' não está configurada em '{CONFIG_PATH}'.")
        return

    # Determinar o caminho de destino local
    local_destination_path = args.output_path # Prioriza o argumento da CLI
    if not local_destination_path:
        if args.type == 'datasets':
            local_destination_path = config.get('local_datasets_path', './datasets_from_gdrive')
        elif args.type == 'models':
            local_destination_path = config.get('local_models_path', './models_from_gdrive')

    # Opcional: pode-se querer que o --type também defina uma subpasta na URL do GDrive
    # Ex: gdrive_url aponta para a raiz, e queremos baixar de gdrive_url/datasets ou gdrive_url/models
    # Por simplicidade, vamos assumir que a gdrive_url já aponta para a pasta correta (datasets OU models)
    # OU que dentro da gdrive_url existem subpastas chamadas "datasets" e "models".
    # Para este script, vamos assumir que a URL em gdrive_config.json é a URL da PASTA RAIZ GERAL,
    # e vamos procurar por subpastas chamadas "datasets" ou "models" dentro dela.

    print(f"Tentando baixar '{args.type}' do Google Drive para '{local_destination_path}'")

    service = get_gdrive_service()
    if not service:
        return

    root_folder_id = extract_folder_id_from_url(gdrive_url)
    if not root_folder_id:
        print(f"Não foi possível determinar o ID da pasta raiz do Google Drive a partir da URL: {gdrive_url}")
        return

    # Encontrar o ID da subpasta específica (datasets ou models) dentro da pasta raiz
    target_folder_name = args.type # "datasets" ou "models"
    target_folder_id = None

    try:
        query = f"name='{target_folder_name}' and '{root_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
        results = service.files().list(q=query, fields="files(id, name)", pageSize=1).execute()
        items = results.get('files', [])
        if items:
            target_folder_id = items[0]['id']
            print(f"Encontrada pasta '{target_folder_name}' com ID: {target_folder_id} dentro da pasta raiz.")
        else:
            print(f"ERRO: Subpasta '{target_folder_name}' não encontrada dentro da pasta do Google Drive especificada por: {gdrive_url}")
            print(f"Certifique-se de que a pasta '{target_folder_name}' existe dentro de '{gdrive_url}'.")
            return

    except Exception as e:
        print(f"Erro ao procurar pela subpasta '{target_folder_name}': {e}")
        return

    # Criar o diretório de destino base se não existir
    os.makedirs(local_destination_path, exist_ok=True)

    # Iniciar o download recursivo da pasta alvo (datasets ou models)
    list_and_download_folder_contents(service, target_folder_id, local_destination_path)

    print(f"\nProcesso de download para '{args.type}' concluído.")

if __name__ == '__main__':
    main()
