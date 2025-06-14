import os.path
import json
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

# Se modificar esses escopos, delete o arquivo token.json.
SCOPES = ['https://www.googleapis.com/auth/drive.readonly'] # Apenas leitura por segurança inicial
TOKEN_PATH = 'gdrive_token.json'
CREDENTIALS_PATH = 'gdrive_credentials.json' # O usuário precisará fornecer este arquivo

def authenticate_gdrive():
    """Autentica com a API do Google Drive usando OAuth 2.0.
    Salva as credenciais em token.json para usos futuros.
    Requer um arquivo credentials.json obtido do Google Cloud Console.
    """
    creds = None
    # O arquivo token.json armazena os tokens de acesso e atualização do usuário,
    # e é criado automaticamente na primeira vez que o fluxo de autorização é completado.
    if os.path.exists(TOKEN_PATH):
        try:
            creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)
        except json.JSONDecodeError:
            print(f"Erro ao decodificar {TOKEN_PATH}. O arquivo pode estar corrompido. Tentando re-autenticar.")
            creds = None # Força re-autenticação
        except ValueError as e:
            print(f"Erro ao carregar credenciais de {TOKEN_PATH}: {e}. Pode ser um problema de escopo. Tentando re-autenticar.")
            creds = None # Força re-autenticação


    # Se não houver credenciais (válidas), permite que o usuário faça login.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                print("Atualizando token de acesso expirado...")
                creds.refresh(Request())
            except Exception as e:
                print(f"Falha ao atualizar token: {e}")
                print(f"Por favor, re-autentique. Removendo {TOKEN_PATH} se existir.")
                if os.path.exists(TOKEN_PATH):
                    os.remove(TOKEN_PATH)
                creds = None # Garante que o fluxo de autenticação abaixo será executado

        if not creds: # creds ainda é None se a atualização falhou ou não era possível
            if not os.path.exists(CREDENTIALS_PATH):
                print(f"ERRO: O arquivo '{CREDENTIALS_PATH}' não foi encontrado.")
                print("Este arquivo é necessário para a autenticação OAuth 2.0.")
                print("Por favor, baixe-o do Google Cloud Console (Credenciais -> Criar Credenciais -> ID do Cliente OAuth)")
                print("e salve-o como 'gdrive_credentials.json' no mesmo diretório deste script.")
                return None

            try:
                flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_PATH, SCOPES)
                # O parâmetro port=0 faz com que o servidor local use uma porta aleatória disponível.
                # O redirect_uri precisa corresponder ao configurado no Google Cloud Console para "URIs de redirecionamento autorizados".
                # Para aplicações de desktop/linha de comando, "http://localhost" ou "urn:ietf:wg:oauth:2.0:oob" são comuns.
                # Se usar "http://localhost", certifique-se de que está adicionado no GCP Console.
                # Usar launch_browser=False e pedir para copiar/colar pode ser mais robusto em alguns ambientes.
                # No entanto, o padrão de InstalledAppFlow tenta abrir o navegador.
                print("\nIniciando fluxo de autenticação do Google Drive...")
                print("Uma janela do navegador deve abrir para você autorizar o acesso.")
                print("Se não abrir, copie o link do terminal e cole no seu navegador.")
                creds = flow.run_local_server(port=0)
            except Exception as e:
                print(f"Falha ao executar o fluxo de autenticação local: {e}")
                return None

        # Salva as credenciais para a próxima execução
        if creds:
            try:
                with open(TOKEN_PATH, 'w') as token_file:
                    token_file.write(creds.to_json())
                print(f"Credenciais salvas em '{TOKEN_PATH}'")
            except Exception as e:
                print(f"Erro ao salvar o token em {TOKEN_PATH}: {e}")
        else:
            print("Não foi possível obter credenciais.")
            return None

    if creds and creds.valid:
        print("Autenticação com Google Drive bem-sucedida.")
    elif creds and creds.expired:
        print("Token expirado e não pôde ser atualizado. Por favor, execute novamente para re-autenticar.")
        return None

    return creds

if __name__ == '__main__':
    print("Este script ajuda a autenticar com o Google Drive.")
    print(f"Ele espera um arquivo '{CREDENTIALS_PATH}' no mesmo diretório.")
    print(f"Após a autenticação bem-sucedida, um arquivo '{TOKEN_PATH}' será criado.")
    print("--------------------------------------------------------------------------")
    authenticate_gdrive()
