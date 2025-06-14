import base64
import os
import random
from flask import Flask, request, send_file
from PIL import Image
import numpy as np
import math
import io
from datetime import datetime
from recognizer import Recognizer
from uuid import uuid4
import urllib.parse
import skimage.draw
import json
from google.oauth2 import service_account
from googleapiclient.discovery import build as build_gdrive_service
from googleapiclient.http import MediaIoBaseUpload, MediaFileUpload

app = Flask(__name__)

# --- Configuração da Aplicação e Google Drive ---
APP_SETTINGS_PATH = 'app_settings.json'
if not os.path.exists(APP_SETTINGS_PATH):
    raise FileNotFoundError(f"Arquivo de configuração '{APP_SETTINGS_PATH}' não encontrado.")
with open(APP_SETTINGS_PATH, 'r') as f:
    app_settings = json.load(f)

GDRIVE_SCOPES = ['https://www.googleapis.com/auth/drive']
GDRIVE_UPLOAD_FOLDER_ID = app_settings.get('google_drive_upload_folder_id')
GDRIVE_SERVICE_ACCOUNT_KEY_PATH = app_settings.get('service_account_key_path')
ALLOW_RAW_UPLOAD_TO_GDRIVE = app_settings.get('allow_raw_upload_to_gdrive', False)
GENERATED_FILES_PARENT_FOLDER_NAME = app_settings.get('generated_files_parent_folder_name', 'JuizDeBochaUploadsApp')

def get_gdrive_service_for_app():
    try:
        creds = service_account.Credentials.from_service_account_file(
            GDRIVE_SERVICE_ACCOUNT_KEY_PATH, scopes=GDRIVE_SCOPES)
        service = build_gdrive_service('drive', 'v3', credentials=creds, static_discovery=False)
        print('Serviço Google Drive inicializado com sucesso para a aplicação.')
        return service
    except Exception as e:
        print(f'Falha ao inicializar serviço Google Drive para a aplicação: {e}')
        return None

_gdrive_service = get_gdrive_service_for_app()
_generated_files_gdrive_folder_id = None # Será resolvido/criado em _upload_image_to_gdrive
# --- Fim da Configuração ---

_rec = Recognizer(
    config_file='models/model/config.yaml',
    weights_file='models/model/weights.pkl',
    confidence_threshold=0.82,
)
__thumbnail_size = tuple(app_settings.get('default_thumbnail_size', [300, 300]))


def _two_centers_distance(ca, cb):
    return math.sqrt(
        math.pow(cb['x'] - ca['x'], 2) +
        math.pow(cb['y'] - ca['y'], 2)
    )


def gaussian_kernel(l=5, sig=1.):
    """\
    creates gaussian kernel with side length `l` and a sigma of `sig`
    """
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)


def __read_image(img_bytes):
    return _rec.read_file(io.BytesIO(img_bytes))


def __process_balls(img):
    instances = _rec.predict_data(img)
    instances = list(filter(lambda i: i['class'] == 'sports ball', instances))
    balls, smallest, winner = [], None, None
    if len(instances) >= 2:
        in_walls = []
        in_middle = []
        margin = 0.10
        x_min = img.shape[1] * margin
        x_max = img.shape[1] * (1-margin)
        y_min = img.shape[0] * margin
        y_max = img.shape[0] * (1-margin)
        for instance in instances:
            in_horizontal_middle = (x_min <= instance['center']['x'] <= x_max)
            in_vertical_middle = (y_min <= instance['center']['y'] <= y_max)
            if in_horizontal_middle and in_vertical_middle:
                in_middle.append(instance)
            else:
                in_walls.append(instance)
        if len(in_middle) > 0:
            in_middle.sort(key=lambda i: i['area'])
            while True:
                smallest = in_middle[0]
                del in_middle[0]
                if len(in_middle) >= 1:
                    largest_area = in_middle[-1]['area']
                    next_area = in_middle[0]['area']
                    ratio_to_larger = largest_area / smallest['area']
                    ratio_to_next = next_area / smallest['area']
                    if math.fabs(ratio_to_larger - ratio_to_next) > 1.5: # Heuristic for dirt
                        del smallest # Dirt, try next smallest
                    else:
                        break # Viable balim
                else:
                    break # Too few balls
            for i in reversed(range(len(in_walls))):
                ball: dict = in_walls[i]
                if smallest and ball['area'] < smallest['area']/2: # Heuristic for dirt on walls
                    del in_walls[i]
            balls = in_middle + in_walls
            if smallest and len(balls) >= 1:
                for i in range(len(balls)):
                    balls[i]['distance'] = _two_centers_distance(smallest['center'], balls[i]['center'])
                balls.sort(key=lambda b: b['distance'])
                winner = balls[0]
    return balls, winner, smallest


def __draw_circle(clean_img, img, instance, color, margin=10, stroke=None):
    if stroke is None:
        stroke = max(img.shape[0], img.shape[1]) * 0.005
    c = int(instance['center']['x'])
    r = int(instance['center']['y'])
    c_radius = (instance['box']['x2'] - instance['box']['x1'])/2
    r_radius = (instance['box']['y2'] - instance['box']['y1'])/2
    rr, cc = skimage.draw.ellipse(
        r, c,
        int(r_radius + margin + stroke),
        int(c_radius + margin + stroke),
        shape=img.shape,
    )
    img[rr, cc] = color
    rr, cc = skimage.draw.ellipse(
        r, c,
        int(r_radius + margin),
        int(c_radius + margin),
        shape=img.shape
    )
    img[rr, cc] = clean_img[rr, cc]


def _process_image(img_bytes, create_thumbnail=False):
    img = __read_image(img_bytes)
    balls, winner, smallest = __process_balls(img)
    img_winner = img.copy()
    for i, ball in reversed(list(enumerate(balls))):
        colors = img[ball.get('mask')]
        avg_color = np.array(colors.mean(axis=0), dtype=np.uint8)
        img[ball.get('mask')] = avg_color
        if i != 0: # Don't recolor winner ball in img_winner yet
            img_winner[ball.get('mask')] = avg_color

    green = np.array([50, 230, 50], dtype=np.uint8)
    if winner:
        __draw_circle(img, img_winner, instance=winner, color=green)
    if smallest:
        margin_s = 0
        stroke_s = max(img.shape[0], img.shape[1]) * 0.0022
        # Crosshair for smallest
        rr, cc = skimage.draw.rectangle(start=(max(int(smallest['box']['y1']-margin_s), 0), max(int(smallest['center']['x']-stroke_s/2), 0)), end=(min(int(smallest['box']['y2']+margin_s), img.shape[0]-1), min(int(smallest['center']['x']+stroke_s/2), img.shape[1]-1)))
        img[rr, cc] = green; img_winner[rr, cc] = green
        rr, cc = skimage.draw.rectangle(start=(max(int(smallest['center']['y']-stroke_s/2.), 0), max(int(smallest['box']['x1']-margin_s), 0)), end=(min(int(smallest['center']['y']+stroke_s/2.), img.shape[0]-1), min(int(smallest['box']['x2']+margin_s), img.shape[1]-1)))
        img[rr, cc] = green; img_winner[rr, cc] = green
        # Circle for smallest
        __draw_circle(img.copy(), img, instance=smallest, color=green, margin=margin_s, stroke=stroke_s)
        __draw_circle(img_winner.copy(), img_winner, instance=smallest, color=green, margin=margin_s, stroke=stroke_s)


    if create_thumbnail:
        axis = int(np.argmin(img_winner.shape[:2]))
        factor = img_winner.shape[axis] / __thumbnail_size[axis]
        thumbnail_pil = Image.fromarray(img_winner)
        thumbnail_pil = thumbnail_pil.resize(
            size=(int(img_winner.shape[1] / factor), int(img_winner.shape[0] / factor))
        )
        thumbnail = np.array(thumbnail_pil)
        cross_axis = (axis+1) % 2
        center = int(thumbnail.shape[cross_axis] / 2)
        if smallest is not None:
            center_coord = smallest['center'][('y', 'x')[cross_axis]]
            center = int(center_coord / factor)

        size_on_cross_axis = __thumbnail_size[cross_axis]
        start = center - int(size_on_cross_axis / 2)
        if start < 0: start = 0
        end = start + size_on_cross_axis
        if end > thumbnail.shape[cross_axis]:
            end = thumbnail.shape[cross_axis]
            start = end - size_on_cross_axis
            if start < 0: start = 0 # Adjust if image is smaller than thumbnail dim

        if cross_axis == 0: thumbnail = thumbnail[start:end, :]
        else: thumbnail = thumbnail[:, start:end]
        return thumbnail, (img, img_winner)
    return img, img_winner


__counter = random.randint(1, 10000)
def _create_filename():
    global __counter
    __counter += 1
    return f"{datetime.now().timestamp()}.{__counter}"


max_shape = (10000, 9000) # Max dimensions for GIF frames to avoid memory issues
def _to_gif_bytes(images):
    gif_bytes = io.BytesIO()
    base, winner = images
    frames = []
    p = 0.0
    new_size = None
    if base.shape[0] > base.shape[1] and base.shape[0] > max_shape[0]:
        factor = max_shape[0] / base.shape[0]
        new_size = (int(base.shape[1] * factor), max_shape[0])
    elif base.shape[1] > max_shape[1]:
        factor = max_shape[1] / base.shape[1]
        new_size = (max_shape[1], int(base.shape[0] * factor))

    for step in [0.25, -0.25]: # Simple animation loop
        for _ in range(int(math.fabs(1.0//step))):
            frame_np = np.array(base * (1-p) + winner * p, dtype=np.uint8)
            frame_pil = Image.fromarray(frame_np)
            if new_size is not None:
                frame_pil = frame_pil.resize(new_size)
            frames.append(frame_pil)
            p += step
    if frames:
        frames[0].save(gif_bytes, format='gif', append_images=frames[1:], save_all=True, duration=100, loop=0, optimize=True)
    return gif_bytes.getvalue()


def _to_bytes(im, format='jpeg'):
    img_pil = Image.fromarray(im)
    img_bytes = io.BytesIO()
    img_pil.save(img_bytes, format=format.upper())
    return img_bytes.getvalue()

def _get_or_create_app_upload_folder(service, parent_folder_id, folder_name):
    query = f"name='{folder_name}' and '{parent_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
    response = service.files().list(q=query, fields='files(id)').execute()
    if response.get('files'):
        return response.get('files')[0].get('id')
    else:
        file_metadata = {'name': folder_name, 'mimeType': 'application/vnd.google-apps.folder', 'parents': [parent_folder_id]}
        folder = service.files().create(body=file_metadata, fields='id').execute()
        print(f'Pasta Criada ID: {folder.get("id")}, Nome: {folder_name}')
        return folder.get('id')

def _upload_image_to_gdrive(filename_on_drive, file_bytes, mime_type, user_id=None):
    global _generated_files_gdrive_folder_id
    if not _gdrive_service:
        print('Serviço Google Drive não inicializado. Upload cancelado.')
        return None
    if not GDRIVE_UPLOAD_FOLDER_ID or GDRIVE_UPLOAD_FOLDER_ID == 'COLE_O_ID_DA_PASTA_DO_GOOGLE_DRIVE_AQUI':
        print('ID da pasta de upload do Google Drive não configurado. Upload cancelado.')
        return None

    if not _generated_files_gdrive_folder_id:
        try:
            _generated_files_gdrive_folder_id = _get_or_create_app_upload_folder(_gdrive_service, GDRIVE_UPLOAD_FOLDER_ID, GENERATED_FILES_PARENT_FOLDER_NAME)
            if not _generated_files_gdrive_folder_id:
                print('Não foi possível criar ou encontrar a pasta de uploads da aplicação no Drive.')
                return None
        except Exception as e_folder_create:
            print(f'Erro ao criar/encontrar pasta da app no Drive: {e_folder_create}')
            return None

    current_parent_folder_id = _generated_files_gdrive_folder_id
    if user_id:
        user_folder_name = f"user_{user_id}"
        try:
            user_gdrive_folder_id = _get_or_create_app_upload_folder(_gdrive_service, _generated_files_gdrive_folder_id, user_folder_name)
            if user_gdrive_folder_id: current_parent_folder_id = user_gdrive_folder_id
        except Exception as e_user_folder:
            print(f'Erro ao criar/encontrar pasta do usuário no Drive: {e_user_folder}, usando pasta da app.')

    file_metadata = {'name': filename_on_drive, 'parents': [current_parent_folder_id]}
    media_body = MediaIoBaseUpload(io.BytesIO(file_bytes), mimetype=mime_type, resumable=True)
    try:
        file = _gdrive_service.files().create(body=file_metadata, media_body=media_body, fields='id, webViewLink').execute()
        print(f"Arquivo '{filename_on_drive}' enviado para Google Drive. ID: {file.get('id')}")
        return file.get('webViewLink')
    except Exception as e:
        print(f"Erro durante o upload para Google Drive: {e}")
        return None

@app.route('/image', methods=['POST'])
def process_image_return_image():
    with_thumbnail = request.args.get('with-thumbnail') is not None
    img_bytes = request.stream.read()
    filename_base = _create_filename()

    user_id_param = request.args.get('userID')

    if ALLOW_RAW_UPLOAD_TO_GDRIVE:
        _upload_image_to_gdrive(filename_base + '.jpg', img_bytes, 'image/jpeg', user_id=user_id_param)

    result = _process_image(img_bytes, create_thumbnail=with_thumbnail)
    thumbnail_bytes = None
    if with_thumbnail:
        thumbnail_np, result_images = result
        thumbnail_bytes = _to_bytes(thumbnail_np)
    else:
        result_images = result

    gif_bytes = _to_gif_bytes(result_images)

    if with_thumbnail and thumbnail_bytes:
        return dict(
            thumbnail=base64.encodebytes(thumbnail_bytes).decode('utf-8'),
            gif=base64.encodebytes(gif_bytes).decode('utf-8'),
        )
    else:
        return send_file(
            io.BytesIO(gif_bytes),
            download_name=filename_base + '.gif',
            mimetype='image/gif',
            as_attachment=True,
        )


@app.route('/url', methods=['POST'])
def process_image_return_url():
    with_thumbnail = request.args.get('with-thumbnail') is not None
    img_bytes = request.stream.read()
    filename_base = _create_filename()
    user_id_param = request.args.get('userID')

    if ALLOW_RAW_UPLOAD_TO_GDRIVE:
         _upload_image_to_gdrive(filename_base + '.jpg', img_bytes, 'image/jpeg', user_id=user_id_param)

    result = _process_image(img_bytes, create_thumbnail=with_thumbnail)
    thumbnail_bytes = None
    if with_thumbnail:
        thumbnail_np, result_images = result
        thumbnail_bytes = _to_bytes(thumbnail_np)
    else:
        result_images = result

    animated_url = _upload_image_to_gdrive(filename_base + '.gif', _to_gif_bytes(result_images), 'image/gif', user_id=user_id_param)

    if with_thumbnail and thumbnail_bytes:
        thumbnail_url = _upload_image_to_gdrive(filename_base + '.thumbnail.jpg', thumbnail_bytes, 'image/jpeg', user_id=user_id_param)
        return dict(
            thumbnail=thumbnail_url or "upload_failed",
            animated=animated_url or "upload_failed",
        )
    else:
        return animated_url or "Upload do GIF falhou."


@app.route('/coordinates', methods=['POST'])
def process_image_return_coordinates():
    img_bytes = request.stream.read()
    filename_base = _create_filename()
    user_id_param = request.args.get('userID')

    # Upload da imagem original para referência, se desejado
    original_image_url = _upload_image_to_gdrive(filename_base + '_original.jpg', img_bytes, 'image/jpeg', user_id=user_id_param)

    img_np = __read_image(img_bytes) # Re-ler para obter dimensões corretas
    balls, winner, smallest = __process_balls(img_np)

    response_balls = []
    if smallest: # Inclui o bolim como a primeira "bola" para consistência com a lógica anterior
        response_balls.append(smallest)
    if winner and (not smallest or winner['id'] != smallest['id']): # Adiciona o vencedor se não for o bolim
         response_balls.extend([b for b in balls if b['id'] == winner['id']]) # Pega o vencedor da lista ordenada
    # Adiciona outras bolas, evitando duplicatas do bolim ou vencedor já adicionados
    other_balls_ids = {b['id'] for b in response_balls}
    response_balls.extend([b for b in balls if b['id'] not in other_balls_ids])


    processed_ball_data = []
    for ball_data in response_balls:
        if not ball_data: continue # Caso smallest seja None
        processed_ball_data.append({
            'center': {
                'x': ball_data['center']['x'] / img_np.shape[1],
                'y': ball_data['center']['y'] / img_np.shape[0],
            },
            'ellipse': {
                'width': (ball_data['box']['x2'] - ball_data['box']['x1']) / img_np.shape[1],
                'height': (ball_data['box']['y2'] - ball_data['box']['y1']) / img_np.shape[0],
            },
            # Adicionar 'id' ou 'class' se necessário para a app Flutter distinguir
            # 'is_balim': ball_data['id'] == smallest['id'] if smallest else False,
            # 'is_winner': ball_data['id'] == winner['id'] if winner else False,
        })

    return {
        'url': original_image_url or "upload_original_failed",
        'smallest': processed_ball_data[0] if len(processed_ball_data) > 0 and smallest else None, # Bolim é o primeiro, se existe
        'winner': next((b for i,b in enumerate(processed_ball_data) if smallest and winner and b['center']['x'] * img_np.shape[1] == winner['center']['x'] and i!=0), None) if smallest and winner else (processed_ball_data[0] if not smallest and winner else None), # Tenta encontrar o vencedor que não seja o bolim
        'balls': processed_ball_data # Lista completa, Flutter pode filtrar/interpretar
    }


if __name__ == "__main__":
    # Para debug local, sobrescrever a configuração de upload para GDrive se necessário
    # ALLOW_RAW_UPLOAD_TO_GDRIVE = False
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
