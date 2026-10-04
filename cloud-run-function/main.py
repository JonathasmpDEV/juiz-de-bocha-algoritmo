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
try:
    with open(APP_SETTINGS_PATH, 'r') as f:
        app_settings = json.load(f)
except FileNotFoundError:
    print(f'AVISO: {APP_SETTINGS_PATH} não encontrado, usando defaults.')
    app_settings = {}

# Configurações de Armazenamento de Saída
OUTPUT_STORAGE_TYPE = app_settings.get('output_storage_type', 'local').lower()
LOCAL_STORAGE_GENERATED_FILES = app_settings.get('local_storage_path_generated_files', '/app/output/generated_files')
LOCAL_STORAGE_RAW_FILES = app_settings.get('local_storage_path_raw_files', '/app/output/raw_files')

# Configurações do Google Drive (usadas se OUTPUT_STORAGE_TYPE == 'gdrive')
GDRIVE_SCOPES = ['https://www.googleapis.com/auth/drive']
GDRIVE_UPLOAD_FOLDER_ID = app_settings.get('google_drive_upload_folder_id', 'COLE_O_ID_DA_PASTA_DO_GOOGLE_DRIVE_AQUI')
GDRIVE_SERVICE_ACCOUNT_KEY_PATH = app_settings.get('service_account_key_path', 'gdrive_service_account.json')
# ALLOW_RAW_UPLOAD_TO_GDRIVE é para GDrive; para local, assumimos que se salva se a função for chamada.
ALLOW_GDRIVE_RAW_UPLOAD = app_settings.get('allow_raw_upload_to_gdrive', False)
GENERATED_FILES_PARENT_FOLDER_NAME = app_settings.get('generated_files_parent_folder_name', 'JuizDeBochaUploadsApp')

# Garantir que as pastas locais de saída existam no container se o modo for local
if OUTPUT_STORAGE_TYPE == 'local':
    os.makedirs(LOCAL_STORAGE_GENERATED_FILES, exist_ok=True)
    os.makedirs(LOCAL_STORAGE_RAW_FILES, exist_ok=True)
    print(f'Armazenamento local configurado em: {LOCAL_STORAGE_GENERATED_FILES} e {LOCAL_STORAGE_RAW_FILES}')
elif OUTPUT_STORAGE_TYPE == 'gdrive':
    print(f'Armazenamento Google Drive configurado para pasta ID: {GDRIVE_UPLOAD_FOLDER_ID}')
else:
    print(f'AVISO: output_storage_type "{OUTPUT_STORAGE_TYPE}" não reconhecido. Nenhum arquivo de saída será salvo permanentemente pela API.')

def get_gdrive_service_for_app():
    if not os.path.exists(GDRIVE_SERVICE_ACCOUNT_KEY_PATH):
        print(f"AVISO: Arquivo de chave de conta de serviço '{GDRIVE_SERVICE_ACCOUNT_KEY_PATH}' não encontrado. Uploads para Google Drive desabilitados.")
        return None
    try:
        creds = service_account.Credentials.from_service_account_file(
            GDRIVE_SERVICE_ACCOUNT_KEY_PATH, scopes=GDRIVE_SCOPES)
        service = build_gdrive_service('drive', 'v3', credentials=creds, static_discovery=False)
        print('Serviço Google Drive inicializado com sucesso para a aplicação.')
        return service
    except Exception as e:
        print(f'Falha ao inicializar serviço Google Drive para a aplicação: {e}')
        return None

_gdrive_service = None
if OUTPUT_STORAGE_TYPE == 'gdrive':
    _gdrive_service = get_gdrive_service_for_app()

_generated_files_gdrive_folder_id = None # Cache para o ID da pasta principal de uploads da app no GDrive
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
    # print("n instances:", len(instances))
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
            # get balim
            while True:
                smallest = in_middle[0]
                del in_middle[0]
                if len(in_middle) >= 1:
                    # check if its too small, then, is probably dirt
                    largest_area = in_middle[-1]['area']
                    next_area = in_middle[0]['area']
                    ratio_to_larger = largest_area / smallest['area']
                    ratio_to_next = next_area / smallest['area']
                    # print("rate difference:")
                    # print(ratio_to_larger)
                    # print(ratio_to_next)
                    # print(ratio_to_larger - ratio_to_next)
                    # print(math.fabs(ratio_to_larger - ratio_to_next))
                    if math.fabs(ratio_to_larger - ratio_to_next) > 1.5:
                        del smallest
                    else:
                        # is viable
                        break
                else:
                    # too few balls to analyse
                    break
            # remove walls dirt (smaller than balim)
            for i in reversed(range(len(in_walls))):
                ball: dict = in_walls[i]
                if ball['area'] < smallest['area']/2:
                    del in_walls[i]
            balls = in_middle + in_walls
            if len(balls) >= 1:
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
    # outer
    rr, cc = skimage.draw.ellipse(
        r, c,
        int(r_radius + margin + stroke),
        int(c_radius + margin + stroke),
        shape=img.shape,
    )
    img[rr, cc] = color
    # inner
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
    # grid
    # if smallest:
    #     width = int(smallest['box']['x2'] - smallest['box']['x1']) +1
    #     height = int(smallest['box']['y2'] - smallest['box']['y1']) +1
    #     grid_color = np.array([80, 80, 80], dtype=np.uint8)
    #     for direction in [-1, 1]:
    #         x = int(smallest['box']['x1'])
    #         while 0 < x < img.shape[1]:
    #             img[:, x-1:x] = grid_color
    #             x += direction * width
    #         y = int(smallest['box']['y1'])
    #         while 0 < y < img.shape[0]:
    #             img[y-1:y, :] = grid_color
    #             y += direction * height
    #         # y = int(smallest['box']['y1'])
    #     # i, j = 0, 0
    img_winner = img.copy()
    for i, ball in reversed(list(enumerate(balls))):
        colors = img[ball.get('mask')]
        avg_color = np.array(colors.mean(axis=0), dtype=np.uint8)
        img[ball.get('mask')] = avg_color
        if i != 0:
            img_winner[ball.get('mask')] = avg_color
    green = np.array([50, 230, 50], dtype=np.uint8)
    if winner:
        __draw_circle(
            img,
            img_winner,
            instance=winner,
            color=green,
        )
        # c = int(winner['center']['x'])
        # r = int(winner['center']['y'])
        # c_radius = int(winner['box']['x2']-winner['box']['x1'])
        # r_radius = int(winner['box']['y2']-winner['box']['y1'])
        # shape = img.shape
        # # outer
        # margin = 8
        # rr, cc = skimage.draw.ellipse(
        #     r, c,
        #     r_radius-margin,
        #     c_radius-margin,
        #     shape=shape
        # )
        # img_winner[rr, cc] = np.array([50, 230, 50], dtype=np.uint8)
        # # inner
        # stroke = max(img.shape[0], img.shape[1]) * 0.007
        # rr, cc = skimage.draw.ellipse(
        #     r, c,
        #     r_radius-margin-stroke,
        #     c_radius-margin-stroke,
        #     shape=shape
        # )
        # img_winner[rr, cc] = img[rr, cc]

    # smaller
    if smallest:
        margin = 0
        stroke = max(img.shape[0], img.shape[1]) * 0.0022
        # vertical line
        rr, cc = skimage.draw.rectangle(
            start=(
                max(int(smallest['box']['y1']-margin), 0),
                max(int(smallest['center']['x']-stroke/2), 0),
            ),
            end=(
                min(int(smallest['box']['y2']+margin), img.shape[0]-1),
                min(int(smallest['center']['x']+stroke/2), img.shape[1]-1),
            ),
        )
        img[rr, cc] = green
        img_winner[rr, cc] = green
        # horizontal line
        rr, cc = skimage.draw.rectangle(
            start=(
                max(int(smallest['center']['y']-stroke/2.), 0),
                max(int(smallest['box']['x1']-margin), 0),
            ),
            end=(
                min(int(smallest['center']['y']+stroke/2.), img.shape[0]-1),
                min(int(smallest['box']['x2']+margin), img.shape[1]-1),
            ),
        )
        img[rr, cc] = green
        img_winner[rr, cc] = green
        # circle
        __draw_circle(
            img,
            img_winner,
            instance=smallest,
            color=green,
            margin=margin,
            stroke=stroke,
        )
        __draw_circle(
            img.copy(),
            img,
            instance=smallest,
            color=green,
            margin=margin,
            stroke=stroke,
        )
        # img[smallest.get('mask')] = np.array([255, 255, 255], dtype=np.uint8)

    # circulo na cor média da bola e piscando em verde
    # for i in reversed(range(len(balls))):
    #     colors = img[balls[i].get('mask')]
    #     avg_color = colors.mean(axis=0)
    #     img[balls[i].get('mask')] = np.array(avg_color, dtype=np.uint8)
    # if winner:
    #     img[winner.get('mask')] = np.array([50, 230, 50], dtype=np.uint8)
        # border = winner.get('mask')
        # dilated = ndimage.binary_dilation(border, gaussian_kernel(10))
        # eroded = ndimage.binary_erosion(border, gaussian_kernel(5))
        # border = dilated ^ eroded
        # border = ndimage.binary_dilation(border, gaussian_kernel(5, 0.5))
        # img[border] = np.array([100, 220, 100], dtype=np.uint8)
    if create_thumbnail:
        # resize/reduce based on smaller axis factor of original image
        axis = int(np.argmin(img_winner.shape[:2]))
        factor = img_winner.shape[axis] / __thumbnail_size[axis]
        thumbnail = Image.fromarray(img_winner)
        thumbnail = thumbnail.resize(
            size=(
                int(img_winner.shape[1] / factor),  # width
                int(img_winner.shape[0] / factor),  # height
            )
        )
        thumbnail = np.array(thumbnail)
        # cut larger axis to fit target size, with smaler ball in the center
        cross_axis = (axis+1) % 2
        center = int(thumbnail.shape[cross_axis] / 2)
        if smallest is not None:
            center = int(smallest['center'][('y', 'x')[cross_axis]])
        center = int(center / factor)
        size = __thumbnail_size[cross_axis]
        start = center - int(size / 2)
        # guarantee that is after image start
        if start < 0:
            start = 0
            end = size
        else:
            end = start + size
            # guarantee that is before image end
            if end > thumbnail.shape[cross_axis]:
                end = thumbnail.shape[cross_axis]
                start = end - size
        if cross_axis == 0:
            thumbnail = thumbnail[start:end, :]
        else:
            thumbnail = thumbnail[:, start:end]
        return thumbnail, (img, img_winner)
    return img, img_winner


__counter = random.randint(1, 10000)


def _create_filename():
    global __counter
    __counter += 1
    return f"{datetime.now().timestamp()}.{__counter}"


max_shape = (10000, 9000)


def _to_gif_bytes(images):
    gif_bytes = io.BytesIO()
    base, winner = images
    frames = []
    p = 0.0
    new_size = None
    if base.shape[0] > base.shape[1] and base.shape[0] > max_shape[0]:  # portrait
        factor = max_shape[0] / base.shape[0]
        new_size = (
            int(base.shape[1] * factor),  # x
            max_shape[0],  # y
        )
    elif base.shape[1] > max_shape[1]:  # landscape
        factor = max_shape[1] / base.shape[1]
        new_size = (
            max_shape[1],  # x
            int(base.shape[0] * factor),  # y
        )
    for step in [0.25, -0.25]:
        for i in range(int(math.fabs(1.0//step))):
            frame = np.array(base * (1-p) + winner * p, dtype=np.uint8)
            frame = Image.fromarray(frame)
            if new_size is not None:
                frame = frame.resize(new_size)
            frames.append(frame)
            p += step
    frames[0].save(
        gif_bytes,
        format='gif',
        append_images=frames[1:],
        save_all=True,
        duration=100,
        optimize=True,
        loop=0,
        minimize_size=True,
        allow_mixed=True,
    )
    return gif_bytes.getvalue()


def _to_bytes(im, format='jpeg'):
    img = Image.fromarray(im)
    img_bytes = io.BytesIO()
    img.save(img_bytes, format=format)
    return img_bytes.getvalue()


def _save_file_locally(filename_on_disk, file_bytes, storage_folder_path):
    try:
        full_path = os.path.join(storage_folder_path, filename_on_disk)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, 'wb') as f: f.write(file_bytes)
        print(f"Arquivo '{filename_on_disk}' salvo localmente em '{full_path}'")
        # Retornar um caminho relativo que possa ser usado para construir uma URL se servido localmente
        return os.path.join("/output", os.path.basename(storage_folder_path), filename_on_disk).replace(os.sep, '/')
    except Exception as e:
        print(f"Erro ao salvar arquivo localmente '{filename_on_disk}': {e}")
        return None

def _get_or_create_gdrive_folder(service, parent_folder_id, folder_name):
    query = f"name='{folder_name}' and '{parent_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
    response = service.files().list(q=query, fields='files(id)', pageSize=1).execute()
    if response.get('files'): return response.get('files')[0].get('id')
    file_metadata = {'name': folder_name, 'mimeType': 'application/vnd.google-apps.folder', 'parents': [parent_folder_id]}
    folder = service.files().create(body=file_metadata, fields='id').execute()
    print(f'Pasta Google Drive Criada: ID {folder.get("id")}, Nome: {folder_name}')
    return folder.get('id')

def _upload_image_to_gdrive(filename_on_drive, file_bytes, mime_type, user_id=None):
    global _generated_files_gdrive_folder_id
    if not _gdrive_service: print('Serviço GDrive não inicializado. Upload GDrive cancelado.'); return None
    if not GDRIVE_UPLOAD_FOLDER_ID or GDRIVE_UPLOAD_FOLDER_ID == 'COLE_O_ID_DA_PASTA_DO_GOOGLE_DRIVE_AQUI':
        print('ID da pasta de upload GDrive não configurado. Upload GDrive cancelado.'); return None
    try:
        if not _generated_files_gdrive_folder_id:
            _generated_files_gdrive_folder_id = _get_or_create_gdrive_folder(_gdrive_service, GDRIVE_UPLOAD_FOLDER_ID, GENERATED_FILES_PARENT_FOLDER_NAME)
            if not _generated_files_gdrive_folder_id: print('Falha ao resolver pasta principal da app no GDrive.'); return None

        current_parent_id = _generated_files_gdrive_folder_id
        if user_id:
            user_folder_id = _get_or_create_gdrive_folder(_gdrive_service, _generated_files_gdrive_folder_id, f"user_{user_id}")
            if user_folder_id: current_parent_id = user_folder_id
            else: print(f"Falha ao resolver pasta do usuário {user_id} no GDrive, usando pasta principal da app.")

        file_metadata = {'name': filename_on_drive, 'parents': [current_parent_id]}
        media_body = MediaIoBaseUpload(io.BytesIO(file_bytes), mimetype=mime_type, resumable=True)
        file = _gdrive_service.files().create(body=file_metadata, media_body=media_body, fields='id, webViewLink').execute()
        print(f"Arquivo '{filename_on_drive}' enviado para GDrive. ID: {file.get('id')}")
        return file.get('webViewLink')
    except Exception as e: print(f"Erro durante upload para GDrive: {e}"); return None

def _handle_file_output(base_filename, file_bytes, mime_type, output_location_type, user_id=None):
    """ output_location_type: 'raw' ou 'generated' """
    filename_with_user = base_filename
    if user_id:
        filename_with_user = os.path.join(f'user_{user_id}', base_filename)

    if OUTPUT_STORAGE_TYPE == 'gdrive':
        # Para GDrive, user_id é usado para subpastas dentro da lógica de _upload_image_to_gdrive
        return _upload_image_to_gdrive(base_filename, file_bytes, mime_type, user_id=user_id)
    elif OUTPUT_STORAGE_TYPE == 'local':
        storage_path = LOCAL_STORAGE_RAW_FILES if output_location_type == 'raw' else LOCAL_STORAGE_GENERATED_FILES
        return _save_file_locally(filename_with_user, file_bytes, storage_path)
    else:
        print(f"Tipo de armazenamento de saída '{OUTPUT_STORAGE_TYPE}' não reconhecido.")
        return None
# --- Fim das Funções de Armazenamento ---

@app.route('/image', methods=['POST'])
def process_image_return_image():
    with_thumbnail = request.args.get('with-thumbnail') is not None
    img_bytes = request.stream.read()
    filename_base = _create_filename()
    user_id_param = request.args.get('userID')

    if (OUTPUT_STORAGE_TYPE == 'gdrive' and ALLOW_GDRIVE_RAW_UPLOAD) or \
       (OUTPUT_STORAGE_TYPE == 'local'): # Para local, sempre salvar raw se a função for chamada
        _handle_file_output(filename_base + '.jpg', img_bytes, 'image/jpeg', 'raw', user_id=user_id_param)

    result = _process_image(img_bytes, create_thumbnail=with_thumbnail)
    thumbnail_bytes = None
    if with_thumbnail: thumbnail_np, result_images = result; thumbnail_bytes = _to_bytes(thumbnail_np)
    else: result_images = result
    gif_bytes = _to_gif_bytes(result_images)

    if with_thumbnail and thumbnail_bytes:
        return dict(thumbnail=base64.encodebytes(thumbnail_bytes).decode('utf-8'), gif=base64.encodebytes(gif_bytes).decode('utf-8'))
    else:
        return send_file(io.BytesIO(gif_bytes), download_name='response.gif', mimetype='image/gif', as_attachment=True)

@app.route('/url', methods=['POST'])
def process_image_return_url():
    with_thumbnail = request.args.get('with-thumbnail') is not None
    img_bytes = request.stream.read()
    filename_base = _create_filename()
    user_id_param = request.args.get('userID')

    if (OUTPUT_STORAGE_TYPE == 'gdrive' and ALLOW_GDRIVE_RAW_UPLOAD) or \
       (OUTPUT_STORAGE_TYPE == 'local'):
        _handle_file_output(filename_base + '.jpg', img_bytes, 'image/jpeg', 'raw', user_id=user_id_param)

    result = _process_image(img_bytes, create_thumbnail=with_thumbnail)
    thumbnail_bytes = None
    if with_thumbnail: thumbnail_np, result_images = result; thumbnail_bytes = _to_bytes(thumbnail_np)
    else: result_images = result

    animated_file_ref = _handle_file_output(filename_base + '.gif', _to_gif_bytes(result_images), 'image/gif', 'generated', user_id=user_id_param)

    if with_thumbnail and thumbnail_bytes:
        thumbnail_file_ref = _handle_file_output(filename_base + '.thumbnail.jpg', thumbnail_bytes, 'image/jpeg', 'generated', user_id=user_id_param)
        return dict(thumbnail=thumbnail_file_ref or "upload_failed", animated=animated_file_ref or "upload_failed")
    else:
        return animated_file_ref or "Upload do GIF falhou."

@app.route('/coordinates', methods=['POST'])
def process_image_return_coordinates():
    img_bytes = request.stream.read()
    filename_base = _create_filename()
    user_id_param = request.args.get('userID')

    if (OUTPUT_STORAGE_TYPE == 'gdrive' and ALLOW_GDRIVE_RAW_UPLOAD) or \
       (OUTPUT_STORAGE_TYPE == 'local'):
        _handle_file_output(filename_base + '.jpg', img_bytes, 'image/jpeg', 'raw', user_id=user_id_param)

    img = __read_image(img_bytes)
    balls, winner, smallest = __process_balls(img)
    # 'url' mantém o nome do contrato original; no armazenamento Drive/local é a referência do arquivo salvo
    url = _handle_file_output(filename_base + '.jpg', img_bytes, 'image/jpeg', 'generated', user_id=user_id_param)
    balls = [smallest, *balls]
    for i, ball in enumerate(balls):
        balls[i] = {
            'center': {
                'x': ball['center']['x'] / img.shape[1],
                'y': ball['center']['y'] / img.shape[0],
            },
            'ellipse': {
                'width': (ball['box']['x2'] - ball['box']['x1']) / img.shape[1],
                'height': (ball['box']['y2'] - ball['box']['y1']) / img.shape[0],
            },
        }
    return {
        'url': url,
        'smallest': balls[0],
        'winner': balls[1],
        'balls': balls[2:],
    }

if __name__ == "__main__":
    # Para debug local, OUTPUT_STORAGE_TYPE será controlado pelo app_settings.json
    # Ex: se app_settings.json tiver "output_storage_type": "local"
    # os arquivos serão salvos localmente nos volumes Docker.
    print(f"Rodando com OUTPUT_STORAGE_TYPE: {OUTPUT_STORAGE_TYPE}")
    if OUTPUT_STORAGE_TYPE == 'gdrive' and not _gdrive_service:
        print("AVISO IMPORTANTE: OUTPUT_STORAGE_TYPE é 'gdrive', mas o serviço Google Drive não pôde ser inicializado. Verifique as configurações e o arquivo de chave da conta de serviço.")

    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
