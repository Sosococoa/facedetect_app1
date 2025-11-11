import cv2
import os
import numpy as np
import base64
import PySimpleGUI as sg
import sys
import shutil

sg.theme('LightBlue3')

# === モデル読み込み ===
embedder = cv2.dnn.readNetFromTorch(
    "/Users/moriwakisou/Desktop/FaceDetect/models/openface_nn4.small2.v1.t7"
)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# === 学習データフォルダ ===
study_dir = "study"
os.makedirs(study_dir, exist_ok=True)

# === あなたの顔データ ===
my_embeddings = []
my_face_vector = None


# === 絵文字の読み込み ===
emoji = cv2.imread("img/b.png", cv2.IMREAD_UNCHANGED)

# === 顔特徴量の学習関数 ===
def study_images_from_folder(folder_path):
    global my_embeddings, my_face_vector

    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not image_files:
        raise ValueError("学習フォルダに画像がありません。")

    my_embeddings.clear()

    for filename in image_files:
        path = os.path.join(folder_path, filename)
        image = cv2.imread(path)
        if image is None:
            print(f"⚠️ 読み込めない画像をスキップ: {filename}")
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.05, 5)

        if len(faces) == 0:
            print(f"❌ 顔検出失敗: {filename}")
            continue

        for (x, y, w, h) in faces:
            face = image[y:y+h, x:x+w]
            blob = cv2.dnn.blobFromImage(
                cv2.resize(face, (96, 96)), 1.0/255, (96, 96),
                (0, 0, 0), swapRB=True, crop=False
            )
            embedder.setInput(blob)
            vec = embedder.forward()
            my_embeddings.append(vec.flatten())

    if len(my_embeddings) == 0:
        raise ValueError("どの画像からも顔が検出できませんでした。")

    my_face_vector = np.mean(my_embeddings, axis=0)
    print(f"✅ {len(my_embeddings)}枚の顔データを学習しました。")


# === OpenCV → Base64変換関数 ===
def cv2_to_base64(img_cv, size=(300, 300)):
    img_resized = cv2.resize(img_cv, size, interpolation=cv2.INTER_AREA)
    _, img_encoded = cv2.imencode(".png", img_resized)
    return base64.b64encode(img_encoded.tobytes())


# === 顔判定 + 絵文字マスク処理 ===
def process_image(img_path):
    if my_face_vector is None:
        raise RuntimeError("⚠️ 先にあなたの顔画像を学習してください。")

    with open(img_path, 'rb') as f:
        data = np.fromfile(f, dtype=np.uint8)
    original_img = cv2.imdecode(data, cv2.IMREAD_COLOR)

    if original_img is None:
        raise FileNotFoundError(f"画像が読み込めません: {img_path}")

    processed_img = original_img.copy()
    gray = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.05, 5)

    for (x, y, w, h) in faces:
        face = processed_img[y:y+h, x:x+w]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(face, (96, 96)), 1.0/255, (96, 96),
            (0, 0, 0), swapRB=True, crop=False
        )
        embedder.setInput(blob)
        vec = embedder.forward().flatten()

        sim = np.dot(my_face_vector, vec) / (np.linalg.norm(my_face_vector) * np.linalg.norm(vec))

        if sim < 0.9:
            print(f"😎 他人の顔を検出 (類似度={sim:.2f})")
            emoji_resized = cv2.resize(emoji, (w, h))
            if emoji_resized.shape[2] == 4:
                alpha_s = emoji_resized[:, :, 3] / 255.0
                alpha_l = 1.0 - alpha_s
                for c in range(3):
                    processed_img[y:y+h, x:x+w, c] = (
                        alpha_s * emoji_resized[:, :, c] +
                        alpha_l * processed_img[y:y+h, x:x+w, c]
                    )
            else:
                processed_img[y:y+h, x:x+w] = emoji_resized

    return original_img, processed_img


# === PySimpleGUI UI ===
image_size = (300, 300)

layout = [
    [sg.Text('', size=(60, 1), key='-STATUS-')],
    [sg.Text("① あなたの顔画像を選択（複数可）して学習フォルダに追加")],
    [sg.Input(key='-FILEPATHS-', enable_events=True, visible=False),
     sg.FilesBrowse('学習画像を追加', target='-FILEPATHS-', file_types=(("Image Files", "*.png *.jpg *.jpeg"),))],
    [sg.Button('📘 再学習する'), sg.Button('🧹 学習データをリセット')],

    [sg.Text("② 判定したい画像を選択")],
    [sg.Input(key='-FILEPATH-', enable_events=True, visible=False),
     sg.FileBrowse('判定画像を選択', target='-FILEPATH-', file_types=(("Image Files", "*.png *.jpg *.jpeg"),))],

    [sg.HSeparator()],
    [
        sg.Column([
            [sg.Text('オリジナル')],
            [sg.Image(size=image_size, key='-IMG_ORIG-')]
        ]),
        sg.VSeparator(),
        sg.Column([
            [sg.Text('処理後')],
            [sg.Image(size=image_size, key='-IMG_PROC-')]
        ])
    ]
]

window = sg.Window('OpenCV 顔認識デモ (PySimpleGUI版)', layout)

# === イベントループ ===
while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED:
        break

    # 学習画像追加
    if event == '-FILEPATHS-':
        filepaths = values['-FILEPATHS-'].split(';')
        for f in filepaths:
            if os.path.exists(f):
                shutil.copy(f, os.path.join(study_dir, os.path.basename(f)))
        window['-STATUS-'].update(f'📂 {len(filepaths)}枚を学習フォルダに追加しました。')

    # 再学習
    if event == '📘 再学習する':
        try:
            study_images_from_folder(study_dir)
            window['-STATUS-'].update('✅ 再学習が完了しました！')
        
        except Exception as e:
            window['-STATUS-'].update(f'学習エラー: {e}')
            print(e)

    # 🧹 学習データをリセット
    if event == '🧹 学習データをリセット':
        try:
            for filename in os.listdir(study_dir):
                file_path = os.path.join(study_dir, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            my_embeddings.clear()
            my_face_vector = None
            window['-STATUS-'].update('🧹 学習データをすべて削除しました。')
            print("🧹 studyフォルダ内のファイルを削除しました。")
        except Exception as e:
            window['-STATUS-'].update(f'リセットエラー: {e}')
            print(e)

    # 判定画像の処理
    if event == '-FILEPATH-':
        img_path = values['-FILEPATH-']
        if img_path:
            try:
                orig, proc = process_image(img_path)
                orig_b64 = cv2_to_base64(orig, size=image_size)
                proc_b64 = cv2_to_base64(proc, size=image_size)
                window['-IMG_ORIG-'].update(data=orig_b64)
                window['-IMG_PROC-'].update(data=proc_b64)
                window['-STATUS-'].update(f'✅ 判定完了: {img_path}')
            except Exception as e:
                window['-STATUS-'].update(f'処理エラー: {e}')
                print(e)

window.close()
