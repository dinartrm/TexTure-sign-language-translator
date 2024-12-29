import os # untuk bekerja dengan sistem file (membaca direktori dan file)
import pickle # untuk menyimpan dan memuat objek (data dan label) ke dalam file
import mediapipe as mp # untuk mendeteksi tangan (hand tracking)
import cv2 # untuk pemrosesan gambar dan video
import matplotlib.pyplot as plt # digunakan untuk visualisasi

# Mengakses solusi deteksi tangan dari mediapipe
mp_hands = mp.solutions.hands 
mp_drawing = mp.solutions.drawing_utils # menggambar landmark tangan
mp_drawing_styles = mp.solutions.drawing_styles

# Inisialisasi objek deteksi tangan dari mediapipe dengan mendukung dua tangan
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.3)

DATA_DIR = './data'

data = []
labels = []

# Membaca daftar semua folder yang ada di direktori 'data'
for dir_ in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_)
    if os.path.isdir(dir_path):  # Memastikan hanya direktori yang diproses
        for img_path in os.listdir(dir_path):
            data_aux = []
            x_ = []
            y_ = []

            # Membaca gambar dari file
            img = cv2.imread(os.path.join(dir_path, img_path))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Mengubah gambar dari BGR ke RGB

            # Memproses gambar untuk mendeteksi tangan dan landmark
            results = hands.process(img_rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        
                        # Menyimpan koordinat x dan y dari semua landmark tangan
                        x_.append(x)
                        y_.append(y)

                    # Normalisasi koordinat landmark untuk tangan
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        data_aux.append(x - min(x_))
                        data_aux.append(y - min(y_))

                data.append(data_aux)
                labels.append(dir_)

# Simpan data dan label ke file pickle
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)
