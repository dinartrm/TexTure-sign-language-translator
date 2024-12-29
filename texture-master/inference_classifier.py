import tkinter as tk
from tkinter import Label, Button, Toplevel
from PIL import Image, ImageTk
import threading
import pickle
import cv2
import mediapipe as mp
import numpy as np
import time
from gtts import gTTS
import pygame
import os

# ------------------------- Inisialisasi Model -------------------------
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "model.p")
with open(model_path, "rb") as f:
    model_dict = pickle.load(f)
model = model_dict['model']

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {
    0: 'Halo', 1: 'Perkenalkan', 2: 'Nama', 3: 'Saya', 4: 'Kamu', 5: 'Siapa',
    6: 'Terima Kasih', 7: 'A', 8: 'B', 9: 'C', 10: 'D', 11: 'E', 12: 'N',
    13: 'K', 14: 'R', 15: 'V', 16: 'I', 17: 'O'
}

# --------------------- Variabel Global ---------------------
last_text = ""
stable_text = ""
stable_time = 0
cap = None  # Objek kamera global
running = False  # Status kamera aktif
bg_image = None
bg_photo = None
bg_image_id = None

# ------------------------ Fungsi Pemutaran Suara -----------------------
def speak(text):
    tts = gTTS(text=text, lang='id')
    filename = "output.mp3"
    tts.save(filename)

    if os.path.exists(filename):
        pygame.mixer.init()
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
        
        pygame.mixer.quit()
        os.remove(filename)

# -------------------------- Fungsi Kamera ----------------------------
def start_camera(label, camera_page):
    global cap, running, last_text, stable_text, stable_time

    cap = cv2.VideoCapture(0)
    mp_drawing = mp.solutions.drawing_utils
    running = True

    while running:
        data_aux = []
        x_ = []
        y_ = []

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        H, W, _ = frame.shape

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    data_aux.append(hand_landmarks.landmark[i].x - min(x_))
                    data_aux.append(hand_landmarks.landmark[i].y - min(y_))

            if len(results.multi_hand_landmarks) == 1:
                data_aux += [0] * 42

            if len(data_aux) == 84:
                prediction = model.predict([np.asarray(data_aux)])
                predicted_character = labels_dict[int(prediction[0])]
                current_time = time.time()
                
                if predicted_character == stable_text:
                    if current_time - stable_time >= 0.5:  # Stabil selama 0.5 detik
                        if predicted_character != last_text:
                            last_text = predicted_character
                            label.config(text=f"Gesture: {predicted_character}")
                else:
                    stable_text = predicted_character
                    stable_time = current_time

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert back to RGB for Tkinter
        frame_image = Image.fromarray(frame)
        frame_photo = ImageTk.PhotoImage(frame_image)

        camera_page.video_label.config(image=frame_photo)
        camera_page.video_label.image = frame_photo  # Keep a reference

        # Allow the camera to continue running
        if cv2.waitKey(1) & 0xFF == 27:  # ESC key to stop
            break

    cap.release()

# ------------------------- Fungsi Resize Background ----------------------------
def resize_background(canvas, image_id, bg_image):
    def inner(event):
        # Mendapatkan ukuran baru berdasarkan ukuran canvas
        canvas_width, canvas_height = event.width, event.height

        # Resize gambar background agar sesuai ukuran canvas
        resized_image = bg_image.resize((canvas_width, canvas_height), Image.ANTIALIAS)
        resized_photo = ImageTk.PhotoImage(resized_image)

        # Simpan referensi gambar untuk mencegah garbage collector
        canvas.image = resized_photo
        canvas.itemconfig(image_id, image=resized_photo)

    return inner

# ------------------------- Fungsi Tombol ----------------------------
def open_camera_page():
    global last_text

    camera_page = tk.Toplevel(root)
    camera_page.title("Web Camera")
    camera_page.attributes('-fullscreen', True)

    # Membuat canvas untuk background di halaman kamera
    camera_canvas = tk.Canvas(camera_page, highlightthickness=0)
    camera_canvas.pack(fill="both", expand=True)

    # Menambahkan gambar background ke halaman kamera
    camera_bg_image_id = camera_canvas.create_image(0, 0, image=bg_photo, anchor="nw")

    # Tambahkan event untuk resize halaman kamera
    camera_canvas.bind("<Configure>", resize_background(camera_canvas, camera_bg_image_id, bg_image))

    # Judul di atas
    title_label = Label(camera_canvas, text="Gesture to Text and Speech", font=("Helvetica", 18, "bold"), bg="#fff3db", fg="black")
    title_label.place(relx=0.5, rely=0.26, anchor="center")

    # Frame untuk kamera dengan border
    camera_frame = tk.Frame(camera_canvas, width=680, height=520, bg="#fef5da", bd=1, relief="solid")
    camera_frame.place(relx=0.7, rely=0.6, anchor="center")
    camera_page.video_label = Label(camera_frame, text="Camera feed will appear here.", font=("Helvetica", 18), bg="#fef5da", fg="black")
    camera_page.video_label.place(relx=0.5, rely=0.5, anchor="center")

    # Frame untuk Gesture Text
    gesture_frame = tk.Frame(camera_canvas, width=350, height=100, bg="#fff6a8", bd=1, relief="solid")
    gesture_frame.place(relx=0.25, rely=0.55, anchor="center")
    gesture_label = Label(gesture_frame, text="Gesture Text: detecting camera...", font=("Antonio", 18, "bold"), bg="#fff6a8", fg="black")
    gesture_label.pack(padx=10, pady=10)

    # Tombol Play Sound
    play_sound_btn = Button(camera_canvas, text="Play Sound", font=("Helvetica", 18, "bold"), width=20, height=1, bg="#f2a216", fg="black", command=lambda: speak(last_text))
    play_sound_btn.place(relx=0.25, rely=0.7, anchor="center")

    # Tombol Back
    def back_to_main():
        camera_page.destroy()

    back_btn = Button(camera_canvas, text="Back", font=("Helvetica", 18, "bold"), width=20, height=1, bg="#f2a216", fg="black", command=back_to_main)
    back_btn.place(relx=0.25, rely=0.8, anchor="center")

    # Memulai thread kamera
    threading.Thread(target=start_camera, args=(gesture_label, camera_page), daemon=True).start()

# ------------------------- Pembuatan GUI ----------------------------
root = tk.Tk()
root.title("Gesture Recognition Interface")
root.attributes('-fullscreen', True)
root.protocol("WM_DELETE_WINDOW", root.quit)
root.geometry(f"{root.winfo_screenwidth()}x{root.winfo_screenheight()}")

# Memuat gambar background menggunakan Pillow
bg_image = Image.open("texture_bg.png")  # Pastikan file ada di direktori yang sama
bg_photo = ImageTk.PhotoImage(bg_image)

# Menambahkan Canvas untuk background
canvas = tk.Canvas(root, highlightthickness=0)
canvas.pack(fill="both", expand=True)

# Menampilkan gambar di Canvas
bg_image_id = canvas.create_image(0, 0, image=bg_photo, anchor="nw")

# Menambahkan event untuk resize window
canvas.bind("<Configure>", resize_background(canvas, bg_image_id, bg_image))

# Menambahkan widget di atas background dengan penempatan di tengah
title_label = Label(root, text="Gesture to Text and Speech", font=("Helvetica", 18, "bold"), bg="#fff3db", fg="black",
                    highlightthickness=0)
title_label.place(relx=0.5, rely=0.26, anchor="center")

start_btn = Button(root, text="Open Camera", font=("Helvetica", 18, "bold"), width=25, height=1, bg="#f2a216", fg="black",
                   command=open_camera_page)
start_btn.place(relx=0.5, rely=0.55, anchor="center")

stop_btn = Button(root, text="Exit", font=("Helvetica", 18, "bold"), width=25, height=1, bg="#f2a216", fg="black",
                  command=root.quit)
stop_btn.place(relx=0.5, rely=0.62, anchor="center")

root.mainloop()
