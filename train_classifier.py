import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np
import pandas as pd
from collections import Counter

try:
    with open('./data.pickle', 'rb') as f:
        data_dict = pickle.load(f)
except FileNotFoundError:
    print("Error: File 'data.pickle' tidak ditemukan.")
    exit()
except Exception as e:
    print(f"Error saat memuat 'data.pickle': {e}")
    exit()

# Verifikasi panjang dari setiap sampel data
data = data_dict['data']
labels = np.asarray(data_dict['labels'])

# distribusi label dalam dataset
print("\nDistribusi Label dalam Dataset:")
print(Counter(labels))

# panjang maksimal dari sampel untuk normalisasi
max_length = max(len(sample) for sample in data)
normalized_data = [sample + [0] * (max_length - len(sample)) for sample in data]
data = np.asarray(normalized_data)

data_df = pd.DataFrame(data)
print("\nJumlah Data Duplikat:", data_df.duplicated().sum())

# validasi ukuran data dan label
if data.shape[0] != labels.shape[0]:
    print("Error: Jumlah data dan label tidak sesuai.")
    exit()

x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels, random_state=42
)

intersect = len(np.intersect1d(x_train, x_test)) > 0
print("\nApakah Data Latih dan Uji Tumpang Tindih?:", intersect)

# RandomForestClassifier parameter
model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

# akurasi prediksi
score = accuracy_score(y_predict, y_test)
print(f'\nAkurasi Model: {score * 100:.2f}%')

# Cross-Validation
cv_scores = cross_val_score(model, data, labels, cv=5)
print("\nHasil Cross-Validation (5-Fold):", cv_scores)
print("Rata-Rata Akurasi Cross-Validation:", np.mean(cv_scores))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_predict))

print("\nClassification Report:")
print(classification_report(y_test, y_predict))

with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
    print("\nModel berhasil disimpan sebagai 'model.p'.")