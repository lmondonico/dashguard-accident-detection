#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input

# 📥 1. Load CSVs & pad IDs
df       = pd.read_csv('./data/nexar-collision-prediction/train.csv')
df_test  = pd.read_csv('./data/nexar-collision-prediction/test.csv')
df['id']       = df['id'].astype(str).str.zfill(5)
df_test['id']  = df_test['id'].astype(str).str.zfill(5)

# 📂 2. Define video folders & filenames
train_dir = './data/nexar-collision-prediction/train/'
test_dir  = './data/nexar-collision-prediction/test/'

df['train_videos'] = df['id']       + '.mp4'
df_test['test_videos'] = df_test['id'] + '.mp4'

# 🔍 3. Quick sanity prints
print(f"Sample Train ID:\n{df['id'].head().to_list()}")
print(f"Sample Test ID:\n{df_test['id'].head().to_list()}")
print(f"Total Train Videos: {len(df['train_videos'])}")
print(f"Total Test Videos:  {len(df_test['test_videos'])}")

# 🖼️ 4. Frame sampling helper
def extract_frames(path, num_frames=16, size=(224,224)):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return np.empty((0, *size, 3), dtype=np.uint8)
    step = max(total // num_frames, 1)
    frames = []
    for i in range(num_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, size)
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(frames) if frames else np.empty((0, *size, 3), dtype=np.uint8)

# 🔧 5. Load CNN for feature extraction
base_model = InceptionV3(weights='imagenet', include_top=False, pooling='avg')
feat_dim   = base_model.output_shape[-1]

def get_features(ids, folder):
    feats = []
    for vid in tqdm(ids, desc=f"Extracting from {folder}"):
        path = os.path.join(folder, f"{vid}.mp4")
        frames = extract_frames(path)
        if frames.size == 0:
            feats.append(np.zeros(feat_dim, dtype=np.float32))
            continue
        x = preprocess_input(frames.astype('float32'))
        f = base_model.predict(x, batch_size=16, verbose=0)
        feats.append(f.mean(axis=0))
    return np.vstack(feats)

# ⚙️ 6. Extract features
X_TRAIN_FEATURES_FILE = 'features/X_train_full_features.npy'
X_TEST_FEATURES_FILE = 'features/X_test_features.npy'

if os.path.exists(X_TRAIN_FEATURES_FILE) and os.path.exists(X_TEST_FEATURES_FILE):
    print(f"Loading pre-computed features from {X_TRAIN_FEATURES_FILE} and {X_TEST_FEATURES_FILE}...")
    X_train_full = np.load(X_TRAIN_FEATURES_FILE)
    X_test = np.load(X_TEST_FEATURES_FILE)
    print("Features loaded successfully.")
else:
    print("Pre-computed features not found. Starting feature extraction...")
    X_train_full = get_features(df['id'], train_dir)
    np.save(X_TRAIN_FEATURES_FILE, X_train_full)
    print(f"Saved training features to {X_TRAIN_FEATURES_FILE}")
    
    X_test = get_features(df_test['id'], test_dir)
    np.save(X_TEST_FEATURES_FILE, X_test)
    print(f"Saved test features to {X_TEST_FEATURES_FILE}")

# X_train_full = get_features(df['id'],      train_dir)
# X_test       = get_features(df_test['id'], test_dir)

# 🔄 7. Scale & split
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train_full)
y        = df['target'].values

X_tr, X_val, y_tr, y_val = train_test_split(
    X_scaled, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# 🌲 8. Train RandomForest baseline
clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
clf.fit(X_tr, y_tr)

# 📈 9. Validate
val_pred = clf.predict_proba(X_val)[:,1]
print("Validation ROC‑AUC:", roc_auc_score(y_val, val_pred))

# 💾 10. Inference & submission
X_test_scaled = scaler.transform(X_test)
test_pred     = clf.predict_proba(X_test_scaled)[:,1]

submission = pd.DataFrame({
    'id':    df_test['id'],
    'score': test_pred
})
submission.to_csv('submission.csv', index=False)
print("✅ Written → submission.csv")