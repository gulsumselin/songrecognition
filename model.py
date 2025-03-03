import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import pandas as pd
import joblib
import json
import matplotlib.pyplot as plt
from model_definition import CNNModel

# Veri setini yükleme ve ön işleme
data = pd.read_csv("normalized_audio.csv")  # Sizin dosyanızın yolunu buraya koyun

# Hedef sütunu ve özellikleri ayırma
target_column = "folder"  # Hedef sütun adı
features = [col for col in data.columns if col not in ["file_name", "folder"]]  # Özellik sütunları

# Özellikleri normalize etme
scaler = StandardScaler()
data[features] = scaler.fit_transform(data[features])

# Hedef sınıfları kodlama
label_encoder = LabelEncoder()
data["folder_encoded"] = label_encoder.fit_transform(data[target_column])

X = data[features].values
y = data["folder_encoded"].values

# Veri dengesini sağlamak için SMOTE uygulama
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

X_train, X_temp, y_train, y_temp = train_test_split(
    X_resampled, y_resampled, test_size=0.3, random_state=42, stratify=y_resampled
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

# CNN giriş verisi için yeniden şekillendirme
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

dataset_train = TensorDataset(X_train_tensor, y_train_tensor)
dataset_val = TensorDataset(X_val_tensor, y_val_tensor)
dataset_test = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(dataset_train, batch_size=64, shuffle=True)
val_loader = DataLoader(dataset_val, batch_size=64, shuffle=False)
test_loader = DataLoader(dataset_test, batch_size=64, shuffle=False)

k_folds = 4
kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
input_length = X.shape[1]
num_classes = len(set(y_resampled))

fold_accuracies = []
best_val_accuracy = 0.0  # En iyi doğruluk başlangıç değeri

for fold, (train_index, val_index) in enumerate(kf.split(X_resampled, y_resampled)):
    print(f"Fold {fold+1}/{k_folds}")
    
    X_train_fold, X_val_fold = X_resampled[train_index], X_resampled[val_index]
    y_train_fold, y_val_fold = y_resampled[train_index], y_resampled[val_index]

    X_train_tensor = torch.tensor(X_train_fold, dtype=torch.float32).unsqueeze(1)
    X_val_tensor = torch.tensor(X_val_fold, dtype=torch.float32).unsqueeze(1)
    y_train_tensor = torch.tensor(y_train_fold, dtype=torch.long)
    y_val_tensor = torch.tensor(y_val_fold, dtype=torch.long)

    dataset_train = TensorDataset(X_train_tensor, y_train_tensor)
    dataset_val = TensorDataset(X_val_tensor, y_val_tensor)
    train_loader = DataLoader(dataset_train, batch_size=64, shuffle=True)
    val_loader = DataLoader(dataset_val, batch_size=64, shuffle=False)

    model = CNNModel(input_length, num_classes)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    for epoch in range(100):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_accuracy = 100 * val_correct / val_total
        scheduler.step(val_loss)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), "best_model.pth")
            print(f"New best model saved with validation accuracy: {best_val_accuracy:.2f}%")

        print(f"Epoch {epoch+1}/100, Loss: {running_loss/len(train_loader):.4f}, Validation Loss: {val_loss/len(val_loader):.4f}, Validation Accuracy: {val_accuracy:.2f}%")

    fold_accuracies.append(val_accuracy)

print(f"Average Cross-Validation Accuracy: {sum(fold_accuracies)/len(fold_accuracies):.2f}%")

if os.path.exists("best_model.pth"):
    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()
    test_correct = 0
    test_total = 0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
            y_true.extend(labels.tolist())
            y_pred.extend(predicted.tolist())

    test_accuracy = 100 * test_correct / test_total
    print(f"Test Accuracy: {test_accuracy:.2f}%")

    # Sınıf Bazlı Performans Raporu
    from sklearn.metrics import classification_report

    print("Sınıf Bazlı Performans Raporu:")
    print(classification_report(y_true, y_pred, target_names=label_encoder.classes_))


else:
    print("Model dosyası bulunamadı. Eğitim sırasında model kaydedilmiş mi kontrol edin.")
# Test Accuracy ve Sınıf Bazlı Performans Raporunu Kaydetme
output_file = "classification_report.txt"

with open(output_file, "w") as f:
    # Test doğruluk oranını dosyaya yaz
    f.write(f"Test Accuracy: {test_accuracy:.2f}%\n\n")
    
    # Sınıf bazlı performans raporunu dosyaya yaz
    f.write("Sınıf Bazlı Performans Raporu:\n")
    f.write(classification_report(y_true, y_pred, target_names=label_encoder.classes_))
    
print(f"Sınıf bazlı performans raporu {output_file} dosyasına kaydedildi.")

joblib.dump(scaler, "scaler.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

model_config = {"input_length": input_length, "num_classes": num_classes}
with open("model_config.json", "w") as f:
    json.dump(model_config, f)