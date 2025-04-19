import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Veriyi yükle
data = pd.read_csv("loan_data.csv")

# LabelEncoder nesnesi
le = LabelEncoder()

# Kategorik sütunları belirle
categorical_columns = [
    'person_gender',
    'person_education',
    'person_home_ownership',
    'loan_intent',
    'previous_loan_defaults_on_file'
]

# Kategorik sütunları dönüştür
for col in categorical_columns:
    data[col] = le.fit_transform(data[col])

# Özellikler ve hedef
X = data.drop("loan_status", axis=1)
y = data["loan_status"]

# Veri bölme
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Modeli kaydet
with open("model/rf_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model başarıyla kaydedildi.")
