import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from flask import Flask, render_template, request

# Inisialisasi Flask app
app = Flask(__name__)

# --- Bagian ML: Training Model KNN ---
# Load dataset Iris
iris = datasets.load_iris()

# Buat DataFrame
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Split fitur dan target
X = df[iris.feature_names]
y = df['target']

# Split data training dan testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Normalisasi
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Training model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Evaluasi model (hanya untuk ditampilkan di terminal)
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Akurasi KNN: {accuracy * 100:.2f}%")

# --- Bagian Flask: Routing ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Ambil data dari form HTML
    sepal_length = float(request.form['sepal_length'])
    sepal_width = float(request.form['sepal_width'])
    petal_length = float(request.form['petal_length'])
    petal_width = float(request.form['petal_width'])

    # Siapkan data dan prediksi
    input_data = scaler.transform([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = knn.predict(input_data)
    species = iris.target_names[prediction[0]]

    return render_template('index.html', species=species)

# --- Jalankan server Flask ---
if __name__ == '__main__':
    app.run(debug=True)