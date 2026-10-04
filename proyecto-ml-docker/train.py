import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def train():
    #Cargar dataset (Clasificación de Flores Iris)
    iris = load_iris()
    X, y = iris.data, iris.target

    #Dividir datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #Entrenar el modelo
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    #Evaluar modelo
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    print(f"Modelo entrenado con precisión (Accuracy): {acc * 100:.2f}%")

    #Guardar el modelo en disco
    joblib.dump(model, "model.pkl")
    print("Modelo guardado exitosamente como 'model.pkl'")

if __name__ == "__main__":
    train()