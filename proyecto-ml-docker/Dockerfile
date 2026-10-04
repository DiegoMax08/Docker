#Imagen base oficial de Python
FROM python:3.10-slim

#Directorio de trabajo dentro del contenedor
WORKDIR /app

#Copiar archivo de requerimientos e instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

#Copiar los archivos del proyecto al contenedor
COPY main.py .
COPY model.pkl .

#Exponer el puerto donde correrá FastAPI
EXPOSE 8000

#Comando para iniciar la API al ejecutar el contenedor
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]