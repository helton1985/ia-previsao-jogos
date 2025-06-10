from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DadosJogo(BaseModel):
    gols_time: int
    gols_adversario: int
    chutes: int
    escanteios: int

dados_ficticios = pd.DataFrame({
    'gols_time': [1, 2, 3, 0, 1, 2],
    'gols_adversario': [0, 2, 1, 1, 3, 2],
    'chutes': [10, 15, 12, 8, 9, 14],
    'escanteios': [5, 6, 4, 3, 7, 5]
})

def preparar_dados(df):
    df = df.dropna()
    df['resultado'] = df.apply(lambda linha: 1 if linha['gols_time'] > linha['gols_adversario'] else (0 if linha['gols_time'] == linha['gols_adversario'] else -1), axis=1)
    X = df.drop(columns=['resultado'])
    y = df['resultado']
    return X, y

X, y = preparar_dados(dados_ficticios)
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X, y)

@app.post("/prever")
def prever_resultado(dados: DadosJogo):
    entrada = pd.DataFrame([dados.dict()])
    resultado = modelo.predict(entrada)[0]
    mapa_resultado = {1: "Vitória", 0: "Empate", -1: "Derrota"}
    return {"previsao": mapa_resultado[resultado]}

@app.get("/")
def status():
    return {"mensagem": "API de previsão de jogos está online."}
