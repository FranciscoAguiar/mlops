# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
import pickle
import os

from flask import Flask, request, jsonify
from flask_basicauth import BasicAuth

# Criação de uma app
app = Flask(__name__)
app.config['BASIC_AUTH_USERNAME'] = os.environ.get('BASIC_AUTH_USERNAME')
app.config['BASIC_AUTH_PASSWORD'] = os.environ.get('BASIC_AUTH_PASSWORD')

# Habilitando autenticação na app
basic_auth = BasicAuth(app)

# Antes das APIs
colunas = ['RevolvingUtilizationOfUnsecuredLines', 'age',
       'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio', 'MonthlyIncome',
       'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate',
       'NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse',
       'NumberOfDependents', 'IncomePerPerson', 'NumOfPastDue', 'MonthlyDebt',
       'NumOfOpenCreditLines', 'MonthlyBalance', 'age_sqr']

def load_model(file_name = 'xgboost_undersampling.pkl'):
    return pickle.load(open(file_name, "rb"))

# Carregar modelo treinado
modelo = load_model('models/xgboost_undersampling.pkl')

# Rota de predição de scores
@app.route('/score/', methods=['POST'])
@basic_auth.required
def get_score():
    # Pegar o JSON da requisição
    dados = request.get_json()
    # Garantir a ordem das colunas
    payload = np.array([dados[col] for col in colunas])
    # Fazer predição
    payload = xgb.DMatrix([payload], feature_names=colunas)
    score = np.float64(modelo.predict(payload)[0])
    status = 'APROVADO'
    if score <= 0.3:
        status = 'REPROVADO'
    elif score <= 0.6: 
        status = 'MESA DE AVALIACAO'
    return jsonify(cpf=dados['cpf'], score=score, status=status)

# Nova rota - recebendo CPF
@app.route('/score/<cpf>')
@basic_auth.required
def show_cpf(cpf):
    return 'Recebendo dados\nCPF: %s'%cpf

# Rota padrão
@app.route('/')
def home():
    return 'API de predição de credito'

# Subir a API
app.run(debug=True, host='0.0.0.0')

