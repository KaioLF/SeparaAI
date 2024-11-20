from django.shortcuts import render
from django.conf import settings
import pandas as pd
from . forms.uploadCSVform import UploadCSVForm
import numpy as np
import matplotlib.pyplot as plt
import io
import os
import base64
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib


def index(request):
    return render(request, 'index.html')

def predictor(request):
    return render(request, 'predictor.html')

def upload_file(request):
    form = UploadCSVForm()
    data_preview = None
    file_path = None

    if request.method == 'POST':
        form = UploadCSVForm(request.POST, request.FILES)
        if form.is_valid():
            file = request.FILES['file']
            file_path = os.path.join(settings.MEDIA_ROOT, 'base.csv')

            # Salvar o arquivo no diretório media
            with open(file_path, 'wb+') as destination:
                for chunk in file.chunks():
                    destination.write(chunk)

            # Pré-visualizar os dados
            df = pd.read_csv(file_path)
            data_preview = df.head(10).to_html()

    return render(request, 'upload_file.html', {'form': form, 'data': data_preview, 'file_path': file_path})

def predictor(request):
    if request.method == 'POST':
        inputs = [int(request.POST.get(f'q{i}', 0)) for i in range(1, 56)]
        model = RandomForestClassifier()  # Substitua pelo modelo treinado
        # Carregue o modelo salvo
        # model = joblib.load('model.pkl')
        prediction = model.predict(np.array(inputs).reshape(1, -1))
        return render(request, 'predictor.html', {'prediction': prediction[0]})
    return render(request, 'predictor.html')

def data_analysis(request, file_path):
    # Carregar o arquivo CSV pelo caminho
    df = pd.read_csv(file_path)

    # Exemplo: Gerar um gráfico simples
    plt.figure(figsize=(10, 5))
    df['coluna_exemplo'].value_counts().plot(kind='bar')
    plt.title('Distribuição da Coluna')
    plt.xlabel('Categorias')
    plt.ylabel('Frequência')

    # Converter o gráfico em imagem para exibição no template
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_png = buf.getvalue()
    buf.close()
    graphic = base64.b64encode(image_png).decode('utf-8')

    return render(request, 'data_analysis.html', {'graphic': graphic})


def train_model(request):
    result = None
    if request.method == 'POST':
        # Obter os dados do formulário
        algorithm = request.POST.get('algorithm')
        max_depth = request.POST.get('max_depth')
        n_estimators = request.POST.get('n_estimators')
        n_neighbors = request.POST.get('n_neighbors')
        penalty = request.POST.get('penalty')
        max_iter = request.POST.get('max_iter')

        # Carregar o dataset do último upload
        file_path = os.path.join(settings.MEDIA_ROOT, 'base.csv')
        if not os.path.exists(file_path):
            return render(request, 'predictor.html', {'error': 'Arquivo de dados não encontrado. Faça o upload primeiro.'})

        # Ler o dataset
        try:
            df = pd.read_csv(file_path, sep=';')
        except Exception as e:
            return render(request, 'predictor.html', {'error': f'Erro ao ler o arquivo: {e}'})

        # Separar features (Q1 a Q54) e target (Divorce)
        try:
            X = df.iloc[:, :-1]  # Todas as colunas exceto a última
            y = df.iloc[:, -1]   # Última coluna
            y = y.values.ravel()  # Garantir que y é um array unidimensional
        except Exception as e:
            return render(request, 'predictor.html', {'error': f'Erro ao processar os dados: {e}'})

        # Validar se X e y não estão vazios
        if X.empty or y.size == 0:
            return render(request, 'predictor.html', {'error': 'As features ou o target estão vazios.'})

        # Tratar valores não numéricos (converter para números ou substituir NaN por 0)
        try:
            X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
        except Exception as e:
            return render(request, 'predictor.html', {'error': f'Erro ao tratar as features: {e}'})

        # Dividir o dataset em treino e teste
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Escolher o algoritmo com base na seleção do usuário
        if algorithm == 'random_forest':
            model = RandomForestClassifier(
                max_depth=int(max_depth) if max_depth else None,
                n_estimators=int(n_estimators) if n_estimators else 100,
                random_state=42
            )
        elif algorithm == 'logistic_regression':
            model = LogisticRegression(
                penalty=penalty if penalty else 'l2',
                max_iter=int(max_iter) if max_iter else 100,
                random_state=42
            )
        elif algorithm == 'knn':
            model = KNeighborsClassifier(
                n_neighbors=int(n_neighbors) if n_neighbors else 5
            )
        else:
            return render(request, 'predictor.html', {'error': 'Algoritmo inválido selecionado.'})

        # Treinar o modelo
        try:
            model.fit(X_train, y_train)
        except Exception as e:
            return render(request, 'predictor.html', {'error': f'Erro ao treinar o modelo: {e}'})

        # Avaliar o modelo
        try:
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
        except Exception as e:
            return render(request, 'predictor.html', {'error': f'Erro ao avaliar o modelo: {e}'})

        # Salvar o modelo treinado
        model_path = os.path.join(settings.MEDIA_ROOT, 'trained_model.pkl')
        try:
            joblib.dump(model, model_path)
        except Exception as e:
            return render(request, 'predictor.html', {'error': f'Erro ao salvar o modelo: {e}'})

        # Preparar o resultado para exibição
        result = {
            'algorithm': algorithm,
            'accuracy': accuracy,
            'model_path': model_path,
        }

    return render(request, 'predictor.html', {'result': result})

def predict_view(request):
    # Caminho para o arquivo de referência e o modelo treinado
    reference_path = os.path.join(settings.MEDIA_ROOT, 'reference.tsv')
    model_path = os.path.join(settings.MEDIA_ROOT, 'trained_model.pkl')  # Modelo treinado

    # Ler o arquivo de referência e corrigir os dados
    reference_data = pd.read_csv(reference_path, sep='\t', header=None, names=["question_number", "question_text"])
    reference_data = reference_data[reference_data['question_number'].str.contains('|', regex=False)]
    reference_data[['question_number', 'question_text']] = reference_data['question_number'].str.split('|', expand=True)
    reference_data['question_number'] = reference_data['question_number'].str.strip()
    reference_data['question_text'] = reference_data['question_text'].str.strip()
    questions = reference_data.to_dict(orient='records')

    divorce_probability = None  # Variável para armazenar a probabilidade de divórcio
    error = None  # Variável para erros

    if request.method == 'POST':
        try:
            # Capturar as respostas do formulário
            answers = [int(request.POST.get(f'q{question["question_number"]}', 0)) for question in questions]

            # Verificar se o modelo existe
            if not os.path.exists(model_path):
                error = 'Modelo treinado não encontrado. Certifique-se de treinar o modelo antes de realizar a predição.'
            else:
                # Carregar o modelo treinado
                model = joblib.load(model_path)

                # Verificar se o modelo suporta probabilidade
                if hasattr(model, "predict_proba"):
                    # Realizar a predição de probabilidade
                    probabilities = model.predict_proba([answers])[0]
                    divorce_probability = probabilities[1]  # Probabilidade da classe 1 (divórcio)
                else:
                    error = 'O modelo treinado não suporta predições de probabilidade.'
        except Exception as e:
            error = f'Erro ao realizar a predição: {e}'

    # Renderizar o template com a probabilidade e/ou erro
    return render(request, 'predict.html', {
        'questions': questions,
        'value_range': range(5),
        'divorce_probability': divorce_probability,
        'error': error
    })





