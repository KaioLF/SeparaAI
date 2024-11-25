import csv
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
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.express as px
from sklearn.metrics import roc_curve, auc
from django.shortcuts import redirect


def index(request):
    return render(request, 'index.html')

def upload_file(request):
    form = UploadCSVForm()
    data_preview = None
    file_path = None
    error_message = None
    target_distribution_plot = None

    if request.method == 'POST':
        form = UploadCSVForm(request.POST, request.FILES)
        if form.is_valid():
            file = request.FILES['file']
            file_path = os.path.join(settings.MEDIA_ROOT, 'base.csv')

            # Salvar o arquivo no diretório media
            try:
                with open(file_path, 'wb+') as destination:
                    for chunk in file.chunks():
                        destination.write(chunk)
            except Exception as e:
                error_message = f"Erro ao salvar o arquivo: {e}"
                return render(request, 'upload_file.html', {'form': form, 'error': error_message})

            # Pré-visualizar os dados
            try:
                df = pd.read_csv(file_path, sep=';')
                data_preview = df.head(10).to_html()
            except Exception as e:
                error_message = f"Erro ao ler o arquivo CSV: {e}"
                return render(request, 'upload_file.html', {'form': form, 'error': error_message})

            # Gerar gráfico interativo de distribuição do target
            try:
                target_distribution_plot = generate_target_distribution_plot(df)
            except Exception as e:
                error_message = f"Erro ao gerar o gráfico de distribuição do target: {e}"
                return render(request, 'upload_file.html', {'form': form, 'error': error_message})

    return render(request, 'upload_file.html', {
        'form': form,
        'data': data_preview,
        'file_path': file_path,
        'error': error_message,
        'target_distribution_plot': target_distribution_plot,
    })

def train_model(request):
    result = None
    retrain = request.GET.get('retrain', 'false').lower() == 'true'  # Checa se é retreino

    if request.method == 'POST' or retrain:
        # Obter parâmetros do formulário ou da sessão
        if retrain:
            algorithm = request.session.get('algorithm')
            max_depth = request.session.get('max_depth')
            n_estimators = request.session.get('n_estimators')
            n_neighbors = request.session.get('n_neighbors')
            penalty = request.session.get('penalty')
            max_iter = request.session.get('max_iter')
        else:
            algorithm = request.POST.get('algorithm')
            max_depth = request.POST.get('max_depth')
            n_estimators = request.POST.get('n_estimators')
            n_neighbors = request.POST.get('n_neighbors')
            penalty = request.POST.get('penalty')
            max_iter = request.POST.get('max_iter')

            # Salvar os parâmetros na sessão
            request.session['algorithm'] = algorithm
            request.session['max_depth'] = max_depth
            request.session['n_estimators'] = n_estimators
            request.session['n_neighbors'] = n_neighbors
            request.session['penalty'] = penalty
            request.session['max_iter'] = max_iter

        try:
            # Carregar dataset
            file_path = os.path.join(settings.MEDIA_ROOT, 'base.csv')
            df = pd.read_csv(file_path, sep=';')
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Seleção de algoritmo
            model = None
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

            # Treinar o modelo
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            classification_rep = classification_report(y_test, y_pred, target_names=["Não Divórcio", "Divórcio"])

            # Gerar gráficos
            confusion_matrix_html = generate_confusion_matrix_plot_interactive(y_test, y_pred)
            roc_curve_html = generate_roc_curve_plot_interactive(y_test, model.predict_proba(X_test))
            feature_importance_html = None

            # Gerar gráfico de importância das variáveis (se suportado)
            feature_importance_html = generate_feature_importance_plot(file_path, model)

            # Salvar modelo treinado
            joblib.dump(model, os.path.join(settings.MEDIA_ROOT, 'trained_model.pkl'))

            # Resultados para o template
            result = {
                'accuracy': f"{accuracy * 100:.2f}%",
                'classification_report': classification_rep,
                'confusion_matrix_html': confusion_matrix_html,
                'roc_curve_html': roc_curve_html,
                'feature_importance_html': feature_importance_html,
            }
        except Exception as e:
            return render(request, 'predictor.html', {'error': f"Erro no treinamento: {e}"})

    return render(request, 'predictor.html', {
        'result': result,
        'error': None,
    })

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
    
def data_analysis(request):
    correlation_matrix_html = None
    response_distribution_html = None

    # Caminho do dataset
    data_path = os.path.join(settings.MEDIA_ROOT, 'base.csv')

    # Verificar se o dataset existe
    if not os.path.exists(data_path):
        return render(request, 'data_analysis.html', {'error': 'Dataset não encontrado. Faça o upload primeiro.'})

    # Carregar o dataset
    try:
        df = pd.read_csv(data_path, sep=';')
        if df.empty:
            raise ValueError("O dataset está vazio.")
        if df.shape[1] < 2:
            raise ValueError("O dataset não possui as colunas esperadas.")
    except Exception as e:
        return render(request, 'data_analysis.html', {'error': f"Erro ao carregar o dataset: {e}"})

    # Gerar a matriz de correlação interativa
    try:
        correlation_matrix_html = generate_correlation_matrix_plot_interactive(df)
    except Exception as e:
        return render(request, 'data_analysis.html', {'error': f"Erro ao gerar a matriz de correlação: {e}"})

    # Gerar a distribuição de respostas por pergunta
    try:
        response_distribution_html = generate_response_distribution_plot(df)
    except Exception as e:
        return render(request, 'data_analysis.html', {'error': f"Erro ao gerar o gráfico de distribuição de respostas: {e}"})

    # Renderizar o template com os gráficos
    return render(request, 'data_analysis.html', {
        'correlation_matrix_html': correlation_matrix_html,
        'response_distribution_html': response_distribution_html,
        'error': None,
    })

def add_record_view(request):
    # Caminho para o arquivo de referência e o dataset
    reference_path = os.path.join(settings.MEDIA_ROOT, 'reference.tsv')
    dataset_path = os.path.join(settings.MEDIA_ROOT, 'base.csv')

    # Ler o arquivo de referência e processar os dados
    try:
        reference_data = pd.read_csv(reference_path, sep='\t', header=None, names=["question_number", "question_text"])
        reference_data = reference_data[reference_data['question_number'].str.contains('|', regex=False)]
        reference_data[['question_number', 'question_text']] = reference_data['question_number'].str.split('|', expand=True)
        reference_data['question_number'] = reference_data['question_number'].str.strip()
        reference_data['question_text'] = reference_data['question_text'].str.strip()
        questions = reference_data.to_dict(orient='records')
    except Exception as e:
        return render(request, 'add_record.html', {'error': f'Erro ao carregar o arquivo de referência: {e}'})

    success_message = None
    error_message = None

    if request.method == 'POST':
        try:
            # Capturar as respostas do formulário
            answers = [request.POST.get(f'q{question["question_number"]}', None) for question in questions]
            divorce_label = request.POST.get('divorce', None)  # Capturar o valor do target

            # Verificar se todas as respostas e o target foram preenchidos
            if None in answers or divorce_label is None:
                error_message = 'Por favor, preencha todas as perguntas e selecione o rótulo de divórcio.'
            else:
                # Adicionar o registro ao dataset
                with open(dataset_path, 'a', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile, delimiter=';')
                    writer.writerow(answers + [divorce_label])
                success_message = 'Registro adicionado ao dataset com sucesso!'
        except Exception as e:
            error_message = f'Erro ao adicionar o registro: {e}'

    # Renderizar o template com as perguntas e mensagens de sucesso/erro
    return render(request, 'add_record.html', {
        'questions': questions,
        'value_range': range(5),
        'success_message': success_message,
        'error_message': error_message
    })

def generate_target_distribution_plot(data):
    """
    Gera um gráfico de barras interativo para distribuição do target.
    """
    # Obter a distribuição do target
    target_counts = data.iloc[:, -1].value_counts()
    labels = ["Não Divórcio", "Divórcio"]
    values = target_counts.values

    # Criar gráfico de barras
    fig = go.Figure(data=[go.Bar(x=labels, y=values, marker_color=['skyblue', 'orange'])])

    # Configurar layout do gráfico
    fig.update_layout(
        title="Distribuição do Target (Balanceamento do Dataset)",
        xaxis_title="Classe",
        yaxis_title="Número de Amostras",
        template="plotly_white",
    )

    # Retornar o gráfico como HTML
    return fig.to_html(full_html=False)

def generate_confusion_matrix_plot_interactive(y_true, y_pred):
    """
    Gera um gráfico interativo de matriz de confusão usando Plotly.

    Args:
        y_true (array): Valores reais.
        y_pred (array): Valores previstos pelo modelo.

    Returns:
        str: HTML contendo o gráfico interativo.
    """
    cm = confusion_matrix(y_true, y_pred)
    labels = ["Não Divórcio", "Divórcio"]

    # Criar o gráfico de matriz de confusão
    fig = ff.create_annotated_heatmap(
        z=cm,
        x=labels,
        y=labels,
        colorscale="Blues",
        showscale=True,
        hoverinfo="z",
    )
    fig.update_layout(
        title="Matriz de Confusão",
        xaxis=dict(title="Previsto"),
        yaxis=dict(title="Real"),
    )

    # Retornar o HTML do gráfico interativo
    return fig.to_html(full_html=False)

def generate_correlation_matrix_plot_interactive(data):
    """
    Gera um gráfico interativo de matriz de correlação usando Plotly.

    Args:
        data (DataFrame): Dados para calcular a matriz de correlação.

    Returns:
        str: HTML contendo o gráfico interativo.
    """
    correlation = data.corr()
    labels = correlation.columns.tolist()

    # Criar o gráfico de matriz de correlação
    fig = go.Figure(
        data=go.Heatmap(
            z=correlation.values,
            x=labels,
            y=labels,
            colorscale="Viridis",
            colorbar=dict(title="Correlação"),
        )
    )
    fig.update_layout(
        title="Matriz de Correlação",
        xaxis=dict(title="Variáveis"),
        yaxis=dict(title="Variáveis"),
    )

    # Retornar o HTML do gráfico interativo
    return fig.to_html(full_html=False)

def generate_roc_curve_plot_interactive(y_true, y_pred_proba):
    """
    Gera um gráfico interativo da Curva ROC usando Plotly.

    Args:
        y_true (array): Valores reais.
        y_pred_proba (array): Probabilidades previstas pelo modelo.

    Returns:
        str: HTML contendo o gráfico interativo.
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])  # Considerando classe positiva como índice 1
    roc_auc = auc(fpr, tpr)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'AUC = {roc_auc:.2f}', line=dict(color='darkorange')))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Aleatório', line=dict(dash='dash', color='gray')))
    fig.update_layout(
        title="Curva ROC",
        xaxis=dict(title="Taxa de Falsos Positivos (FPR)"),
        yaxis=dict(title="Taxa de Verdadeiros Positivos (TPR)"),
        showlegend=True,
    )

    return fig.to_html(full_html=False)

def generate_feature_importance_plot(filepath, model):
    try:
        # Carregar o dataset
        df = pd.read_csv(filepath, sep=';')
        feature_names = df.columns[:-1]  # Todas as colunas menos o target

        # Verificar se o modelo suporta importâncias
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = model.coef_[0]
        else:
            return None  # Retornar None se o modelo não suportar importâncias

        # Criar DataFrame de importâncias
        importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        importance_df = importance_df.sort_values(by='Importance', ascending=False)

        # Gerar o gráfico interativo
        fig = px.bar(
            importance_df,
            x='Importance',
            y='Feature',
            orientation='h',
            title='Importância das Variáveis',
            labels={'Importance': 'Importância', 'Feature': 'Variável'}
        )
        fig.update_layout(yaxis=dict(autorange="reversed"))  # Reverter a ordem do eixo Y

        return fig.to_html(full_html=False)
    except Exception as e:
        print(f"Erro ao gerar gráfico de importância das variáveis: {e}")
        return None
  
def generate_prediction_probability_distribution_plot(y_test, y_pred_proba):
    # Criar histogramas separados para cada classe
    fig = go.Figure()

    for i, label in enumerate(["Não Divórcio", "Divórcio"]):
        fig.add_trace(go.Histogram(
            x=y_pred_proba[:, i],
            name=f"Classe {label}",
            opacity=0.7,
        ))

    # Ajustar layout
    fig.update_layout(
        title="Distribuição de Probabilidades de Previsões",
        xaxis_title="Probabilidade",
        yaxis_title="Frequência",
        barmode="overlay",
        legend_title="Classe",
    )
    fig.update_traces(opacity=0.75)

    return fig.to_html(full_html=False)

def generate_response_distribution_plot(df):
    # Garantir que as colunas estão nomeadas corretamente
    question_columns = [f"Q{i+1}" for i in range(df.shape[1] - 1)]  # Excluir a coluna target
    response_data = df.iloc[:, :-1]  # Todas as colunas menos o target
    response_data.columns = question_columns

    # Calcular a distribuição relativa para todas as perguntas
    distribution = pd.DataFrame()
    for column in response_data.columns:
        counts = response_data[column].value_counts(normalize=True).sort_index()  # Frequência relativa
        counts.name = column
        distribution = pd.concat([distribution, counts], axis=1)

    distribution = distribution.fillna(0).T  # Transpor para facilitar o plot

    # Criar gráfico de barras empilhadas
    fig = go.Figure()

    for response in range(5):  # Respostas de 0 a 4
        fig.add_trace(go.Bar(
            x=distribution.index,
            y=distribution[response] if response in distribution.columns else [0] * len(distribution),
            name=f"Resposta {response}"
        ))

    # Personalizar o layout do gráfico
    fig.update_layout(
        barmode='stack',
        title="Distribuição das Respostas por Pergunta",
        xaxis_title="Perguntas",
        yaxis_title="Proporção das Respostas",
        legend_title="Respostas (0 a 4)",
        xaxis_tickangle=45,
        width=1200,
        height=600
    )

    return fig.to_html(full_html=False)


