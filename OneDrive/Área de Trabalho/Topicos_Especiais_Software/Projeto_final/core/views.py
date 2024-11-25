import csv
import os

import pandas as pd
from django.conf import settings
from django.shortcuts import render
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
import joblib
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.express as px

from .forms.uploadCSVform import UploadCSVForm


def index(request):
    """
    View para renderizar a página inicial.
    """
    return render(request, 'index.html')

def upload_file(request):
    """
    View para realizar o upload de um arquivo CSV. 
    Também exibe uma pré-visualização dos dados e o gráfico interativo do target.
    """
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
    """
    View para treinar ou retreinar um modelo de Machine Learning.
    Salva os parâmetros de configuração e exibe os resultados, como métricas e gráficos.
    """
    result = None
    retrain = request.GET.get('retrain', 'false').lower() == 'true'  # Checa se é retreino

    if request.method == 'POST' or retrain:
        if retrain:
            # Recuperar parâmetros da sessão no retreino
            algorithm = request.session.get('algorithm')
            max_depth = request.session.get('max_depth')
            n_estimators = request.session.get('n_estimators')
            n_neighbors = request.session.get('n_neighbors')
            penalty = request.session.get('penalty')
            max_iter = request.session.get('max_iter')
        else:
            # Obter parâmetros do formulário e salvar na sessão
            algorithm = request.POST.get('algorithm')
            max_depth = request.POST.get('max_depth')
            n_estimators = request.POST.get('n_estimators')
            n_neighbors = request.POST.get('n_neighbors')
            penalty = request.POST.get('penalty')
            max_iter = request.POST.get('max_iter')

            request.session['algorithm'] = algorithm
            request.session['max_depth'] = max_depth
            request.session['n_estimators'] = n_estimators
            request.session['n_neighbors'] = n_neighbors
            request.session['penalty'] = penalty
            request.session['max_iter'] = max_iter

        try:
            # Carregar dataset e dividir em treino e teste
            file_path = os.path.join(settings.MEDIA_ROOT, 'base.csv')
            df = pd.read_csv(file_path, sep=';')
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Configurar o modelo
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
    """
    View para realizar predições com base nas respostas do formulário.
    Carrega o modelo treinado e calcula a probabilidade de divórcio.
    """
    # Caminho para o arquivo de referência e o modelo treinado
    reference_path = os.path.join(settings.MEDIA_ROOT, 'reference.tsv')
    model_path = os.path.join(settings.MEDIA_ROOT, 'trained_model.pkl')

    # Ler o arquivo de referência e processar as questões
    reference_data = pd.read_csv(reference_path, sep='\t', header=None, names=["question_number", "question_text"])
    reference_data = reference_data[reference_data['question_number'].str.contains('|', regex=False)]
    reference_data[['question_number', 'question_text']] = reference_data['question_number'].str.split('|', expand=True)
    reference_data['question_number'] = reference_data['question_number'].str.strip()
    reference_data['question_text'] = reference_data['question_text'].str.strip()
    questions = reference_data.to_dict(orient='records')

    divorce_probability = None
    error = None

    if request.method == 'POST':
        try:
            # Capturar as respostas do formulário
            answers = [int(request.POST.get(f'q{question["question_number"]}', 0)) for question in questions]

            # Verificar se o modelo treinado existe
            if not os.path.exists(model_path):
                error = 'Modelo treinado não encontrado. Certifique-se de treinar o modelo antes de realizar a predição.'
            else:
                # Carregar o modelo treinado
                model = joblib.load(model_path)

                # Realizar a predição
                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba([answers])[0]
                    divorce_probability = probabilities[1] * 100 # Probabilidade de divórcio
                else:
                    error = 'O modelo treinado não suporta predições de probabilidade.'
        except Exception as e:
            error = f'Erro ao realizar a predição: {e}'

    return render(request, 'predict.html', {
        'questions': questions,
        'value_range': range(5),
        'divorce_probability': divorce_probability,
        'error': error
    })

def data_analysis(request):
    """
    View para realizar análise de dados.
    Gera e exibe a matriz de correlação e a distribuição de respostas por pergunta.
    """
    data_path = os.path.join(settings.MEDIA_ROOT, 'base.csv')
    correlation_matrix_html = None
    response_distribution_html = None

    # Verificar se o dataset existe
    if not os.path.exists(data_path):
        return render(request, 'data_analysis.html', {'error': 'Dataset não encontrado. Faça o upload primeiro.'})

    try:
        # Carregar o dataset
        df = pd.read_csv(data_path, sep=';')
        if df.empty:
            raise ValueError("O dataset está vazio.")
        if df.shape[1] < 2:
            raise ValueError("O dataset não possui as colunas esperadas.")
    except Exception as e:
        return render(request, 'data_analysis.html', {'error': f"Erro ao carregar o dataset: {e}"})

    try:
        # Gerar gráficos
        correlation_matrix_html = generate_correlation_matrix_plot_interactive(df)
        response_distribution_html = generate_response_distribution_plot(df)
    except Exception as e:
        return render(request, 'data_analysis.html', {'error': f"Erro ao gerar os gráficos: {e}"})

    return render(request, 'data_analysis.html', {
        'correlation_matrix_html': correlation_matrix_html,
        'response_distribution_html': response_distribution_html,
        'error': None,
    })

def add_record_view(request):
    """
    View para adicionar um novo registro ao dataset.
    Atualiza o CSV base com as respostas fornecidas pelo usuário.
    """
    reference_path = os.path.join(settings.MEDIA_ROOT, 'reference.tsv')
    dataset_path = os.path.join(settings.MEDIA_ROOT, 'base.csv')

    try:
        # Carregar as questões de referência
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
            # Capturar respostas e target
            answers = [request.POST.get(f'q{question["question_number"]}', None) for question in questions]
            divorce_label = request.POST.get('divorce', None)

            if None in answers or divorce_label is None:
                error_message = 'Preencha todas as perguntas e selecione o rótulo de divórcio.'
            else:
                # Adicionar registro ao dataset
                with open(dataset_path, 'a', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile, delimiter=';')
                    writer.writerow(answers + [divorce_label])
                success_message = 'Registro adicionado ao dataset com sucesso!'
        except Exception as e:
            error_message = f'Erro ao adicionar o registro: {e}'

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
    try:
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

        return fig.to_html(full_html=False)
    except Exception as e:
        print(f"Erro ao gerar gráfico de matriz de correlação: {e}")
        return None

def generate_roc_curve_plot_interactive(y_true, y_pred_proba):
    """
    Gera um gráfico interativo da Curva ROC usando Plotly.

    Args:
        y_true (array): Valores reais.
        y_pred_proba (array): Probabilidades previstas pelo modelo.

    Returns:
        str: HTML contendo o gráfico interativo.
    """
    try:
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])  # Considerar classe positiva (índice 1)
        roc_auc = auc(fpr, tpr)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, mode='lines', 
            name=f'AUC = {roc_auc:.2f}', 
            line=dict(color='darkorange')
        ))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode='lines', 
            name='Aleatório', 
            line=dict(dash='dash', color='gray')
        ))
        fig.update_layout(
            title="Curva ROC",
            xaxis=dict(title="Taxa de Falsos Positivos (FPR)"),
            yaxis=dict(title="Taxa de Verdadeiros Positivos (TPR)"),
            showlegend=True,
        )

        return fig.to_html(full_html=False)
    except Exception as e:
        print(f"Erro ao gerar gráfico da Curva ROC: {e}")
        return None

def generate_feature_importance_plot(filepath, model):
    """
    Gera um gráfico interativo de importância das variáveis baseado no modelo treinado.

    Args:
        filepath (str): Caminho para o dataset.
        model: Modelo treinado que contém as importâncias das features.

    Returns:
        str: HTML contendo o gráfico interativo ou None caso não suportado.
    """
    try:
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
    """
    Gera um gráfico interativo de distribuição de probabilidades das previsões.

    Args:
        y_test (array): Valores reais.
        y_pred_proba (array): Probabilidades previstas pelo modelo.

    Returns:
        str: HTML contendo o gráfico interativo.
    """
    try:
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
    except Exception as e:
        print(f"Erro ao gerar gráfico de distribuição de probabilidades: {e}")
        return None

def generate_response_distribution_plot(df):
    """
    Gera um gráfico de barras empilhadas interativo para a distribuição das respostas por pergunta.

    Args:
        df (DataFrame): Dataset contendo as respostas das perguntas.

    Returns:
        str: HTML contendo o gráfico interativo.
    """
    try:
        question_columns = [f"Q{i+1}" for i in range(df.shape[1] - 1)]  # Excluir o target
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

        # Ajustar layout do gráfico
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
    except Exception as e:
        print(f"Erro ao gerar gráfico de distribuição de respostas: {e}")
        return None
