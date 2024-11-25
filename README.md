# Projeto de Predição de Divórcios

## Pré-requisitos

Certifique-se de ter as seguintes dependências instaladas:

- Python 3.10+
- Django 4.2+
- Bibliotecas Python:
  - pandas
  - numpy
  - matplotlib
  - joblib
  - scikit-learn
  - plotly
  - bootstrap (somente para frontend, via template CDN)
- Banco de dados SQLite (instalado automaticamente com Django)

Para instalar as bibliotecas necessárias, execute:

```bash
pip install django pandas numpy matplotlib joblib scikit-learn plotly
```

# Como Configurar o Projeto

Clone o Repositório:
```bash
git clone <link-do-repositorio>
cd <nome-do-repositorio>
```

Configuração do Ambiente Virtual (opcional):

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\\Scripts\\activate    # Windows
```

Navegue até a pasta principal onde manage.py está localizado.

Execute as migrações do banco de dados:
```bash
python manage.py makemigrations
python manage.py migrate
```

Inicialize o Servidor de Desenvolvimento:
```bash
python manage.py runserver
```

# Acesse o Sistema:

Abra o navegador e vá para: http://127.0.0.1:8000/

Funcionalidades do Sistema
1. Upload de Arquivos
Navegue até a aba "Upload CSV".
Envie um arquivo .csv no formato esperado.
Pré-visualize os dados enviados e visualize a distribuição do target.
2. Análise de Dados
Navegue até a aba "Análise de Dados".
Explore:
Matriz de correlação.
Distribuição das respostas por pergunta.
3. Treinamento de Modelos
Na aba "Treinamento e Predição":
Selecione o algoritmo de Machine Learning.
Configure os parâmetros do modelo.
Clique em "Treinar Modelo".
Visualize métricas como Matriz de Confusão, Curva ROC e Importância das Variáveis.
4. Predição
Responda o formulário baseado nas perguntas do dataset.
O sistema exibirá a probabilidade de divórcio com base nas respostas.
5. Adicionar Registro
Navegue até "Adicionar Registro".
Preencha o formulário com as respostas e o valor do target.
O registro será adicionado ao dataset para re-treinamento.
Estrutura do Projeto
core: Contém as principais views e lógica do projeto.
templates: Arquivos HTML para renderização das páginas.
media: Armazena arquivos como dataset e modelo treinado.
Observações
Para re-treinar o modelo com novos dados:

Adicione os registros necessários na aba "Adicionar Registro".
Volte para a aba "Treinamento e Predição" e clique em "Treinar Modelo".
Qualquer modelo treinado anteriormente será substituído pelo novo durante o treinamento.

