# NexTripAI - Análise Preditiva de Clientes

Este projeto contém uma solução de ponta a ponta de Ciência de Dados para a ClickBus, focada em entender e prever o comportamento dos clientes. A solução inclui segmentação de clientes, previsão de próxima compra e um dashboard interativo para visualização dos resultados.

## Funcionalidades

- **Segmentação de Clientes (RFM + K-Means):** Agrupamento de clientes em personas acionáveis (Ex: "Clientes Fiéis", "Super Clientes", "Clientes em Risco") com base no seu histórico de Recência, Frequência e Valor (RFM).
- **Previsão de Próxima Rota (RandomForest):** Um modelo que prevê, com alta acurácia (78%), qual será o próximo trecho de viagem que um cliente irá comprar.
- **Previsão de Recompra (XGBoost):** Um modelo para identificar clientes com alta probabilidade de realizar uma nova compra nos próximos 30 dias, otimizado para campanhas de marketing de grande alcance.
- **Dashboard Interativo (Streamlit):** Uma interface web para que as equipes de negócio possam explorar os segmentos de clientes e testar os modelos preditivos de forma intuitiva.

## Arquitetura da Solução

A arquitetura deste protótipo é focada em um fluxo de trabalho local, separando o treinamento dos modelos da sua visualização, o que garante performance e organização.

![Arquitetura do Protótipo Local](Arquitetura_NexTrip_AI.jpg)

## Tecnologias Utilizadas

- **Linguagem:** Python 3.9+
- **Análise e Manipulação de Dados:** Pandas
- **Modelagem de Machine Learning:** Scikit-learn, XGBoost
- **Dashboard Interativo:** Streamlit
- **Visualização de Dados:** Plotly
- **Serialização de Modelos:** Joblib

## Estrutura do Projeto

```
/projeto_clickbus/
|
|-- data/                             # Pasta para os dados de entrada e saída
|   |-- clientes_segmentados_rfm.csv (gerado por main.py)
|   |-- dashboard_clickbus.py         # Script para executar o dashboard interativo
|   |-- df_t.csv                      # Arquivo com os dados fornecidos para o projeto (não incluso no repositório por questões de segurança)
|
|-- models/                           # Pasta para os modelos treinados (gerados por main.py)
|   |-- modelo_recompra.joblib
|   |-- modelo_proxima_rota.joblib
|   |-- encoder_proxima_rota.joblib
|
|-- Arquitetura_NextTrip_AI.jpg       # Lista de dependências do projeto
|-- main.py                           # Script principal para treinar e salvar todos os modelos e a lista segmentada de clientes
|-- README.md                         # Documentação do projeto
|-- requirements.txt                  # Lista de dependências do projeto
```

## Como Executar o Projeto

Siga os passos abaixo para configurar e executar a solução em sua máquina local.

### 1. Pré-requisitos
- Python 3.9 ou superior
- Git

### 2. Clone o Repositório
```bash
git clone https://github.com/gscardini1/projeto_clickbus.git
cd projeto_clickbus
```

### 3. Crie e Ative um Ambiente Virtual
É uma boa prática isolar as dependências do projeto.
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 4. Instale as Dependências
```bash
pip install -r requirements.txt
```

### 5. Adicione os dados
```bash
Coloque o arquivo df_t.csv dentro da pasta data/.
```

### 6. Treine os Modelos
Execute o script `main.py` para processar os dados e treinar todos os modelos. Os artefatos serão salvos nas pastas `data/` e `models/`.
```bash
python main.py
```

### 7. Execute o Dashboard
Após o treinamento, inicie o dashboard interativo com o Streamlit.
```bash
cd data; streamlit run dashboard_clickbus.py
```
Uma nova aba abrirá em seu navegador com a aplicação web.