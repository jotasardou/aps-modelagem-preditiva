# APS - Modelagem Preditiva Básica

Projeto colaborativo desenvolvido para a Atividade Prática Supervisionada da disciplina de Modelagem Preditiva Básica

Professor:

Paulo C. Marques F.

Alunos:  

João Pedro Sardou de Oliveira
Pietra Rosa de Jesus Araújo
Sávio da Silva de Souza
Silas André Pereira de Brito


## Conteúdo

### Parte teórica e experimental

1. **Regressão logística versus árvore de classificação:** treinamento com `Q1_training.csv`, avaliação em `Q1_test.csv` e discurssão dos resultados.

2. **Bootstrap e bagging de árvores de regressão:** explicação dos métodos, implementação de bagging em California Housing e avaliação do ganho obtido com o método do subespaço aleatório.

3. **Random FOrests:** aleatorização das variáveis candidatas em cada divisão, relação com o compromisso entre viés e variÂncia e distinção entre a proposta de Breiman e o subespaço aleatório de Ho.

4. **Erro out-of-bag em regressão:** explicação do cálculo e comparação com o erro de teste em California Housing conforme aumenta o número de árvores.

5. **Importância das variáveis em Random Forests:** pesquisa sobre seu cálculo e construção de um exemplo ilustrativo.

### Aplicações práticas

6. **Churn bancário** comparação de k-NN, Regressão Logística, Árvore de Classificação, Random Forest e Boosting, utilizando acurácia, curvas ROC e AUC.

7. **Preços de automóveis usados:** comparação de Regressão Linear Múltipla, Árvore de Regressão, Random Forest e Boosting, utilizando RMSE e gráficos de valores previstos versus observados.

## Estado atual

Em desenvolvimento. As aplicações práticas estão sendo construídas com explicações das escolhas metodológicas e avaliação em dados de teste.

## Ambiente das aplicações em Python

Python 3.12. Dependências registradas em `requirements.txt`

Para instalar as depêndencias do projeto: `python -m pip install -r requirements.txt`
