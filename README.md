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

1. **Regressão logística versus árvore de classificação:** treinamento com `Q1_training.csv`, avaliação em `Q1_test.csv` e discussão dos resultados.

2. **Bootstrap e bagging de árvores de regressão:** explicação dos métodos, implementação de bagging em California Housing e avaliação do ganho obtido com o método do subespaço aleatório.

3. **Random Forests:** aleatorização das variáveis candidatas em cada divisão, relação com o compromisso entre viés e variância e distinção entre a proposta de Breiman e o subespaço aleatório de Ho.

4. **Erro out-of-bag em regressão:** explicação do cálculo e comparação com o erro de teste em California Housing conforme aumenta o número de árvores.

5. **Importância das variáveis em Random Forests:** pesquisa sobre seu cálculo e construção de um exemplo ilustrativo.

### Aplicações práticas

6. **Churn bancário:** comparação de k-NN, Regressão Logística, Árvore de Classificação, Random Forest e Boosting, utilizando acurácia, curvas ROC e AUC.

7. **Preços de automóveis usados:** comparação de Regressão Linear Múltipla, Árvore de Regressão, Random Forest e Boosting, utilizando RMSE e gráficos de valores previstos versus observados.

## Estado atual

As duas aplicações práticas estão implementadas com explicações das escolhas metodológicas e avaliação em dados de teste.

## Ambiente das aplicações em R

O código principal das aplicações é desenvolvido em R, conforme o enunciado da APS:

- `aplicacoes.Rmd`: aplicação de churn bancário.
- `aplicacao_automoveis.Rmd`: aplicação de preços de automóveis usados.

O notebook `aplicacao_automoveis.ipynb` registra a exploração inicial e a tradução metodológica da segunda aplicação para R.

Pacotes utilizados:

- `tidyverse`
- `rsample`
- `class`
- `tree`
- `ranger`
- `gbm`
- `pROC`
- `knitr`
- `rmarkdown`

Para instalá-los no R:

```r
install.packages(c(
  "tidyverse", "rsample", "tree", "ranger", "gbm", "pROC",
  "knitr", "rmarkdown"
))
```
