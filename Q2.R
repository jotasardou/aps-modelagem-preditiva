# QUESTAO 2 ----

dados <- read.csv("data/california.csv", stringsAsFactors = TRUE)

# Remocao de observacoes com dados faltantes (NAs em total_bedrooms)
dados <- na.omit(dados)
dados$ocean_proximity <- as.factor(dados$ocean_proximity)

str(dados)
cat("Total de observacoes validas:", nrow(dados), "\n")

# Divisao 50/50 em Treinamento e Teste
set.seed(123)
n <- nrow(dados)
idx_treino <- sample(1:n, size = floor(0.5 * n), replace = FALSE)

treino <- dados[idx_treino, ]
teste  <- dados[-idx_treino, ]
n_treino <- nrow(treino)
n_teste  <- nrow(teste)

# Modelo Baseline: uma unica arvore de regressao
arvore_unica <- tree(median_house_value ~ ., data = treino)
summary(arvore_unica)

pred_unica <- predict(arvore_unica, newdata = teste)
rmse_unica <- sqrt(mean((teste$median_house_value - pred_unica)^2))
cat("RMSE de Teste - Arvore de Regressao Unica:", round(rmse_unica, 2), "\n")

# Bagging de Arvores de Regressao (B = 500 arvores)
B <- 500
preds_bag <- matrix(NA, nrow = n_teste, ncol = B)
rmse_acum_bag <- numeric(B)

set.seed(123)
for (b in 1:B) {
  # Sorteio da amostra bootstrap (com reposicao) do conjunto de treinamento
  idx_boot <- sample(1:n_treino, size = n_treino, replace = TRUE)
  treino_boot <- treino[idx_boot, ]
  
  # Ajuste da arvore de regressao na amostra bootstrap b
  mod_b <- tree(median_house_value ~ ., data = treino_boot)
  
  # Previsao no conjunto de teste
  preds_bag[, b] <- predict(mod_b, newdata = teste)
  
  # RMSE acumulado usando as primeiras b arvores de regressao
  pred_media_b <- rowMeans(preds_bag[, 1:b, drop = FALSE])
  rmse_acum_bag[b] <- sqrt(mean((teste$median_house_value - pred_media_b)^2))
}

rmse_bag_final <- rmse_acum_bag[B]
cat("RMSE de Teste - Bagging (B =", B, "):", round(rmse_bag_final, 2), "\n")

# Metodo do Subespaco Aleatorio (avaliando diferentes tamanhos m < p)
vars_explicativas <- setdiff(names(treino), "median_house_value")
p <- length(vars_explicativas) # p = 9 variaveis explicativas

# Testando subespacos de tamanho m = 3, 5, 6, 7 e 8 variaveis por arvore

valores_m <- c(3, 5, 6, 7, 8)
resultados_subespaco <- data.frame(
  m = valores_m,
  RMSE_Subespaco_Puro = NA,
  RMSE_Bagging_Subespaco = NA
)

for (i in seq_along(valores_m)) {
  m_atual <- valores_m[i]
  preds_sub_puro <- matrix(NA, nrow = n_teste, ncol = B)
  preds_bag_sub  <- matrix(NA, nrow = n_teste, ncol = B)
  
  set.seed(123)
  for (b in 1:B) {
    # Sorteio do subespaco aleatorio de m variaveis explicativas para a arvore b
    vars_sorteadas <- sample(vars_explicativas, size = m_atual, replace = FALSE)
    formula_sub <- as.formula(paste("median_house_value ~", paste(vars_sorteadas, collapse = " + ")))
    
    # (a) Subespaco Aleatorio puro (amostra de treino completa, sorteando variaveis por arvore)
    mod_sub_puro <- tree(formula_sub, data = treino)
    preds_sub_puro[, b] <- predict(mod_sub_puro, newdata = teste)
    
    # (b) Bagging + Subespaco Aleatorio (amostra bootstrap + sorteio de variaveis por arvore)
    idx_boot <- sample(1:n_treino, size = n_treino, replace = TRUE)
    mod_bag_sub <- tree(formula_sub, data = treino[idx_boot, ])
    preds_bag_sub[, b] <- predict(mod_bag_sub, newdata = teste)
  }
  
  pred_media_sub_puro <- rowMeans(preds_sub_puro)
  pred_media_bag_sub  <- rowMeans(preds_bag_sub)
  
  resultados_subespaco$RMSE_Subespaco_Puro[i]    <- round(sqrt(mean((teste$median_house_value - pred_media_sub_puro)^2)), 2)
  resultados_subespaco$RMSE_Bagging_Subespaco[i] <- round(sqrt(mean((teste$median_house_value - pred_media_bag_sub)^2)), 2)
}

# Resultados e Grafico Comparativo

tabela_q2 <- data.frame(
  Metodo = c("Árvore de Regressao Unica",
             paste0("Bagging Padrão (m = p = ", p, ")"),
             paste0("Bagging + Subespaço Aleatório (m = ", resultados_subespaco$m, ")"),
             paste0("Subespaço Aleatório Puro (m = ", resultados_subespaco$m, ")")),
  RMSE_Teste = c(round(rmse_unica, 2),
                 round(rmse_bag_final, 2),
                 resultados_subespaco$RMSE_Bagging_Subespaco,
                 resultados_subespaco$RMSE_Subespaco_Puro)
)
print(tabela_q2)

# Grafico da convergencia do RMSE do Bagging conforme B cresce
plot(1:B, rmse_acum_bag, type = "l", col = "#0072B2", lwd = 2,
     ylim = range(c(rmse_acum_bag, rmse_unica)),
     xlab = "Número de Árvores de Regressão (B)",
     ylab = "RMSE no Conjunto de Teste",
     main = "Evolução do Erro de Teste: Árvore Única vs. Bagging")
abline(h = rmse_unica, col = "#D55E00", lwd = 2, lty = 2)
legend("right",
       legend = c("Bagging de Árvores de Regressão", "Árvore de Regressão Única"),
       col = c("#0072B2", "#D55E00"), lwd = 2, lty = c(1, 2))
