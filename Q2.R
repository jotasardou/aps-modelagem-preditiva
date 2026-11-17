# QUESTAO 2

dados <- read.csv("california.csv", stringsAsFactors = TRUE)

# Remoção de observações com dados faltantes (e.g., NAs em total_bedrooms)
dados <- na.omit(dados)
dados$ocean_proximity <- as.factor(dados$ocean_proximity)

str(dados)
cat("Total de observacoes validas:", nrow(dados), "\n")

# Divisão 50/50 em Treinamento e Teste
set.seed(123)
n <- nrow(dados)
idx_treino <- sample(1:n, size = floor(0.5 * n), replace = FALSE)

treino <- dados[idx_treino, ]
teste  <- dados[-idx_treino, ]
n_treino <- nrow(treino)
n_teste  <- nrow(teste)

# Modelo Baseline: Uma única árvore de regressão
arvore_unica <- tree(median_house_value ~ ., data = treino)
summary(arvore_unica)

pred_unica <- predict(arvore_unica, newdata = teste)
rmse_unica <- sqrt(mean((teste$median_house_value - pred_unica)^2))
cat("RMSE de Teste - Arvore de Regressao Unica:", round(rmse_unica, 2), "\n")

# Árvores profundas para o "bagging" (sem poda e sem o critério de parada padrão do tree()).
# O padrão do tree() (mindev = 0.01, minsize = 10) gera arvores rasas (9 folhas aqui),
# com viés alto: o "bagging" apenas reduz variância, logo quase não se ganha nada com elas.
# mindev = 0 desliga a parada por redução de "deviance"; minsize = 10 / mincut = 5
# deixam folhas com pelo menos 5 observações (e.g., min.node.size = 5 do ranger).
# Se aparecer o erro "maximum depth reached", aumente minsize (ex.: 20) e mincut (ex.: 10).
ctrl_profunda <- tree.control(nobs = n_treino, mincut = 5, minsize = 10, mindev = 0)

arvore_profunda <- tree(median_house_value ~ ., data = treino, control = ctrl_profunda)
pred_profunda <- predict(arvore_profunda, newdata = teste)
rmse_profunda <- sqrt(mean((teste$median_house_value - pred_profunda)^2))
cat("RMSE de Teste - Arvore de Regressao Unica PROFUNDA:", round(rmse_profunda, 2), "\n")

# "Bagging" de Árvores de Regressão
# Com árvores profundas, o RMSE do "bagging" estabiliza bem antes de 200 árvores.
B <- 200

soma_pred_bag <- numeric(n_teste) # soma acumulada das previsões (mais rápido que guardar a matriz)
rmse_acum_bag <- numeric(B)

set.seed(123)
inicio <- Sys.time()
pb <- txtProgressBar(min = 0, max = B, style = 3)
for (b in 1:B) {
  # Sorteio da amostra "bootstrap" (com reposição) do conjunto de treinamento
  idx_boot <- sample(1:n_treino, size = n_treino, replace = TRUE)
  
  # Ajuste da árvore de regressão profunda na amostra "bootstrap" "b"
  mod_b <- tree(median_house_value ~ ., data = treino[idx_boot, ], control = ctrl_profunda)
  
  # Previsão no conjunto de teste e RMSE acumulado usando as primeiras "b" árvores
  soma_pred_bag <- soma_pred_bag + predict(mod_b, newdata = teste)
  rmse_acum_bag[b] <- sqrt(mean((teste$median_house_value - soma_pred_bag / b)^2))
  setTxtProgressBar(pb, b)
}
close(pb)
print(Sys.time() - inicio)

rmse_bag_final <- rmse_acum_bag[B]
cat("RMSE de Teste - Bagging (B =", B, "):", round(rmse_bag_final, 2), "\n")

# Método do Subespaço Aleatório (avaliando diferentes tamanhos m < p)
vars_explicativas <- setdiff(names(treino), "median_house_value")
p <- length(vars_explicativas) # p = 9 variáveis explicativas

valores_m <- c(3, 5, 6, 7, 8)
B_sub <- 200

resultados_subespaco <- data.frame(
  m = valores_m,
  RMSE_Subespaco_Puro = NA,
  RMSE_Bagging_Subespaco = NA
)

# No subespaço puro (amostra de treino completa), duas árvores com o mesmo subconjunto
# de variáveis são idênticas. Guardamos a previsão de cada subconjunto já ajustado
# para não ajustar a mesma árvore de novo (o resultado é o mesmo, apenas fica mais rápido).
cache_puro <- new.env()

# Com poucas variáveis (m pequeno), a árvore profunda pode passar do limite de
# profundidade do pacote tree ("maximum depth reached"). Aqui, tentamos de novo
# com folhas um pouco maiores (minsize 20, 40, 80...), até a árvore caber no limite.
# Guardamos quantas vezes isso foi necessário.
n_ajustes_folha <- 0
arvore_profunda_segura <- function(formula, dados) {
  for (tam in c(10, 20, 40, 80, 160)) {
    ctrl <- tree.control(nobs = nrow(dados), mincut = tam / 2, minsize = tam, mindev = 0)
    ajuste <- tryCatch(tree(formula, data = dados, control = ctrl),
                       error = function(e) NULL)
    if (!is.null(ajuste)) {
      if (tam > 10) n_ajustes_folha <<- n_ajustes_folha + 1
      return(ajuste)
    }
  }
  stop("Não foi possível ajustar a árvore nem com minsize = 160.")
}

inicio <- Sys.time()
for (i in seq_along(valores_m)) {
  m_atual <- valores_m[i]
  soma_puro <- numeric(n_teste)
  soma_bag_sub <- numeric(n_teste)
  
  cat("\nm =", m_atual, "\n")
  pb <- txtProgressBar(min = 0, max = B_sub, style = 3)
  set.seed(123)
  for (b in 1:B_sub) {
    # Sorteio do subespaço aleatório de "m" variáveis explicativas para a árvore "b"
    vars_sorteadas <- sample(vars_explicativas, size = m_atual, replace = FALSE)
    formula_sub <- as.formula(paste("median_house_value ~", paste(vars_sorteadas, collapse = " + ")))
    
    # Subespaço Aleatório puro (amostra de treino completa, sorteando variáveis por árvore)
    chave <- paste(sort(vars_sorteadas), collapse = "|")
    if (is.null(cache_puro[[chave]])) {
      mod_sub_puro <- arvore_profunda_segura(formula_sub, treino)
      cache_puro[[chave]] <- predict(mod_sub_puro, newdata = teste)
    }
    soma_puro <- soma_puro + cache_puro[[chave]]
    
    # "Bagging" + Subespaço Aleatório (amostra "bootstrap" + sorteio de variáveis por árvore)
    idx_boot <- sample(1:n_treino, size = n_treino, replace = TRUE)
    mod_bag_sub <- arvore_profunda_segura(formula_sub, treino[idx_boot, ])
    soma_bag_sub <- soma_bag_sub + predict(mod_bag_sub, newdata = teste)
    
    setTxtProgressBar(pb, b)
  }
  close(pb)
  
  resultados_subespaco$RMSE_Subespaco_Puro[i]    <- round(sqrt(mean((teste$median_house_value - soma_puro / B_sub)^2)), 2)
  resultados_subespaco$RMSE_Bagging_Subespaco[i] <- round(sqrt(mean((teste$median_house_value - soma_bag_sub / B_sub)^2)), 2)
}
print(Sys.time() - inicio)
cat("Árvores que precisaram de folhas maiores (minsize > 10):", n_ajustes_folha, "\n")

# Resultados e Gráfico Comparativo
tabela_q2 <- data.frame(
  Metodo = c("Árvore de Regressao Unica (padrao do tree)",
             "Árvore de Regressao Unica (profunda)",
             paste0("Bagging Padrão (m = p = ", p, ")"),
             paste0("Bagging + Subespaço Aleatório (m = ", resultados_subespaco$m, ")"),
             paste0("Subespaço Aleatório Puro (m = ", resultados_subespaco$m, ")")),
  RMSE_Teste = c(round(rmse_unica, 2),
                 round(rmse_profunda, 2),
                 round(rmse_bag_final, 2),
                 resultados_subespaco$RMSE_Bagging_Subespaco,
                 resultados_subespaco$RMSE_Subespaco_Puro)
)
print(tabela_q2)

# Guarda os resultados 
saveRDS(list(tabela_q2 = tabela_q2, rmse_acum_bag = rmse_acum_bag,
             rmse_unica = rmse_unica, rmse_profunda = rmse_profunda),
        "resultados_q2.rds")

# Grafico da convergência do RMSE do "Bagging" conforme "B" cresce
plot(1:B, rmse_acum_bag, type = "l", col = "#0072B2", lwd = 2,
     ylim = range(c(rmse_acum_bag, rmse_unica, rmse_profunda)),
     xlab = "Número de Árvores de Regressão (B)",
     ylab = "RMSE no Conjunto de Teste",
     main = "Evolução do Erro de Teste: Árvore Única vs. Bagging")
abline(h = rmse_unica, col = "#D55E00", lwd = 2, lty = 2)
abline(h = rmse_profunda, col = "#009E73", lwd = 2, lty = 3)
legend("right",
       legend = c("Bagging de Árvores de Regressão (profundas)",
                  "Árvore de Regressão Única (padrão) - Ref. fixa",
                  "Árvore de Regressão Única (profunda) - Ref. fixa"),
       col = c("#0072B2", "#D55E00", "#009E73"), lwd = 2, lty = c(1, 2, 3))
