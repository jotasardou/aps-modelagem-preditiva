# APS MODELAGEM PREDITIVA

# QUESTAO 1 ----

install.packages('tree')
library(tree)

install.packages('pROC')
library(pROC)

install.packages('ggplot2')
library(ggplot2)


train <- read.csv2("data/Q1_training.csv")
test  <- read.csv2("data/Q1_test.csv")

# Versao fator de y para a arvore de classificacao
train$y_fac <- factor(train$y, levels = c(0, 1))
test$y_fac  <- factor(test$y, levels = c(0, 1))

# Analise Exploratoria Visual

ggplot(train, aes(x = x1, y = x2, color = y_fac)) +
  geom_point(alpha = 0.7, size = 2) +
  scale_color_manual(values = c("0" = "#0072B2", "1" = "#D55E00"), name = "Classe (y)") +
  labs(title = "Conjunto de Treinamento: Distribuição no Plano (x1, x2)",
       x = "x1", y = "x2") +
  theme_bw()

# Modelo 1: Regressao Logistica
mod_logit <- glm(y ~ x1 + x2, data = train, family = binomial(link = "logit"))
summary(mod_logit)

prob_logit <- predict(mod_logit, newdata = test, type = "response")
pred_logit <- ifelse(prob_logit >= 0.5, 1, 0)

tab_logit <- table(Previsto = factor(pred_logit, levels = c(0, 1)),
                   Observado = factor(test$y, levels = c(0, 1)))
print(tab_logit)

acc_logit  <- mean(pred_logit == test$y)
sens_logit <- sum(pred_logit == 1 & test$y == 1) / sum(test$y == 1)
esp_logit  <- sum(pred_logit == 0 & test$y == 0) / sum(test$y == 0)

# Modelo 2: Arvore de Classificacao

mod_tree <- tree(y_fac ~ x1 + x2, data = train)
summary(mod_tree)

# Gráfico da Árvore de Classificação
plot(mod_tree)
text(mod_tree, pretty = 2, cex = 0.8, font = 2)
title("Árvore de Classificação - Treinamento")

# Particao do espaco (x1, x2)
partition.tree(mod_tree, label = "y_fac",
               main = "Partição do Espaço (x1, x2) pela Árvore de Classificação", font = 2)
points(train$x1, train$x2, col = ifelse(train$y == 1, "#D55E00", "#0072B2"), pch = 19, cex = 0.7)

prob_tree <- predict(mod_tree, newdata = test, type = "vector")[, "1"]
pred_tree <- ifelse(prob_tree >= 0.5, 1, 0)

tab_tree <- table(Previsto = factor(pred_tree, levels = c(0, 1)),
                  Observado = factor(test$y, levels = c(0, 1)))
print(tab_tree)

acc_tree  <- mean(pred_tree == test$y)
sens_tree <- sum(pred_tree == 1 & test$y == 1) / sum(test$y == 1)
esp_tree  <- sum(pred_tree == 0 & test$y == 0) / sum(test$y == 0)

# Curvas ROC e AUC

roc_logit <- roc(test$y, prob_logit, quiet = TRUE)
roc_tree  <- roc(test$y, prob_tree, quiet = TRUE)

plot(roc_tree, col = "#0072B2", lwd = 2, main = "Curvas ROC no Conjunto de Teste")
plot(roc_logit, col = "#D55E00", lwd = 2, add = TRUE)
legend("bottomright",
       legend = c(paste0("Árvore de Classificação (AUC = ", round(auc(roc_tree), 4), ")"),
                  paste0("Regressão Logística (AUC = ", round(auc(roc_logit), 4), ")")),
       col = c("#0072B2", "#D55E00"), lwd = 2)

# Tabela Resumo de Comparacao no Teste

resumo_q1 <- data.frame(
  Modelo          = c("Regressao Logistica", "Arvore de Classificacao"),
  Acuracia        = round(c(acc_logit, acc_tree), 4),
  Erro_Teste      = round(1 - c(acc_logit, acc_tree), 4),
  Sensibilidade   = round(c(sens_logit, sens_tree), 4),
  Especificidade  = round(c(esp_logit, esp_tree), 4),
  AUC             = round(c(as.numeric(auc(roc_logit)), as.numeric(auc(roc_tree))), 4)
)
print(resumo_q1)

