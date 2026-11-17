# QUESTÃO 4

# Erro "out-of-bag" (OOB) "versus" erro de teste conforme aumenta-se o número de árvores

library(tidyverse)
library(rsample)
library(ranger)

theme_set(theme_bw())

california <- read_csv("california.csv", show_col_types = FALSE) |>
  mutate(ocean_proximity = as_factor(ocean_proximity)) |>
  rename(y = median_house_value)

set.seed(42)

split <- initial_split(california, prop = 0.5)

trn <- training(split)
tst <- testing(split)

### Uma única floresta com "B_max" árvores, guardando quem ficou fora de cada amostra "bootstrap"
B_max <- 500

rf <- ranger(y ~ ., data = trn, num.trees = B_max, keep.inbag = TRUE)

# inbag[i, b] = número de vezes que a observação "i" entrou na amostra "bootstrap" da árvore "b"
inbag <- simplify2array(rf$inbag.counts)
oob <- inbag == 0

mean(oob) # proporção de pares (observação, árvore) fora da amostra: deve ficar perto de e^(-1) = 0.368

# Previsões de cada árvore, separadamente (matrizes n x B_max)
T_trn <- predict(rf, data = trn, predict.all = TRUE)$predictions
T_tst <- predict(rf, data = tst, predict.all = TRUE)$predictions

### Cálculo do erro OOB "na mão" para as florestas formadas pelas "B" primeiras árvores
# Previsão OOB da observação "i": média das previsões das árvores b <= B em que "i" ficou fora.
soma_oob <- t(apply(T_trn * oob, 1, cumsum)) # soma acumulada das previsões OOB
n_oob    <- t(apply(oob, 1, cumsum))         # número acumulado de árvores OOB
soma_tst <- t(apply(T_tst, 1, cumsum))       # soma acumulada das previsões no teste

grade_B <- 1:B_max

curvas <- map_dfr(grade_B, \(B) {
  tem_oob <- n_oob[, B] > 0
  y_hat_oob <- soma_oob[tem_oob, B] / n_oob[tem_oob, B]
  y_hat_tst <- soma_tst[, B] / B
  tibble(
    B = B,
    rmse_oob = sqrt(mean((trn$y[tem_oob] - y_hat_oob)^2)),
    rmse_teste = sqrt(mean((tst$y - y_hat_tst)^2)),
    frac_sem_oob = mean(!tem_oob) # observações ainda sem previsão OOB
  )
})

# Conferência: com todas as árvores, o cálculo manual coincide com o erro OOB do ranger
c(manual = curvas$rmse_oob[B_max]^2, ranger = rf$prediction.error)

# Tabela 
curvas |>
  filter(B %in% c(1, 2, 5, 10, 25, 50, 100, 200, 300, 500)) |>
  mutate(diferenca = rmse_oob - rmse_teste) |>
  print(n = Inf)

### RMSE OOB "versus" RMSE de teste em função do número de árvores
curvas |>
  select(B, OOB = rmse_oob, Teste = rmse_teste) |>
  pivot_longer(-B, names_to = "erro", values_to = "rmse") |>
  ggplot(aes(x = B, y = rmse, color = erro)) +
  geom_line(linewidth = 1) +
  scale_x_log10(breaks = c(1, 2, 5, 10, 25, 50, 100, 200, 500)) +
  scale_y_continuous(labels = scales::dollar_format()) +
  scale_color_manual(values = c(OOB = "dark orange", Teste = "steelblue")) +
  labs(x = "Número de árvores da floresta (escala log)", y = "RMSE", color = NULL,
       title = "California housing: erro out-of-bag x erro de teste")

### Diferença entre os dois erros
ggplot(curvas, aes(x = B, y = rmse_oob - rmse_teste)) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  geom_line(linewidth = 1, color = "dark green") +
  scale_x_log10(breaks = c(1, 2, 5, 10, 25, 50, 100, 200, 500)) +
  scale_y_continuous(labels = scales::dollar_format()) +
  labs(x = "Número de árvores da floresta (escala log)", y = "RMSE OOB − RMSE de teste")
