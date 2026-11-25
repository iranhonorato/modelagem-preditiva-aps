# Execute na raiz do projeto: Rscript src/aps.R
# Dependências: install.packages(c("tree", "rpart", "class", "ranger", "gbm"))
for (pacote in c("tree", "rpart", "class", "ranger", "gbm")) {
  if (!requireNamespace(pacote, quietly = TRUE)) stop("Instale o pacote ", pacote)
}
saida <- file.path("docs", "resultados_aps")
dir.create(saida, recursive = TRUE, showWarnings = FALSE)

ler <- function(nome) {
  dados <- read.csv(file.path("data", paste0(nome, ".csv")), stringsAsFactors = TRUE)
  stopifnot(!anyNA(dados))
  dados
}
salvar <- function(dados, nome) {
  write.csv(dados, file.path(saida, paste0(nome, ".csv")), row.names = FALSE)
}
grafico <- function(nome, largura = 1200, altura = 780) {
  png(file.path(saida, paste0(nome, ".png")), largura, altura, res = 130)
}
rmse <- function(y, previsto) sqrt(mean((y - previsto)^2))
auc <- function(y, escore) {
  n <- sum(y == 1)
  (sum(rank(escore)[y == 1]) - n * (n + 1) / 2) / (n * sum(y == 0))
}
classificacao <- function(y, previsoes) {
  linhas <- lapply(names(previsoes), function(modelo) {
    escore <- previsoes[[modelo]]
    stopifnot(length(escore) == length(y), all(is.finite(escore)))
    positivo <- escore >= 0.5
    vp <- sum(positivo & y == 1)
    fp <- sum(positivo & y == 0)
    vn <- sum(!positivo & y == 0)
    fn <- sum(!positivo & y == 1)
    data.frame(modelo, acuracia = (vp + vn) / length(y),
               precisao = if (vp + fp > 0) vp / (vp + fp) else NA_real_,
               sensibilidade = vp / (vp + fn), auc = auc(y, escore),
               vn, fp, fn, vp)
  })
  do.call(rbind, linhas)
}
roc <- function(y, previsoes, nome) {
  cores <- c("#1A4968", "#AD451B", "#377343", "#70519D", "#7E6415")
  grafico(nome)
  plot(c(0, 1), c(0, 1), type = "n", asp = 1,
       xlab = "Taxa de falsos positivos", ylab = "Sensibilidade")
  abline(0, 1, col = "gray65", lty = 2)
  for (i in seq_along(previsoes)) {
    escore <- previsoes[[i]]
    ordem <- order(escore, decreasing = TRUE)
    fim_grupo <- c(which(diff(escore[ordem]) != 0), length(y))
    lines(c(0, cumsum(y[ordem] == 0)[fim_grupo] / sum(y == 0)),
          c(0, cumsum(y[ordem] == 1)[fim_grupo] / sum(y == 1)),
          col = cores[i], lwd = 2)
  }
  rotulos <- sprintf("%s (AUC %.3f)", names(previsoes),
                     vapply(previsoes, function(x) auc(y, x), numeric(1)))
  legend("bottomright", legend = rotulos, col = cores[seq_along(previsoes)],
         lwd = 2, bty = "n", cex = 0.83)
  invisible(dev.off())
}
dobras <- function(grupos) {
  fold <- integer(length(grupos))
  for (indices in split(seq_along(grupos), grupos)) {
    fold[indices] <- sample(rep(1:5, length.out = length(indices)))
  }
  fold
}
metade <- function(grupos) {
  indices <- split(seq_along(grupos), grupos)
  treino <- unlist(lapply(indices, function(i) sample(i, floor(length(i) / 2))),
                   use.names = FALSE)
  faltam <- floor(length(grupos) / 2) - length(treino)
  if (faltam > 0) treino <- c(treino, sample(setdiff(seq_along(grupos), treino), faltam))
  sort(treino)
}
faixas <- function(y) {
  cut(rank(y, ties.method = "first"), 10, labels = FALSE)
}
arvore_cv <- function(dados, formula, y, fold, tamanhos, metodo, avaliar) {
  probabilidades <- metodo == "class"
  controle <- function(m) rpart::rpart.control(cp = 0, minsplit = 2 * m,
                                               minbucket = m, xval = 0)
  pontuacao <- function(m) mean(vapply(1:5, function(k) {
    dentro <- fold != k
    arvore <- rpart::rpart(formula, data = dados[dentro, ], method = metodo,
                          parms = if (probabilidades) list(split = "information") else NULL,
                          control = controle(m))
    previsto <- predict(arvore, dados[!dentro, ], type = if (probabilidades) "prob" else "vector")
    avaliar(y[!dentro], if (probabilidades) previsto[, "1"] else previsto)
  }, numeric(1)))
  qualidade <- vapply(tamanhos, pontuacao, numeric(1))
  melhor <- which.max(qualidade)
  list(modelo = rpart::rpart(formula, data = dados, method = metodo,
                             parms = if (probabilidades) list(split = "information") else NULL,
                             control = controle(tamanhos[melhor])),
       cv = data.frame(minfolha = tamanhos, criterio = qualidade,
                       escolhido = seq_along(tamanhos) == melhor))
}

# Questão 1
q1 <- ler("Q1_training")
q1_teste <- ler("Q1_test")
logit <- glm(y ~ x1 + x2, data = q1, family = binomial())
radial <- glm(y ~ I(x1^2 + x2^2), data = q1, family = binomial())
q1$classe <- factor(q1$y)
arvore_q1 <- rpart::rpart(classe ~ x1 + x2, data = q1, method = "class",
                         parms = list(split = "information"),
                         control = rpart::rpart.control(cp = 0, minsplit = 2,
                                                         minbucket = 1, xval = 0))
escores_q1 <- function(dados) list(
  Logística = predict(logit, dados, type = "response"),
  Árvore = predict(arvore_q1, dados, type = "prob")[, "1"],
  'Logística radial' = predict(radial, dados, type = "response"))
q1_resultado <- classificacao(q1_teste$y, escores_q1(q1_teste))
q1_resultado$acuracia_treino <- classificacao(q1$y, escores_q1(q1))$acuracia
salvar(q1_resultado, "q1")
roc(q1_teste$y, escores_q1(q1_teste), "q1_roc")

# Questão 2. Cada coluna numérica codificada pertence a um dos nove preditores.
california <- ler("california")
set.seed(42)
cal_idx <- sample.int(nrow(california), floor(nrow(california) / 2))
cal_treino <- california[cal_idx, ]
cal_teste <- california[-cal_idx, ]
x_cal <- model.matrix(median_house_value ~ ., california)
grupo_colunas <- attr(x_cal, "assign")[-1]
x_cal <- as.data.frame(x_cal[, -1, drop = FALSE])
names(x_cal) <- make.names(names(x_cal), unique = TRUE)
y_cal <- california$median_house_value
ajustar_tree <- function(indices, colunas) {
  dados <- data.frame(valor = y_cal[indices], x_cal[indices, colunas, drop = FALSE])
  tree::tree(valor ~ ., dados, control = tree::tree.control(nrow(dados),
                                                             mincut = 10, minsize = 20, mindev = 0.001))
}
avaliar_cal <- function(modelo, colunas) {
  predict(modelo, x_cal[-cal_idx, colunas, drop = FALSE])
}
colunas <- seq_len(ncol(x_cal))
pred_unica <- avaliar_cal(ajustar_tree(cal_idx, colunas), colunas)
set.seed(43)
B <- 300L
m <- 6L
pred_bag <- pred_sub <- numeric(nrow(cal_teste))
bag <- data.frame(arvores = c(1, 5, 10, 25, 50, 100, B), bagging = NA_real_,
                  subespaco = NA_real_, unica = rmse(cal_teste$median_house_value, pred_unica))
for (b in seq_len(B)) {
  amostra <- sample(cal_idx, length(cal_idx), replace = TRUE)
  grupos <- sample(unique(grupo_colunas), m)
  subconjunto <- which(grupo_colunas %in% grupos)
  pred_bag <- pred_bag + avaliar_cal(ajustar_tree(amostra, colunas), colunas)
  pred_sub <- pred_sub + avaliar_cal(ajustar_tree(amostra, subconjunto), subconjunto)
  linha <- match(b, bag$arvores)
  if (!is.na(linha)) {
    bag$bagging[linha] <- rmse(cal_teste$median_house_value, pred_bag / b)
    bag$subespaco[linha] <- rmse(cal_teste$median_house_value, pred_sub / b)
  }
}
salvar(bag, "bagging")

# Questões 4 e 5. Mesma divisão usada no bagging.
floresta <- ranger::ranger(median_house_value ~ ., data = cal_treino, num.trees = B,
                           mtry = 3, min.node.size = 5, keep.inbag = TRUE,
                           replace = TRUE, sample.fraction = 1,
                           respect.unordered.factors = "partition",
                           importance = "permutation", seed = 42, num.threads = 1)
por_arvore <- function(dados) {
  predict(floresta, data = dados, predict.all = TRUE, num.threads = 1)$predictions
}
pred_treino <- por_arvore(cal_treino)
pred_teste <- por_arvore(cal_teste)
soma_oob <- votos <- numeric(nrow(cal_treino))
soma_teste <- numeric(nrow(cal_teste))
curva <- data.frame(arvores = seq_len(B), cobertura = NA_real_,
                    rmse_oob = NA_real_, rmse_teste = NA_real_)
for (b in curva$arvores) {
  fora <- floresta$inbag.counts[[b]] == 0
  soma_oob[fora] <- soma_oob[fora] + pred_treino[fora, b]
  votos[fora] <- votos[fora] + 1
  coberto <- votos > 0
  soma_teste <- soma_teste + pred_teste[, b]
  curva$cobertura[b] <- mean(coberto)
  curva$rmse_oob[b] <- rmse(cal_treino$median_house_value[coberto],
                            soma_oob[coberto] / votos[coberto])
  curva$rmse_teste[b] <- rmse(cal_teste$median_house_value, soma_teste / b)
}
stopifnot(isTRUE(all.equal(tail(curva$rmse_oob, 1), sqrt(floresta$prediction.error))),
          isTRUE(all.equal(soma_teste / B, predict(floresta, data = cal_teste)$predictions)))
salvar(curva[c(1, 5, 10, 50, 100, B), ], "oob")
grafico("erro_oob")
matplot(curva$arvores, curva[c("rmse_oob", "rmse_teste")], type = "l", lty = 1:2,
        col = c("#1A4968", "#AD451B"), lwd = 2,
        xlab = "Número de árvores", ylab = "RMSE (US$)")
legend("topright", c("OOB", "Teste"), col = c("#1A4968", "#AD451B"),
       lty = 1:2, lwd = 2, bty = "n")
invisible(dev.off())
importancia <- sort(floresta$variable.importance, decreasing = TRUE)
salvar(data.frame(variavel = names(importancia), aumento_mse = unname(importancia)),
       "importancia")

# Aplicação 1
churn <- ler("churn")
churn$y <- as.integer(churn$Exited == "Yes")
churn$Exited <- NULL
set.seed(42)
ch_idx <- metade(churn$y)
ch_treino <- churn[ch_idx, ]
ch_teste <- churn[-ch_idx, ]
produtos <- with(ch_treino, table(NumOfProducts, y))
salvar(data.frame(produtos = rownames(produtos), ficou = produtos[, "0"],
                  saiu = produtos[, "1"]), "churn_produtos")
ch_x <- model.matrix(y ~ ., churn)[, -1, drop = FALSE]
padronizado <- scale(ch_x[ch_idx, ])
ch_x_teste <- scale(ch_x[-ch_idx, ], attr(padronizado, "scaled:center"),
                    attr(padronizado, "scaled:scale"))
k <- 2L * round((sqrt(nrow(ch_treino)) - 1) / 2) + 1L
vizinho <- class::knn(padronizado, ch_x_teste, factor(ch_treino$y), k = k,
                      prob = TRUE, use.all = FALSE)
p_vencedor <- attr(vizinho, "prob")
p_vizinho <- ifelse(vizinho == "1", p_vencedor, 1 - p_vencedor)
ch_logit <- glm(y ~ ., data = ch_treino, family = binomial())
ch_class <- transform(ch_treino, y = factor(y))
ch_arvore <- arvore_cv(ch_class, y ~ ., ch_treino$y, dobras(ch_treino$y),
                       c(1, 5, 10, 20, 50, 100, 200), "class", auc)
ch_rf <- ranger::ranger(y ~ ., ch_class, probability = TRUE, num.trees = 500,
                        mtry = 3, seed = 42, num.threads = 1)
ch_gbm <- gbm::gbm(y ~ ., data = ch_treino, distribution = "bernoulli",
                   n.trees = 300, interaction.depth = 3, shrinkage = 0.05,
                   bag.fraction = 0.5, n.cores = 1, verbose = FALSE)
ch_escores <- list('k-NN' = p_vizinho,
                   Logística = predict(ch_logit, ch_teste, type = "response"),
                   Árvore = predict(ch_arvore$modelo, ch_teste, type = "prob")[, "1"],
                   'Random Forest' = predict(ch_rf, ch_teste)$predictions[, "1"],
                   'Gradient Boosting' = predict(ch_gbm, ch_teste, n.trees = 300,
                                                  type = "response"))
salvar(classificacao(ch_teste$y, c(ch_escores, list(Baseline = rep(0, nrow(ch_teste))))),
       "churn")
ch_cv <- ch_arvore$cv
ch_cv$folhas <- sum(ch_arvore$modelo$frame$var == "<leaf>")
salvar(ch_cv, "churn_cv")
roc(ch_teste$y, ch_escores, "churn_roc")

# Aplicação 2
carros <- ler("used_cars")
set.seed(42)
ca_idx <- metade(faixas(carros$price))
ca_treino <- carros[ca_idx, ]
ca_teste <- carros[-ca_idx, ]
ca_linear <- lm(price ~ ., data = ca_treino)
ca_arvore <- arvore_cv(ca_treino, price ~ ., ca_treino$price,
                       dobras(faixas(ca_treino$price)), c(1, 2, 5, 10, 20, 50, 100),
                       "anova", function(y, previsto) -rmse(y, previsto))
ca_rf <- ranger::ranger(price ~ ., ca_treino, num.trees = 500, mtry = 3,
                        seed = 42, num.threads = 1)
ca_gbm <- gbm::gbm(price ~ ., data = ca_treino, distribution = "gaussian",
                   n.trees = 500, interaction.depth = 4, shrinkage = 0.05,
                   bag.fraction = 0.5, n.cores = 1, verbose = FALSE)
ca_prev <- list('Regressão linear' = predict(ca_linear, ca_teste),
                Árvore = predict(ca_arvore$modelo, ca_teste),
                'Random Forest' = predict(ca_rf, ca_teste)$predictions,
                'Gradient Boosting' = predict(ca_gbm, ca_teste, n.trees = 500))
observado <- ca_teste$price
ca_resultado <- do.call(rbind, lapply(names(ca_prev), function(modelo) {
  previsto <- ca_prev[[modelo]]
  data.frame(modelo, rmse = rmse(observado, previsto),
             mae = mean(abs(observado - previsto)),
             r2 = 1 - sum((observado - previsto)^2) / sum((observado - mean(observado))^2),
             negativas = sum(previsto < 0))
}))
salvar(ca_resultado, "carros")
ca_cv <- ca_arvore$cv
ca_cv$criterio <- -ca_cv$criterio
ca_cv$folhas <- sum(ca_arvore$modelo$frame$var == "<leaf>")
salvar(ca_cv, "carros_cv")
grafico("carros_previsto", altura = 1000)
par(mfrow = c(2, 2), mar = c(4, 4, 2.5, 1))
limites <- range(c(observado, unlist(ca_prev)))
for (modelo in names(ca_prev)) {
  previsto <- ca_prev[[modelo]]
  plot(observado, previsto, xlim = limites, ylim = limites, asp = 1,
       xlab = "Preço observado", ylab = "Preço previsto",
       main = sprintf("%s | RMSE %.3f", modelo, rmse(observado, previsto)),
       pch = 16, cex = 0.5, col = adjustcolor("#1A4968", alpha.f = 0.25))
  abline(0, 1, lty = 2, col = "#AD451B", lwd = 2)
}
invisible(dev.off())
