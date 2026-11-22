# Execute na raiz do projeto com Rscript src/random_forest.R
# Dependência: install.packages("ranger")
library(ranger)
saida <- "src/markdown/figuras_random_forest"
dir.create(saida, recursive = TRUE, showWarnings = FALSE)
dados <- read.csv("data/california.csv")
dados$ocean_proximity <- factor(dados$ocean_proximity)
stopifnot(!anyNA(dados))
set.seed(42)
indices <- sample.int(nrow(dados), floor(nrow(dados) / 2))
treino <- dados[indices, ]
teste <- dados[-indices, ]

# Uma única floresta permite comparar as primeiras B árvores de cada vez.
floresta <- ranger(median_house_value ~ ., data = treino, num.trees = 300,
                   mtry = 3, min.node.size = 5, keep.inbag = TRUE,
                   replace = TRUE, sample.fraction = 1,
                   respect.unordered.factors = "partition",
                   importance = "permutation", seed = 42, num.threads = 1)
por_arvore <- function(dados) {
  predict(floresta, data = dados, predict.all = TRUE, num.threads = 1)$predictions
}
pred_treino <- por_arvore(treino)
pred_teste <- por_arvore(teste)
soma_oob <- votos <- numeric(nrow(treino))
soma_teste <- numeric(nrow(teste))
curva <- data.frame(arvores = seq_len(floresta$num.trees), cobertura = NA_real_,
                    rmse_oob = NA_real_, rmse_teste = NA_real_)
for (b in curva$arvores) {
  fora <- floresta$inbag.counts[[b]] == 0
  soma_oob[fora] <- soma_oob[fora] + pred_treino[fora, b]
  votos[fora] <- votos[fora] + 1
  coberto <- votos > 0
  soma_teste <- soma_teste + pred_teste[, b]
  curva$cobertura[b] <- mean(coberto)
  curva$rmse_oob[b] <- sqrt(mean((treino$median_house_value[coberto] -
                                  soma_oob[coberto] / votos[coberto])^2))
  curva$rmse_teste[b] <- sqrt(mean((teste$median_house_value - soma_teste / b)^2))
}
stopifnot(isTRUE(all.equal(tail(curva$rmse_oob, 1), sqrt(floresta$prediction.error))),
          isTRUE(all.equal(soma_teste / floresta$num.trees,
                           predict(floresta, data = teste)$predictions)))
write.csv(curva, file.path(saida, "erros.csv"), row.names = FALSE)
png(file.path(saida, "erro_oob.png"), width = 1000, height = 550, res = 120)
matplot(curva$arvores, curva[c("rmse_oob","rmse_teste")], type = "l", lty = 1:2,
        col = c("steelblue4","darkorange3"), lwd = 2,
        xlab = "Número de árvores", ylab = "RMSE (US$)")
legend("topright", c("Treino fora da amostra (OOB)", "Teste"),
       col = c("steelblue4","darkorange3"), lty = 1:2, lwd = 2, bty = "n")
invisible(dev.off())
importancia <- sort(floresta$variable.importance, decreasing = TRUE)
write.csv(data.frame(variavel = names(importancia), aumento_mse = unname(importancia)),
          file.path(saida, "importancia.csv"), row.names = FALSE)
print(curva[c(1,5,10,50,100,300), ], row.names = FALSE)
print(round(importancia))
