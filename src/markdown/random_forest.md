# Random Forest na regressão dos preços dos imóveis

## Item 3. Como a floresta escolhe um corte

Imagine uma árvore que prevê o preço mediano dos imóveis de um distrito. No bagging, cada árvore recebe uma amostra bootstrap de distritos. Se a renda for um preditor forte, várias árvores poderão começar pelo mesmo corte de renda. Quando todas cometem erros parecidos nos mesmos distritos, tirar a média corrige pouco esses erros.

No algoritmo de Breiman com sorteio de atributos, a árvore recebe uma amostra bootstrap e, a cada nó, sorteia sem reposição `mtry` variáveis candidatas. Entre os cortes possíveis nessas variáveis, escolhe aquele que mais reduz a soma dos quadrados dos resíduos. Cada folha prevê a média dos preços de treino que chegaram até ela. A floresta calcula a média das previsões das árvores. [Breiman, *Random Forests*, 2001](https://doi.org/10.1023/A:1010933404324)

Considere quatro preditores: renda, latitude, idade e população. Com duas candidatas por nó, um sorteio na raiz pode oferecer renda e idade. A árvore escolhe renda. No ramo de menor renda, um novo sorteio pode oferecer latitude e população, permitindo separar distritos pela localização. No método de Ho, se a árvore recebe renda e idade, apenas essas duas variáveis ficam disponíveis ao longo de todos os seus ramos. Assim, **Breiman não se limitou a aplicar o subespaço de Ho**. O momento do sorteio muda inclusive quais interações uma árvore consegue representar. [Ho, *The Random Subspace Method for Constructing Decision Forests*, 1998](https://doi.org/10.1109/34.709601)

Para um distrito fixo, suponha que cada previsão tenha variância σ² e que os pares de previsões tenham correlação ρ nas repetições do treinamento. Somando as B variâncias e as B(B−1) covariâncias, a variância da média fica **σ²[ρ + (1−ρ)/B]**. Por exemplo, com σ² = 100 e B = 100, reduzir ρ de 0,8 para 0,3 reduz essa variância de 80,2 para 30,7. A hipótese de variâncias iguais serve para isolar o efeito da correlação.

Há um custo possível: quando uma variável decisiva fica fora do sorteio, a árvore pode escolher um corte inferior. Se isso acontece repetidamente, a previsão média pode se afastar do valor esperado do imóvel e o viés aumentar. Como o erro quadrático esperado inclui viés ao quadrado, variância e ruído, reduzir `mtry` só melhora o erro total se o ganho compensar esse custo.

## Item 4. Como calcular o erro out-of-bag

Em um bootstrap de n sorteios com reposição, a chance de um distrito específico ficar de fora é (1−1/n)ⁿ, aproximadamente 36,8% para n grande. Essas árvores podem prever o distrito sem terem usado seu preço no ajuste. O conjunto de árvores elegíveis muda de um distrito para outro.

**Exemplo para desenhar no relatório.** Imagine três árvores e dois distritos com preços em milhares de dólares. Um traço indica que a árvore recebeu aquele distrito no bootstrap.

| Distrito | Preço observado | Árvore 1 | Árvore 2 | Árvore 3 | Média OOB |
|---|---:|---:|---:|---:|---:|
| A | 100 | 90 | — | 110 | 100 |
| B | 160 | — | 140 | — | 140 |

Primeiro tiramos a média das previsões elegíveis **dentro de cada linha**. Só então calculamos os erros por distrito. O RMSE OOB do exemplo é √[(0² + 20²)/2] = 14,14 mil dólares. Isso dá o mesmo peso aos dois distritos, embora A tenha recebido mais previsões. Se uma linha ainda não recebeu previsão OOB, ela fica fora do denominador. A cobertura é a fração de linhas que já receberam pelo menos uma previsão.

**Experimento.** Dividimos aleatoriamente o California housing em 10.216 distritos de treino e 10.217 de teste, com semente 42. Usamos os nove preditores originais e tratamos a proximidade do oceano como categórica. Ajustamos uma floresta de regressão com 300 árvores, bootstrap de tamanho igual ao treino e três candidatas por nó. Para cada B de 1 a 300, usamos as primeiras B árvores da mesma floresta. Cada distrito de teste recebe B previsões. Cada distrito de treino recebe somente as previsões das árvores que o deixaram de fora. O código verifica o último RMSE OOB contra a raiz do MSE informado pelo próprio pacote. [Documentação do ranger](https://imbs-hl.github.io/ranger/reference/ranger.html)

![RMSE OOB e de teste ao acrescentar árvores](figuras_random_forest/erro_oob.png)

| Árvores | Cobertura OOB | RMSE OOB (US$) | RMSE de teste (US$) |
|---:|---:|---:|---:|
| 1 | 36,93% | 78.023 | 76.882 |
| 5 | 89,97% | 67.991 | 57.297 |
| 10 | 98,95% | 60.134 | 53.702 |
| 50 | 100% | 51.950 | 51.064 |
| 100 | 100% | 51.082 | 50.801 |
| 300 | 100% | 50.497 | 50.649 |

Com cinco árvores, a diferença ultrapassou US$ 10 mil. Mesmo tendo previsão OOB, muitos distritos ainda dependiam de apenas uma ou duas árvores. Com 50 árvores, todos os distritos já tinham previsão OOB e a diferença caiu para US$ 886. Com 300, o erro OOB ficou US$ 152 abaixo do erro de teste, uma diferença de 0,30%. A aproximação observada tem oscilações: as curvas usam distritos distintos e cada média OOB usa menos árvores que a previsão de teste correspondente.

## Item 5. Como construir a tabela de importância

A definição da coluna determina o significado do ranking. Na **importância por redução de impureza**, cada nó de regressão contribui com a redução da soma dos quadrados obtida pelo corte: SQ do nó pai menos a soma das SQ dos filhos. Credita-se o ganho à variável que definiu o corte, somam-se seus ganhos na árvore e agrega-se entre árvores. Se a tabela for normalizada para somar 100%, a porcentagem representa a participação nos ganhos de corte. Uma variável com muitos pontos de corte possíveis pode ter vantagem nessa medida. [Strobl et al., 2007](https://doi.org/10.1186/1471-2105-8-25)

Para a tabela deste estudo, usamos **importância por permutação OOB**. Em cada árvore, calculamos o MSE nos seus distritos OOB. Em seguida, embaralhamos os valores de uma variável entre esses distritos, preservamos os preços observados e as demais colunas e voltamos a prever com a árvore já ajustada. Subtraímos o MSE original do MSE após a permutação. Repetimos para cada variável e calculamos a média das diferenças entre árvores. Ordenamos a tabela pelo aumento médio do MSE, em dólares ao quadrado, sem padronizar. [Breiman e Cutler, importância por permutação](https://www.stat.berkeley.edu/~breiman/forests/cc_home.htm#varimp)

**Exemplo numérico.** Dois distritos OOB valem 100 e 200 mil dólares. Uma árvore prevê 110 e 190, com MSE igual a 100 em (mil dólares)². Ao trocar as rendas entre os distritos, suponha que passe a prever 190 e 110. O MSE sobe para 8.100 e a renda recebe contribuição de 8.000 nessa árvore. Se trocar as idades preservar as previsões, a contribuição da idade será zero. Na floresta, uma variável pode até ter valor negativo quando a permutação melhora a previsão por acaso. Variáveis correlacionadas também podem substituir umas às outras. O ranking descreve a dependência preditiva da floresta ajustada e não mede efeito causal sobre o preço.

| Posição | Variável | Aumento médio do MSE (bilhões de US$²) |
|---:|---|---:|
| 1 | Renda mediana | 9,25 |
| 2 | Longitude | 6,37 |
| 3 | Latitude | 5,66 |
| 4 | Proximidade do oceano | 3,81 |
| 5 | População | 2,45 |
| 6 | Total de cômodos | 2,28 |
| 7 | Total de quartos | 1,54 |
| 8 | Domicílios | 1,37 |
| 9 | Idade mediana dos imóveis | 1,05 |

Embaralhar a renda mediana provocou a maior perda de precisão OOB. Longitude e latitude vieram em seguida, indicando que a floresta depende da localização para prever os preços. A unidade é quadrática porque o cálculo compara MSEs. O valor 9,25 da primeira linha não representa uma variação de preço em dólares nem uma porcentagem.

## Reprodução

Na raiz do projeto, execute `Rscript src/random_forest.R` com o pacote `ranger` instalado. O script gera o gráfico, a [curva completa de erro](figuras_random_forest/erros.csv) e a [tabela de importância](figuras_random_forest/importancia.csv).
