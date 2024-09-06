# Classificador Bayesiano e LDA com Validação Cruzada

Este repositório contém a implementação de um classificador Bayesiano e Linear Discriminant Analysis (LDA) com validação cruzada K-fold, como parte do Trabalho 4 da disciplina de Tópicos Especiais em Telecomunicações I da Universidade Federal do Ceará (Campus Sobral).

## Descrição

O projeto visa aplicar técnicas de classificação para duas bases de dados com amostras contínuas, utilizando:
- Classificador Bayesiano com atributos contínuos;
- LDA seguido de um classificador unidimensional baseado em limiar.

## Estrutura do Código

- **Classificador Bayesiano**: Implementado sem uso de funções prontas do MATLAB para estimativa de parâmetros.
- **LDA**: Realiza a projeção dos dados para uma dimensão, seguido de um classificador baseado em limiar.
- **Validação K-fold**: K=10, com cálculo da acurácia média e desvio padrão da acurácia.
- **Gráficos**: Geração de gráficos de dispersão 2D para visualização das amostras.

## Bases de Dados

- **Input1**: 4000 amostras, 2 atributos de entrada contínuos, 2 classes.
- **Input2**: Outra base de dados utilizada para análise comparativa de desempenho.

## Resultados

- **Acurácia Média**: Calculada para os 10 folds.
- **Desvio-Padrão da Acurácia**: Analisado para medir a variação de desempenho.
