# Classificação com Árvore de Decisão e Validação Cruzada K-Fold

Este projeto implementa uma árvore de decisão para classificar uma base de dados com atributos categóricos, utilizando o critério de **ganho de informação (entropia)**. O trabalho foi desenvolvido como parte da disciplina **Tópicos Especiais em Telecomunicações I (ECO0080)** da **Universidade Federal do Ceará - Campus Sobral**.

## Descrição

O projeto utiliza uma base de dados fornecida, que contém 876 amostras com três atributos de entrada categóricos (binários) e duas classes de saída. A classificação é feita utilizando uma **Árvore de Decisão**, onde os nós são selecionados com base na entropia.

### Informações da Base de Dados
- **876 amostras**
- **Atributos de entrada**:
  - Empregado? (“sim” = 1, “não” = 0)
  - Devedor? (“sim” = 1, “não” = 0)
  - Salário acima de 5 SM? (“sim” = 1, “não” = 0)
- **Classes de saída**: 
  - Empréstimo aprovado? (“sim” = 1, “não” = 0)
  - 462 amostras da classe “não”
  - 414 amostras da classe “sim”

### Regras do Algoritmo
- O critério para escolha dos nós na árvore de decisão é o **ganho de informação (entropia)**.
- Cada atributo só pode ser utilizado uma vez na construção da árvore.
- A árvore para de crescer quando a entropia não pode ser mais reduzida ou quando todos os atributos foram utilizados.
- Não é necessário realizar podas.
- A validação cruzada será feita com **K=10** (K-fold).

### Métricas de Saída
- Acurácia
- Sensibilidade
- Especificidade
- Precisão
- F1-score

## Objetivo

O objetivo é classificar corretamente os dados utilizando uma árvore de decisão, sem o uso de funções prontas para cálculo de entropia ou para construção da árvore. O código também deve gerar as métricas médias após a validação cruzada com K=10.

## Resultados Esperados

- **Acurácia**: Taxa média de acerto nos 10 folds.
- **Sensibilidade**: Proporção de verdadeiros positivos.
- **Especificidade**: Proporção de verdadeiros negativos.
- **Precisão**: Proporção de positivos identificados corretamente.
- **F1-score**: Média harmônica entre precisão e sensibilidade.

## Personalizações

- **Atributos**: Os atributos categóricos já estão definidos na base de dados.
- **Critério de Parada**: A árvore para quando não é possível reduzir mais a entropia ou quando todos os atributos forem usados.

## Observações

O algoritmo foi desenvolvido do zero, sem o uso de funções prontas do MATLAB ou Python para árvore de decisão ou cálculo de entropia.
O código foi escrito de maneira organizada e comentada para facilitar o entendimento e correção.
