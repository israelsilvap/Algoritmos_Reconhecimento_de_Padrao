# Classificação de Sinais de Movimento de Mão com k-NN e Validação Cruzada k-Fold

Este projeto implementa um classificador de k-Nearest Neighbors (k-NN) para classificar sinais de movimento de mão. A implementação inclui a extração de atributos e o uso de validação cruzada com k-fold (K=10), conforme solicitado na disciplina **Tópicos Especiais em Telecomunicações I (ECO0080)** da **Universidade Federal do Ceará - Campus Sobral**.

## Descrição

O conjunto de dados utilizado consiste em sinais capturados por uma luva sensorial equipada com 3 acelerômetros. Cada sinal corresponde a um dos dois tipos de movimento de mão:
- **Abrir a mão** (Classe -1)
- **Mão para baixo** (Classe +1)

### Estrutura dos Dados
- **InputData.mat**: Matriz de tamanho 1500 x 120, onde cada uma das 120 colunas corresponde a um sinal de 1500 pontos.
- **OutputData.mat**: Vetor de tamanho 120, contendo as classes de saída (-1 ou +1) para cada sinal.

## Objetivo

O objetivo principal do projeto é implementar um classificador k-NN capaz de identificar corretamente os movimentos de mão. O projeto inclui:
- Extração de pelo menos 10 atributos diferentes dos sinais de entrada.
- Teste de diferentes valores de **k** no classificador k-NN.
- Implementação de validação cruzada com **K=10**.
- Geração de gráficos de dispersão para os atributos extraídos.

## Funcionalidades

- **Classificação com k-NN**: Implementação de um classificador k-NN do zero (sem usar bibliotecas prontas).
- **Extração de Atributos**: Teste de pelo menos 10 atributos dos sinais de movimento.
- **Validação Cruzada k-Fold**: Utilização de validação cruzada com K=10 para garantir maior robustez na avaliação do modelo.
- **Geração de Gráficos**: Gráficos de dispersão 1D e 2D para visualização dos atributos e sua relevância.

## Resultados Esperados

- **Acurácia**: O código exibirá a taxa média de acerto nos 10 folds da validação cruzada.
- **Gráficos**: Gráficos de dispersão serão gerados para ajudar na visualização e avaliação dos atributos escolhidos.

## Personalizações

- **Atributos**: É possível testar diferentes combinações de atributos e ajustar a seleção de acordo com os gráficos de dispersão.
- **Valor de k**: O valor de k no k-NN pode ser ajustado para obter melhores resultados de acurácia.

## Observações

- Não foram utilizadas funções prontas para o k-NN nem para a validação cruzada k-fold, conforme solicitado.
- O código está organizado e comentado para facilitar o entendimento.
