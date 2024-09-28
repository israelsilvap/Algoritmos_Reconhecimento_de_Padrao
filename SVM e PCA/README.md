# Reconhecimento de Padrões com PCA e SVM

Este projeto implementa um classificador utilizando **PCA (Análise de Componentes Principais)** para redução de dimensionalidade e **SVM (Máquinas de Vetores de Suporte)** para a classificação de comandos de voz. A implementação inclui a extração de atributos e o uso de validação cruzada com k-fold (K=10), conforme solicitado na disciplina **Tópicos Especiais em Telecomunicações I (ECO0080)** da **Universidade Federal do Ceará - Campus Sobral**.

## Descrição

O conjunto de dados utilizado consiste em gravações de áudio com três comandos de voz:
- **Abrir** (Classe 1)
- **Fechar** (Classe 2)
- **Ligar** (Classe 3)

### Estrutura dos Dados
- **Amostras de áudio**: Um total de 30 amostras de áudio por classe.
- **Atributos extraídos**: Foram utilizados atributos derivados dos sinais de áudio, como FFT (Transformada Rápida de Fourier), energia do sinal, entre outros, para caracterizar os padrões vocais.

## Objetivo

O objetivo principal do projeto é implementar um classificador SVM capaz de identificar corretamente os comandos de voz após a aplicação de PCA para redução de dimensionalidade. O projeto inclui:
- Extração de atributos relevantes dos sinais de áudio.
- Aplicação de PCA para redução da dimensionalidade.
- Teste de diferentes parâmetros no classificador SVM.
- Implementação de validação cruzada com **K=10**.
- Geração da matriz de confusão para os resultados do conjunto de melhores hiperparâmetros encontrados.

## Funcionalidades

- **Classificação com SVM**: Implementação do classificador SVM utilizando três tipos de kernel: linear, RBF e polinomial (grau 2).
- **Otimização de Hiperparâmetros**: Uso de grid search para otimizar os parâmetros de kernel, constante de relaxamento **C**, e escala do kernel.
- **Validação Cruzada k-Fold**: Utilização de validação cruzada com **K=10** para garantir maior robustez na avaliação do modelo.

## Resultados Esperados

- **Acurácia**: O código exibirá a taxa média de acerto nos 10 folds da validação cruzada.
- **Matriz de Confusão**: Será gerada uma matriz de confusão para análise do desempenho do classificador.

## Personalizações

- **Atributos**: É possível testar diferentes combinações de atributos e ajustar a seleção e os resultados da classificação.
- **Parâmetros de SVM**: Os parâmetros do SVM, como o tipo de kernel, a constante de relaxamento **C** e a escala do kernel, podem ser ajustados para obter melhores resultados.

## Observações

- Foram utilizadas funções prontas para o SVM e para a validação cruzada k-fold, conforme permitido.
- O código está organizado e comentado para facilitar o entendimento.