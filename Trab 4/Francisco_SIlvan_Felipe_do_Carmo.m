% Inicializando o ambiente de trabalho
clc;
clear;
close all;

% Carregar o arquivo .mat
load('Input1.mat'); 

X = Input1';
y = [ones(2000, 1); 2*ones(2000, 1)];


% Gráfico de dispersão 2D dos dados
figure;
gscatter(X(:,1), X(:,2), y, 'rb', '..', 12);
title('Gráfico de Dispersão 2D dos Dados de Entrada');
xlabel('Atributo 1');
ylabel('Atributo 2');
legend('Classe 1', 'Classe 2');
grid off;


% Definir K para K-fold
K = 10;
indices = crossvalind('Kfold', y, K);

% Inicializar vetores para armazenar acurácias
accuracy_bayes = zeros(K, 1);
accuracy_lda = zeros(K, 1);

% Loop K-fold
for k = 1:K
    test_idx = (indices == k);
    train_idx = ~test_idx;
    
    X_train = X(train_idx, :);
    y_train = y(train_idx);
    X_test = X(test_idx, :);
    y_test = y(test_idx);
    
    % Classificador Bayesiano
    y_pred_bayes = bayesian_classifier(X_train, y_train, X_test);

    % Classificador Linear LDA
    y_pred_lda = LDA_linear_classifier(X_train, y_train, X_test);
    
    % Calcular acurácia
    accuracy_lda(k) = sum(y_pred_lda == y_test) / length(y_test);

    % Calcula a acurácia
    accuracy_bayes(k) = sum(y_pred_bayes == y_test) / length(y_test);
    
end

% Calcular métricas de desempenho
mean_acc_bayes = mean(accuracy_bayes);
std_acc_bayes = std(accuracy_bayes);

mean_acc_lda = mean(accuracy_lda);
std_acc_lda = std(accuracy_lda);

% Exibir resultados
fprintf('Acurácia do Classificador Bayesiano: %.2f%%\n', mean_acc_bayes * 100);
fprintf('Desvio Padrão do Classificador Bayesiano: %.2f%%\n', std_acc_bayes * 100);
fprintf('\n')

fprintf('Acurácia do LDA: %.2f%%\n', mean_acc_lda * 100);
fprintf('Desvio Padrão do LDA: %.2f%%\n', std_acc_lda * 100);

function y_pred_bayes = bayesian_classifier(X_train, y_train, X_test)
    classes = unique(y_train);
    priori = zeros(1, length(classes));
    posteriori = zeros(size(X_test, 1), length(classes));

    for i = 1:length(classes)
        idx_samples = find(y_train == classes(i));
        priori(i) = length(idx_samples)/length(X_train);
        samples = X_train(idx_samples, :);
        mu = mean(samples, 1);
        cv = cov(samples);
        for j = 1:length(X_test)
            x = X_test(j, :);
            likelihood = gaussian_likelihood(x, mu, cv); 
            posteriori(j, i) = likelihood * priori(i);
        end
    end
    % Para cada exemplo de teste, escolhe a classe com maior valor de posteriori
    [~, y_pred_idx] = max(posteriori, [], 2);
    y_pred_bayes = classes(y_pred_idx);
end

function likelihood = gaussian_likelihood(x, mu, cov_matrix)
    % x: Vetor de amostras de teste
    % mu: Média da distribuição
    % cov_matrix: Matriz de covariância

    % Dimensão do vetor de entrada
    N = length(x); 
    
    % Termo constante na frente do exponencial
    const_term = 1 / ((2 * pi)^(N/2) * sqrt(det(cov_matrix)));
    
    % Diferença entre o vetor de amostras e a média
    diff = x - mu;
    
    % Termo exponencial
    exp_term = exp(-0.5 * (diff * (cov_matrix \ diff')));
    
    % Likelihood
    likelihood = const_term * exp_term;
end

function y_pred_lda = LDA_linear_classifier(X_train, y_train, X_test)
    % Calcular médias de cada classe
    mu1 = mean(X_train(y_train == 1, :), 1);
    mu2 = mean(X_train(y_train == 2, :), 1);

    % Calcular matrizes de covariância de cada classe
    S1 = cov(X_train(y_train == 1, :));
    S2 = cov(X_train(y_train == 2, :));

    % Matriz de dispersão within-class
    Sw = S1 + S2;

    % Calcular vetor de projeção w
    w = inv(Sw) * (mu1 - mu2)';

    % Normalizar w
    w = w / norm(w);

    % Projetar dados de treino e teste em w
    z_train = X_train * w;
    z_test = X_test * w;

    % Calcular médias projetadas
    mean_z1 = mean(z_train(y_train == 1));
    mean_z2 = mean(z_train(y_train == 2));

    % Definir limiar como ponto médio
    threshold = (mean_z1 + mean_z2) / 2;

    % Classificar dados de teste
    y_pred_lda = zeros(length(z_test), 1);
    y_pred_lda(z_test >= threshold) = 1;
    y_pred_lda(z_test < threshold) = 2;

end
