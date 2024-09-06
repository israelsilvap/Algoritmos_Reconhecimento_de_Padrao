clear;

% Leitura dos dados
inputData = load('InputData.mat');
outputData = load('OutputData.mat');

X = inputData.InputData';
y = outputData.OutputData;

% Extração de Atributos
num_samples = size(X, 1);
num_features = size(X, 2);
num_folds = 10;
k_values = 1:10; % Valores de K para testar

% Extrair atributos para todas as amostras
X_features = [];
for i = 1:num_samples
    X_features(i, :) = extract_features(X(i, :));
end

% Separar o conjunto de dados em treino, validação e teste utilizando a abordagem de 80% para treino e validação e 20% para teste
train_val_ratio = 0.8; % 80% para treino e validação
train_ratio = 0.7; % 70% de treino dentro dos 80%
test_ratio = 0.2; % 20% para teste

indices = randperm(num_samples);
train_val_idx = indices(1:round(train_val_ratio * num_samples));
test_idx = indices(round(train_val_ratio * num_samples) + 1:end);

% Dentro do conjunto de treino e validação, dividir em treino e validação
num_train_val = length(train_val_idx);
train_idx = train_val_idx(1:round(train_ratio * num_train_val));
val_idx = train_val_idx(round(train_ratio * num_train_val) + 1:end);

% Dados de treino, validação e teste
X_train = X_features(train_idx, :);
y_train = y(train_idx);

X_val = X_features(val_idx, :);
y_val = y(val_idx);

X_test = X_features(test_idx, :);
y_test = y(test_idx);

% Testar vários valores de k e escolher o melhor
best_k = 1;
best_accuracy = 0;
for k = k_values
    accuracy = k_fold_cross_validation(X_train, y_train, k, num_folds);
    fprintf('k = %d, Acurácia = %.2f%%\n', k, accuracy * 100);
    if accuracy > best_accuracy
        best_accuracy = accuracy;
        best_k = k;
    end
end

fprintf('Melhor k: %d com Acurácia: %.2f%%\n', best_k, best_accuracy * 100);

% Avaliar no conjunto de teste com o melhor k
num_correct = 0;
for i = 1:length(y_test)
    prediction = kNN(X_train, y_train, X_test(i, :), best_k);
    if prediction == y_test(i)
        num_correct = num_correct + 1;
    end
end
test_accuracy = num_correct / length(y_test);
fprintf('Acurácia no conjunto de teste: %.2f%%\n', test_accuracy * 100);

% Gerar gráficos de dispersão dos atributos calculados
figure;
for i = 1:size(X_features, 2)
    subplot(5, 2, i);
    gscatter(X_features(:, i), y);
    xlabel(sprintf('Atributo %d', i));
    ylabel('Classe');
    title(sprintf('Dispersão do Atributo %d', i));
end

% Função para extrair atributos
function features = extract_features(signal)
    features = [];
    features = [features mean(signal)]; % Média
    features = [features median(signal)]; % Mediana
    features = [features std(signal)]; % Desvio padrão
    features = [features var(signal)]; % Variância
    features = [features max(signal)]; % Valor máximo
    features = [features min(signal)]; % Valor mínimo
    features = [features range(signal)]; % Amplitude
    features = [features skewness(signal)]; % Assimetria
    features = [features kurtosis(signal)]; % Curtose
    features = [features sum(signal.^2)]; % Energia
end

% Função para validação cruzada k-fold
function accuracy = k_fold_cross_validation(X, y, k, num_folds)
    fold_size = floor(length(y) / num_folds);
    indices = randperm(length(y));
    accuracies = zeros(num_folds, 1);
    
    for i = 1:num_folds
        val_indices = indices((i-1)*fold_size + 1:i*fold_size);
        train_indices = setdiff(indices, val_indices);
        
        X_train = X(train_indices, :);
        y_train = y(train_indices);
        X_val = X(val_indices, :);
        y_val = y(val_indices);
        
        num_correct = 0;
        for j = 1:length(y_val)
            prediction = kNN(X_train, y_train, X_val(j, :), k);
            if prediction == y_val(j)
                num_correct = num_correct + 1;
            end
        end
        accuracies(i) = num_correct / length(y_val);
    end
    accuracy = mean(accuracies);
end

% Implementação do k-NN
function prediction = kNN(X_train, y_train, test_instance, k)
    distances = zeros(size(X_train, 1), 1);
    for i = 1:size(X_train, 1)
        distances(i) = euclidean_distance(X_train(i, :), test_instance);
    end
    [sorted_distances, sorted_indices] = sort(distances);
    nearest_neighbors = y_train(sorted_indices(1:k));
    prediction = mode(nearest_neighbors);
end

% Função para calcular a distância euclidiana
function d = euclidean_distance(a, b)
    d = sqrt(sum((a - b).^2));
end
