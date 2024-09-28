clear all;
clc;

amostras = load('amostras_labels.mat', 'amostras').amostras;
labels = load('amostras_labels.mat', 'labels').labels;
%% 
% |_Extração de atributos:_|

amostras = amostras';
features = [];
features = [features; sum(diff(sign(amostras)) ~= 0, 1)]; % Zero Crossing Rate
features = [features; kurtosis(amostras)]; % Curtose
features = [features; skewness(amostras)]; % Assimetria
features = [features; max(amostras)]; % Valor máximo
features = [features; mean(amostras)]; % Média
features = [features; range(amostras)]; % Amplitude
features = [features; var(amostras)]; % Variância
features = [features; sum(amostras.^2)]; % Energia
features = [features; sum(amostras .* amostras)]; % Power
features = [features; std(amostras)]; % Desvio padrão
features = [features; rms(amostras)]; % Root Mean Square
features = [features; squeeze(mean(mfcc(amostras, 8000), 1))]; % Coeficientes MFCC (taxa de amostragem = 8000)

Y = fftshift(fft(amostras));
mag_fft = abs(Y);

features = [features; mean(mag_fft)]; % Média
features = [features; median(mag_fft)]; % Mediana
features = [features; var(mag_fft)]; % Variância
features = [features; max(mag_fft)]; % Valor máximo
features = [features; min(mag_fft)]; % Valor mínimo
features = [features; range(mag_fft)]; % Amplitude
features = [features; skewness(mag_fft)]; % Assimetria
features = [features; kurtosis(mag_fft)]; % Curtose
features = [features; sum(mag_fft .^ 2)]; % Energia
features = [features; sum(diff(sign(mag_fft)) ~= 0, 1)]; % Zero Crossing Rate
features = [features; log10(sqrt(sum(diff(mag_fft).^2)))]; % Max Fractal Length
features = [features; rms(mag_fft)]; % Root Mean Square
features = [features; max(diff(mag_fft) / length(mag_fft))]; % Rate Attack Magnitude FFT

phase_fft = angle(Y);

features = [features; skewness(phase_fft)]; % Assimetria
features = [features; kurtosis(phase_fft)]; % Curtose
features = [features; mean(phase_fft)]; % Média
features = [features; var(phase_fft)]; % Variância
features = [features; max(phase_fft)]; % Valor máximo
features = [features; min(phase_fft)]; % Valor mínimo
features = [features; max(diff(phase_fft) / length(phase_fft))]; % Rate of Attack ANGLE


features = features';
features = normalize(features); 

%% 
% |_Aplicação do PCA:_|

disp(['Quantidades de atributos originais extraidos: ', num2str(size(features, 2))]);
[coeff, featuresPCA, ~, ~, explained] = pca(features);
% Escolher o número de componentes que mantém 95% da variância
explainedVariance = cumsum(explained);
numComponents = find(explainedVariance >= 95, 1);

% Reduzir as features com base no número de componentes selecionados
features = featuresPCA(:, 1:numComponents);

disp(['Quantidades de componentes principais: ', num2str(size(features, 2))]);
%% 
% |_Aplicação do grid search e k-folds:_|

% Definir o número de folds
K = 10;
cv = cvpartition(labels, 'KFold', K); 

% Definir os hiperparâmetros para grid search
kernels = {'linear', 'rbf', 'polynomial'};
C_values = [1e-2, 1e-1, 1, 10, 100];
kernel_scales = [1e-2, 1e-1, 1, 10, 100];

unique_classes = unique(labels); % Lista de classes únicas
num_classes = length(unique_classes); % Número de classes
num_samples = size(features, 1); % Número de amostras
num_features = size(features, 2); % Número de features

% Variáveis para armazenar o melhor resultado
bestAccuracy = 0;
bestModel = [];
bestKernel = '';
bestC = 0;
bestKernelScale = 0;
bestConfMat = []; % Variável para armazenar as matrizes de confusão dos melhores folds


for i = 1:length(kernels)
    for j = 1:length(C_values)
        for k = 1:length(kernel_scales)

            
            foldAccuracies = zeros(K, 1);
            foldConfMats = cell(K, 1); 

            for fold = 1:K
                % Separar dados de treino e teste para o fold atual
                trainIdx = training(cv, fold);
                testIdx = test(cv, fold);

                features_train = features(trainIdx, :);
                labels_train = labels(trainIdx);
                features_test = features(testIdx, :);
                labels_test = labels(testIdx);

                % Para cada classe, treinamos um modelo binário (one-vs-all)
                models = cell(num_classes, 1); % Armazena os modelos
                decision_values = zeros(length(labels_test), num_classes); % Para armazenar os scores de decisão

                for class_idx = 1:num_classes
                    % Criar os rótulos binários para a classe atual
                    binary_labels_train = -1 * ones(size(labels_train)); % Todos como negativo
                    binary_labels_train(labels_train == unique_classes(class_idx)) = 1; % Positivos para a classe corrente

                    if strcmp(kernels{i}, 'polynomial')
                        t = templateSVM('KernelFunction', kernels{i}, 'BoxConstraint', C_values(j), 'KernelScale', kernel_scales(k), 'PolynomialOrder', 2);
                    else
                        t = templateSVM('KernelFunction', kernels{i}, 'BoxConstraint', C_values(j), 'KernelScale', kernel_scales(k));
                    end

                    % Treinar o modelo binário para essa classe
                    models{class_idx} = fitcsvm(features_train, binary_labels_train, 'KernelFunction', kernels{i}, 'BoxConstraint', C_values(j), 'KernelScale', kernel_scales(k));
                end

                % Predizer os valores de decisão para cada modelo
                for class_idx = 1:num_classes
                    [~, score] = predict(models{class_idx}, features_test);
                    decision_values(:, class_idx) = score(:, 2); 
                end

                % Decidir a classe final (aquela com o maior score)
                [~, predicted_labels] = max(decision_values, [], 2);

                % Calcular a acurácia no fold atual
                foldAccuracies(fold) = sum(predicted_labels == labels_test) / length(labels_test);

                % Gerar a matriz de confusão para o fold atual
                foldConfMats{fold} = confusionmat(labels_test, predicted_labels);
            end

            meanAccuracy = mean(foldAccuracies);

            % Atualizar o melhor modelo e armazenar as matrizes de confusão 
            if meanAccuracy > bestAccuracy
                bestAccuracy = meanAccuracy;
                bestKernel = kernels{i};
                bestC = C_values(j);
                bestKernelScale = kernel_scales(k);
                bestConfMat = foldConfMats;  % Guardar as matrizes de confusão dos melhores folds
            end
        end
    end
end

disp(['Melhor Acurácia Média: ', num2str(bestAccuracy)]);
disp(['Melhor Kernel: ', bestKernel]);
disp(['Melhor C: ', num2str(bestC)]);
disp(['Melhor Kernel Scale: ', num2str(bestKernelScale)]);
% Somar as matrizes de confusão dos K folds
cumulativeConfMat = zeros(size(bestConfMat{1}));

for fold = 1:K
    cumulativeConfMat = cumulativeConfMat + bestConfMat{fold};
end

figure;
confChartCumulative = confusionchart(cumulativeConfMat, {'abrir', 'fechar', 'ligar'});
confChartCumulative.Title = 'Matriz de Confusão Acumulada';
confChartCumulative.RowSummary = 'row-normalized';
confChartCumulative.ColumnSummary = 'column-normalized';
confChartCumulative.FontSize = 12;