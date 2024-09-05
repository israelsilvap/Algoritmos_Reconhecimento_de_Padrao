clc
clear

% Carregar dados e separá-los em uma tabela
load('dataset.mat', 'Dataset');

% Criar uma tabela a partir dos dados
X = array2table(Dataset(:, 1:3), 'VariableNames', {'empregado?', 'devedor?', 'salário acima de 5SM?'});
y = Dataset(:, 4);


[media_acuracia, media_sensibilidade, media_especificidade, media_precisao, media_f1_score] = K_fold(X, y);
% Exibir os resultados
fprintf('Média das métricas nos 10 folds:\n');
fprintf('Acurácia: %.2f%%\n', media_acuracia*100);
fprintf('Sensibilidade: %.2f%%\n', media_sensibilidade*100);
fprintf('Especificidade: %.2f%%\n', media_especificidade*100);
fprintf('Precisão: %.2f%%\n', media_precisao*100);
fprintf('F1-score: %.2f%%\n', media_f1_score*100);

function entropia = calcular_entropia(y)
    classes = unique(y);
    entropia = 0;
    for i = 1:length(classes)
        p_i = sum(y == classes(i)) / length(y);
        if p_i > 0
            entropia = entropia - p_i * log2(p_i);
        end
    end
end

function infoGanho = calcular_informacao(atributo, y)
    entropia_inicial = calcular_entropia(y);
    valores = unique(atributo);
    entropia_subdivisao = zeros(length(valores), 1);
    for i = 1:length(valores)
        subdivisao = y(atributo == valores(i));
        p = length(subdivisao) / length(atributo);
        entropia_subdivisao(i) = p * calcular_entropia(subdivisao);
    end
    infoGanho = entropia_inicial - sum(entropia_subdivisao);
end

function melhorAtributo = calcular_melhor_atributo(data, y)
    maxGanhoInfo = -inf;
    melhorAtributo = '';
    
    % Percorre as colunas da tabela
    for atributo = data.Properties.VariableNames
        ganhoInfo = calcular_informacao(data{:, atributo{1}}, y);
        if ganhoInfo > maxGanhoInfo
            maxGanhoInfo = ganhoInfo;
            melhorAtributo = atributo{1};
        end
    end
end

function arvore = criar_arvore(data, y)
    % Fim da arvore: os dados são todos de uma mesma classe (nó folha) ou se acabar os atributos disponíveis
    if numel(unique(y)) == 1 || isempty(data.Properties.VariableNames)
        arvore = struct('Atributo', [], 'Esquerda', [], 'Direita', [], 'Classe', mode(y));
        return;
    end

    % Escolha do melhor atributo para o nó
    melhorAtributo = calcular_melhor_atributo(data, y);

    % Dados para a ramificação esquerda e direita
    dados_Esquerda = data(data{:, melhorAtributo} == 0, :);
    rotulos_Esquerda = y(data{:, melhorAtributo} == 0);
    
    dados_Direita = data(data{:, melhorAtributo} == 1, :);
    rotulos_Direita = y(data{:, melhorAtributo} == 1);

    % Remover o melhor atributo dos dados para não reutilizá-lo em descendentes
    dados_Esquerda = removevars(dados_Esquerda, melhorAtributo);
    dados_Direita = removevars(dados_Direita, melhorAtributo);

    arvore = struct('Atributo', melhorAtributo, 'Esquerda', [], 'Direita', []);
    arvore.Esquerda = criar_arvore(dados_Esquerda, rotulos_Esquerda);
    arvore.Direita = criar_arvore(dados_Direita, rotulos_Direita);
end

function classe = testar_amostra(arvore, amostra)
    % Percorrer a árvore até encontrar uma folha
    while ~isempty(arvore.Atributo)    
        % Obter o valor da amostra para o atributo atual
        valor_atributo = amostra.(arvore.Atributo);

        % Decidir se vai para a esquerda ou direita na árvore
        if valor_atributo == 0
            arvore = arvore.Esquerda;
        elseif valor_atributo == 1
            arvore = arvore.Direita;
        end
    end
    % Quando chegar a uma folha, retornar a classe
    classe = arvore.Classe;
end

function [media_acuracia, media_sensibilidade, media_especificidade, media_precisao, media_f1_score] = K_fold(X, y, num_folds)
   
    % Define valores padrão para num_folds e k_values se não forem fornecidos
    if nargin < 3
        num_folds = 10; % Quantidade de folds padrão
    end

    % Parâmetros para validação cruzada
    K = num_folds;
    cv = cvpartition(y, 'KFold', K);
    
    % Inicializar variáveis para acumular métricas
    acuracia = zeros(K, 1);
    sensibilidade = zeros(K, 1);
    especificidade = zeros(K, 1);
    precisao = zeros(K, 1);
    f1_score = zeros(K, 1);
    
    % Loop através de cada fold
    for i = 1:K
        % Separar dados de treino e teste
        trainIdx = training(cv, i);
        testIdx = test(cv, i);
        
        X_train = X(trainIdx, :);
        y_train = y(trainIdx);
        X_test = X(testIdx, :);
        y_test = y(testIdx);
        
        % Chamar a função para criar a árvore de decisão
        arvore = criar_arvore(X_train, y_train);
        
        % Testar os dados de teste
        y_pred = zeros(size(y_test));
        for j = 1:size(X_test, 1)
            nova_amostra = X_test(j, :);
            y_pred(j) = testar_amostra(arvore, nova_amostra);
        end
        
        % Calcular métricas
        [acuracia(i), sensibilidade(i), especificidade(i), precisao(i), f1_score(i)] = calcular_metricas(y_test, y_pred);
    end
    
    % Média das métricas nos 10 folds
    media_acuracia = mean(acuracia);
    media_sensibilidade = mean(sensibilidade);
    media_especificidade = mean(especificidade);
    media_precisao = mean(precisao);
    media_f1_score = mean(f1_score);
end


function [acuracia, sensibilidade, especificidade, precisao, f1_score] = calcular_metricas(y_true, y_pred)
    TP = sum((y_true == 1) & (y_pred == 1));
    TN = sum((y_true == 0) & (y_pred == 0));
    FP = sum((y_true == 0) & (y_pred == 1));
    FN = sum((y_true == 1) & (y_pred == 0));
    
    acuracia = (TP + TN) / (TP + TN + FP + FN);
    sensibilidade = TP / (TP + FN);
    especificidade = TN / (TN + FP);
    precisao = TP / (TP + FP);
    f1_score = 2 * (precisao * sensibilidade) / (precisao + sensibilidade);
end