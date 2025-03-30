
data = csvread('dataset1_Skin.csv');


[num_samples, num_columns] = size(data);
disp(['Número de muestras: ', num2str(num_samples)]);
disp(['Número de columnas: ', num2str(num_columns)]);

% Separa características y clases
X = data(:, 1:3)'; 
t = data(:, 4)';   

% Verificar que X y t tienen el mismo número de muestras
[~, samples_X] = size(X);
samples_t = length(t);
disp(['Muestras en X: ', num2str(samples_X)]);
disp(['Muestras en t: ', num2str(samples_t)]);

if samples_X ~= samples_t
    error('El número de muestras en X y t no coincide');
end

% Crea la red neuronal usando la sintaxis actualizada
RN = feedforwardnet([5, 3]);  % capas ocultas de 5 y 3 neuronas

% funciones de activación
RN.layers{1}.transferFcn = 'logsig';
RN.layers{2}.transferFcn = 'logsig';
RN.layers{3}.transferFcn = 'purelin';

% algoritmo de entrenamiento
RN.trainFcn = 'trainlm';

% configuracion del entrenamiento
RN.trainParam.epochs = 9;      % epocas
RN.trainParam.goal = 1e-5;        % error
RN.trainParam.max_fail = 6;       % fallos en validación antes que el entrenamiento pare
RN.divideFcn = 'divnone';
% Entrenamiento de la red
[RNE, tr] = train(RN, X, t);

% Simulación con los datos de entrenamiento
y = sim(RNE, X);

% Redondear las salidas para clasificación
y_class = round(y);

% Cálculo del error
error_cuadratico = perform(RNE, y, t);

m = length(t);
aciertos = 0;
for i=1:m
    if(round(y(i))==t(i))
        aciertos = aciertos+1;
    end
end
porcentaje = (aciertos/m)*100;

% Mostrar resultados
disp(['Error cuadrático medio: ', num2str(error_cuadratico)]);
disp(['Precisión de clasificación: ', num2str(porcentaje), '%']);

figure;
% Usamos la función built-in de MATLAB calificándola con "builtin"
builtin('plot', tr.epoch, tr.perf, 'b-', tr.epoch, tr.vperf, 'g-', tr.epoch, tr.tperf, 'r-');
legend('Entrenamiento', 'Validación', 'Test');
xlabel('Épocas');
ylabel('Error Cuadrático Medio');
title('Evolución del Entrenamiento');

save('modelo1_Skin.mat', 'RNE');