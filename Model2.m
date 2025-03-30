

%{
    MODELO BEANS
    DATASET: 17 COLUMNAS, 13611 FILAS

    CLASES:
    - N:1
    - O:2
%}

data = csvread('dataset2_Fertility.csv');

% Verificar dimensiones de los datos cargados
[num_samples, num_columns] = size(data);
disp(['Número de muestras: ', num2str(num_samples)]);
disp(['Número de columnas: ', num2str(num_columns)]);

% Separar características y clases
X = data(:, 1:9)'; % Transponer para que cada columna sea una muestra
t = data(:, 10)';   % Vector de clases

% Verificar que X y t tienen el mismo número de muestras
[~, samples_X] = size(X);
samples_t = length(t);
disp(['Muestras en X: ', num2str(samples_X)]);
disp(['Muestras en t: ', num2str(samples_t)]);

if samples_X ~= samples_t
    error('El número de muestras en X y t no coincide');
end

% red neuronal usando la sintaxis actualizada
RN = feedforwardnet([6, 6, 4]);  % 3,4,4 

% funciones de activación
RN.layers{1}.transferFcn = 'logsig';
RN.layers{2}.transferFcn = 'logsig';
RN.layers{3}.transferFcn = 'purelin';

% algoritmo de entrenamiento
RN.trainFcn = 'trainlm';

% Configuración del entrenamiento
RN.trainParam.epochs = 20;      % Número máximo de épocas
RN.trainParam.goal = 0.001;%1e-5;        % Error objetivo
RN.trainParam.max_fail = 6;       % Máximo número de fallos en validación

% Entrenamiento de la red
[RNE, tr] = train(RN, X, t);

% Simulación con los datos de entrenamiento
y = sim(RNE, X);

% Cálculo del error
error_cuadratico = perform(RNE, y, t);

% Redondear las salidas para clasificación
y_class = round(y);

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

% Graficar evolución del entrenamiento - usamos la función figure para evitar conflictos
figure;
% Usamos la función built-in de MATLAB calificándola con "builtin"
builtin('plot', tr.epoch, tr.perf, 'b-', tr.epoch, tr.vperf, 'g-', tr.epoch, tr.tperf, 'r-');
legend('Entrenamiento', 'Validación', 'Test');
xlabel('Épocas');
ylabel('Error Cuadrático Medio');
title('Evolución del Entrenamiento');

% Guardar el modelo entrenado
save('modelo2_Fertility.mat', 'RNE');