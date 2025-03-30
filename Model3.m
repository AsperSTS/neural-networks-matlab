%{
    MODELO FERTILIDAD
    DATASET: 10 COLUMNAS, 100 FILAS

    CLASES:
    - SEKER: 1
    - BARBUNYA: 2
    - BOMBAY: 3
    - CALI: 4
    - HOROZ: 5
    - SIRA: 6
    - DERMASON: 7
%}

data = csvread('dataset3_Beans.csv');

% Verificar dimensiones de los datos cargados
[num_samples, num_columns] = size(data);
disp(['Número de muestras: ', num2str(num_samples)]);
disp(['Número de columnas: ', num2str(num_columns)]);

% Separar características y clases
X = data(:, 1:16)'; % Transponer para que cada columna sea una muestra
t = data(:, 17)';   % Vector de clases

% Verificar que X y t tienen el mismo número de muestras
[~, samples_X] = size(X);
samples_t = length(t);
disp(['Muestras en X: ', num2str(samples_X)]);
disp(['Muestras en t: ', num2str(samples_t)]);

if samples_X ~= samples_t
    error('El número de muestras en X y t no coincide');
end

% Crear la red neuronal usando la sintaxis actualizada
RN = feedforwardnet([10, 5, 5]);  % 

% Configurar funciones de activación
RN.layers{1}.transferFcn = 'logsig';
RN.layers{2}.transferFcn = 'logsig';
RN.layers{3}.transferFcn = 'purelin';

% Configurar algoritmo de entrenamiento
RN.trainFcn = 'trainlm';

% Configuración del entrenamiento
RN.trainParam.epochs = 1000;      % Número máximo de épocas
RN.trainParam.goal = 1e-5;        % Error objetivo
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

% Graficar comparación entre salida deseada y obtenida
figure;
builtin('plot', 1:length(t), t, 'bo-', 'LineWidth', 2);
hold on;
builtin('plot', 1:length(t), y, 'r*-');
legend('Clases Deseadas', 'Salidas de la Red');
xlabel('Muestra');
ylabel('Clase');
title('Comparación entre Salidas Deseadas y Obtenidas');


% Guardar el modelo entrenado
save('modelo3_Beans.mat', 'RNE');