% Cargar datos de Iris
% Asumiendo que iris.txt está en el directorio actual
data = csvread('iris.txt');

% Verificar dimensiones de los datos cargados
[num_samples, num_columns] = size(data);
disp(['Número de muestras: ', num2str(num_samples)]);
disp(['Número de columnas: ', num2str(num_columns)]);

% Separar características y clases
X = data(:, 1:4)'; % Transponer para que cada columna sea una muestra
t = data(:, 5)';   % Vector de clases

% Verificar que X y t tienen el mismo número de muestras
[~, samples_X] = size(X);
samples_t = length(t);
disp(['Muestras en X: ', num2str(samples_X)]);
disp(['Muestras en t: ', num2str(samples_t)]);

if samples_X ~= samples_t
    error('El número de muestras en X y t no coincide');
end

% Crear la red neuronal usando la sintaxis actualizada
RN = feedforwardnet([5, 3]);  % Red feedforward con capas ocultas de 5 y 3 neuronas

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
perf = perform(RNE, y, t);

% Redondear las salidas para clasificación
y_class = round(y);

% Calcular precisión
accuracy = sum(y_class == t) / length(t) * 100;

% Mostrar resultados
disp(['Error cuadrático medio: ', num2str(perf)]);
disp(['Precisión de clasificación: ', num2str(accuracy), '%']);

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

% Visualización de los datos en el espacio de características
figure;
gscatter(X(1,:), X(2,:), t);
xlabel('Longitud de Sépalo');
ylabel('Ancho de Sépalo');
title('Visualización de clases: Longitud vs Ancho de Sépalo');

figure;
gscatter(X(3,:), X(4,:), t);
xlabel('Longitud de Pétalo');
ylabel('Ancho de Pétalo');
title('Visualización de clases: Longitud vs Ancho de Pétalo');

% Guardar el modelo entrenado
save('modelo_iris.mat', 'RNE');