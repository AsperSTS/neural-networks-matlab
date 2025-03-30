% Cargar el modelo entrenado
load('modelo1_Skin.mat', 'RNE');

% Cargar nuevos datos para probar el modelo
nuevos_datos = csvread('dataset1_Skin.csv');

% Separar características y etiquetas
X_nuevos = nuevos_datos(:, 1:3)';
t = nuevos_datos(:, 4)'; % Asumiendo que la columna 4 contiene las etiquetas reales

% Simular la red neuronal
y = sim(RNE, X_nuevos);

% Calcular precisión
m = length(t);
aciertos = 0;
for i=1:m
    if(round(y(i))==t(i))
        aciertos = aciertos+1;
    end
end
porcentaje = (aciertos/m)*100;

% Mostrar resultados
fprintf('Total de datos: %d\n', m);
fprintf('Predicciones correctas: %d\n', aciertos);
fprintf('Precisión de clasificación: %.2f%%\n', porcentaje);
