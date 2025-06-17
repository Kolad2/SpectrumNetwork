function Converter()
    % Укажите путь к папке с .mat-файлами
    input_folder = 'data';  % Замените на ваш путь
    output_folder = 'SpectrumAscii';     % Папка для сохранения ASCII
    
    params_output_file = 'all_parameters.csv';

% Создаем папку для результатов
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

% Инициализация таблицы для параметров
all_parameters = table();

% Обработка всех .mat-файлов
mat_files = dir(fullfile(input_folder, '**', '*.mat'));
for i = 1:length(mat_files)
    file_name = mat_files(i).name;
    folder = mat_files(i).folder;
    file_path = fullfile(folder, file_name);
        % Загрузка данных
        data = load(file_path);
        
        % --- Обработка Spectrum ---
        if isfield(data, 'Spectrum')
            spectrum = data.Spectrum;
            if ismember('F', spectrum.Properties.VariableNames) && ...
               ismember('D', spectrum.Properties.VariableNames)
                F = spectrum.F;
                D = log(spectrum.D);  % Логарифмируем D
                
                % Сохранение спектра
                [~, base_name, ~] = fileparts(file_name);
                spectrum_output = fullfile(output_folder, [base_name '.txt']);
                
                fid = fopen(spectrum_output, 'w');
                fprintf(fid, 'F\tD\n');
                for j = 1:length(F)
                    fprintf(fid, '%.6f\t%.6f\n', F(j), D(j));
                end
                fclose(fid);
            else
                warning('Пропуск файла %s: нет колонок F или D', file_name);
                continue;
            end
        else
            warning('Пропуск файла %s: нет таблицы Spectrum', file_name);
            continue;
        end
        
       if isfield(data, 'Parameters')
    params = table2struct(data.Parameters);
    % Проверяем наличие N или NNN и нормализуем до одного поля N
    if isfield(params, 'N')
        params.N = params.N;  % оставляем как есть
    elseif isfield(params, 'NNN')
        params.N = params.NNN;
        warning('Поле NNN переименовано в N для файла %s', file_name);
    else
        error('Нет ни N, ни NNN в структуре Parameters для файла %s', file_name);
    end

    % Удаляем лишние поля, оставляем только нужные
    keepFields = {'N', 'Date', 'St', 'R', 'Mo', 'Es'};
    allFields = fieldnames(params)';
    
    % Определяем, какие поля существуют в текущей структуре
    existingKeepFields = intersect(keepFields, allFields);
    
    % Создаем новую структуру только с нужными полями
    cleanedParams = rmfield(params, setdiff(allFields, existingKeepFields));

    % Добавляем имя файла как первую колонку
    cleanedParams.FileName = {file_name};

    % Объединяем с общей таблицей
    if isempty(all_parameters)
        all_parameters = cleanedParams;
    else
        all_parameters = [all_parameters; cleanedParams];
    end
else
    warning('Пропуск параметров для %s: нет таблицы Parameters', file_name);
end
        
        disp(['Успешно обработан: ' file_name]);
end

% Сохранение всех параметров
if ~isempty(all_parameters)
    all_parameters = struct2table(all_parameters)

    writetable(all_parameters, params_output_file);
    disp(['Все параметры сохранены в: ' params_output_file]);
else
    disp('Нет данных параметров для сохранения.');
end
end

