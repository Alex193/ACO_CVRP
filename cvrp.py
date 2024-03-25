import csv
import os
import re
import numpy
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt


def list_of_routs_solver(vertices, demand, edges, pheromones, capacity_limit, alpha, beta):
    # Решение задачи - список маршрутов
    solution_paths = []
    # Список доступных вершин для посещения
    remaining_vertices = list(vertices)
    # Продолжаем, пока есть доступные вершины
    while remaining_vertices:
        # Текущий маршрут и выбор случайной начальной вершины
        current_path = []
        current_vertex = numpy.random.choice(remaining_vertices)
        current_capacity = capacity_limit - demand[current_vertex]
        # Добавляем выбранную вершину в маршрут и удаляем из списка доступных
        current_path.append(current_vertex)
        remaining_vertices.remove(current_vertex)
        # Продолжаем, пока есть доступные вершины
        while remaining_vertices:
            probabilities = []
            # Расчёт вероятностей для перехода к следующим вершинам
            for next_vertex in remaining_vertices:
                edge = (min(next_vertex, current_vertex), max(next_vertex, current_vertex))
                if edges[edge] == 0:
                    prob = 0
                else:
                    # Формула для расчета вероятности выбора пути
                    prob = (pheromones[edge] ** alpha) * ((1 / edges[edge]) ** beta)
                probabilities.append(prob)
            # Нормализация вероятностей и выбор следующей вершины
            if numpy.sum(probabilities) > 0:
                probabilities = probabilities / numpy.sum(probabilities)
                current_vertex = numpy.random.choice(remaining_vertices, p=probabilities)
            else:
                # Если вероятности нулевые, выбираем случайную вершину
                current_vertex = numpy.random.choice(remaining_vertices)
            # Проверяем, хватит ли вместимости для добавления вершины в маршрут
            current_capacity -= demand[current_vertex]
            if current_capacity >= 0:
                current_path.append(current_vertex)
                remaining_vertices.remove(current_vertex)
            else:
                break
        # Добавляем построенный маршрут в решение
        solution_paths.append(current_path)
    # Возвращаем построенное решение
    return solution_paths


def pheromones_renew(
    trail_strengths,  # Изначальная сила феромонов на рёбрах
    route_solutions,  # Список решений и их стоимостей
    optimal_route,  # Лучшее найденное решение и его стоимость
    enhancement_factor,  # Фактор усиления феромонов
    evaporation_rate,  # Скорость испарения феромонов
    additional_influence  # Дополнительное влияние на феромоны
):
    # Перерасчет силы феромонов с учетом испарения и дополнительного влияния
    average_cost = numpy.mean([solution[1] for solution in route_solutions])
    trail_strengths = {edge: (evaporation_rate + additional_influence / average_cost) * intensity for edge, intensity in trail_strengths.items()}

    # Обновление лучшего решения на основе сравнения с новыми решениями
    if optimal_route is None or optimal_route[1] > route_solutions[0][1]:
        optimal_route = min(route_solutions, key=lambda x: x[1])

    # Усиление феромонов на лучшем пути
    for segment in optimal_route[0]:
        for j in range(len(segment) - 1):
            segment_edge = (min(segment[j], segment[j + 1]), max(segment[j], segment[j + 1]))
            trail_strengths[segment_edge] += enhancement_factor / optimal_route[1]

    # Дополнительное усиление феромонов для топ решений
    for index, (routes, score) in enumerate(route_solutions[:enhancement_factor]):
        for route in routes:
            for j in range(len(route) - 1):
                route_edge = (min(route[j], route[j + 1]), max(route[j], route[j + 1]))
                trail_strengths[route_edge] += (enhancement_factor - (index + 1)) / score ** (index + 1)
    # Возвращаем обновленное лучшее решение
    return optimal_route


def run_optimization(file_name):
    """
    Основная функция для оптимизации маршрутов доставки с использованием алгоритма муравьиной колонии.

    :param file_name: Путь к файлу с данными задачи.
    :param disable_tqdm: Если True, то прогресс выполнения не будет отображаться.
    :return: Кортеж с результатами выполнения алгоритма.
    """
    # Загрузка данных
    # -------------------------------------------------------------------------------
    with open(file_name) as file:
        data = file.read()
    # Извлечение данных с использованием регулярных выражений
    num_ants = int(re.search('o of trucks: (\d+)', data).group(1))
    optimal_value = int(re.search('Optimal value: (\d+)', data).group(1))
    capacity_limit = int(re.search('CAPACITY : (\d+)', data).group(1))
    node_positions = re.findall(r'(\d+) (\d+) (\d+)', data)
    node_demands = re.findall(r'(\d+) (\d+)', data)
    # Координаты узлов
    graph = {int(node): (int(x), int(y)) for node, x, y in node_positions}
    # Потребности узлов
    demand = {int(node): int(demand) for node, demand in node_demands}
    # -------------------------------------------------------------------------------

    # Инициализация вершин и дорог
    vertices = list(graph.keys())
    vertices.remove(1)  # Удаление депо из списка вершин

    # Расчет веса дорог между вершинами
    edges = {
        (min(a, b), max(a, b)): numpy.sqrt((graph[a][0] - graph[b][0]) ** 2 + (graph[a][1] - graph[b][1]) ** 2)
        for a in graph for b in graph if a != b
    }
    # Начальное распределение феромонов
    pheromones = {edge: 1.0 for edge in edges}

    # Параметры алгоритма
    alpha = 0.9  # Значимость следа феромона для выбора направления движения.
    beta = 5.0  # Роль эвристической информации, как значимо расстояние до следующего узла.
    sigma = 4  # Вклад лучшего решения в увеличение феромона на тропе.
    rho = 0.6  # Процент испарения феромона, определяющий уменьшение следа с временем.
    theta = 2.0  # Бонус к феромону для наилучшего текущего решения.
    iterations = 1000  # Общее число циклов поиска решений алгоритмом.
    stop_count = 300  # Максимальное количество циклов без обнаружения лучшего решения до остановки алгоритма.
    cur_count = 0  # Текущее количество итераций без улучшения решения.
    best_solution = ([], float('inf'))

    # Процесс оптимизации
    t = tqdm(range(iterations))
    for _ in t:
        solutions = []
        for _ in range(num_ants):
            solution = list_of_routs_solver(vertices, demand, edges, pheromones, capacity_limit, alpha, beta)
            # Вычиcление общую длину маршрутов в предложенном решении.
            # -------------------------------------------------------------------------------
            cost = 0
            for route in solution:
                # Добавляем в начало и конец маршрута депо, предполагая его индекс 1
                adjusted_route = [1] + route + [1]
                # Вычисление длины маршрута путем суммирования расстояний между последовательными городами
                for start, end in zip(adjusted_route[:-1], adjusted_route[1:]):
                    distance = edges.get((min(start, end), max(start, end)), 0)
                    cost += distance
            solutions.append((solution, cost))
            # -------------------------------------------------------------------------------
        current_best = min(solutions, key=lambda x: x[1])
        if current_best[1] < best_solution[1]:
            best_solution = current_best
            cur_count = 0
        else:
            cur_count += 1
        pheromones_renew(pheromones, solutions, best_solution, sigma, rho, theta)
        t.set_description(f"Итерация: Стоимость={int(best_solution[1])}, Оптимальная стоимость={optimal_value}")
        if cur_count >= stop_count:
            tqdm.write('Остановлено из-за отсутствия прогресса.')
            break
    elapsed_time = t.format_dict['elapsed']
    t.close()
    return best_solution[1], optimal_value, elapsed_time, len(graph), num_ants


def calculate():
    analysis_results = []  # Список для хранения результатов обработки каждого файла
    # directories = ['A', 'B', 'E']
    directories = ['A', 'B', 'E']
    file_save_to = 'results.csv'
    # Перебор всех директорий
    for dir_path in tqdm(directories, desc="Обработка директорий"):
        # Получение списка файлов в директории
        for file in tqdm(os.listdir(dir_path), desc=f"Анализ файлов в {dir_path}"):
            # Обработка только файлов с расширением .vrp
            if file.endswith('.vrp'):
                full_file_path = os.path.join(dir_path, file)  # Полный путь к файлу
                file_analysis_result = run_optimization(full_file_path)  # Вызов функции run_optimization для анализа файла
                analysis_results.append((dir_path, file, file_analysis_result))  # Сохранение результата в список

    # Открытие файла для записи
    with open(file_save_to, 'w', newline='') as csv_file:
        # Создание объекта writer для записи данных в CSV
        csv_writer = csv.writer(csv_file)
        # Заголовки столбцов в CSV файле
        headers = ['Имя файла', 'Стоимость решения', 'Оптимальное значение',
                   'Время выполнения', 'Размер графа', 'Количество грузовиков']
        csv_writer.writerow(headers)

        # Перебор данных для записи в файл
        for directory, filename, analysis_result in analysis_results:
            csv_writer.writerow([
                filename,
                analysis_result[0], analysis_result[1],
                analysis_result[2], analysis_result[3], analysis_result[4]
            ])

def analyse():
    # Загрузка данных из файла CSV
    results = pd.read_csv('results.csv', delimiter=';', engine='python')

    # считаем среднюю ошибку по каждому из сетов и визуализируем ее
    #------------------------------------------------------------------------------------------------------------------------
    # Преобразование столбцов 'Стоимость решения' и 'Оптимальное значение' в числовой формат
    results['Стоимость решения'] = pd.to_numeric(results['Стоимость решения'], errors='coerce')
    results['Оптимальное значение'] = pd.to_numeric(results['Оптимальное значение'], errors='coerce')
    # Расчет ошибки для каждой записи
    results['Ошибка'] = (results['Стоимость решения'] - results['Оптимальное значение']) / results['Оптимальное значение']
    # Группировка данных по первой букве в названии файла (сету) и вычисление средней ошибки для каждого сета
    average_errors = results.groupby(results['Имя файла'].str[0])['Ошибка'].mean()
    # Вывод средних ошибок по сетам
    print(average_errors)

    results['Время выполнения'] = pd.to_numeric(results['Время выполнения'], errors='coerce')
    results['Размер графа'] = pd.to_numeric(results['Размер графа'], errors='coerce')

    # Разделение данных по сетам
    set_a = results[results['Имя файла'].str.startswith('A')].sort_values(by='Размер графа')
    set_b = results[results['Имя файла'].str.startswith('B')].sort_values(by='Размер графа')
    set_e = results[results['Имя файла'].str.startswith('E')].sort_values(by='Размер графа')

    # Для этого сначала рассчитаем среднее время выполнения для каждого размера графа в каждом сете
    set_a_mean = set_a.groupby('Размер графа')['Время выполнения'].mean()
    set_b_mean = set_b.groupby('Размер графа')['Время выполнения'].mean()
    set_e_mean = set_e.groupby('Размер графа')['Время выполнения'].mean()

    # Визуализация
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.bar(set_a_mean.index, set_a_mean.values, color='blue')
    plt.title('Среднее время выполнения алгоритма: Сет A')
    plt.xlabel('Размер графа')
    plt.ylabel('Среднее время выполнения')

    plt.subplot(1, 3, 2)
    plt.bar(set_b_mean.index, set_b_mean.values, color='green')
    plt.title('Среднее время выполнения алгоритма: Сет B')
    plt.xlabel('Размер графа')
    plt.ylabel('Среднее время выполнения')

    plt.subplot(1, 3, 3)
    plt.bar(set_e_mean.index, set_e_mean.values, color='red')
    plt.title('Среднее время выполнения алгоритма: Сет E')
    plt.xlabel('Размер графа')
    plt.ylabel('Среднее время выполнения')

    plt.tight_layout()
    plt.show()

    results['Время выполнения'] = pd.to_numeric(results['Время выполнения'], errors='coerce')

    # Рассчитываем среднее время выполнения для каждого сета
    results['Сет'] = results['Имя файла'].str[0]
    average_time = results.groupby('Сет')['Время выполнения'].mean()

    # Визуализация
    average_time.plot(kind='bar', color=['blue', 'orange', 'green'])
    plt.title('Среднее время работы алгоритма по сетам')
    plt.xlabel('Сет')
    plt.ylabel('Среднее время выполнения (секунды)')
    plt.xticks(rotation=0)
    plt.show()

    # Визуализация результатов в виде гистограммы
    average_errors.plot(kind='bar')
    plt.title('Средняя ошибка по сетам A, B, E')
    plt.xlabel('Сет')
    plt.ylabel('Средняя ошибка')
    plt.xticks(rotation=0)  # Горизонтальные метки на оси X
    plt.show()

# calculate()
analyse()
