import os
from multiprocessing import Process
from multiprocessing import Pool
from synthesis_utils import GlobaMultiSynth

###### Для заполнения ######
observation_range = '0306'
path_to_calib_tables = '/home/dmitry/Documents/calib_tables/2024/240514 - 0306 - 02-02-03.json'
directory_of_data = 'data'
directory_of_result = 'results'
flags_freq = []
number_of_clean_iter = 100000
threshold = 400000
######                ######

GlobaMultiSynth = GlobaMultiSynth()
GlobaMultiSynth.start_procedures()

list_of_freqs = GlobaMultiSynth.indicate_observation_range(observation_range)
GlobaMultiSynth.create_places(directory_of_result, list_of_freqs)
files = sorted(os.listdir(directory_of_data))
print(f'Список файлов: {files}')

# if __name__ == '__main__':
#     procs = []

#     for index, file in enumerate(files):
#         for freq in list_of_freqs:
#             # Создаем процесс для каждого файла и частоты
#             proc = Process(target=GlobaMultiSynth.image_maker, args=(file, freq, flags_freq, path_to_calib_tables, directory_of_data, directory_of_result, number_of_clean_iter, threshold))
#             procs.append(proc)
#             proc.start()

#     for proc in procs:
#         proc.join()

#     GlobaMultiSynth.finish_procedures(directory_of_result, list_of_freqs)

if __name__ == '__main__':
    # Определяем количество одновременно работающих процессов
    max_processes = 4  # Ограничим до 4 процессов

    # Создаем пул с ограниченным числом процессов
    with Pool(processes=max_processes) as pool:
        # Генерируем список задач для обработки: каждый файл, частота и аргументы функции
        tasks = [(file, index, flags_freq, path_to_calib_tables, directory_of_data, directory_of_result, number_of_clean_iter, threshold) for file in files for index, freq in enumerate(list_of_freqs)]
        print(tasks)

        # Запускаем процессы, распределяя задачи по пулу
        pool.starmap(GlobaMultiSynth.image_maker, tasks)

    GlobaMultiSynth.finish_procedures(directory_of_result, list_of_freqs)