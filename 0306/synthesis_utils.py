import srhdata
import os
import multiprocessing
from multiprocessing import Process
import logging

class GlobaMultiSynth():

    def __init__(self) -> None:
        self.list_of_freqs_0306 = [2800, 3000, 3200, 3400, 3600, 3800, 4000, 4200, 4400, 4600, 4800, 5000, 5200, 5400, 5600, 5800]
        self.list_of_freqs_0612 = [6000, 6400, 6800, 7200, 7600, 8000, 8400, 8800, 9200, 9600, 10000, 10400, 10800, 11200, 11600, 12000]
        self.list_of_freqs_1224 = [12200, 12960, 13720, 14480, 15240, 16000, 16760, 17520, 18280, 19040, 19800, 20560, 21320, 22080, 23000, 23400]

    def start_procedures(self):
        os.system("tracker3 daemon -k")
        os.system("tracker3 daemon --list-miners-running")

        multiprocessing.log_to_stderr()
        logger = multiprocessing.get_logger()
        logger.setLevel(logging.INFO)

    def finish_procedures(self, directory_of_result, list_of_freqs):
        os.system("tracker3 daemon -s")
        os.system("rm -rf casa*.log")
        for fq in list_of_freqs:
            os.system(f"rm -rf {directory_of_result}/{fq}/*_mask")

    def create_places(self, directory_of_result, list_of_freqs):
        try:
            os.mkdir(f'{directory_of_result}')
            for fq in list_of_freqs:
                os.mkdir(f'{directory_of_result}/{fq}')
            print('Папка для сохранения результатов создана')
        except:
            print('Папка для сохранения результатов уже существует')

    def indicate_observation_range(self, observation_range):
        if observation_range == '0306':
            return self.list_of_freqs_0306
        elif observation_range == '0612':
            return self.list_of_freqs_0612
        elif observation_range == '1224':
            return self.list_of_freqs_1224

    def image_maker(self, file_of_data, freq, flags_freq, path_to_calib_tables, directory_of_data, directory_of_result, number_of_clean_iter, threshold):
        """
        Мультипроцессинговая функция по созданию радиоизображений для конкретной частоты
        """
        for scan in range(0, 20):
            if freq in flags_freq:
                pass
            else:
                proc_id = os.getpid()
                print(f'{file_of_data} in process id: {proc_id}, frequency: {freq}')

                (srhdata.open(f'{directory_of_data}/{file_of_data}')).makeImage(
                    path = f'./{directory_of_result}/{freq}',
                    calibtable = path_to_calib_tables,
                    remove_tables = True,
                    frequency = freq,
                    scan = scan,
                    average = 0,
                    compress_image = False,
                    RL = True,
                    clean_disk = True,
                    calibrate = False,
                    cell = 2.45,
                    imsize = 1024,
                    niter = number_of_clean_iter,
                    threshold = threshold,
                    stokes = 'RRLL'
                )