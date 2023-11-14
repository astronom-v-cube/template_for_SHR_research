import srhdata
import os
import multiprocessing
from multiprocessing import Process
import logging

os.system("tracker3 daemon -k")
os.system("tracker3 daemon --list-miners-running")

multiprocessing.log_to_stderr()
logger = multiprocessing.get_logger()
logger.setLevel(logging.INFO)

###### Для заполнения ######
observation_range = '0306'
path_to_calib_tables = 'path'
directory_of_data = 'data'
directory_of_result = 'results'
flags_freq = []
number_of_clean_iter = 350
######                ######

files = sorted(os.listdir(directory_of_data))
print(f'Список файлов: {files}')

if observation_range == '0612':
    list_of_freqs = [5800, 6200, 6600, 7000, 7400, 7800, 8200, 8600, 9000, 9400, 9800, 10200, 10600, 11000, 11400, 11800]
elif observation_range == '0306':
    list_of_freqs = [2800, 3000, 3200, 3400, 3600, 3800, 4000, 4200, 4400, 4600, 4800, 5000, 5200, 5400, 5600, 5800]
elif observation_range == '1224':
    list_of_freqs = [12000, 12800, 13600, 14400, 15200, 16000, 16800, 17600, 18400, 19200, 20000, 20800, 21600, 22400, 23200, 24000]

try:
    os.mkdir(f'{directory_of_result}')
    for fq in list_of_freqs:
        os.mkdir(f'{directory_of_result}/{fq}')
    print('Папка для сохранения результатов создана')
except:
    print('Папка для сохранения результатов уже существует')


def image_maker(file_of_data):
    """
    Мультипроцессинговая функция по созданию радиоизображений
    """
    for freq in range(0, 16):
        
        for scan in range(0, 20):
            
            if list_of_freqs[freq] in flags_freq:
                pass
            
            else:
                proc_id = os.getpid()
                print(f'{file_of_data} in process id: {proc_id}')
    
                (srhdata.open(f'{directory_of_data}/{file_of_data}')).makeImage(
                    path = f'./{directory_of_result}/{list_of_freqs[freq]}', 
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
                    threshold = 35000, 
                    stokes = 'RRLL'
                    )
 
if __name__ == '__main__':
    
    procs = []
    
    for index, file in enumerate(files):
        proc = Process(target=image_maker, args=(file,))
        procs.append(proc)
        proc.start()
    
    for proc in procs:
        proc.join()
        
    os.system("tracker3 daemon -s")
    os.system("rm -rf casa*.log")
    for fq in list_of_freqs:
        os.system(f"rm -rf {directory_of_result}/{fq}/*_mask")