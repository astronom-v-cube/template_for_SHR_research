import srhdata
import os

os.system("tracker3 daemon -k")
os.system("tracker3 daemon --list-miners-running")

directory = 'data'
files = sorted(os.listdir(directory))
print(files[1:])

for file in files[6:8]:

    srh_file = srhdata.open(f'data/{file}')

    for freq in range(16):

        for scan in range(16):

            srh_file.makeImage(path = './results', calibtable = '2.json', remove_tables = True, frequency = freq, scan = scan, average = 0, compress_image = False, RL = True, clean_disk = True, calibrate = False, cell = 2.45, imsize = 1024, niter = 350, threshold = 35000, stokes = 'RRLL')
            
            os.system("rm -rf *_mask")
            
os.system("tracker3 daemon -s")