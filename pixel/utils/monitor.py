from tqdm import tqdm
import psutil, GPUtil, time

import torch as T


def main():
    if T.cuda.is_available():
        CPU = tqdm(total=100, desc='CPU %', position=1, colour='RED')
        GPU = tqdm(total=100, desc='GPU %', position=2, colour='GREEN')
        RAM = tqdm(total=100, desc='RAM %', position=3, colour='BLUE')
        with CPU, GPU, RAM:
            while True:
                CPU.n = psutil.cpu_percent()
                CPU.refresh()
                GPU.n = GPUtil.showUtilization()
                GPU.refresh()
                RAM.n = psutil.virtual_memory().percent
                RAM.refresh()
                time.sleep(0.1)
    else:
        CPU = tqdm(total=100, desc='CPU %', position=1, colour='RED')
        RAM = tqdm(total=100, desc='RAM %', position=2, colour='BLUE')
        with CPU, RAM:
            while True:
                CPU.n = psutil.cpu_percent()
                CPU.refresh()
                RAM.n = psutil.virtual_memory().percent
                RAM.refresh()
                time.sleep(0.1)

if __name__ == '__main__':
    main()
