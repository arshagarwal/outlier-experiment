import os 
import shutil
os.mkdir('cacd2/10-30')
dest = 'cacd2/10-30'
path = 'cacd/10-30'
for i in os.listdir('cacd/10-30')[:10000]:
    shutil.move(path + '/' + i, dest)
    print(i)
