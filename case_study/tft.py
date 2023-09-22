import json
import numpy as np
import os
from collections import defaultdict
import numpy as np
from PIL import Image

def read_dir(data_dir, maxuser=10000):
    clients = []
    groups = []
    data = defaultdict(lambda : None)

    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.json')]
    tot = 0
    for f in files:
        file_path = os.path.join(data_dir,f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        tot += len(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        data.update(cdata['user_data'])
        if tot > maxuser:
            break

    clients = list(sorted(data.keys()))
    return clients, groups, data

def arr2img(arr):
    mat = np.reshape(arr,(28,28))
    return Image.fromarray(np.uint8(mat * 255) , 'L')
    
def reorganize(indir, outdir):
    clients, groups, data = read_dir(indir)
    print("done loading original data")
    N=62
    cl = len(data.keys())
    for clid, username in enumerate(data.keys()):
        path = os.path.join(outdir, username)
        os.mkdir(path)
        class_lists = [os.path.join(path, str(n)) for n in range(N)]
        for i in range(N):
            os.mkdir(class_lists[i])
        for order, x in enumerate(data[username]["x"]):
            img = arr2img(x)
            img.save(os.path.join(class_lists[int(data[username]["y"][order])],"t_{}.png".format(order)))
        #break
        print("{} percent finished".format(clid/cl * 100))

def datareorganize(data, outdir):
    #clients, groups, data = read_dir(indir)
    print("---transforming---")
    N=62
    cl = len(data.keys())
    for clid, username in enumerate(data.keys()):
        path = os.path.join(outdir, username)
        #print(os.listdir(outdir))
        #print(username)
        #if username in os.listdir(outdir):
        if (os.path.isfile(path)) or (os.path.isdir(path)):
            #print("")
            continue
        os.mkdir(path)
        class_lists = [os.path.join(path, str(n)) for n in range(N)]
        for i in range(N):
            os.mkdir(class_lists[i])
        for order, x in enumerate(data[username]["x"]):
            img = arr2img(x)
            img.save(os.path.join(class_lists[int(data[username]["y"][order])],"t_{}.png".format(order)))
        #break
        print("{} percent finished".format(clid/cl * 100))

def mem_reorganize(data_dir, outdir):
    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.json')]
    lf = len(files)
    for fid,f in enumerate(files):
        data = defaultdict(lambda : None)
        clients = []
        groups = []

        file_path = os.path.join(data_dir,f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        data.update(cdata['user_data'])
        print("finished loading {} percent".format(fid/lf*100))
        datareorganize(data, outdir)

    clients = list(sorted(data.keys()))
    return clients, groups, data

if __name__ == "__main__":
    pass