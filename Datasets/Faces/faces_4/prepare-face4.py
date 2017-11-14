import os
import numpy as np

def read_pgm(filename):
    with open(filename, 'rb') as f:
        f.readline()
        (width, height) = [int(i) for i in f.readline().split()]
        depth = int(f.readline())
        buffer = f.read()
        data = np.array(buffer.split()).astype(int)
        image = data.reshape([height, width])
    return image

def show_image(filename):
    from matplotlib import pyplot
    image = read_pgm(filename)
    pyplot.imshow(image, pyplot.cm.gray)
    pyplot.show()

def list2row(list):
    return ','.join([str(x) for x in list])

def images2database(files,shape=(30,32)):
    orientation_class = ['left', 'right', 'up', 'straight']
    sentiment_class = ['angry', 'sad', 'happy', 'neutral']
    accessory_class = ['sunglasses', 'open']
    database = []
    for af in files:
        image = read_pgm(af)
        orientation = [x for x in orientation_class if af.find(x) != -1][0]
        sentiment = [x for x in sentiment_class if af.find(x) != -1][0]
        accessory = [x for x in accessory_class if af.find(x) != -1][0]
        assert image.shape == shape
        database.append(list(image.reshape(image.shape[0] * image.shape[1])) + [orientation, sentiment, accessory])
    return database

def database2file(database,fname='faces.csv'):
    f = open(fname,'w')
    cnames = ['Id'] + ['Col'+str(i) for i in range(len(database[0][0:-3]))] + ['Orientation','Sentiment','Eyes']
    f.write(list2row(cnames)+"\n")
    for i,inst in enumerate(database):
        r = list2row(inst)
        f.write(str(i) + ',' + r +'\n')
    f.close()

def create_database():
    curr_path = os.getcwd()
    all_files = os.listdir(curr_path)
    ascii_files = filter(lambda f: str.find(f,'ascii') != -1, all_files)
    db = images2database(ascii_files)
    database2file(db)
