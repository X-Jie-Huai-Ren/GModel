
from dataset import SolarDataset



if __name__ == '__main__':

    solardataset = SolarDataset(file_path='./datasets/solar.csv', header=None)

    for x in solardataset:
        print(x)
        break