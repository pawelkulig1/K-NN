import numpy as np

class Normalizer:
    def __init__(self, data):
        self.data = data
        self.size = len(data)
        self.max_arr = np.zeros(self.size)
        self.min_arr = np.array(data.data[0])
        self.findMinMax()


    def findMinMax(self):
        for i, e in enumerate(self.data.data):
            for j in range(self.size):
                if e[j] > self.max_arr[j]:
                    self.max_arr[j] = e[j]

                if e[j] < self.min_arr[j]:
                    self.min_arr[j] = e[j]

    def normalize(self):
        #normalize data to be in range [0, 1]

        for i, e in enumerate(self.data.data):
            for j in range(self.size):
                e[j] = (e[j] - self.min_arr[j]) / (self.max_arr[j] - self.min_arr[j])

        return self.data
