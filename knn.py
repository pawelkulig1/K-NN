import numpy as np


class KNN:
    def __init__(self, data):
        self.data = data

    def findKNearest(self, vector, k):
        if len(vector) != len(self.data['data'][0]):
            raise ("findKNearest: Vectors must be the same length!")

        if len(self.data['data']) != len(self.data['target']):
            raise ("findKNearest: Vectors 'data' and 'target' must be the same length")

        if k > len(self.data['data']):
            raise ("findKNearest: 'k' can't be larger than 'data' size")

        dist = []
        for i, e in enumerate(self.data['data']):
            dist.append([self.calculateDistance(e, vector), self.data['target'][i]])

        dist = sorted(dist, key=lambda x: x[0])

        return dist[0:k]

    def makeGuess(self, vector, k):
        nearest = self.findKNearest(vector, k)
        counter = 0
        for i, e in enumerate(nearest):
            counter += e[1]

        if(counter > len(nearest)/2):
            return 1
        return 0

    def calculateDistance(self, vector1, vector2):
        dist = 0
        for i, e in enumerate(vector1):
            dist += (e - vector2[i])**2
        
        return dist**0.5