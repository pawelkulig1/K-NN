import numpy as np
import unittest
from knn import KNN
from normalizer import Normalizer
from sklearn.neighbors import KNeighborsClassifier #for scikit-learn KNN model
from sklearn.datasets import load_breast_cancer, make_classification  # breast cancer data


class TestKNN(unittest.TestCase):

    def setUp(self):
        self.data = load_breast_cancer()

    def test_distance_calculation(self):
        knn = KNN(None) #params not needed to test this functionality
        vector1 = [0,0,0,0,0]
        vector2 = [5,5,5,5,5]

        expectedDistance = (5**2 + 5**2 + 5**2 + 5**2 + 5**2) ** 0.5

        self.assertAlmostEqual(knn.calculateDistance(vector1, vector2), expectedDistance)

    def test_compare_to_scikit_learn_changing_k(self):
        normalizer = Normalizer(self.data)
        data = normalizer.normalize()

        testSize = 100
        trainSize = len(data.data) - testSize
        for i in range(1, 12):
            with self.subTest(i=i):
                print("k: ", i)
                neighbours = i

                trainData = {}
                testData = {}

                trainData['data'] = data.data[:trainSize]
                trainData['target'] = data.target[:trainSize]

                testData['data'] = data.data[trainSize:]
                testData['target'] = data.target[:trainSize]
                knn = KNN(trainData)

                #scikit-learn model:
                model = KNeighborsClassifier(n_neighbors=neighbours)
                model.fit(trainData['data'], trainData['target'])

                ourCounter = 0
                sciCounter = 0
                for i, e in enumerate(testData['data']):
                    if knn.makeGuess(e, neighbours) == testData['target'][i]:
                        ourCounter+=1

                    if model.predict([e]) == testData['target'][i]:
                        sciCounter+=1

                self.assertAlmostEqual(ourCounter/(testSize), sciCounter/(testSize), 3)

    def test_compare_to_scikit_learn_changing_test_size(self):
        normalizer = Normalizer(self.data)
        data = normalizer.normalize()

        for i in range(50, 130, 10):
            with self.subTest(i=i):
                testSize = i
                trainSize = len(data.data) - testSize

                print("test size: ", i)

                neighbours = 5

                trainData = {}
                testData = {}

                trainData['data'] = data.data[:trainSize]
                trainData['target'] = data.target[:trainSize]

                testData['data'] = data.data[trainSize:]
                testData['target'] = data.target[:trainSize]
                knn = KNN(trainData)

                #scikit-learn model:
                model = KNeighborsClassifier(n_neighbors=neighbours)
                model.fit(trainData['data'], trainData['target'])

                ourCounter = 0
                sciCounter = 0
                for i, e in enumerate(testData['data']):
                    if knn.makeGuess(e, neighbours) == testData['target'][i]:
                        ourCounter+=1

                    if model.predict([e]) == testData['target'][i]:
                        sciCounter+=1

                self.assertAlmostEqual(ourCounter/(testSize), sciCounter/(testSize), 3)

    def test_find_nearest(self):
        tab = {}
        tab['data'] = np.array([[0.2, 0.31], [0.1, 0.4], [2, 3], [2, 2], [1, 4], [1.5, 2], [2, 1.4], [1.3, 3.1],
                                [-1, 2.3], [-1.4,-1.6], [-0.2, -0.3], [-0.9, 1.7], [2, -1.7], [0.5, -0.7], [1.2, -1.2]])

        tab['target'] = np.array([0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1])

        expectedNearestPoint = np.array([[-0.2, -0.3], [0.2, 0.31], [0.1, 0.4], [0.5, -0.7]])
        expectedNearestDist = [[((-0.2)**2 + (-0.3)**2) ** 0.5, 0], [((0.2)**2 + (0.31)**2) ** 0.5, 0],
                               [((0.1)**2 + (0.4)**2) ** 0.5, 0], [((0.5)**2 + (-0.7)**2) ** 0.5, 1]]

        k = 4

        knn = KNN(tab)
        nearest = knn.findKNearest(np.zeros(2), k)

        self.assertListEqual(nearest, expectedNearestDist)

    def test_make_guess(self):
        tab = {}
        tab['data'] = np.array([[0.2, 0.31], [0.1, 0.4], [2, 3], [2, 2], [1, 4], [1.5, 2], [2, 1.4], [1.3, 3.1],
                                [-1, 2.3], [-1.4, -1.6], [-0.2, -0.3], [-0.9, 1.7], [2, -1.7], [0.5, -0.7],
                                [1.2, -1.2]])

        tab['target'] = np.array([0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1])

        knn = KNN(tab)

        self.assertAlmostEqual(knn.makeGuess(np.zeros(2), 4), 0) #3 points belong to group '0' and 1 point to group '1'
                                                                 #  expected result 0


    def test_exception(self):
        tab = {}
        tab['data'] = np.array([[0.2, 0.31], [0.1, 0.4], [2, 3], [2, 2], [1, 4]]) #length 5

        tab['target'] = np.array([0, 0, 1, 1, 0, 1]) #length 6

        knn = KNN(tab)

        self.assertRaises(Exception, knn.findKNearest, [2, 3, 4], 2)
        self.assertRaises(Exception, knn.findKNearest, np.zeros(2), 3)
        self.assertRaises(Exception, knn.findKNearest, np.zeros(2), 6)

if __name__ == '__main__':
    unittest.main()
