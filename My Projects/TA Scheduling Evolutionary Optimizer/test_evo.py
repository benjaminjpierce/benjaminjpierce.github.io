import unittest
from ta_evolution import Evo
import numpy as np


class TestEvo(unittest.TestCase):
    """
        Test class for Evo objective functions
    """

    def setUp(self):
        """
            Set up the Evo instance for testing
        """

        self.evo = Evo("tas.csv", "sections.csv", None)


    def get_test_data(self, test_data_filename):
        """
            Retrieve test data from CSV file

            Parameters:
            - test_data_filename (str): filename of CSV file containing test data

            Returns:
            - np.array: array representing test data
        """

        data = np.genfromtxt(test_data_filename, delimiter=',')
        return data


    def test_overallocation_objective(self):
        """
            Test the overallocation objective function using test cases
        """

        results = []

        for i in range(1, 4):
            test_data_filename = f'test{i}.csv'
            formatted_test_data = self.get_test_data(test_data_filename)
            result = self.evo.overallocation_objective(formatted_test_data)
            results.append(result)
            expected_scores = [37, 41, 23]

        self.assertEqual(results, expected_scores, f"Test failed for overallocation_objective with {test_data_filename}")


    def test_conflicts_objective(self):
        """
            Test the conflicts objective function using test cases
        """

        results = []

        for i in range(1, 4):
            test_data_filename = f'test{i}.csv'
            formatted_test_data = self.get_test_data(test_data_filename)
            result = self.evo.conflicts_objective(formatted_test_data)
            results.append(result)
            expected_scores = [8, 5, 2]

        self.assertEqual(results, expected_scores, f"Test failed for conflicts_objective with {test_data_filename}")


    def test_undersupport_objective(self):
        """
            Test the undersupport objective function using test cases
        """

        results = []

        for i in range(1, 4):
            test_data_filename = f'test{i}.csv'
            formatted_test_data = self.get_test_data(test_data_filename)
            result = self.evo.undersupport_objective(formatted_test_data)
            results.append(result)
            expected_scores = [1, 0, 7]

        self.assertEqual(results, expected_scores, f"Test failed for undersupport_objective with {test_data_filename}")


    def test_unwilling_objective(self):
        """
            Test the unwilling objective function using test cases
        """

        results = []

        for i in range(1, 4):
            test_data_filename = f'test{i}.csv'
            formatted_test_data = self.get_test_data(test_data_filename)
            result = self.evo.unwilling_objective(formatted_test_data)
            results.append(result)
            expected_scores = [53, 58, 43]

        self.assertEqual(results, expected_scores, f"Test failed for unwilling_objective with {test_data_filename}")


    def test_unpreferred_objective(self):
        """
            Test the unpreferred objective function using test cases
        """

        results = []

        for i in range(1, 4):
            test_data_filename = f'test{i}.csv'
            formatted_test_data = self.get_test_data(test_data_filename)
            result = self.evo.unpreferred_objective(formatted_test_data)
            results.append(result)
            expected_scores = [15, 19, 10]

        self.assertEqual(results, expected_scores, f"Test failed for unpreferred_objective with {test_data_filename}")


if __name__ == '__main':
    unittest.main()
