"""
Generator classes for plan generation
"""
import pickle
import numpy as np


class PlanFromDiskGenerator():
    """
    Load precomputed plans from disk
    """

    def __init__(self, file_list, num_plans_per_file, plan_dim, plan_len, flattened=True):
        self.file_list = file_list
        self.num_plans_per_file = num_plans_per_file
        self.plan_dim = plan_dim
        self.plan_len = plan_len
        self.flattened = flattened

        self.size = len(self.file_list) * self.num_plans_per_file

        print(f"PlanFromDiskGenerator uses {self.size} precomputed plans")

    def test_consistency(self):
        """
        Assert that the given plans are of correct dimensionality and saved
        in the correct format
        """
        for filename in self.file_list:
            with open(filename, 'rb') as data_stream:
                data_now = pickle.load(data_stream)

            if self.flattened:
                assert data_now.shape == (
                    self.num_plans_per_file,
                    self.plan_len*self.plan_dim
                )
            else:
                assert data_now.shape == (
                    self.num_plans_per_file,
                    self.plan_len,
                    self.plan_dim
                )

    def sample(self, sampled_index=None):
        """
        Sample plan from disk
        """
        if sampled_index is not None:
            assert sampled_index < self.size, "sampled_index larger than number of saved plans"
        else:
            sampled_index = np.random.randint(self.size)

        sampled_file = self.file_list[sampled_index//self.num_plans_per_file]
        sampled_index_in_file = sampled_index % self.num_plans_per_file

        with open(sampled_file, 'rb') as data_stream:
            data = pickle.load(data_stream)[sampled_index_in_file]

        if self.flattened:
            data = data.reshape(self.plan_len, self.plan_dim)

        return{
            'finger_position': data[0, :2],
            'box_position': data[0, 3:5],
            'goal_position': data[-1, 3:5],
            'precomputed_plan': data.reshape(-1)
        }
