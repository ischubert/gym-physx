"""
Generator classes for plan generation
"""
import pickle
import numpy as np


class PlanFromDiskGenerator():
    """
    Load precomputed plans from disk
    """

    def __init__(
            self,
            plan_dim, plan_len,
            file_list=None, num_plans_per_file=None,
            plan_array=None, flattened=True
    ):
        self.plan_dim = plan_dim
        self.plan_len = plan_len
        self.file_list = file_list
        self.num_plans_per_file = num_plans_per_file
        self.plan_array = plan_array
        self.flattened = flattened

        if self.plan_array is None:
            # In this case, the data is read directly from the disk
            assert self.file_list is not None, "Both plan_array and file_list are None"
            print("PlanFromDiskGenerator reads plans from disk")
            self.size = len(self.file_list) * self.num_plans_per_file
        else:
            # In this case, the data is given as an object
            assert self.file_list is None, "Both plan_array and file_list are given"
            print("PlanFromDiskGenerator was provided with list of plans")
            self.size = len(self.plan_array)

        self.test_consistency()
        print(f"PlanFromDiskGenerator uses {self.size} precomputed plans")

    def test_consistency(self):
        """
        Assert that the given plans are of correct dimensionality and saved
        in the correct format
        """
        if self.plan_array is None:
            # If plans are read from disk...
            for filename in self.file_list:
                # ...assert that all files exist and can be opened
                with open(filename, 'rb') as data_stream:
                    data_now = pickle.load(data_stream)
                # ...and assert that the content of each file is of the expected format
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
        else:
            # If plans are provided as an object, assert that array has expected format
            if self.flattened:
                assert self.plan_array.shape == (
                    self.size,
                    self.plan_len*self.plan_dim
                )
            else:
                assert self.plan_array.shape == (
                    self.size,
                    self.plan_len,
                    self.plan_dim
                )

    def sample(self, sampled_index=None):
        """
        Sample plan from disk
        """
        # Validate index if given and sample index if not
        if sampled_index is not None:
            assert sampled_index < self.size, "sampled_index larger than number of saved plans"
        else:
            sampled_index = np.random.randint(self.size)

        if self.plan_array is None:
            # Read plan from the disk
            sampled_file = self.file_list[sampled_index//self.num_plans_per_file]
            sampled_index_in_file = sampled_index % self.num_plans_per_file
            with open(sampled_file, 'rb') as data_stream:
                data = pickle.load(data_stream)[sampled_index_in_file]
        else:
            # Sample plan from object
            data = self.plan_array[sampled_index]

        if self.flattened:
            # reshape into format (time, dim)
            data = data.reshape(self.plan_len, self.plan_dim)

        return{
            'finger_position': data[0, :2],
            'box_position': data[0, 3:5],
            'goal_position': data[-1, 3:5],
            'precomputed_plan': data.reshape(-1)
        }
