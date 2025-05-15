import numpy as np

class DistributedTensor:
    """Class to simulate a distributed tensor across multiple devices""" 
    
    def __init__(self, tensor=None, device=None):
        """
        Initialize a distributed tensor
        
        Args:
            tensor: The full tensor data (numpy array)
            device_map: Dictionary mapping device IDs to tensor chunks
                        If None, the tensor is considered to be on a single device (0)
        """
        self.__full_tensor = tensor  # programmers shouldn't be able to access this
        self.device_map = {device: self.__full_tensor}
        self.cur_dev_group = [[device]]
        self.is_shard = False
        self.is_replicated = False
    
    def shard(self, device_group, all_devices):
        if self.is_shard:
            raise RuntimeError("You shouldn't shard an already sharded tensor, please use a collective first!")
        
        devices_being_used = []
        for row in device_group:
            for device in row:
                if device not in all_devices:
                    raise RuntimeError("Device in device group not an initialized device")
                
                devices_being_used.append(device)
        
        for key, value in self.device_map.items():
            if key not in devices_being_used:
                raise RuntimeError("Current tensor device is not included in provided device group")

        w_t, h_t = self.__full_tensor.shape
        w_d, h_d = device_group.shape
            
        area_dg = w_d * h_d
        area_tensor = w_t * h_t
        if  area_dg > area_tensor:
            raise RuntimeError("More Devices than possible shards")
        
        if (w_t % w_d != 0) or (h_t % h_d != 0):
            raise RuntimeError("Tensor dimensions are not evenly divisible by device group dimensions, uneven sharding not allowed.")
        
        repeat_factors = (w_t // w_d, h_t // h_d)
    
        mapped_array = np.tile(device_group, repeat_factors)

        unique_values = np.unique(mapped_array)
        grouped_elements = {}
        
        for value in unique_values:
            positions = np.where(mapped_array == value)
        
            rows, cols = positions
            unique_rows = np.unique(rows)
            
            structured_result = []
            for row in unique_rows:
                row_indices = np.where(rows == row)[0]
                row_cols = cols[row_indices]
                sorted_indices = np.argsort(row_cols)
                row_cols = row_cols[sorted_indices]
                row_indices = row_indices[sorted_indices]
                
                row_values = [self.__full_tensor[rows[i], cols[i]] for i in row_indices]
                structured_result.append(row_values)
            
            grouped_elements[int(value)] = np.array(structured_result)
        
        self.device_map = grouped_elements
        self.cur_dev_group = device_group
        self.is_shard = True
    
    def gather(self, dst, all_devices):
        if not self.is_shard:
            raise RuntimeError("Tensor is not sharded, cannot unshard!")
        
        found = False
        for row in self.cur_dev_group:
            for device in row:
                if dst == device:
                    found = True
        
        if not found:
            raise RuntimeError("Provided dst not part of current device group")
        
        # w_t, h_t = self.__full_tensor.shape
        w_d, h_d = self.cur_dev_group.shape

        # Get a sample shard to determine dimensions
        sample_device = list(self.device_map.keys())[0]
        sample_shard = self.device_map[sample_device]
        shard_rows, shard_cols = sample_shard.shape
        
        # Calculate full tensor dimensions
        w_t = w_d * shard_rows
        h_t = h_d * shard_cols
        
        repeat_factors = (w_t // w_d, h_t // h_d)
        mapped_array = np.tile(self.cur_dev_group, repeat_factors)

        reassembled = np.zeros((w_t, h_t))
        
        for device_id, device_data in self.device_map.items():
            positions = np.where(mapped_array == device_id)
            rows, cols = positions
        
            data_row_idx = 0
            data_col_idx = 0
            prev_row = -1
            
            for i in range(len(rows)):
                current_row = rows[i]
                current_col = cols[i]
                
                if current_row != prev_row:
                    data_row_idx += (0 if prev_row == -1 else 1)
                    data_col_idx = 0
                    prev_row = current_row
                
                if data_row_idx < device_data.shape[0] and data_col_idx < device_data.shape[1]:
                    reassembled[current_row, current_col] = device_data[data_row_idx, data_col_idx]
                    data_col_idx += 1

        self.__full_tensor = reassembled
        self.device_map = {dst: self.__full_tensor}
        self.cur_dev_group = [[dst]]
        self.is_shard = False
    
    def replicate(self, device_group, all_devices):
        if self.is_shard:
            raise RuntimeError("You shouldn't replicate an already sharded tensor, please use a collective first!")
         
        devices_being_used = []
        for row in device_group:
            for device in row:
                if device not in all_devices:
                    raise RuntimeError("Device in device group not an initialized device")
                devices_being_used.append(device)
                
        for key, value in self.device_map.items():
            if key not in devices_being_used:
                raise RuntimeError("Current tensor device is not included in provided device group")
        
        new_map = {}
        for row in device_group:
            for device in row:
                new_map[device] = self.__full_tensor
        
        self.device_map = new_map
        self.cur_dev_group = device_group
        self.is_replicated = True
    
    def reduce(self, dst, all_devices):
        if not self.is_replicated:
            raise RuntimeError("Tensor is not replicated, cannot reduce")
        
        found = False
        for row in self.cur_dev_group:
            for device in row:
                if dst == device:
                    found = True
        
        if not found:
            raise RuntimeError("Provided dst not part of current device group")

        # tensors should all be the same size
        tensor_list = [value for key, value in self.device_map.items()]
        result = np.sum(tensor_list, axis=0)

        self.__full_tensor = result
        self.device_map = {dst: self.__full_tensor}
        self.cur_dev_group = [[dst]]
        self.is_replicated = False

    
    def __repr__(self):
        return f"DistributedTensor(data={self.__full_tensor}, devices={list(self.device_map.keys())})"