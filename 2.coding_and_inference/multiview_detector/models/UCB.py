import numpy as np

class UCB:
    def __init__(self, num_cameras, alpha=1.0, input_threshold=0.80, penalty=-100, delay_threshold=0.5, exploration_rate=0.1):
        self.num_cameras = num_cameras
        self.alpha = alpha
        self.counts = np.zeros(num_cameras)
        self.values = np.zeros(num_cameras)
        self.input_threshold = input_threshold
        self.penalty = penalty
        self.delay_threshold = delay_threshold
        self.exploration_rate = exploration_rate

    def select_arms(self, delays):
        ucb_values = self.values + self.alpha * np.sqrt((2 * np.log(np.sum(self.counts) + 1)) / (self.counts + 1e-5))
        num_to_select = np.random.randint(1, self.num_cameras + 1)
        
        # Ensure that we select arms according to the exploration rate
        if np.random.rand() < self.exploration_rate:
            # Exploration: select random arms
            valid_cam_indices = np.random.choice(np.where(delays <= self.delay_threshold)[0], num_to_select, replace=False)
        else:
            # Exploitation: select arms based on UCB values
            valid_cam_indices = [i for i in np.argsort(ucb_values) if delays[i] <= self.delay_threshold]
            valid_cam_indices = valid_cam_indices[-num_to_select:]
        
        return sorted(valid_cam_indices)

    def update(self, arms, reward):
        for arm in arms:
            self.counts[arm] += 1
            n = self.counts[arm]
            value = self.values[arm]
            new_value = ((n - 1) / n) * value + (1 / n) * reward
            self.values[arm] = new_value

    def calculate_reward(self, bits_loss, moda, moda_baseline):
        if moda >= moda_baseline * self.input_threshold:
            reward = -bits_loss
        else:
            reward = self.penalty  # 惩罚
        return reward
