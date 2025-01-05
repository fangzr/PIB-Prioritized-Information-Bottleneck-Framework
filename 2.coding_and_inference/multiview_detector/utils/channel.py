import numpy as np

class Channel:
    def __init__(self, distance_to_base_station):
        self.distance_to_base_station = distance_to_base_station
        self.frequency = 2.4e9  # 2.4 GHz
        self.light_speed = 3e8  # speed of light (m/s)
        self.path_loss_exponent = 3.5  # Urban environment
        self.shadowing_deviation = 8  # dB
        self.interference_power = 0.1  # Watts per device
        self.num_devices_per_100m2 = 10  # Device density needs to be adjusted to observe different scenarios
        self.bandwidth = 2000000  # Hz (2 MHz)

    def calculate_capacity(self, distance):
        path_loss = self.calculate_path_loss(distance)
        shadow_fading = np.random.normal(0, self.shadowing_deviation)
        total_path_loss = path_loss + shadow_fading

        area = np.pi * distance**2  # m^2
        num_devices = (area / 10000) * self.num_devices_per_100m2
        total_interference = num_devices * self.interference_power

        received_power = -total_path_loss + 30  # dBm assuming transmitter power

        noise_power = -174 + 10 * np.log10(self.bandwidth)  # 热噪声功率，BW=2MHz
        sinr_db = received_power - (10 * np.log10(total_interference) + noise_power)
        sinr_linear = 10**(sinr_db / 10)

        capacity = self.bandwidth * np.log2(1 + sinr_linear)
        return capacity / 1000  # Convert capacity to Kbps

    def calculate_path_loss(self, distance):
        return 20 * np.log10(distance) + 20 * np.log10(self.frequency) + 20 * np.log10(4 * np.pi / self.light_speed) + 10 * self.path_loss_exponent * np.log10(distance)

    def get_delays(self, C_th):
        delays = []
        for distance in self.distance_to_base_station:
            capacity = self.calculate_capacity(distance)
            if capacity >= C_th:
                delay = 0
            else:
                delay = int(np.ceil(C_th / capacity))
            delays.append(delay)
        return delays
    
    @staticmethod
    def calculate_delays(num_cameras=7, distance_to_base_station=None, C_th=43):
        # np.random.seed(2)
        if distance_to_base_station is None:
            distance_to_base_station = [100] * num_cameras  # meters

        channel = Channel(distance_to_base_station)
        delays = channel.get_delays(C_th)
        print(f"Delays: {delays}")
        return delays

if __name__ == "__main__":
    np.random.seed(5)  
    delays = Channel.calculate_delays()
    print(f"Delays: {delays}")