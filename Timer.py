import time


class Timer:

    def __init__(self):
        self.start_time = time.time()

    def check_timer(self, duration):
        end_time = time.time()
        if int(end_time - self.start_time) >= duration:
            # self.reset_timer()  # Uncomment this line if you desire a repetitive time check
            return True
        else:
            return False

    def reset_timer(self):
        self.start_time = time.time()
