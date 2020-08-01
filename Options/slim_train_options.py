import argparse


class options():
    def __init__(self):
        parser = argparse.ArgumentParser()

        parser.add_argument('--s_e_freq', type=int, default=10,
                            help="save epoch frequency")
        parser.add_argument('--b_size', type=int, default=50)
        parser.add_argument('--epochs', type=int, default=50)
        self.parser = parser.parse_args()