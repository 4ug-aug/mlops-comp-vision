import numpy as np
import pandas as pd


def get_bird_names():
    data = pd.read_csv("data/raw/BUTTERFLIES.csv")

    # get names
    names = np.unique(data["labels"])

    with open("data/raw/names.txt", mode="w") as file:
        for name in names:
            file.write(f"{name}\n")


if __name__ == "__main__":
    get_bird_names()
