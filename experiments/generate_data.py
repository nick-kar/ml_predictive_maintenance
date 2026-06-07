import numpy as np
import pandas as pd

def generate(n=5000):
    data = []

    for i in range(n):
        temp = np.random.normal(70, 10)
        vib = np.random.normal(0.5, 0.2)
        pressure = np.random.normal(30, 5)
        hours = np.random.randint(1, 1000)

        failure = int(temp > 85 or vib > 0.9 or hours > 800)

        data.append([temp, vib, pressure, hours, failure])

    df = pd.DataFrame(data, columns=[
        "temp", "vibration", "pressure", "hours", "failure"
    ])

    df.to_csv("data/sensor_data.csv", index=False)

    return df


if __name__ == "__main__":
    df = generate()
    print("Dataset created:", df.shape)
    print(df.head())