import numpy as np
import pandas as pd
import smogn

rng = np.random.default_rng(42)
n = 200

df = pd.DataFrame({
    "x1": rng.normal(size=n),
    "x2": rng.normal(size=n),
})
df["y"] = np.exp(df["x1"] + rng.normal(scale=0.3, size=n))

print("antes:", df.shape)
saida = smogn.smoter(data=df, y="y")
print("depois:", saida.shape)