import pandas as pd

from utils import *

application = pd.read_csv(APPLICATION_FEATURES)
bureau = pd.read_csv(BUREAU_FEATURES)

abt = (
    application
    .merge(bureau, on="SK_ID_CURR", how="left")
)

abt.fillna(0, inplace=True)

abt.to_csv(ABT, index=False)