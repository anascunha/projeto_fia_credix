from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DADOS = BASE_DIR / "Dados"
RAW_DATA = "raw_data"
CLEAN_DATA = "clean_data"

APPLICATION = DADOS / RAW_DATA /"application_train.csv"
BUREAU = DADOS / RAW_DATA /"bureau.csv"
PREVIOUS = DADOS / RAW_DATA /"previous_application.csv"
INSTALLMENTS = DADOS / RAW_DATA /"installments_payments.csv"
CREDIT_CARD = DADOS / RAW_DATA /"credit_card_balance.csv"


APPLICATION_FEATURES = DADOS / CLEAN_DATA /"application_features.csv"
BUREAU_FEATURES = DADOS / CLEAN_DATA /"bureau_features.csv"
PREVIOUS_FEATURES = DADOS / CLEAN_DATA /"previous_features.csv"
INSTALLMENTS_FEATURES = DADOS / CLEAN_DATA /"installments_features.csv"
CREDIT_CARD_FEATURES = DADOS / CLEAN_DATA /"credit_card_features.csv"

ABT = DADOS / "abt.csv"