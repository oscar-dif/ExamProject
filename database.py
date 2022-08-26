from sqlalchemy import create_engine
import psycopg2
import pandas as pd

from sqlalchemy import create_engine
engine = create_engine('postgresql://houses_user:123@localhost:5432/houses_db')

df = pd.read_csv('transformed.csv')
df.to_sql('new_table', engine)