import seaborn as sns
from pandas import DataFrame

def draw_basic_plot(df: DataFrame, col: str):
    sns.barplot(x=df.index, y=df['col'])
