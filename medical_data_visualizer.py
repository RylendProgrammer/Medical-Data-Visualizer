import pandas as pd # type: ignore
import seaborn as sns # type: ignore
import matplotlib.pyplot as plt # type: ignore
import numpy as np # type: ignore

# 1
df = pd.read_csv('medical_examination.csv')
#print(df.head())

# 2
df['overweight'] = None

counter = 0
for row in df.itertuples(index=True):
    height = row.height * 0.01
    weight = row.weight
    BMI = weight / (height * height)
    if BMI > 25:
        df.at[row.Index, 'overweight'] = 1
    else:
        df.at[row.Index, 'overweight'] = 0
    counter += 1
    if counter == 5:
        break
#print(df.head())

# 3
df['cholesterol'] = df['cholesterol'].apply(lambda x: 0 if x == 1 else 1)
df['gluc'] = df['gluc'].apply(lambda x: 0 if x == 1 else 1)
#print(df.head())


# 4
def draw_cat_plot():
    df = pd.read_csv('medical_examination.csv')

    df['overweight'] = (df['weight'] / ((df['height'] / 100) ** 2) > 25).astype(int)
    
    df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
    df['gluc'] = (df['gluc'] > 1).astype(int)

    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['active', 'alco', 'cholesterol', 'gluc', 'overweight', 'smoke'])

    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='total')

    g = sns.catplot(
        x='variable', y='total', hue='value', col='cardio',
        data=df_cat, kind='bar', height=6, aspect=1.2
    )

    g.set_axis_labels('variable', 'total')
    g.set_titles('cardio = {col_name}')
    g.despine(left=True)
    
    g.savefig('catplot.png')

    fig = g.figure

    fig.savefig('catplot.png')
    return fig
draw_cat_plot()


# 10
def draw_heat_map():
    df_heat = pd.read_csv('medical_examination.csv')
    
    df_heat = df_heat[
        (df_heat['height'] >= df_heat['height'].quantile(0.025)) & 
        (df_heat['height'] <= df_heat['height'].quantile(0.975)) &
        (df_heat['weight'] >= df_heat['weight'].quantile(0.025)) & 
        (df_heat['weight'] <= df_heat['weight'].quantile(0.975))
    ]

    corr = df_heat.corr()

    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(11, 9))

    sns.heatmap(corr, mask=mask, cmap='coolwarm', vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)

    fig.savefig('heatmap.png')
    return fig
draw_heat_map()
