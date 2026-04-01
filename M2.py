import pandas as pd
import numpy as np
from linearmodels.panel import PanelOLS
import statsmodels.api as sm
from linearmodels.panel import RandomEffects
from scipy import stats

df = pd.read_csv('dataset.csv')
df = df.set_index(['Country', 'Year'])

df_clean = df[df['NATO_EXP_EQUIPMENT'] > 0].copy()

df_clean['log_GFCF'] = np.log(df_clean['GFCF'])
df_clean['log_Equip'] = np.log(df_clean['NATO_EXP_EQUIPMENT'])
df_clean['Social_Exp'] = df_clean['Education'] + df_clean['Health']
df_clean['log_Social'] = np.log(df_clean['Social_Exp'])

producers = ['Poland', 'Greece']
is_prod = df_clean.index.get_level_values('Country').isin(producers)
df_clean['Equip_Importers'] = df_clean['log_Equip']
df_clean.loc[is_prod, 'Equip_Importers'] = 0
df_clean['Equip_Producers'] = df_clean['log_Equip']
df_clean.loc[~is_prod, 'Equip_Producers'] = 0

X = sm.add_constant(df_clean[['Equip_Importers', 'Equip_Producers', 'log_Social', 'GDP_Growth']])
model_nato = PanelOLS(df_clean['log_GFCF'], X, 
                      entity_effects=True, 
                      time_effects=True).fit(cov_type='clustered', cluster_entity=True)

print(model_nato.summary)




