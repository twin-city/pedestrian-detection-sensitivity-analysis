
#%% Analysis of p-value of cofactors (and plot it) for both MR and FPPI

import statsmodels.api as sm
# df_frame_metadata[["blizzard", "smog", "thunder"]] = pd.get_dummies(df_frame_metadata["weather"])[['BLIZZARD', 'SMOG', 'THUNDER']]


"""

#%% All at once
cofactors = ["is_night", 'pitch', 'roll', 'x', 'y', 'z','is_moving', "blizzard", "smog", "thunder"]
X = df_frame_metadata[cofactors]
Y = df_mr_fppi[df_mr_fppi["threshold"]==0.5]["MR"].loc[listint2liststr(X.index)]
X = sm.add_constant(X)
fit = sm.OLS(Y, X).fit()
for i, cofactor in enumerate(cofactors):
    print(fit.pvalues[i], cofactor)


#%% Separated
cofactors = ["is_night", 'pitch', 'roll', 'x', 'y', 'z','is_moving', "blizzard", "smog", "thunder"]

for cofactor in cofactors:
    X = df_frame_metadata[cofactor]
    Y = df_mr_fppi[df_mr_fppi["threshold"]==0.5]["MR"].loc[listint2liststr(X.index)]
    X = sm.add_constant(X)
    fit = sm.OLS(Y, X).fit()
    for i, cofactor in enumerate([cofactor]):
        print(fit.pvalues[i], cofactor)
"""



"""
Corriger BOnferroni + DataViz

Matrice de correlation des cofacteurs

Peut-être plutot les performances au niveau de la séquence ??? 
Par exemple pour weather oui... Car sinon les tests sont non indépendants

"""

"""

#%%
import pandas as pd
import seaborn as sns


# compute the correlation matrix using the corr method
correlation_matrix = df_frame_metadata[cofactors].corr()

# plot the correlation matrix using the heatmap function from seaborn
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()
"""

