import random

import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import numpy as np

# l=[]
# for x in range(80):
#     l.append([1,1])
# for y in range(1):
#     l.append([2,1])
# exog=np.array(l)
# q,r = np.linalg.qr(exog)
# print(np.dot(q,r))
a=np.array([[-0.5,-0.5,-0.5,-0.5],
            [-0.5,-0.5,0.5,0.5]])
b=np.array([2,2,2,2])
print(a,b)
print(np.dot(a,b))
# coh_list=[0,1,3,3]
coh_list=[]
for x in range(4):
     coh_list.append(2)
id=[]
for x in range(2):
    id.append(2)
for x in range(2):
    id.append(1)

data = {'id': id,
        'coherence': coh_list
        }

df = pd.DataFrame(data)
print(df)
print(ols('coherence ~ C(id)', df).fit())
anova_res = anova_lm(ols('coherence ~ C(id)', df).fit(), typ=1)

print(anova_res)
print(type(anova_res.loc['C(id)']['PR(>F)']))