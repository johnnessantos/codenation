#!/usr/bin/env python
# coding: utf-8

# # Desafio 1
# 
# Para esse desafio, vamos trabalhar com o data set [Black Friday](https://www.kaggle.com/mehdidag/black-friday), que reúne dados sobre transações de compras em uma loja de varejo.
# 
# Vamos utilizá-lo para praticar a exploração de data sets utilizando pandas. Você pode fazer toda análise neste mesmo notebook, mas as resposta devem estar nos locais indicados.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Set up_ da análise

# In[3]:


import pandas as pd
import numpy as np


# In[4]:


black_friday = pd.read_csv("black_friday.csv")


# ## Inicie sua análise a partir daqui

# In[5]:


black_friday.head()


# In[6]:


black_friday.info()


# In[7]:


black_friday.describe().T


# In[8]:


black_friday_info = pd.DataFrame(
    {
        'colunas': black_friday.columns,
        'types': black_friday.dtypes,
        'missing': black_friday.isna().sum(),
        'missing_percentage': black_friday.isna().sum() / black_friday.shape[0]
    })

black_friday_info


# In[9]:


# Consultar os dados com todos os registros preenchidos
black_friday_not_nan = black_friday.dropna()

black_friday_not_nan.head()


# ## Questão 1
# 
# Quantas observações e quantas colunas há no dataset? Responda no formato de uma tuple `(n_observacoes, n_colunas)`.

# In[10]:


def q1():
    # Retorne aqui o resultado da questão 1.
    return black_friday.shape


# ## Questão 2
# 
# Há quantas mulheres com idade entre 26 e 35 anos no dataset? Responda como um único escalar.

# In[11]:


def q2():
    return black_friday[(black_friday.Gender == 'F')
            & (black_friday.Age == '26-35')].User_ID.nunique()


# ## Questão 3
# 
# Quantos usuários únicos há no dataset? Responda como um único escalar.

# In[12]:


def q3():
    return black_friday['User_ID'].nunique()


# ## Questão 4
# 
# Quantos tipos de dados diferentes existem no dataset? Responda como um único escalar.

# In[13]:


def q4():
    return black_friday.dtypes.nunique()


# ## Questão 5
# 
# Qual porcentagem dos registros possui ao menos um valor null (`None`, `ǸaN` etc)? Responda como um único escalar entre 0 e 1.

# In[14]:


def q5():
    black_friday_not_nan = black_friday.dropna()
    return 1 - black_friday_not_nan.shape[0]/black_friday.shape[0]


# ## Questão 6
# 
# Quantos valores null existem na variável (coluna) com o maior número de null? Responda como um único escalar.

# In[15]:


def q6():
    black_friday_nan_acc = black_friday.isna().sum()
    return max(black_friday_nan_acc)


# ## Questão 7
# 
# Qual o valor mais frequente (sem contar nulls) em `Product_Category_3`? Responda como um único escalar.

# In[16]:


def q7():
    return black_friday.Product_Category_3.dropna().value_counts().sort_values(ascending=False).index[0]


# ## Questão 8
# 
# Qual a nova média da variável (coluna) `Purchase` após sua normalização? Responda como um único escalar.

# In[20]:


def q8():
    purchase_min = black_friday['Purchase'].min()
    purchase_max = black_friday['Purchase'].max()
    purshase_min_max = purchase_max-purchase_min

    purchase_norm = (black_friday['Purchase'] - purchase_min)/purshase_min_max
    return float(purchase_norm.mean())


# ## Questão 9
# 
# Quantas ocorrências entre -1 e 1 inclusive existem da variável `Purchase` após sua padronização? Responda como um único escalar.

# In[18]:


def q9():
    purchase_mean = black_friday['Purchase'].mean()
    purchase_std = black_friday['Purchase'].std()
    black_friday['Purchase_Stand'] = (black_friday['Purchase']-purchase_mean)/purchase_std
    
    purchase_stand_count = black_friday[black_friday['Purchase_Stand'].between(-1, 1)].shape[0]
    return purchase_stand_count


# ## Questão 10
# 
# Podemos afirmar que se uma observação é null em `Product_Category_2` ela também o é em `Product_Category_3`? Responda com um bool (`True`, `False`).

# In[19]:


def q10():
    black_friday_products_nan = black_friday.loc[black_friday['Product_Category_2'].isna(), ['Product_Category_2', 'Product_Category_3']]
    return black_friday_products_nan['Product_Category_2'].equals(black_friday_products_nan['Product_Category_3'])


# In[ ]:




