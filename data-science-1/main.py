#!/usr/bin/env python
# coding: utf-8

# # Desafio 3
# 
# Neste desafio, iremos praticar nossos conhecimentos sobre distribuições de probabilidade. Para isso,
# dividiremos este desafio em duas partes:
#     
# 1. A primeira parte contará com 3 questões sobre um *data set* artificial com dados de uma amostra normal e
#     uma binomial.
# 2. A segunda parte será sobre a análise da distribuição de uma variável do _data set_ [Pulsar Star](https://archive.ics.uci.edu/ml/datasets/HTRU2), contendo 2 questões.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sct
import seaborn as sns
from statsmodels.distributions.empirical_distribution import ECDF


# In[4]:


#%matplotlib inline

from IPython.core.pylabtools import figsize


figsize(12, 8)

sns.set()


# ## Parte 1

# ### _Setup_ da parte 1

# In[6]:


np.random.seed(42)
    
dataframe = pd.DataFrame({"normal": sct.norm.rvs(20, 4, size=10000),
                     "binomial": sct.binom.rvs(100, 0.2, size=10000)})


# ## Inicie sua análise a partir da parte 1 a partir daqui

# In[ ]:





# In[8]:


# Sua análise da parte 1 começa aqui.

# Processo de verificaçao dos dados
# 1. Verificar a quantidade de linhas e colunas
# 2. Verificar os primeiros registros
# 3. Verificar os tipos de dados
# 4. Verificar informações estatisticas


# 1. Verificar a quantidade de linhas e colunas (ok)
dataframe.shape


# In[10]:


# 2. Verificar os primeiros registros (ok)

dataframe.head()


# In[12]:


# 3. Verificar os tipos de dados (ok)

dataframe.info()


# In[14]:


# 4. Verificar informações estatisticas

dataframe.describe().T


# # Analise exploratória dos dados

# ## Questão 1
# 
# Qual a diferença entre os quartis (Q1, Q2 e Q3) das variáveis `normal` e `binomial` de `dataframe`? Responda como uma tupla de três elementos arredondados para três casas decimais.
# 
# Em outra palavras, sejam `q1_norm`, `q2_norm` e `q3_norm` os quantis da variável `normal` e `q1_binom`, `q2_binom` e `q3_binom` os quantis da variável `binom`, qual a diferença `(q1_norm - q1 binom, q2_norm - q2_binom, q3_norm - q3_binom)`?

# In[16]:


def q1():
    dataframe_stats = dataframe.describe()
    return (round(dataframe_stats['normal']['25%'] - dataframe_stats['binomial']['25%'], 3), 
            round(dataframe_stats['normal']['50%'] - dataframe_stats['binomial']['50%'], 3),
            round(dataframe_stats['normal']['75%'] - dataframe_stats['binomial']['75%'], 3)
    )


# Para refletir:
# 
# * Você esperava valores dessa magnitude?
# 
# * Você é capaz de explicar como distribuições aparentemente tão diferentes (discreta e contínua, por exemplo) conseguem dar esses valores?

# ## Questão 2
# 
# Considere o intervalo $[\bar{x} - s, \bar{x} + s]$, onde $\bar{x}$ é a média amostral e $s$ é o desvio padrão. Qual a probabilidade nesse intervalo, calculada pela função de distribuição acumulada empírica (CDF empírica) da variável `normal`? Responda como uma único escalar arredondado para três casas decimais.

# In[18]:


def q2():
    normal_mean = dataframe['normal'].mean()
    normal_std = dataframe['normal'].std()
    
    # Calculating the intervals
    interval1 = normal_mean - normal_std
    interval2 = normal_mean + normal_std

    # Values applied to the ECDF function
    interval1 = ECDF(dataframe['normal'])(interval1)
    interval2 = ECDF(dataframe['normal'])(interval2)
    
    return float(round(interval2 - interval1, 3))


# Para refletir:
# 
# * Esse valor se aproxima do esperado teórico?
# * Experimente também para os intervalos $[\bar{x} - 2s, \bar{x} + 2s]$ e $[\bar{x} - 3s, \bar{x} + 3s]$.

# ## Questão 3
# 
# Qual é a diferença entre as médias e as variâncias das variáveis `binomial` e `normal`? Responda como uma tupla de dois elementos arredondados para três casas decimais.
# 
# Em outras palavras, sejam `m_binom` e `v_binom` a média e a variância da variável `binomial`, e `m_norm` e `v_norm` a média e a variância da variável `normal`. Quais as diferenças `(m_binom - m_norm, v_binom - v_norm)`?

# In[20]:


def q3():
    binomial_mean, binomial_variance = dataframe['binomial'].mean(), dataframe['binomial'].var()
    normal_mean, normal_variance = dataframe['normal'].mean(), dataframe['normal'].var()
    
    return (round(binomial_mean - normal_mean, 3), round(binomial_variance - normal_variance, 3))


# Para refletir:
# 
# * Você esperava valore dessa magnitude?
# * Qual o efeito de aumentar ou diminuir $n$ (atualmente 100) na distribuição da variável `binomial`?

# ## Parte 2

# ### _Setup_ da parte 2

# In[22]:


stars = pd.read_csv("pulsar_stars.csv")

stars.rename({old_name: new_name
              for (old_name, new_name)
              in zip(stars.columns,
                     ["mean_profile", "sd_profile", "kurt_profile", "skew_profile", "mean_curve", "sd_curve", "kurt_curve", "skew_curve", "target"])
             },
             axis=1, inplace=True)

stars.loc[:, "target"] = stars.target.astype(bool)


# ## Inicie sua análise da parte 2 a partir daqui

# In[24]:


# Sua análise da parte 2 começa aqui.
# Processo de verificaçao dos dados
# 1. Verificar a quantidade de linhas e colunas
# 2. Verificar os primeiros registros
# 3. Verificar os tipos de dados
# 4. Verificar informações estatisticas


# In[26]:


# 1. Verificar a quantidade de linhas e colunas
print(f'shape of stars: {stars.shape}')

# 2. Verificar os primeiros registros
stars.head()


# In[28]:


# 3. Verificar os tipos de dados
print(stars.info())


# In[30]:


# 4. Verificar informações estatisticas
stars.describe().T


# ## Questão 4
# 
# Considerando a variável `mean_profile` de `stars`:
# 
# 1. Filtre apenas os valores de `mean_profile` onde `target == 0` (ou seja, onde a estrela não é um pulsar).
# 2. Padronize a variável `mean_profile` filtrada anteriormente para ter média 0 e variância 1.
# 
# Chamaremos a variável resultante de `false_pulsar_mean_profile_standardized`.
# 
# Encontre os quantis teóricos para uma distribuição normal de média 0 e variância 1 para 0.80, 0.90 e 0.95 através da função `norm.ppf()` disponível em `scipy.stats`.
# 
# Quais as probabilidade associadas a esses quantis utilizando a CDF empírica da variável `false_pulsar_mean_profile_standardized`? Responda como uma tupla de três elementos arredondados para três casas decimais.

# In[32]:


def q4():
    false_pulsar_mean_profile = stars.loc[stars['target']==False, 'mean_profile']
    
    false_pulsar_mean_profile_standardized = sct.zscore(false_pulsar_mean_profile)
    
    # Percent point function
    ppf_q80 = sct.norm.ppf(0.80, loc=0, scale=1)
    ppf_q90 = sct.norm.ppf(0.90, loc=0, scale=1)
    ppf_q95 = sct.norm.ppf(0.95, loc=0, scale=1)
    
    # Create the ecdf function with standardized star data
    compute_ecdf_stars = ECDF(false_pulsar_mean_profile_standardized)
    
    return (round(compute_ecdf_stars(ppf_q80), 3),
           round(compute_ecdf_stars(ppf_q90), 3),
           round(compute_ecdf_stars(ppf_q95), 3)
    )


# Para refletir:
# 
# * Os valores encontrados fazem sentido?
# * O que isso pode dizer sobre a distribuição da variável `false_pulsar_mean_profile_standardized`?

# ## Questão 5
# 
# Qual a diferença entre os quantis Q1, Q2 e Q3 de `false_pulsar_mean_profile_standardized` e os mesmos quantis teóricos de uma distribuição normal de média 0 e variância 1? Responda como uma tupla de três elementos arredondados para três casas decimais.

# In[34]:


def q5():
    false_pulsar_mean_profile = stars.loc[stars['target']==False, 'mean_profile']
    
    false_pulsar_mean_profile_standardized = sct.zscore(false_pulsar_mean_profile)
    
    # Quartis q1, q2 and q3 stardardized
    ppf_q1 = sct.norm.ppf(0.25, loc=0, scale=1)
    ppf_q2 = sct.norm.ppf(0.50, loc=0, scale=1)
    ppf_q3 = sct.norm.ppf(0.75, loc=0, scale=1)
    
    false_pulsar_mean_profile_standardized_q1 = np.percentile(false_pulsar_mean_profile_standardized, 25)
    false_pulsar_mean_profile_standardized_q2 = np.percentile(false_pulsar_mean_profile_standardized, 50)
    false_pulsar_mean_profile_standardized_q3 = np.percentile(false_pulsar_mean_profile_standardized, 75)
    
    return (round(false_pulsar_mean_profile_standardized_q1 - ppf_q1, 3),
            round(false_pulsar_mean_profile_standardized_q2 - ppf_q2, 3),
            round(false_pulsar_mean_profile_standardized_q3 - ppf_q3, 3)
    )


# Para refletir:
# 
# * Os valores encontrados fazem sentido?
# * O que isso pode dizer sobre a distribuição da variável `false_pulsar_mean_profile_standardized`?
# * Curiosidade: alguns testes de hipóteses sobre normalidade dos dados utilizam essa mesma abordagem.
