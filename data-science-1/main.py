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

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sct
import seaborn as sns
from statsmodels.distributions.empirical_distribution import ECDF


# In[2]:


#%matplotlib inline

from IPython.core.pylabtools import figsize


figsize(12, 8)

sns.set()


# ## Parte 1

# ### _Setup_ da parte 1

# In[3]:


np.random.seed(42)
    
dataframe = pd.DataFrame({"normal": sct.norm.rvs(20, 4, size=10000),
                     "binomial": sct.binom.rvs(100, 0.2, size=10000)})


# ## Inicie sua análise a partir da parte 1 a partir daqui

# In[ ]:





# In[4]:


# Sua análise da parte 1 começa aqui.

# Processo de verificaçao dos dados
# 1. Verificar a quantidade de linhas e colunas
# 2. Verificar os primeiros registros
# 3. Verificar os tipos de dados
# 4. Verificar informações estatisticas


# 1. Verificar a quantidade de linhas e colunas (ok)
dataframe.shape


# In[5]:


# 2. Verificar os primeiros registros (ok)

dataframe.head()


# In[6]:


# 3. Verificar os tipos de dados (ok)

dataframe.info()


# In[7]:


# 4. Verificar informações estatisticas

dataframe.describe().T


# # Analise exploratória dos dados

# ### Demonstrando a distribuição dos dados

# In[8]:


sns.pairplot(data=dataframe)
plt.suptitle('Gráficos de histograma e dispersão das váriaveis')
plt.show()


# ### Bloxplot: Observar a variação por quartis

# In[9]:


plt.subplot(221)
fig1 = sns.boxplot(dataframe['normal'], orient='v')

plt.subplot(222)
fig2 = sns.boxplot(dataframe['binomial'], orient='v')

plt.suptitle('Distribuição das váriaveis ao longo dos quartis')
plt.show()


# # Plotando a função ECDF

# In[10]:


# Definition of function
# Ref. https://campus.datacamp.com/courses/statistical-thinking-in-python-part-1/graphical-exploratory-data-analysis?ex=12
def compute_ecdf(data):
    """Compute ECDF for a one-dimensional array of measurements."""
    # Number of data points: n
    n = len(data)

    # x-data for the ECDF: x
    x = np.sort(data)

    # y-data for the ECDF: y
    y = np.arange(1, n+1) / n

    return x, y


# In[11]:


# Calculating series to plot ecdf
x_normal, y_normal = compute_ecdf(dataframe['normal'])
x_binomial, y_binomial = compute_ecdf(dataframe['binomial'])

# Using number of lines equal to 1 and columns 2 for plotting
plt.subplot(1, 2, 1)
ax_normal = plt.plot(x_normal, y_normal, marker='.', linestyle='none')
plt.xlabel('normal')
plt.ylabel('ECDF')

plt.subplot(1, 2, 2)
ax_binomial = plt.plot(x_binomial, y_binomial, marker='.', linestyle='none')
plt.xlabel('binomial')
plt.ylabel('ECDF')

plt.suptitle('Função de distribuição acumulada empírica')
plt.show()


# ## Questão 1
# 
# Qual a diferença entre os quartis (Q1, Q2 e Q3) das variáveis `normal` e `binomial` de `dataframe`? Responda como uma tupla de três elementos arredondados para três casas decimais.
# 
# Em outra palavras, sejam `q1_norm`, `q2_norm` e `q3_norm` os quantis da variável `normal` e `q1_binom`, `q2_binom` e `q3_binom` os quantis da variável `binom`, qual a diferença `(q1_norm - q1 binom, q2_norm - q2_binom, q3_norm - q3_binom)`?

# In[12]:


def q1():
    return tuple(round(dataframe['normal'].quantile([0.25, 0.50, 0.75])-dataframe['binomial'].quantile([0.25, 0.50, 0.75]), 3))


# Para refletir:
# 
# * Você esperava valores dessa magnitude?
# 
# * Você é capaz de explicar como distribuições aparentemente tão diferentes (discreta e contínua, por exemplo) conseguem dar esses valores?

# ## Questão 2
# 
# Considere o intervalo $[\bar{x} - s, \bar{x} + s]$, onde $\bar{x}$ é a média amostral e $s$ é o desvio padrão. Qual a probabilidade nesse intervalo, calculada pela função de distribuição acumulada empírica (CDF empírica) da variável `normal`? Responda como uma único escalar arredondado para três casas decimais.

# In[13]:


def q2():
    normal_mean = dataframe['normal'].mean()
    normal_std = dataframe['normal'].std()
    
    # Calculating the intervals
    interval1 = normal_mean - normal_std
    interval2 = normal_mean + normal_std

    # Definition of function ecdf
    ecdf_normal_func = ECDF(dataframe['normal'])

    # Values applied to the ECDF function
    interval1 = ecdf_normal_func(interval1)
    interval2 = ecdf_normal_func(interval2)
    
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

# In[14]:


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

# In[15]:


stars = pd.read_csv("pulsar_stars.csv")

stars.rename({old_name: new_name
              for (old_name, new_name)
              in zip(stars.columns,
                     ["mean_profile", "sd_profile", "kurt_profile", "skew_profile", "mean_curve", "sd_curve", "kurt_curve", "skew_curve", "target"])
             },
             axis=1, inplace=True)

stars.loc[:, "target"] = stars.target.astype(bool)


# ## Inicie sua análise da parte 2 a partir daqui

# In[16]:


# Sua análise da parte 2 começa aqui.
# Processo de verificaçao dos dados
# 1. Verificar a quantidade de linhas e colunas
# 2. Verificar os primeiros registros
# 3. Verificar os tipos de dados
# 4. Verificar informações estatisticas


# In[17]:


# 1. Verificar a quantidade de linhas e colunas
print(f'shape of stars: {stars.shape}')

# 2. Verificar os primeiros registros
stars.head()


# In[18]:


# 3. Verificar os tipos de dados
print(stars.info())


# In[19]:


# 4. Verificar informações estatisticas
stars.describe().T


# # Analise exploratória

# In[20]:


# Obtain numerical data for analysis
stars_numeric = stars.select_dtypes(include=np.number)

ax = sns.pairplot(stars_numeric)

#plt.suptitle('Visão geral dos dados')
plt.show()


# In[21]:


# Counter of target
stars_target_count = stars['target'].value_counts()

# Plot
ax = sns.barplot(x=stars_target_count.index, y=stars_target_count.values)

plt.title('Quantidade por categoria no target')
plt.show()


# In[22]:


# Plotting correlations
ax = sns.heatmap(stars.corr(), annot=True, cmap='Blues')
plt.show()


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

# In[23]:


def q4():
    false_pulsar_mean_profile = stars.loc[stars['target']==False, 'mean_profile']
    
    false_pulsar_mean_profile_standardized = sct.zscore(false_pulsar_mean_profile)
    
    # Create the ecdf function with standardized star data
    compute_ecdf_stars = ECDF(false_pulsar_mean_profile_standardized)
    
    # Percent point function
    ppf_qs = sct.norm.ppf([0.80, 0.90, 0.95], loc=0, scale=1)
    
    return tuple([round(compute_ecdf_stars(x), 3) for x in ppf_qs])


# In[ ]:





# Para refletir:
# 
# * Os valores encontrados fazem sentido?
# * O que isso pode dizer sobre a distribuição da variável `false_pulsar_mean_profile_standardized`?

# ## Questão 5
# 
# Qual a diferença entre os quantis Q1, Q2 e Q3 de `false_pulsar_mean_profile_standardized` e os mesmos quantis teóricos de uma distribuição normal de média 0 e variância 1? Responda como uma tupla de três elementos arredondados para três casas decimais.

# In[24]:


def q5():
    false_pulsar_mean_profile = stars.loc[stars['target']==False, 'mean_profile']
    
    false_pulsar_mean_profile_standardized = sct.zscore(false_pulsar_mean_profile)

    # Calculating quartiles
    ppf_qs = sct.norm.ppf([0.25, 0.50, 0.75], loc=0, scale=1)
    false_pulsar_mean_profile_standardized_qs = np.percentile(false_pulsar_mean_profile_standardized, [25, 50, 75])
    
    return tuple([round(false_pulsar_mean_profile_standardized_qs[i]-ppf_qs[i], 3) for i in range(0, len(ppf_qs))])


# Para refletir:
# 
# * Os valores encontrados fazem sentido?
# * O que isso pode dizer sobre a distribuição da variável `false_pulsar_mean_profile_standardized`?
# * Curiosidade: alguns testes de hipóteses sobre normalidade dos dados utilizam essa mesma abordagem.
