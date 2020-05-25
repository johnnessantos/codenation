
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import scipy.stats as sct
import statsmodels.api as sm
"""
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer
"""

class Experimentos():
    def __init__(self):
        """
        Classe responsavel pela aplicação de experimentos
        """
        self.tranformadores_ = None
    
    def aplicar_tranformadores(self, X, transformadores=None, random_state = 42):
        """
        https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html
        Função responsavel por aplicar transformadores nas variáveis
        Parâmetros:
        ----------
            X: numpy.array
                A distribuição a analisar
            transformadores: list(tuples)
                Lista de tuplas de objetos sklearn preprocessing
        Retornos:
        --------
            g: grafico
                Um grafico do matplot
        """
        """
        # Atualizando os tranformadores aplicados nos dados
        if transformadores:
            self.tranformadores_ = transformadores
        else:
            self.transformadores_ = [
                ('StandardScaler', StandardScaler().fit_transform(X)),
                ('MinMaxScaler', MinMaxScaler().fit_transform(X)),
                ('MaxAbsScaler', MaxAbsScaler().fit_transform(X)),
                ('RobustScaler', RobustScaler().fit_transform(X)), 
                ('PowerTransformerYeoJohnson', PowerTransformer(method='yeo-johnson').fit_transform(X)), 
                ('PowerTransformerBoxCox', PowerTransformer(method='box-cox').fit_transform(X)),
                ('QuantileTransformerUniform', QuantileTransformer(output_distribution='uniform').fit_transform(X)),
                ('QuantileTransformerNorm', QuantileTransformer(output_distribution='normal').fit_transform(X))
            ]

        transformador = self.tranformadores_[0]
        return transformador
        """
        pass

    def criar_histogramas(self, df, colunas, col_wrap=3, height=4):
        """
        Gera grafico de histograma de váriaveis númericas
        Parâmetros:
        ----------
            df: DataFrame
                DataFrame com os dados
            colunas: list or array
                Lista de colunas para exibir o histograma
            col_wrap: int
                Ajuste no número de gráficos por linha
            height: int
                Altura de cada elemento do grafico
        Retornos:
        --------
            g: Gráfico
        """
        f = pd.melt(df, value_vars=colunas, var_name='var', value_name='valor') 
        g = sns.FacetGrid(f, col='var',  col_wrap=col_wrap, height=height, sharex=False, sharey=False)
        g = g.map(sns.distplot, 'valor')
        return g

    def verificar_distribuicao_normal(self, arr, p_value=0.05):
        """
        Função responsavel por verificar a normalidade de uma distribuição utilizando o método jarque bera.
        Sendo assim rejeitar_h0 sendo `False` então a distribuição é normal
        Parâmetros:
            arr: list or array
            p-value: float (Nível de significância padrão p_value=0.05)
        Retornos:
            rejeitar_h0: bool
        """
        return (sct.jarque_bera(arr)[1], bool(sct.jarque_bera(arr)[1] >= p_value))
    
    def deteccao_anomalia(self, arr, remover_nans=True):
        """
        Função para detecção de anomalia e retorna a lista de indices dos registros
        Parâmetros:
        ----------
            arr: list or array
            remover_nans: bool (default=True)
        Retornos:
        --------
            xt: tuple(limite_inferior, limite_superior, indeces dos registros)
        """
        if remover_nans:
            arr = pd.Series(arr).dropna()
        else:
            arr = pd.Series(arr)
        
        quartiles = arr.quantile([0.25, 0.75]).values
        IQR = quartiles[1]-quartiles[0]
        limite_inferior, limite_superior = quartiles[0]-1.5*IQR, quartiles[1]+1.5*IQR
        return (limite_inferior, 
                limite_superior, 
                list(arr.index[np.logical_or(arr<limite_inferior, arr>limite_superior)]))