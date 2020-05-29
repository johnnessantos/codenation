
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as sct
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer


class Experimentos():
    def __init__(self):
        """
        Classe responsavel pela aplicação de experimentos
        """
        self.tranformadores_ = None
    
    def aplicar_tranformadores(self, X, transformadores=None, random_state = 42):
        """
        Função responsavel por aplicar transformadores nas variáveis
        Parâmetros:
        ----------
            X: numpy.array
                A distribuição a analisar
            transformadores: list(tuples)
                Lista de tuplas de objetos sklearn preprocessing
        Retornos:
        --------
            df: DataFrame
                Dataframe pandas com as as colunas com os dados transformados

        Ref.: https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html
        """
        
        # Atualizando os tranformadores aplicados nos dados
        if not transformadores:
            transformadores = [
                ('StandardScaler', StandardScaler()),
                ('MinMaxScaler', MinMaxScaler()),
                ('MaxAbsScaler', MaxAbsScaler()),
                ('RobustScaler', RobustScaler()), 
                ('PowerTransformerYeoJohnson', PowerTransformer(method='yeo-johnson')), 
                ('PowerTransformerBoxCox', PowerTransformer(method='box-cox')),
                ('QuantileTransformerUniform', QuantileTransformer(output_distribution='uniform')),
                ('QuantileTransformerNorm', QuantileTransformer(output_distribution='normal'))
            ]
        
        dados_transformados = pd.DataFrame()
        for transformador in transformadores:
            dados_transformados[transformador[0]] = transformador[1].fit_transform(X)[:, 0]
        
        return dados_transformados

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

    def criar_diagrama_caixa(self, df, colunas, n_colunas=3, altura=4):
        """
        Gera gráfico com diversos diagramas de caixa conforme lista de `colunas`.
        A figura contém o tamanho sendo figsize(14, quantidade_linhas*altura) 
        mas pode ser redimensionado com fig.set_size_inches(12, 4).
        Parâmetros:
        ----------
            df: DataFrame
                Dados a serem plotados
            colunas: List
                Lista com as colunas do `df` para exibir
            n_colunas: int (default n_colunas=3)
                Número de colunas por linha
            altura: int (default altura=4)
                Altura de cada subplot em polegadas
        Retornos:
        --------
            fig: Figure
                Figura matplotlib
            ax: Axes
                Eixos do matpolotlib
        """
        quantidade_graficos = len(colunas)
        quantidade_linhas = quantidade_graficos//n_colunas
        
        if quantidade_graficos%n_colunas > 0:
            quantidade_linhas += 1


        fig, ax = plt.subplots(quantidade_linhas, n_colunas, figsize=(14, quantidade_linhas*altura))

        if quantidade_linhas > 1:
            for i in range(quantidade_graficos):
                sns.boxplot(df[colunas[i]].dropna(axis=0), orient='v', ax=ax[i//n_colunas, i%n_colunas])
                
            if quantidade_linhas*n_colunas != quantidade_graficos:
                for i in range(quantidade_graficos, quantidade_linhas*n_colunas):
                    ax[i//n_colunas, i%n_colunas].remove()
        else:
            for i in range(quantidade_graficos):
                sns.boxplot(df[colunas[i]].dropna(axis=0), orient='v', ax=ax[i])
            
            if quantidade_linhas*n_colunas != quantidade_graficos:
                for i in range(quantidade_graficos, quantidade_linhas*n_colunas):
                    ax[i].remove()
        return fig, ax

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

"""
if __name__ == "__main__":
    arr = np.arange(1, 1001).reshape(-1, 1)
    experimentos = Experimentos()
    print(experimentos.aplicar_tranformadores(arr))
"""