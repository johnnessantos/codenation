import os

import numpy as np
import pandas as pd

class Utils:
    """
    Classe responsavel por conter metodos que podem ser executados por qualquer arquivo python
    """
    def __init__(self):
        self.diretorio_dados = os.path.abspath(os.path.join(os.path.join(os.path.dirname(__file__), '..'), 'data'))
        self.dicionario_variaveis = pd.read_csv(os.path.join(self.diretorio_dados, 'features_dictionary.csv'), sep=';')

    def descrever_variavel(self, variavel=None):
        """
        Função com intenção de obter a descrição de uma coluna especifica dos dados fonte.
        Parâmetros:
        ----------
            variavel: String
        """

        if variavel:
            variavel_descricao = self.dicionario_variaveis[self.dicionario_variaveis.feature == variavel].values
            if variavel_descricao.size > 0:
                return (tuple(variavel_descricao[0]))
            else:
                return ((variavel, 'Variável não encontrada no dicionário'))

    def descrever_dataframe(self, df):
        """
        Função geradora de uma descrição de informações do dataframe, como: colunas, tipos das colunas,
        quantidade de nulos, percentual de nulos.
        Parâmetros:
        ----------
            df: DataFrame
        Retornos:
        ----------
            df_novo: DataFrame
        """
        df_novo = pd.DataFrame({
            'coluna': df.columns.values,
            'tipo': df.dtypes.values,
            'quantidade_nulos': df.isna().sum().values
        })
        df_novo['porcentagem_nulos'] = df_novo['quantidade_nulos']/df.shape[0]

        return df_novo
