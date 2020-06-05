import os
import sys

import pandas as pd

class FonteDados():
    def __init__(self):
        """
        Classe responsavel pela leitura dos dados.
        
        """
        self.diretorio_raiz = os.path.abspath(os.path.join(os.path.join(os.path.dirname(__file__), '..')))
        self.diretorio_dados = os.path.join(self.diretorio_raiz, 'data')

    def carregar_dados(self):
        """
        Carrega os dados de e os retorna em um DataFrame
        Parâmetros:
        ----------
            None
        Retornos:
        --------
            X: DataFrame
        """

        return pd.read_csv(os.path.join(self.diretorio_dados, 'estaticos_market.csv'), index_col=0)

    def carregar_portifolio(self, portifolio=1):
        """
        Metodo para retornar o dataframe do portifolio de acordo com o numero
        Parâmetros:
        ----------
            portifolio: int (padrão 1)
        Retornos:
        --------
            X: DataFrame
        """

        return pd.read_csv(os.path.join(self.diretorio_dados, f'estaticos_portfolio{portifolio}.csv'), index_col=0)


    def carregar_dicionario_dados(self):
        """
        Metodo que carrega os dados do dicionario dos dados
        Parâmetros:
        ----------
            None
        Retornos:
        --------
            X: DataFrame
        """

        return pd.read_csv(os.path.join(self.diretorio_dados, 'features_dictionary.csv'), sep=';')

    def gerar_portifolio(self, dados_fonte, portifolio_ids):
        """
        Metodo responsável por gerar um portifolio realizando o join com os dados fonte `market`
        Parâmetros:
        ----------
            dados_fonte: DataFrame
                Dados de `market` que serão utilizados como fonte.
            portifolio_ids: DataFrame
                DataFrame com os `ids` das empresas.

        Retornos:
        --------
            df_novo: DataFrame
                DataFrame com os dodos de `dados_fonte` de acordo com os ids de `portifolio_ids`
        """

        return portifolio_ids[['id']].join(dados_fonte.set_index('id'), on='id', how='inner')

"""
if __name__ == "__main__":
    fonte_dados = FonteDados()

    print(fonte_dados.diretorio_raiz)
"""