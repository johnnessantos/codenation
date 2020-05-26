import os
import sys

import numpy as np
import pandas as pd
import pytest

# Adicionando o diretorio raiz no ambiente para possibilitar realizar importações
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils import Utils


class TestUtils():
    
    @pytest.fixture(scope='module')
    def instancia_utils(self):
        return Utils()

    @pytest.mark.parametrize("variavel, esperado", [
        ('fl_matriz', 'boolean value, true if the CNPJ corresponds to the "matriz".'),
        ('meses_ultima_contratacaco', 'numeric, months since the last hire.')
    ])
    def test_descrever_variavel_pertencente(self, variavel, esperado, instancia_utils):
        assert instancia_utils.descrever_variavel(variavel)[1] == esperado
    
    def test_descrever_variavel_nao_pertencente(self, instancia_utils):
        esperado = 'Variável não encontrada no dicionário'
        assert instancia_utils.descrever_variavel('id')[1] == esperado

    def test_descrever_dataframe(self, instancia_utils):
        df = pd.DataFrame({
            'coluna1': [1, 2, 3, 4, np.nan]
        })

        df = instancia_utils.descrever_dataframe(df)
        assert  df.quantidade_nulos.sum() == 1



    
