import os
import sys

import pytest
import numpy as np
import scipy.stats as sct
from sklearn.preprocessing import StandardScaler

# Adicionando o diretorio raiz no ambiente para possibilitar realizar importações
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.experimentos import Experimentos

class TestExperimentos():
    @pytest.fixture(scope='class')
    def instancia_experimentos(self):
        return Experimentos()

    @pytest.fixture(scope='class')
    def dados(self):
        n_elementos = 1000
        return np.arange(1, n_elementos+1).reshape(-1, 1)

    def test_aplicar_tranformadores_numero_linhas(self, instancia_experimentos, dados):
        df = instancia_experimentos.aplicar_tranformadores(dados)
        assert df.shape[0] == 1000

    def test_aplicar_tranformadores_numero_colunas(self, instancia_experimentos, dados):
        df = instancia_experimentos.aplicar_tranformadores(dados)
        assert df.shape[1] == 8

    def test_aplicar_tranformadores_customizado(self, instancia_experimentos, dados):
        df = instancia_experimentos.aplicar_tranformadores(dados,
                                                        transformadores = [('StandardScaler', StandardScaler())])
        assert df.shape[0] == 1000
        assert df.shape[1] == 1

    def test_verificar_distribuicao_normal(self, instancia_experimentos):
        arr_norm = sct.norm.rvs(loc=10, scale=4, size=1000)
        assert instancia_experimentos.verificar_distribuicao_normal(arr_norm)[1] == True