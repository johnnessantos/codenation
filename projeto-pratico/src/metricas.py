import os
import sys

from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score

class Metricas():
    def __init__(self):
        """
        Classe responsável pela metrica aplicada para se basear na escolha do método
        """

    def calcular_pontuacao(self, X, categorias, metrica='euclidean', tam_amostra=1000):
        """
        Metrica para calcula a silhueta do cluster com o método `silhouette`, `calinski_harabasz`
        e `davies_bouldin`.
        A metrica silhouette com distancia euclidiana o melhor valor é 1 e o pior é -1. Em caso
        onde o valor for 0 significa que os agrupametos estão sobrepostos.
        A metrica calinski_harabasz quanto maior mais denso e mais bem separados está o cluster.
        A metrica davies_bouldin quanto mais próximo de zero melhor, ele se baseia em similaridade
        entre agrupamentos se baseando na média das distancias do próprio agrupamento.
        Parâmetros:
        ----------
            X: List
                Resultado do cluster
            tam_amostra: List
                Resposta para os dados
        """

        return {
            f'silhouette_score_{metrica}': silhouette_score(X, categorias, 
                                            metric=metrica, sample_size=tam_amostra, random_state=42),
            'calinski_harabasz_score': calinski_harabasz_score(X, categorias),
            'davies_bouldin_score': davies_bouldin_score(X, categorias)
        }
"""
if __name__ == "__main__":
    metricas = Metricas()
    X  = [[ 0.6749994 ,  0.32689971,  1.15461469,  5.19933758, -5.19933758,
        -5.19933758, -5.19933758, -5.19933758, -5.19933758,  5.19933758,
         5.19933758,  5.19933758, -5.19933758,  2.15550892,  1.26806948,
        -5.19933758, -5.19933758,  5.19933758, -5.19933758, -5.19933758,
        -5.19933758, -5.19933758, -5.19933758,  5.19933758, -5.19933758,
        -5.19933758, -5.19933758, -5.19933758],
       [-1.0663039 , -1.6252744 , -0.21363612,  5.19933758, -5.19933758,
        -5.19933758,  5.19933758, -5.19933758, -5.19933758,  5.19933758,
         5.19933758,  5.19933758, -5.19933758,  0.06528392, -0.00627288,
        -5.19933758, -5.19933758, -5.19933758, -5.19933758, -5.19933758,
         5.19933758, -5.19933758, -5.19933758, -5.19933758, -5.19933758,
        -5.19933758,  5.19933758, -5.19933758],
       [ 0.05246503, -0.83689206, -0.21363612,  5.19933758, -5.19933758,
        -5.19933758,  5.19933758, -5.19933758, -5.19933758, -5.19933758,
         5.19933758,  5.19933758, -5.19933758, -1.23127984, -1.25298763,
        -5.19933758, -5.19933758, -5.19933758, -5.19933758, -5.19933758,
         5.19933758, -5.19933758, -5.19933758, -5.19933758, -5.19933758,
        -5.19933758,  5.19933758, -5.19933758]]

    print(metricas.calcular_pontuacao(X, [0, 1, 1]))
"""