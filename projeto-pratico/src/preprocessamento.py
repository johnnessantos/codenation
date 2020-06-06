import os
import sys
import time

import numpy as np
import pandas as pd

from fonte_dados import FonteDados
from experimentos import Experimentos


class Preprocessamento():
    def __init__(self):
        """
        Classe responsavel pelo préprocessamento dos dados incluidos todas as funções e definições
        sobre os dados para que possa ser utilizado para servir os modelos.
        """
        self.variaveis_deletar = ['fl_epp', 'vl_total_tancagem', 'vl_total_veiculos_antt',
            'vl_total_veiculos_leves', 'vl_total_veiculos_pesados', 'qt_art',
            'vl_total_tancagem_grupo', 'vl_total_veiculos_antt_grupo',
            'vl_potenc_cons_oleo_gas', 'de_indicador_telefone', 'vl_frota',
            'qt_socios_st_suspensa', 'qt_socios_pep',
            'qt_alteracao_socio_total', 'qt_alteracao_socio_90d',
            'qt_alteracao_socio_180d', 'qt_alteracao_socio_365d',
            'qt_socios_pj_ativos', 'qt_socios_pj_nulos',
            'qt_socios_pj_baixados', 'qt_socios_pj_suspensos',
            'qt_socios_pj_inaptos', 'vl_idade_media_socios_pj',
            'vl_idade_maxima_socios_pj', 'vl_idade_minima_socios_pj',
            'qt_coligados', 'qt_socios_coligados', 'qt_coligados_matriz',
            'qt_coligados_ativo', 'qt_coligados_baixada',
            'qt_coligados_inapta', 'qt_coligados_suspensa',
            'qt_coligados_nula', 'idade_media_coligadas',
            'idade_maxima_coligadas', 'idade_minima_coligadas',
            'coligada_mais_nova_ativa', 'coligada_mais_antiga_ativa',
            'idade_media_coligadas_ativas', 'coligada_mais_nova_baixada',
            'coligada_mais_antiga_baixada', 'idade_media_coligadas_baixadas',
            'qt_coligados_sa', 'qt_coligados_me', 'qt_coligados_mei',
            'qt_coligados_ltda', 'qt_coligados_epp', 'qt_coligados_norte',
            'qt_coligados_sul', 'qt_coligados_nordeste', 'qt_coligados_centro',
            'qt_coligados_sudeste', 'qt_coligados_exterior',
            'qt_ufs_coligados', 'qt_regioes_coligados', 'qt_ramos_coligados',
            'qt_coligados_industria', 'qt_coligados_agropecuaria',
            'qt_coligados_comercio', 'qt_coligados_serviço',
            'qt_coligados_ccivil', 'qt_funcionarios_coligados',
            'qt_funcionarios_coligados_gp', 'media_funcionarios_coligados_gp',
            'max_funcionarios_coligados_gp', 'min_funcionarios_coligados_gp',
            'vl_folha_coligados', 'media_vl_folha_coligados',
            'max_vl_folha_coligados', 'min_vl_folha_coligados',
            'vl_folha_coligados_gp', 'media_vl_folha_coligados_gp',
            'max_vl_folha_coligados_gp', 'min_vl_folha_coligados_gp',
            'faturamento_est_coligados', 'media_faturamento_est_coligados',
            'max_faturamento_est_coligados', 'min_faturamento_est_coligados',
            'faturamento_est_coligados_gp',
            'media_faturamento_est_coligados_gp',
            'max_faturamento_est_coligados_gp',
            'min_faturamento_est_coligados_gp', 'total_filiais_coligados',
            'media_filiais_coligados', 'max_filiais_coligados',
            'min_filiais_coligados', 'qt_coligados_atividade_alto',
            'qt_coligados_atividade_medio', 'qt_coligados_atividade_baixo',
            'qt_coligados_atividade_mt_baixo',
            'qt_coligados_atividade_inativo', 'qt_coligadas',
            'sum_faturamento_estimado_coligadas', 'qt_ex_funcionarios',
            'qt_funcionarios_grupo', 'percent_func_genero_masc',
            'percent_func_genero_fem', 'idade_ate_18', 'idade_de_19_a_23',
            'idade_de_24_a_28', 'idade_de_29_a_33', 'idade_de_34_a_38',
            'idade_de_39_a_43', 'idade_de_44_a_48', 'idade_de_49_a_53',
            'idade_de_54_a_58', 'idade_acima_de_58',
            'grau_instrucao_macro_analfabeto',
            'grau_instrucao_macro_escolaridade_fundamental',
            'grau_instrucao_macro_escolaridade_media',
            'grau_instrucao_macro_escolaridade_superior',
            'grau_instrucao_macro_desconhecido', 'total',
            'meses_ultima_contratacaco', 'qt_admitidos_12meses',
            'qt_desligados_12meses', 'qt_desligados', 'qt_admitidos',
            'media_meses_servicos_all', 'max_meses_servicos_all',
            'min_meses_servicos_all', 'media_meses_servicos',
            'max_meses_servicos', 'min_meses_servicos', 'qt_funcionarios',
            'qt_funcionarios_12meses', 'qt_funcionarios_24meses',
            'tx_crescimento_12meses', 'tx_crescimento_24meses',
            'tx_rotatividade']

    def converter_sim_nao(self, df, colunas):
        """
        Função responsável pela conversão de valores de sim e não para 1 e 0. Em caso de `sim` o valor é 1.
        Parâmetros:
        ----------
            df: DataFrame
                Dados a serem convertidos
            colunas: array ou list
                Lista de colunas a aplicar a conversão
        Retornos:
        --------
            df_novo: DataFrame
                DataFrame com as colunas convertidas
        """
        df_novo = df.copy()
        df_novo[colunas] = df[colunas].replace({'SIM': 1, 'NAO': 0})
        return df_novo


    def converter_booleano(self, df, colunas):
        """
        Função responsável pela conversão de booleanos para 1 e 0. Em caso de `True` o valor é 1.
        Parâmetros:
        ----------
            df: DataFrame
                Dados a serem convertidos
            colunas: array ou list
                Lista de colunas a aplicar a conversão
        Retornos:
        --------
            df_novo: DataFrame
                DataFrame com as colunas convertidas
        """
        df_novo = df.copy()
        df_novo[colunas] = df[colunas].replace({True: 1, False: 0})
        return df_novo

    def moda_preenchendo_zero(self, X):
        """
        Metodo da moda estatistica modificado para que onde não haja moda retorne zero
        Parâmetros:
        ----------
            X: DataFrame
        Retornos:
        --------
            moda: float
        """
        moda = X.mode()
        return 0 if len(moda)==0 else moda

    def preencher_qt_socios(self, X):
        """
        Função que preenche os dados de `qt_socios` com a moda por agrupamento de natureza juridica
        Parâmetros:
        ----------
            X: DataFrame
        Retornos:
        --------
            X_novo: DataFrame
                O dataframe após aplicar transformação.
        """
        X_novo = X.copy()
        qt_socios_agrupado = X_novo.groupby(by=['de_natureza_juridica']).agg(
            {
                'qt_socios': self.moda_preenchendo_zero
            }).to_dict().get('qt_socios')
        
        for i, valor in X_novo[X_novo.qt_socios.isnull()].iterrows():
            X_novo.at[i, 'qt_socios'] = qt_socios_agrupado.get(valor.de_natureza_juridica)
        return X_novo
    
    def preencher_censo_renda(self, X):
        """
        Função responsável por preencher a variável `empsetorcensitariofaixarendapopulacao`
        com a média por micro região
        Parâmetros:
        ----------
            X: DataFrame
        Retornos:
        --------
            X_novo: DataFrame
                O dataframe após aplicar transformação.
        """
        X_novo = X.copy()
        censo_renda_populacao = X_novo.groupby(by=['nm_micro_regiao']).agg({
            'empsetorcensitariofaixarendapopulacao': 'mean'
        }).to_dict().get('empsetorcensitariofaixarendapopulacao')
        
        # Iterando sobre as linhas do dataframe que contém nulos
        for i, valor in X_novo[X_novo.empsetorcensitariofaixarendapopulacao.isnull()].iterrows():
            X_novo.at[i, 'empsetorcensitariofaixarendapopulacao'] = censo_renda_populacao.get(valor.nm_micro_regiao)
        
        return X_novo

    def preencher_setor(self, df):
        """
        Função para preencher os dados faltantes da variável `setor` com a constante `OUTRO`
        Parâmetros:
        ----------
            df: DataFrame
        Retornos:
        --------
            df_novo: DataFrame
                O dataframe após aplicar preenchimento do setor.
        """
        df_novo = df.copy()
        df_novo.loc[df_novo.setor.isnull(), 'setor'] = 'OUTRO'
        return df_novo

    def preencher_faturamento(self, df):
        """
        Função para preencher as variáveis de faturamento, sendo elas `vl_faturamento_estimado_aux` e
        `vl_faturamento_estimado_grupo_aux`.
        Parâmetros:
        ----------
            df: DataFrame
        Retornos:
        --------
            df_novo: DataFrame
                O dataframe após aplicar preenchimento do setor.
        """

        df_novo = df.copy()
        df_novo['vl_faturamento_estimado_aux'].fillna(df_novo.vl_faturamento_estimado_aux.mean(), inplace=True)
        df_novo['vl_faturamento_estimado_grupo_aux'].fillna(df_novo.vl_faturamento_estimado_grupo_aux.mean(), inplace=True)
        return df_novo


    def executar(self):
        fonte_dados = FonteDados()
        experimentos = Experimentos()

        # Lendo os dados originais
        dados = fonte_dados.carregar_dados()

        # Deletando colunas que não serão utilizadas
        dados.drop(labels=self.variaveis_deletar, axis=1, inplace=True)
        
        # Identificação e eliminação de outliers
        outliers = experimentos.deteccao_anomalia(dados['idade_minima_socios'])
        dados.loc[outliers[2], 'idade_minima_socios'] = np.nan

        outliers = experimentos.deteccao_anomalia(dados['idade_maxima_socios'])
        dados.loc[outliers[2], 'idade_maxima_socios'] = np.nan

        outliers = experimentos.deteccao_anomalia(dados['idade_media_socios'])
        dados.loc[outliers[2], 'idade_media_socios'] = np.nan

        # Conversões dos dados
        dados = self.converter_sim_nao(dados, ['fl_rm'])
        
        variaveis_booleanas = ['fl_matriz', 'fl_me', 'fl_sa', 'fl_mei', 'fl_ltda', 'fl_st_especial', 
        'fl_email', 'fl_telefone', 'fl_rm', 'fl_spa', 'fl_antt', 'fl_optante_simples', 
        'fl_optante_simei', 'fl_simples_irregular', 'fl_passivel_iss']

        dados = self.converter_booleano(dados, variaveis_booleanas)

        # Preenchimento dos dados faltantes
        dados = self.preencher_qt_socios(dados)
        dados = self.preencher_censo_renda(dados)
        dados = self.preencher_setor(dados)
        dados = self.preencher_faturamento(dados)
        return dados

if __name__ == "__main__":
    preprocessamento = Preprocessamento()

    inicio_execucao = time.time()
    try:
        dados_processados = preprocessamento.executar()
    except KeyboardInterrupt:
        fim_execucao = time.time()
        tempo_executacao = fim_execucao - inicio_execucao
        print(f'inicio execucao: {inicio_execucao}')
        print(f'fim da execução: {fim_execucao}')
        print(f'tempo de execução: {tempo_executacao} segundos')
