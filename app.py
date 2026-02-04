import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import sys                              # 
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath("__file__"))))  # Adiciona raiz

from config_path import RAW_DATA_DIRECTORY_PATH, PROCESSED_DATA_DIRECTORY_PATH, REPORTS_DIRECTORY_PATH
from config_path import METRICS_DIRECTORY_PATH, MODELS_DIRECTORY_PATH, FEATURES_DIRECTORY_PATH
from config_path import ROOT_DIR, INTERACTIVE_REPORTS_PATH, IMGS_DIRECTORY_PATH

# Configuração da página
st.set_page_config(page_title="Desafio ZettaLab - Ciência e Governança de Dados", layout="wide")

# --- FUNÇÕES DE APOIO ---
@st.cache_data
def carregar_e_processar_dados():
    # Substitua pelo caminho do seu arquivo ou carregue o DataFrame atual
    # df = pd.read_csv('seu_arquivo.csv')
    # Exemplo hipotético:
    return pd.DataFrame() 

def executar_kmeans(df, n_clusters):
    # Seleção de colunas numéricas para o modelo
    colunas_cluster = ['taxa_va_agropecuaria', 'taxa_va_servicos', 'taxa_va_industria', 
                       'taxa_va_address', 'participacao_impostos', 'pib_per_capita']
    
    X = df[colunas_cluster]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
    df['cluster_original'] = kmeans.fit_predict(X_scaled)
    
    # Ordenação Hierárquica pelo PIB (conforme seu código)
    medias_pib = df.groupby('cluster_original')['pib_per_capita'].mean().sort_values()
    mapeamento = {antigo: novo for novo, antigo in enumerate(medias_pib.index)}
    df['cluster_hierarquico'] = df['cluster_original'].map(mapeamento)
    
    return df

# --- INTERFACE STREAMLIT ---

st.title("📊 Relatório dos Insights Obtidos")
st.markdown("""
Este website relata de forma interativa, os insights obtidos do Desafio Data Science do ZettaLab utilizando a metodologia CRISP-DM.
A qual incluirá documentação das fontes utilizadas, representação dos dados, modelos de Machine Learning usados, e visualizações estáticas e interativas.
""")

# --- LAYOUT PRINCIPAL ---
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Inicio", "Dados", "EDA", "IA", "Conclusões"])
 
with tab1: # Inicio
    st.subheader("Descrição do Projeto")
    st.write("""O projeto é um trabalho realizado individualmente na trilha 'Ciência e Governança de Dados' do projeto ZettaLab da UFLA 
             (Universidade Federal de Lavras). O desafio proposto foi de acessar e reunir diversas bases de dados brasileiras relevantes 
             que incluem informações capazes de avaliar as condições socioeconômicas e seus determinantes multifatoriais, a qual seja feito 
             limpeza e tratamento dos dados, análise estatística inicial e identificação de padrões, correlações e tendências relevantes, 
             e realizar análise exploratória cujo proposito é gerar insights sobre eles. Importante ressaltar que devido ao tamanho dos 
             datasets escolhidos, nosso foco será no múnicipio de Lavras, localizado em Minas Gerais, a fim de facilitar a analise e processamento.""")
    st.subheader("Metodologia CRISP-DM")
    st.write("""Este projeto segue a metodologia CRISP-DM (Cross-Industry Standard Process for Data Mining), 
             padrão mundial para projetos de Ciência de Dados.""")
    st.image(f"{IMGS_DIRECTORY_PATH}/Diagrama-de-funcionamento-do-modelo-CRISP-DM.png", caption="Diagrama da Metodologia CRISP-DM")
    st.write("""
            - **Entendimento do negócio**: A primeira etapa é, possivelmente, a mais importante de todo o processo. 
            Caso ela não seja feita da maneira correta, todo o resto do projeto pode ser invalidado futuramente. 
            Nesta etapa, é definido o objetivo do projeto e as necessidades da empresa ou projeto em análise. Por isso, 
            é necessário que todos estejam bem-informados e completamente alinhados.
            
            - **Compreensão dos dados**: Depois da primeira etapa, podemos começar a pensar nos dados que serão utilizados 
            no processo. Para isso podemos fazer várias perguntas, como: “A empresa tem banco de dados? Os dados serão 
            acessados de que forma? Quantas fontes de dados serão utilizadas? Quais serão os formatos dos dados? 
            Os dados estão estruturados?”. A partir delas, é feita a coleta dos dados, tomando cuidado para que nenhuma 
            informação importante fique de fora. 
            
            - **Preparação dos dados**: Com os dados já coletados, é preciso organizá-los de modo a conseguirmos 
            enxergar o que eles contam. Esta etapa também pode ser guiada por algumas perguntas: “Como os valores nulos 
            devem ser tratados? Os atributos estão nos formatos corretos? Será necessário fazer alguma fusão com outros 
            dados? Quais variáveis serão utilizadas na modelagem?”. Esta costuma ser a parte mais demorada e trabalhosa 
            de todas, porém um bom trabalho aqui significa menos retrabalho futuro.
            
            - **Modelagem**: Nesta etapa o modelo começa a tomar forma e podemos ver os primeiros resultados.
            O tipo de modelagem a ser utilizada normalmente é definida de acordo com a necessidade do negócio e com o 
            tipo de variável a ser analisada. Com a definição de qual modelo será utilizado, devem ser definidos quais
            atributos serão variáveis na construção deste modelo. “Aqui pode ser muito útil voltar à primeira etapa para
            conferir objetivos e encontrar novas possibilidades”, aconselha Prado.
            
            - **Implementação**: Com o modelo já em mãos, podemos avaliar se o se o resultado corresponde à expectativa
            do projeto. Caso a resposta seja negativa ou a equipe considere que há espaço para melhorias, todas as forças
            devem ser direcionadas para fazer as mudanças necessárias. Estas mudanças podem ter diversas formas, como a
            retirada de atributos estatisticamente insignificantes, correção na entrada de dados, correção no tratamento
            dos atributos etc.
            
            - **Entrega (Deployment)**: Caso o processo tenha sido feito da maneira correta, esta será a última etapa.
            Aqui, o modelo deve ser colocado em produção, de modo a agregar valor para o negócio. A forma como isso é
            feito varia muito, dependendo do tipo de modelo e projeto. Esse modelo deve ficar exposto para acesso,
            normalmente armazenado na nuvem ou em servidores locais da própria empresa.
             """)


with tab2: # Dados
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Dispersão: PIB vs Setores")
        # Exemplo de gráfico interativo
        # fig = px.scatter(df_final, x='taxa_va_servicos', y='pib_per_capita', color='cluster_hierarquico')
        # st.plotly_chart(fig, use_container_width=True)
        st.info("Espaço para gráfico de dispersão interativo.")

    with col2:
        st.subheader("Mapa de Calor (Intensidade)")
        st.info("Espaço para o Heatmap de médias normalizadas.")
        
with tab3: # EDA
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Dispersão: PIB vs Setores")
        # Exemplo de gráfico interativo
        # fig = px.scatter(df_final, x='taxa_va_servicos', y='pib_per_capita', color='cluster_hierarquico')
        # st.plotly_chart(fig, use_container_width=True)
        st.info("Espaço para gráfico de dispersão interativo.")

    with col2:
        st.subheader("Mapa de Calor (Intensidade)")
        st.info("Espaço para o Heatmap de médias normalizadas.")

with tab4: # IA
    st.subheader("Modelos de Machine Learning")
    st.markdown("#### Clusterização por K-Means")
    st.image(f"{IMGS_DIRECTORY_PATH}/K-Means-Clusterizacao.png", caption="K-Means-Clusterizacao")
    st.write("""
             
             """)
    
    st.markdown("#### Random Forest Regressor")
    st.image(f"{IMGS_DIRECTORY_PATH}/Random-Forest-Regressao.png", caption="Random Forest Regressao")
    st.write("""
             
             """)
    
    st.markdown("#### Regressão Linear")
    st.image(f"{IMGS_DIRECTORY_PATH}/Regressao-Linear.jpg", caption="Regressao Linear")
    st.write("""
             
             """)

# Rodapé
st.divider()
st.caption("Trilha Ciência e Governança de Dados - UFLA - ZettaLab")