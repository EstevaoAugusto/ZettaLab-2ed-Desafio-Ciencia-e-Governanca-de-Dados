import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go                               #
import streamlit.components.v1 as components
import sys
import os

# --- CONFIGURAÇÃO DE AMBIENTE E CAMINHOS ---

# Garante que o Python encontre os módulos do projeto independentemente de onde o script é executado.
# O sys.path.append adiciona a raiz do projeto ao path, permitindo importações de arquivos locais como 'config_path'.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath("__file__"))))

# Importação de constantes de diretórios definidas centralizadamente.
# Isso facilita a manutenção: se você mudar a pasta de dados, muda apenas no config_path.py.
from config_path import (
    RAW_DATA_DIRECTORY_PATH,       # Dados brutos (como baixados do IBGE)
    PROCESSED_DATA_DIRECTORY_PATH, # Dados limpos e prontos para o modelo
    REPORTS_DIRECTORY_PATH,        # Gráficos estáticos (PNG/PDF)
    METRICS_DIRECTORY_PATH,        # Arquivos de performance do modelo (JSON/CSV)
    MODELS_DIRECTORY_PATH,         # Modelos serializados (arquivos .pkl)
    FEATURES_DIRECTORY_PATH,       # Versões específicas das colunas utilizadas
    ROOT_DIR,                      # Diretório base do projeto
    INTERACTIVE_REPORTS_PATH,      # Onde os arquivos HTML interativos são salvos
    IMGS_DIRECTORY_PATH            # Logos e outras imagens da interface
)

# Configuração da página
st.set_page_config(page_title="Desafio ZettaLab - Ciência e Governança de Dados", layout="wide")

# --- FUNÇÕES DE APOIO ---
# 1. Função com Cache para performance
@st.cache_data
def gerar_mapa_interativo(df_aux):
    # Filtragem temporal dinâmica
    df_plot = df_aux[df_aux['ano'] == 2019].copy()
    
    # Localização de Lavras
    df_lavras = df_plot[df_plot['id_municipio'] == 3138203]
    
    clusters = sorted(df_plot['cluster_hierarquico'].unique())
    colors = px.colors.qualitative.Plotly
    
    fig = go.Figure()

    # 2. Adição dos Traces por Cluster
    for i, cluster in enumerate(clusters):
        df_cluster = df_plot[df_plot['cluster_hierarquico'] == cluster]
        
        fig.add_trace(
            go.Scattermap(
                lat=df_cluster["latitude"],
                lon=df_cluster["longitude"],
                mode='markers',
                marker=go.scattermap.Marker(
                    size=10, 
                    color=colors[i % len(colors)],
                    opacity=0.7
                ),
                text=df_cluster["nome_municipio"],
                name=f"Cluster {cluster}",
                hoverinfo="text"
            )
        )

    # 3. Destaque Especial: Lavras
    if not df_lavras.empty:
        fig.add_trace(
            go.Scattermap(
                lat=df_lavras["latitude"],
                lon=df_lavras["longitude"],
                mode='markers+text',
                marker=go.scattermap.Marker(
                    size=22, 
                    color='gold', 
                    symbol='star'
                ),
                text="🌟 Lavras (MG)",
                textposition="top center",
                name="Destaque: Lavras",
                hoverinfo="text"
            )
        )

    # 4. Lógica dos Botões (Mantida como você criou)
    buttons = []
    buttons.append(dict(
        method="update",
        label="Todos os Clusters",
        args=[{"visible": [True] * (len(clusters) + 1)}]
    ))

    for i in range(len(clusters)):
        visibility = [False] * (len(clusters) + 1)
        visibility[i] = True
        visibility[-1] = True 
        buttons.append(dict(
            method="update",
            label=f"Apenas Cluster {clusters[i]}",
            args=[{"visible": visibility}]
        ))

    fig.update_layout(
    # 1. Título mais elegante
    title={
        'text': f"<b>ANÁLISE DE CLUSTERS: 2019 </b>",
        'y': 0.98,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': dict(size=20, color="#FFFFFF") # Cor padrão do Streamlit
    },
    
    # 2. Estilização pesada nos botões
    updatemenus=[dict(
        buttons=buttons,
        direction="down",
        showactive=True,
        x=0.05,        # Posicionado levemente à esquerda
        y=0.95,        # No topo
        xanchor="left",
        yanchor="top",
        bgcolor="rgba(255, 255, 255, 0.9)", # Branco levemente transparente
        bordercolor="#4B4B4B",
        borderwidth=1,
        font=dict(size=14, color="#262730"), # Fonte maior e cor legível
        pad={"r": 10, "t": 10} # Espaçamento para o texto não sufocar
    )],
    
    # 3. Legenda lateral mais discreta
    legend=dict(
        orientation="v",
        yanchor="top",      # Mudei para top para alinhar melhor se o x for 0.98
        y=0.98,             # Subi um pouco para ficar no topo direito
        xanchor="right",
        x=0.98,
        bgcolor="rgb(255, 255, 255)", # Branco sólido (sem transparência)
        bordercolor="#000000",       # Borda preta fina
        borderwidth=1,
        font=dict(
            family="Arial, sans-serif",
            size=14,
            color="#000000"           # PRETO PURO para as letras
        )
    ),

    map=dict(
        style="carto-positron",
        center={"lat": -18.5, "lon": -44.5},
        zoom=5.5
    ),
        margin={"r":0,"t":50,"l":0,"b":0}, # Reduzi o topo para o título não flutuar
        height=800
    )
    
    return fig

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
    st.image(f"{IMGS_DIRECTORY_PATH}/K-Means-Clusterizacao.png", caption="K-Means-Clusterizacao", width=600)
    st.write("""
            O agrupamento K-Means é um algoritmo de aprendizado não supervisionado utilizado para agrupamento de dados, 
            que agrupa pontos de dados não rotulados em grupos ou clusters. É um dos métodos de agrupamento mais populares usados 
            em aprendizado de máquina. Diferentemente do aprendizado supervisionado, os dados de treinamento que esse 
            algoritmo utiliza não são rotulados, o que significa que os pontos de dados não têm uma estrutura de 
            classificação definida.

            O K-Means é um algoritmo de agrupamento baseado em centroides iterativo, que divide um conjunto de dados em 
            grupos semelhantes com base na distância entre seus centroides. O centroide, ou centro do cluster, é a média
            ou a mediana de todos os pontos dentro do cluster, dependendo das características dos dados.

            K-Means foi utilizado para o projeto a fim de categorizar municipio que sejam semelhantes entre si utilizando
            os seguintes fatores: impostos_liquidos, va_agropecuaria, va_industria, va_servicos, e va_adespss.
            Isso ajudaria a obter insights gerais como "municipios de Cluster X tendem a possuir mais investimento em
            area Y", além de ajudar na implementação do Random Forest.

            Para a seleção do número de clusters que serão utilizados no k-means, foi-se usado o Método do Cotovelo
            (Elbow Method) para encontrar o 'K' ideal. Tal método é baseada na análise do within-cluster sum of squares (WCSS),
            que mede a variação dentro dos clusters. A ideia é identificar o “cotovelo” no gráfico, onde a taxa de
            diminuição muda para cada k significativamente. Nesse projeto foi utilizado 5 clusters. Abaixo é o gráfico
            do Elbow Method.
             """)

    st.image(f"{REPORTS_DIRECTORY_PATH}/08_barplot_metodo_cotovelo.png", caption="Grafico Método Cotovelo", width=600)
    st.markdown("""
            Adicionalmente, quando o K-Means é executado, obtem se dados com clusters as quais não se sabem exatamente
            o que eles significam, afinal K-Means é um algoritmo de Machine Learning Não-Supervisionado, os clusters são
            apenas pontos aos quais registros vão estar mais proximos. Para resover isso, utilizou-se de Clusters
            Hierarquicos, aos quais utilizam os Clusters Originais do K-Means como base. Eles analisam a Média do PIB de
            cada Cluster Original, e de ordem crescente dão um novo valor para cada um, e com isso, cria Clusters
            Hierarquicos. Abaixo está o código para a implementação dos Clusters Hierarquicos:

            ```python
            # Executando o K-Means final com 5 clusters
            n_clusters = 5
            kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
            clusters = kmeans.fit_predict(X_scaled)

            # Adicionando o resultado de volta ao DataFrame original
            df_aux.loc[df_aux.index, 'cluster_original'] = clusters

            # Verificando o perfil médio de cada grupo. Os valores são ordenados de forma crescente
            medias_por_cluster = df_aux.groupby('cluster_original')['pib_per_capita'].mean().sort_values()

            # Como o index de medias_por_cluster são o valor númerico do cluster, utiliza-se enumarate para então
            # criar os Clusters Hierarquicos
            mapeamento_ordenado = {cluster_antigo: i for i, cluster_antigo in enumerate(medias_por_cluster.index)}
            # Exemplo de valor: mapeamento_ordenado = { '1' : 0, '4' : 1, ...}

            # Cria-se uma nova coluna apenas para os Clusters Hieraquicos
            df_aux.loc[df_aux.index,'cluster_hierarquico'] = df_aux['cluster_original'].map(mapeamento_ordenado)
            ```

            Abaixo estão os gráficos gerados com os Clusters Hierarquicos:

             """)
    
    col1, col2 = st.columns(2)

    with col1:
        st.image(f"{REPORTS_DIRECTORY_PATH}/09_distribuicao_municipios_por_nivel_de_riqueza_colorido.png", 
             caption="Distribuição Municipios Por Nivel de Riqueza Colorido", width="stretch")
    with col2:
        st.image(f"{REPORTS_DIRECTORY_PATH}/10_distribuicao_municipios_por_nivel_de_riqueza_lavras.png", 
             caption="Distribuição Municipios Por Nivel de Riqueza Lavras", width="stretch")
        
    st.markdown("""
                Dentre os **5570** múnicipios no Brasil:

                - **1990** pertencem ao Cluster 0.
                - **1532** pertencem ao Cluster 1. 
                - **1248** pertencem ao Cluster 2.
                - **734** pertencem ao Cluster 3.
                - **66** pertencem ao Cluster 4.

                Os municipios do Cluster 4, que são aqueles com maior PIB per Capita, representam quase que 1%.
                O que revela uma enorme discrepância entre quantidade de múnicipios e bens produzidos. 
                O gráfico abaixo melhor reforça essa diferença:
                """)

    st.image(f"{REPORTS_DIRECTORY_PATH}/10_distribuicao_municipios_pizza_porcentagem.png", 
             caption="Distribuição Municipios Por Nivel de Riqueza Pizza", width=600)
    
    st.markdown("""
                No geral, 63,2% dos múnicipios brasileiros ou se encontram no Cluster 1 ou Cluster 0. Mais da metade do páis. 
                Sabendo que Lavras concentra-se no cluster 0, sendo que nela reside a UFLA (Universidade Federal de Lavras), 
                umas das maiores referências no setor de Cafeicultura, Zootecnia, Veterinária e Ciências do Solo, que atua desde 
                de 1908 (originalmente fundada como Escola Agrícola de Lavras) e é considerada umas das melhores universidades 
                federais do páis. Pode-se assumir que maioria dos investimentos de Lavras vão para a área de serviços, nisso 
                inclui a UFLA, quando estudantes finalizam o curso, muitas vezes podem não continuar atuando naquele mesmo múnicipio, 
                saindo do local a fim de encontrar mais oportunidades. Em outras palavras, investe-se na Educação em Lavras, contudo 
                esse talento nem sempre permanece no múnicipio, especialmente se os futuros profissionais forem de áreas a 
                qual Lavras não oferece espaço para trabalho.
                """)

    st.image(f"{REPORTS_DIRECTORY_PATH}/09_barplot_media_pib_per_capita_por_cluster_hierarquico.png", 
             caption="Grafico de Barras da Média PIB per Capita", width=600)
    
    st.markdown("""
                Apesar de termos 66 municipios no cluster 4, quase 1% da quantidade total de municipios, a
                tendência é de possuirem um PIB per Capita extremamente maior comparado com o resto dos clusters.
                A diferença entre essa e o cluster 3, que é a segunda maior média de PIB per Capita, tendo mais de
                700 múnicipios agrupados é exorbitante, apenas reforçando a perda de talentos de múnicipios após os
                estudos na universidade. De resto, Cluster 1 e 2 são os mais estáveis, tendo menor diferença de média
                entre eles. Enquanto o Cluster 0 permanece por último.
                """)
    
    
    st.markdown("""
                #### Visualização Interativa dos Clusters no Brasil
                
                Abaixo é um mapa interativo do Brasil onde os pontos são múnicipios categorizados pelos seus clusters. 
                Nele conseguimos ver como que municipios de Cluster 0 e 1 se espalham por boa parte do pais, diferente
                dos outros.
                """)
    
    
    df = pd.read_csv(f"{FEATURES_DIRECTORY_PATH}/informacoes_municipios_com_clusters.csv")

    # Chamada da função
    fig_final = gerar_mapa_interativo(df)
    # Renderização nativa
    st.plotly_chart(fig_final, use_container_width=True)
    
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