import streamlit as st
import pandas as pd
import numpy as np
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

df = pd.read_csv(f"{FEATURES_DIRECTORY_PATH}/informacoes_municipios_com_clusters.csv")
shap_values = np.load(f"{ROOT_DIR}/shap_values_backup.npy")
X_test_amostra = pd.read_csv(f"{FEATURES_DIRECTORY_PATH}/amostras_shap_values.csv")

# Configuração da página
st.set_page_config(page_title="Desafio ZettaLab - Ciência e Governança de Dados", layout="wide")

# --- FUNÇÕES DE APOIO ---
def renderizar_mapa_valor_3d(df_aux):
    """
    Renderiza o gráfico de dispersão 3D comparando Indústria, Serviços e PIB,
    colorido por cluster, diretamente no Streamlit.
    """
    
    # 1. Criação do gráfico 3D com Plotly Express
    fig = px.scatter_3d(
        df_aux, 
        x='taxa_va_industria', 
        y='taxa_va_servicos', 
        z='pib_per_capita',
        color='cluster_hierarquico', 
        size='pib_per_capita',       # Cidades mais ricas ficam com esferas maiores
        opacity=0.8,
        hover_name='nome_municipio',
        title='Mapa de Valor: Indústria vs Serviços no PIB',
        color_discrete_sequence=px.colors.qualitative.Bold,
        template='plotly_white' # Força o tema claro do Plotly
    )

    # 2. Atualização dos títulos dos eixos
    # AJUSTE DAS LETRAS (CORES PRETAS)
    fig.update_layout(
        font=dict(color='black'), 
        paper_bgcolor='white',
        plot_bgcolor='white',
        scene=dict(
            xaxis=dict(
                backgroundcolor="white", 
                gridcolor="lightgrey", 
                showbackground=True,
                tickfont=dict(color='black'),
                title=dict(font=dict(color='black')) # A correção está aqui
            ),
            yaxis=dict(
                backgroundcolor="white", 
                gridcolor="lightgrey", 
                showbackground=True,
                tickfont=dict(color='black'),
                title=dict(font=dict(color='black')) # E aqui
            ),
            zaxis=dict(
                backgroundcolor="white", 
                gridcolor="lightgrey", 
                showbackground=True,
                tickfont=dict(color='black'),
                title=dict(font=dict(color='black')) # E aqui
            ),
            xaxis_title='V.A. Indústria',
            yaxis_title='V.A. Serviços',
            zaxis_title='PIB Per Capita'
        ),
        legend=dict(font=dict(color='black')),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    # 3. Exibição no Streamlit
    
    st.plotly_chart(fig, width='stretch')

def renderizar_hierarquia_impacto_backup(shap_values_backup, feature_names):
    """
    Renderiza a hierarquia de impacto SHAP usando valores carregados de um backup.
    
    Args:
        shap_values_backup: O objeto ou array carregado do seu repositório.
        feature_names: Lista com os nomes das colunas (ex: X.columns).
    """

    # 2. Cálculo da importância média absoluta
    # O SHAP pode vir em 2D (regressão) ou 3D (classificação). 
    # Para o seu PIB (regressão), o .mean(axis=0) resolve.
    avg_shap = np.abs(shap_values_backup).mean(axis=0)

    # 3. Criar DataFrame para o Plotly
    importance_df = pd.DataFrame({
        'Agente Socioeconômico': feature_names.columns,
        'Impacto Médio (SHAP)': avg_shap
    })

    # 4. Ordenar para o gráfico ficar em escada
    importance_df = importance_df.sort_values(by='Impacto Médio (SHAP)', ascending=True)

    # 5. Criar a figura com o tema branco e letras pretas
    fig = px.bar(
        importance_df, 
        x='Impacto Médio (SHAP)', 
        y='Agente Socioeconômico',
        orientation='h',
        color='Impacto Médio (SHAP)',
        color_continuous_scale='Sunsetdark',
        title='Hierarquia de Impacto: Quais agentes explicam o PIB?',
        template='plotly_white'
    )

    fig.update_layout(
        font=dict(color='black'),
        title_font=dict(color='black', size=18),
        paper_bgcolor='white',
        plot_bgcolor='white',
        xaxis=dict(
            tickfont=dict(color='black'),
            title_font=dict(color='black'),
            gridcolor='lightgrey'
        ),
        yaxis=dict(
            tickfont=dict(color='black'),
            title_font=dict(color='black')
        ),
        coloraxis_colorbar=dict(
            title=dict(text="Magnitude", font=dict(color='black')),
            tickfont=dict(color='black')
        ),
        margin=dict(l=20, r=20, t=60, b=20),
        height=450
    )

    # 6. Exibir no Streamlit
    st.plotly_chart(fig, width='stretch')
    
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

def exibir_dicionario_variaveis():
    # Criando o dataset da tabela
    dados_variaveis = {
        "Variável": [
            "id_uf", "id_municipio", "id_municipio_nome", "sigla_uf", "sigla_uf_nome", 
            "pib", "pib per capita", "impostos_liquidos", "va_total", "va_agropecuaria", 
            "va_industria", "va_servicos", "va_adespss", "AR_UF_2024", "AR_MUN_2024", 
            "populacao_estado", "gini_pib", "gini_va_agro", "gini_va_industria", 
            "gini_va_servicos", "gini_va_adespss", "cluster_original", "cluster_hierarquico", 
            "centroide", "latitude", "longitude", "total_alfabetizados_nao_alfabetizados", 
            "total_alfabetizados", "total_nao_alfabetizados"
        ],
        "Tipo": [
            "Int", "Int", "String", "String", "String", "Float", "Float", "Float", "Int", 
            "Int", "Float", "Float", "Float", "Float", "Float", "Float", "Float", "Float", 
            "Float", "Float", "Float", "Integer", "Integer", "Objeto", "Float", "Float", 
            "Integer", "Integer", "Integer"
        ],
        "Descrição": [
            "Código identificador único do estado (2 dígitos).", 
            "Código identificador único do múnicipio (7 dígitos).", 
            "Nome do múnicipio.", "Sigla de um estado (UF).", "Nome do estado (UF).", 
            "Valor adicionado bruto da indústria ao PIB.", 
            "Valor adicionado bruto da indústria ao PIB.", 
            "Impostos, líquidos de subsídios, sobre produtos a preços correntes.", 
            "Valor adicionado bruto a preços correntes total.", 
            "Valor adicionado bruto a preços correntes da agropecuária.", 
            "Valor adicionado bruto a preços correntes da indústria.", 
            "Valor adicionado bruto a preços correntes dos serviços (excl. adm pública).", 
            "Valor adicionado bruto da administração, defesa, educação e saúde públicas.", 
            "Tamanho da área em km² do estado (UF).", 
            "Tamanho da área em km² do município.", 
            "Valor de contribuição da feature para a predição final.", 
            "Índice de Gini da distribuição do PIB.", 
            "Índice de Gini do valor adicionado da agropecuária.", 
            "Índice de Gini do valor adicionado da indústria.", 
            "Índice de Gini do valor adicionado dos serviços.", 
            "Índice de Gini do valor adicionado da administração pública.", 
            "Cluster criado via K-Means (va_adespss, va_servicos, va_industria, va_agro, pib per capita).", 
            "Cluster organizado hierarquicamente com base no PIB per capita.", 
            "Dados de latitude e longitude.", "Coordenada Norte-Sul.", "Coordenada Leste-Oeste.", 
            "Quantidade de pessoas alfabetizadas e não alfabetizadas.", 
            "Quantidade de pessoas alfabetizadas.", "Quantidade de pessoas não alfabetizadas."
        ],
        "Fonte": [
            "IBGE", "IBGE", "IBGE", "IBGE", "IBGE", "IBGE", "IBGE", "IBGE", "IBGE", "IBGE", 
            "IBGE", "IBGE", "IBGE", "IBGE", "IBGE", "IBGE", "Basedosdados", "Basedosdados", 
            "Basedosdados", "Basedosdados", "Basedosdados", "K-Means", "Derivado", 
            "IBGE", "IBGE", "IBGE", "IBGE", "IBGE", "IBGE"
        ]
    }

    df_vars = pd.DataFrame(dados_variaveis)

    # Renderização no Streamlit
    with st.expander("🌐 Clique para expandir o Dicionário de Variáveis"):
        st.markdown("""
        Esta tabela detalha cada feature utilizada no modelo de clustering e na análise espacial. 
        Você pode ordenar as colunas clicando no cabeçalho.
        """)
        st.dataframe(
            df_vars, 
            width='stretch', 
            hide_index=True
        )

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
    
    st.markdown("""
                ## Instalação

                O projeto usou o Jupyter Notebook, e por padrão, os dados csv necessários já estão salvos numa pasta dedicada a eles. 
                Não é preciso executar o projeto pois as saídas originais já estão preservadas. Logo, certas partes do 
                tutorial de como acessar podem ser puladas (esses que estarão explicitos ao decorrer do README.md). 
                Contudo, caso queira realizar o build completo do zero e executar tudo no Notebook, siga todo que 
                esteja escrito.
                """)
    
    st.markdown("""
                ### Pré-requisitos

                - [Python 3.13.1](https://www.python.org/)
                - pip (Gerenciador de Pacotes do Python)

                ### Passo-a-Passo

                No terminal Git Bash, clone o repositório:
                ```bash
                git clone https://github.com/EstevaoAugusto/ZettaLab-2ed-Desafio-Ciencia-e-Governanca-de-Dados.git
                ```

                Entre na pasta:
                ```bash
                cd ZettaLab-2ed-Desafio-Ciencia-e-Governanca-de-Dados
                ```

                Criar e ative o Ambiente Virtual do Python:
                ```bash
                python -m venv .venv
                source ./.venv/Scripts/activate
                ```

                Atualize o pip e instale as dependências:
                ```bash
                python -m pip install --upgrade pip
                pip install -r requirements_dev.txt
                ```

                Crie um arquivo .env
                ```bash
                touch .env
                ```

                Cadastre uma conta no Google Cloud, e crie um projeto. Após isso, coloque seu ID Project no .env
                ```bash
                echo "GOOGLE_CLOUD_ID_PROJECT='<seu-projeto-id>'" > .env
                ```

                Execute os notebooks sequencialmente na pasta "notebooks". Após isso, acesse a dashboard a partir do comando abaixo:

                ```bash
                streamlit run app.py
                ```
                """)


with tab2: # Dados
    st.markdown("""
                ## Datasets Escolhidos

                Abaixo estão os datasets que foram selecionados para o projeto, sua descrição, e uso. 
                O foco do problema é estudar os fatores socioeconomicos e multifatoriais que afetam o múnicipio 
                de Lavras, localizado em Minas Gerais. Para isso, utilizamos desde de dados gerais do Brasil todo,
                como tambem, sobre seus estados, e Lavras em si. O foco da análise foi a partir de questões mais sociais, 
                demográficas, e economicas, as quais estão apresentadas abaixo:

                ### [Base dos Dados](https://basedosdados.org/)

                #### [Indice de Gini - UF](https://basedosdados.org/dataset/fcf025ca-8b19-4131-8e2d-5ddb12492347?table=a5e13468-e1e4-4125-92e6-89d3b9c85e18)

                O índice de Gini, chamado também de coeficiente de Gini, é um indicador que mensura a distribuição 
                de renda em um território (no caso do dataset, a nível estadual). Por meio dele, é possível 
                determinar a desigualdade social e a concentração de renda em diferentes níveis territoriais, além 
                de estabelecer comparativos entre eles. Utilizei os dados que vão de 2002 a 2021.

                #### [Produto Interno Bruto (PIB) Por Municipio](https://basedosdados.org/dataset/fcf025ca-8b19-4131-8e2d-5ddb12492347?table=fbbbe77e-d234-4113-8af5-98724a956943)

                Dados do PIB por Município permitem realizar comparações entre Lavras e outros múnicipios do Brasil, 
                podendo usar seus fatores economicos para entender seu desenvolvimento e quais são mais relevantes. 
                Utilizei os dados que vão de 2016 a 2020.

                #### [Produto Interno Bruto (PIB) Por UF](https://basedosdados.org/dataset/fcf025ca-8b19-4131-8e2d-5ddb12492347?table=93007431-7ce9-42ee-8740-8c2274d345ad)

                Dados do PIB por UF permitem realizar comparações entre diferentes estados do Brasil.
                Como Lavras é um múnicipio de Minas Gerais, o intuito é entender se ele está comforme os padrões e 
                tendências do estado de Minas Gerais. Utilizei os dados que vão de 2016 a 2020.

                #### [População Brasileira](https://basedosdados.org/dataset/1e2b9a88-9dc7-4f0e-a3a5-e8d2a13869bf?table=1a8d9636-c11d-443b-ae83-1b00576f0b70)

                Dados da População Brasileira inteira permitem realizar diversas operações em todo o brasil, 
                desde de pequenas como em múnicipios e para todo o Brasil. A partir dele, são realizadas 
                estimativas da quantidade populacional dos UFs a partir das projeções do número de habitantes 
                em cada Múnicipio. Utilizei os dados que vão de 2016 a 2020, esses a quais foram integrados nos 
                datasets de PIB por UF e PIB por Munícipio.

                #### [Tabela de Tradução de Múnicipios](https://basedosdados.org/api/tables/downloadTable?p=YnJfYmRfZGlyZXRvcmlvc19icmFzaWw=&q=bXVuaWNpcGlv&d=dHJ1ZQ==&s=ZnJlZQ==)

                Esta tabela tem como propósito auxiliar a Ciência de Dados ao traduzir identificadores de UFs e Múnicipios. 
                Ele serve como forma de integrar diversas informações diferentes atráves de chaves universais e padronizadas 
                do governo.

                #### [Censo 2022 - Alfabetização por Sexo, Raça e Grupo de Idade](https://basedosdados.org/dataset/08a1546e-251f-4546-9fe0-b1e6ab2b203d?table=cf9537b5-6198-455f-a8b0-7c762e94d79c)

                Tabela contem dados de pessoas de 15 anos ou mais de idade, total e as alfabetizadas, por sexo, cor ou 
                raça e grupos de idade. Alfabetização é diferente de Educação Básica, já que se refere a capacidade 
                de ler e escrever, enquanto a outra trata de aspectos formal, por exemplo: Ensino Médio Incompleto, 
                Mestrado Completo, entre outros. Os dados utilizados vêm de 2022, pois é o único ano disponível.

                #### [Educação Básica - Sexo Raça Cor](https://basedosdados.org/dataset/386927a4-4ee8-4975-8ff3-beece3474942?table=2eaf0bb5-8d7b-4d54-ae1a-85edf58c6978)

                A base conta com o total de matrículas por município para todas as etapas de ensino, sexo e raça/cor. 
                Utilizei os dados que vão de 2016 a 2020, esses a quais foram integrados nos datasets de PIB por UF 
                e PIB por Munícipio.

                ### [Area IBGE](https://www.ibge.gov.br/geociencias/organizacao-do-territorio/estrutura-territorial/15761-areas-dos-municipios.html?t=acesso-ao-produto&c=1)

                Essa tabela oferecem informações relacionadas ao tamanho territorial do brasil ao decorrer dos anos. 
                Foi a partir dele que cálculos como densidade populacional foram realizados. Os dados são do ano de 2024.

                ## Dicionário de Dados

                O projeto faz o uso de uma vasta quantidade de dados. Sem o contexto do que eles significam, eles são reduzidos à 
                numeros numa tela, impossibilitando a obtenção de insights significativos para a tomada de decisões. 
                Alguns dos dados utilizados significam a mesma coisa, contudo sua redundância facilitou no 
                tratamento de Integração de Dados e Análise de Dados Exploratória. Abaixo está uma descrição da maioria dos 
                dados utilizados:
                """)
    
    exibir_dicionario_variaveis()
        
with tab3: # EDA
    st.title("🔍 Exploration Data Analytics (EDA)")
    st.markdown("""
    Abaixo estão os insights obtidos após a execução de todos os notebooks. 
    Veremos como dados multifatoriais e socioeconômicos podem afetar o município de **Lavras**.
    """)

    # --- 1. ALFABETIZAÇÃO ---
    st.header("Distribuição de Taxa de Alfabetização em Minas Gerais")
    
    col1, col2, col3 = st.columns([1, 4, 1])
    with col2:
        st.image(f"{REPORTS_DIRECTORY_PATH}/01_taxa_alfabetizacao_lavras_vs_mg.png", 
                 caption="Comparação de Alfabetização: Lavras vs MG")
    
    st.write("""
            Nessa tabela, vemos a distribuição de taxa de alfabetização em MG, onde comparamos Lavras 
            com outros múnicipios de Minas Gerais. Os dados da alfabetização são de 2022, 
            devido à dificuldade de encontrar datasets que pudessem se encaixar nos anos de 2016 a 2020, 
            o foco do projeto, usa-se a intuinção de que o número de alfabetizados e não-alfabetizados 
            não tenha mudado muito desde então.

            Na estátistica acima, vemos que Lavras possui uma taxa de alfabetização maior que mais da metade dos 
            múnicipios, ao ter um valor aproximado de 80%. 
            Isso indica que boa parte da população sabe ler e escrever, fator esse que afeta suas 
            oportunidades para conseguirem melhores empregos, continuar os estudos, e possibilidade de participar 
            numa democracia.
    """)

    st.divider()

    # --- 2. POPULAÇÃO TOP 10 ---
    st.header("Top 10 municípios de MG por população (2020)")
    
    col1, col2, col3 = st.columns([1, 4, 1])
    with col2:
        st.image(f"{REPORTS_DIRECTORY_PATH}/02_pop_lavras_vs_top10_mun_mg.png", 
                 caption="Lavras em relação aos gigantes de MG")
    
    st.write("""
            Comparando o número total populacional de Lavras com os 10 municípios mais populosos de Minas Gerais, 
            notamos que seu valor se encontra muito abaixo deles. 
            Como entendemos que Lavras possuí uma Taxa de Alfabetização maior que a média, 
            pode-se dizer que seu número de habitantes mais baixo afetam na quantidade de pessoas que conseguem ler 
            e escrever no estado de Minas Gerais.    
            """)

    st.divider()

    # --- 3. PIB E VALOR ADICIONADO ---
    st.header("PIB e Valor Adicionado em Lavras (milhões)")
    
    col1, col2, col3 = st.columns([1, 4, 1])
    with col2:
        st.image(f"{REPORTS_DIRECTORY_PATH}/03_evolucao_pib_vs_va_lavras.png", 
                 caption="Evolução Temporal do PIB e VA")
    
    st.info("""
            A comparação entre o PIB e o Valor Adicionado (VA) Total de Lavras revela 
            uma trajetória de crescimento sólido e paralelo entre 2016 e 2019. O gap 
            constante entre as métricas evidencia a estabilidade na geração de impostos líquidos sobre a
            produção local. Em 2020, observa-se uma leve redução, reflexo direto dos impactos econômicos da 
            pandemia de COVID-19, embora a magnitude da queda indique uma resiliência estrutural superior à de
            municípios vizinhos menos diversificados.
            """)

    st.divider()

    # --- 4. ÍNDICE DE GINI ---
    st.header("Evolução do Índice de Gini Médio")
    
    col1, col2, col3 = st.columns([1, 4, 1])
    with col2:
        st.image(f"{REPORTS_DIRECTORY_PATH}/04_grafico_evo_indice_gini_medio_por_setor.png", 
                 caption="Desigualdade por Setor (2001-2021)")
    
    st.write("""
                O gráfico acima mostra o Indice de Gini Médio dos UFs por Setor entre os anos de 2001 a 2021. 
                O Indice de Gini é um indicador que mensura a distribuição de renda em um território. Seu valor
                varia entre 0 e 1: quando mais próximo de 1, mais desigual é a distribuição de renda em um país; 
                quanto mais próximo de 0, menor é essa desigualdade. 
                O gráfico apresenta a média do indicador ao decorrer dos anos, nota-se que a maioria dos valores
                permaneceu estável ao decorrer dos anos, com exceção da Agropecuária a qual teve um leve aumento a
                partir de 2017.
            """)

    st.divider()

    # --- 5. MATRIZES DE CORRELAÇÃO ---
    st.header("Matrizes de Correlação (2016-2020)")
    
    with st.expander("📝 Sobre o Coeficiente de Pearson"):
        st.write("""
                Ambas as matrizes de correlação apresentadas utilizam o Coeficiente de Correlação de Pearson. Este cálculo estatístico é utilizado para medir o grau de relação linear entre duas variáveis quantitativas.

                O coeficiente varia em um intervalo de -1 a 1:

                - 1 (Correlação Positiva Perfeita): Quando uma variável aumenta, a outra aumenta na mesma proporção.
                - 0 (Ausência de Correlação): Não existe uma relação linear aparente entre as variáveis.
                - -1 (Correlação Negativa Perfeita): Quando uma variável aumenta, a outra diminui proporcionalmente.
                """)

    st.subheader("Contexto: Brasil")
    st.image(f"{REPORTS_DIRECTORY_PATH}/05_matriz_correlacao_mun_brasil_2016_a_2020.png")
    
    st.markdown("""
                Na matriz de correlação, observamos como alguns indicadores de Lavras se relacionam entre si:

                - O PIB per capita e o Valor Bruto Total a Preços Correntes per capita apresentam forte correlação, indicando que municípios com maior PIB per capita tendem a ter também maior Valor Bruto Total per capita.
                - A correlação entre a Taxa de Alfabetização e o PIB per capita, assim como entre a Taxa de Alfabetização e o Valor Bruto Total per capita, foi moderada, sugerindo que municípios com maior renda tendem a ter maior alfabetização, embora essa relação não seja sempre consistente.
                - A correlação entre a Taxa de Alfabetização e a Taxa de Matrícula foi a menor positiva, mostrando que a matrícula escolar nem sempre se traduz em alfabetização efetiva.
                - Os demais indicadores apresentaram correlação negativa, indicando que aumentos em certos indicadores estão associados a reduções em outros.
    """)

    st.subheader("Contexto: Minas Gerais")
    st.image(f"{REPORTS_DIRECTORY_PATH}/06_matriz_correlacao_todos_os_municipios_mg_2016_a_2020.png")
    st.write("Analisando essa matriz utilizando apenas informações dos municipios de Minas Gerais, vemos que os resultados são semelhantes do gráfico anterior, com pequenas difereças de valores.")

    st.divider()

    # --- 6. POPULAÇÃO VS PIB ---
    st.header("Relação entre População e PIB")
    
    col1, col2, col3 = st.columns([1, 4, 1])
    with col2:
        st.image(f"{REPORTS_DIRECTORY_PATH}/07_scatter_plot_pib_pop_brasil_2016_2020.png")
    
    st.success("""
    O gráfico acima demonstra uma relação linear entre as variáveis População e PIB (Produto Interno Bruto) no Brasil durante os anos de 2016 a 2020. Isso mostra que quanto maior a quantidade de habitantes em um múnicipio, a tendência é que o PIB (Produto Interno Bruto) aumente.
    """)


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
    
    # Chamada da função
    fig_final = gerar_mapa_interativo(df)
    # Renderização nativa
    st.plotly_chart(fig_final, width='stretch')
    
    st.markdown("#### Random Forest Regressor")
    st.image(f"{IMGS_DIRECTORY_PATH}/Random-Forest-Regressao.png", caption="Random Forest Regressao")
    st.write("""
                #### O que é?
                Random forest é um algoritmo de aprendizado de máquina amplamente utilizado, que combina a 
                saída de múltiplas decision trees (Arvores de Decisão) para alcançar um único resultado. 
                Sua facilidade de uso e flexibilidade impulsionaram sua adoção, pois lida com problemas de classificação 
                e regressão. Arvores de Decisão são únidades basicas simulam o processo humano de tomada de 
                decisão. Ela utiliza uma estrutura de fluxograma para dividir os dados em grupos cada vez menores e 
                mais específicos até chegar a uma conclusão. Tais unidades sózinhas são altamente sensíveis a variações no grupo 
                de dados aos quais estão sendo utilizados, a fim de minimizar esse problema, o Random Forest é um 
                algoritmo que gera diversas Arvores de Decisão, a quais para problemas de classificação, são determinados pelo 
                voto da maioria, e para casos de regressão, são determinados a partir da média dos resultados.
             """)
    
    st.image(f"{REPORTS_DIRECTORY_PATH}/13_barplot_importancia_das_variaveis_predicao_do_pib_per_capita.png", caption="Importancia das Variaveis")
    
    st.markdown("""
                Este gráfico destaca quais variáveis são mais eficientes para organizar e separar os dados em árvores de decisão.

                - Predomínio Absoluto (0.548): A taxa_va_addess detém mais de 50% da importância do modelo, reafirmando-se como o melhor preditor de riqueza.
                - Protagonismo do Cluster (0.354): Diferente da regressão, aqui o cluster_hierarquico é a segunda variável mais importante. Isso indica que o agrupamento capturou nuances não lineares essenciais para o modelo.
                - Diluição dos Outros Agentes: Variáveis como participacao_impostos (0.029), taxa_va_servicos (0.025), taxa_va_agropecuaria (0.024) e taxa_va_industria (0.021) possuem pesos residuais.
                """)
    
    col3, col4 = st.columns(2)

    with col3:
        st.image(f"{REPORTS_DIRECTORY_PATH}/14_scatterplot_comparacao_predito_real_random_forest_pib_per_capita.png", caption="Importancia das Variaveis")
    with col4:
        st.image(f"{REPORTS_DIRECTORY_PATH}/15_histplot_distribuicao_dos_erros_random_forest_pib_per_capita.png", 
             caption="Distribuição Municipios Por Nivel de Riqueza Lavras", width="stretch")

    st.markdown("""
                Análisando a performance do modelo, vemos que ele obteu um R2 (Coeficiente de Determinação) de 0,8991, indicando que o modelo explica quase 90% da variação do PIB per capita. A maioria dos pontos (municípios) está sobre ou muito próxima à linha tracejada vermelha, o que demonstra uma alta taxa de acerto do algoritmo. Mesmo em valores de PIB muito elevados (acima de 200.000), o modelo mantém uma consistência muito superior à da regressão simples, embora a dispersão aumente levemente nos casos extremos.

                Sobre a distribuição de resíduos, a grande massa de dados está concentrada na linha vermelha vertical (erro zero), indicando que o modelo não possui um viés sistemático (não está "viciado" em chutar sempre para cima ou para baixo). A "montanha" roxa é muito alta e estreita, o que significa que, para a vasta maioria dos municípios brasileiros, o erro de predição é extremamente pequeno. Existem alguns erros maiores (caudas que se estendem para a direita e esquerda), representando cidades com comportamentos econômicos únicos que fogem à regra geral da "floresta"..

                Ademais, utilizando o MEA (Média de Erros Absoluta), obtemos um valor de 4383.76, o modelo estima o PIB per capita com um desvio médio de aproximadamente 4,3 mil reais, uma margem considerada baixa dada a disparidade econômica entre os municípios brasileiros.
                """)
    
    st.markdown("#### Regressão Linear Múltipla")
    st.image(f"{IMGS_DIRECTORY_PATH}/Regressao-Linear.jpg", caption="Regressao Linear")
    st.markdown("""#### O que é?""")
    
    st.markdown(fr"""
                O modelo de regressão linear é um modelo estatístico versátil para avaliar relacionamentos entre uma resposta contínua e os preditores.

                Os preditores podem ser campos contínuos, categóricos ou derivados para que relacionamentos não lineares também sejam suportados. O modelo é linear porque consiste em termos aditivos em que cada termo é um preditor que é multiplicado por um coeficiente estimado. Um termo constante (intercepto) também é geralmente incluído no modelo.

                A regressão linear é usada para gerar insights para gráficos que contêm pelo menos dois campos contínuos com um identificado como o destino e o outro como um preditor. Além disso, um preditor categórico e dois campos contínuos auxiliares podem ser especificados em um gráfico e usados para gerar um modelo de Regressão apropriado. 

                No nosso caso, utilizamos a Regressão Linear Múltipla, pois trabalhamos com diversos preditores simultaneamente para explicar a variação do PIB.

                Para esse projeto, foi utilizamos as features: 'taxa_va_agroupecuaria', 'taxa_va_servicos', 'taxa_va_industria', 'taxa_va_addess', 'participacao_impostos' e 'cluster_hierarquico', a fim de prever o target pib_per_capita. A razão dessa escolha se deve que em valores brutos PIB equivale a soma entre o total de valores adicionados mais os impostos liquidos de um municipio, o cluster hierarquico é um valor adicional para ver se possui alguma relação com o PIB per Capita. Todos os dados utilizados estão a nivel municipal.

                A equação geral segue o formato:

                $$
                \^text{{PIB}}_{{pc}} = \^beta_0 + \beta_1(\text{{agro}}) + \beta_2(\text{{ind}}) + \beta_3(\text{{serv}}) + \beta_4(\text{{adm}}) + \beta_5(\text{{imp}}) + \beta_6(\text{{cluster}}) + \epsilon
                $$

                PIBpc (Variável Dependente): É o nosso alvo de predição, representando o Produto Interno Bruto per capita do município.

                β0 (Intercepto): É o valor constante do PIB quando todos os preditores são iguais a zero. Ele representa o ponto de partida da nossa reta de regressão.

                β1 a β6 (Coeficientes/Pesos): Representam a força e a direção do impacto de cada feature.

                Se um β é positivo, aquela variável contribui para o aumento do PIB.

                Se for negativo, ela indica uma relação inversamente proporcional.

                Features (x): São os dados reais de Lavras (Taxas de VA por setor, impostos e o cluster), que alimentam o modelo para gerar a estimativa.

                ϵ (Erro Aleatório ou Resíduo): Representa a diferença entre o valor real observado e o valor previsto pelo modelo, capturando variações que as nossas variáveis não conseguem explicar.

                Utilizamos a regressão para entender não apenas o "valor" previsto, mas a importância relativa de cada setor. Isso nos permite afirmar, por exemplo, o quanto o aumento em 1% na participação industrial de um município impactaria o seu PIB per Capita final, mantendo as outras variáveis constantes.
                """)
    
    col5, col6 = st.columns(2)

    with col5:
        st.image(f"{REPORTS_DIRECTORY_PATH}/16_scatterplot_diferenca_entre_pib_per_capita_real_vs_predicao.png", caption="Diferença entre Real e Prediçao")
    with col6:
        st.image(f"{REPORTS_DIRECTORY_PATH}/17_scatterplot_analise_de_residuos_pib_per_capita_real_predicao_regressao_linear.png", 
             caption="Analise de Resíduos", width="stretch")
    
    st.markdown(fr"""
Para o modelo de Regressão Linear Múltipla, treinamos com dados de 2016 a 2019 a fim de prever os de 2020. Analisando o gráfico à esquerda, vemos que a "nuvem" de pontos azuis concentrada no início mostra que a grande maioria dos municípios brasileiros possui um PIB per capita relativamente baixo e estável. Para essa massa de dados, o modelo é bastante preciso. 

A linha vermelha representa o ideal, mas os pontos azuis se "achatam" ao decorrer do gráfico. Isso indica que variáveis lineares baseadas apenas em taxas setoriais (Agro, Indústria, etc.) não conseguem captar totalmente o que diferencia uma cidade comum de um "ponto fora da curva" nacional, como capitais financeiras ou polos exportadores.

#### Análise de Resíduos
Os erros assumem uma forma de **"leque" ou "funil"**, o que é um reflexo direto da desigualdade econômica brasileira. O modelo erra pouco em cidades pequenas e médias, mas o erro expande em cidades ricas. Como quase todos os resíduos altos estão acima da linha zero, o modelo é **conservador**: ele prevê uma riqueza "média" baseada nos setores, mas os municípios no topo da pirâmide possuem fatores de riqueza (como valor agregado tecnológico ou financeiro) que a regressão linear simples não enxerga.

Como **Lavras** se encontra no Cluster 0 (que possui menor PIB per capita médio), o modelo consegue predizer seu valor com maior confiança, uma vez que ele tende a performar melhor em dados com menor dispersão.

---

### 📏 Métricas de Desempenho

#### 1. Coeficiente de Determinação ($R^2$)
O $R^2$ compara o erro do modelo (distância entre pontos reais e predição) com o erro da "média". 

$$R^2 = 1 - \frac{{\sum_{{i}} (y_i - \hat{{y}}_i)^2}}{{\sum_{{i}} (y_i - \bar{{y}})^2}}$$

Onde:
- $y_i$: Valor real do PIB per capita.
- $$\hat{{y}}_i$$: Valor previsto pelo modelo.
- $$\bar{{y}}$$: Média de todos os valores reais.

O modelo obteve um **$R^2$ de 0,43**, significando que explica 43% da variabilidade econômica. Os 57% restantes derivam de fatores não capturados (política local, infraestrutura, logística) ou complexidades não lineares.

#### 2. Erro Médio Absoluto (MAE)
O MAE representa a média das magnitudes dos erros. Ele nos dá a distância real do "palpite" em relação à realidade, em termos absolutos (Reais).

$$MAE = \frac{{1}}{{n}}\sum_{{i=1}}^{{n}}|y_i - \hat{{y}}_i|$$

O modelo obteve um **MAE de 9.095,34**. Isso indica que, em média, as previsões divergem cerca de 9 mil reais. No entanto, como visto no gráfico de resíduos, esse valor é inflado pelos *outliers* de alta renda. Para municípios com o perfil de Lavras, o erro real tende a ser significativamente menor que essa média geral.
""")
    st.image(f"{REPORTS_DIRECTORY_PATH}/18_impacto_rel_agentes_socioeconomicos_pib_per_capita.png", caption="Importância das variaveis")
    st.write("O gráfico de barras representa o impacto relativo de cada agente socioeconômico sobre a variável alvo (PIB per capita). Em uma Regressão Linear Múltipla, o coeficiente indica o quanto o PIB per capita aumenta para cada unidade de variação na variável independente, mantendo as outras constantes. A taxa de Valor Adicionado em Administração, Defesa, Educação e Saúde Pública aparece como o maior coeficiente. Isso indica que, no conjunto de dados, investimentos e gastos públicos nesses setores têm uma correlação fortíssima com o PIB per capita municipal. Logo em seguida, a taxa_va_agropecuaria e a taxa_va_industria mostram impactos elevados. Cidades com forte produção industrial ou agrícola tendem a apresentar saltos significativos na renda per capita. Embora importante, a taxa_va_servicos aparece em quarto lugar, sugerindo que, embora onipresente, ela gera um impacto proporcionalmente menor no PIB per capita do que a especialização industrial ou pública em alguns municípios. Os Impostos possui um impacto positivo, mas moderado em comparação aos setores produtivos. Por fim, o Cluster Hierárquico tem coeficiente próximo a zero para o que sugere a categoria do grupo, por si só, não é um preditor linear tão forte quanto os valores brutos de produção (VA).")

    st.markdown(f"""
                #### Visualização Interativa Mapa Valor Industria e Servicos ao PIB

                Abaixo está um link para acessar um gráfico 3D que mapei os valores de industria e serviço publico, ao PIB. 
                """)
    
    renderizar_mapa_valor_3d(df)
    
    st.markdown("""
## Importância das Variáveis via SHAP

### O que é SHAP

Os valores SHAP (SHapley Additive exPlanations) são uma forma de explicar o resultado de qualquer modelo de aprendizado de máquina. Ele usa uma abordagem teórica de jogos que mede a contribuição de cada jogador para o resultado final. No aprendizado de máquina, cada recurso recebe um valor de importância que representa sua contribuição para o resultado do modelo.

Os valores de SHAP mostram como cada recurso afeta cada previsão final, a importância de cada recurso em comparação com outros e a dependência do modelo na interação entre os recursos.

Os valores SHAP são independentes de modelo, o que significa que podem ser usados para interpretar qualquer modelo de aprendizado de máquina, inclusive:

- Regressão linear
- Árvores de decisão
- Florestas aleatórias
- Modelos de aumento de gradiente
- Redes neurais

Para os resultados obtidos, utilizamos o modelo Random Forest.

### Valores SHAP obtidos
                """)
    st.image(f"{REPORTS_DIRECTORY_PATH}/19_SHAP_impacto_agentes_socioeconomicos_pib_per_capita.png", caption="Importância das variaveis")
    
    st.markdown("""
            O gráfico SHAP oferece uma visão detalhada de como cada agente socioeconômico influencia a predição do PIB per capita de forma individualizada para os municípios. Diferente da importância global, aqui vemos a direção e a intensidade do impacto.

            **Como ler o gráfico:**

            **Eixo X (SHAP value):** Pontos à direita do zero indicam que a variável aumentou a previsão do PIB; pontos à esquerda indicam que ela diminuiu.

            **Cores (Feature value):** O rosa representa valores altos da variável, enquanto o azul representa valores baixos.

            **Destaques da Análise:**

            **Taxa de Administração Pública (taxa_va_addess):** É o fator com maior dispersão. Valores baixos (azul) têm um forte impacto negativo, "puxando" o PIB para baixo, enquanto valores altos (rosa) estão distribuídos, mostrando que o investimento público é uma base necessária, mas não a única garantia de PIB altíssimo.

            **Cluster Hierárquico:** Observa-se um comportamento binário interessante. Pontos rosa (clusters específicos) geram impactos positivos extremos (acima de 80.000 no SHAP value), validando que a classificação por grupos captura municípios com dinâmicas de riqueza fora da curva.

            **Setores Produtivos (Serviços, Agro e Indústria):** Apresentam impactos mais concentrados em torno de zero, com pequenos grupos (pontos rosa) conseguindo empurrar o PIB para valores positivos. Isso sugere que a especialização extrema nesses setores beneficia apenas uma elite de municípios.

            **Participação de Impostos:** Mostra um impacto neutro a levemente negativo quando os valores são baixos, reforçando que a arrecadação caminha junto com a geração de valor local.
                """)
    
    
    
    st.markdown("""
                Este último gráfico apresenta a Hierarquia de Impacto Médio (SHAP), que quantifica a contribuição média de cada variável para a formação do preço final do PIB per capita previsto pelo modelo. O gráfico de impacto médio SHAP consolida a importância de cada agente econômico na decisão final do modelo Random Forest. Ele revela o "peso" médio que cada variável tem ao deslocar a previsão do PIB para longe da média global.

                **Destaques da Hierarquia:**

                **A Supremacia do Setor Público (taxa_va_addess):** Com um impacto médio superior a 12k, esta variável é o pilar central da previsão. Isso confirma que a estrutura de administração, educação e saúde pública é o fator que mais altera (positiva ou negativamente) a estimativa de riqueza de um município.

                **O Valor Estratégico do Grupo (cluster_hierarquico):** O cluster aparece como o segundo fator mais influente, com um impacto médio próximo a 3k. Isso valida a metodologia de agrupamento: pertencer a um determinado perfil socioeconômico é um preditor de riqueza mais forte do que o desempenho isolado de setores como agro ou indústria.

                **Setores de Base (Serviços, Agro e Indústria):** Estes agentes apresentam impactos médios menores (entre 1k e 2k). Isso indica que, embora importantes, sua contribuição para o PIB per capita é mais homogênea entre os municípios, não causando desvios tão drásticos quanto a gestão pública ou a classificação do cluster.
                
                ### Visualização interativa Hierarquia
                """)
    
    renderizar_hierarquia_impacto_backup(shap_values, X_test_amostra)
    
    st.write("Este último gráfico apresenta a Hierarquia de Impacto Médio (SHAP), que quantifica a contribuição média de cada variável para a formação do preço final do PIB per capita previsto pelo modelo. O gráfico de impacto médio SHAP consolida a importância de cada agente econômico na decisão final do modelo Random Forest. Ele revela o 'peso' médio que cada variável tem ao deslocar a previsão do PIB para longe da média global.")


with tab5:
    st.markdown("""
                ### Insights
                
                Este projeto seguiu um pipeline completo de Ciência de Dados para entender a dinâmica do PIB per capita dos municípios brasileiros em 2020. Abaixo, destacamos os principais insights de cada etapa:

                **1. Análise Exploratória (EDA) e Clusterização**

                A exploração inicial revelou uma economia heterogênea, com grandes disparidades regionais. A aplicação da Clusterização Hierárquica foi fundamental para organizar essa complexidade:

                - Segmentação Estratégica: Em vez de tratar todos os municípios como iguais, a clusterização criou grupos com perfis socioeconômicos similares.
                - Poder Preditivo: O cluster tornou-se uma das variáveis mais importantes nos modelos de Machine Learning, provando que a "vizinhança econômica" (perfil do grupo) dita o ritmo da riqueza local.

                **2. Regressão Linear Múltipla: A Base Estrutural**

                A Regressão Linear serviu para identificar a tendência média e a elasticidade dos setores:

                - Interpretabilidade: Mostrou que, em uma visão simplificada, os setores de Administração Pública, Agropecuária e Indústria possuem os maiores pesos diretos na formação do PIB.
                - Limitação: Com um R2 de 0,43, o modelo linear deixou claro que a economia não cresce em linha reta e que existem fatores complexos que exigem modelos mais robustos.

                **3. Random Forest: O Salto de Precisão**

                Ao mudarmos para o Random Forest, o desempenho saltou para um R2 de 0,899, reduzindo o erro médio (MAE) pela metade (R$ 4.383,76).

                - Não-Linearidade: O modelo de floresta entendeu que o impacto de uma indústria em Lavras é diferente do impacto de uma indústria em um polo minerador, ajustando a previsão de acordo com o contexto do cluster.
                - Consistência: A distribuição de resíduos concentrada no zero confirmou a alta confiabilidade do algoritmo para a vasta maioria dos municípios brasileiros.

                **4. Inteligência Explicável (SHAP)**

                O uso do SHAP permitiu "abrir a caixa-preta" do Random Forest:

                - Hierarquia de Impacto: Confirmou que a Administração Pública (taxa_va_addess) é o fator que mais move o ponteiro da economia, seguida de perto pelo enquadramento no Cluster.
                - Direcionamento: O modelo provou que baixos investimentos em serviços públicos são o principal "freio" para o PIB per capita municipal.

                O projeto demonstra que a riqueza municipal brasileira é um fenômeno multidimensional. Enquanto a Regressão Linear nos dá a direção dos setores produtivos, o Random Forest mapeia a realidade com precisão ao considerar as interdependências entre gestão pública e o perfil regional (cluster). Para cidades como Lavras, o sucesso econômico está intrinsecamente ligado à manutenção da sua qualidade administrativa e à sua força como polo regional.
                
                ## Limitações e Futuras Implementações

                O trabalho realizado, apesar de sua complexidade, possui diversas limitações. 
                Primeiramente, maioria dos dados utilizados trata de questões de investimento e produção de bens, 
                não levando em consideração a composição salarial dos trabalhadores, disponibilidade à saûde, 
                custos de vida, gênero, cor de pele, etnia, entre outros. 
                
                Além disso, alguns dos dados usados são de tempos diferentes, por exemplo, usou-se a quantidade 
                de pessoas alfabetizadas no ano de 2022, sendo que os dados utilizados nas análises de EDA e IA, 
                são de 2016 a 2020, tais dados de alfabetização foram simplificados, já que os originais consideravam 
                coisas como tipo cor de pele e gênero, a simplificação das informações foi feita para tornar o 
                projeto mais viável, e focar em quesitos economicos e administrativos, buscando gerar comparações entre o
                municipio de Lavras com os padrões de Minas Gerais e Brasil.  

                Futuramente, espera-se adicionar mais dados para preencher os gaps e que estejam conforme a 
                temporalidade dos registros, sem ter que usar suposições de que o numero de alfabetizados de 2022, 
                não seja muito diferente comparado os de 2016 a 2020. Como tambem tratar da reutilização de código 
                encontrada frequentemente no projeto, optando por criar scripts python separados que tratam de 
                questões de mapas e gráficos, pré-processamento, e obtenção de informações, assim melhorando a 
                manuntenção, coesão e desenvolvimento de código.

                ## Responsável

                - Estevão Augusto da Fonseca Santos, Graduando em Ciência de Computação (6° Período)
                """)


# Rodapé
st.divider()
st.caption("Trilha Ciência e Governança de Dados - UFLA - ZettaLab")