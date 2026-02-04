# Desafio ZettaLab - Ciência e Governança de Dados

O projeto é um trabalho realizado individualmente na trilha "Ciência e Governança de Dados" do projeto ZettaLab da UFLA (Universidade Federal de Lavras). O desafio proposto foi de acessar e reunir diversas bases de dados brasileiras relevantes que incluem informações capazes de avaliar as condições socioeconômicas e seus determinantes multifatoriais, a qual seja feito limpeza e tratamento dos dados, análise estatística inicial e identificação de padrões, correlações e tendências relevantes, e realizar análise exploratória cujo proposito é gerar insights sobre eles. Importante ressaltar que devido ao tamanho dos datasets escolhidos, nosso foco será no múnicipio de Lavras, localizado em Minas Gerais, a fim de facilitar a analise e processamento.

## Índices

- [Estrutura do Projeto](#estrutura-do-projeto)
- [Metodologia CRISP-DM](#metodologia-crisp-dm)
- [Instalação](#instalação)
- [Datasets Escolhidos](#datasets-escolhidos)
- [Dicionário de Dados](#dicionário-de-dados)
- [EDA](#exploration-data-analyticseda)
- [Modelo de IA](#modelo-de-ia)
    - [K-Means](#clusterização-por-k-means)
    - [Random Forest Regressor](#random-forest-regressor)
    - [Regressão Linear](#regressão-linear-múltipla)
- [Importância das Variáveis via SHAP](#importância-das-variáveis-via-shap)
- [Responsável](#responsável)

## Estrutura do Projeto

```bash
.
├── CONTRIBUTING.md
├── Desafio 1.pdf
├── Desafio 2.pdf
├── E-book Ciência e Governança de Dados.pdf
├── README.md
├── __pycache__
│   └── config_path.cpython-313.pyc
├── anotacao.txt
├── config_path.py
├── data
│   ├── processed
│   │   ├── alfabetizacao_processada.csv
│   │   ├── brasil_info.csv
│   │   ├── indicadores_lavras_completo.csv
│   │   ├── lavras_info.csv
│   │   ├── mg_info.csv
│   │   ├── quantidade_total_matriculas_alfabetizacao.csv
│   │   ├── quantidade_total_matriculas_alfabetizacao_dividida_mun.csv
│   │   ├── tamanho_municipios_2024.csv
│   │   ├── tamanho_populacional_mun.csv
│   │   ├── tamanho_populacional_uf.csv
│   │   ├── tamanho_uf_2024.csv
│   │   └── trad_municipio_tratados.csv
│   └── raw
│       ├── Dados_Geograficos_Brasil_Inteiro.ods
│       ├── alfabetizadas_por_sexo_cor_ou_raca_e_idade_original.ods
│       ├── educacao_basica_sexo_raca_cor.csv
│       ├── indice_de_gini_uf.csv
│       ├── pip_por_municipio_original.csv
│       ├── pip_por_uf_original.csv
│       ├── populacao_brasileira.csv
│       └── traducao_municipios.csv
├── features
│   └── informacoes_cada_municipio_2019_clusterizado.csv
├── imgs
│   ├── Diagrama-de-funcionamento-do-modelo-CRISP-DM.png
│   ├── K-Means-Clusterizacao.png
│   ├── Random-Forest-Regressao.png
│   └── Regressao-Linear.jpg
├── interactive_reports
│   ├── 01_industria_vs_servicos_no_pib.html
│   ├── 02_hierarquia_de_impacto_agentes_explicam_PIB.html
│   └── 03_mapa_clusters_2020_destaque_lavras.html
├── metrics
├── models
│   ├── clusterizacao_k_means_municipios_2019.pkl
│   ├── ia_random_forest_regressao.joblib
│   └── regressao_linear_prever_pib_per_capita.pkl
├── notebooks
│   ├── 1_coleta_preparacao_dados.ipynb
│   ├── 2_analise_exploratoria.ipynb
│   ├── 3_aplicacao_ia.ipynb
│   └── run_notebooks.py
├── reports
│   ├── 01_taxa_alfabetizacao_lavras_vs_mg.png
│   ├── 02_pop_lavras_vs_top10_mun_mg.png
│   ├── 03_evolucao_pib_vs_va_lavras.png
│   ├── 04_grafico_evo_indice_gini_medio_por_setor.png
│   ├── 05_matriz_correlacao_mun_brasil_2016_a_2020.png
│   ├── 06_matriz_correlacao_todos_os_municipios_mg_2016_a_2020.png
│   ├── 07_scatter_plot_pib_pop_mg_2016_2020.png
│   ├── 08_barplot_metodo_cotovelo.png
│   ├── 09_barplot_media_pib_per_capita_por_cluster_hierarquico.png
│   ├── 09_distribuicao_municipios_por_nivel_de_riqueza_colorido.png
│   ├── 10_distribuicao_municipios_por_nivel_de_riqueza_lavras.png
│   ├── 11_distribuicao_clusters_minas_gerais.png
│   ├── 12_boxplot_distribuicao_pib_per_capita_por_cluster_hierarquico.png
│   ├── 13_barplot_importancia_das_variaveis_predicao_do_pib_per_capita.png
│   ├── 14_scatterplot_comparacao_predito_real_random_forest_pib_per_capita.png
│   ├── 15_histplot_distribuicao_dos_erros_random_forest_pib_per_capita.png
│   ├── 16_scatterplot_diferenca_entre_pib_per_capita_real_vs_predicao.png
│   ├── 17_scatterplot_analise_de_residuos_pib_per_capita_real_predicao_regressao_linear.png
│   ├── 18_impacto_rel_agentes_socioeconomicos_pib_per_capita.png
│   └── 19_SHAP_impacto_agentes_socioeconomicos_pib_per_capita.png
├── requirements.txt
├── shap_values_backup.npy
└── teste.txt
```

## Metodologia CRISP-DM

Este projeto segue a metodologia **CRISP-DM** (Cross-Industry Standard Process for Data Mining), padrão mundial para projetos de ciência de dados.

<img src="./imgs/Diagrama-de-funcionamento-do-modelo-CRISP-DM.png">

- **Entendimento do negócio**: A primeira etapa é, possivelmente, a mais importante de todo o processo. Caso ela não seja feita da maneira correta, todo o resto do projeto pode ser invalidado futuramente. Nesta etapa, é definido o objetivo do projeto e as necessidades da empresa ou projeto em análise. Por isso, é necessário que todos estejam bem-informados e completamente alinhados.
- **Compreensão dos dados**: Depois da primeira etapa, podemos começar a pensar nos dados que serão utilizados no processo. Para isso podemos fazer várias perguntas, como: “A empresa tem banco de dados? Os dados serão acessados de que forma? Quantas fontes de dados serão utilizadas? Quais serão os formatos dos dados? Os dados estão estruturados?”. A partir delas, é feita a coleta dos dados, tomando cuidado para que nenhuma informação importante fique de fora. 
- **Preparação dos dados**: Com os dados já coletados, é preciso organizá-los de modo a conseguirmos enxergar o que eles contam. Esta etapa também pode ser guiada por algumas perguntas: “Como os valores nulos devem ser tratados? Os atributos estão nos formatos corretos? Será necessário fazer alguma fusão com outros dados? Quais variáveis serão utilizadas na modelagem?”. Esta costuma ser a parte mais demorada e trabalhosa de todas, porém um bom trabalho aqui significa menos retrabalho futuro.
- **Modelagem**: Nesta etapa o modelo começa a tomar forma e podemos ver os primeiros resultados. O tipo de modelagem a ser utilizada normalmente é definida de acordo com a necessidade do negócio e com o tipo de variável a ser analisada. Com a definição de qual modelo será utilizado, devem ser definidos quais atributos serão variáveis na construção deste modelo. “Aqui pode ser muito útil voltar à primeira etapa para conferir objetivos e encontrar novas possibilidades”, aconselha Prado.
- **Implementação**: Com o modelo já em mãos, podemos avaliar se o se o resultado corresponde à expectativa do projeto. Caso a resposta seja negativa ou a equipe considere que há espaço para melhorias, todas as forças devem ser direcionadas para fazer as mudanças necessárias. Estas mudanças podem ter diversas formas, como a retirada de atributos estatisticamente insignificantes, correção na entrada de dados, correção no tratamento dos atributos etc.
- **Entrega (Deployment)**: Caso o processo tenha sido feito da maneira correta, esta será a última etapa. Aqui, o modelo deve ser colocado em produção, de modo a agregar valor para o negócio. A forma como isso é feito varia muito, dependendo do tipo de modelo e projeto. Esse modelo deve ficar exposto para acesso, normalmente armazenado na nuvem ou em servidores locais da própria empresa.

## Instalação

O projeto usou o Jupyter Notebook, e por padrão, os dados csv necessários já estão salvos numa pasta dedicada a eles. Não é preciso executar o projeto pois as saídas originais já estão preservadas. Logo, certas partes do tutorial de como acessar podem ser puladas (esses que estarão explicitos ao decorrer do README.md). Contudo, caso queira realizar o build completo do zero e executar tudo no Notebook, siga todo que esteja escrito.

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
pip install -r requirements.txt
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

## Datasets Escolhidos

Abaixo estão os datasets que foram selecionados para o projeto, sua descrição, e uso. O foco do problema é estudar os fatores socioeconomicos e multifatoriais que afetam o múnicipio de Lavras, localizado em Minas Gerais. Para isso, utilizamos desde de dados gerais do Brasil todo, como tambem, sobre seus estados, e Lavras em si. O foco da análise foi a partir de questões mais sociais, demográficas, e economicas, as quais estão apresentadas abaixo:

### [Base dos Dados](https://basedosdados.org/)

#### [Indice de Gini - UF](https://basedosdados.org/dataset/fcf025ca-8b19-4131-8e2d-5ddb12492347?table=a5e13468-e1e4-4125-92e6-89d3b9c85e18)

O índice de Gini, chamado também de coeficiente de Gini, é um indicador que mensura a distribuição de renda em um território (no caso do dataset, a nível estadual). Por meio dele, é possível determinar a desigualdade social e a concentração de renda em diferentes níveis territoriais, além de estabelecer comparativos entre eles. Utilizei os dados que vão de 2002 a 2021.

#### [Produto Interno Bruto (PIB) Por Municipio](https://basedosdados.org/dataset/fcf025ca-8b19-4131-8e2d-5ddb12492347?table=fbbbe77e-d234-4113-8af5-98724a956943)

Dados do PIB por Município permitem realizar comparações entre Lavras e outros múnicipios do Brasil, podendo usar seus fatores economicos para entender seu desenvolvimento e quais são mais relevantes. Utilizei os dados que vão de 2016 a 2020.

#### [Produto Interno Bruto (PIB) Por UF](https://basedosdados.org/dataset/fcf025ca-8b19-4131-8e2d-5ddb12492347?table=93007431-7ce9-42ee-8740-8c2274d345ad)

Dados do PIB por UF permitem realizar comparações entre diferentes estados do Brasil. Como Lavras é um múnicipio de Minas Gerais, o intuito é entender se ele está comforme os padrões e tendências do estado de Minas Gerais. Utilizei os dados que vão de 2016 a 2020.

#### [População Brasileira](https://basedosdados.org/dataset/1e2b9a88-9dc7-4f0e-a3a5-e8d2a13869bf?table=1a8d9636-c11d-443b-ae83-1b00576f0b70)

Dados da População Brasileira inteira permitem realizar diversas operações em todo o brasil, desde de pequenas como em múnicipios e para todo o Brasil. A partir dele, são realizadas estimativas da quantidade populacional dos UFs a partir das projeções do número de habitantes em cada Múnicipio. Utilizei os dados que vão de 2016 a 2020, esses a quais foram integrados nos datasets de PIB por UF e PIB por Munícipio.

#### [Tabela de Tradução de Múnicipios](https://basedosdados.org/api/tables/downloadTable?p=YnJfYmRfZGlyZXRvcmlvc19icmFzaWw=&q=bXVuaWNpcGlv&d=dHJ1ZQ==&s=ZnJlZQ==)

Esta tabela tem como propósito auxiliar a Ciência de Dados ao traduzir identificadores de UFs e Múnicipios. Ele serve como forma de integrar diversas informações diferentes atráves de chaves universais e padronizadas do governo.

#### [Censo 2022 - Alfabetização por Sexo, Raça e Grupo de Idade](https://basedosdados.org/dataset/08a1546e-251f-4546-9fe0-b1e6ab2b203d?table=cf9537b5-6198-455f-a8b0-7c762e94d79c)

Tabela contem dados de pessoas de 15 anos ou mais de idade, total e as alfabetizadas, por sexo, cor ou raça e grupos de idade. Alfabetização é diferente de Educação Básica, já que se refere a capacidade de ler e escrever, enquanto a outra trata de aspectos formal, por exemplo: Ensino Médio Incompleto, Mestrado Completo, entre outros. Os dados utilizados vêm de 2022, pois é o único ano disponível.

#### [Educação Básica - Sexo Raça Cor](https://basedosdados.org/dataset/386927a4-4ee8-4975-8ff3-beece3474942?table=2eaf0bb5-8d7b-4d54-ae1a-85edf58c6978)

A base conta com o total de matrículas por município para todas as etapas de ensino, sexo e raça/cor. Utilizei os dados que vão de 2016 a 2020, esses a quais foram integrados nos datasets de PIB por UF e PIB por Munícipio.

### [Area IBGE](https://www.ibge.gov.br/geociencias/organizacao-do-territorio/estrutura-territorial/15761-areas-dos-municipios.html?t=acesso-ao-produto&c=1)

Essa tabela oferecem informações relacionadas ao tamanho territorial do brasil ao decorrer dos anos. Foi a partir dele que cálculos como densidade populacional foram realizados. Os dados são do ano de 2024.

## Dicionário de Dados

O projeto faz o uso de uma vasta quantidade de dados. Sem o contexto do que eles significam, eles são reduzidos à numeros numa tela, impossibilitando a obtenção de insights significativos para a tomada de decisões. Alguns dos dados utilizados significam a mesma coisa, contudo sua redundância facilitou no tratamento de Integração de Dados e Análise de Dados Exploratória. Abaixo está uma descrição da maioria dos dados utilizados:

<details>
  <summary>🌐 Clique para expandir a Tabela de Variáveis</summary>

| Variável | Tipo | Descrição | Fonte |
| :--- | :--- | :--- | :--- |
| `id_uf` | Int | Código identificador único do estado (2 dígitos). | IBGE |
| `id_municipio` | Int | Código identificador único do múnicipio (7 dígitos). | IBGE |
| `id_municipio_nome` | String | Nome do múnicipio. | IBGE |
| `sigla_uf` | String | Sigla de um estado (UF). | IBGE |
| `sigla_uf_nome` | String | Nome do estado (UF). | IBGE |
| `pib`| Float | Valor adicionado bruto da indústria ao PIB. | IBGE |
| `pib per capita`| Float | Valor adicionado bruto da indústria ao PIB. | IBGE |
| `impostos_liquidos` | Float | Impostos, líquidos de subsídios, sobre produtos a preços correntes. | IBGE |
| `valor_adicionado_precos_correntes_total (va)` | Int | Valor adicionado bruto a preços correntes total. | IBGE |
| `valor_adicionado_correntes_agropecuaria (va_agropecuaria)` | Int | Valor adicionado bruto a preços correntes da agropecuária. | Saída do Modelo |
| `valor_adicionado_correntes_industria (va_industria)` | Float | Valor adicionado bruto a preços correntes da indústria. | Saída do SHAP |
| `valor_adicionado_correntes_servicos (va_servicos)` | Float | Valor adicionado bruto a preços correntes dos serviços, exclusive administração, defesa, educação e saúde públicas e seguridade social. | Saída do SHAP |
| `valor_adicionado_correntes_adm_defesa_edu_saude_seguranca_social (va_adespss)` | Float | Valor adicionado bruto a preços correntes da administração, defesa, educação e saúde públicas e seguridade social. | Saída do SHAP |
| `AR_UF_2024` | Float | Tamanho da area em quilômetros de cada um dos estados (UFs). | IBGE |
| `AR_MUN_2024` | Float | Tamanho da area em quilômetros de cada um dos múnicipios. | IBGE |
| `populacao_estado` | Float | Valor de contribuição da feature para a predição final. | Saída do SHAP |
| `gini_pib` | Float | Índice de Gini da distribuição do produto interno bruto a preços correntes. | Basedosdados |
| `gini_va_agro` | Float | Índice de Gini da distribuição do valor adicionado bruto a preços correntes da agropecuária. | Basedosdados |
| `gini_va_industria` | Float | Índice de Gini da distribuição do valor adicionado bruto a preços correntes da indústria. | Basedosdados |
| `gini_va_servicos` | Float | Índice de Gini da distribuição do valor adicionado bruto a preços correntes dos serviços, exclusive administração, defesa, educação e saúde públicas e seguridade social. | Basedosdados |
| `gini_va_adespss` | Float | Índice de Gini da distribuição do valor adicionado bruto a preços correntes da administração, defesa, educação e saúde públicas e seguridade social. | Basedosdados |
| `cluster_original` | Integer | Cluster criado a partir de va_adespss, va_servicos, va_industria, va_agropecuaria, e pib per capita. Cada cluster define qual o registro mais se assemelha. |  Valor derivado, criado com o K-Means |
| `cluster_hierarquico` | Integer | Cluster organizado de forma hierarquica, quanto onde o tamanho do pib per capita determina o cluster. | Cluster criado a partir do cluster original |
| `centroide` | Objeto |  |  |
| `latitude` | Float |  |  |
| `longitude` | Float |  |  |
| `total_alfabetizados_e_nao_alfabetizados` | Integer |  |  |
| `total_alfabetizados` | Integer |  |  |
| `total_nao_alfabetizados` | Integer |  |  |
| `alfabetizacao` | Integer |  |  |

</details>

## Exploration Data Analytics(EDA)

Abaixo são os insights obtidos após a execução de todos os notebooks. Veremos como dados multifatorais e socioeconomicos podem afetar o múnicipio de Lavras.

### Distribuição de Taxa de Alfabetização em Minas Gerais

<img src="./reports/01_taxa_alfabetizacao_lavras_vs_mg.png" alt="Tabela de Comparação entre Lavras e 10 mais populosos municípios de MG" width="600" >

Nessa tabela, vemos a distribuição de taxa de alfabetização em MG, onde comparamos Lavras com outros múnicipios de Minas Gerais. Os dados da alfabetização são de 2022, devido à dificuldade de encontrar datasets que pudessem se encaixar nos anos de 2016 a 2020, o foco do projeto, usa-se a intuinção de que o número de alfabetizados e não-alfabetizados não tenha mudado muito desde então.

Na estátistica acima, vemos que Lavras possui uma taxa de alfabetização maior que mais da metade dos múnicipios, ao ter um valor aproximado de 80%. Isso indica que boa parte da população sabe ler e escrever, fator esse que afeta suas oportunidades para conseguirem melhores empregos, continuar os estudos, e possibilidade de participar numa democracia.

### Top 10 municípios de MG por população em 2020

<img src="./reports/02_pop_lavras_vs_top10_mun_mg.png" alt="Matriz de Correlação de Indicadores Socioeconomicos em Lavras" width="600" >

Comparando o número total populacional de Lavras com os 10 municípios mais populosos de Minas Gerais, notamos que seu valor se encontra muito abaixo deles. Como entendemos que Lavras possuí uma Taxa de Alfabetização maior que a média, pode-se dizer que seu número de habitantes mais baixo afetam na quantidade de pessoas que conseguem ler e escrever no estado de Minas Gerais.

### Impacto relativo dos agentes socioeconômicos sobre PIB per capita

<img src="./reports/06_matriz_correlacao_todos_os_municipios_mg_2016_a_2020.png" alt="Matriz de Confusao de Dados Correlacionados em Lavras" width="600" >

Tabela acima demonstra o impacto que determinados agentes socioeconomicos possuem sobre o PIB per Capita. Pode-se obter os seguintes insights a partir dele:

- A Taxa de Alfabetização possui o maior Coeficiente de Regressão, sendo ele de valor positivo. Isso indica que o aumento desse agente é o principal motor para o crescimento do PIB per capita.

- O Total Populacional é um agente que traz impacto positivo, mas que é o mais fraco de todos. Não causando muita diferença no PIB per capita.

- A Taxa de Matricula apresenta um impacto negativo, porém fraco. Um aumento nesta taxa está associado a uma ligeira queda no PIB per capita, o que pode indicar custos de curto prazo ou complexidades no modelo.

### PIB e Valor Adicionado em Lavras (milhões)

<img src="./reports/03_evolucao_pib_vs_va_lavras.png" alt="PIB e Valor Bruto Total de Preços adicionado em Lavras" width="600" >

A tabela acima compara o PIB e o Valor Adicionado Total do município de Lavras. Nele, vemos que o PIB é superior ao VA Total. O principal insight é que a diferença entre esses dois valores, que representa os Impostos Líquidos sobre Produtos, é um componente substancial na composição da riqueza total gerada em Lavras, destacando a importância da arrecadação fiscal sobre bens e serviços para a economia municipal.

### Evolução do Indice de Gini dos UFs Médio ao decorrer dos anos

<img src="./reports/04_grafico_evo_indice_gini_medio_por_setor.png" alt="Grafico" width="600">

O gráfico acima mostra o Indice de Gini Médio dos UFs por Setor entre os anos de 2001 a 2021. O Indice de Gini é um indicador que mensura a distribuição de renda em um território. Seu valor varia entre 0 e 1: quando mais próximo de 1, mais desigual é a distribuição de renda em um país; quanto mais próximo de 0, menor é essa desigualdade. O gráfico apresenta a média do indicador ao decorrer dos anos, nota-se que a maioria dos valores permaneceu estável ao decorrer dos anos, com exceção da Agropecuária a qual teve um leve aumento a partir de 2017.

### Matriz de Correlação de Brasil durante 2016 a 2020

<img src="./reports/05_matriz_correlacao_mun_brasil_2016_a_2020.png" alt="Grafico" width="600">

Na matriz de correlação, observamos como alguns indicadores de Lavras se relacionam entre si:

- O PIB per capita e o Valor Bruto Total a Preços Correntes per capita apresentam forte correlação, indicando que municípios com maior PIB per capita tendem a ter também maior Valor Bruto Total per capita.

- A correlação entre a Taxa de Alfabetização e o PIB per capita, assim como entre a Taxa de Alfabetização e o Valor Bruto Total per capita, foi moderada, sugerindo que municípios com maior renda tendem a ter maior alfabetização, embora essa relação não seja sempre consistente.

- A correlação entre a Taxa de Alfabetização e a Taxa de Matrícula foi a menor positiva, mostrando que a matrícula escolar nem sempre se traduz em alfabetização efetiva.

- Os demais indicadores apresentaram correlação negativa, indicando que aumentos em certos indicadores estão associados a reduções em outros.
<img src="./reports/06_matriz_correlacao_todos_os_municipios_mg_2016_a_2020.png" alt="Grafico" width="600">

### Gráfico entre valores de População e PIB

<img src="./reports/07_scatter_plot_pib_pop_brasil_2016_2020.png" alt="Grafico" width="600">

O gráfico acima demonstra uma relação linear entre as variáveis População e PIB (Produto Interno Bruto) no Brasil durante os anos de 2016 a 2020. Isso mostra que quanto maior a quantidade de habitantes em um múnicipio, a tendência é que o PIB (Produto Interno Bruto) aumente.

## Modelos de IA

### Clusterização por K-Means

<img src="./imgs/K-Means-Clusterizacao.png" alt="Imagem com fundo azul escuro com algumas partes claras ou brancas. Um gráfico na esquerda com a frase 'Before K-Means' monstra um conjunto de registros isolados em diferentes cantos, enquanto no gráfico na esquerda com a frase 'After K-Means' o conjunto de registros foram agrupados em 4 grupos diferentes baseados em sua similaridade." width="600">

#### O que é?

O agrupamento k-means é um algoritmo de aprendizado não supervisionado utilizado para agrupamento de dados, que agrupa pontos de dados não rotulados em grupos ou clusters. É um dos métodos de agrupamento mais populares usados em aprendizado de máquina. Diferentemente do aprendizado supervisionado, os dados de treinamento que esse algoritmo utiliza não são rotulados, o que significa que os pontos de dados não têm uma estrutura de classificação definida.

Embora existam vários tipos de algoritmos de agrupamento, incluindo exclusivos, sobrepostos, hierárquicos e probabilísticos, o algoritmo de agrupamento k-means é um exemplo de um método de agrupamento exclusivo ou "hard". Essa forma de agrupamento estipula que um ponto de dados pode existir em apenas um cluster. Esse tipo de análise de cluster é comumente utilizado em ciência de dados para segmentação de mercado, agrupamento de documentos, segmentação de imagens e compactação de imagens. O algoritmo k-means é um método amplamente utilizado na análise de clusters porque é eficiente, eficaz e simples.

O k-means é um algoritmo de agrupamento baseado em centroides iterativo, que divide um conjunto de dados em grupos semelhantes com base na distância entre seus centroides. O centroide, ou centro do cluster, é a média ou a mediana de todos os pontos dentro do cluster, dependendo das características dos dados.

<img src="./reports/08_barplot_metodo_cotovelo.png" alt="Grafico" width="600">
  
Para a seleção do número de clusters que serão utilizados no k-means, foi-se usado o Método do Cotovelo (Elbow Method) para encontrar o 'K' ideal. Tal método é baseada na análise do within-cluster sum of squares (WCSS), que mede a variação dentro dos clusters. A ideia é identificar o “cotovelo” no gráfico, onde a taxa de diminuição muda para cada k significativamente. Nesse projeto foi utilizado 5 clusters.

#### Implementação do Cluster Hierarquico

Adicionalmente, quando o K-Means é executado, obtem se dados com clusters a quais não se sabem exatamente o que eles significam, afinal K-Means é um algoritmo de Machine Learning Não-Supervisionado, os clusters são apenas pontos aos quais registros vão estar mais proximos. Para resover isso, utilizou-se de Clusters Hierarquicos, aos quais utilizam os Clusters Originais do K-Means como base. Eles analisam a Média do PIB de cada Cluster Original, e com isso, cria Clusters Hierarquicos a partir dos dados que estejam mais proximos desses.

Abaixo estão os gráficos gerados com os Clusters Hierarquicos, e a média do PIB per Capita:

<img src="./reports/09_distribuicao_municipios_por_nivel_de_riqueza_colorido.png" alt="Grafico" width="600">

<img src="./reports/09_barplot_media_pib_per_capita_por_cluster_hierarquico.png" alt="Grafico" width="600">

Análisando esses gráficos, vemos que a maioria dos múnicipios do Brasil se encontra em Cluster 0, o quao.

Apesar disso tudo, a abordagem utilizada possui limitações como por exemplo: . 

<img src="./reports/10_distribuicao_municipios_por_nivel_de_riqueza_lavras.png" alt="Grafico" width="600">

Ademais, o município de Lavras se encontra no Cluster 0, indicando ter menores níveis de PIB per Capita comparado com outros Clusters. Isso significa que [REDACTED].

#### Visualização Interativa dos Clusters no Brasil

<a href="./interactive_reports/03_mapa_clusters_2020_destaque_lavras.html" target="_blank">addsadas</a>

#### Análise de Gráficos

##### Média do PIB per Capita por Cluster Hierárquico

Utilizando a implementação do Cluster Hierarquico, foi possivel

<img src="./reports/11_distribuicao_clusters_minas_gerais.png" alt="Grafico" width="600">

<img src="./reports/12_boxplot_distribuicao_pib_per_capita_por_cluster_hierarquico.png" alt="Grafico" width="600">

### Random Forest Regressor

<img src="./imgs/Random-Forest-Regressao.png" alt="Imagem que descreve o funcionamento do Random Forest Regressão" width="600">

#### O que é?

Random forest é um algoritmo de aprendizado de máquina amplamente utilizado, registrado por Leo Breiman e Adele Cutler, que combina a saída de múltiplas decision trees para alcançar um único resultado. Sua facilidade de uso e flexibilidade impulsionaram sua adoção, pois lida com problemas de classificação e regression.

Métodos de aprendizado em conjunto são compostos por um conjunto de classificadores (por exemplo, árvores de decisão), e suas previsões são agregadas para identificar o resultado mais popular. Os métodos em conjunto mais conhecidos são bagging, também conhecido como agregação bootstrap, e boosting. Em 1996, Leo Breiman lançou o método bagging; nesse método, uma amostra aleatória de dados em um conjunto de treinamento é selecionada com reposição, o que significa que os pontos de dados individuais podem ser escolhidos mais de uma vez. Após a geração de várias amostras de dados, esses modelos são, então, treinados de forma independente, e dependendo do tipo de tarefa (por exemplo, regressão ou classificação), a média ou a maioria dessas previsões resulta em uma estimativa mais precisa. Essa abordagem é comumente usada para reduzir a variância em um conjunto de dados ruidoso.


### Regressão Linear Múltipla

<img src="./imgs/Regressao-Linear.jpg" alt="Grafico" width="600">

#### O que é?

O modelo de regressão linear múltipla é um modelo estatístico versátil para avaliar relacionamentos entre uma resposta contínua e os preditores.

Os preditores podem ser campos contínuos, categóricos ou derivados para que relacionamentos não lineares também sejam suportados. O modelo é linear porque consiste em termos aditivos em que cada termo é um preditor que é multiplicado por um coeficiente estimado. Um termo constante (intercepto) também é geralmente incluído no modelo.

A regressão linear é usada para gerar insights para gráficos que contêm pelo menos dois campos contínuos com um identificado como o destino e o outro como um preditor. Além disso, um preditor categórico e dois campos contínuos auxiliares podem ser especificados em um gráfico e usados para gerar um modelo de Regressão apropriado.

## Importância das Variáveis via SHAP

### O que é SHAP

### Valores SHAP obtidos

<img src="./reports/19_SHAP_impacto_agentes_socioeconomicos_pib_per_capita.png" alt="Grafico" width="600">


## Visualizações Interativas

## Limitações

## Recomendações Estratégicas

## Responsável

- Estevão Augusto da Fonseca Santos, Graduando em Ciência de Computação (6° Período)