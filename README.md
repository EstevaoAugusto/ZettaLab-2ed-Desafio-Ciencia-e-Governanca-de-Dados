# Desafio ZettaLab - Ciência e Governança de Dados

O projeto é um trabalho realizado individualmente na trilha "Ciência e Governança de Dados" do projeto ZettaLab da UFLA (Universidade Federal de Lavras). O desafio proposto foi de acessar e reunir diversas bases de dados brasileiras relevantes que incluem informações capazes de avaliar as condições socioeconômicas e seus determinantes multifatoriais, a qual seja feito limpeza e tratamento dos dados, análise estatística inicial e identificação de padrões, correlações e tendências relevantes., e realizar análise exploratória cujo proposito é gerar insights sobre eles.

## Link dos Banco de Dados

- [Base dos Dados](https://basedosdados.org/)
    - [Produto Interno Bruto (PIB)](https://basedosdados.org/dataset/fcf025ca-8b19-4131-8e2d-5ddb12492347?table=fbbbe77e-d234-4113-8af5-98724a956943)
    - [População Brasileira](https://basedosdados.org/dataset/1e2b9a88-9dc7-4f0e-a3a5-e8d2a13869bf?table=1a8d9636-c11d-443b-ae83-1b00576f0b70)
    - 
- [Dados Gov](https://dados.gov.br/home)
    - 
    - 

## Como acessar

O projeto usou o Jupyter Notebook, e por padrão, os dados csv necessários já estão salvos numa pasta dedicada a eles. Não é preciso executar o projeto pois as saídas originais já estão preservadas. Logo, certas partes do tutorial de como acessar podem ser puladas (esses que estarão explicitos ao decorrer do README.md). Contudo, caso queira realizar o build completo do zero e executar tudo no Notebook, siga todo que esteja escrito.

### Pré-requisitos

- [Python 3.13.1](https://www.python.org/)
- pip (Gerenciador de Pacotes do Python)

### Passo-a-Passo

No terminal Git Bash, clone o repositório
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
./.venv/Scripts/activate
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

Cadastre uma conta no Google Cloud, e crie um projeto. Após isso, coloque seu ID no .env
```bash
echo "GOOGLE_CLOUD_PROJECT='<seu-projeto-id>'" > .env
```

Execute o projeto