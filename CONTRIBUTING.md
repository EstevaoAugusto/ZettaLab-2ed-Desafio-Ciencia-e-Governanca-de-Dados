# Guia de Contribuição e Estilo de Commit

Este documento serve como um lembrete rápido de como manter a organização do código e a consistência das mensagens de commit para este projeto.

## Conventional Commits

Cada commit realizado no projeto deve seguir o padrão especificado pelo [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/), uma convenção que define um conjunto de regras para criar um histórico de commit explícito, o que facilita a criação de ferramentas automatizadas baseadas na especificação. A mensagem do commit deve ser estruturada da seguinte forma:

```git
<tipo>[escopo opcional]: <descrição>

[corpo opcional]

[rodapé(s) opcional(is)]
```

### Tipos de Commit


| Tipo | Quando usar? |
| :--- | :--- |
| **feat** | Inclui um novo recurso na sua base de código. |
| **fix** | Soluciona um problema na sua base de código. |
| **docs** | Alterações neste arquivo, no README ou comentários explicativos no código. |
| **data** | Quando houver atualização de arquivos brutos na pasta `/data` ou `/features`. |
| **refactor** | Melhoria no código de limpeza sem alterar o resultado final. |
| **style** | Mudanças puramente visuais (ex: paleta de cores do Seaborn). |
| **chore** | Manutenção de pastas, salvamento de modelos `.pkl` ou atualização de bibliotecas. |

---

## Escopos Específicos do Projeto
Para facilitar a busca no histórico (`git log`), use estes escopos entre parênteses:

- **(data)**: Processamento de arquivos brutos.
- **(eda)**: Análises na matriz de correlação e Seaborn.
- **(model)**: Treinamento, salvamento e métricas da Regressão Linear.
- **(plots)**: Visualizações interativas com Plotly.
- **(lavras)**: Análises específicas do município de Lavras.

---

## Exemplos Reais para Revisitar

- `feat(eda): adiciona matriz de correlação de Spearman para VA agropecuário`
- `fix(features): garante ordem das colunas para compatibilidade entre 2019 e 2020`
- `data(processed): resolve redundância de IDs para integração de novos municípios`
- `model(lavras): salva regressor treinado e scaler em /models`
- `style(plots): muda escala de cores do heatmap para RdBu_r`

---
