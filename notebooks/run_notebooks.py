"""
Pipeline Automático - Análise Preditiva
Executa sequencialmente: 1_coleta → 2_eda → 3_modelo
"""

import sys                  # Biblioteca de interação com o interpretador Python
from pathlib import Path    # Biblioteca de tratamento de caminhos representados via classes
import time                 # Biblioteca de gestão de tempo e conversão
import nbformat
from nbclient import NotebookClient


def executar_notebook(nome_notebook: str) -> bool:
    """Executa um notebook Jupyter e retorna True se sucesso"""
    caminho = Path(nome_notebook)
    
    if not caminho.exists():
        raise FileNotFoundError(f"ERRO: {nome_notebook} não encontrado!")
    
    print(f"\n[{time.strftime('%H:%M:%S')}] Executando: {caminho.name}")
    
    try:
        nb = nbformat.read(caminho, as_version=4)
        
        client = NotebookClient(nb)
        client.execute()
        
        print(f"OK: {caminho.name}")
        return True
            
    except Exception as e:
        print(f"EXCEÇÃO: {caminho.name} - {e}")
        return False

def main():
    print("PIPELINE DE ANÁLISE PREDITIVA")
    print("=" * 50)
    
    # Seus notebooks EXATAMENTE na ordem
    notebooks = [
        "1_coleta_preparacao_dados.ipynb",
        "2_analise_exploratoria.ipynb", 
        "3_modelo_predicao.ipynb"
    ]
    
    sucessos = 0
    total = len(notebooks)
    
    for i, notebook in enumerate(notebooks, 1):
        print(f"[{i}/{total}] : {notebook}")
        if executar_notebook(notebook):
            sucessos += 1
        else:
            raise RuntimeError(f"ERRO: {notebook} não foi executado com sucesso!")
        print("-" * 50)
    
    # Relatório final
    print("\n RESUMO FINAL")
    print(f" Sucessos: {sucessos}/{total}")
    print(f" Falhas:   {total - sucessos}/{total}")
    
    if sucessos == total:
        print("\n PIPELINE CONCLUÍDO COM SUCESSO!")
        sys.exit(0) # Script finalizou com sucesso
    else:
        print("\n ALGUNS NOTEBOOKS FALHARAM!")
        sys.exit(1) # Script finalizou com falhas

if __name__ == "__main__":
    main()