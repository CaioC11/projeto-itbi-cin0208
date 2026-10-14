# Decisões de integração

- As branches `origin/eda-alvo`, `origin/eda-numericas` e `origin/eda-categoricas` já eram ancestrais da `main` no commit `66bc1e8`. A branch `fabinho` parte desse histórico, preservando os commits dos participantes.
- A versão completa `01b_eda_numericas (3).ipynb`, entregue por NyckolasJGF, foi consolidada como `01b_eda_numericas.ipynb`. As versões preliminares continuam disponíveis no histórico do Git.
- As análises de Caio César (alvo) e Gabriel Ramos (categóricas) foram renomeadas conforme a ordem documentada. A leitura do CSV foi centralizada sem descarte de linhas e sem clonar repositórios dentro dos notebooks.
- O dataset bruto permanece intacto. Campos numéricos com vírgula são convertidos como notação brasileira; coordenadas com ponto decimal mantêm esse formato.
- A EDA propõe manter valores altos e investigar valores muito baixos. Não se removeu nenhuma observação nem se tomou decisão de modelagem durante esta integração.
- `area_terreno`, `fracao_ideal` e `sfh` permanecem candidatas a experimentos; redundância e correlação baixa não foram tratadas como descarte definitivo.
- Pearson de 0,92 não significa que área e localização expliquem 92% da variância. O README distingue achados exploratórios de métricas de modelagem.
- Não havia branch `pipeline`, notebook de treinamento ou imagens `Assets/` nas referências remotas após o fetch. O README mantém o estilo enviado, remove referências a imagens locais inexistentes e identifica os experimentos pendentes.

## Validação da integração

- Python 3.12.15 com as versões de dependências de `requirements.txt`.
- Os três notebooks executaram do início ao fim em kernels novos, com diretório de trabalho em `notebooks/`, sem erros.
- As colunas numéricas do carregador foram comparadas à leitura original com `decimal=","`: valores preservados, incluindo o alvo mínimo de R$ 19,52.
- Dataset com 16.318 linhas e 22 colunas; CSV bruto sem alterações.
- Formato dos notebooks validado e células Python compiladas; `git diff --check` sem problemas.
