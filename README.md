# tcc-detector

Experimento do TCC: **generalização de uma CNN (ResNet-50, transferência de aprendizado) na detecção de imagens geradas por IA**, comparando geradores vistos × não vistos no treino. Este repositório corresponde ao **MVP do TCC-I** (escala reduzida); o TCC-II aumenta a escala e as análises.

## Fluxo de trabalho

- **Notebook local (CPU):** escrever código, rodar `make test`, pré-processamento e testes de vazamento.
- **Kaggle (GPU):** gerar imagens, treinar e avaliar. Abra `notebooks/kaggle_entry.ipynb`, clone este repositório e execute.
- **GitHub:** versionamento. O que é gerado (`data/`, `results/`) não é versionado.

## Estrutura

| Pasta | Conteúdo |
|---|---|
| `configs/` | `experiment.yaml` (parâmetros), `generators.yaml` (geradores/versões), `smoke.yaml` (teste rápido) |
| `src/tccdet/generate/` | Adaptadores de geradores (diffusers, BigGAN, StyleGAN2) e gerador em lote |
| `src/tccdet/` | `collect_real`, `preprocess`, `split`, `dataset`, `model`, `train`, `evaluate`, `aggregate`, `shortcuts`, `metrics` |
| `tests/` | Testes de vazamento, balanceamento, deduplicação e métricas |

## Passo a passo

```bash
make setup                  # instala dependências
make test                   # 13 testes (sem GPU)
make smoke                  # pipeline completo com dados FALSOS (rode primeiro no Kaggle)

# dados reais
make generate G=sd15        # repetir para cada gerador
make real                   # amostra as imagens reais (ajuste real.source_dir)

# pipeline
make preprocess split shortcuts
make train eval SEED=0
make aggregate              # média ± desvio entre sementes
```

Saídas em `results/<split>/seed<N>/`: `metrics.json` (Testes A e B, ICs 95%, ΔF1/Δacc/ΔAUC), `per_group.csv` (por gerador e família), `scores_*.csv`.

## Decisões que ainda precisam ser fechadas (e registradas no capítulo 3)

1. **StyleGAN2:** preencher `checkpoints` em `generators.yaml`. Confira quais categorias possuem checkpoint público; se faltar, **troque a categoria**, não improvise.
2. **Índices de classe do BigGAN** e versões dos modelos: conferir antes de gerar em massa.
3. **SDXL-Turbo** como "não visto" é da mesma empresa que o SD 1.5. Se quiser mais distância entre vistos e não vistos, troque por um modelo de outro laboratório (`generators.yaml`).
4. **Licenças e termos de uso** de cada modelo e da base de imagens reais.
5. **Reais da ImageNet:** o backbone pré-treinado já viu essas imagens. Registre como limitação ou use outra fonte (LSUN, COCO, Open Images).
6. **Regra de dimensionamento:** `real.n_per_category` ≥ nº de geradores × `generation.n_per_category`. O `split` falha com erro claro se faltarem reais.

## Mapa para o texto do TCC

| Objetivo específico | Onde está no código |
|---|---|
| b — conjunto de dados | `generate/`, `collect_real`, `preprocess`, `split` |
| c — treinar a ResNet-50 | `model`, `train` |
| d — avaliar (acurácia, precisão, revocação, F1, AUC, matriz) | `metrics`, `evaluate` |
| e — variação entre geradores/famílias | `per_group.csv`; no TCC-II, `protocol: logo` |

## Limitações do MVP

- `make test` e `make smoke` cobrem a lógica de dados e métricas; os módulos que dependem de GPU/geradores reais (`generate/`, `train`, `evaluate`) só foram checados quanto à sintaxe e devem ser validados no Kaggle com `make smoke` antes de gerar dados.
- Com ~240 imagens por gerador os resultados são **preliminares**: servem para provar viabilidade, não para concluir.
- O baseline de atalhos (`shortcuts`) deve ficar próximo de AUC 0,5 nos atributos `processed` com dados reais; se ficar alto, revise a padronização antes de treinar.
