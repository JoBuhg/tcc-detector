# tcc-detector

Experimento do TCC **"Avaliação da generalização de uma rede neural convolucional na detecção de imagens geradas por inteligência artificial"**. Este repositório é o **MVP do TCC-I** (escala reduzida). O TCC-II amplia os dados, as repetições e as análises.

### Treinamento e avaliação

- **Treino em duas fases:** (1) só a camada final, taxa 1e-3, 3 épocas; (2) último bloco convolucional (`layer4`), taxa 1e-4, até 20 épocas com parada antecipada pela perda de validação (paciência 3). Adam, entropia cruzada binária, lote de 32, único aumento de dados: espelhamento horizontal. Limiar de decisão 0,5.
- **Sementes:** uma no TCC-I (`seeds: [0]`);
- **Métricas:** acurácia, precisão, revocação, F1-score, taxa de falsos positivos, AUC e matriz de confusão, com intervalos de confiança de 95% por *bootstrap*; também por gerador.
- **Baseline de atalhos:** classificador simples sobre atributos não semânticos (tamanho de arquivo, nitidez, estatísticas de cor). AUC próxima de 0,5 nos atributos `processed` indica ausência de atalhos óbvios.
- **Teste de sensibilidade:** reavalia os Testes A e B sob JPEG 70 e 50, desfoque e redução de escala. Queda acentuada indica dependência de artefatos frágeis, entre eles marcas d'água invisíveis.

## Ambiente

Tudo roda no **notebook, em CPU** (sem GPU). Os comandos abaixo são para Windows (PowerShell); `make` normalmente não existe no Windows, então os comandos estão por extenso. O `notebooks/kaggle_entry.ipynb` e `configs/generators_open.yaml` (Stable Diffusion, StyleGAN2, BigGAN) ficam como opção para o TCC-II.

## Instalação

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
pytest -q            # 19 testes
```

Na primeira execução com `pretrained: true`, o PyTorch baixa os pesos da ResNet-50 (cerca de 100 MB).

## Teste de fumaça (imagens fictícias)

Rode **primeiro**: é a primeira execução real de `train`, `evaluate` e `robustness`. Depois apague o conteúdo de `data/raw`, `data/processed`, `data/splits` e `results`.

```powershell
python scripts/make_fake_data.py --config configs/smoke.yaml
python -m tccdet.preprocess --config configs/smoke.yaml
python -m tccdet.split --config configs/smoke.yaml
python -m tccdet.shortcuts --config configs/smoke.yaml
python -m tccdet.train --config configs/smoke.yaml --split fixed --seed 0
python -m tccdet.evaluate --config configs/smoke.yaml --split fixed --seed 0
python -m tccdet.robustness --config configs/smoke.yaml --split fixed --seed 0
```

## Fluxo com dados reais

```powershell
# 1. imagens reais: coloque >= 80 por pasta em data/real_source/{cat,car,horse}/
python -m tccdet.collect_real

# 2. lista de prompts (os mesmos para todos os geradores)
python -m tccdet.manual prompts          # -> data/manual/prompts.csv

# 3. gere as imagens nas ferramentas e salve em data/inbox/<gerador>/<categoria>/
#    geradores: chatgpt, gemini, copilot, ideogram | categorias: cat, car, horse

# 4. organizar e conferir
python -m tccdet.manual ingest           # -> data/raw/ e data/registry.csv
python -m tccdet.manual status           # mostra quantas imagens faltam

# 5. preparar os dados
python -m tccdet.preprocess
python -m tccdet.split
python -m tccdet.shortcuts

# 6. treinar e avaliar
python -m tccdet.train --split fixed --seed 0
python -m tccdet.evaluate --split fixed --seed 0
python -m tccdet.robustness --split fixed --seed 0
```

Com `make` (Linux/macOS): `make prompts ingest status real preprocess split shortcuts train eval robustness`.

### Regras da geração manual

- Use **o mesmo prompt** em todos os geradores, uma imagem por prompt e por gerador, na ordem do CSV (20 por categoria).
- Baixe as imagens na ordem de geração: o `ingest` ordena por data de modificação e depois por nome, e o prompt registrado segue essa ordem.
- No `data/registry.csv`, preencha `visible_watermark` (sim/não) e anote plano de uso e data. Marcas visíveis devem ser cortadas pelo recorte, ou a imagem descartada.
- Confira, nos termos de cada ferramenta, se o uso em pesquisa é permitido e quais marcas d'água ou metadados de procedência são aplicados.
- Regra de dimensionamento: `real.n_per_category` ≥ nº de geradores × `generation.n_per_category` (4 × 20 = 80). O `split` falha com erro claro se faltarem reais.

## Saídas

Em `results/fixed/seed0/`:

| Arquivo | Conteúdo |
|---|---|
| `model.pt`, `history.json` | melhor modelo e curva de treino |
| `metrics.json` | Testes A e B, intervalos de confiança, ΔAUC/ΔF1/Δacc |
| `per_group.csv` | métricas por gerador |
| `scores_test_a.csv`, `scores_test_b.csv` | probabilidade de cada imagem |
| `robustness.csv` | AUC e F1 sob cada perturbação |

Também: `results/shortcuts.json` (baseline de atalhos) e `data/splits/fixed/split_info.json` (contagens por conjunto).

## Estrutura

| Pasta | Conteúdo |
|---|---|
| `configs/` | `experiment.yaml` (parâmetros), `generators.yaml` (comerciais, manuais), `generators_open.yaml` (modelos abertos), `smoke.yaml` |
| `src/tccdet/` | `manual`, `collect_real`, `preprocess`, `split`, `dataset`, `model`, `train`, `evaluate`, `robustness`, `perturb`, `aggregate`, `shortcuts`, `metrics`, `prompts`, `generate/` (modelos abertos, 

## Mapa para o texto do TCC

| Objetivo específico | Onde está no código |
|---|---|
| a — investigar conceitos e métodos | revisão da literatura (documento) |
| b — conjunto de dados controlado | `manual`, `collect_real`, `preprocess`, `split` |
| c — treinar a ResNet-50 | `model`, `train` |
| d — avaliar (acurácia, precisão, revocação, F1, AUC, matriz) | `metrics`, `evaluate`, `robustness` |
| e — variação entre geradores | `per_group.csv` |
| f — pipeline reprodutível, versionado e testado | repositório, `tests/`, `configs/` |

## Limitações do MVP

- Os módulos que usam PyTorch (`train`, `evaluate`, `robustness`) foram verificados apenas quanto à sintaxe; rode o teste de fumaça primeiro.
- Sem geradores de arquitetura conhecida, não há análise por família (GAN × difusão) no TCC-I.

## TCC-II

Mais imagens (5 categorias, 60 por gerador e categoria), cinco sementes (`train.seeds`), `python -m tccdet.aggregate` para média e desvio, `split.protocol: logo` (*leave-one-generator-out*) e geradores abertos (`generators_open.yaml`) para a análise por família.
