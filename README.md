# Previsão de Churn com Otimização por Lucro

[![CI](https://github.com/juliamunizcarvalh0/previsao-churn/actions/workflows/ci.yml/badge.svg)](https://github.com/juliamunizcarvalh0/previsao-churn/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-API-009688)

Modelo de machine learning que estima a probabilidade de um cliente de telecom cancelar o serviço e decide **quem deve receber uma oferta de retenção**, escolhendo o limiar de decisão que maximiza o lucro da campanha, e não só uma métrica estatística.

O projeto cobre o ciclo completo: geração dos dados, comparação de modelos com validação cruzada, avaliação honesta em dados de teste, explicabilidade, API REST, testes automatizados, Docker e CI.

## Resultados

| Métrica (conjunto de teste) | Valor |
|---|---|
| ROC-AUC | **0,81** |
| PR-AUC (taxa base de churn: 24,6%) | 0,57 |
| Recall no limiar escolhido | 80% dos clientes que cancelariam são abordados |
| Lucro da campanha guiada pelo modelo | **R$ 61,6 mil** |
| Lucro abordando todos os clientes | R$ 26,3 mil |

Usar o modelo para escolher quem abordar **mais que dobra o lucro** em relação a oferecer desconto para todo mundo.

![Lucro por limiar](reports/figures/lucro_por_limiar.png)

![Importância das variáveis](reports/figures/importancia_variaveis.png)

O tipo de contrato e o tempo de casa são os fatores mais fortes. Região e streaming praticamente não influenciam, o que bate com o processo que gerou os dados.

## Decisões técnicas

- **Limiar escolhido pelo negócio, não pelo padrão de 0,5.** Cada abordagem custa R$ 50; cada cliente perdido custa 6 meses de receita; a oferta convence 35% de quem ia sair. O limiar que maximiza o lucro é escolhido nas previsões *fora da amostra* do treino (via `cross_val_predict`) para não vazar informação do teste.
- **Comparação entre um modelo simples e um complexo.** A regressão logística (ROC-AUC 0,807 na validação cruzada) empatou tecnicamente com o gradient boosting (0,800) e foi escolhida por ser mais simples e interpretável. Começar por um baseline forte é uma prática que muitos projetos pulam.
- **Pipeline único do scikit-learn.** Pré-processamento e modelo ficam no mesmo objeto, então a API aplica exatamente as mesmas transformações do treino.
- **Importância por permutação** no conjunto de teste, que mede o impacto real de cada variável na qualidade do modelo.
- **Validação de entrada na API** com Pydantic: valores fora do domínio (como `tipo_contrato: "semanal"`) retornam erro 422 em vez de uma previsão sem sentido.

### Limitações (e próximos passos)

- Os dados são **sintéticos** (`src/churn/data.py`), gerados com padrões conhecidos do setor, para o projeto rodar sem download. O próximo passo natural é aplicá-lo a um dataset público como o *Telco Customer Churn*.
- O `class_weight="balanced"` melhora o recall, mas descalibra as probabilidades (Brier 0,18). Uma calibração (`CalibratedClassifierCV`) deixaria as probabilidades mais fiéis.
- Monitorar *data drift* em produção e retreinar periodicamente.

## Como rodar

```bash
git clone https://github.com/juliamunizcarvalh0/previsao-churn.git
cd previsao-churn
python -m venv .venv && source .venv/bin/activate
make install   # dependências
make train     # treina e gera reports/
make test      # roda os testes
make api       # sobe a API em http://localhost:8000/docs
```

Ou com Docker (o modelo é treinado durante o build):

```bash
docker build -t previsao-churn .
docker run -p 8000:8000 previsao-churn
```

### Exemplo de uso da API

```bash
curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d '{
  "idade": 29, "tempo_cliente_meses": 2, "valor_mensal": 189.9,
  "chamados_suporte_90d": 5, "atrasos_pagamento_12m": 3,
  "tipo_contrato": "mensal", "forma_pagamento": "boleto", "internet": "fibra",
  "regiao": "sudeste", "tem_streaming": true, "tem_fidelidade_movel": false
}'
```

```json
{"probabilidade_churn": 0.9905, "risco": "alto", "recomendar_oferta_retencao": true}
```

Endpoints: `GET /health`, `POST /predict` e `POST /predict/batch` (até 1.000 clientes).

## Estrutura

```
├── src/churn/
│   ├── data.py     # geração dos dados sintéticos
│   ├── model.py    # pipelines, cálculo de lucro e escolha do limiar
│   ├── train.py    # treino, avaliação, gráficos e artefatos
│   └── api.py      # API FastAPI
├── tests/          # testes de dados, modelo e API (pytest)
├── reports/        # métricas e gráficos gerados no treino
├── Dockerfile
└── .github/workflows/ci.yml   # lint + testes + build Docker
```

## Tecnologias

Python · pandas · NumPy · scikit-learn · FastAPI · Pydantic · pytest · Docker · GitHub Actions

## Licença

MIT
