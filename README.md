# Predictive Maintenance

Sistema de alerta temprana de fallas para una flota de 100 máquinas (Microsoft Azure Predictive
Maintenance), proyecto del curso Machine Learning y Deep Learning. Tarea principal: probabilidad de
falla de cualquier componente en las siguientes 24 h; tarea secundaria: qué componente (5 clases).

## Requisitos

- Python 3.10 o superior
- Jupyter Notebook o JupyterLab

```bash
python -m pip install -r requirements.txt
```

Las redes corren en CPU o GPU (PyTorch detecta CUDA). En CPU de 2 núcleos la semana 3 completa
tarda unos 25 min si los modelos ya están en `models/week3/`, y unos 45 min si hay que entrenarlos.

## Organización de los datos

Los CSV originales deben estar dentro de `archive/` (no se suben al repositorio):

```text
archive/
├── PdM_telemetry.csv
├── PdM_errors.csv
├── PdM_maint.csv
├── PdM_failures.csv
└── PdM_machines.csv
```

Todos los notebooks se ejecutan desde la raíz del proyecto.

## Orden de ejecución

| Paso | Archivo | Qué hace |
|---|---|---|
| 1 | `week1_predictive_maintenance_serie_cruda.ipynb` | EDA, relación mantenimiento-fallas, etiquetas H = 24 h, partición temporal, pruebas de fuga, líneas base (regresión logística y XGBoost sobre la serie cruda de 13 canales × 24 h). Escribe `data/processed/{train,validation,test}.parquet` y `week1_validation_predictions.csv.gz`. |
| 2 | `week2_deep_learning_baselines_week1.ipynb` | Tensor de entrada, LSTM / TCN / CNN-LSTM con L ∈ {24, 72, 168}, estrategias de desbalance, Optuna. Guarda `models/week2_best.pt` (TCN, L = 24, sobremuestreo). |
| 3 | `week3_advanced_reliability_policy.ipynb` | Transformer con token de máquina, ensamble de 5 TCN y MC Dropout, calibración, gradientes integrados, política de mantenimiento por costo, **evaluación única en prueba**, costo anual y sensibilidad. |

`week1_predictive_maintenance.ipynb` es la versión anterior de la semana 1, con variables
construidas (promedios móviles); se conserva como referencia, pero las semanas 2 y 3 usan la serie cruda.

Código reutilizable:

- `pm_dl.py`: canales horarios, ventanas bajo demanda, arquitecturas (LSTM/GRU, TCN, CNN-LSTM y
  Transformer), estrategias de desbalance, entrenamiento con early stopping y métricas.
- `pm_week3.py`: ensamble y MC Dropout, *temperature scaling*, gradientes integrados, simulación de
  la política de alertas y del reemplazo por calendario.
- `train_week3.py`: entrena (o carga) las redes de la semana 3 en `models/week3/`. El notebook de
  la semana 3 usa exactamente los mismos argumentos, así que correrlo antes solo llena el caché.

## Partición temporal

```text
Entrenamiento:  2015-01-01 hasta antes de 2015-08-31
Validación:     2015-09-01 hasta antes de 2015-10-31
Prueba:         2015-11-01 hasta 2016-01-01
```

Se deja un margen de 24 h antes de cada frontera para que ninguna etiqueta dependa de fallas
posteriores al periodo. Las estadísticas de normalización se calculan solo con horas de entrenamiento.

## Protocolo de prueba (semana 3)

Todas las decisiones (modelo final, temperatura, umbral de alerta, periodos de calendario) se toman con
entrenamiento y validación y se guardan en `data/processed/week3_decisions.json` antes de abrir el
conjunto de prueba. La sección *Evaluación final en prueba* del notebook lee ese archivo y aplica
las decisiones tal cual.

## Archivos generados

```text
data/processed/
├── train.parquet / validation.parquet / test.parquet   (semana 1)
├── week1_validation_predictions.csv.gz                  (semana 1)
├── week2_*.csv                                          (semana 2)
├── week3_decisions.json          decisiones congeladas antes de prueba
├── week3_validation_comparison.csv
├── week3_calibration_validation.csv
├── week3_ig_channel_share.csv
├── week3_test_metrics.csv
├── week3_annual_cost.csv         costo anual de las tres políticas
├── week3_annual_cost_bootstrap.csv  intervalos de 95 % (bootstrap por máquina) en prueba
├── week3_sensitivity.csv         sensibilidad a la razón falla / falsa alarma
└── week3_summary.json
figures/week2/, figures/week3/   figuras de los reportes
docs/                            reporte final (PDF y fuente LaTeX) y presentación
models/week2_best.pt, models/week3/*.pt
```

Nota: `models/week2_best.pt` contiene los hiperparámetros del ensayo 3 de Optuna (dropout 0.023,
λ = 0.25, lr 1.3e-3, 64 canales). La salida impresa en el notebook de la semana 2 corresponde a una
ejecución anterior; el archivo de pesos coincide con `runs/week2/tuned.npz` (la semana 3 lo verifica).

Los archivos `.csv` y `.csv.gz` están excluidos por `.gitignore`.
