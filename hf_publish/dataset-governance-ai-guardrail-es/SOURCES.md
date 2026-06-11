# Sources and Upstream Notes

Este repositorio consolida material derivado de múltiples datasets de terceros y de trabajo correctivo propio.

## Upstream datasets integrados

- `Necent/llm-jailbreak-prompt-injection-dataset`
- `bogdanminko/Catch_the_prompt_injection_or_jailbreak_or_benign`
- `J1N2/mix-prompt-injection-dataset`
- `huyhoangdinhcong/guardrails-dataset-full`

## Transformaciones aplicadas

- traducción a español
- mutaciones lingüísticas en español
- balanceo por clase
- downsampling por fuente y etiqueta
- consolidación en formato parquet
- construcción de hard negatives
- creación de dataset correctivo desde errores del golden set

## Nota de reutilización

Antes de reutilizar o redistribuir este dataset, revisar:

1. licencia de cada fuente aguas arriba
2. limitaciones sobre contenido sensible
3. restricciones sobre redistribución o uso comercial
4. políticas internas de cumplimiento para contenido dañino o adversarial

## Trabajo adicional propio

El valor agregado principal de Governance AI en este repositorio es:

- normalización y esquema de etiquetas unificado
- traducción y mutaciones en español
- balanceo realista para fine-tuning
- corrección de falsos positivos a partir de evaluación adversarial
- empaquetado reproducible para entrenamiento guardrail
