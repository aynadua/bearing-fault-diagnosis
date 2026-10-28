# Results

Trained only on condition(s) **A**, tested on every condition. Operating points: A=1500 rpm, B=1800 rpm, C=2400 rpm, D=3000 rpm.

## Accuracy by operating condition

| Model | A (source) | B | C | D | worst drop |
|---|---:|---:|---:|---:|---:|
| envelope features + GBM | 0.993 | 0.972 | 0.922 | 0.832 | 0.161 |
| CNN (raw) | 1.000 | 0.500 | 0.324 | 0.293 | 0.707 |
| CNN (order) | 1.000 | 0.961 | 0.399 | 0.250 | 0.750 |
| CNN (envelope_order) | 0.982 | 0.957 | 0.884 | 0.758 | 0.224 |

![domain shift](domain_shift.png)

## Domain gap

| Model | In-domain | Mean out-of-domain | Worst out-of-domain |
|---|---:|---:|---:|
| envelope features + GBM | 0.993 | 0.909 | 0.832 |
| CNN (raw) | 1.000 | 0.373 | 0.293 |
| CNN (order) | 1.000 | 0.537 | 0.250 |
| CNN (envelope_order) | 0.982 | 0.866 | 0.758 |