#### Śledzenie trajektorii (estymata ESKF, bez zakłóceń)

| Scenariusz | Regulator | RMSE [m] | Błąd max [m] | Energia [J] | Chybienie bramki [m] |
|---|---|---|---|---|---|
| figure8 | PID | 0.0963 | 0.227 | 5612.3 |  |
| figure8 | Geometric | 0.0869 | 0.2032 | 5480.7 |  |
| figure8 | LQR | 0.094 | 0.2197 | 5569.1 |  |
| figure8 | MPC | 0.1055 | 0.2555 | 5568.3 |  |

#### Odporność: RMSE śledzenia [m] (estymata ESKF)

| Scenariusz | Regulator | nominal |
|---|---|---|
| figure8 | PID | 0.0963 |
| figure8 | Geometric | 0.0869 |
| figure8 | LQR | 0.094 |
| figure8 | MPC | 0.1055 |

#### Odpowiedź na skok 1 m (po przekątnej, stan idealny)

| Regulator | Czas ustalania (5 cm) [s] | Przeregulowanie [m] | Energia [J] |
|---|---|---|---|
| PID | 2.19 | 0.0 | 3527.4 |
| Geometric | 1.39 | 0.029 | 3518.3 |
| LQR | 1.89 | 0.09 | 3690.9 |
| MPC | 1.84 | 0.053 | 3658.0 |
