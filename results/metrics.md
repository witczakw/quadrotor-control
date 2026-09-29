#### Śledzenie trajektorii (estymata ESKF, bez zakłóceń)

| Scenariusz | Regulator | RMSE [m] | Błąd max [m] | Energia [J] | Chybienie bramki [m] |
|---|---|---|---|---|---|
| figure8 | PID | 0.0803 | 0.214 | 7344.4 |  |
| figure8 | Geometric | 0.0725 | 0.1998 | 7231.9 |  |
| figure8 | LQR | 0.0802 | 0.2108 | 7291.4 |  |
| figure8 | MPC | 0.0847 | 0.2443 | 7280.3 |  |
| helix | PID | 0.0855 | 0.2979 | 8341.0 |  |
| helix | Geometric | 0.0733 | 0.174 | 8188.7 |  |
| helix | LQR | 0.0815 | 0.2462 | 8333.2 |  |
| helix | MPC | 0.0918 | 0.2948 | 8339.3 |  |
| slalom | PID | 0.176 | 0.4615 | 5970.9 | 0.192 |
| slalom | Geometric | 0.0799 | 0.2056 | 5640.7 | 0.12 |
| slalom | LQR | 0.133 | 0.3139 | 5858.2 | 0.075 |
| slalom | MPC | 0.1282 | 0.3072 | 5861.6 | 0.11 |
| aggressive_turn | PID | 0.1435 | 0.2864 | 4256.4 |  |
| aggressive_turn | Geometric | 0.1103 | 0.212 | 3998.6 |  |
| aggressive_turn | LQR | 0.1502 | 0.3116 | 4255.5 |  |
| aggressive_turn | MPC | 0.1623 | 0.3238 | 4305.8 |  |

#### Śledzenie trajektorii z idealną informacją o stanie (bez EKF)

| Scenariusz | Regulator | RMSE [m] | Błąd max [m] | Energia [J] | Chybienie bramki [m] |
|---|---|---|---|---|---|
| figure8 | PID | 0.0511 | 0.172 | 7365.5 |  |
| figure8 | Geometric | 0.0247 | 0.0855 | 7240.4 |  |
| figure8 | LQR | 0.0385 | 0.1377 | 7307.8 |  |
| figure8 | MPC | 0.0463 | 0.1368 | 7296.4 |  |
| helix | PID | 0.0498 | 0.1838 | 8331.0 |  |
| helix | Geometric | 0.0243 | 0.0823 | 8195.8 |  |
| helix | LQR | 0.0458 | 0.1719 | 8328.7 |  |
| helix | MPC | 0.0591 | 0.1698 | 8330.7 |  |
| slalom | PID | 0.1364 | 0.2868 | 5881.9 | 0.143 |
| slalom | Geometric | 0.0296 | 0.0586 | 5607.0 | 0.033 |
| slalom | LQR | 0.11 | 0.1825 | 5799.3 | 0.139 |
| slalom | MPC | 0.1021 | 0.1805 | 5790.5 | 0.148 |
| aggressive_turn | PID | 0.1129 | 0.1839 | 4248.1 |  |
| aggressive_turn | Geometric | 0.0525 | 0.0968 | 3995.5 |  |
| aggressive_turn | LQR | 0.1224 | 0.2505 | 4252.3 |  |
| aggressive_turn | MPC | 0.1404 | 0.2777 | 4300.5 |  |

#### Odporność: RMSE śledzenia [m] (estymata ESKF)

| Scenariusz | Regulator | nominal | wind | mass+20% | mass-20% | rotor_fault_30% |
|---|---|---|---|---|---|---|
| figure8 | PID | 0.0803 | 0.0884 | 0.0914 | 0.0796 | 0.0849 |
| figure8 | Geometric | 0.0725 | 0.0873 | 0.1023 | 0.0766 | 0.1362 |
| figure8 | LQR | 0.0802 | 0.0897 | 0.0828 | 0.081 | 0.0859 |
| figure8 | MPC | 0.0847 | 0.0956 | 0.0885 | 0.0844 | 0.0928 |
| helix | PID | 0.0855 | 0.0951 | 0.0963 | 0.0869 | 0.0903 |
| helix | Geometric | 0.0733 | 0.0863 | 0.0973 | 0.0825 | 0.1268 |
| helix | LQR | 0.0815 | 0.09 | 0.0841 | 0.0822 | 0.0873 |
| helix | MPC | 0.0918 | 0.1 | 0.0956 | 0.0917 | 0.1003 |
| slalom | PID | 0.176 | 0.1869 | 0.2807 | 0.1423 | 0.2311 |
| slalom | Geometric | 0.0799 | 0.0916 | 0.1407 | 0.1143 | 0.1832 |
| slalom | LQR | 0.133 | 0.1428 | 0.1451 | 0.1237 | 0.1389 |
| slalom | MPC | 0.1282 | 0.137 | 0.1481 | 0.1119 | 0.1377 |
| aggressive_turn | PID | 0.1435 | 0.1469 | 0.1923 | 0.1368 | 0.1712 |
| aggressive_turn | Geometric | 0.1103 | 0.1324 | 0.1606 | 0.1358 | 0.2531 |
| aggressive_turn | LQR | 0.1502 | 0.156 | 0.1622 | 0.1415 | 0.1561 |
| aggressive_turn | MPC | 0.1623 | 0.1657 | 0.1773 | 0.1494 | 0.1703 |

#### Odpowiedź na skok 1 m (po przekątnej, stan idealny)

| Regulator | Czas ustalania (5 cm) [s] | Przeregulowanie [m] | Energia [J] |
|---|---|---|---|
| PID | 2.19 | 0.0 | 3527.4 |
| Geometric | 1.39 | 0.029 | 3518.3 |
| LQR | 1.89 | 0.09 | 3690.9 |
| MPC | 1.84 | 0.053 | 3658.0 |
