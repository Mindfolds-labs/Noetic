# Resultado de treinamento real (PyFolds + Digits)

- Dataset: sklearn digits (real), train=1437, test=360
- Épocas: 10
- Device: cpu

## PyFolds + Head linear

| epoch | train_loss | train_acc | test_acc |
|---:|---:|---:|---:|
| 1 | 2.3184 | 0.1009 | 0.0889 |
| 2 | 2.2971 | 0.112 | 0.125 |
| 3 | 2.2917 | 0.1406 | 0.1167 |
| 4 | 2.2755 | 0.1621 | 0.1361 |
| 5 | 2.261 | 0.1531 | 0.1444 |
| 6 | 2.2563 | 0.1587 | 0.1417 |
| 7 | 2.2571 | 0.1496 | 0.1361 |
| 8 | 2.2589 | 0.1468 | 0.1278 |
| 9 | 2.2653 | 0.1406 | 0.1111 |
| 10 | 2.2955 | 0.1239 | 0.1111 |

## Baseline MLP (sem PyFolds)

| epoch | train_loss | train_acc | test_acc |
|---:|---:|---:|---:|
| 1 | 2.18 | 0.4809 | 0.7083 |
| 2 | 1.8355 | 0.801 | 0.8306 |
| 3 | 1.3991 | 0.8553 | 0.8583 |
| 4 | 0.9945 | 0.881 | 0.8778 |
| 5 | 0.7169 | 0.9074 | 0.9028 |
| 6 | 0.5461 | 0.9207 | 0.9139 |
| 7 | 0.4407 | 0.9318 | 0.9278 |
| 8 | 0.37 | 0.9339 | 0.9361 |
| 9 | 0.3183 | 0.9402 | 0.9444 |
| 10 | 0.2784 | 0.9485 | 0.9389 |

PyFolds teste final: **0.1111**
Baseline teste final: **0.9389**

Conclusão inicial: baseline MLP ainda supera bastante esta configuração de PyFolds; a integração precisa de tuning (input encoding, reward schedule e parâmetros MPJRD).