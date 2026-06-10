"""
Lección: 03-cnns-desde-lenet-hasta-resnet
Fase: 04
Prerrequisitos: 02-convoluciones-desde-cero
"""
from __future__ import annotations
import sys
import numpy as np


def bloque_conv_bn_relu(x_shape, in_c, out_c, k=3):
    """Representa un bloque Conv -> BN -> ReLU por sus dimensiones."""
    # x: (N, H, W, C_in)
    H, W, C = x_shape
    # Conv con padding='same'
    out_H, out_W = H, W
    return (out_H, out_W, out_c)


def bloque_residual(x_shape, out_c):
    """Bloque residual: y = x + Conv(x). Mantiene C si x.shape[-1] == out_c."""
    H, W, C = x_shape
    if C != out_c:
        # Proyeccion 1x1 para cambiar canales
        intermedio = (H, W, C)  # conv 1x1 cambia canales
        intermedio = (H, W, out_c)
    else:
        intermedio = (H, W, out_c)
    return intermedio


def receptive_field(capas):
    """Calcula receptive field de una secuencia de Conv k=3.
    Para k=3 con stride 1: RF += 2 por capa. Para stride 2: *= 2."""
    rf = 1
    stride_acum = 1
    for capa in capas:
        if capa == "conv3":
            rf += (3 - 1) * stride_acum
        elif capa == "conv1":
            rf += (1 - 1) * stride_acum
        elif capa == "pool2":
            stride_acum *= 2
    return rf


def cuente_params_conv(in_c, out_c, k=3):
    """Numero de parametros: (in_c * k * k) * out_c + out_c (bias)."""
    return in_c * k * k * out_c + out_c


def cuente_params_fc(in_f, out_f):
    return in_f * out_f + out_f


def total_params_lenet():
    """LeNet-5: 2 conv + 2 fc. Cuento approx."""
    # Conv1: 1->6, k=5
    c1 = cuente_params_conv(1, 6, 5)
    # Conv2: 6->16, k=5
    c2 = cuente_params_conv(6, 16, 5)
    # FC: 16*5*5 -> 120
    f1 = cuente_params_fc(16 * 5 * 5, 120)
    f2 = cuente_params_fc(120, 84)
    f3 = cuente_params_fc(84, 10)
    return {"conv1": c1, "conv2": c2, "fc1": f1, "fc2": f2, "fc3": f3, "total": c1 + c2 + f1 + f2 + f3}


def main() -> int:
    print("LeNet-5 params:")
    for k, v in total_params_lenet().items():
        print(f"  {k}: {v}")
    print(f"RF tras [conv3, conv3, pool2, conv3, conv3, pool2]: {receptive_field(['conv3', 'conv3', 'pool2', 'conv3', 'conv3', 'pool2'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())