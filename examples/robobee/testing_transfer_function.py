import numpy as np
from matplotlib import pyplot as plt

A = 151.4*0.0063
b_eq = 0.1959
m_eq = 3.82e-4
k_eq = 500.4331

def wGw(w):
    w = 2 * np.pi * w
    return (w * w * A * A)/((b_eq * w)**2 + (k_eq - m_eq * w * w)**2)

ws = np.linspace(0, 250, 200)
transfer = [wGw(w) for w in ws]

plt.figure()
plt.plot(ws, transfer)
plt.axvline(x=100, linestyle='--', color='b')
plt.axvline(x=180, linestyle='--', color='b')
plt.title("Transfer function term (w G(w))^2")
plt.show()
