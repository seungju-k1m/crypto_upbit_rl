from scipy import optimize
import time
from scipy.optimize.nonlin import NoConvergence

m = 0
def F(x):
    if x * -1 < 0:
        return -m + (x + 1) ** 0.5 -1 + 0.001 * x
    else:
        return-m +  (-x + 1) ** 0.5 - 1 - 0.001 * x


start = -3
end = 3
delta = 1e-3

xx = [start + i * delta for i in range(int((end - start)/delta))]

mz = time.time()
output = []
for m in xx:
    def F(x):
        if m * -1 < 0:
            # 양수
            return -m + (x + 1) ** 0.5 -1 + 0.001 * x
        else:
            return -m - ((-x + 1) ** 0.5 - 1) + 0.001 * x

    try:
        sol = optimize.broyden1(F, 0, f_tol=1e-6)
    except NoConvergence as e:
        sol = e.args[0]
    output.append(float(sol))

print(output)
print(time.time() - mz)