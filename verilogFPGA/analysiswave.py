from vcdvcd import VCDVCD

import numpy as np
import matplotlib.pyplot as plt


class WarpVCDVisualizer:
    def __init__(self, vcd_file):
        self.vcd = VCDVCD(vcd_file)
        self.signals = self.vcd.signals

    # --------------------------
    # RAW ACCESS
    # --------------------------
    def get(self, name):
        return self.vcd[name].tv

    def values(self, name):
        return [v for _, v in self.vcd[name].tv]

    # --------------------------
    # SWITCHING = ENERGY PROXY
    # --------------------------
    def switching(self, name):
        vals = self.values(name)
        return sum(vals[i] != vals[i - 1] for i in range(1, len(vals)))

    # --------------------------
    # ALIGN SIGNALS BY TIME
    # --------------------------
    def align(self, *names):
        """
        Convert signals into aligned time series.
        """
        series = [self.get(n) for n in names]

        all_times = sorted(set(t for s in series for t, _ in s))

        aligned = []
        for s in series:
            d = dict(s)
            aligned.append([d.get(t, np.nan) for t in all_times])

        return all_times, aligned

    # --------------------------
    # WARP VISUALIZATION
    # --------------------------
    def plot_execution(self, op_name, y_name):
        times, (op, y) = self.align(op_name, y_name)

        fig, ax = plt.subplots(3, 1, figsize=(10, 7))

        # --- OP (control)
        ax[0].step(times, op, where="post")
        ax[0].set_title("Control (op)")
        ax[0].set_ylabel("op")

        # --- OUTPUT (datapath)
        ax[1].step(times, y, where="post")
        ax[1].set_title("Output (y)")
        ax[1].set_ylabel("y")

        # --- ENERGY PROXY (switching intensity)
        energy = [0]
        for i in range(1, len(op)):
            energy.append(int(op[i] != op[i - 1]) + int(y[i] != y[i - 1]))

        ax[2].plot(times, energy)
        ax[2].set_title("Switching Activity (Energy Proxy)")
        ax[2].set_ylabel("switches")

        plt.tight_layout()
        plt.show()

    # --------------------------
    # UTILIZATION METRIC
    # --------------------------
    def utilization(self, name):
        vals = self.values(name)
        active = sum(v != 0 for v in vals)
        return active / len(vals)

viz = WarpVCDVisualizer("wave.vcd")

print("OP switching:", viz.switching("TOP.op[1:0]"))
print("Y switching:", viz.switching("TOP.y[7:0]"))

viz.plot_execution("TOP.op[1:0]", "TOP.y[7:0]")
