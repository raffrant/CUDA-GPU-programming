from vcdvcd import VCDVCD
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


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

    def times(self, name):
        return [t for t, _ in self.vcd[name].tv]

    # --------------------------
    # HELPERS
    # --------------------------
    @staticmethod
    def _to_int(v):
        if v in (None, 'x', 'z', 'X', 'Z'):
            return None
        if isinstance(v, (int, float, np.integer, np.floating)):
            return int(v)
        if isinstance(v, str):
            s = v.strip()
            if set(s) <= {'0', '1'}:
                return int(s, 2)
            try:
                return int(s, 0)
            except Exception:
                return None
        try:
            return int(v)
        except Exception:
            return None

    def _state_of(self, v):
        iv = self._to_int(v)
        if iv is None:
            return 'unknown'
        if iv == 0:
            return 'idle'
        if iv in (1, 2):
            return 'compute'
        if iv in (3, 4):
            return 'memory'
        return 'other'

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
        series = [self.get(n) for n in names]
        all_times = sorted(set(t for s in series for t, _ in s))
        aligned = []
        for s in series:
            d = dict(s)
            aligned.append([d.get(t, np.nan) for t in all_times])
        return all_times, aligned

    def align_with_prev(self, *names):
        times, aligned = self.align(*names)
        filled = []
        for arr in aligned:
            out = []
            last = np.nan
            for v in arr:
                if not (isinstance(v, float) and np.isnan(v)):
                    last = v
                out.append(last)
            filled.append(out)
        return times, filled

    # --------------------------
    # METRICS
    # --------------------------
    def utilization(self, name):
        vals = self.values(name)
        active = sum(v not in (0, '0', 'x', 'z') for v in vals)
        return active / len(vals) if vals else 0.0

    def energy_proxy(self, *names, op_costs=None, switching_weight=0.5):
        op_costs = op_costs or {}
        summary = {}
        total = 0.0
        for name in names:
            vals = self.values(name)
            sw = self.switching(name)
            base = 0.0
            for v in vals:
                iv = self._to_int(v)
                if iv is None:
                    base += 0.0
                else:
                    base += op_costs.get(iv, 1.0 if iv != 0 else 0.0)
            energy = base + switching_weight * sw
            summary[name] = {'base_cost': base, 'switches': sw, 'energy': energy}
            total += energy
        return summary, total

    def state_fractions(self, name):
        vals = self.values(name)
        n = len(vals) or 1
        counts = {'idle': 0, 'compute': 0, 'memory': 0, 'other': 0, 'unknown': 0}
        for v in vals:
            counts[self._state_of(v)] += 1
        return {k: counts[k] / n for k in counts}

    def segment_metrics(self, name):
        vals = self.values(name)
        states = [self._state_of(v) for v in vals]
        segments = []
        if not states:
            return segments
        cur = states[0]
        length = 1
        for s in states[1:]:
            if s == cur:
                length += 1
            else:
                segments.append((cur, length))
                cur = s
                length = 1
        segments.append((cur, length))
        return segments

    # --------------------------
    # PLOTTING
    # --------------------------
    def plot_execution(self, op_name, y_name, out_file=None):
        times, (op, y) = self.align_with_prev(op_name, y_name)
        op_num = np.array([np.nan if v is np.nan else (self._to_int(v) if self._to_int(v) is not None else np.nan) for v in op], dtype=float)
        y_num = np.array([np.nan if v is np.nan else (self._to_int(v) if self._to_int(v) is not None else np.nan) for v in y], dtype=float)
        energy = [0]
        for i in range(1, len(op_num)):
            energy.append(int(op_num[i] != op_num[i - 1]) + int(y_num[i] != y_num[i - 1]))

        fig, ax = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
        ax[0].step(times, op_num, where='post')
        ax[0].set_title(f'Control ({op_name})')
        ax[0].set_ylabel('op')

        ax[1].step(times, y_num, where='post')
        ax[1].set_title(f'Output ({y_name})')
        ax[1].set_ylabel('y')

        ax[2].plot(times, energy)
        ax[2].set_title('Switching Activity (Energy Proxy)')
        ax[2].set_ylabel('switches')
        ax[2].set_xlabel('time')

        plt.tight_layout()
        if out_file:
            plt.savefig(out_file, dpi=180, bbox_inches='tight')
        plt.show()

    def plot_breakdown(self, signal_name, out_file=None):
        times, (sig,) = self.align_with_prev(signal_name)
        vals = [self._to_int(v) for v in sig]
        states = [self._state_of(v) for v in vals]
        n = len(states)
        comp = np.array([1 if s == 'compute' else 0 for s in states], dtype=float)
        mem = np.array([1 if s == 'memory' else 0 for s in states], dtype=float)
        idle = np.array([1 if s == 'idle' else 0 for s in states], dtype=float)
        unk = np.array([1 if s == 'unknown' else 0 for s in states], dtype=float)
        sw = np.array([0] + [int(vals[i] != vals[i - 1]) for i in range(1, n)], dtype=float)

        fig, ax = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
        ax[0].plot(times, comp, label='compute', color='tab:blue')
        ax[0].plot(times, mem, label='memory', color='tab:orange')
        ax[0].plot(times, idle, label='idle', color='tab:green')
        ax[0].plot(times, unk, label='unknown', color='tab:red', alpha=0.7)
        ax[0].set_title(f'State Breakdown ({signal_name})')
        ax[0].set_ylabel('state')
        ax[0].legend(loc='upper right', ncols=4, fontsize=9)

        ax[1].plot(times, sw, color='black')
        ax[1].set_title('Switch Activity per Step')
        ax[1].set_ylabel('switch')

        # energy model
        op_costs = {0: 0.0, 1: 1.0, 2: 2.0, 3: 3.5, 4: 3.5}
        energy = np.array([(op_costs.get(v, 1.0) if v is not None else 0.0) for v in vals], dtype=float) + 0.5 * sw
        ax[2].plot(times, energy, color='tab:purple')
        ax[2].set_title('Energy Model per Step')
        ax[2].set_ylabel('energy')

        # running totals / fractions
        cum_energy = np.cumsum(np.nan_to_num(energy))
        ax[3].plot(times, cum_energy, color='tab:brown')
        ax[3].set_title('Cumulative Energy')
        ax[3].set_ylabel('cum energy')
        ax[3].set_xlabel('time')

        plt.tight_layout()
        if out_file:
            plt.savefig(out_file, dpi=180, bbox_inches='tight')
        plt.show()

    def plot_memory_compute_heatmap(self, signal_names, out_file=None, title='Memory vs Compute Heatmap'):
        times, aligned = self.align_with_prev(*signal_names)
        arr = np.array([[self._state_of(v) for v in row] for row in aligned], dtype=object)
        mapping = {'idle': 0, 'compute': 1, 'memory': 2, 'other': 3, 'unknown': 4}
        num = np.vectorize(mapping.get)(arr)

        fig, ax = plt.subplots(figsize=(12, max(4, 0.35 * len(signal_names))))
        im = ax.imshow(num, aspect='auto', interpolation='nearest', cmap='viridis')
        ax.set_yticks(range(len(signal_names)))
        ax.set_yticklabels(signal_names)
        ax.set_xlabel('time index')
        ax.set_title(title)
        cbar = plt.colorbar(im, ax=ax, ticks=list(mapping.values()))
        cbar.ax.set_yticklabels(list(mapping.keys()))
        plt.tight_layout()
        if out_file:
            plt.savefig(out_file, dpi=180, bbox_inches='tight')
        plt.show()


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('vcd_file', nargs='?', default='wave.vcd')
    p.add_argument('--op', default='TOP.op[1:0]')
    p.add_argument('--y', default='TOP.y[7:0]')
    p.add_argument('--out-prefix', default='warp')
    args = p.parse_args()

    viz = WarpVCDVisualizer(args.vcd_file)
    print('OP switching:', viz.switching(args.op))
    print('Y switching:', viz.switching(args.y))
    print('Utilization OP:', viz.utilization(args.op))
    print('Utilization Y :', viz.utilization(args.y))

    op_costs = {0: 0, 1: 1, 2: 2, 3: 3.5, 4: 3.5}
    summary, total = viz.energy_proxy(args.op, args.y, op_costs=op_costs, switching_weight=0.5)
    print('Energy summary:', summary)
    print('Total energy   :', total)
    print('State fractions OP:', viz.state_fractions(args.op))
    print('State fractions Y :', viz.state_fractions(args.y))

    viz.plot_execution(args.op, args.y, out_file=f'{args.out_prefix}_execution.png')
    viz.plot_breakdown(args.op, out_file=f'{args.out_prefix}_breakdown_op.png')
    viz.plot_breakdown(args.y, out_file=f'{args.out_prefix}_breakdown_y.png')
    viz.plot_memory_compute_heatmap([args.op, args.y], out_file=f'{args.out_prefix}_mc_heatmap.png', title='Memory vs Compute Heatmap')
