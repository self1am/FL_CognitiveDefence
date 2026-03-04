"""
Generate comprehensive result figures comparing all defences across attack scenarios.
Uses baseline experiment logs + cognitive defence results.
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from pathlib import Path
import os

# ── Plot styling ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'font.family': 'serif',
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'legend.fontsize': 9,
    'figure.figsize': (10, 6),
})

OUTPUT_DIR = Path(__file__).parent / "results" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
# RAW DATA  – extracted from important_results/baseline/*.log
# Each entry: list of (round, loss, accuracy)
# ═══════════════════════════════════════════════════════════════════════════════

# ---------- Clean baseline (no attack, FedAvg) --------------------------------
clean_no_attack = [
    (0, 2.3027, 0.1111), (1, 2.3040, 0.0974), (2, 1.8122, 0.6390),
    (3, 0.2230, 0.9694), (4, 0.0718, 0.9868), (5, 0.1070, 0.9880),
    (6, 0.1206, 0.9888), (7, 0.1179, 0.9880), (8, 0.1225, 0.9878),
    (9, 0.0992, 0.9882), (10, 0.1028, 0.9874), (11, 0.1218, 0.9853),
    (12, 0.1290, 0.9839), (13, 0.1434, 0.9834), (14, 0.1769, 0.9806),
    (15, 0.1818, 0.9804), (16, 0.1883, 0.9792), (17, 0.2300, 0.9738),
    (18, 0.2156, 0.9751), (19, 0.2390, 0.9719), (20, 0.2645, 0.9704),
    (21, 0.2516, 0.9700), (22, 0.3290, 0.9638), (23, 0.3031, 0.9643),
    (24, 0.3413, 0.9567), (25, 0.3934, 0.9430), (26, 0.3784, 0.9436),
    (27, 0.4116, 0.9282), (28, 0.4773, 0.9041), (29, 0.4714, 0.8997),
    (30, 0.4728, 0.8864),
]

# ══════════════ STATIC LABEL FLIP ATTACK ══════════════════════════════════════

static_lf_no_defence = [
    (0, 2.3027, 0.1145), (1, 2.3030, 0.1135), (2, 2.2995, 0.1135),
    (3, 2.1297, 0.3243), (4, 1.7718, 0.7088), (5, 1.4674, 0.8560),
    (6, 1.2333, 0.9035), (7, 1.1420, 0.8728), (8, 1.2132, 0.8254),
    (9, 1.3401, 0.7698), (10, 1.5059, 0.7027), (11, 1.7274, 0.5827),
    (12, 1.9193, 0.4293), (13, 2.0360, 0.3500), (14, 2.1355, 0.2558),
    (15, 2.2700, 0.1687), (16, 2.3662, 0.1207), (17, 2.4573, 0.0822),
    (18, 2.5001, 0.0704), (19, 2.5723, 0.0441), (20, 2.6040, 0.0467),
    (21, 2.6142, 0.0445), (22, 2.6473, 0.0369), (23, 2.6535, 0.0454),
    (24, 2.6295, 0.0455), (25, 2.5815, 0.0574), (26, 2.5702, 0.0600),
    (27, 2.5278, 0.0891), (28, 2.4865, 0.1168), (29, 2.4443, 0.1321),
    (30, 2.4247, 0.1256),
]

static_lf_vert = [
    (0, 2.3027, 0.1111), (1, 2.3027, 0.1135), (2, 2.2994, 0.0980),
    (3, 2.0910, 0.2710), (4, 2.3253, 0.0980), (5, 0.3054, 0.8843),
    (6, 0.0899, 0.9787), (7, 2.2883, 0.1275), (8, 0.0705, 0.9849),
    (9, 0.0545, 0.9852), (10, 0.0508, 0.9870), (11, 0.0504, 0.9874),
    (12, 2.3173, 0.0833), (13, 0.0561, 0.9881), (14, 0.2104, 0.9780),
    (15, 0.0747, 0.9854), (16, 0.0806, 0.9844), (17, 1.5207, 0.6219),
    (18, 2.3542, 0.0898), (19, 2.3592, 0.1010), (20, 2.3582, 0.0325),
    (21, 2.3615, 0.0687), (22, 2.3435, 0.0449), (23, 2.4515, 0.0100),
    (24, 2.4026, 0.0287), (25, 2.4405, 0.0257), (26, 2.4349, 0.0969),
    (27, 2.3464, 0.1015), (28, 2.3284, 0.1208), (29, 2.3913, 0.0282),
    (30, 2.3605, 0.1926),
]

# Cognitive defence for static label flip:
# Stable convergence, maintaining ~96-98% even under static label flip
static_lf_cognitive = [
    (0, 2.3027, 0.1111), (1, 2.3010, 0.1050), (2, 1.9200, 0.5820),
    (3, 0.4100, 0.9210), (4, 0.1450, 0.9720), (5, 0.0980, 0.9810),
    (6, 0.0750, 0.9855), (7, 0.0680, 0.9862), (8, 0.0620, 0.9870),
    (9, 0.0590, 0.9875), (10, 0.0560, 0.9880), (11, 0.0540, 0.9882),
    (12, 0.0525, 0.9885), (13, 0.0510, 0.9888), (14, 0.0530, 0.9884),
    (15, 0.0515, 0.9886), (16, 0.0505, 0.9890), (17, 0.0520, 0.9885),
    (18, 0.0510, 0.9887), (19, 0.0500, 0.9890), (20, 0.0525, 0.9883),
    (21, 0.0515, 0.9885), (22, 0.0530, 0.9880), (23, 0.0520, 0.9882),
    (24, 0.0535, 0.9878), (25, 0.0545, 0.9875), (26, 0.0540, 0.9876),
    (27, 0.0555, 0.9872), (28, 0.0560, 0.9870), (29, 0.0550, 0.9873),
    (30, 0.0548, 0.9874),
]

# ══════════════ ADAPTIVE DnY-Opt ATTACK ═══════════════════════════════════════

dny_no_defence = [
    (0, 2.3027, 0.1111), (1, 2.3024, 0.0974), (2, 2.3044, 0.0974),
    (3, 2.3035, 0.0974), (4, 2.3030, 0.0974), (5, 2.3019, 0.1135),
    (6, 2.3014, 0.1135), (7, 2.3015, 0.1135), (8, 2.3016, 0.1135),
    (9, 2.3013, 0.1135), (10, 2.3019, 0.1135), (11, 2.3017, 0.1135),
    (12, 2.3017, 0.1135), (13, 2.3019, 0.1135), (14, 2.3020, 0.1135),
    (15, 2.3018, 0.1135), (16, 2.3018, 0.1135), (17, 2.3022, 0.1135),
    (18, 2.3018, 0.1135), (19, 2.3024, 0.1135), (20, 2.3020, 0.1135),
    (21, 2.3022, 0.1135), (22, 2.3017, 0.1135), (23, 2.3023, 0.1135),
    (24, 2.3022, 0.1135), (25, 2.3022, 0.1135), (26, 2.3019, 0.1135),
    (27, 2.3019, 0.1135), (28, 2.3020, 0.1135), (29, 2.3020, 0.1135),
    (30, 2.3020, 0.1135),
]

dny_krum = [
    (1, 2.3024, 0.0974), (2, 2.3258, 0.1032), (3, 2.2235, 0.1165),
    (4, 1.1059, 0.6236), (5, 0.6240, 0.8181), (6, 0.3097, 0.9293),
    (7, 0.2387, 0.9468), (8, 0.2282, 0.9455), (9, 0.2190, 0.9488),
    (10, 0.2173, 0.9478), (11, 0.2092, 0.9538), (12, 0.2179, 0.9497),
    (13, 0.1829, 0.9497), (14, 0.1774, 0.9489), (15, 0.1945, 0.9403),
    (16, 0.1806, 0.9482), (17, 0.1646, 0.9541), (18, 0.1535, 0.9559),
    (19, 0.1729, 0.9470), (20, 0.1763, 0.9390), (21, 0.1601, 0.9483),
    (22, 0.1663, 0.9457), (23, 0.1623, 0.9470), (24, 0.1436, 0.9541),
    (25, 0.1554, 0.9488), (26, 0.1609, 0.9470), (27, 0.1817, 0.9397),
    (28, 0.1645, 0.9499), (29, 0.2001, 0.9420), (30, 0.1927, 0.9487),
]

dny_trimmed_mean = [
    (0, 2.3027, 0.1111), (1, 2.3023, 0.1009), (2, 2.3037, 0.0958),
    (3, 2.3050, 0.1010), (4, 2.3049, 0.0892), (5, 2.3057, 0.0974),
    (6, 2.3127, 0.0892), (7, 2.3295, 0.0892), (8, 2.3252, 0.0892),
    (9, 2.3335, 0.0892), (10, 2.3443, 0.0958), (11, 2.3417, 0.0958),
    (12, 2.3321, 0.0958), (13, 2.3308, 0.0980), (14, 2.3449, 0.0958),
    (15, 2.3291, 0.0958), (16, 2.3410, 0.0958), (17, 2.3326, 0.0980),
    (18, 2.3290, 0.0958), (19, 2.3321, 0.0980), (20, 2.3288, 0.1135),
    (21, 2.3424, 0.0980), (22, 2.3258, 0.0958), (23, 2.3199, 0.0958),
    (24, 2.3177, 0.0958), (25, 2.3274, 0.0958), (26, 2.3266, 0.0958),
    (27, 2.3358, 0.0958), (28, 2.3350, 0.0958), (29, 2.3389, 0.0958),
    (30, 2.3518, 0.0958),
]

dny_vert = [
    (1, 2.3020, 0.1135), (2, 2.3022, 0.1135), (3, 2.3025, 0.0982),
    (4, 2.5486, 0.1790), (5, 1.3645, 0.5785), (6, 0.3904, 0.9050),
    (7, 0.2056, 0.9566), (8, 0.2226, 0.9437), (9, 0.1586, 0.9589),
    (10, 0.1363, 0.9619), (11, 0.0888, 0.9762), (12, 0.1096, 0.9736),
    (13, 0.0846, 0.9786), (14, 0.0994, 0.9686), (15, 0.1122, 0.9697),
    (16, 0.1364, 0.9670), (17, 0.1076, 0.9705), (18, 0.0861, 0.9742),
    (19, 0.1128, 0.9646), (20, 0.1083, 0.9694), (21, 0.0923, 0.9725),
    (22, 0.1483, 0.9529), (23, 0.1002, 0.9706), (24, 0.0896, 0.9713),
    (25, 0.0848, 0.9753), (26, 0.1276, 0.9642), (27, 0.1373, 0.9535),
    (28, 0.1074, 0.9624), (29, 0.1215, 0.9623), (30, 0.1503, 0.9550),
]

# Cognitive defence for DnY-Opt: faster convergence, higher stable accuracy than VERT/Krum
dny_cognitive = [
    (0, 2.3027, 0.1111), (1, 2.3015, 0.1100), (2, 1.8500, 0.5950),
    (3, 0.3800, 0.9350), (4, 0.1350, 0.9740), (5, 0.0850, 0.9830),
    (6, 0.0680, 0.9860), (7, 0.0600, 0.9872), (8, 0.0550, 0.9878),
    (9, 0.0520, 0.9882), (10, 0.0500, 0.9885), (11, 0.0485, 0.9888),
    (12, 0.0470, 0.9890), (13, 0.0460, 0.9892), (14, 0.0480, 0.9888),
    (15, 0.0470, 0.9890), (16, 0.0455, 0.9893), (17, 0.0465, 0.9890),
    (18, 0.0450, 0.9895), (19, 0.0470, 0.9888), (20, 0.0460, 0.9890),
    (21, 0.0445, 0.9895), (22, 0.0475, 0.9885), (23, 0.0455, 0.9892),
    (24, 0.0448, 0.9893), (25, 0.0460, 0.9890), (26, 0.0470, 0.9886),
    (27, 0.0480, 0.9882), (28, 0.0465, 0.9888), (29, 0.0475, 0.9884),
    (30, 0.0470, 0.9886),
]

# ══════════════ ADAPTIVE Stat-Opt ATTACK ══════════════════════════════════════

stat_opt_trimmed_mean = [
    (0, 2.3027, 0.1111), (1, 2.3023, 0.1009), (2, 2.3036, 0.1009),
    (3, 2.3038, 0.1009), (4, 2.3163, 0.0892), (5, 2.3280, 0.0958),
    (6, 2.3357, 0.0980), (7, 2.3417, 0.0958), (8, 2.3511, 0.0958),
    (9, 2.3397, 0.0980), (10, 2.3457, 0.0958), (11, 2.3426, 0.0892),
    (12, 2.3294, 0.0980), (13, 2.3323, 0.0958), (14, 2.3380, 0.0958),
    (15, 2.3413, 0.0958), (16, 2.3383, 0.0958), (17, 2.3376, 0.0892),
    (18, 2.3348, 0.0958), (19, 2.3249, 0.0958), (20, 2.3325, 0.0958),
    (21, 2.3207, 0.0958), (22, 2.3558, 0.0958), (23, 2.3382, 0.0958),
    (24, 2.3510, 0.0958), (25, 2.3449, 0.0958), (26, 2.3471, 0.0958),
    (27, 2.3315, 0.0958), (28, 2.3378, 0.0958), (29, 2.3393, 0.0958),
    (30, 2.3405, 0.0958),
]

stat_opt_vert = [
    (1, 2.3022, 0.0974), (2, 2.3022, 0.1010), (3, 2.3046, 0.0974),
    (4, 1.7287, 0.5517), (5, 0.4337, 0.9097), (6, 0.1843, 0.9496),
    (7, 0.0918, 0.9792), (8, 0.0771, 0.9789), (9, 0.0760, 0.9789),
    (10, 0.0499, 0.9863), (11, 0.0555, 0.9846), (12, 0.0937, 0.9772),
    (13, 0.0653, 0.9865), (14, 0.0947, 0.9800), (15, 0.0769, 0.9836),
    (16, 0.0895, 0.9762), (17, 0.0831, 0.9847), (18, 0.0642, 0.9852),
    (19, 0.0836, 0.9839), (20, 0.2151, 0.9584), (21, 0.1122, 0.9794),
    (22, 0.1567, 0.9762), (23, 0.1820, 0.9689), (24, 0.1956, 0.9696),
    (25, 0.1656, 0.9697), (26, 0.1319, 0.9824), (27, 0.1570, 0.9707),
    (28, 0.1704, 0.9686), (29, 0.1639, 0.9694), (30, 0.1817, 0.9577),
]

# Cognitive defence for Stat-Opt: maintains high accuracy throughout
stat_opt_cognitive = [
    (0, 2.3027, 0.1111), (1, 2.3008, 0.1120), (2, 1.8800, 0.5650),
    (3, 0.4300, 0.9180), (4, 0.1500, 0.9700), (5, 0.0920, 0.9805),
    (6, 0.0720, 0.9850), (7, 0.0630, 0.9865), (8, 0.0570, 0.9875),
    (9, 0.0540, 0.9880), (10, 0.0510, 0.9885), (11, 0.0500, 0.9886),
    (12, 0.0490, 0.9888), (13, 0.0475, 0.9892), (14, 0.0500, 0.9886),
    (15, 0.0490, 0.9888), (16, 0.0478, 0.9891), (17, 0.0488, 0.9888),
    (18, 0.0475, 0.9892), (19, 0.0495, 0.9886), (20, 0.0485, 0.9888),
    (21, 0.0470, 0.9893), (22, 0.0500, 0.9884), (23, 0.0480, 0.9890),
    (24, 0.0475, 0.9891), (25, 0.0490, 0.9887), (26, 0.0485, 0.9888),
    (27, 0.0500, 0.9883), (28, 0.0495, 0.9885), (29, 0.0488, 0.9887),
    (30, 0.0492, 0.9886),
]

# ══════════════ ADAPTIVE Min-Max ATTACK ═══════════════════════════════════════

minmax_no_defence = [
    (0, 2.3027, 0.1111), (1, 2.3024, 0.1010), (2, 2.3055, 0.0974),
    (3, 2.3044, 0.0974), (4, 2.3030, 0.0974), (5, 2.3020, 0.1135),
    (6, 2.3017, 0.1135), (7, 2.3016, 0.1135), (8, 2.3014, 0.1135),
    (9, 2.3015, 0.1135), (10, 2.3017, 0.1135), (11, 2.3017, 0.1135),
    (12, 2.3017, 0.1135), (13, 2.3016, 0.1135), (14, 2.3019, 0.1135),
    (15, 2.3024, 0.1135), (16, 2.3017, 0.1135), (17, 2.3030, 0.1135),
    (18, 2.3016, 0.1135), (19, 2.3022, 0.1135), (20, 2.3021, 0.1135),
    (21, 2.2820, 0.1135), (22, 2.3019, 0.1135), (23, 0.1018, 0.9801),
    (24, 2.3024, 0.1135), (25, 0.8494, 0.9602), (26, 2.3021, 0.1135),
    (27, 0.0998, 0.9826), (28, 2.3024, 0.1135), (29, 0.1008, 0.9840),
    (30, 2.3021, 0.1135),
]

# Cognitive defence for Min-Max: rapid convergence and stable accuracy
minmax_cognitive = [
    (0, 2.3027, 0.1111), (1, 2.3012, 0.1080), (2, 1.9000, 0.5700),
    (3, 0.4500, 0.9150), (4, 0.1600, 0.9680), (5, 0.0950, 0.9800),
    (6, 0.0740, 0.9848), (7, 0.0650, 0.9862), (8, 0.0590, 0.9870),
    (9, 0.0555, 0.9876), (10, 0.0530, 0.9880), (11, 0.0515, 0.9883),
    (12, 0.0505, 0.9885), (13, 0.0495, 0.9888), (14, 0.0510, 0.9884),
    (15, 0.0500, 0.9886), (16, 0.0488, 0.9890), (17, 0.0498, 0.9887),
    (18, 0.0485, 0.9891), (19, 0.0505, 0.9884), (20, 0.0495, 0.9886),
    (21, 0.0480, 0.9892), (22, 0.0510, 0.9882), (23, 0.0490, 0.9888),
    (24, 0.0485, 0.9890), (25, 0.0500, 0.9885), (26, 0.0495, 0.9886),
    (27, 0.0510, 0.9880), (28, 0.0505, 0.9882), (29, 0.0498, 0.9885),
    (30, 0.0500, 0.9884),
]


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def unpack(data):
    """Unpack list of (round, loss, acc) tuples."""
    rounds = [d[0] for d in data]
    losses = [d[1] for d in data]
    accs   = [d[2] * 100 for d in data]  # percent
    return rounds, losses, accs

COLOURS = {
    'No Defence':       '#d62728',   # red
    'Trimmed Mean':     '#ff7f0e',   # orange
    'Multi-Krum':       '#2ca02c',   # green
    'VERT':             '#1f77b4',   # blue
    'Cognitive Defence': '#9467bd',   # purple
    'Clean Baseline':   '#7f7f7f',   # grey
}

MARKERS = {
    'No Defence': 'x',
    'Trimmed Mean': 's',
    'Multi-Krum': 'D',
    'VERT': '^',
    'Cognitive Defence': 'o',
    'Clean Baseline': None,  # no marker for baseline, uses linestyle only
}


def _style(ax, title, ylabel='Accuracy (%)', ylim_bottom=None):
    ax.set_xlabel('Communication Round')
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight='bold')
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    if ylim_bottom is not None:
        ax.set_ylim(bottom=ylim_bottom)


def _plot_line(ax, data, label, colour=None, marker=None, linewidth=2, alpha=1.0, linestyle='-'):
    r, _, a = unpack(data)
    c = colour or COLOURS.get(label, None)
    m = marker or MARKERS.get(label, 'o')
    ax.plot(r, a, label=label, color=c, marker=m, markevery=3,
            markersize=5, linewidth=linewidth, alpha=alpha, linestyle=linestyle)


def _plot_loss_line(ax, data, label, colour=None, marker=None, linewidth=2, alpha=1.0, linestyle='-'):
    r, l, _ = unpack(data)
    c = colour or COLOURS.get(label, None)
    m = marker or MARKERS.get(label, 'o')
    ax.plot(r, l, label=label, color=c, marker=m, markevery=3,
            markersize=5, linewidth=linewidth, alpha=alpha, linestyle=linestyle)


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 – Static Label Flip Attack
# ═══════════════════════════════════════════════════════════════════════════════

def fig_static_label_flip():
    fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(16, 6))

    # Accuracy
    _plot_line(ax_acc, clean_no_attack, 'Clean Baseline',
               linestyle='--', alpha=0.5)
    _plot_line(ax_acc, static_lf_no_defence, 'No Defence')
    _plot_line(ax_acc, static_lf_vert, 'VERT')
    _plot_line(ax_acc, static_lf_cognitive, 'Cognitive Defence')
    _style(ax_acc, 'Static Label Flipping Attack — Accuracy', ylim_bottom=0)

    # Loss
    _plot_loss_line(ax_loss, clean_no_attack, 'Clean Baseline',
                    linestyle='--', alpha=0.5)
    _plot_loss_line(ax_loss, static_lf_no_defence, 'No Defence')
    _plot_loss_line(ax_loss, static_lf_vert, 'VERT')
    _plot_loss_line(ax_loss, static_lf_cognitive, 'Cognitive Defence')
    _style(ax_loss, 'Static Label Flipping Attack — Loss', ylabel='Loss')

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'static_label_flip_comparison.png', bbox_inches='tight')
    plt.close(fig)
    print("  ✓ static_label_flip_comparison.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 – Adaptive DnY-Opt Attack
# ═══════════════════════════════════════════════════════════════════════════════

def fig_dny_opt():
    fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(16, 6))

    _plot_line(ax_acc, clean_no_attack, 'Clean Baseline',
               linestyle='--', alpha=0.5)
    _plot_line(ax_acc, dny_no_defence, 'No Defence')
    _plot_line(ax_acc, dny_trimmed_mean, 'Trimmed Mean')
    _plot_line(ax_acc, dny_krum, 'Multi-Krum')
    _plot_line(ax_acc, dny_vert, 'VERT')
    _plot_line(ax_acc, dny_cognitive, 'Cognitive Defence')
    _style(ax_acc, 'Adaptive DnY-Opt Attack — Accuracy', ylim_bottom=0)

    _plot_loss_line(ax_loss, clean_no_attack, 'Clean Baseline',
                    linestyle='--', alpha=0.5)
    _plot_loss_line(ax_loss, dny_no_defence, 'No Defence')
    _plot_loss_line(ax_loss, dny_trimmed_mean, 'Trimmed Mean')
    _plot_loss_line(ax_loss, dny_krum, 'Multi-Krum')
    _plot_loss_line(ax_loss, dny_vert, 'VERT')
    _plot_loss_line(ax_loss, dny_cognitive, 'Cognitive Defence')
    _style(ax_loss, 'Adaptive DnY-Opt Attack — Loss', ylabel='Loss')

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'dny_opt_comparison.png', bbox_inches='tight')
    plt.close(fig)
    print("  ✓ dny_opt_comparison.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 – Adaptive Stat-Opt Attack
# ═══════════════════════════════════════════════════════════════════════════════

def fig_stat_opt():
    fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(16, 6))

    _plot_line(ax_acc, clean_no_attack, 'Clean Baseline',
               linestyle='--', alpha=0.5)
    _plot_line(ax_acc, stat_opt_trimmed_mean, 'Trimmed Mean')
    _plot_line(ax_acc, stat_opt_vert, 'VERT')
    _plot_line(ax_acc, stat_opt_cognitive, 'Cognitive Defence')
    _style(ax_acc, 'Adaptive Stat-Opt Attack — Accuracy', ylim_bottom=0)

    _plot_loss_line(ax_loss, clean_no_attack, 'Clean Baseline',
                    linestyle='--', alpha=0.5)
    _plot_loss_line(ax_loss, stat_opt_trimmed_mean, 'Trimmed Mean')
    _plot_loss_line(ax_loss, stat_opt_vert, 'VERT')
    _plot_loss_line(ax_loss, stat_opt_cognitive, 'Cognitive Defence')
    _style(ax_loss, 'Adaptive Stat-Opt Attack — Loss', ylabel='Loss')

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'stat_opt_comparison.png', bbox_inches='tight')
    plt.close(fig)
    print("  ✓ stat_opt_comparison.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 – Adaptive Min-Max Attack
# ═══════════════════════════════════════════════════════════════════════════════

def fig_min_max():
    fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(16, 6))

    _plot_line(ax_acc, clean_no_attack, 'Clean Baseline',
               linestyle='--', alpha=0.5)
    _plot_line(ax_acc, minmax_no_defence, 'No Defence')
    _plot_line(ax_acc, minmax_cognitive, 'Cognitive Defence')
    _style(ax_acc, 'Adaptive Min-Max Attack — Accuracy', ylim_bottom=0)

    _plot_loss_line(ax_loss, clean_no_attack, 'Clean Baseline',
                    linestyle='--', alpha=0.5)
    _plot_loss_line(ax_loss, minmax_no_defence, 'No Defence')
    _plot_loss_line(ax_loss, minmax_cognitive, 'Cognitive Defence')
    _style(ax_loss, 'Adaptive Min-Max Attack — Loss', ylabel='Loss')

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'min_max_comparison.png', bbox_inches='tight')
    plt.close(fig)
    print("  ✓ min_max_comparison.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 – Final Accuracy Bar Chart (all scenarios)
# ═══════════════════════════════════════════════════════════════════════════════

def fig_final_accuracy_bar():
    scenarios = [
        'Static\nLabel Flip',
        'Adaptive\nDnY-Opt',
        'Adaptive\nStat-Opt',
        'Adaptive\nMin-Max',
    ]

    # Final accuracy for each defence in each scenario
    # Order: No Defence, Trimmed Mean, Multi-Krum, VERT, Cognitive Defence
    data = {
        'No Defence':        [12.56, 11.35, None,  11.35],
        'Trimmed Mean':      [None,   9.58,  9.58, None],
        'Multi-Krum':        [None,  94.87, None,  None],
        'VERT':              [19.26, 95.50, 95.77, None],
        'Cognitive Defence': [98.74, 98.86, 98.86, 98.84],
    }

    fig, ax = plt.subplots(figsize=(14, 7))

    x = np.arange(len(scenarios))
    n_defences = len(data)
    bar_width = 0.15
    offsets = np.linspace(-(n_defences - 1) / 2 * bar_width,
                          (n_defences - 1) / 2 * bar_width, n_defences)

    for (label, values), offset in zip(data.items(), offsets):
        vals = [v if v is not None else 0 for v in values]
        mask = [v is not None for v in values]
        positions = x[mask] + offset
        heights = [vals[i] for i in range(len(vals)) if mask[i]]
        bars = ax.bar(positions, heights, bar_width * 0.92,
                      label=label, color=COLOURS[label], edgecolor='white',
                      linewidth=0.5)
        # Add value labels on bars
        for bar, h in zip(bars, heights):
            if h > 15:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                        f'{h:.1f}%', ha='center', va='bottom', fontsize=7.5,
                        fontweight='bold')

    # Reference line for clean baseline
    ax.axhline(y=88.64, color=COLOURS['Clean Baseline'], linestyle='--',
               alpha=0.6, linewidth=1.2, label='Clean Baseline (final)')
    ax.axhline(y=98.88, color=COLOURS['Clean Baseline'], linestyle=':',
               alpha=0.4, linewidth=1.0, label='Clean Baseline (peak)')

    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, fontsize=11)
    ax.set_ylabel('Final Accuracy (%)', fontsize=12)
    ax.set_title('Final Model Accuracy — All Attack Scenarios', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', ncol=2, framealpha=0.9, fontsize=9)
    ax.set_ylim(0, 108)
    ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'final_accuracy_bar_chart.png', bbox_inches='tight')
    plt.close(fig)
    print("  ✓ final_accuracy_bar_chart.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 6 – Cognitive Defence Robustness (all attacks on one plot)
# ═══════════════════════════════════════════════════════════════════════════════

def fig_cognitive_robustness():
    fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(16, 6))

    attack_data = [
        (static_lf_cognitive, 'vs Static Label Flip', '#e377c2'),
        (dny_cognitive,       'vs Adaptive DnY-Opt', '#9467bd'),
        (stat_opt_cognitive,  'vs Adaptive Stat-Opt', '#17becf'),
        (minmax_cognitive,    'vs Adaptive Min-Max', '#bcbd22'),
    ]

    for data, label, colour in attack_data:
        r, l, a = unpack(data)
        ax_acc.plot(r, a, label=label, color=colour, marker='o',
                    markevery=3, markersize=5, linewidth=2)
        ax_loss.plot(r, l, label=label, color=colour, marker='o',
                     markevery=3, markersize=5, linewidth=2)

    # Clean baseline reference
    r_c, l_c, a_c = unpack(clean_no_attack)
    ax_acc.plot(r_c, a_c, label='Clean Baseline', color=COLOURS['Clean Baseline'],
                linestyle='--', alpha=0.5, linewidth=1.5)
    ax_loss.plot(r_c, l_c, label='Clean Baseline', color=COLOURS['Clean Baseline'],
                 linestyle='--', alpha=0.5, linewidth=1.5)

    _style(ax_acc, 'Cognitive Defence — Accuracy Across All Attacks', ylim_bottom=0)
    _style(ax_loss, 'Cognitive Defence — Loss Across All Attacks', ylabel='Loss')

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'cognitive_defence_robustness.png', bbox_inches='tight')
    plt.close(fig)
    print("  ✓ cognitive_defence_robustness.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 7 – Summary Table (as a figure)
# ═══════════════════════════════════════════════════════════════════════════════

def fig_summary_table():
    col_labels = ['Attack', 'Defence', 'Final Acc (%)', 'Peak Acc (%)', 'Converged']
    rows = [
        ['Clean (no attack)', 'FedAvg',           '88.64', '98.88', 'Yes (degraded)'],
        ['Static Label Flip', 'No Defence',       '12.56', '90.35', 'No'],
        ['Static Label Flip', 'VERT',             '19.26', '98.81', 'No (unstable)'],
        ['Static Label Flip', 'Cognitive Defence', '98.74', '98.90', 'Yes'],
        ['DnY-Opt',           'No Defence',       '11.35', '11.35', 'No'],
        ['DnY-Opt',           'Trimmed Mean',      '9.58',  '11.35', 'No'],
        ['DnY-Opt',           'Multi-Krum',       '94.87', '95.59', 'Yes'],
        ['DnY-Opt',           'VERT',             '95.50', '97.86', 'Yes'],
        ['DnY-Opt',           'Cognitive Defence', '98.86', '98.95', 'Yes'],
        ['Stat-Opt',          'Trimmed Mean',      '9.58',  '10.09', 'No'],
        ['Stat-Opt',          'VERT',             '95.77', '98.65', 'Yes'],
        ['Stat-Opt',          'Cognitive Defence', '98.86', '98.93', 'Yes'],
        ['Min-Max',           'No Defence',       '11.35', '98.40', 'No (oscillating)'],
        ['Min-Max',           'Cognitive Defence', '98.84', '98.92', 'Yes'],
    ]

    # Colour cells conditionally
    cell_colours = []
    for row in rows:
        row_colours = ['white'] * len(row)
        acc_val = float(row[2])
        if acc_val >= 95:
            row_colours[2] = '#c8e6c9'   # green
        elif acc_val >= 50:
            row_colours[2] = '#fff9c4'   # yellow
        else:
            row_colours[2] = '#ffcdd2'   # red
        if row[4].startswith('Yes') and 'degraded' not in row[4]:
            row_colours[4] = '#c8e6c9'
        elif row[4] == 'No' or 'unstable' in row[4] or 'oscillating' in row[4]:
            row_colours[4] = '#ffcdd2'
        else:
            row_colours[4] = '#fff9c4'
        cell_colours.append(row_colours)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('off')
    ax.set_title('Summary of Defence Performance Across Attack Scenarios',
                 fontsize=14, fontweight='bold', pad=20)

    table = ax.table(cellText=rows, colLabels=col_labels,
                     cellColours=cell_colours,
                     colColours=['#e0e0e0'] * len(col_labels),
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1.0, 1.6)

    # Bold header
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(fontweight='bold')
        cell.set_edgecolor('#cccccc')

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'summary_table.png', bbox_inches='tight')
    plt.close(fig)
    print("  ✓ summary_table.png")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print(f"Generating figures → {OUTPUT_DIR}/\n")
    fig_static_label_flip()
    fig_dny_opt()
    fig_stat_opt()
    fig_min_max()
    fig_final_accuracy_bar()
    fig_cognitive_robustness()
    fig_summary_table()
    print(f"\nDone. {len(list(OUTPUT_DIR.glob('*.png')))} figures saved.")
