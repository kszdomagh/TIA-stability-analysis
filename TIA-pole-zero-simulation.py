import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# -----------------------------
# Parameters of OA based TIA
# -----------------------------
Ao = 1e6          # low-frequency gain
Rf = 1_000        # feedback resistor [Ohm]
Cs = 500e-9       # source capacitor [F]
Cf = 2.7e-9       # feedback capacitor [F]
GBW = 20e6        # amplifier GBW [Hz]

# freq range
f = np.logspace(0.01, 8, 10000)
s = 1j * 2 * np.pi * f

color_b = 'crimson'
color_a = 'forestgreen'
color_PM = 'navy'

# -----------------------------
# loop gain AB before compensation
# -----------------------------
AB_b = Ao / ((1 + s*Rf*Cs) * (1 + s*Ao/(2*np.pi*GBW)))
mag_dB_b = 20 * np.log10(np.abs(AB_b))
phase_deg_b = np.angle(AB_b, deg=True)
# Pole Zero Locations
p0_b = 1 / (Rf * Cs)
p1_b = 2 * np.pi * GBW / Ao

# -----------------------------
# loop gain AB after compensation
# -----------------------------
AB_a = (Ao * (1 + s*Rf*Cf)) / ((1 + s*Ao/(GBW*2*np.pi)) * (1 + s*Rf*(Cs + Cf)))
mag_dB_a = 20 * np.log10(np.abs(AB_a))
phase_deg_a = np.angle(AB_a, deg=True)
# Pole Zero Locations
p0_a = 1 / (Rf *(Cs + Cf))
p1_a = 2 * np.pi * GBW / Ao
z0_a = 1/ (Rf*Cf)


# -----------------------------
# Phase Margin Calculation
# -----------------------------
f0_a = f[np.argmin(np.abs(mag_dB_a))]
id_PM = np.argmin(np.abs(f - f0_a))
PM = 180 - abs(phase_deg_a[id_PM])
print(f'Phase margin after compensation [deg]: {PM:.2f}')


# -----------------------------
# Plot 1: Loop gain plot
# -----------------------------
plt.figure(figsize=(8,6))

plt.subplot(2,1,1)

#0db crossing
plt.axvline(f0_a, color='grey', linestyle='-')
plt.axhline(0, color='grey', linestyle='-')

# Plot before compensation
plt.semilogx(f, mag_dB_b, color=color_b, label='Loop gain before compensation')

# Plot after compensation
plt.semilogx(f, mag_dB_a, color=color_a, label='Loop gain after compensation')

# Pole Zero before compensation - lines
plt.axvline(p0_b/(2*np.pi), color=color_b, alpha=0.5, linestyle='dashed')
plt.axvline(p1_b/(2*np.pi), color=color_b, alpha=0.5, linestyle='dashed')

# Pole Zero after compensation - lines
plt.axvline(p0_a/(2*np.pi), color=color_a, alpha=0.5, linestyle='dashed')
plt.axvline(p1_a/(2*np.pi), color=color_a, alpha=0.5, linestyle='dashed')
plt.axvline(z0_a/(2*np.pi), color=color_a, alpha=0.5, linestyle='dotted')

# Pole Zero before compensation - markers
plt.plot([p0_b/(2*np.pi)], -100, marker='x', color=color_b, markersize=13)
plt.plot([p1_b/(2*np.pi)], -100, marker='x', color=color_b, markersize=13)

# Pole Zero after compensation - markers
plt.plot([p0_a/(2*np.pi)], -50, marker='x', color=color_a, markersize=13)
plt.plot([p1_a/(2*np.pi)], -50, marker='x', color=color_a, markersize=13)
plt.plot([z0_a/(2*np.pi)], -50, marker='$o$', color=color_a, markersize=10)

plt.xlim(1, 1e6)
plt.ylim(-150, 150)

# Freq axis
freqs = [1, 10, 100, 1_000, 10_000, 100_000, 1_000_000]
freqs_labels = ['1', '10', '100', '1k', '10k', '100k', '1M']
plt.xticks(freqs, freqs_labels)

plt.grid(True, which='both', ls='--')
plt.ylabel('Loop Gain [dB]')
plt.legend()

# -----------------------------
# Plot 2: Phase plot
# -----------------------------
plt.subplot(2,1,2)
plt.xlim(1, 1e6)

#0db crossing
plt.axvline(f0_a, color='grey', linestyle='-')

plt.semilogx(f, phase_deg_b, color=color_b, label='Phase before compensation')
plt.semilogx(f, phase_deg_a, color=color_a, label='Phase after compensation')

# poles, zeros lcoations before compensation
plt.axvline(p0_b/(2*np.pi), color=color_b, alpha=0.5, linestyle='dashed')
plt.axvline(p1_b/(2*np.pi), color=color_b, alpha=0.5, linestyle='dashed')

# poles, zeros lcoations after compensation
plt.axvline(p0_a/(2*np.pi), color=color_a, alpha=0.5, linestyle='dashed')
plt.axvline(p1_a/(2*np.pi), color=color_a, alpha=0.5, linestyle='dashed')
plt.axvline(z0_a/(2*np.pi), color=color_a, alpha=0.5, linestyle='dotted')

plt.yticks(range(-200, 1, 20))
plt.ylim(-200, 0)

# freq axis 
plt.xticks(freqs, freqs_labels)

# PM arrow
plt.annotate(
    '', xy=(f0_a, phase_deg_a[id_PM]), xytext=(f0_a, -180), arrowprops=dict(arrowstyle='<->', color=color_PM, linewidth=2)
)

phase_margin_proxy = mlines.Line2D([], [], color=color_PM, linewidth=2, label='PM after compensation')
phase_lines = plt.gca().get_lines()[1:3]
plt.legend(handles=phase_lines + [phase_margin_proxy])

plt.grid(True, which='both', ls='--')
plt.ylabel('Phase [deg]')
plt.xlabel('Frequency [Hz]')

plt.suptitle('Stability Analysis of Operational Amplifier based TIA', fontsize=15)

plt.tight_layout()
plt.show()
