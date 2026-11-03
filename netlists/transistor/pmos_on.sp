* Example Single-Transistor Netlist: PMOS_ON
* Tech node: 16nm_HP | Nominal VDD = 0.9V

.INCLUDE ../../16nm_HP.pm
.OPTIONS GMIN=1e-18 ABSTOL=1e-18 RELTOL=1e-6 VNTOL=1e-9
.TEMP 27.0

Mp drain gate source body pmos W=32n L=16n

Vd drain  0 0.0
Vg gate   0 0.0
Vs source 0 0.9
Vb body   0 0.9

.control
op
print i(Vd) i(Vg) i(Vs) i(Vb)
.endc
.END
