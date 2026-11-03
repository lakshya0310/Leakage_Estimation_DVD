* Example SPICE Netlist: INV
* Tech node: 16nm_HP | Nominal VDD = 0.9V

.INCLUDE ../../16nm_HP.pm
.OPTIONS GMIN=1e-18 ABSTOL=1e-18 RELTOL=1e-6 VNTOL=1e-9
.TEMP 27.0

Vdd supply 0 0.9
Va a 0 0.0

Mp out a supply supply pmos W=64n L=1.6e-08
Mn out a 0 0 nmos W=32n L=1.6e-08

.control
op
print i(Vdd) v(out)
.endc
.END
