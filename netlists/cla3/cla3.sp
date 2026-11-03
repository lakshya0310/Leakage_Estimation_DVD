* Example SPICE Netlist: 3-bit Carry Lookahead Adder (CLA3)
* Tech node: 16nm_HP | Nominal VDD = 0.9V | Temp = 27C

.INCLUDE ../../16nm_HP.pm
.OPTIONS GMIN=1e-18 ABSTOL=1e-18 RELTOL=1e-6 VNTOL=1e-9
.TEMP 27.0

Vdd supply 0 0.9
Va0 a0 0 0.0
Va1 a1 0 0.0
Va2 a2 0 0.0
Vb0 b0 0 0.0
Vb1 b1 0 0.0
Vb2 b2 0 0.0
Vc0 c0 0 0.0


* Standard Static CMOS Gate Library Subcircuits
.subckt inv in out vdd gnd
Mp out in vdd vdd pmos W=64n L=16n
Mn out in gnd gnd nmos W=32n L=16n
.ends

.subckt nand2 a b out vdd gnd
Mp1 out a vdd vdd pmos W=64n L=16n
Mp2 out b vdd vdd pmos W=64n L=16n
Mn1 out a x gnd nmos W=64n L=16n
Mn2 x b gnd gnd nmos W=64n L=16n
.ends

.subckt nor2 a b out vdd gnd
Mp1 y a vdd vdd pmos W=128n L=16n
Mp2 out b y vdd pmos W=128n L=16n
Mn1 out a gnd gnd nmos W=32n L=16n
Mn2 out b gnd gnd nmos W=32n L=16n
.ends

.subckt and2 a b out vdd gnd
Mp1 n_out a vdd vdd pmos W=64n L=16n
Mp2 n_out b vdd vdd pmos W=64n L=16n
Mn1 n_out a x gnd nmos W=64n L=16n
Mn2 x b gnd gnd nmos W=64n L=16n
Mp3 out n_out vdd vdd pmos W=64n L=16n
Mn3 out n_out gnd gnd nmos W=32n L=16n
.ends

.subckt or2 a b out vdd gnd
Mp1 y a vdd vdd pmos W=128n L=16n
Mp2 n_out b y vdd pmos W=128n L=16n
Mn1 n_out a gnd gnd nmos W=32n L=16n
Mn2 n_out b gnd gnd nmos W=32n L=16n
Mp3 out n_out vdd vdd pmos W=64n L=16n
Mn3 out n_out gnd gnd nmos W=32n L=16n
.ends

.subckt xor2 a b out vdd gnd
Xor a b or_out vdd gnd or2
Xnand a b nand_out vdd gnd nand2
Xand or_out nand_out out vdd gnd and2
.ends

* 3-Bit Carry-Lookahead Adder (CLA3)
.subckt cla3 a0 a1 a2 b0 b1 b2 c0 s0 s1 s2 c3 vdd gnd
* Bit 0 generate & propagate
Xg0 a0 b0 g0 vdd gnd and2
Xp0 a0 b0 p0 vdd gnd or2

* Bit 1 generate & propagate
Xg1 a1 b1 g1 vdd gnd and2
Xp1 a1 b1 p1 vdd gnd or2

* Bit 2 generate & propagate
Xg2 a2 b2 g2 vdd gnd and2
Xp2 a2 b2 p2 vdd gnd or2

* Carry Lookahead Logic: C1
Xt01 p0 c0 t01 vdd gnd and2
Xc1 g0 t01 c1 vdd gnd or2

* Carry Lookahead Logic: C2
Xt10 p1 g0 t10 vdd gnd and2
Xt11 p1 p0 t11 vdd gnd and2
Xt12 t11 c0 t12 vdd gnd and2
Xo21 g1 t10 o21 vdd gnd or2
Xc2 o21 t12 c2 vdd gnd or2

* Carry Lookahead Logic: C3 (Carry-out)
Xt21 p2 c2 t21 vdd gnd and2
Xc3 g2 t21 c3 vdd gnd or2

* Sum bits via XOR2
Xx01 a0 b0 x01 vdd gnd xor2
Xs0 x01 c0 s0 vdd gnd xor2

Xx11 a1 b1 x11 vdd gnd xor2
Xs1 x11 c1 s1 vdd gnd xor2

Xx21 a2 b2 x21 vdd gnd xor2
Xs2 x21 c2 s2 vdd gnd xor2
.ends


Xcla a0 a1 a2 b0 b1 b2 c0 s0 s1 s2 c3 supply 0 cla3

.control
op
print i(Vdd) v(s0) v(s1) v(s2) v(c3)
.endc
.END
