K_nod = 8
N_nod = 2 ** K_nod
Beta = 16
a=Beta/N_nod


from systems.Nonrel.oscillator import Oscillator
from systems.Nonrel.x4 import X4
from systems.Nonrel.Morse import Morse
from systems.Nonrel.twowells import Two_wells
from systems.Nonrel.Columb import Columb

from systems.Rel.rel_oscillator import Rel_Oscillator
from systems.Rel.rel_Morse import Rel_Morse
from systems.Rel.rel_twowells import Rel_two_wells
from systems.Rel.rel_Columb import Rel_Columb

from systems.Ultrarel.ur_oscillator import Ur_Oscillator



osc = Oscillator(N_nod,Beta)
anh = X4(N_nod,Beta,g=1)
morse = Morse(N_nod,Beta,alpha=0.5)
tw = Two_wells(N_nod,Beta,g=1,x0=1.41)
columb = Columb(N_nod,Beta,alpha=1,R=1)

rel_osc = Rel_Oscillator(N_nod,Beta,sigma=1)
rel_morse = Rel_Morse(N_nod,Beta,m=1,alpha=0.5)
rel_tw = Rel_two_wells(N_nod,Beta,m=1,g=1,x0=1.41)
rel_columb = Rel_Columb(N_nod,Beta,m=1,alpha=1,R=1)

ur = Ur_Oscillator(N_nod,Beta)