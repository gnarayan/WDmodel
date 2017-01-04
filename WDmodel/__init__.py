"""
Contains the core class to generate DA White Dwarf model spectra from the model
grid The model grid is generated from model atmosphere structure files of Ivan
Hubeny (UA) by Jay Holberg (UA) The model grid limits are log(g)=7--9.5 and
T_{eff } = 16000-90000 Kelvin As log(g) goes below the lower limit, there it is
no longer in hydrostatic equilibrium, and there is a stellar wind as it goes
over the Eddington limit. Below the lower T_{eff} limit, there are issues with
convective energy transport based on current mixing length theory (and DA white
dwarfs end up at higher log(g) values by about 0.15 dex). Above the high
T_{eff} limit, the spectral lines are too shallow to be measured reliably. 
"""
from .WDmodel import WDmodel
