# README for npzd

#### This code is to allow rapid exploration of parameter and functional choices in NPZD code. In particular it implements a time-dependent 1-D (vertical) model following either Fennel or Banas specifications.

---

#### driver.py
This runs the 1-D model and produces a couple of plots of the solution. To change from the fennel to banas parameters you edit `modname` around line 16.

The only non-standard dependency is:
```
from lo_tools import plotting_functions as pfun
```
which implies it is meant to be run inside the (loenv) environment, but this can be omitted. It only makes the fontsize easier to read in the plots.

#### npzd_equations.py
This is a module of functions used by driver.py. The most important one is update_v() which does all the biogeochemical transformations. It uses a backwards-implicit integration, just like in ROMS, which can be non-intuitive at first. It has good stability and conservation properties.

#### parameters.py
This has all the parameter choices for both the fennel and banas models.

---

## References

Fennel, K., Hu, J., Laurent, A., Marta-Almeida, M., & Hetland, R. (2013). Sensitivity of hypoxia predictions for the northern Gulf of Mexico to sediment oxygen consumption and model nesting. Journal of Geophysical Research: Oceans, 118(2), 990-1002. doi:10.1002/jgrc.20077
**The primary reference for the Fennel model. The one coded in ROMS is only slightly different.**

Banas, N. S., Lessard, E. J., Kudela, R. M., MacCready, P., Peterson, T. D., Hickey, B. M., & Frame, E. (2009). Planktonic growth and grazing in the Columbia River plume region: A biophysical model study. Journal of Geophysical Research, 114, C00B06. doi:10.1029/2008JC004993
**The primary reference for the Banas model. The one coded here is only slightly different, incorporating differences describes in Davis et al. (2014).**

Fennel, K., Mattern, J. P., Doney, S. C., Bopp, L., Moore, A. M., Wang, B., & Yu, L. (2022). Ocean biogeochemical modelling. Nature Reviews Methods Primers, 2(1). doi:10.1038/s43586-022-00154-2
**A great overview of biogeochemical modeling.**

Siedlecki, S. A., Banas, N. S., Davis, K. A., Giddings, S. N., Hickey, B. M., MacCready, P., Connolly, T. P., & Geier, S. (2015). Seasonal and interannual oxygen variability on the Washington and Oregon continental shelves. Journal of Geophysical Research: Oceans, 120(2), 608-633. doi:10.1002/2014jc010254
**Added oxygen to the Banas model.**

Davis, K. A., Banas, N. S., Giddings, S. N., Siedlecki, S. A., MacCready, P., Lessard, E. J., Kudela, R. M., & Hickey, B. M. (2014). Estuary-enhanced upwelling of marine nutrients fuels coastal productivity in the U.S. Pacific Northwest. Journal of Geophysical Research: Oceans, 119(12), 8778-8799. doi:10.1002/2014jc010248
**Some modifications to the phytoplankton forumlation in the Banas model.**
