# "Global Bouguer" using global SH models and SHTOOLS, a notebook

[![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.jupyter.org/github/apasto/ggm_bg_maps_binder/blob/master/ggm_bg.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/apasto/ggm_bg_maps_binder/master?filepath=ggm_bg.ipynb)

![Global map of the computed topography-reduced gravity disturbance](./readme_figures/bg_gd.png)

The notebook in this repository presents a *topography-reduced gravity disturbance* (or *complete Bouguer disturbance*, or *NETC disturbance*) computation, starting from the spherical harmonics coefficients of a global gravity field model and a synthetic model of the effect of topography, ice, and water. The code relies *heavily* on the Python components of [SHTOOLS](https://shtools.oca.eu/shtools/public/) (Wieczorek and Meschede, 2018, [doi:10.1029/2018GC007529](https://doi.org/10.1029/2018GC007529)).

It was set up to try SHTOOLS out of curiosity, to get familiar with Jupyter notebooks and their deployment on Binder, and, after a good deal of trial and error, to successfully compute a Bouguer disturbance, which *should* be a trivial task. The operation is defined in Eq. 33 of ICGEM report [STR09/02](https://doi.org/10.2312/GFZ.b103-0902-26).

To keep the notebook lean, the default parameters in the example result in a computation at relatively low degrees (*lmax* is 359, resulting in a 0.25° spaced grid). This is way less than the complete spectral content of the models involved, therefore a re-implementation of the gentle-cut truncation proposed by [Barthelmes (2008)](http://icgem.gfz-potsdam.de/gentlecut_engl.pdf) is included in the [`shtaper`](./shtaper.py) module.
It is successful in avoiding sidelobes/ringing.

This notebook was set up with the following gravity models in mind, their SH coefficients downloaded in the gfc format from [ICGEM](http://icgem.gfz-potsdam.de/home):

* Input global gravity model: **XGM2019e**, Zingerle, P., Pail, R., Gruber, T., Oikonomidou, X. (2019): The experimental gravity field model XGM2019e. *GFZ Data Services*. [doi:10.5880/ICGEM.2019.007](http://doi.org/10.5880/ICGEM.2019.007).

* Topographic effect model: **dV_ELL_Earth2014**, Rexer, M., Hirt, C., Claessens, S., Tenzer, R. (2016): Layer-based modelling of the Earth's gravitational potential up to 10km-scale in spherical harmonics in spherical and ellipsoidal approximation. *Surveys in Geophysics*. [doi:10.1007/s10712-016-9382-2](https://doi.org/10.1007/s10712-016-9382-2).

When run in Binder, these two models are downloaded and unzipped after the container is built, using the [`PostBuild`](./binder/postBuild) script provided this git repository.
If run locally, `PostBuild` must be run beforehand and the downloaded files moved accordingly (or the argument of `read_icgem_gfc` edited with the correct path).
The code below should work with any other combination of models with a maximum SH degree equal or greater than the maximum SH degree of the required grid. Otherwise, zero-padding the missing high degrees is needed.
Note that the topographic effect model used in the example does not include the normal potential.

#### ⚠️ Disclaimer
Keep in mind that this example deals with computing a *topography-reduced gravity disturbance*, as described in Eq. 33 of [STR09/02](https://doi.org/10.2312/GFZ.b103-0902-26), using SHTOOLS and gfc files downloaded from ICGEM. 
Its aim is keeping formal consistency in the computation - a trivial task, albeit prone to errors.
Apart from this, for this purposes, any other issue was deemed of secondary importance (to a reasonable extent). Thus I must warn against using the 'Bouguer disturbance' here obtained in any application.

There are some maps of the tensor (2nd derivative of the topography-reduced disturbing potential), at the notebook far end.
They were done out of curiosity, the same disclaimer applies.
