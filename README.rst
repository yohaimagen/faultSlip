faultSlip
======================

faultSlip is a personal Python package for performing static earthquake slip inversion of geodetic datasets with inversion condition number optimization.

faultSlip can handle one or more of the following datasets:
----------------------------------
* InSAR (raster displacement data in the satellite line-of-sight)
* Optical and SAR pixel correlation (raster displacement data in two perpendicular horizontal directions)
* Burst Overlap Interferometry (strips of data in the SAR satellite flight direction)
* GNSS data (tabular displacement data)
* Strain data (GNSS-derived strain data)

Details regarding the condition number optimization and inversion of the 2019 Ridgecrest, CA earthquake can be found in this publication:

Magen, Y., Ziv, A., Inbal, A., Baer, G., & Hollingsworth, J. (2020). Fault rerupture during the july 2019 ridgecrest earthquake pair from joint slip inversion of insar, optical imagery, and gps. Bulletin of the Seismological Society of America. https://doi.org/10.1785/0120200024



