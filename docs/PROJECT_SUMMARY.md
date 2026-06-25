# Project Summary

## Project name

Multisource UAV Inspection Path Planning for Rice-Crayfish Co-Culture Areas

## Research problem

Rice-crayfish co-culture farms require UAV inspection of both areal features and facility-related risk points. Uniform coverage routes are easy to execute but may under-serve high-risk locations such as inlet/outlet regions, feeding sites, aerators and abnormal water-quality areas.

## Proposed idea

The project combines baseline coverage path planning with multisource risk-aware revisiting. A risk map is generated from five layers:

```text
R_i = 0.24 C_i + 0.20 V_i + 0.16 T_i + 0.20 S_i + 0.20 K_i
```

High-risk cells are converted into candidate revisiting points. These points are inserted into the baseline route according to benefit-to-cost ratio. Route segments crossing expanded no-fly zones are repaired using A* detours.

## Main result

In the default simulation scenario, the proposed method achieved:

- high-risk coverage: 0.9381
- abnormality detection rate: 0.7937
- missed-risk rate: 0.2063
- route distance: 12811.70 m

Compared with standard CPP:

- standard CPP high-risk coverage: 0.8053
- standard CPP detection rate: 0.7343
- standard CPP route distance: 12731.26 m

Across five random scenarios, the proposed method achieved:

- high-risk coverage: 0.9717 ± 0.0194
- detection rate: 0.8235 ± 0.0386

## Limitation

The project is a simulation validation. Real-world validation requires UAV imagery, field water-quality data, flight logs and expert-labelled abnormality regions.
