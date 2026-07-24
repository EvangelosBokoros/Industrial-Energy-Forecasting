# Damavand Energy Forecasting API Contract

## Contract identification

**API version:** `v1`  
**Schema version:** `1.0`  
**Forecasting model:** `2.0_ensemble_70_30`  
**Behavioral evaluation:** `2.1`  
**Target:** `active_energy_kWh`

## Purpose

The API provides daily active-energy forecasts for the Damavand industrial process.

The service uses the official weighted ensemble:

```text
70% post-only Extra Trees
30% full-history AdaBoost