import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.timeseries import BoxLeastSquares
import os

def simulate_lightcurve(period, radius, n_points=1000):
    time = np.linspace(0, max(period)*3, n_points)
    flux = np.ones_like(time)
    
    for p, r in zip(period, radius):
        depth = (r * 0.009158)** 2
        duration = 0.1 * p
        phase = (time % p) / p
        in_transit = (phase < duration/p)
        flux[in_transit] -= depth
        
    return time, flux

def bls_implementation(csv_path, max_planets=5):
    # Load kaggle data
    df = pd.read_csv(csv_path, comment='#')
    
    # Extract planets with valid orbital period and radius (Earth radii)
    planets = df[['pl_orbper', 'pl_rade']].dropna()
    
    # Filter realistic values
    mask = (planets['pl_orbper'] > 0) & (planets['pl_rade'] > 0)
    planets = planets[mask]
    
    # Limit to first 'max_planets' for simulation clarity
    sample = planets.head(max_planets)
    periods = sample['pl_orbper'].values
    radii = sample['pl_rade'].values

    # Simulate combined light curve for selected planets
    time, flux = simulate_lightcurve(periods, radii, n_points=5000)

    # Normalize flux
    flux = flux / np.nanmedian(flux)

    # Compute BLS periodogram
    bls = BoxLeastSquares(time, flux)
    periods_grid = np.linspace(0.5, 1.5*max(periods), 20000)
    results = bls.power(periods_grid, 0.1)
    
    # Find best period peak
    best_idx = np.argmax(results.power)
    best_period = periods_grid[best_idx]
    best_power = results.power[best_idx]
    
    # Plot combined light curve & BLS periodogram
    best_t0 = results.transit_time[best_idx]
    best_dur = results.duration[best_idx]
    model = bls.model(time, best_period, best_dur, best_t0)

    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(10,7), gridspec_kw={'height_ratios':[2,1]})
    ax1.plot(time, flux, 'k-', markersize=2, alpha=0.6, label='Simulated Flux')
    ax1.plot(time, model, 'r-', lw=1.5, label='BLS Transit Model')
    ax1.axvspan(best_t0 - 0.5*best_dur, best_t0 + 0.5*best_dur, color='r', alpha=0.1)
    ax1.set_xlabel('Time (days)')
    ax1.set_ylabel('Normalized Flux')
    ax1.set_title('Simulated Exoplanet Transit Light Curve')
    ax1.legend()

    ax2.plot(periods_grid, results.power, 'k-')
    ax2.axvline(best_period, color='purple', linestyle='--', label=f'Best Period = {best_period:.5f} d')
    ax2.set_xlabel('Period (days)')
    ax2.set_ylabel('BLS Power')
    ax2.legend()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    csv_file = 'PS_2025.02.03_05.09.36.csv'
    bls_implementation(csv_file)
