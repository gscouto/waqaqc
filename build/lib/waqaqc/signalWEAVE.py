from waqaqc.signal2noise import signal, totalSeeing


def signalWEAVE(mag, time, instrument_mode='redLR', airmass=1.2, band='R',
                seeing_input=1.0, rn=3., dark=0., eff=None, sb=None, skysb=22.7,
                skyband="B", profile='Gaussian', betam=None, offset=0.0,
                LIFU=False, rv=False, rvscale=0.6, verbose=True):
    """
    Compute signal-to-noise ratio and optionally radial velocity accuracy
    using instrument-specific parameters.

    Parameters match the original script logic.

    Returns
    -------
    dict
        Result dictionary with S/N and additional metrics.
    """

    # Fiber injection and optics
    fFP = 3.2
    fcol = 3.1
    fcam = 1.8

    # Instrument mode settings
    if instrument_mode == 'blueLR':
        res = 5750.
        cwave = 4900.
        pfEff = 0.839
        fiberEff = 0.784
        specEff = 0.567
        QE = 0.924
    elif instrument_mode == 'redLR':
        res = 5750.
        cwave = 7950.
        pfEff = 0.870
        fiberEff = 0.852
        specEff = 0.607
        QE = 0.805
    elif instrument_mode == 'blueHR':
        res = 21000.
        cwave = 4250.
        pfEff = 0.800
        fiberEff = 0.691
        specEff = 0.337
        QE = 0.923
    elif instrument_mode == 'greenHR':
        res = 21000.
        cwave = 5130.
        pfEff = 0.833
        fiberEff = 0.794
        specEff = 0.498
        QE = 0.927
    elif instrument_mode == 'redHR':
        res = 21000.
        cwave = 6450.
        pfEff = 0.861
        fiberEff = 0.837
        specEff = 0.505
        QE = 0.947
    else:
        raise ValueError("Invalid instrument_mode")

    # Seeing
    seeing = totalSeeing(seeing_input)

    # Fiber diameter
    fiberD = 1.3045
    if LIFU:
        res = res / 2.
        fiberD = fiberD * 2.

    # Handle Moffat profile
    if profile != 'Moffat':
        betam = None

    # Initialize signal object
    S = signal(QE, fiberEff, specEff, pfEff, res, offset, fiberD, fFP, fcol,
               fcam, cwave=cwave, profile=profile, betam=betam)

    # Calculate S/N
    snr = S.S2N(time, mag, band, airmass, fiberD, seeing, rn, dark,
                eff, sb, skysb, skyband)
    s2n = snr['SNR']

    if verbose:
        print(f"Spectral resolving power R = {int(S.res):5}")
        print(f"Average dispersion (Ang/pixel) = {S.disp:8.4f}")
        print(f"Number of pixels/fiber along slit = {S.fiberCCD:5.2f}")
        print(f"Resolution element (Ang) = {S.fiberCCD * S.disp:6.3f}")
        print(f"Efficiency = {snr['efficiency']:4.2f} Effective area = {snr['effectivearea']:5.2f} m^2")
        print(
            f"Object photons/pixel = {int(snr['objectphotons'])} sky photons (between lines)/pixel = {int(snr['skyphotons'])}")
        print("(both integrated over spatial direction)")
        print(f"S/N/Ang = {s2n:8.2f} S/N/resolution element = {snr['SNRres']:8.2f}")
        print(f"S/N/pix = {snr['SNRpix']:8.2f}")
        if rv:
            sigmarv = S.RVaccuracy(s2n, scale=rvscale)
            print(f"RV error (km/s) = {sigmarv:8.2f}")

    # Optionally return the full result
    return snr
