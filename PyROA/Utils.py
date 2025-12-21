import matplotlib.pyplot as plt
import pickle
import numpy as np
from scipy.stats import median_abs_deviation as mad
from matplotlib import gridspec#
import scipy.interpolate as interpolate
import corner
import matplotlib.ticker as mtick
import matplotlib.ticker as ticker
import astropy.units as u
import astropy.constants as ct
from pandas import DataFrame
import matplotlib.ticker as mtick
import scipy.optimize as opt
from uncertainties import ufloat
from pathlib import Path

def Chains(nparam, objname, filters, delay_ref,
           burnin=0, samples_file='samples_flat.obj',
           outputdir='./', initial=0,
           savefig=True, figname=None):
    
    """Parameter Chain Plot of MCMC from PyROA outputs
	nparam : str
		Parameters to show in the corner plot. Can choose
		individual ones such as: 'A','B','tau','sig'
		or plot all parameters: 'all'
		(NOTE, the latter can create very large files)
    filters : list
        List of filters used in the PyROA fit.
    delay_ref : str
        Name of the filter used as the reference. Must be contained in
        "filters".


	burnin : float, optional
        Number of samples to discard in the fit, from 0 to burnin.
        This cut is applied to the samples_flat.obj.
        Use the "convergence" or "chains" plots to determine this 
        number.  Default: 0
    samples_file : str, optional
        File name of the MCMC samples. Default: "samples_flat.obj"
        This is the PyROA standard output.
    outputdir : str, optional
        Directory path where PyROA "*.obj" are stored. 
        This is the PyROA standard output. Default: Current directory "./"
    initial : int, optional
        Initial chain number to discard. Default: 0
    savefig : bool, optional
        Save figure as a PDF Default: True.
    figname : str, optional
        Name of the figure to be saved. If not provided, the default
        name is 'pyroa_chains.pdf'


    Returns
    -------
    None

    Example
    -------
	import pyroa_utils
	
	importlib.reload(utils)
	filters=['u','B','g']
	burnin = 250000
	delay_ref = 'g'
    # Plot chain for the delay "tau" parameters
	pyroa_utils.Chains('tau',filters,delay_ref,
	                  burnin=burnin)

	"""

    # 使用 Path 处理目录，并确保存在
    outputdir = Path(outputdir)
    outputdir.mkdir(parents=True, exist_ok=True)
    objname = objname

    # 读取样本
    samples_path = outputdir / samples_file
    with open(samples_path, 'rb') as f:
        samples = pickle.load(f)

    samples = samples[burnin:, :]

    ss = np.where(np.array(filters) == delay_ref)[0][0]

    labels = []
    for i in range(len(filters)):
        for j in ["A", "B", r"$\tau$", r"$\sigma$"]:
            labels.append(j + r'$_{' + filters[i] + r'}$')
    labels.append(r'$\Delta$')
    all_labels = labels.copy()
    del labels[ss * 4 + 2]

    # -------- 几种不同的 nparam 模式 --------
    if isinstance(nparam, int):
        ndim = nparam
        fig, axes = plt.subplots(ndim, figsize=(10, 2 * ndim), sharex=True)
        ct = 0
        for i in range(initial, initial + ndim):
            ax = axes[ct]
            ax.plot(samples[:, i], "k", alpha=0.3)
            ax.set_xlim(0, len(samples))
            ax.set_ylabel(labels[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)
            ct += 1
        axes[-1].set_xlabel("Chain number")

    elif nparam == 'all':
        ndim = samples.shape[1]
        fig, axes = plt.subplots(ndim, figsize=(10, 2 * ndim), sharex=True)
        ct = 0
        for i in range(ndim):
            ax = axes[ct]
            ax.plot(samples[:, i], "k", alpha=0.3)
            ax.set_xlim(0, len(samples))
            ax.set_ylabel(labels[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)
            ct += 1
        axes[-1].set_xlabel("Chain number")

    elif nparam in ('tau', 'A', 'B', 'sig'):
        if nparam == 'A':
            shifter = 0
        elif nparam == 'B':
            shifter = 1
        elif nparam == 'tau':
            shifter = 2
        else:  # 'sig'
            shifter = 3

        ndim = len(filters)
        fig, axes = plt.subplots(ndim - 1, figsize=(10, 2 * ndim), sharex=True)
        ct = 0
        mm = 0
        for i in range(ndim):
            if i != ss:
                ax = axes[ct]
                ax.plot(samples[:, i * 4 + shifter + mm], "k", alpha=0.3)
                ax.set_xlim(0, len(samples))
                ax.set_ylabel(all_labels[i * 4 + shifter], fontsize=20)
                ax.yaxis.set_label_coords(-0.1, 0.5)
                ct += 1
            else:
                mm = -1
        axes[-1].set_xlabel("Chain number")

    elif nparam == 'delta':
        fig, ax = plt.subplots(1, figsize=(10, 2))
        ax.plot(samples[:, -1], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(all_labels[-1], fontsize=20)
        ax.yaxis.set_label_coords(-0.1, 0.5)
        ax.set_xlabel("Chain number")

    # -------- 保存图片 --------
    if savefig:
        outname = f'{objname}_pyroa_chains.pdf'
        outpath = outputdir / outname
        plt.savefig(outpath)


def CornerPlot(nparam, objname, filters, delay_ref,
               burnin=0,
               samples_file='samples_flat.obj',
               outputdir='./',
               savefig=True, figname=None):

    """
    Corner Plot of MCMC parameters from PyROA outpu
	nparam : str
		Parameters to show in the corner plot. Can choose
		individual ones such as: 'A','B','tau','sig'
		or plot all parameters: 'all'
		(NOTE, the latter can create very large files)
    filters : list
        List of filters used in the PyROA fit.
    delay_ref : str
        Name of the filter used as the reference. Must be contained in
        "filters".


	burnin : float, optional
        Number of samples to discard in the fit, from 0 to burnin.
        This cut is applied to the samples_flat.obj.
        Use the "convergence" or "chains" plots to determine this 
        number.  Default: 0
    samples_file : str, optional
        File name of the MCMC samples. Default: "samples_flat.obj"
        This is the PyROA standard output.
    outputdir : str, optional
        Directory path where PyROA "*.obj" are stored. 
        This is the PyROA standard output. Default: Current directory "./"
    savefig : bool, optional
        Save figure as a PDF Default: True.
    figname : str, optional
        Name of the figure to be saved. If not provided, the default
        name is 'pyroa_corner.pdf'


    Returns
    -------
    None

    Example
    -------
	import pyroa_utils
	
	importlib.reload(utils)
	filters=['u','B','g']
	burnin = 250000
	delay_ref = 'g'
	pyroa_utils.corner_plot('tau',filters,delay_ref,
	                  burnin=burnin)
	"""

    # 用 Path 处理目录，并保证存在
    outputdir = Path(outputdir)
    outputdir.mkdir(parents=True, exist_ok=True)

    # 读取 samples
    samples_path = outputdir / samples_file
    with open(samples_path, 'rb') as f:
        samples = pickle.load(f)[burnin:]

    ss = np.where(np.array(filters) == delay_ref)[0][0]

    labels = []
    for i in range(len(filters)):
        for j in ["A", "B", r"$\tau$", r"$\sigma$"]:
            labels.append(j + r'$_{' + filters[i] + r'}$')
    labels.append(r'$\Delta$')
    all_labels = labels.copy()
    del labels[ss * 4 + 2]

    # 只画某一类参数：A / B / tau / sig
    if nparam in ('tau', 'A', 'B', 'sig'):
        if nparam == 'A':
            shifter = 0
        elif nparam == 'B':
            shifter = 1
        elif nparam == 'tau':
            shifter = 2
        else:  # 'sig'
            shifter = 3

        list_only = []
        mm = 0
        for i in range(len(filters)):
            if i != ss:
                list_only.append(i * 4 + shifter + mm)
            else:
                mm = -1

        gg = corner.corner(
            samples[:, list_only],
            show_titles=True,
            labels=np.array(labels)[list_only],
            title_kwargs={'fontsize': 19}
        )

    # 画所有参数
    elif nparam == 'all':
        gg = corner.corner(
            samples,
            show_titles=True,
            labels=all_labels,
            title_kwargs={'fontsize': 19}
        )

    # 保存图像
    if savefig:
        outname = f'{objname}_pyroa_corner.pdf'
        outpath = outputdir / outname
        plt.savefig(outpath)

def LagSpectrum(filters, objname, delay_ref, wavelengths,
                burnin=0, samples_file='samples_flat.obj',
                outputdir='./',
                band_colors=None,
                redshift=0.0,
                savefig=True, figname=None):

    # 统一用 Path，并确保目录存在
    outputdir = Path(outputdir)
    outputdir.mkdir(parents=True, exist_ok=True)
    objname = objname
    # 读样本
    samples_path = outputdir / samples_file
    with open(samples_path, 'rb') as f:
        samples = pickle.load(f)[burnin:]

    ss = np.where(np.array(filters) == delay_ref)[0][0]

    labels = []
    for i in range(len(filters)):
        for j in ["A", "B", r"$\tau$", r"$\sigma$"]:
            labels.append(j + r'$_{' + filters[i] + r'}$')
    labels.append(r'$\Delta$')
    all_labels = labels.copy()
    del labels[ss * 4 + 2]

    # 只要 tau
    shifter = 2
    list_only = []
    mm = 0
    ndim = len(filters)
    for i in range(ndim):
        if i != ss:
            list_only.append(i * 4 + shifter + mm)
        if i == ss:
            mm = -1

    # 计算各波段 lag 及误差
    lag = np.zeros(ndim - 1)
    lag_m = np.zeros(ndim - 1)
    lag_p = np.zeros(ndim - 1)
    for j, i in enumerate(list_only):
        q50 = np.percentile(samples[:, i], 50)
        q84 = np.percentile(samples[:, i], 84)
        q16 = np.percentile(samples[:, i], 16)
        lag[j] = q50
        lag_m[j] = q50 - q16
        lag_p[j] = q84 - q50

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111)

    plt.axhline(y=0, ls='--', alpha=0.5)

    # 颜色：默认给每个点一个 'k'
    if band_colors is None:
        band_colors = ['k'] * lag.size

    for i in range(lag.size):
        plt.errorbar(
            wavelengths[i] / (1 + redshift),
            lag[i] / (1 + redshift),
            yerr=lag_m[i],
            marker='o',
            color=band_colors[i]
        )

    if redshift > 0:
        plt.xlabel(r'Rest Wavelength / $\mathrm{\AA}$')
        plt.ylabel(r'$\tau_{\rm rest}$ / day')
    else:
        plt.xlabel(r'Observed Wavelength / $\mathrm{\AA}$')
        plt.ylabel(r'$\tau$ / day')

    if savefig:
        outpath = outputdir / f'{objname}_pyroa_lagspectrum.pdf'
        plt.savefig(outpath)

def Lightcurves(objName, filters, delay_ref,
                lc_file="Lightcurve_models.obj",
                samples_file='samples_flat.obj',
                slow_comp_file='Slow_Comps.obj',
                outputdir='./', datadir='./',
                burnin=0, band_colors=None,
                limits=None, grid=False, grid_step=5.0,
                show_delay_ref=False, ylab=None,
                filter_labels=None, savefig=True, figname=None,
                include_slow_comp=False, slow_comp_delta=30.0):
    """Plots the Lightcurve data and best fit as measured by PyROA
    （原始 docstring 略）
    """

    plt.rcParams.update({
        "font.family": "Serif",
        "font.serif": ["Times New Roman"],
        "figure.figsize": [40, 30],
        "font.size": 19
    })

    outputdir = Path(outputdir)
    datadir = Path(datadir)
    outputdir.mkdir(parents=True, exist_ok=True)

    if ylab is None:
        ylab = r"F$_{\nu}$" + "\nmJy"
    if filter_labels is None:
        filter_labels = filters

    ss = np.where(np.array(filters) == delay_ref)[0][0]

    # 读 MCMC 样本 & 模型
    with open(outputdir / samples_file, 'rb') as f:
        samples_flat = pickle.load(f)
    samples_flat = samples_flat[burnin:, :]

    with open(outputdir / lc_file, 'rb') as f:
        models = pickle.load(f)

    if include_slow_comp:
        with open(outputdir / slow_comp_file, 'rb') as f:
            slow_comps = pickle.load(f)

    # 按 A, B, tau, sig 切 chunk
    chunk_size = 4
    transpose_samples = np.transpose(samples_flat)
    # 在参考滤光片的位置插入 tau=0
    transpose_samples = np.insert(
        transpose_samples,
        [ss * 4 + 2],
        np.array([0.0] * len(transpose_samples[1])),
        axis=0
    )
    samples_chunks = [
        transpose_samples[i:i + chunk_size]
        for i in range(0, len(transpose_samples), chunk_size)
    ]

    # 准备画布：每个滤光片两列（左：lightcurve/residual，右：lag posterior）
    fig = plt.figure(figsize=(20, len(filters) * 3.5))
    corro = 1
    if show_delay_ref:
        corro = 0

    gs = fig.add_gridspec(len(filters) - corro, 2, hspace=0, wspace=0,
                          width_ratios=[5, 1])
    axs = gs.subplots(sharex='col')

    if band_colors is None:
        band_colors = ['k'] * len(filters)

    data = []
    ko = 0

    if limits is not None:
        xmin = limits[0]
        xmax = limits[1]

    # ======================== 主循环：逐滤光片处理 ========================
    for i in range(len(filters)):
        # 读数据
        data_file = datadir / f"{objName}_{filters[i]}.dat"
        this_data = np.loadtxt(data_file)
        data.append(this_data)

        mjd = this_data[:, 0]
        flux = this_data[:, 1]
        err = this_data[:, 2]

        if (i == 0) and (limits is None):
            xmin = np.nanmin(mjd) - 10
            xmax = np.nanmax(mjd) + 10

        # 额外方差（sig）并入误差
        B = np.percentile(samples_chunks[i][1], 50)
        sig = np.percentile(samples_chunks[i][3], 50)
        err = np.sqrt(err**2 + sig**2)

        # 非参考滤光片，或者显示参考滤光片
        if filters[i] != delay_ref:

            # 在左侧大格子 gs[i-ko, 0] 里再切一列 6x1 的子格子
            gs00 = gridspec.GridSpecFromSubplotSpec(
                6, 1, subplot_spec=gs[i - ko, 0], hspace=0
            )
            ax1 = fig.add_subplot(gs00[:-2, :])  # lightcurve
            ax2 = fig.add_subplot(gs00[-2:, :])  # residuals

            # y 轴范围用 MAD 控制
            ax1.set_ylim(
                np.median(flux) - 4.8 * mad(flux),
                np.median(flux) + 4.8 * mad(flux)
            )

            if i < len(filters) - 1:
                ax2.set_xticklabels([])
            else:
                ax2.set_xlabel("MJD")
            ax1.set_xticklabels([])
            axs[i - ko][0].set_yticklabels([])

            # 画数据
            ax1.errorbar(
                mjd, flux, yerr=err,
                ls='none', marker=".", color=band_colors[i], ms=2
            )

            # 画模型 & 残差
            t, m, errs = models[i]
            new_m = np.interp(mjd, t, m)

            ax2.axhline(y=0, ls='--', color='k')
            ax2.errorbar(
                mjd, (flux - new_m) / err, yerr=1,
                ls='none', marker=".", color=band_colors[i], ms=2
            )
            ax2.set_ylim(-4.9, 4.9)

            if grid:
                for hh in np.arange(59330, xmax, grid_step):
                    ax1.axvline(x=hh, ls='--', color='grey', alpha=0.4)

            ax1.set_xlim(xmin, xmax)
            ax2.set_xlim(xmin, xmax)

            ax1.plot(t, m, color="black", lw=3)

            # 慢变成分
            if include_slow_comp:
                slow_comp = slow_comps[i]
                ax1.plot(mjd, slow_comp(mjd) + B,
                         linestyle="dashed", color="black")

            filto = filter_labels[i]

            # 处理可能的 inf 模型误差
            inf_mask = errs == np.inf
            errs[inf_mask] = 1e32
            inf_mask = errs == -np.inf
            errs[inf_mask] = -1e32

            ax1.text(
                0.1, 0.2, filto,
                color=band_colors[i], fontsize=19,
                transform=ax1.transAxes
            )
            ax1.fill_between(
                t, m + errs, m - errs,
                alpha=0.5, color="black"
            )
            ax1.set_ylabel(ylab)
            ax2.set_ylabel(r"$\chi$")

            # 右侧：时间延迟后验直方图
            tau_samples = samples_chunks[i][2]
            axs[i - ko][1].hist(
                tau_samples, color=band_colors[i],
                bins=50, histtype='stepfilled'
            )
            q16, q50, q84 = np.percentile(tau_samples, [16, 50, 84])
            axs[i - ko][1].axvline(x=q50, color="black")
            axs[i - ko][1].axvline(x=q16, color="black", ls="--")
            axs[i - ko][1].axvline(x=q84, color="black", ls="--")
            axs[i - ko][1].axvline(x=0, color="grey", ls="-")
            axs[i - ko][1].set_xlabel("Time Delay")
            axs[i - ko][1].set_yticklabels([])
            axs[i - ko][1].axes.get_yaxis().set_visible(False)
            axs[i - ko][0].set_xticklabels([])

            axs[0][0].set_title(objName)
            axs[0][1].set_title("Time Delay")

        # 参考滤光片
        if filters[i] == delay_ref:

            if show_delay_ref:
                gs00 = gridspec.GridSpecFromSubplotSpec(
                    6, 1, subplot_spec=gs[i - ko, 0], hspace=0
                )
                ax1 = fig.add_subplot(gs00[:-2, :])
                ax2 = fig.add_subplot(gs00[-2:, :])

                ax1.set_ylim(
                    np.median(flux) - 4.8 * mad(flux),
                    np.median(flux) + 4.8 * mad(flux)
                )

                if i < len(filters) - 1:
                    ax2.set_xticklabels([])
                else:
                    ax2.set_xlabel("MJD")
                ax1.set_xticklabels([])
                axs[i - ko][0].set_yticklabels([])

                ax1.errorbar(
                    mjd, flux, yerr=err,
                    ls='none', marker=".", color=band_colors[i], ms=2
                )

                t, m, errs = models[i]
                new_m = np.interp(mjd, t, m)
                ax2.axhline(y=0, ls='--', color='k')
                ax2.errorbar(
                    mjd, (flux - new_m) / err, yerr=1,
                    ls='none', marker=".", color=band_colors[i], ms=2
                )
                ax2.set_ylim(-4.9, 4.9)

                if grid:
                    for hh in np.arange(59330, xmax, 5):
                        ax1.axvline(x=hh, ls='--',
                                    color='grey', alpha=0.4)

                ax1.set_xlim(xmin, xmax)
                ax2.set_xlim(xmin, xmax)

                ax1.plot(t, m, color="black", lw=3)

                filto = filter_labels[i]

                inf_mask = errs == np.inf
                errs[inf_mask] = 1e32
                inf_mask = errs == -np.inf
                errs[inf_mask] = -1e32

                ax1.text(
                    0.1, 0.2, filto,
                    color=band_colors[i], fontsize=19,
                    transform=ax1.transAxes
                )
                ax1.fill_between(
                    t, m + errs, m - errs,
                    alpha=0.5, color="black"
                )
                ax1.set_ylabel(ylab)
                ax2.set_ylabel(r"$\chi$")

                ko = 0
            else:
                # 不单独画参考滤光片：后面索引要减一
                ko = 1

    for ax in axs.flat:
        ax.label_outer()

    if savefig:
        outpath = outputdir/f"{objName}_pyroa_lightcurves.pdf"
        plt.savefig(outpath)


def FluxFlux(objName, filters, delay_ref, gal_ref,wavelengths,
            lc_file="Lightcurve_models.obj",
            samples_file='samples_flat.obj',
            xt_file='X_t.obj',
            outputdir = './', datadir='./',
            burnin=0, band_colors = None,
            input_units='mJy',output_units='mJy',
            redshift=0.0, ebv=0.0,
            limits=None, ylab = None,
            savefig=True, figname=None):
    """Flux-Flux analysis and Spectral energy distribution as
    estimated by PyROA.

    Parameters
    ----------
    objName : str
        Name used in PyROA for the data files.
    filters : list
        List of filters used in the PyROA fit.
    delay_ref : str
        Name of the filter used as the reference. Must be contained in
        "filters".
    gal_ref : str
        Name of the filter used as the reference to construct
        the galaxy spectrum. Usually the one that crosses the x-axis first
        *** Must be contained in "filters" array ***.
    wavelengths : list
        List of wavelengths corresponding to each filter. Assumed to be in 
        Angstroms.

    lc_file : str, optional
        File name of lightcurve models. Default: "Lightcurve_models.obj"
        This is the PyROA standard output.
    samples_file : str, optional
        File name of the MCMC samples. Default: "samples_flat.obj"
        This is the PyROA standard output.
    xt_file : str, optional
        File name of the driving lightcurve model. Default: "X_t.obj"
        This is the PyROA standard output.
    outputdir : str, optional
        Directory path where PyROA "*.obj" are stored. 
        This is the PyROA standard output. Default: Current directory "./"
    datadir : str, optional
        Directory path where PyROA "*.dat" are stored. 
        This is the PyROA standard input. Default: Current directory "./"
    burnin : float, optional
        Number of samples to discard in the fit, from 0 to burnin.
        This cut is applied to the samples_flat.obj.
        Use the "convergence" or "chains" plots to determine this 
        number.  Default: 0
    band_colors : list, optional
        List of colours for each filter. List must be the same size as
        the filters array. Default: all lightcurves will be black.
    flux_fnu : bool, optional
        Units to be used in the Flux-Flux analysis.
        Default: True
    input_unit : float, optional
        Units of flux values. Valid options:
        'mJy', 'Jy', 'fnu'= erg/s/cm^2/Hz,'flam'=erg/s/cm^2/Ang
        Default: 'mJy'
    output_unit : float, optional
        Units of flux values. Valid options:
        'mJy', 'Jy', 'fnu'= erg/s/cm^2/Hz,'flam'=erg/s/cm^2/Ang
        Default: 'mJy'
    redshift : float, optional
        Redshift of the AGN. Default: 0.0
    ebv : float, optional
        E(B-V) value of the line-of-sight extinction towards the AGN. 
        The SED plot will be corrected by this amount following 
        Fitzpatrick (1999) parametrisation. Default: 0.0
    limits : list[2], optional
        Limits on the shared Y-axis of all plots e.g., [xmin,xmax]
        Default: It is determined by the data.
    ylab : str, optional
        Label of Y-axis. Default: "F$_{\nu}$"+"\nmJy"
    filter_labels : list, optional
        List of filter names that overrides the original names as given by
        "filters" one. Default: None.
    savefig : bool, optional
        Save figure as a PDF Default: True.
    figname : str, optional
        Name of the figure to be saved. If not provided, the default
        name is 'pyroa_fluxflux.pdf' and 'pyroa_sed.pdf'

    Returns
    -------
    None

    Example
    -------
    import pyroa_utils
    
    waves = [3580,4392,4770,5468,6215,7545,8700]
    objName="NGC_4151"
    datadir = "pyroa_yr1/"
    gal_ref = 'u'
    burnin = 100000
    filters=['u','B','g','g1','V','r','i','z']
    band_colors=['#0652DD','#1289A7','#006266','#006266','#A3CB38',
    			 'orange','#EE5A24','brown']
    
    utils.fluxflux(objName,filters,delay_ref,gal_ref,
                    datadir=datadir,outputdir='./',
                    burnin=burnin,band_colors=band_colors,
                    wavelengths=waves,ebv=0.027,redshift=0.003326,
                    limits=[7,120])
	"""

    plt.rcParams.update({
        "font.family": "Serif",  
        "font.serif": ["Times New Roman"],
    "figure.figsize":[40,30],
    "font.size": 19})  
	
    # To Do: 改成Path
    outputdir = Path(outputdir)
    datadir = Path(datadir)
    outputdir.mkdir(parents=True, exist_ok=True)

    #if filter_labels == None: filter_labels = filters

    if input_units == 'mJy': funits = 1*u.mJy
    if input_units == 'Jy': funits = 1*u.Jy
    if input_units == 'fnu': funits = 1*u.erg/u.s/(u.cm**2)/u.Hz
    if input_units == 'flam': funits = 1e-15 * u.erg/u.s/(u.cm**2)/u.Angstrom
    if output_units == 'mJy': 
        ylab = r"F$_{\nu}$"+" / mJy"
    if output_units == 'Jy': 
        ylab = r"F$_{\nu}$"+" / Jy"
    if output_units == 'fnu': 
        ylab = r"F$_{\nu}$"+r" / erg s$^{-1}$ cm$^{-2}$ ${\rm Hz}^{-1}$"
    if output_units == 'flam': 
        ylab = r"F$_{\lambda}$"+r" / $\times10^{-15}$ erg s$^{-1}$ cm$^{-2}$ ${\rm \AA}^{-1}$"

    ss = np.where(np.array(filters) == delay_ref)[0][0]

    with open(outputdir/samples_file, 'rb') as f:
        samples_flat = pickle.load(f)
    samples_flat = samples_flat[burnin:,:]

    with open(outputdir/lc_file, 'rb') as f:
         models = pickle.load(f)
    
    with open(outputdir/xt_file, 'rb') as f:
        norm_lc = pickle.load(f)

    wave = np.array(wavelengths)

    #Split samples into chunks, 4 per lightcurve i.e A, B, tau, sig
    chunk_size=4
    transpose_samples = np.transpose(samples_flat)
    #Insert zero where tau_0 would be 
    transpose_samples = np.insert(transpose_samples, [ss*4+2], np.array([0.0]*len(transpose_samples[1])), axis=0)
    samples_chunks = [transpose_samples[i:i + chunk_size] for i in range(0, len(transpose_samples), chunk_size)] 

    gal_spectrum,gal_spectrum_err,fnu_f,fnu_b,slope,slope_err = [],[],[],[],[],[]
    fnu_f_err,fnu_b_err = [], []
    
    fig = plt.figure(figsize=(10,7))
    xx = np.linspace(-15,5,300) # ？？？
    max_flux = 0.0
    
    fac_flux = np.ones(len(wavelengths))

    # ---- 基于 gal_ref 预先计算 x_gal_mcmc（供所有滤光片复用）----
    if gal_ref not in filters:
        raise ValueError(f"gal_ref='{gal_ref}' 不在 filters 列表中")

    j_gal = filters.index(gal_ref)
    snu_gal = samples_chunks[j_gal][0]
    cnu_gal = samples_chunks[j_gal][1]

    # 计算 x_gal_mcmc，并屏蔽除零/无效
    with np.errstate(divide='ignore', invalid='ignore'):
        x_gal_mcmc_full = -cnu_gal / snu_gal
    x_gal_mcmc = x_gal_mcmc_full[np.isfinite(x_gal_mcmc_full)]
    if x_gal_mcmc.size == 0:
        raise ValueError("x_gal_mcmc 全为非数值；请检查 gal_ref 的样本（可能 s=0）。")

    x_gal     = np.median(x_gal_mcmc)
    x_gal_std = np.std(x_gal_mcmc)

    # ---- 主循环：包含参考波段；不再使用 kk ----
    fac_flux = np.ones(len(wavelengths), dtype=float)

    for i, flt in enumerate(filters):
        # 读取每个滤光片对应的 .dat 文件
        data_file =datadir/f"{objName}_{flt}.dat"
        data = np.loadtxt(data_file)
        snu_mcmc = samples_chunks[i][0]
        cnu_mcmc = samples_chunks[i][1]
        sig = np.percentile(samples_chunks[i][3], 50)

        # 初始化并生成 200 条模拟直线
        mc_pl = np.zeros((200, xx.size))
        for lo in range(200):
            jj = int(np.random.uniform(0, snu_mcmc.size))
            mc_pl[lo] = cnu_mcmc[jj] + xx * snu_mcmc[jj]

        # 若长度不一致，将 x_gal_mcmc 重采样到与 snu_mcmc 相同长度
        if x_gal_mcmc.size != snu_mcmc.size:
            idx = np.random.randint(0, x_gal_mcmc.size, size=snu_mcmc.size)
            xg = x_gal_mcmc[idx]
            xg_std = x_gal_std
        else:
            xg = x_gal_mcmc
            xg_std = x_gal_std

        # Galaxy + 端点/斜率
        # gal_spectrum_mcmc = np.median(cnu_mcmc) + (xg + xg_std) * np.median(snu_mcmc)
        gal_spectrum_mcmc = np.median(cnu_mcmc) + (xg + xg_std) * np.median(snu_mcmc)
        gal_spectrum.append(gal_spectrum_mcmc.mean())
        gal_spectrum_err.append(gal_spectrum_mcmc.std())

        fnu_f_mcmc = snu_mcmc * (np.min(norm_lc[1]) - xg)
        fnu_b_mcmc = snu_mcmc * (np.max(norm_lc[1]) - xg)
        fnu_f.append(fnu_f_mcmc.mean());   fnu_f_err.append(fnu_f_mcmc.std())
        fnu_b.append(fnu_b_mcmc.mean());   fnu_b_err.append(fnu_b_mcmc.std())

        slope.append(np.median(snu_mcmc))
        slope_err.append(np.std(snu_mcmc))

        # 拟合直线
        lin_fit = np.median(snu_mcmc) * xx + np.median(cnu_mcmc)

        # 光度单位转换
        if wavelengths is not None:
            wave_i = wavelengths[i] * u.Angstrom
            if (input_units != 'flam') and (output_units != 'flam'):
                dd = funits
                fac_flux[i] = dd.cgs.to(output_units).value if output_units != 'fnu' \
                            else dd.cgs.to('erg s^-1 cm^-2 Hz^-1').value
            elif (input_units != 'flam') and (output_units == 'flam'):
                dd = funits / (wave_i**2) * ct.c
                fac_flux[i] = dd.cgs.to('erg s^-1 cm^-2 Angstrom^-1').value / 1e-15
            elif (input_units == 'flam') and (output_units != 'flam'):
                dd = funits / ct.c * (wave_i**2)
                fac_flux[i] = dd.cgs.to(output_units).value if output_units != 'fnu' \
                            else dd.cgs.to('erg s^-1 cm^-2 Hz^-1').value

        # 绘制 Flux-Flux 图
        plt.fill_between(xx,
                        (mc_pl.mean(axis=0) + mc_pl.std(axis=0)) * fac_flux[i],
                        (mc_pl.mean(axis=0) - mc_pl.std(axis=0)) * fac_flux[i],
                        color=band_colors[i], alpha=0.3)

        interp_xt = np.interp(data[:, 0], norm_lc[0], norm_lc[1])
        plt.errorbar(interp_xt, data[:, 1] * fac_flux[i],
                    yerr=np.sqrt(data[:, 2]**2 + sig**2) * fac_flux[i],
                    color=band_colors[i], ls='None', alpha=0.8)

        plt.plot(xx, lin_fit * fac_flux[i], color=band_colors[i], lw=3)
        max_flux = max(max_flux, np.max(data[:, 1] * fac_flux[i]))

    fnu_f = np.array(fnu_f)
    fnu_f_err = np.array(fnu_f_err)
    fnu_b = np.array(fnu_b)
    fnu_b_err = np.array(fnu_b_err)
    slope = np.array(slope)
    slope_err = np.array(slope_err)
    gal_spectrum = np.array(gal_spectrum)
    gal_spectrum_err = np.array(gal_spectrum_err)

    plt.axvline(x=np.median(x_gal_mcmc + x_gal_mcmc.std()), color='r',
                linestyle='-.', label=r'Galaxy')
    plt.axvline(x=np.min(norm_lc[1]), color='k',
                linestyle='--', label=r'F$_{\rm faint}$')
    plt.axvline(x=np.max(norm_lc[1]), color='grey',
                linestyle='--', label=r'F$_{\rm bright}$')

    lg = plt.legend(ncol=4)
    plt.xlim(x_gal - 1, 3)
    plt.ylim(-0.04 * fac_flux[-1], max_flux * 1.2)

    if limits != None:
        plt.ylim(-0.04, limits[1])

    plt.xlabel(r'$X_0 (t)$, Normalised driving light curve flux')
    plt.ylabel(ylab)
    plt.tight_layout()

    if savefig:
        plt.savefig(outputdir/f'{objName}_fluxflux.pdf')

    # 绘制 SED 图
    if wavelengths != None:
        wave = np.array(wavelengths)
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111)
        xxx = np.arange(2000, 9300)

        # 绘制 AGN 变异范围
        plt.fill_between(wave / (1 + redshift),
                        (np.array(unred(wave, fnu_b, ebv))) * fac_flux,
                        (np.array(unred(wave, fnu_f, ebv))) * fac_flux,
                        color='k', alpha=0.1, label='AGN variability')

        # 绘制 F_bright - F_faint
        plt.errorbar(wave / (1 + redshift), (np.array(unred(wave, fnu_b, ebv)) -
                                            np.array(unred(wave, fnu_f, ebv))) * fac_flux,
                    yerr=np.sqrt((np.array(fnu_f_err)) ** 2 + (np.array(fnu_b_err)) ** 2) * fac_flux,
                    marker='.', linestyle='-', color='k',
                    label=r'F$_{\rm bright}$ - F$_{\rm faint}$', ms=15)

        # 绘制 AGN RMS
        plt.errorbar(wave / (1 + redshift), np.array(unred(wave, slope, ebv)) * fac_flux,
                    yerr=0, marker='o', linestyle='--', color='grey', label='AGN RMS')

        # 绘制银河光谱
        plt.errorbar(wave / (1 + redshift), unred(wave, gal_spectrum, ebv) * fac_flux,
                    yerr=gal_spectrum_err * fac_flux,
                    marker='s', color='r', label='Galaxy', linestyle='-.')

        plt.xscale('log')
        plt.yscale('log')
        plt.xlim(np.min(wave / (1 + redshift)) - 100, np.max(wave / (1 + redshift)) + 100)
        plt.ylim(np.min(np.array(unred(wave, slope, ebv))) * 0.7 * fac_flux[-1], max_flux * 1.2)

        if limits != None:
            plt.ylim(limits[0], limits[1])

        lg = plt.legend(ncol=2)
        if redshift > 0:
            plt.xlabel(r'Rest Wavelength / $\mathrm{\AA}$')
        else:
            plt.xlabel(r'Observed Wavelength / $\mathrm{\AA}$')

        plt.ylabel(ylab)

        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
        ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f'))
        ax.xaxis.set_minor_formatter(mtick.FormatStrFormatter('%.0f'))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(2000))
        plt.tight_layout()

        if savefig:
            plt.savefig(outputdir/f'{objName}_SED.pdf')

        # === Power-law fit: Y = (lambda_rest) * (F_bright - F_faint) vs X = lambda_rest ===
        # 构造 X, Y 与误差
        x_rest = wave / (1.0 + redshift)
        F_bright_minus_faint = (np.array(unred(wave, fnu_b, ebv)) -
                                np.array(unred(wave, fnu_f, ebv))) * fac_flux
        F_bright_minus_faint_err = np.sqrt((np.array(fnu_b_err))**2 +
                                           (np.array(fnu_f_err))**2) * fac_flux

        Y = x_rest * F_bright_minus_faint
        Y_err = x_rest * F_bright_minus_faint_err  # 假设 x 无误差

        # 只保留可用于拟合的点（正值、有限、非零误差）
        m = np.isfinite(x_rest) & np.isfinite(Y) & np.isfinite(Y_err) & \
            (x_rest > 0) & (Y > 0) & (Y_err > 0)

        if np.count_nonzero(m) >= 3:
            # 幂律模型：Y = A * X^beta
            def powerlaw(x, A, beta):
                return A * x**beta

            # 使用 curve_fit 进行加权最小二乘拟合（线性空间）
            popt, pcov = opt.curve_fit(
                powerlaw,
                x_rest[m], Y[m],
                sigma=Y_err[m],
                absolute_sigma=True
            )

            A_best, beta_best = popt
            A_err, beta_err = np.sqrt(np.diag(pcov))  # 参数标准差，库直接给

            # ---- log10(A) 及其误差：尽量交给库做误差传播 ----
            try:
                from uncertainties import ufloat, umath
                A_u = ufloat(A_best, A_err)
                a_u = umath.log10(A_u)   # 自动误差传播
                a_best = a_u.n
                sa = a_u.s
            except ImportError:
                # 若没有 uncertainties，则用一行解析公式作为 fallback
                a_best = np.log10(A_best)
                sa = A_err / (A_best * np.log(10.0))

            sb = beta_err   # 斜率误差直接用 curve_fit 给的

            # 计算 χ²_red（在线性空间，带权重）
            res = (Y[m] - powerlaw(x_rest[m], A_best, beta_best)) / Y_err[m]
            chi2 = np.sum(res**2)
            dof = max(np.count_nonzero(m) - 2, 1)
            chi2_red = chi2 / dof

            expected = -4.0 / 3.0
            zscore = (beta_best - expected) / (sb if sb > 0 else np.inf)

            # 打印结果（格式保持不变，只是内部计算方式简化了）
            print("=== Power-law fit of Y = λ_rest * (F_bright - F_faint) vs X = λ_rest ===")
            print(f"N = {np.count_nonzero(m)} usable points")
            print(f"Slope β (expected -4/3 ≈ -1.3333): {beta_best:.4f} ± {sb:.4f}")
            print(f"Intercept a (log10 A): {a_best:.4f} ± {sa:.4f}  ->  A = {A_best:.4e}")
            print(f"Weighted χ²_red = {chi2_red:.3f}")
            print(f"Deviation from -4/3: {(beta_best - expected):.4f}  ({zscore:.2f} σ)")

            # 画图（errorbar 数据 + 拟合曲线），对数坐标
            fig_fit = plt.figure(figsize=(8, 6))
            ax_fit = fig_fit.add_subplot(111)
            ax_fit.errorbar(
                x_rest[m], Y[m], yerr=Y_err[m],
                fmt='o', ms=6, alpha=0.85,
                label=r'$\lambda_{\rm rest}\,[F_{\rm bright}-F_{\rm faint}]$'
            )

            xfit = np.logspace(np.log10(np.min(x_rest[m])),
                               np.log10(np.max(x_rest[m])), 256)
            yfit = powerlaw(xfit, A_best, beta_best)
            ax_fit.plot(
                xfit, yfit, '-', lw=2,
                label=fr'Fit: $Y = A\,X^{{\beta}}$,  $\beta={beta_best:.2f}$'
            )

            ax_fit.set_xscale('log')
            ax_fit.set_yscale('log')
            ax_fit.set_xlabel(
                r'Rest Wavelength / $\mathrm{\AA}$'
                if redshift > 0 else r'Observed Wavelength / $\mathrm{\AA}$'
            )
            ax_fit.set_ylabel(r'$\lambda_{\rm rest}\,[F_{\rm bright}-F_{\rm faint}]$')
            ax_fit.legend()
            ax_fit.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f'))
            ax_fit.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
            plt.tight_layout()

            if savefig:
                plt.savefig(outputdir/f'{objName}_powerlaw_fit.pdf')
        else:
            print(" [PyROA] Not enough valid points for log-log fit; skipped power-law fitting.")

    else:
        print(' [PyROA] No wavelength list. Skipping SED plot.')

    # 输出 fluxflux 数据
    d = {'wave': wave,
        'agn_b': np.array(fnu_b),
        'agn_b_err': np.array(fnu_b_err),
        'agn_f': np.array(fnu_f),
        'agn_f_err': np.array(fnu_f_err),
        'agn_rms': np.array(slope),
        'agn_rms_err': np.array(slope_err),
        'gal': np.array(gal_spectrum),
        'gal_err': np.array(gal_spectrum_err),
        'unred_agn_b': np.array(unred(wave, fnu_b, ebv)),
        'unred_agn_b_err': np.array(unred(wave, fnu_b_err, ebv)),
        'unred_agn_f': np.array(unred(wave, fnu_f, ebv)),
        'unred_agn_f_err': np.array(unred(wave, fnu_f_err, ebv)),
        'unred_agn_rms': np.array(unred(wave, slope, ebv)),
        'unred_agn_rms_err': np.array(unred(wave, slope_err, ebv)),
        'unred_gal': np.array(unred(wave, gal_spectrum, ebv)),
        'unred_gal_err': np.array(unred(wave, gal_spectrum_err, ebv))}
    
    # 假设 d['unred_agn_b'] 和 d['unred_agn_b_err'] 是数组或列表，逐一创建 ufloat 对象
    unred_agn_b = [ufloat(b, e) for b, e in zip(d['unred_agn_b'], d['unred_agn_b_err'])]
    unred_agn_F = [ufloat(f, e) for f, e in zip(d['unred_agn_f'], d['unred_agn_f_err'])]

    # 计算 epsilon（亮度比）并处理误差传播
    epsilon = [f / b for f, b in zip(unred_agn_F, unred_agn_b)]

    # 提取 epsilon 的标准值和误差
    epsilon_cern = [e.nominal_value for e in epsilon]
    epsilon_err = [e.std_dev for e in epsilon]

    # 将计算结果添加到字典
    d['epsilon'] = epsilon_cern
    d['epsilon_err'] = epsilon_err

    unred_agn_b_Jy = [flam_to_jy(b.nominal_value, wave[i]) for i, b in enumerate(unred_agn_b)]
    unred_agn_b_Jy_err = [flam_to_jy(b.std_dev, wave[i]) for i, b in enumerate(unred_agn_b)]
    d['unred_agn_b_Jy'] = unred_agn_b_Jy
    d['unred_agn_b_Jy_err'] = unred_agn_b_Jy_err
    df = DataFrame(data=d)
    
    df.to_csv(outputdir/f'{objName}_fluxflux.csv', index=False)


def Convergence(objname,
                outputdir='./',
                samples_file='samples_flat.obj',
                burnin=0,
                init_chain_length=100,
                savefig=True):

    # 用 Path 处理目录并确保存在
    outputdir = Path(outputdir)
    outputdir.mkdir(parents=True, exist_ok=True)
    
    # 读 samples
    samples_path = outputdir / samples_file
    with open(samples_path, 'rb') as f:
        samples = pickle.load(f)

    # 去掉 burnin
    chain = samples[burnin:, :]          # 形状 (Nsamples, Ndim)

    # 计算不同长度的链对应的自相关估计
    N = np.exp(
        np.linspace(np.log(init_chain_length),
                    np.log(chain.shape[0]), 10)
    ).astype(int)

    # 转置后再传给自相关函数（假设期望形状是 (Ndim, Nsamples)）
    chain_T = chain.T                    # 形状 (Ndim, Nsamples)

    gw2010 = np.empty(len(N))
    new = np.empty(len(N))
    for i, n in enumerate(N):
        gw2010[i] = autocorr_gw2010(chain_T[:, :n])
        new[i] = autocorr_new(chain_T[:, :n])

    fig = plt.figure(figsize=(8, 6))
    plt.loglog(N, gw2010, "o-", label="G&W 2010")
    plt.loglog(N, new, "o-", label="new")
    ylim = plt.gca().get_ylim()
    plt.plot(N, N / 50., "--k", label=r"$\tau = N/50$")
    plt.ylim(ylim)
    plt.xlabel("number of samples, $N$")
    plt.ylabel(r"$\tau$ estimates")
    plt.legend(fontsize=14)

    if savefig:
        outpath = outputdir / f'{objname}_pyroa_convergence.pdf'
        plt.savefig(outpath)


# Automated windowing procedure following Sokal (1989)
def auto_window(taus, c):
    m = np.arange(len(taus)) < c * taus
    if np.any(m):
        return np.argmin(m)
    return len(taus) - 1


# Following the suggestion from Goodman & Weare (2010)
def autocorr_gw2010(y, c=5.0):
    f = autocorr_func_1d(np.mean(y, axis=0))
    taus = 2.0 * np.cumsum(f) - 1.0
    window = auto_window(taus, c)
    return taus[window]


def autocorr_new(y, c=5.0):
    f = np.zeros(y.shape[1])
    for yy in y:
        f += autocorr_func_1d(yy)
    f /= len(y)
    taus = 2.0 * np.cumsum(f) - 1.0
    window = auto_window(taus, c)
    return taus[window]

def next_pow_two(n):
    i = 1
    while i < n:
        i = i << 1
    return i


def autocorr_func_1d(x, norm=True):
    x = np.atleast_1d(x)
    if len(x.shape) != 1:
        raise ValueError("invalid dimensions for 1D autocorrelation function")
    n = next_pow_two(len(x))

    # Compute the FFT and then (from that) the auto-correlation function
    f = np.fft.fft(x - np.mean(x), n=2 * n)
    acf = np.fft.ifft(f * np.conjugate(f))[: len(x)].real
    acf /= 4 * n

    # Optionally normalize
    if norm:
        acf /= acf[0]

    return acf

def unred(wave, flux, ebv, R_V=3.1, LMC2=False, AVGLMC=False):
    """
     Deredden a flux vector using the Fitzpatrick (1999) parameterization

     Parameters
     ----------
     wave :   array
              Wavelength in Angstrom
     flux :   array
              Calibrated flux vector, same number of elements as wave.
     ebv  :   float, optional
              Color excess E(B-V). If a negative ebv is supplied,
              then fluxes will be reddened rather than dereddened.
              The default is 3.1.
     AVGLMC : boolean
              If True, then the default fit parameters c1,c2,c3,c4,gamma,x0
              are set to the average values determined for reddening in the
              general Large Magellanic Cloud (LMC) field by
              Misselt et al. (1999, ApJ, 515, 128). The default is
              False.
     LMC2 :   boolean
              If True, the fit parameters are set to the values determined
              for the LMC2 field (including 30 Dor) by Misselt et al.
              Note that neither `AVGLMC` nor `LMC2` will alter the default value
              of R_V, which is poorly known for the LMC.

     Returns
     -------
     new_flux : array
                Dereddened flux vector, same units and number of elements
                as input flux.

     Notes
     -----

     .. note:: This function was ported from the IDL Astronomy User's Library.

     :IDL - Documentation:

      PURPOSE:
       Deredden a flux vector using the Fitzpatrick (1999) parameterization
      EXPLANATION:
       The R-dependent Galactic extinction curve is that of Fitzpatrick & Massa
       (Fitzpatrick, 1999, PASP, 111, 63; astro-ph/9809387 ).
       Parameterization is valid from the IR to the far-UV (3.5 microns to 0.1
       microns).    UV extinction curve is extrapolated down to 912 Angstroms.

      CALLING SEQUENCE:
        FM_UNRED, wave, flux, ebv, [ funred, R_V = , /LMC2, /AVGLMC, ExtCurve=
                          gamma =, x0=, c1=, c2=, c3=, c4= ]
      INPUT:
         WAVE - wavelength vector (Angstroms)
         FLUX - calibrated flux vector, same number of elements as WAVE
                  If only 3 parameters are supplied, then this vector will
                  updated on output to contain the dereddened flux.
         EBV  - color excess E(B-V), scalar.  If a negative EBV is supplied,
                  then fluxes will be reddened rather than dereddened.

      OUTPUT:
         FUNRED - unreddened flux vector, same units and number of elements
                  as FLUX

      OPTIONAL INPUT KEYWORDS
          R_V - scalar specifying the ratio of total to selective extinction
                   R(V) = A(V) / E(B - V).    If not specified, then R = 3.1
                   Extreme values of R(V) range from 2.3 to 5.3

       /AVGLMC - if set, then the default fit parameters c1,c2,c3,c4,gamma,x0
                 are set to the average values determined for reddening in the
                 general Large Magellanic Cloud (LMC) field by Misselt et al.
                 (1999, ApJ, 515, 128)
        /LMC2 - if set, then the fit parameters are set to the values determined
                 for the LMC2 field (including 30 Dor) by Misselt et al.
                 Note that neither /AVGLMC or /LMC2 will alter the default value
                 of R_V which is poorly known for the LMC.

         The following five input keyword parameters allow the user to customize
         the adopted extinction curve.    For example, see Clayton et al. (2003,
         ApJ, 588, 871) for examples of these parameters in different interstellar
         environments.

         x0 - Centroid of 2200 A bump in microns (default = 4.596)
         gamma - Width of 2200 A bump in microns (default  =0.99)
         c3 - Strength of the 2200 A bump (default = 3.23)
         c4 - FUV curvature (default = 0.41)
         c2 - Slope of the linear UV extinction component
              (default = -0.824 + 4.717/R)
         c1 - Intercept of the linear UV extinction component
              (default = 2.030 - 3.007*c2
    """

    x = 10000./ wave # Convert to inverse microns
    curve = x*0.

    # Set some standard values:
    x0 = 4.596
    gamma =  0.99
    c3 =  3.23
    c4 =  0.41
    c2 = -0.824 + 4.717/R_V
    c1 =  2.030 - 3.007*c2

    if LMC2:
        x0    =  4.626
        gamma =  1.05
        c4   =  0.42
        c3    =  1.92
        c2    = 1.31
        c1    =  -2.16
    elif AVGLMC:
        x0 = 4.596
        gamma = 0.91
        c4   =  0.64
        c3    =  2.73
        c2    = 1.11
        c1    =  -1.28

    # Compute UV portion of A(lambda)/E(B-V) curve using FM fitting function and
    # R-dependent coefficients
    xcutuv = np.array([10000.0/2700.0])
    xspluv = 10000.0/np.array([2700.0,2600.0])

    iuv = np.where(x >= xcutuv)[0]
    N_UV = len(iuv)
    iopir = np.where(x < xcutuv)[0]
    Nopir = len(iopir)
    if (N_UV > 0): xuv = np.concatenate((xspluv,x[iuv]))
    else:  xuv = xspluv

    yuv = c1  + c2*xuv
    yuv = yuv + c3*xuv**2/((xuv**2-x0**2)**2 +(xuv*gamma)**2)
    yuv = yuv + c4*(0.5392*(np.maximum(xuv,5.9)-5.9)**2+0.05644*(np.maximum(xuv,5.9)-5.9)**3)
    yuv = yuv + R_V
    yspluv  = yuv[0:2]  # save spline points

    if (N_UV > 0): curve[iuv] = yuv[2::] # remove spline points

    # Compute optical portion of A(lambda)/E(B-V) curve
    # using cubic spline anchored in UV, optical, and IR
    xsplopir = np.concatenate(([0],10000.0/np.array([26500.0,12200.0,6000.0,5470.0,4670.0,4110.0])))
    ysplir   = np.array([0.0,0.26469,0.82925])*R_V/3.1
    ysplop   = np.array((np.polyval([-4.22809e-01, 1.00270, 2.13572e-04][::-1],R_V ),
            np.polyval([-5.13540e-02, 1.00216, -7.35778e-05][::-1],R_V ),
            np.polyval([ 7.00127e-01, 1.00184, -3.32598e-05][::-1],R_V ),
            np.polyval([ 1.19456, 1.01707, -5.46959e-03, 7.97809e-04, -4.45636e-05][::-1],R_V ) ))
    ysplopir = np.concatenate((ysplir,ysplop))

    if (Nopir > 0):
      tck = interpolate.splrep(np.concatenate((xsplopir,xspluv)),np.concatenate((ysplopir,yspluv)),s=0)
      curve[iopir] = interpolate.splev(x[iopir], tck)

    #Now apply extinction correction to input flux vector
    curve *= ebv

    return flux * 10.**(0.4*curve)

def flam_to_jy(flux, wavelength):
    c = 3e18  # 光速，单位为 Å/s
    jy_conversion_factor = 1e8
    jy_flux = flux * (wavelength ** 2) / c * jy_conversion_factor
    
    return jy_flux