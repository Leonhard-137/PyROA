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
from scipy import optimize
from uncertainties import ufloat, umath
import astropy.constants as ct
from pathlib import Path
from dust_extinction.parameter_averages import F99  # 银河系模型 (Fitzpatrick 99)
from dust_extinction.averages import G03_SMCBar     # SMC模型 (AGN核区常用)
import warnings
import pandas as pd

warnings.filterwarnings('ignore')

def Chains(nparam, objname, filters, delay_ref,
           burnin=0, samples_file='samples_flat.obj',
           outputdir='./', initial=0,
           savefig=True, figname=None):
    
    """Parameter Chain Plot of MCMC from PyROA outputs
	nparam : str
		Parameters to show in the corner plot. Can choose
		individual ones such as: 'A','A','tau','sig'
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
        for j in ["R", "A", r"$\tau$", r"$\sigma$"]:
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

    elif nparam in ('tau', 'R', 'A', 'sig'):
        if nparam == 'R':
            shifter = 0
        elif nparam == 'A':
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

    # 用 Path 处理目录，并保证存在
    outputdir = Path(outputdir)
    outputdir.mkdir(parents=True, exist_ok=True)

    # 读取 samples
    samples_path = outputdir / samples_file
    with open(samples_path, 'rb') as f:
        # 【修改点 1】：增加 ::10 进行稀疏化，防止 MemoryError
        # 如果还是爆内存，可以改成 ::50 或 ::100
        samples = pickle.load(f)[burnin::10]

    ss = np.where(np.array(filters) == delay_ref)[0][0]

    labels = []
    for i in range(len(filters)):
        for j in ["R", "A", r"$\tau$", r"$\sigma$"]:
            labels.append(j + r'$_{' + filters[i] + r'}$')
    labels.append(r'$\Delta$')
    
    # 【修改点 2】：修复标签错位 BUG
    # 必须先删除参考波段的 tau，再复制给 all_labels
    del labels[ss * 4 + 2]     # 先删
    all_labels = labels.copy() # 后复制 (或者直接用 labels)

    # 只画某一类参数：R / A / tau / sig
    if nparam in ('tau', 'R', 'A', 'sig'):
        if nparam == 'R':
            shifter = 0
        elif nparam == 'A':
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
            labels=np.array(labels)[list_only], # 注意：这里实际上labels已经是删减过的了，逻辑可能需要根据你之前的bug修复微调，但通常画单类参数不受那个bug影响太大，主要是画all的时候
            title_kwargs={'fontsize': 19}
        )

    # 画所有参数
    elif nparam == 'all':
        gg = corner.corner(
            samples,
            show_titles=True,
            labels=all_labels, # 这里现在是对的了
            title_kwargs={'fontsize': 19},
            # 额外保险：不画散点，只画等高线，进一步省内存
            plot_datapoints=False, 
            fill_contours=True
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
        for j in ["R", "A", r"$\tau$", r"$\sigma$"]:
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

def Lightcurves(objname, filters, delay_ref,
                lc_file="Lightcurve_models.obj",
                samples_file='samples_flat.obj',
                slow_comp_file='Slow_Comps.obj',
                outputdir='./', datadir='./',
                burnin=0, band_colors=None,
                limits=None, grid=False, grid_step=5.0,
                show_delay_ref=False, ylab=None,
                filter_labels=None, savefig=True, figname=None,
                include_slow_comp=False, slow_comp_delta=30.0):

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

    # 按 R, A, tau, sig 切 chunk
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
        data_file = datadir / f"{objname}_{filters[i]}.dat"
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

            axs[0][0].set_title(objname)
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
        outpath = outputdir/f"{objname}_pyroa_lightcurves.pdf"
        plt.savefig(outpath)

def extract_flux_components(objname, filters, delay_ref, gal_ref, wavelengths,
                            lc_file="Lightcurve_models.obj",
                            samples_file='samples_flat.obj',
                            xt_file='X_t.obj',
                            outputdir='./', datadir='./',
                            burnin=0, band_colors=None,
                            input_units='mJy', output_units='flam',
                            ebv_galactic=0.0,
                            savefig=True):
    """
    Step 1 (Complete): 提取光变分量并绘制完整的 Flux-Xt 相关图 (Overlaid Version)。
    
    修改说明：
    1. 绘图改为单图叠加模式 (Single Axes Overlay)。
    2. 散点颜色与拟合线颜色保持一致。
    3. 参考线 (Galaxy/Faint/Bright) 统一绘制，图例样式复刻 Reference 图。
    4. 保持所有输出变量名和 CSV Header 不变。
    """
    outputdir = Path(outputdir)
    datadir = Path(datadir)
    
    # --- 1. 内部辅助函数：银河系去红化 (MW F99) ---
    def unred_mw(flux, wave, ebv_val):
        if ebv_val == 0: return flux
        ext = F99(Rv=3.1)
        k_lambda = ext(wave * u.Angstrom)
        return flux * 10**(0.4 * k_lambda * ebv_val)

    # --- 2. 加载 PyROA 结果 ---
    try:
        with open(outputdir / samples_file, 'rb') as f:
            samples_flat = pickle.load(f)
        samples_flat = samples_flat[burnin:, :]
        
        with open(outputdir / xt_file, 'rb') as f:
            norm_lc = pickle.load(f) # 格式 [t, X, err]
            
        with open(outputdir / lc_file, 'rb') as f:
            lc_models = pickle.load(f)
    except Exception as e:
        print(f"Error loading PyROA results: {e}")
        return None

    # --- 3. 参数切片与参考点计算 ---
    ss = np.where(np.array(filters) == delay_ref)[0][0]
    chunk_size = 4 # R, A, tau, sig
    transpose_samples = np.transpose(samples_flat)
    transpose_samples = np.insert(transpose_samples, [ss*4+2], np.array([0.0]*len(transpose_samples[1])), axis=0)
    samples_chunks = [transpose_samples[i:i + chunk_size] for i in range(0, len(transpose_samples), chunk_size)]

    # 计算全局基准点 x_gal
    j_gal = filters.index(gal_ref)
    snu_gal = samples_chunks[j_gal][0] 
    cnu_gal = samples_chunks[j_gal][1] 
    with np.errstate(divide='ignore', invalid='ignore'):
        x_gal_mcmc = -cnu_gal / snu_gal
    x_gal = np.median(x_gal_mcmc[np.isfinite(x_gal_mcmc)])
    
    X_min = np.min(norm_lc[1])
    X_max = np.max(norm_lc[1])

    # --- 4. 容器与绘图初始化 (修改为单图) ---
    results = []
    if band_colors is None:
        band_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                       '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    # 修改点：创建一个单一的大图，而不是 subplots 网格
    fig, ax = plt.subplots(figsize=(10, 7))

    # --- 5. 主循环：逐波段处理 ---
    for i, flt in enumerate(filters):
        wave_i = wavelengths[i]
        color = band_colors[i % len(band_colors)]
        
        # A. 单位转换系数
        # 注意：这里保持 scale_val = 1，如果数据需要从 1e-15 变到 1，请将此处改为 1e-15
        scale_val = 1.0 
        
        fac_flux = 1.0
        if input_units == 'mJy' and output_units == 'flam':
            f_nu_unit = 1e-23 * u.erg / u.s / (u.cm**2) / u.Hz 
            w_unit = wave_i * u.Angstrom
            f_lam_equiv = f_nu_unit.to(u.erg / u.s / (u.cm**2) / u.Angstrom, 
                                       equivalencies=u.spectral_density(w_unit))
            fac_flux = f_lam_equiv.value / scale_val
        elif input_units == 'flam' and output_units == 'flam':
            fac_flux = 1.0 / scale_val

        # B. 提取分量
        snu_mcmc = samples_chunks[i][0] 
        cnu_mcmc = samples_chunks[i][1] 
        tau_mcmc = samples_chunks[i][2] 
        sig_mcmc = samples_chunks[i][3] 
        
        R_obs = np.median(snu_mcmc) * fac_flux
        R_err = np.std(snu_mcmc) * fac_flux
        A_obs = np.median(cnu_mcmc) * fac_flux
        A_err = np.std(cnu_mcmc) * fac_flux
        tau_val = np.median(tau_mcmc)
        tau_err = np.std(tau_mcmc)
        sig_val = np.median(sig_mcmc)
        sig_err = np.std(sig_mcmc)

        # C. 计算亮暗态与宿主星系流量
        f_bright_dist = snu_mcmc * (X_max - x_gal) * fac_flux
        f_faint_dist  = snu_mcmc * (X_min - x_gal) * fac_flux
        f_gal_dist    = (cnu_mcmc + snu_mcmc * x_gal) * fac_flux
        
        epsilon = (X_min - x_gal) / (X_max - x_gal)

        # D. 银河系消光修正
        mw_corr = unred_mw(1.0, wave_i, ebv_galactic)
        
        R_unred = R_obs * mw_corr
        A_unred = A_obs * mw_corr
        F_b_unred = np.median(f_bright_dist) * mw_corr
        F_b_err_unred = np.std(f_bright_dist) * mw_corr
        F_f_unred = np.median(f_faint_dist) * mw_corr
        F_f_err_unred = np.std(f_faint_dist) * mw_corr
        Gal_unred = np.median(f_gal_dist) * mw_corr
        Gal_unred_err = np.std(f_gal_dist) * mw_corr

        # E. 收集数据 (保持 Key 不变)
        results.append({
            'Filter': flt, 'Wavelength': wave_i,
            'R_obs': R_obs, 'R_obs_err': R_err,
            'A_obs': A_obs, 'A_obs_err': A_err,
            'R_unred_gal': R_unred, 'R_unred_gal_err': R_err * mw_corr,
            'A_unred_gal': A_unred, 'A_unred_gal_err': A_err* mw_corr,
            'F_unred_gal_bright': F_b_unred, 'F_unred_gal_bright_err':F_b_err_unred,
            'F_unred_gal_faint': F_f_unred, 'F_unred_gal_faint_err':F_f_err_unred,
            'Epsilon': epsilon, 'F_Galaxy': Gal_unred,
            'F_Galaxy_err': Gal_unred_err,
            'tau': tau_val, 'tau_err': tau_err,
            'sig': sig_val, 'sig_err': sig_err,
        })

        # F. 绘制 Flux-Xt 图 (叠加在同一个 ax 上)
        obs_data = np.loadtxt(datadir / f"{objname}_{flt}.dat")
        t_obs, f_obs, e_obs = obs_data[:, 0], obs_data[:, 1], obs_data[:, 2]
        
        # 时延修正
        X_at_t = np.interp(t_obs - tau_val, norm_lc[0], norm_lc[1])
        
        # 修改点：散点颜色改为 color (原为 'k')，使其与拟合线一致
        # alpha 设置稍低一点，避免点太密把线挡住
        ax.errorbar(X_at_t, f_obs * fac_flux, yerr=np.sqrt(e_obs**2 + sig_val**2) * fac_flux, 
                    fmt='.', color=color, alpha=0.3, zorder=1)
        
        # 拟合直线及误差带
        xx = np.linspace(x_gal - 1, X_max + 1, 100)
        mc_lines = np.zeros((200, len(xx)))
        for lo in range(200):
            idx = np.random.randint(0, len(snu_mcmc))
            mc_lines[lo] = (snu_mcmc[idx] * xx + cnu_mcmc[idx]) * fac_flux
        
        ax.fill_between(xx, np.mean(mc_lines, axis=0) - np.std(mc_lines, axis=0),
                        np.mean(mc_lines, axis=0) + np.std(mc_lines, axis=0), 
                        color=color, alpha=0.15, zorder=2)
        ax.plot(xx, (np.median(snu_mcmc) * xx + np.median(cnu_mcmc)) * fac_flux, 
                color=color, lw=2, label=f'{flt}', zorder=3)

    # --- 6. 绘制统一参考线与修饰 (Loop 外部) ---
    # 复刻第一张图的图例样式：只显示参考线的图例，不显示每个波段的图例(太乱)
    # 若需要波段图例，可去掉 label='_nolegend_' 并使用 ax.legend(ncol=2)
    
    l1 = ax.axvline(x=x_gal, color='red', ls='-.', alpha=0.8, lw=1.5, label='Galaxy')
    l2 = ax.axvline(x=X_min, color='black', ls='--', alpha=0.7, lw=1.5, label=r'F$_{\rm faint}$')
    l3 = ax.axvline(x=X_max, color='grey', ls='--', alpha=0.7, lw=1.5, label=r'F$_{\rm bright}$')

    ax.set_xlabel(r'$X_0(t)$, Normalised driving light curve flux', fontsize=14)
    # 修改点：强制指定单位标签，符合你的要求
    ax.set_ylabel(r'F$_{\lambda}$ / $\times 10^{-15} \mathrm{erg\ s^{-1}\ cm^{-2}\ \AA^{-1}}$', fontsize=14)
    
    # 设置刻度字体大小
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    # 放置图例：只放参考线，置于顶部中央 (根据参考图 1 风格)
    # handles=[l1, l2, l3] 确保只显示这三条线
    ax.legend(handles=[l1, l2, l3], loc='upper center', ncol=3, fontsize=13, frameon=True)

    # --- 7. 保存与输出 ---
    plt.tight_layout()
    if savefig:
        plt.savefig(outputdir / f"{objname}_flux_xt_correlation.pdf", dpi=300)
    plt.close()

    df = pd.DataFrame(results)
    df.to_csv(outputdir / f"{objname}_extracted_components.csv", index=False)
    print(f"\n[Step 1] Components saved to {objname}_extracted_components.csv ")
    
    return df

def calculate_nuclear_ebv(df, outputdir, objname):
    """
    Step 2: Calculate Nuclear E(B-V) using smart anchor pairs.
    
    Physics Assumptions:
    1. Intrinsic Spectrum: Standard Thin Disk f_lambda ~ lambda^(-7/3)
    2. Extinction Law: SMC Bar (Gordon et al. 2003) for AGN nucleus
    3. Input Data: Uses 'R_unred' (Slope corrected for Galactic extinction)
    """

    outputdir = Path(outputdir)
    filters = df['Filter'].tolist()
    
    # --- 1. 智能波段对选择策略 ---
    # 逻辑：优先选择波长跨度最大、且受 BLR (Small Blue Bump) 污染最小的组合
    pairs_to_try = [
        # Swift UVOT (最佳组合，W2/M2 在巴耳末跳变左侧远端，V 在右侧远端)
        ('W2', 'V'), ('M2', 'V'), ('W1', 'V'),
        # Ground based (U波段虽然有污染，但比B好一点点；B-V 是下策)
        ('U', 'V'), ('B', 'V'),
        # SDSS (g-i 跨度大且避开了 r 波段可能的 H-alpha)
        ('g', 'i'), ('g', 'r')
    ]
    
    target_blue, target_red = None, None
    for b, r in pairs_to_try:
        if b in filters and r in filters:
            target_blue, target_red = b, r
            break
            
    if not target_blue:
        print("[Step 2] Warning: No suitable filter pair found. Assuming Nuclear E(B-V)=0.")
        return ufloat(0.0, 0.0)

    print(f"\n[Step 2] Calculating Nuclear Extinction using pair: {target_blue} - {target_red}")

    # --- 2. 获取数据 ---
    # 注意：必须使用 'R_unred' (已去除银河系消光的光变幅值)
    row_b = df[df['Filter'] == target_blue].iloc[0]
    row_r = df[df['Filter'] == target_red].iloc[0]

    # 使用 uncertainties 处理误差
    R_blue = ufloat(row_b['R_unred_gal'], row_b['R_unred_gal_err'])
    R_red  = ufloat(row_r['R_unred_gal'], row_r['R_unred_gal_err'])
    
    lam_b = row_b['Wavelength']
    lam_r = row_r['Wavelength']

    # --- 3. 计算理论比值 (Intrinsic Ratio) ---
    # 假设标准吸积盘 f_lambda ∝ lambda^(-7/3)
    # Ratio = f_blue / f_red = (lam_blue / lam_red)^(-7/3) = (lam_red / lam_blue)^(7/3)
    ratio_theo = (lam_r / lam_b)**(7.0/3.0)
    
    # --- 4. 计算观测比值 (Observed Ratio) ---
    # 这里使用的是去除了银河系消光后的观测值
    ratio_obs = R_blue / R_red
    
    # --- 5. 计算该波段对的色余 E(pair) ---
    # 公式：-2.5 * log10(Obs / Theo)
    E_pair = -2.5 * umath.log10(ratio_obs / ratio_theo)
    
    # --- 6. 换算为标准核内 E(B-V) ---
    # 使用 SMC 消光曲线
    # k(lambda) = A_lambda / E(B-V)
    ext_model = G03_SMCBar()
    
    # dust_extinction 需要带单位的波长 (Angstrom)
    k_b = ext_model(lam_b * u.Angstrom)
    k_r = ext_model(lam_r * u.Angstrom)
    
    # 核心公式: E(pair) = A_blue - A_red = E(B-V) * (k_b - k_r)
    # 所以: E(B-V) = E(pair) / (k_b - k_r)
    # 核心公式: E(pair) = A_blue - A_red = E(B-V) * (k_b - k_r)
    conversion_factor = k_b - k_r
    
    # 计算 E(B-V)
    if conversion_factor == 0:
        ebv_val = ufloat(0.0, 0.0)
    else:
        ebv_val = E_pair / conversion_factor

    # ================= 核心修改开始 =================
    # 策略：处理非物理的负值
    # 如果标称值小于 0，这通常意味着测量误差或吸积盘本身很蓝
    # 在物理上尘埃不能是负的，所以我们将其截断为 0
    if ebv_val.nominal_value < 0:
        print(f"[Step 2 Info] Calculated E(B-V) is negative ({ebv_val.nominal_value:.5f}).")
        print(f"              This is consistent with 0 within errors or implies a very blue disk.")
        print(f"              -> Forcing Nuclear E(B-V) = 0.0 for subsequent analysis.")
        
        # 强制设为 0，但保留原始误差作为参考（或者误差也设为0，视你的需求而定）
        # 这里建议保留误差，表示“0 +/- 0.02”
        ebv_val = ufloat(0.0, ebv_val.std_dev)
    # ================= 核心修改结束 =================

    # --- 7. 输出与保存 ---
    # 保留两位有效数字的显示逻辑 (配合你上一个问题)
    def format_sci(val):
        """辅助函数：保留两位有效数字的科学计数法，或者直接保留3位小数"""
        if abs(val) < 1e-3:
            return f"{val:.2e}"
        else:
            return f"{val:.4f}"

    print(f"  Lambda {target_blue}: {lam_b:.0f} A, k(SMC)={k_b:.3f}")
    print(f"  Lambda {target_red}:  {lam_r:.0f} A, k(SMC)={k_r:.3f}")
    print(f"  Ratio Theo ({target_blue}/{target_red}): {ratio_theo:.4f}")
    print(f"  Ratio Obs  ({target_blue}/{target_red}): {ratio_obs:.4f}")
    print(f"  E({target_blue}-{target_red}): {E_pair:.4f}")
    
    # 打印最终结果（带误差）
    print(f"  ==> Nuclear E(B-V): {ebv_val.nominal_value:.4f} +/- {ebv_val.std_dev:.4f}\n")
    
    # 保存到 txt 文件
    with open(outputdir / f'{objname}_extinction.txt', 'w') as f:
        f.write(f"Object: {objname}\n")
        f.write(f"Anchor Pair: {target_blue}-{target_red}\n")
        # 写入文件时也可以做格式化
        f.write(f"Calculated Nuclear E(B-V): {ebv_val.nominal_value:.4f} +/- {ebv_val.std_dev:.4f}\n")
        if ebv_val.nominal_value == 0.0 and E_pair < 0:
             f.write(f"Note: Raw calculated value was negative ({E_pair:.4f}), forced to 0.0.\n")
        f.write(f"Method: Standard Disk (lambda^-7/3) + SMC Extinction\n")
        f.write(f"Input Data: Galactic-extinction-corrected Slope (R_unred)\n")
        
    return ebv_val


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize
from pathlib import Path
from astropy import units as u
from dust_extinction.parameter_averages import G03_SMCBar
from uncertainties import unumpy, ufloat

def analyze_sed_powerlaw(df, ebv_nuclear, outputdir, objname, redshift=0.0, savefig=True):
    """
    Step 3 (Full Reconstruction - Refined Version): 
    1. 使用 unumpy 自动处理包括核区消光误差在内的全量误差传递。
    2. 计算本质 (Intrinsic) 的亮态、暗态、差分光谱及其误差。
    3. 执行 lambda*F_lambda vs lambda 的 Power-law 拟合。
    4. 输出包含所有误差项的完整 CSV 和拟合诊断。
    """
    outputdir = Path(outputdir)
    
    # --- 1. 数据准备 (读取带误差的原始数据) ---
    # 将 df 转换为带误差的 unumpy 数组 (uarray)
    # 使用你提供的 CSV Title
    Fb_u = unumpy.uarray(df['F_unred_gal_bright'].values, df['F_unred_gal_bright_err'].values)
    Ff_u = unumpy.uarray(df['F_unred_gal_faint'].values, df['F_unred_gal_faint_err'].values)
    R_u = unumpy.uarray(df['R_unred_gal'].values, df['R_unred_gal_err'].values)
    Gal_u = unumpy.uarray(df['F_Galaxy'].values, df['F_Galaxy_err'].values)
    
    wave = df['Wavelength'].values
    wave_rest = wave / (1 + redshift)
    
    # 确保 ebv_nuclear 是 ufloat 类型
    if hasattr(ebv_nuclear, 'nominal_value'):
        enuc_u = ebv_nuclear
    else:
        enuc_u = ufloat(ebv_nuclear, 0.0)

    # --- 2. 应用核内消光修正 (SMC Bar Model) ---
    ext_model = G03_SMCBar()
    k_nuc = ext_model(wave * u.Angstrom)
    
    # 修正因子 C = 10^(0.4 * k * E_nuc)，这是一个带误差的数组
    corr_nuc_u = 10**(0.4 * k_nuc * enuc_u)
    
    # 得到本质 (Intrinsic) 物理量 (计算会自动处理误差传递)
    R_int_u = R_u * corr_nuc_u
    Fb_int_u = Fb_u * corr_nuc_u
    Ff_int_u = Ff_u * corr_nuc_u
    
    # 差分光谱 (Difference Spectrum) 和 Epsilon (Ratio)
    delta_F_u = Fb_int_u - Ff_int_u
    epsilon_int_u = Ff_int_u / Fb_int_u

    # --- 3. Power-law 拟合：lambda * F_lambda vs lambda ---
    # 提取数值用于 scipy 拟合
    X_fit = wave_rest
    Y_fit_all = wave_rest * delta_F_u
    Y_fit_val = unumpy.nominal_values(Y_fit_all)
    Y_fit_err = unumpy.std_devs(Y_fit_all)
    
    def pl_model(lam, norm, beta):
        return norm * (lam**beta)

    # 初始猜测：经过 V 波段点，斜率 -1.33
    idx_v = (np.abs(wave - 5500)).argmin()
    p0 = [Y_fit_val[idx_v] / (X_fit[idx_v]**-1.33), -1.33]

    try:
        popt, pcov = optimize.curve_fit(
            pl_model, X_fit, Y_fit_val, 
            sigma=Y_fit_err, p0=p0, absolute_sigma=True
        )
        beta_fit = popt[1]
        beta_err = np.sqrt(np.diag(pcov))[1]
        
        # 计算统计评价
        residuals = Y_fit_val - pl_model(X_fit, *popt)
        chi2 = np.sum((residuals / Y_fit_err)**2)
        dof = len(X_fit) - 2
        red_chi2 = chi2 / dof
        
        theory_beta = -4.0/3.0
        sigma_dev = np.abs(beta_fit - theory_beta) / beta_err
    except Exception as e:
        print(f"Power-law fit failed for {objname}: {e}")
        beta_fit, beta_err, red_chi2, sigma_dev = -999, 0, 0, 0

    # --- 4. 绘图 1: 全成分 SED 分解图 ---
    plt.rcParams.update({'font.family': 'serif', 'font.size': 12})
    fig1, ax1 = plt.subplots(figsize=(10, 7))
    
    # 提取中心值用于绘图
    f_b_val, f_f_val = unumpy.nominal_values(Fb_int_u), unumpy.nominal_values(Ff_int_u)
    df_val, df_err = unumpy.nominal_values(delta_F_u), unumpy.std_devs(delta_F_u)
    gal_val, gal_err = unumpy.nominal_values(Gal_u), unumpy.std_devs(Gal_u)
    r_val = unumpy.nominal_values(R_int_u)

    ax1.fill_between(wave_rest, f_f_val, f_b_val, color='gray', alpha=0.2, label='AGN Variability Range')
    ax1.errorbar(wave_rest, gal_val, yerr=gal_err, fmt='s--', color='red', ms=6, capsize=3, label='Host Galaxy (MW corr)')
    ax1.errorbar(wave_rest, df_val, yerr=df_err, fmt='o-', color='black', ms=8, lw=2, label=r'Difference Spectrum ($\Delta f_\lambda$)')
    ax1.plot(wave_rest, r_val, 'o:', color='gray', alpha=0.7, label='AGN RMS (R)')

    ax1.set_xscale('log'); ax1.set_yscale('log')
    ax1.set_xlabel(r'Rest Wavelength $\lambda_{rest}$ ($\AA$)')
    ax1.set_ylabel(r'Flux $f_\lambda$ ($10^{-15}$ erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$)')
    ax1.set_title(f"{objname} - Full SED Decomposition")
    ax1.legend(loc='best')
    
    if savefig:
        fig1.savefig(outputdir / f"{objname}_SED_Full_Decomposition.pdf", bbox_inches='tight')

    # --- 5. 绘图 2: Power-law 拟合验证图 ---
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.errorbar(X_fit, Y_fit_val, yerr=Y_fit_err, fmt='o', color='#1f77b4', label='De-reddened Data')
    
    x_range = np.logspace(np.log10(min(X_fit)*0.9), np.log10(max(X_fit)*1.1), 100)
    if beta_fit != -999:
        ax2.plot(x_range, pl_model(x_range, *popt), 'r-', lw=2, label=fr'Fit: $\beta = {beta_fit:.2f} \pm {beta_err:.2f}$')
    
    y_theory = Y_fit_val[idx_v] * (x_range / X_fit[idx_v])**(-1.333)
    ax2.plot(x_range, y_theory, 'k--', alpha=0.6, label=r'Theory: $\beta = -4/3$')

    ax2.set_xscale('log'); ax2.set_yscale('log')
    ax2.set_xlabel(r'Rest Wavelength $\lambda_{rest}$ ($\AA$)')
    ax2.set_ylabel(r'Flux $\lambda \cdot \Delta f_\lambda$')
    ax2.legend(loc='upper right')
    ax2.text(0.05, 0.05, f"$\chi^2/dof = {red_chi2:.2f}$\nDev. = ${sigma_dev:.1f}\sigma$", 
             transform=ax2.transAxes, bbox=dict(facecolor='white', alpha=0.7))
    
    if savefig:
        fig2.savefig(outputdir / f"{objname}_Powerlaw_Fit.pdf", bbox_inches='tight')
    plt.close('all')

    # --- 6. 结果导出 (完整包含中心值和误差) ---
    export_df = pd.DataFrame({
        'Filter': df['Filter'],
        'Wave_Rest': wave_rest,
        'R_intrinsic': unumpy.nominal_values(R_int_u),
        'R_intrinsic_err': unumpy.std_devs(R_int_u),
        'F_bright_int': unumpy.nominal_values(Fb_int_u),
        'F_bright_int_err': unumpy.std_devs(Fb_int_u),
        'F_faint_int': unumpy.nominal_values(Ff_int_u),
        'F_faint_int_err': unumpy.std_devs(Ff_int_u),
        'Delta_F_int': unumpy.nominal_values(delta_F_u),
        'Delta_F_int_err': unumpy.std_devs(delta_F_u),
        'Epsilon_int': unumpy.nominal_values(epsilon_int_u),
        'Epsilon_int_err': unumpy.std_devs(epsilon_int_u),
        'Galaxy_MW_only': unumpy.nominal_values(Gal_u),
        'Galaxy_MW_only_err': unumpy.std_devs(Gal_u)
    })
    
    csv_path = outputdir / f"{objname}_intrinsic_sed_components.csv"
    export_df.to_csv(csv_path, index=False)

    # Diagnostics TXT
    diag_path = outputdir / f"{objname}_fit_diagnostics.txt"
    with open(diag_path, 'w') as f:
        f.write(f"--- Fit Diagnostics for {objname} ---\n")
        f.write(f"Nuclear E(B-V) applied: {enuc_u:.5f}\n")
        f.write(f"Power-law Fit Beta: {beta_fit:.4f} +/- {beta_err:.4f}\n")
        f.write(f"Reduced Chi-squared: {red_chi2:.3f}\n")
        f.write(f"Deviation from -4/3 Theory: {sigma_dev:.2f} sigma\n")

    print(f"\n[Step 3] Analysis complete for {objname}.")
    print(f"  -> Beta = {beta_fit:.3f}, Chi2/dof = {red_chi2:.2f}")
    return beta_fit
    
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

def unred(wave, flux, ebv, model_type='MW', R_V=3.1):
    """
    Deredden flux using dust_extinction library.
    
    Parameters
    ----------
    wave : array
        Wavelength in Angstroms.
    flux : array
        Flux values.
    ebv : float
        Color excess E(B-V).
    model_type : str
        'MW' for Galactic extinction (Fitzpatrick 99, with UV bump).
        'SMC' for AGN intrinsic extinction (Gordon 03, no UV bump, steep UV rise).
    R_V : float
        Ratio of total to selective extinction (only for MW model). Default 3.1.
    """
    
    # 如果色余为 0，直接返回，节省计算
    if ebv == 0:
        return flux

    # 1. 选择消光模型
    if model_type == 'SMC':
        # SMC Bar 模型 (适合 AGN 核区)
        ext_model = G03_SMCBar()
    else:
        # 默认为银河系模型 (F99)
        ext_model = F99(Rv=R_V)

    # 2. 计算消光量 A_lambda = E(B-V) * k(lambda)
    # dust_extinction 库直接返回 A(lambda) / E(B-V) 的值吗？
    # 不，它的 extinguish 方法直接计算透射率 (transmissivity) = 10^(-0.4 * A_lambda)
    # 但我们需要支持负 EBV (加红) 或 正 EBV (去红)。
    
    # 我们可以手动计算: A_lambda = ext_model(wave) * ebv
    # ext_model(wave) 返回的是 k(lambda) = A(lambda) / E(B-V)
    # 注意：传入 wave 需要带单位
    
    try:
        # 获取 k(lambda) 值
        k_lambda = ext_model(wave * u.Angstrom)
        
        # 计算 A_lambda
        A_lambda = k_lambda * ebv
        
        # 3. 执行去红化
        # 公式: F_int = F_obs * 10^(0.4 * A_lambda)
        correction_factor = 10**(0.4 * A_lambda)
        
        return flux * correction_factor
        
    except Exception as e:
        print(f"Error in unred: {e}")
        return flux

def flam_to_jy(flux, wavelength):
    c = 3e18  # 光速，单位为 Å/s
    jy_conversion_factor = 1e8
    jy_flux = flux * (wavelength ** 2) / c * jy_conversion_factor
    return jy_flux