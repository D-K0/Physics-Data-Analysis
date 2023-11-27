import numpy as np
inf = np.inf


# BEST FIT:
def getBestFit(x_raw, y_raw, y_unc, model_func, param_guess=None, ignore_unc=False, bounds=(-inf, inf)):
    def best_fit_param(x_raw, y_raw, y_unc, model_func, param_guess=None, ignore_unc = False, bounds=(-inf, inf)):
        from scipy.optimize import curve_fit
        import numpy as np
        # p_opt, p_cov = curve_fit(model_func, x_raw, y_raw, sigma = y_unc, absolute_sigma = True, p0=param_guess, bounds = bounds)
        if ignore_unc:
            p_opt, p_cov = curve_fit(model_func, x_raw, y_raw, p0=param_guess, bounds=bounds)
        else:
            p_opt, p_cov = curve_fit(model_func, x_raw, y_raw, sigma = y_unc, absolute_sigma = True, p0=param_guess, bounds=bounds)
        param_unc = np.sqrt(np.diag(p_cov))
        return p_opt, param_unc
    param, param_unc = best_fit_param(x_raw, y_raw, y_unc, model_func,param_guess, ignore_unc, bounds)
    y_fit = model_func(x_raw, *param)
    return y_fit, param, param_unc

# CHI SQUARE
def getRedChiSqr(y_raw, y_fit, y_unc, n_parameters):
    import numpy as np
    dof = len(y_raw) - n_parameters
    red_chi_sqr = (1 / dof) * np.sum(((y_raw - y_fit) / y_unc)**2)
    print("Chi square = ", red_chi_sqr)

# RESIDUALS
def getRes(y_raw, y_fit):
    return y_raw - y_fit

# CONTROL FUNCTION
def analysis(x_raw, y_raw, y_unc, model_func, param_guess=None, ignore_unc=False, bounds=(-inf, inf)):
    y_fit, param, param_unc = getBestFit(x_raw, y_raw, y_unc, model_func, param_guess, ignore_unc, bounds)
    getRedChiSqr(y_raw, y_fit, y_unc, len(param))
    y_res = getRes(y_raw, y_fit)
    return y_fit, y_res, param, param_unc



# ploting portion

# SETUP
def plot_settings(x_label, y_label, 
                  seperate=True, 
                  x_min=None, x_max=None):
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'serif'})
    plt.xlabel(x_label, fontsize = 36, fontweight = "bold")
    plt.ylabel(y_label, fontsize = 36, fontweight = "bold")
    plt.tick_params(axis = 'both', which = 'major', length = 10, width = 4)
    plt.tick_params(axis = 'both', which = 'minor', length = 6, width = 2.5)
    plt.xticks(fontsize = 36)
    plt.yticks(fontsize = 36)
    plt.legend(loc = "best", fontsize = 32)
    plt.xlim(x_min, x_max)
    plt.grid()
    if seperate:
        plt.show()

def norm(arr):
    '''Takes either 1d or 2d array and makes it 2d so that the for loop can traverse single sets of data'''
    import numpy as np
    if not isinstance(arr, (list, np.ndarray)):
        raise ValueError("not a list")
    elif not isinstance(arr[0], (list, np.ndarray)):
        return [arr]
    elif not isinstance(arr[0][0], (list, np.ndarray)):
        return arr
    elif isinstance(arr[0][0], (list, np.ndarray)):
        raise ValueError("3d+ list")
    
def normalize_input(x_raw=None, y_raw=None, 
                    x_unc=None, y_unc=None, 
                    y_fits=None, y_res=None):
    if x_raw is not None:
        x_raw = norm(x_raw)
    elif y_raw is not None:
        y_raw = norm(y_raw)
    elif x_unc is not None:
        x_unc = norm(x_unc)
    elif y_unc is not None:
        y_unc = norm(y_unc)
    elif y_fits is not None:
        y_fits = norm(y_fits)
    elif y_res is not None:
        y_res = norm(y_res)
    return x_raw, y_raw, x_unc, y_unc, y_fits, y_res
    
# BROKEN DOWN PLOTS
def plot_horizontal(y_hori, color = "black", linestyle='-', linewidth = 4, label = None):
    # >>> plot_horizontal(y_hori=[0, .0002], color=["black", "black"], linestyle=['-', '-'], linewidth=[4, 4])   
    import matplotlib.pyplot as plt
    import numpy as np
    if y_hori is None:
        return None
    elif not isinstance(y_hori, (list, np.ndarray)):
        if label is not None:
            plt.axhline(y_hori, color=color, linestyle=linestyle, linewidth=linewidth, label = label)
        else:
            plt.axhline(y_hori, color=color, linestyle=linestyle, linewidth=linewidth)    
    elif isinstance(y_hori[0], (list, np.ndarray)):
        raise ValueError("2d+ list in a list of horizontal line values")
    else:
        if label is not None:
            for i in range(len(y_hori)):
                plt.axhline(y_hori[i], color=color[i], linestyle=linestyle[i], linewidth=linewidth[i], label = label[i])
        else:
            for i in range(len(y_hori)):
                plt.axhline(y_hori[i], color=color[i], linestyle=linestyle[i], linewidth=linewidth[i])

def plot_vertical(x_vert, color = "black", linestyle='-', linewidth = 4, label = None):
    # plot_single_horizontal([0, .0002], ["black", "black"], ['-', '-'], [4, 4])    
    import matplotlib.pyplot as plt
    import numpy as np
    if x_vert is None:
        return None
    elif not isinstance(x_vert, (list, np.ndarray)):
        if label is not None:
            plt.axvline(x_vert, color=color, linestyle=linestyle, linewidth=linewidth, label = label)
        else: 
            plt.axvline(x_vert, color=color, linestyle=linestyle, linewidth=linewidth)
    elif isinstance(x_vert[0], (list, np.ndarray)):
        raise ValueError("2d+ list in a list of horizontal line values")
    else:
        if label is not None:
            for i in range(len(x_vert)):
                plt.axvline(x_vert[i], color=color[i], linestyle=linestyle[i], linewidth=linewidth[i], label = label[i])
        else:
            for i in range(len(x_vert)):
                plt.axvline(x_vert[i], color=color[i], linestyle=linestyle[i], linewidth=linewidth[i])

def residual_one(x_raw, y_res, y_unc, 
                 x_axis='y axis[unit]', y_axis='Residuals [unit]', 
                 seperate=True, x_min=None, x_max=None):
    import matplotlib.pyplot as plt
    plot_horizontal(0)    
    plt.errorbar(x_raw, y_res, yerr=y_unc, fmt='x', color='black', ecolor='red', capsize=10, elinewidth=4, markersize=15, markeredgewidth=6)
    plot_settings(x_axis, y_axis, seperate=seperate, x_min=x_min, x_max=x_max)

def residual_several(x_raw, y_res, y_unc, 
                     x_axis='y axis[unit]', y_axis='Residuals [unit]', label=["res1", "res2", "res3"], 
                     colour_bar = ['#4FDFDF', '#9664AC', '#B0B06E'], colour_dot=['#008080', '#642C7C', '#808000'], 
                     seperate=True, x_min=None, x_max=None):
    import matplotlib.pyplot as plt
    plot_horizontal(0)    
    x_raw, _, _, y_unc, _, y_res = normalize_input(x_raw, None, None, y_unc, None, y_res)
    for i in range(len(y_res)):
        plt.errorbar(x_raw[i], y_res[i], yerr=y_unc[i], fmt='x', color=colour_dot[i], ecolor=colour_bar[i], capsize=10+i*3, elinewidth=4, markersize=15, markeredgewidth=6, label=label[i], alpha=.75)
    plot_settings(x_axis, y_axis, seperate=seperate, x_min=x_min, x_max=x_max)


def plot_raw_data(x_raw, y_raw, x_unc=None, y_unc=None, bar_color = ['#008080', '#642C7C', '#808000'], label = "Raw data"):
    import matplotlib.pyplot as plt
    if x_unc is None:
        x_unc = [None] * len(x_raw)
    if y_unc is None:
        y_unc = [None] * len(x_raw)
    for i in range(len(x_raw)):
        plt.errorbar(x_raw[i], y_raw[i], yerr=y_unc[i], xerr=x_unc[i], fmt='x', color='black', ecolor=bar_color[i], capsize=10, elinewidth=4, markersize=15, markeredgewidth=6, label=label,alpha=.75, zorder=0)
    
def plot_fit(x_raw, y_fits, format = ['-', '--', '--'], colour = ['#008080', '#642C7C', '#808000'], label=["fit1", "fit2", "fit3"]):
    import matplotlib.pyplot as plt
    if y_fits is not None:
        for i in range(len(y_fits)):
            plt.plot(x_raw[i], y_fits[i],format[i],color=colour[i],linewidth = 6, label=label[i])

def plot_data_only(x_raw, y_raw, x_unc=None, y_unc=None, y_fits=None, 
                   x_axis='x axis[unit]', y_axis='y axis[unit]', label=["fit1", "fit2", "fit3"], label_raw = "Raw data",
                   fit_colour = ['#008080', '#642C7C', '#808000'], bar_color=['#4FDFDF', '#9664AC', '#B0B06E'],
                   x_vert = None, vert_color = "black", vert_linestyle='-', vert_linewidth = 4, vert_label = None,
                   y_hori = None, hori_color = "black", hori_linestyle='-', hori_linewidth = 4, hori_label = None,
                   format = ['-', '--', '--'], seperate=True, x_log = False, y_log=False, x_min=None, x_max=None):
    import matplotlib.pyplot as plt
    if seperate:
        plt.figure(figsize=(20, 12))
    else:
        x_axis = None
    x_raw, y_raw, x_unc, y_unc, y_fits, y_res = normalize_input(x_raw, y_raw, x_unc, y_unc, y_fits, None)
    plot_raw_data(x_raw, y_raw, x_unc, y_unc, bar_color, label_raw)
    plot_fit(x_raw, y_fits, format, fit_colour, label)
    plot_vertical(x_vert, vert_color, vert_linestyle, vert_linewidth, vert_label)
    plot_horizontal(y_hori, hori_color, hori_linestyle, hori_linewidth, hori_label)
    if x_log:
        plt.xscale('log') 
    if y_log:
        plt.yscale('log') 
    plot_settings(x_axis, y_axis, seperate=seperate, x_min=x_min, x_max=x_max)

# COMBINED PLOTS
def combined_figures(x_raw, y_raw, x_unc=None, y_unc=None, y_fits=None, y_res=None, 
                     x_axis='x axis[unit]',
                     y_axis='y axis[unit]',
                     res_axis='res [unit]',
                     res_labels=["res1", "res2", "res3"],
                     fit_labels=["fit1", "fit2", "fit3"],
                     fit_colour=['#008080', '#642C7C', '#808000'],
                     bar_color=['#4FDFDF', '#9664AC', '#B0B06E'],
                     fit_line_format=['-', '--', '--'],
                     x_vert = None, vert_color = "black", vert_linestyle='-', 
                     vert_linewidth = 4, vert_label = None,
                     y_hori = None, hori_color = "black", hori_linestyle='-', 
                     hori_linewidth = 4, hori_label = None,
                     x_log = False, y_log=False, x_min=None, x_max=None): 
    import matplotlib.pyplot as plt
    plt.figure(figsize=(20, 12)) 
    if y_res is not None:
        plt.subplots(2, 1, figsize=(20, 20), gridspec_kw={'height_ratios': [0.7, 0.3]})
        # residual plot
        plt.subplot(2, 1, 2)  
        residual_several(x_raw, y_res, y_unc, x_axis=x_axis, y_axis=res_axis, label=res_labels, colour_bar = bar_color, colour_dot=fit_colour, seperate=False, x_min=x_min, x_max=x_max)
        # data plot
        plt.subplot(2, 1, 1)  
    plot_data_only(x_raw, y_raw, x_unc, y_unc, y_fits, x_axis=x_axis, y_axis=y_axis, label=fit_labels, fit_colour = fit_colour, bar_color=bar_color, 
                   x_vert = x_vert, vert_color = vert_color, vert_linestyle=vert_linestyle, vert_linewidth = vert_linewidth, vert_label = vert_label,
                   y_hori = y_hori, hori_color = hori_color, hori_linestyle=hori_linestyle, hori_linewidth = hori_linewidth, hori_label = hori_label,
                   format = fit_line_format, seperate=False, x_log = x_log, y_log=y_log, x_min=x_min, x_max=x_max)
    # set up
    plt.tight_layout()
    plt.show()


# HISTOGRAM
def getPoisson(x, mu):
    from scipy.stats import poisson
    poisson_pmf = poisson.pmf(x, mu) 
    return poisson_pmf
def getGaussian(x, mu, sigma):
    from scipy.stats import norm
    return norm.pdf(x, mu, sigma) 

def plotOneHistogram(data,n_bin=None, labels=["Histogram", "Poisson", "Guassian"], 
                     color = ['#008080', '#642C7C', '#808000'], x_label = "x_axis", y_label="Normalized Frequency",
                     normalization_extra_G = 1, normalization_extra_P = 1, x_min=None, x_max=None):
    import matplotlib.pyplot as plt
    import numpy as np
    if n_bin is None:
        n_bin = int(np.sqrt(len(data)))
    mu = np.mean(data)
    print("mu = ", mu)
    sigma = np.sqrt(mu)
    print("sigma = ", sigma)
    x = np.arange(0, np.max(data) + 1)
    poisson = getPoisson(x, mu) * normalization_extra_P
    gaussian = getGaussian(x, mu, sigma) * normalization_extra_G
    plt.figure(figsize=(20, 12))
    plt.hist(data, bins=n_bin, alpha=0.7, color=color[0],edgecolor='black', density=True, label=labels[0], zorder=0)
    plt.plot(x, poisson, '-', color = color[1], label=labels[1], lw=5, zorder=1)
    plt.plot(x, gaussian, '--', color = color[2], label=labels[2], lw=4, zorder=2)
    plot_settings(x_label, y_label, x_min=x_min, x_max=x_max)
