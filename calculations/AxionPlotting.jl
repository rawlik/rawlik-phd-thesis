# to supress strange warnings when importing matplotlib
using PyCall
@everywhere @PyCall.pyimport warnings
@everywhere warnings.filterwarnings("ignore")
using PyPlot
# @everywhere PyPlot.matplotlib[:style][:use]("rawlik")



function detectionreport(as::AxionSim, data, dataperiodogram, nullperiodograms;
                         freqs = [],
                         timeunit = DAY,
                         fitoffsets = false,
                         simple = false,
                         readyaxes = false,
                         amplitudeunits = false,
                         conversionfactor = 1,
                         label="data",
                         normalization=:noise)

    # change the normalization to be compatible with Nick
    nrm = if normalization == :noise
        # power is already defined with a noise normalization
        1
    elseif normalization == :signal
        # revert the normalization done when calculating power
        4 / length(freqs)
    else
        NaN
    end

    # define the conversion axis for the vertical axis
    cnv = amplitudeunits ? x -> conversionfactor * sqrt(x * nrm) : x -> conversionfactor^2 * x * nrm

    if !readyaxes
        if simple
            figure(figsize=(8,6))
        else
            figure(figsize=(10,10))
            subplot(221)
        end
    end

    nullcolor = "green"

    if isempty(freqs)
        freqs = autofrequencies(as, 100, 10)
    end

    mcmultiplicity = size(nullperiodograms, 2)

    # The avarage periodogram for zero-amplitude MC dataa
    nullmeans = mapslices(mean, nullperiodograms, 2) |> vec
    plot(freqs * timeunit, nullmeans |> cnv, color=nullcolor, lw=2, label="null hypothesis")


    # Calculate confidence level bands around the average periodogram for zero-amplitude MC data.
    # They visually represent the distribution of the periodogram as an estimator.
    # ref. https://en.wikipedia.org/wiki/Normal_distribution#Standard_deviation_and_tolerance_intervals
    confidencelevels = [ erf(1 / sqrt(2)),  # 1 sigma
                         erf(2 / sqrt(2)),  # 2 sigma
                         erf(3 / sqrt(2)) ] # 3 sigma

    pups = map(cl -> mapslices(d -> quantile(d, cl), nullperiodograms, 2) |> vec,
               confidencelevels)
    pdowns = map(cl -> mapslices(d -> quantile(d, 1 - cl), nullperiodograms, 2) |> vec,
                 confidencelevels)

    for (pup, pdown) in zip(pups, pdowns)
#         plot(freqs * timeunit, pup, color=nullcolor, lw=0.5)
#         plot(freqs * timeunit, pdown, color=nullcolor, lw=0.5)
        fill_between(freqs * timeunit, pup |> cnv, pdown |> cnv, color=nullcolor, alpha=0.2)
    end


    # Calulate the correct false-alarm boundaries with brute-force counting.
    # This works poorly for far tails, as huge amount of Monte Carlo samples
    # is needed in order to reproduce the tails well. This might be refined by
    # Approximating the tails with their functional form: exp(-P)
    #
    # For the functional form ref. eq. (13) in
    # J. D. Scargle, “Studies in astronomical time series analysis.
    # II - Statistical aspects of spectral analysis of unevenly spaced data”
    # Astrophys. J., vol. 263, p. 835, Dec. 1982.
    #
    # For the false-alarm boundaries ref. Dashed lines in Fig. 4 in
    # L. Pandola, “Search for time modulations in the Gallex/GNO solar neutrino data”
    # Astropart. Phys., vol. 22, no. 2, pp. 219–226, Nov. 2004.
    singleconfidencelevels = confidencelevels.^(1 / length(freqs))
    poverallups = map(cl -> mapslices(d -> quantile(d, cl), nullperiodograms, 2) |> vec,
                      singleconfidencelevels)

#     for (i, poverallup) in enumerate(poverallups)
#         plot(freqs * timeunit, poverallup |> cnv, lw=1, color="orange",
#              label = i == 1 ? "false-alarm thresholds (1-, 2-, 3-sigma)" : nothing)
#     end


    # Estimate false-alarm thresholds with exp extrapolation
    # CDF is 1 - exp(-P), ref. Scargle eq. (13)
    # We fit line to logarithm:
    # pr = 1 - A * exp(B * P)
    # 1 - pr = A * exp(B * P)
    # log(1 - pr) = log(A * exp(B * P))
    # log(1 - pr) = log(A) + B * P
    # So the fit parameters are log(A) and B
#     extrconfidencelevels = linspace(0.7, 0.99, 10) |> collect
#     extrpoverallups = map(cl -> mapslices(d -> quantile(d, cl), nullperiodograms, 2) |> vec,
#                           extrconfidencelevels)

#     fits = mapslices(extrpoverallup -> [linreg(extrpoverallup, log(-extrconfidencelevels + 1))...],
#                      hcat(extrpoverallups...), 2)

    fits = mapslices(extrapolateCDF, nullperiodograms, 2)
    pvals(power) = mapslices(fitres -> pvalfromfitres(fitres, power), fits, 2)
    powers_on_pval(pval) = mapslices(fitres -> invpvalfromfitres(fitres, pval), fits, 2)

    # Try the alternative way to calculate the global CL - brute force
    # Find the maximum CDF, i.e. the minimal p-value for each of the null periodograms
    # minpvals = mapslices(power -> minimum(pvals(power)), nullperiodograms, 1)
    minpvals = Float64[
        # Calculate p-value for each frequency in j-th periodogram and take the minimum
        Float64[ pvalfromfitres(fits[i,:], nullperiodograms[i,j]) for i in 1:length(freqs) ] |> minimum
        for j in 1:size(nullperiodograms, 2) ]

    #qs = linspace(0.7, 0.99, 10) |> collect
    # qs = linspace(0.01, 0.99, 100) |> collect
    # effective_n = log(quantile(1 - vec(minpvals), qs)) \ log(qs)
    # effective_n = effective_n[1]

    # use Nick's definition of effective n
    # Calculated as the maximum likelihood estimator
    nicks_effective_n = - length(minpvals) / sum(log(1 - minpvals))
    effective_n = nicks_effective_n

    print("Effective number of independent frequencies: $effective_n, normal frequencies: $(length(freqs)).")


#     println("min pvals", minpvals)
#     fitminpvals = extrapolateCDF(minpvals, 1 - linspace(0.7, 0.99, 10))
#     globalpval(localpval) = pvalfromfitres(fitminpvals, localpval)
#     localpval(globalpval) = invpvalfromfitres(fitminpvals, globalpval)

    # erf((1:5) / sqrt(2)) are 1..5- sigma levels
    globalcls = erf((1:5) / sqrt(2))
    globalpvals = 1 - globalcls
    localcls = globalcls .^ (1 / effective_n)
    localpvals = 1 - localcls
    # localcls = map(localCL, globalcls)

    println("Old local p-values: ", 1 - globalcls.^(1 / length(freqs)))
    println("New local p-values: ", localpvals)

    false_alarm_thresholds = map(powers_on_pval, localpvals) |> hcat

    for (i, th) in enumerate(false_alarm_thresholds)
        plot(freqs * timeunit, th |> cnv, lw=1, color="orange",
            label = i == 1 ? "false-alarm thresholds (1-, 2-, 3-, 4-, 5-sigma)" : nothing)
    end

#     for (i, localcl) in enumerate(localcls)
#         # traditional method
#         # globalcl = cl^(1 / length(freqs))

#         # brute-force method
#         # localcl = localCL(globalcl)

#         # plot(freqs * timeunit, exp(fits[:,1] .+ fits[:,2] * globalcl) |> cnv, lw=1, color="red",
#         # We need CDF^1
#         # pr = 1 - A * exp(B * P)
#         # A * exp(B * P) = 1 - pr
#         # exp(B * P) = (1 - pr) / A
#         # B * P = log((1 - pr) / A)
#         # B * P = log(1 - pr) - log(A)
#         # P = (log(1 - pr) - log(A)) / B
#         plot(freqs * timeunit, invCDFs(localcl) |> cnv, lw=1, color="orange",
#         label = i == 1 ? "false-alarm thresholds (1-, 2-, 3-, 4-, 5-sigma)" : nothing)
#     end

    # erf((1:5) / sqrt(2)) are 1..5- sigma levels
    # old method
    # false_alarm_thresholds = map(globalcl -> invCDFs(globalcl),
    #                             erf((1:5) / sqrt(2)).^(1 / length(freqs)) ) |> hcat
    # brute-force method
    # false_alarm_thresholds = map(globalcl -> invCDFs(c), localcls) |> hcat



    # Calculate the periodogram of the dataset
    # it is now passed as an argument
    # dataperiodogram = periodogramLSSA(freqs, as, data, offsets = fitoffsets)
    plot(freqs * timeunit, dataperiodogram |> cnv, lw=1, color="black", label=label)


    # for each frequency:
    #    calculate the confidence level at which the null hypothesis may be rejected
    #    against a strong signal alternative hypothesis
    # nullrejectionCL(fi) = count(pow -> pow <= dataperiodogram[fi], nullperiodograms[fi,:]) / length(nullperiodograms[fi,:])
    # nullrejectionCLs = map(nullrejectionCL, eachindex(freqs))

    # Replace that with an extrapolated version.
    # Evaluate global CDF at a given value
    nullrejectionCLs = similar(dataperiodogram)
    for i in eachindex(freqs)
        CLi = 1 - exp(fits[i, 1] + fits[i, 2] * dataperiodogram[i])
        nullrejectionCLs[i] = CLi
    end


    # Determine the least-likely peak in the data spectrum
    maxnullrejectionCL, idatamax = findmax(nullrejectionCLs)
    datamax = dataperiodogram[idatamax]

    if !simple
        axhline(datamax |> cnv, ls="--", color="black", lw=1.5)
        axvline(freqs[idatamax] * timeunit, ls="--", color="black")
    end


    # adjust the figure
    yscale("log")
    xscale("log")
    xunit = timeunit == SECOND ? "Hz" : "1 / $(timeunits[timeunit])"
    xlabel("frequency ($xunit)")
    ylabel(amplitudeunits ? "amplitude" : "power")
    margins(0, 0.2)
    ylimits = ylim()
    xlim(extrema(freqs * timeunit)...)
    legend(loc="best")
    axvline(1 / duration(as.ct) * timeunit, color="0.5", lw=1, ls="--", zorder=1)
    axvline(1 / median(separations(as.ct)) * timeunit, color="0.5", lw=1, ls="--", zorder=1);

    if !simple
        # Plot the vertical cut through the first figure at the least likely peak
        subplot(222)
        bins = logspace(log10(ylimits[1]), log10(ylimits[2]), 100)
        plt[:hist](nullperiodograms[idatamax,:] |> vec |> cnv, bins=bins, orientation="horizontal", color=nullcolor)
        axhline(datamax |> cnv, ls="--", color="grey", lw=1.5)
        yscale("log")
        ylim(ylimits...)
        title("null hypothesis power at least likely peak");


        # Plot the confidence levels of the per-frequency null hypothesis rejection
        subplot(223)
        axhline(0.95, ls="-", color="grey")
        plot(freqs * timeunit, nullrejectionCLs, color="black")
        xscale("log")
        # ref. https://en.wikipedia.org/wiki/Logit
        yscale("logit")
        ylim(1 / mcmultiplicity, 1 - 1 / mcmultiplicity)
        xlim(extrema(freqs * timeunit)...)
        xlabel("frequency (1/$(timeunits[timeunit]))")
        ylabel("CL on null hypothesis rejection at a fixed frequency")

        # make red lines to indicate overflows - none of the generated periodograms with
        # the null hypothesis assumption reached as high power as the data periodogram
        for i in find(abs(nullrejectionCLs - 1) .<= 0.001)
            axvline(freqs[i] * timeunit, lw=2, color="red", ymin=0.95)
        end
        # and underflows
        for i in find(abs(nullrejectionCLs) .<= 0.001)
            axvline(freqs[i] * timeunit, lw=2, color="red", ymax=0.05)
        end
    end

    println("minimal p-value on null hypothesis rejection over all frequencies: $(1 - maxnullrejectionCL)")
    overallnullrejectionCL = maxnullrejectionCL ^ length(freqs)
    println("p-value on null hypothesis rejection over all frequencies: $(1 - overallnullrejectionCL)")

    effective_n, [freqs cnv(dataperiodogram) cnv(nullmeans) cnv(hcat(false_alarm_thresholds...))]
end


function exclusionreportplot(CLmap, as, freqs, ampls; timeunit = DAY, conversionfactor = 1, plotline = true)
    figure(figsize=(8, 8))

    # boundsgrid(a) = logspace(extrema(log10(a))..., length(a) + 1)
    pcolormesh(freqs * timeunit, ampls * conversionfactor, CLmap',
               vmax=1, vmin=0, cmap="gist_heat_r")
    colorbar(label="CL on signal hypothesis exclusion")
    if plotline
        contour(freqs * timeunit, ampls * conversionfactor, CLmap', levels=[0.95], colors="white")
    end
#     axvline(1 / duration(as.ct) * timeunit, color="cyan", lw=1)
#     axvline(1 / mean(separations(as.ct)) * timeunit, color="cyan", lw=1)
    xscale("log")
    xunit = timeunit == SECOND ? "Hz" : "1 / $(timeunits[timeunit])"
    xlabel("frequency ($xunit)")
    yscale("log")
    ylabel("amplitude")
    title("CL on signal hypothesis exclusion")
end
