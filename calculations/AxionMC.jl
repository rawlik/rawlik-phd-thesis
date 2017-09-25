using ProgressMeter
using HDF5
using DataFrames

# physical constants
const γn = 29.1646943e6
const γHg = 7.590118e6
const h = 6.626070040e-34
const e_charge = 1.6021766208e-19
const cm = 0.01
const kV = 1000

# properties of the experiment
const chamber_height = 12cm
# deltah = 0.33cm
const deltah = 0.235cm
const HV_potential = 132kV
const E_field_value = HV_potential / chamber_height
const avB0 = 7.8585 / γHg

# conversion cefficients
const fn2dn = h / (2 * E_field_value)
const fn2dn1026ecm = fn2dn / (1e-26 * e_charge * cm)

const R2dn = h / (2 * E_field_value) * avB0 * γHg
const R2dn1026ecm = R2dn / (1e-26 * e_charge * cm)

const meV2f = e_charge / h

A1026ecm2gd(A, m) = (A * 1e-26 / 5e-22 * 3.7e-19) .* (m / 1e-22)

# epoch for time handling - the time zero
# we are lucky that no leap second ocurred during the data taking period
# https://en.wikipedia.org/wiki/Leap_second#Insertion_of_leap_seconds
const epoch = DateTime(2015,7)


# First define the AxionSim object that holds experiment-specitic information,
# i.a. timing and precision of every measurement.

@everywhere begin
    import Base.length

    const SECOND = 1
    const MINUTE = 60SECOND
    const HOUR = 60MINUTE
    const DAY = 24HOUR
    const timeunits = Dict(SECOND => "seconds",
                           HOUR   => "hours",
                           DAY    => "days")


    immutable CycleTimes
        # each point has a corresponding start and end time
        t::Array{Float64,2}
    end

    CycleTimes(tstart, tend) = CycleTimes([tstart'; tend'])
    length(ct::CycleTimes) = size(ct.t, 2)
    midtimes(ct::CycleTimes) = vec(mean(ct.t, 1))
    lengths(ct::CycleTimes) = diff(ct.t)
    duration(ct::CycleTimes) = -(-(extrema(ct.t)...))
    separations(ct::CycleTimes) = diff(midtimes(ct))

    "describtion of simulation"
    immutable AxionSim
        # each point has a corresponding start and end time
        ct::CycleTimes

        # precalculated middle of each run
        mt::SharedVector{Float64}

        sR::Vector{Float64}
        runlengths::Vector{Int}
        cumsum_runlengths::Vector{Int}
    end

    "Constructor for the AxionSim object."
    function AxionSim(;cyclelength = 270SECOND,
                       runlengths::Vector{Int} = 10 * ones(Int, 3),
                       sR=0.1,
                       startrandomization=1MINUTE,
                       ct=CycleTimes(Array{Float64,2}()))
        N = sum(runlengths)

        # if time array not supplied explicitely, create one
        if length(ct) == 0
            tstart = linspace(0, N * cyclelength, N)
            # randomize the times a bit, so that the points are not equally spaced
            tstart += startrandomization * randn(N)
            tend = tstart + cyclelength
            ct = CycleTimes(tstart, tend)
        end

        if isa(sR, Number)
            sR = fill(sR, length(ct))
        end

        if length(ct) != length(sR)
            throw("t and sR must have equal length")
        end

        AxionSim(ct, midtimes(ct), sR, runlengths, cumsum(runlengths))
    end

    function sequencemask(as, i)
        cs = [0; as.cumsum_runlengths]
        cs[i] + 1 : cs[i+1]
    end

end


# Now define functions specific to the axion-model and the Monte-Carlo functions.

@everywhere begin
    model(t, A, f, φ) = A * sin(2π * f * t + φ)
    noise(sim::AxionSim) = sim.sR .* randn(length(sim.ct))

    "High-performance version without memory allocation."
    function noise!(R::AbstractArray, sim::AxionSim)
       for i in eachindex(R)
           R[i] += sim.sR[i] * randn()
       end
    end

    pointlikemcR(sim::AxionSim, A, f, φ) = map(t -> model(t, A, f, φ), midtimes(as.ct)) .+ noise(sim)

    integratedmodel(t, A, f, φ) = -A * cos(2π * f * t + φ) / (2π * f)
    averagedmodel(t1, t2, A, f, φ) = (integratedmodel(t2, A, f, φ) .- integratedmodel(t1, A, f, φ)) ./ (t2 - t1)
    simR(sim::AxionSim, A, f, φ) = vec(mapslices(p -> averagedmodel(p..., A, f, φ), sim.ct.t, 1))

    "High-performance version without memory allocation."
    function simR!(tempR::AbstractArray, sim::AxionSim, A, f, φ)
       for i in eachindex(tempR)
           tempR[i] = averagedmodel(sim.ct.t[1,i], sim.ct.t[2,i], A, f, φ)
       end
    end

    mcR(sim::AxionSim, A, f, φ) = simR(sim, A, f, φ) .+ noise(sim)

    "High-performance version without memory allocation."
    function mcR!(tempR::AbstractArray, sim::AxionSim, A, f, φ)
        simR!(tempR, sim, A, f, φ)
        noise!(tempR, sim)
    end

    nonoisemcR(sim::AxionSim, A, f, φ) = simR(sim, A, f, φ)
    nonoisemcR!(tempR::AbstractArray, sim::AxionSim, A, f, φ) =
        simR!(tempR, sim, A, f, φ)

    randomsteps(sim::AxionSim) = vcat(map(ones, sim.runlengths) .* randn(length(sim.runlengths))...)

    "High-performance version without memory allocation."
    function randomsteps!(tempR::AbstractArray, sim::AxionSim, s::Number = 1)
        i = 1
        for j in sim.cumsum_runlengths
            offset = randn() * s
            for k in i:j
                @inbounds tempR[k] += offset
            end
        i = j + 1
        end
    end

    mcR(sim::AxionSim, A, f, φ, Astep) = mcR(sim, A, f, φ) .+ Astep .* randomsteps(sim)

    "High-performance version without memory allocation."
    function mcR!(tempR::AbstractArray, sim::AxionSim, A, f, φ, Astep)
        mcR!(tempR, sim, A, f, φ)
        randomsteps!(tempR, sim, Astep)
    end

    pointlikemcR(sim::AxionSim, A, f, φ, Astep) = pointlikemcR(sim, A, f, φ) .+ Astep .* randomsteps(sim)
end


# define the Hann function for optional use in periodograms
hanning(x, x0::Number, x1::Number) = 0.5(1 - cos(2π .* (x - x0) / (x1 - x0)))
hanning(A::Array) = hanning(A, extrema(A)...)

@everywhere begin
    """Utility function to generate for each run an array with ones for data points of this run
    and zeros for all the other data points.
    """
    offsetmask(a) = hcat([ vcat(zeros(sum(a[1:i-1])), ones(a[i]), zeros(sum(a[i+1:end]))) for i in eachindex(a) ]...)
end


# Define functions for calculating the periodograms

@everywhere begin
    """Calculate power with Linear Least-Squares method.
    ref. https://en.wikipedia.org/wiki/Least-squares_spectral_analysis"""
    function fitLSSA(f::Number, times, values, sigmas, runlengths=[length(times)])
        φ = 2π * f * times
        sqrtw = 1 ./ sigmas

        M = [ sin.(φ) cos.(φ) offsetmask(runlengths) ]
        # fit without offsets
        # M = [ sin(φ) cos(φ) ]
        res = (M .* sqrtw) \ (values .* sqrtw)
    end

    """High-performance version that needs pre-allocated memory.
    The result is stored in tempR"""
    function fitLSSA!(tempM::AbstractMatrix, tempR::AbstractArray, f::Number, times, sigmas, runlengths=[length(times)])
        for i in eachindex(tempR)
            φ = 2π * f * times[i]
            tempM[i,1] = sin(φ) / sigmas[i]
            tempM[i,2] = cos(φ) / sigmas[i]
        end

        # fill the elements of the matrix for the offsets
        imin = 1
        imax = 0
        for j in eachindex(runlengths)
            imax += runlengths[j]
            for i in eachindex(tempR)
                tempM[i,2+j] = (imin <= i <= imax) ? 1 : 0
                tempM[i,2+j] /= sigmas[i]
            end
            imin = imax + 1
        end

        tempR ./= sigmas

        LinAlg.LAPACK.gels!('N', tempM, tempR)
    end

    fitLSSA(frequencies::AbstractVector, times, values, sigmas, runlengths=[length(times)]) =
        @parallel (hcat) for f in frequencies
            fitLSSA(f, times, values, sigmas, runlengths)
        end

    fitLSSA(f, as::AxionSim, values; offsets = false) =
        fitLSSA(f, as.mt, values, as.sR, offsets ? as.runlengths : [length(as.mt)])

    power(N::Number, res) = N / 4 * (res[1, :].^2 .+ res[2, :].^2)
    power(as::AxionSim, res) = power(length(as.mt), res)
    phase(res) = atan2.(res[2, :], res[1, :])

    periodogramLSSA(frequencies, times, values, sigmas, runlengths = [length(times)]) =
        power(length(times), fitLSSA(frequencies), times, values, sigmas, runlengths) |> vec

    periodogramLSSA(frequencies, as::AxionSim, values; offsets = false) =
        power(as, fitLSSA(frequencies, as, values, offsets = offsets)) |> vec
end


"""Generate a set of frequencies appropriate for the given AxionSim object.
With extend=1 the highest is the Nyquist frequency and the lowest 1 / (4 * duration)."""
function autofrequencies(as, num=100, extend=[1,1]; traditionaltoo = false)
    freq1 = 1 / 4duration(as.ct) / extend[1]
    freqend = 1 / 2mean(separations(as.ct)) * extend[2]

    logspaced = logspace(log10(freq1), log10(freqend), num)

    if traditionaltoo
        # define frequencies as union of "traditional" frequencies (spectral resolution spacing) and
        # of log-spaced ones, so that it nicely looks on a log-scale
        return sort([ logspaced; 1 / duration(as.ct) : 1 / duration(as.ct) : 1 / mean(separations(as.ct)) ])
    else
        return logspaced
    end
end


@everywhere function cleverfrequencies(as::AxionSim; extendleft = nothing, extendright = nothing)
    freqs = 1 / duration(as.ct) : 1 / duration(as.ct) : 1 / mean(separations(as.ct))

    if isa(extendleft, Tuple)
        freqs = sort([logspace(log10(extendleft[1]), log10(minimum(freqs)), extendleft[2])[1:end-1] ; freqs])
    end

    if isa(extendright, Number)
        freqs = sort([ freqs ; (maximum(freqs) : 1 / duration(as.ct) : extendright)[2:end] ])
    end

    freqs
end


function writeas(as::AxionSim, data::AbstractArray, name::AbstractString)
    writedlm("$name.txt", [as.ct.t' data as.sR])
    writedlm("$(name)_runlengths.txt", as.runlengths)
end

function readas(name::AbstractString)
    d = readdlm("$name.txt")
    runlengths = readdlm("$(name)_runlengths.txt", Int64) |> vec
    as = AxionSim(runlengths = runlengths, sR = d[:,4], ct = CycleTimes(d[:,[1,2]]'))
    data = d[:,3]

    as, data
end


function mcperiodograms(as::AxionSim, freqs, signalamplitude, signalfrequency;
                        mcmultiplicity = 10000,
                        fitoffsets = false,
                        simjumpsize = 1)
    p = Progress(mcmultiplicity, 1)
    hcat(Array{Float64}[
            (next!(p); periodogramLSSA(
                freqs, as, mcR(as, signalamplitude, signalfrequency, rand() * 2π, simjumpsize),
                offsets = fitoffsets))
            for i in 1:mcmultiplicity]...)
end

readmc(filename, name) = h5read("cache.hdf5", name)
storemc(name, periodograms) = h5write("cache.hdf5", name, periodograms)
deletemc(name) = h5open("cache.hdf5", "r+") do file
    o_delete(file[name])
end
function appendmc(name, periodograms)
    old_periodograms = readmc(name)
    deletemc(name)
    storemc(name, cat(length(size(periodograms)), old_periodograms, periodograms))
end


@everywhere begin
    # CDF extrapolation
    function extrapolateCDF(powers)
        # Estimate false-alarm thresholds with exp extrapolation
        # CDF is 1 - exp(-P), ref. Scargle eq. (13)
        # We fit line to logarithm:
        # pr = 1 - A * exp(B * P)
        # 1 - pr = A * exp(B * P)
        # log(1 - pr) = log(A * exp(B * P))
        # log(1 - pr) = log(A) + B * P
        # So the fit parameters are log(A) and B

        # extrconfidencelevels = linspace(0.7, 0.99, 10) |> collect
        # need to exclude the last point, as then we have log(0)
        fitres = linreg(sort(powers)[1 : end-1],
                        log.(1 - collect(eachindex(powers)) / length(powers))[1 : end-1])
        # if rand() > 0.9
        #     figure()
        #     plot(sort(powers), log(1 - collect(eachindex(powers)) / length(powers)), ",")
        #     println(
        #         all(isfinite(sort(powers))), " ",
        #         all(isfinite(log(1 - collect(eachindex(powers)) / length(powers)))), " ",
        #         fitres
        #     )
        #     readline(STDIN)
        #     exit()
        # end
        collect(fitres)
    end

    pvalfromfitres(fitres, p) = exp(fitres[1] + fitres[2] * p)
    invpvalfromfitres(fitres, pval) = (log.(pval) - fitres[1]) ./ fitres[2]

    CLfromfitres(fitres, p) = 1 - pvalfromfitres(fitres, p)
    invCLfromfitres(fitres, cl) = invpvalfromfitres(fitres, 1 - cl)

    function extrapolateCDF2(minlocalpvals, confidencelevels = linspace(0.7, 0.99, 10))
        # Estimate false-alarm thresholds with exp extrapolation
        # CDF is CL^n, ref. http://math.stackexchange.com/questions/313390/probability-density-of-the-maximum-of-samples-from-a-uniform-distribution
        maxlocalCLs = 1 - minlocalpvals
        quantiles = quantile(maxlocalCLs |> vec, confidencelevels)
        fitres = linreg(log(quantiles), log(confidencelevels))
    end

    globalCL(fitres, localcl) = exp(fitres[1]) * localcl^fitres[2]
    localCL(fitres, globalcl) = (globalcl * exp(-fitres[1]))^(1 / fitres[2])

    globalpval(fitres, localpval) = 1 - globalCL(fitres, 1 - localpval)
    localpval(fitres, globalpval) = 1- localCL(fitres, 1 - globalpval)
end


# calculate the phase in the middle of the signal - the phase which is not correlated to frequency
phasemid(res, as, freq) = mod(phase(res) .+ 2pi * freq' * -(-(extrema(as.mt)...)) / 2, 2pi)

@everywhere function mcsignalpowers(as::AxionSim, hypothesisfrequencies::AbstractVector, hypothesisamplitudes::AbstractVector;
                        mcmultiplicity = 1000,
                        fitoffsets = false,
                        simjumpsize = 1)
    task = [ (hf, ha) for hf in hypothesisfrequencies, ha in hypothesisamplitudes ]
    function calc(hfha)
        println("frequency $(hfha[1]), amplitude $(hfha[2])")
        Float64[
            periodogramLSSA(hfha[1], as, mcR(as, hfha[2], hfha[1], rand() * 2π, simjumpsize), offsets = fitoffsets)[1]
        for i in 1:mcmultiplicity]
    end
    # calc(hfha) = Float64[
    #         periodogramLSSA(hfha[1], as, mcR(as, hfha[2], hfha[1], rand() * 2π, simjumpsize), offsets = fitoffsets)[1]
    #     for i in 1:mcmultiplicity]
    calcres = reshape(pmap(calc, task), size(task))
    res = zeros(length(hypothesisfrequencies), length(hypothesisamplitudes), mcmultiplicity)
    for i in eachindex(hypothesisfrequencies), j in eachindex(hypothesisamplitudes), k in 1:mcmultiplicity
        res[i,j,k] = calcres[i,j][k]
    end
    res
end


@everywhere function mcsignalpowers(as::AxionSim, hypothesisfrequency::Number, hypothesisamplitude::Number;
                        mcmultiplicity = 1000,
                        fitoffsets = false,
                        simjumpsize = 1)
    Float64[
        periodogramLSSA(hypothesisfrequency, as,
            mcR(as, hypothesisamplitude, hypothesisfrequency, rand() * 2π, simjumpsize), offsets = fitoffsets)[]
            for i in 1:mcmultiplicity]
end


@everywhere function exclusionreport(as::AxionSim, data, freqs, ampls, nullperiodograms, signalperiodograms;
                         grid = 10,
                         mcmultiplicity = 1000,
                         useCLsmethod = true,
                         Astep = 2,
                         fitoffsets = true)
    # iterate over the space of the signal hypotheses
    CLmap = zeros(length(freqs), length(ampls))

    for ihypothesisfreq in eachindex(freqs)
        nullpowers = nullperiodograms[ihypothesisfreq, :]
        datapower = periodogramLSSA(freqs[ihypothesisfreq], as, data, offsets = fitoffsets)[1]

        for ihypothesisampl in eachindex(ampls)
            hypothesispowers = signalperiodograms[ihypothesisfreq, ihypothesisampl, :]

            # ref. Read "Modified frequentist analysis of search results (the CLs method)"
            # in Workshop on Confidence Limits, CERN 2000.
            CLsignalbackground = count(pow -> pow <= datapower, hypothesispowers) / length(hypothesispowers)

            # fall back to extrapolation
            # if CLsignalbackground == 0
            if false
                # DO NOT DO THIS WEIRD EXTRAPOLATION
                # Not necessary where we care
            # if CLsignalbackground < 0.3
                confidencelevels = linspace(0.01, 0.3, 10) |> collect
                quantiles = map(cl -> quantile(hypothesispowers |> vec, cl), confidencelevels)


                #a, b = linreg(quantiles, log(confidencelevels))
                #CLsignalbackground = exp(a + b * datapower)

                # linear fit without an offset
                a = quantiles \ confidencelevels
                # # improve to evaluate each point of CDF
                # v = sort(vec(hypothesispowers))
                # # Extrapolate only CDF < 0.1
                # v = v[1 : round(Int, length(v) * 0.1)]
                # a = sort(v) \ ( collect(eachindex(v)) / length(v) )
                CLsignalbackground = a[1] * datapower

                # CLsignalbackground = CLfromfitres(extrapolateCDF(hypothesispowers, linspace(0.01, 0.3, 10)), datapower)
            end
#             CLbackground = count(pow -> pow <= datapower, nullpowers) / length(nullpowers)
            # for the background use extrapolation
            # to avoid a situation where the above count(...) would yield zero and CLsignal would be NaN
            CLbackground = CLfromfitres(extrapolateCDF(vec(nullpowers)), datapower)

            CLsignal = CLsignalbackground / CLbackground

            CLmap[ihypothesisfreq, ihypothesisampl] = useCLsmethod ? (1 - CLsignal) : (1 - CLsignalbackground)
        end
    end

    CLmap
end


function getB0direction(runnumber, importdatadir)
    h5readattr(joinpath(importdatadir, @sprintf "%06dRAW.hdf5" runnumber),
        @sprintf "Run%06d" runnumber)["B0direction"][]
end


function getcycletimes(runnumber, cyclenumber, importdatadir)
    fname = joinpath(importdatadir, @sprintf "%06dRAW.hdf5" runnumber)

    datearray = h5readattr(fname, @sprintf "Run%06d/Cycle%04d" runnumber cyclenumber)["CycleStartTime"]

    # get the timing of the neutron spin-flip pulses
    stepmask = h5read(fname, @sprintf "Run%06d/uTimer/stepmask" runnumber)
    nrfgate = stepmask .& 2^3 .!= 0
    nrf = stepmask .& 2^5 .!= 0

    stepduration = h5read(fname, @sprintf "Run%06d/uTimer/stepduration" runnumber)
    nrf1_start = cumsum(stepduration)[ find(nrfgate .& nrf)[1] - 1]
    nrf1_end = cumsum(stepduration)[ find(nrfgate .& nrf)[1] ]

    nrf2_start = cumsum(stepduration)[ find(nrfgate .& nrf)[2] - 1]
    nrf2_end = cumsum(stepduration)[ find(nrfgate .& nrf)[2] ]

    # get the absolute time of the cycle's start
    epochcyclestarttime = Dates.value(DateTime(datearray[1:end-1]...) - epoch) / 1000

    [ epochcyclestarttime + mean([nrf1_start, nrf1_end]),
      epochcyclestarttime + mean([nrf2_start, nrf2_end]) ]
end


# functions to get compount HDF5 columns
# work only if all columns have the same size (as in sizeof)
# getcompoundcolumn(d, i::Int) = reinterpret(
#     d.membertype[i], d.data)[i : length(d.membertype) : end]
# getcompoundcolumn(d, s::String) = getcompoundcolumn(d, findfirst(d.membername, s))
# needed to change after an update of HDF5.jl
getcompoundcolumn(d, i::Int) = [ cmp.data[i] for cmp in d ]
getcompoundcolumn(d, s::String) = [
    cmp.data[ findfirst(cmp.membername, s) ] for cmp in d ]


"""Fieldsconf can be 'parallel', 'anti-parallel' or 'zero'
Or now "up", "down", which ignores the electric field.
"""
function getRdataset(resultdatadir, importdatadir, fieldsconf;
        gradientdriftcorrection = true)
    allfiles = filter(r"Axion201[56]_[0-9]+_AnalysisRunLevel.hdf5",
        readdir(resultdatadir))

    rrun = Int64[]
    ccycle = Int64[]
    R = Float64[]
    RErr = Float64[]
    ct = Array{Float64,2}(0, 2)
    sequencelengths = Int64[]

    gcstr = gradientdriftcorrection ? "_WithGz" : ""

    @showprogress "Loading $fieldsconf... " for fname in allfiles
        dataR = h5read(joinpath(subsetdatadir, fname),
            "Ramsey/Asymmetry_SF2combo_WithHgErr$gcstr/SubDataset01/Rratio")
        dataHV = h5read(joinpath(subsetdatadir, fname),
            "Ramsey/Asymmetry_SF2combo_WithHgErr$gcstr/SubDataset01/HighVoltage")

        runnumbers = getcompoundcolumn(dataR, "RunNumber")
        cyclenumbers = getcompoundcolumn(dataR, "Cycle")

        # new file is a start of a new sequence
        # inside a sequence the magnetic field environment is assummed to be constant
        push!(sequencelengths, 0)

        for i in eachindex(runnumbers)
            # for each row in the datafile

            # check for bad data points, if bad - continue
            isfinite(getcompoundcolumn(dataR, "R")[i]) || continue
            isfinite(getcompoundcolumn(dataR, "RErr")[i]) || continue

            # get the B0 field direction
            b0direction = getB0direction(runnumbers[i], importdatadir)

            # get the HV
            hv = getcompoundcolumn(dataHV, "ReadHV")[i]
            inthv = round(Int, hv)

            # determine if the right field configuration
            if fieldsconf == "zero"
                inthv != 0 && continue
            elseif fieldsconf == "parallel"
                # in the parallel configuration (b0direction * hv) < 0
                # remember that we charge the *upper* electrode
                (inthv * b0direction < 0 ) || continue
            elseif fieldsconf == "anti-parallel"
                # in the anti-parallel configuration (b0direction * hv) > 0
                (inthv * b0direction > 0 ) || continue
            elseif fieldsconf == "up"
                b0direction > 0 || continue
            elseif fieldsconf == "down"
                b0direction < 0 || continue
            else
                println("There is no such field configuration: $fieldsconf")
            end

            # store the data
            push!(rrun, runnumbers[i])
            push!(ccycle, cyclenumbers[i])
            push!(R, getcompoundcolumn(dataR, "R")[i])
            push!(RErr, getcompoundcolumn(dataR, "RErr")[i])
            ct = [ct; getcycletimes(runnumbers[i], cyclenumbers[i], importdatadir)']
            sequencelengths[end] += 1
        end
    end

    R, AxionSim(runlengths = sequencelengths, sR = RErr, ct = CycleTimes(ct'))
end
