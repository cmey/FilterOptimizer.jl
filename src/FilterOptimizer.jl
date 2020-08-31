module FilterOptimizer

const debug = false
const fullblown_optim = true  # false: LsqFit, true: Optim

using DSP
using LinearAlgebra
using NPZ
using Optim
using LineSearches
using LsqFit
if !debug
    using Plots
end
using Polynomials
using Printf
using Random
using Statistics


function main_simple_plot(; ensemble_size=8, cutoff_frac=0.05, noDC=false, killOutputs=1:0)
    # cutoff_frac = 0.05  # [fraction of Nyquist, i.e. of fs/2]
    num_freqs = 100

    filter = make_initial_filter(ensemble_size, cutoff_frac)

    # K88  Filter given from Matlab is transposed.
    # filter = [ 0.101928620359513  -0.144692907668316   0.001818686042631   0.034079418317095   0.017585080356053   0.001315214737816  -0.003444416529060  -0.001436438714533
    #           -0.182345096610246   0.360756599962924  -0.147994984888741  -0.059129118196972   0.002868426988251   0.015712495777574   0.007502730016391  -0.002913799498315
    #            0.325318278686291  -0.644192151825324   0.366462705585208  -0.039188548203213  -0.002501377627620   0.008040320659969   0.004771188361656  -0.001218136494786
    #            0.214429240579235   0.020907800279936  -0.640407310078860   0.438171965206373  -0.001984312130525   0.000672744414457   0.000815968852531   0.000020116755318
    #            0.039388771853078   0.158515140964055   0.021611283755305  -0.627238094500448   0.444963982609944  -0.001482793266190  -0.000658657133152   0.000289435629589
    #           -0.028159300455806   0.079366422909548   0.158022482575611   0.012192601874105  -0.632146169468256   0.444503863315704  -0.000536386191742   0.000149049199835
    #           -0.023962585339180   0.005858919704412   0.078943901425177   0.150008766716844   0.008032778760745  -0.632505199692507   0.445310958035830   0.000012891795086
    #           -0.006277091297826  -0.015051750417134   0.005747354831077   0.076845015215617   0.148923596940809   0.007947473300587  -0.632293311502353   0.445417719102865];
    # filter = filter';

    if noDC
        filter = -mean(filter, dims=1) .+ filter;
    end

    filter[:, killOutputs] .= 0

    # path = "$(homedir())/Downloads/"
    path = "$(homedir())/Code/ButterflyNetwork/software/imaging/imaging/recon/doppler/wallfilter/data/"
    filepath = path * "optimized_filter_cutoff$(@sprintf("%.2f", cutoff_frac))_ensemble$(ensemble_size).npy"
    npzwrite(filepath, filter)

    input_vec, freq_vec = make_input(ensemble_size)
    output_vec = apply_wallfilter(input_vec, filter)
    plot_response(freq_vec, make_desired_response(cutoff_frac), make_response(input_vec), make_response(output_vec))

    FilterOptimizer.PlotWallFilter(FilterOptimizer.EvaluateWallFilter(filter)...)
    filter
end


function main_objective(; noDC=false)
    ensemble_size = 8
    cutoff_frac = 0.10  # [fraction of Nyquist, i.e. of fs/2]
    num_freqs = 100

    input_vec, freq_vec = make_input(ensemble_size, num_freqs)
    input_response = make_response(input_vec)

    desired_response = make_desired_response(cutoff_frac, num_freqs)
    @show desired_response

    initial_filter = make_initial_filter(ensemble_size, cutoff_frac)
    # objective = objective_autocorr
    objective = objective_consistency

    # warm up and debug
    x = initial_filter
    f, fout = EvaluateWallFilter(x, num_freqs)
    output_vec = apply_wallfilter(input_vec, x)
    output_response = make_response(output_vec)
    loss = objective(desired_response, output_response, fout)
    if !debug
        PlotWallFilter(f, fout)
    end

    function optim_iter(x)
        _, fout = EvaluateWallFilter(x, num_freqs)
        output_vec = apply_wallfilter(input_vec, x)
        output_response = make_response(output_vec)
        if fullblown_optim
            loss = objective(desired_response, output_response, fout)
            return loss
        else
            # c = vec(output_response[1] .* exp.(1im .* output_response[2]))
            # return vec([real.(c) ; imag.(c)])
            return [vec(real.(fout)) ; vec(imag.(fout))]
        end
    end

    optim_iter(initial_filter)

    if fullblown_optim
        # Continuous, multivariate, optimization
        # Simulated Annealing, Evolutionary algo
        # CG, etc.: Compute / Auto Grad of f?
        # SPSA: black box, doesnt need gradient of f. Perturbs all params at the same time.
        # FD: black box, doesnt need gradient of f. Perturbs one param at a time.
        result = optimize(optim_iter, copy(initial_filter),
            # LBFGS(linesearch=LineSearches.BackTracking()),
            BFGS(linesearch=LineSearches.BackTracking()),
            # ConjugateGradient(linesearch=LineSearches.BackTracking()),
            # NelderMead(),
            debug ? Optim.Options(show_trace = true, iterations = 1) : Optim.Options(show_trace = true),
            ; autodiff = :forward
            )
        @show result

        optimized_filter = Optim.minimizer(result)
    else
        # t: array of independent variable
        # p: array of model parameters
        model(t, p) = optim_iter(reshape(p, size(initial_filter)))
        tdata = vec(input_vec)
        c = vec(desired_response[1] .* exp.(1im .* desired_response[2]))
        c = repeat(c, ensemble_size)
        ydata = [vec(real.(c)) ; vec(imag.(c))]
        p0 = vec(initial_filter)
        fit = curve_fit(model, tdata, ydata, p0)
        optimized_filter = reshape(fit.param, size(initial_filter))
    end

    if noDC
        optimized_filter = mean(optimized_filter, dims=1) .- optimized_filter;
    end

    npzwrite("$(homedir())/Downloads/optimized_filter_cutoff$(cutoff_frac)_ensemble$(ensemble_size).npy", optimized_filter)

    output_vec = apply_wallfilter(input_vec, optimized_filter)
    if !debug
        PlotWallFilter(EvaluateWallFilter(optimized_filter, num_freqs)...)
        plot_response(freq_vec, desired_response, input_response, make_response(output_vec))
    end

    optimized_filter
end


function make_from_iir_to_matrix(ensemble_size, cutoff_frac)
    responsetype = Highpass(cutoff_frac)
    n = 2  # poles
    rp = 0.5  # [dB] ripple in the passband
    rs = 60  # [dB] stopband attentuation
    # designmethod = Chebyshev1(n, rp)  # ripple in passband (no ripple in stopband)
    designmethod = Chebyshev2(n, rs)  # ripple in stopband (no ripple in passband)
    # designmethod = Elliptic(n, rp, rs)  # n pole elliptic (Cauer) filter with rp dB ripple in the passband and rs dB attentuation in the stopband

    filter = digitalfilter(responsetype, designmethod)
    @show filter
    filter = convert(Biquad, filter)
    # DSP.Filters.Biquad(b0, b1, b2, a1, a2) form:
    # H(z) = (b0 +b1z^−1 +b2z^−2) / (1 +a1z^−1 +a2z^−2)

    display(plot(20log10.(abs.(freqz(filter))), title="20log10.(abs.(freqz(filter)))"))
    display(plot(abs.(phasez(filter)), title="abs.(phasez(filter))"))

    # Put in form: Edward S. Chornoboy Lincoln Lab MIT Technical Report 828, 31 December 1990
    # H(z) = (α0 +α1z^-1 +α2z^-2) / (1 + β1z^-1 + β2z^-2)
    # by association:
    α0 = filter.b0
    α1 = filter.b1
    α2 = filter.b2
    β1 = filter.a1
    β2 = filter.a2

    A = [α1-α0*β1
         α2-α0*β2]
    B = [-β1  -β2
           1    0]
    C = [  1
           0]

    F = zeros(ensemble_size, size(A, 1))
    for n in 1:ensemble_size
        F[n, :] = A' * B^(n-1)
    end

    G = zeros(ensemble_size, ensemble_size)
    for n in 1:ensemble_size
        G[n,n] = α0
    end
    for irow in 1:ensemble_size
        for icol in 1:irow-1
            G[irow,icol] = A' * B^(irow-icol-1) * C
        end
    end

    Pf = F * (F' * F)^-1 * F'

    filter = (I - Pf) * G
    filter'
end


function objective_autocorr(desired_response, output_response, fout)
    desired_mag, desired_pha = desired_response
    output_mag, output_pha = output_response
    e = [desired_mag .- output_mag ; desired_pha .- output_pha]
    norm(e)
end


function objective_consistency(desired_response, output_response, fout)
    # size(fout) = (num_freqs, ensemble_size_out)
    desired_mag, desired_pha = desired_response
    output_mag, output_pha = output_response
    # e = desired_mag .- output_mag  # OK
    # e = [desired_mag .- output_mag ; desired_pha .- output_pha]  # OK
    # e = vec(abs.(fout) .- desired_mag)  # OK
    # e = vec(std(diff(angle.(fout), dims=2), dims=2))  # NaN
    # e = vec(std(angle.(fout), dims=2))  # NaN
    # e = vec(diff(angle.(fout), dims=2))  # OK
    # e = [vec(abs.(fout) .- desired_mag) ; vec(diff(angle.(fout), dims=2))]  # OK but not great filter.
    # e = vec(angle.(fout))  # OK. Bad result with UNWRAP.
    # e = [vec(abs.(fout) .- desired_mag) ; vec(angle.(fout))]  # OK but not great filter.
    # e = vec(diff(diff(angle.(fout), dims=2), dims=2))  # OK
    # e = [vec(abs.(fout) .- desired_mag) ; vec(diff(diff(angle.(fout), dims=2), dims=2))]  # NaN

    # e = [desired_mag .- output_mag ; desired_pha .- output_pha ;
    #      vec(abs.(fout) .- desired_mag) ; vec(diff(unwrap(angle.(fout), dims=1), dims=2))]  # OK

    e = [vec(abs.(fout) .- desired_mag) ; vec(diff(angle.(fout), dims=2))]

    # e = [desired_mag .- output_mag ; vec(angle.(fout))]
    # e = [desired_mag .- output_mag ; vec(std(angle.(fout), dims=2))]  # NaN
    # e = [desired_mag .- output_mag ; vec(std(diff(angle.(fout), dims=2), dims=2))]  # NaN
    # e = abs.(fout) .- desired_mag
    # e = [desired_mag .- output_mag ; vec(diff(abs.(fout), dims=2))]
    # e = [desired_mag .- output_mag ; vec(diff(abs.(fout), dims=2))]  # NaN
    # e = [desired_mag .- output_mag ; vec(diff(abs.(fout), dims=2)) ; vec(diff(angle.(fout), dims=2))]  # NaN
    # e = [vec(desired_mag .- fout) ; vec(diff(abs.(fout), dims=2)) ; vec(diff(angle.(fout), dims=2))]
    # println(length(e))
    norm(e)
end


function make_initial_filter(ensemble_size, cutoff_frac)
    # #SHIFTED TRUNCATED FIR
    # responsetype = Highpass(cutoff_frac)
    # # ripple = 3.0  # [dB] in the passband
    # # n = 2  # poles
    # # designmethod = Chebyshev1(n, ripple)
    # window = hanning(5)
    # designmethod = FIRWindow(window; scale=true)
    # digitalfilter(responsetype, designmethod)

    # POLYNOMIAL REGRESSION FILTER
    # filter_order = 1
    # polynomial_regression_filter(ensemble_size, filter_order)

    # FIR 2zero FILTER
    # fir_2zero_wallfilter(ensemble_size)

    # RANDOM matrix
    Random.seed!(1);
    # need randn and not rand b/c need some negative coefficients in order to get a highpass filter
    # randn(ensemble_size, ensemble_size)
    # rand uniform [-1, +1)
    2*rand(ensemble_size, ensemble_size) .- 1

    # IIR
    make_from_iir_to_matrix(ensemble_size, cutoff_frac)
end


function make_input(ensemble_size, num_freqs=100)
    # generate input freqs
    time_vec = linspace_with_endpoint(0, ensemble_size-1, ensemble_size)
    freq_vec = linspace_with_endpoint(0, 1 / 2, num_freqs)
    outer = freq_vec .* time_vec'
    input_vec = exp.(1im .* outer .* 2π)
    (input_vec, freq_vec)
end


linspace_with_endpoint(start, last, length) = range(start, last, length=length)

linspace_without_endpoint(start, stop, length) = range(start, stop, length=length+2)[2:end-1]


function apply_wallfilter(input_vec, filter_mat)
    input_vec * complex(filter_mat)
end


function make_response(vec)
    mag = abs.(autocorrelation(vec, lag=0))
    pha = angle.(autocorrelation(vec, lag=1))
    (mag, pha)
end


function make_desired_response(cutoff_frac, num_freqs=100)
    n_stops = Int(round(num_freqs * cutoff_frac))
    n_pass = num_freqs - n_stops
    mag = [repeat([0], n_stops) ; repeat([1], n_pass)]
    phase = collect(linspace_with_endpoint(0, π, num_freqs))
    (mag, phase)
end


"""
    Autocorrelation over 1st dimension for the given non-negative lag of 0, 1, 2, ...
    Arguments:
        data -- 2D matrix of complex samples where the first dimension is of
                ensemble size and the second dimension is of 1D reshaped pixels size.
        lag  -- Number of 1-way "lag" samples to offset in autocorrelation.
"""
function autocorrelation(data; lag=0)
    @assert (lag >= 0) "autocorrelation lag must be positive (currently $lag)"
    ens = size(data)[2]
    @assert (ens >= 2 * lag) "autocorrelation requires an ensemble of at least size $(lag*2) (currently $ens)"
    mean(data[:, 1+lag:end] .* conj(circshift(data, (0, lag))[:, 1+lag:end]), dims=2)
end


"""
    Generate the wall filter matrix that can reject polynomials up to
    order filter_order i.e. f[k]= sum over i from 0 to filter_order (a_i*k^i)
    Arguments:
    ensemble_size    -- Number of acquisitions in an ensemble
    filter_order     -- Maximum order of polynomial we want to reject.
"""
function polynomial_regression_filter(filter_length, filter_order)
    # Python:
    #  xx = np.linspace(-1, 1, num=filter_length)
    #  A = xx[:, np.newaxis] ** np.arange(filter_order + 1)
    xx = linspace_with_endpoint(-1, 1, filter_length)
    A = xx .^ (0:filter_order)'
    # Find the column basis of the matrix
    Uonly, = svd(A; full=true)
    # Construct a matrix where each column is a basis of the perp. space of Range(A).
    Uc = Uonly[:, (1 + filter_order+1):end]
    HiP = Uc * Uc'
end


function fir_2zero_wallfilter(ensemble_size)
    """Generate a 2 zero wall filter:  [-1 2 -1]/4
        Arguments:
            ensemble_size    -- Number of acquisitions in an ensemble
    """
    wallfilter = zeros(ensemble_size, ensemble_size)
    for k in 1:ensemble_size - 3
        wallfilter[k:k+2, k] = [-1, 2, -1] / 4.0
    end
    wallfilter
end


function plot_response(freq_vec, desired_response, input_response, output_response)
    desired_mag, desired_pha = desired_response
    input_mag, input_pha = input_response
    output_mag, output_pha = output_response

    # plot frequency response (should attenuate lower freqs)
    # fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    p1 = plot(2freq_vec, [20log10.(desired_mag.+1e-6) 20log10.(input_mag) 20log10.(output_mag)],
        label=["Desired" "Input Ensemble" "WF'ed Ensemble"],
        title="Autocorrelation (frequency) response of wall filter",
        ylabel="Autocor0 abs [log a.u.]",
        xlims=(0, 1),
        ylims=(-100, Inf),
        lw=2,
    )

    # plot phase response
    p2 = plot(2freq_vec, [desired_pha input_pha output_pha],
        label=["Desired" "Input Ensemble" "WF'ed Ensemble"],
        xlabel="Frequency (0 to Nyquist) [fraction of fs/2]",
        ylabel="Autocor1 phase [radians]",
        xlims=(0, 1),
        ylims=(-π, π),
        lw=2,
    )

    # A Plot is only displayed when returned (a semicolon will suppress the return),
    # or if explicitly displayed with display(plt), gui(), or by adding show = true to your plot command.
    display(plot(p1, p2, size=(800, 500), layout = (2, 1), legend = true))
end


"""
    Func from Karl
"""
function EvaluateWallFilter(wf, num_freqs=100, PhaseOffset=0, PhaseMultiplier=1)
    # Matlab, old: Assume out (Nx1) = WF (NxM) x in (Mx1)
    # M = ensemble size, N = number of filters(output samples)
    # Assume in (1xM) x WF (MxN) = out (1xN)
    M, N = size(wf)

    # Assume Real
    f = linspace_with_endpoint(0, 1, num_freqs)
    xf = exp.(1im * π * f)
    P = length(f)

    # Fourier Output
    fout = zeros(Complex, P, N)
    for n=1:N
        coefs = wf[:,n]
        # Matlab coefs in descending powers, Julia in ascending powers!
        # fcoefs = polyval(coefs[:], xf[:])
        p = Polynomial(reverse(coefs[:]))
        fcoefs = p.(xf[:])
        fcoefs = fcoefs .* xf[:] .^(n-M+PhaseOffset)*PhaseMultiplier  # .* exp.(1im*(n-M/2).*xf[:])
        fout[:,n] = fcoefs[:]
    end

    f, fout
end


function PlotWallFilter(f, fout)
    # plot mag response
    p1 = plot(f, 20log10.(abs.(fout)),
        title="Magnitude Consistency Across Wall Filter Polyphases",
        ylabel="Abs [log a.u.]",
        xlims=(0, 1),
        ylims=(-80, Inf),
        lw=2,
    )
    # plot phase response
    uwpha = reverse(unwrap(reverse(angle.(fout .+ eps()), dims=1), dims=1), dims=1)
    p2 = plot(f, uwpha,
        title="Phase Consistency Across Wall Filter Polyphases",
        xlabel="Frequency (0 to Nyquist) [fraction of fs/2]",
        ylabel="Phase [radians]",
        xlims=(0, 1),
        ylims=(-π*1.1, π*1.1),
        lw=2,
    )
    # plot group delay (d phase / d w)
    p3 = plot(f[1:end-1], diff(uwpha, dims=1) ./ diff(f, dims=1) ./ π,
        title="Group Delay Consistency Across Wall Filter Polyphases",
        xlabel="Frequency (0 to Nyquist) [fraction of fs/2]",
        ylabel="Group delay [samples]",
        xlims=(0, 1),
        ylims=(-2, 2),
        lw=2,
    )
    display(plot(p1, p2, p3, size=(800, 800), layout = (3, 1), legend = true))
end


end # module


# wf = FilterOptimizer.main_simple_plot(noDC=true, killOutputs=1:0);
# wf = FilterOptimizer.main_simple_plot(ensemble_size=8, cutoff_frac=0.05, noDC=true, killOutputs=[1, 2, 7, 8]);

# wf = FilterOptimizer.main_simple_plot(ensemble_size=8, cutoff_frac=0.02, noDC=true, killOutputs=[1, 2, 3]);  # 0.05 in Bfly
wf = FilterOptimizer.main_simple_plot(ensemble_size=8, cutoff_frac=0.03, noDC=true, killOutputs=[1, 2, 3]);
# wf = FilterOptimizer.main_simple_plot(ensemble_size=8, cutoff_frac=0.04, noDC=true, killOutputs=[1, 2, 3]);
# wf = FilterOptimizer.main_simple_plot(ensemble_size=10, cutoff_frac=0.04, noDC=true, killOutputs=[1, 2, 3]);

# wf = FilterOptimizer.main_simple_plot(ensemble_size=8, cutoff_frac=0.03, noDC=false, killOutputs=[1, 2, 3]);

# wf = FilterOptimizer.main_simple_plot(ensemble_size=10, cutoff_frac=0.05, noDC=true, killOutputs=[1, 2, 8, 9, 10]);
# wf = FilterOptimizer.main_simple_plot(ensemble_size=10, cutoff_frac=0.05, noDC=true, killOutputs=[1, 2, 8, 9, 10]);
# wf = FilterOptimizer.main_simple_plot(ensemble_size=12, cutoff_frac=0.50, noDC=true, killOutputs=[1, 2, 10, 11, 12]);

# wf = FilterOptimizer.main_objective(noDC=true);

show(stdout, "text/plain", wf)
