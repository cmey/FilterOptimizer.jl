module FilterOptimizer

using DSP
using LinearAlgebra
using NPZ
using Optim
using Plots
using Polynomials
using Statistics


function main_simple_plot()
    ensemble_size = 8
    cutoff_frac = 0.1  # [fraction of Nyquist, i.e. of fs/2]

    input_vec, freq_vec = make_input(ensemble_size)

    initial_filter = make_initial_filter(ensemble_size, cutoff_frac)
    EvaluateWallFilter(initial_filter)

    output_vec = apply_wallfilter(input_vec, initial_filter)
    plot_response(freq_vec, make_response(input_vec), make_response(output_vec), ensemble_size)
end


function main_objective()
    ensemble_size = 8
    cutoff_frac = 0.2  # [fraction of Nyquist, i.e. of fs/2]
    num_freqs = 100

    input_vec, freq_vec = make_input(ensemble_size, num_freqs)
    input_response = make_response(input_vec)

    desired_response = make_desired_response(cutoff_frac, num_freqs)
    @show desired_response

    initial_filter = make_initial_filter(ensemble_size, cutoff_frac)
    EvaluateWallFilter(initial_filter)

    function optim_iter(x)
        output_vec = apply_wallfilter(input_vec, x)
        output_response = make_response(output_vec)
        # plot_response(freq_vec, input_response, output_response, ensemble_size)
        loss = loss_function(desired_response, output_response)
    end

    # Continuous, multivariate, optimization
    # Simulated Annealing, Evolutionary algo
    # CG, etc.: Compute / Auto Grad of f?
    # SPSA: black box, doesnt need gradient of f. Perturbs all params at the same time.
    # FD: black box, doesnt need gradient of f. Perturbs one param at a time.
    result = optimize(optim_iter, copy(initial_filter), BFGS(),
        Optim.Options(show_trace = true),
        ; autodiff = :forward
        )
    @show result

    optimized_filter = Optim.minimizer(result)
    show(stdout, "text/plain", optimized_filter)
    npzwrite("$(homedir())/Downloads/optimized_filter_cutoff$(cutoff_frac)_ensemble$(ensemble_size).npy", optimized_filter)

    output_vec = apply_wallfilter(input_vec, optimized_filter)
    plot_response(freq_vec, input_response, make_response(output_vec), ensemble_size)
    EvaluateWallFilter(optimized_filter)
    optimized_filter
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
    filter_order = 2
    polynomial_regression_filter(ensemble_size, filter_order)

    # FIR 2zero FILTER
    # fir_2zero_wallfilter(ensemble_size)

    # RANDOM matrix
    # need randn and not rand b/c need some negative coefficients in order to get a highpass filter
    # randn(ensemble_size, ensemble_size)
end


function loss_function(desired_response, output_response)
    desired_mag, desired_pha = desired_response
    output_mag, output_pha = output_response
    diff = [desired_mag .- output_mag ; desired_pha .- output_pha]
    sqrt(sum(abs2, diff))
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
    mean(data[:, (1+lag):end] .* conj(circshift(data, (0, lag))[:, (1+lag:end)]), dims=2)
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


function plot_response(freq_vec, input_response, output_response, ensemble_size)
    input_mag, input_pha = input_response
    output_mag, output_pha = output_response

    # plot frequency response (should attenuate lower freqs)
    # fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    p1 = plot(2freq_vec, [20log10.(input_mag) 20log10.(output_mag)],
        label=["Input Ensemble" "WF'ed Ensemble"],
        title="Autocorrelation (frequency) response of wall filter, Ens=$ensemble_size",
        ylabel="Autocor0 abs [log a.u.]",
        xlims=(0, 1),
        ylims=(-60, Inf),
    )

    # plot phase response
    p2 = plot(2freq_vec, [input_pha output_pha],
        label=["Input Ensemble" "WF'ed Ensemble"],
        xlabel="Frequency (0 to Nyquist) [fraction of fs/2]",
        ylabel="Autocor1 phase [radians]",
        xlims=(0, 1),
    )

    # A Plot is only displayed when returned (a semicolon will suppress the return),
    # or if explicitly displayed with display(plt), gui(), or by adding show = true to your plot command.
    display(plot(p1, p2, layout = (2, 1), legend = true))
end


function unwrap(v, inplace=false)
    # currently assuming a matrix
    unwrapped = inplace ? v : copy(v)
    N,M = size(v)
    for row in 1:N
        for i in 2:M
            d = unwrapped[row, i] - unwrapped[row, i-1]
            if abs(d) > π
                unwrapped[row, i] -= floor((d+π) / (2π)) * 2π
            end
        end
    end

    return unwrapped
end


"""
    Func from Karl
"""
function EvaluateWallFilter(wf, PhaseOffset=0, PhaseMultiplier=1)
    # Matlab, old: Assume out (Nx1) = WF (NxM) x in (Mx1)
    # M = ensemble size, N = number of filters(output samples)
    # Assume in (1xM) x WF (MxN) = out (1xN)
    M, N = size(wf)

    # Assume Real
    f = 0:0.01:1
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

    p1 = plot(f, 20log10.(abs.(fout)),
        title="Magnitude Consistency Across Wall Filter Polyphases",
        ylabel="Abs [log a.u.]",
        xlims=(0, 1),
        ylims=(-60, Inf),
    )
    # plot phase response
    # unwrap starting from the right (Nyquist)
    p2 = plot(f, angle.(fout),
        title="Phase Consistency Across Wall Filter Polyphases",
        xlabel="Frequency (0 to Nyquist) [fraction of fs/2]",
        ylabel="Phase [radians]",
        xlims=(0, 1),
    )
    display(plot(p1, p2, layout = (2, 1), legend = true))
end


end # module


# FilterOptimizer.main_simple_plot();
FilterOptimizer.main_objective();

# K88 = [ 0.101928620359513  -0.144692907668316   0.001818686042631   0.034079418317095   0.017585080356053   0.001315214737816  -0.003444416529060  -0.001436438714533
#        -0.182345096610246   0.360756599962924  -0.147994984888741  -0.059129118196972   0.002868426988251   0.015712495777574   0.007502730016391  -0.002913799498315
#         0.325318278686291  -0.644192151825324   0.366462705585208  -0.039188548203213  -0.002501377627620   0.008040320659969   0.004771188361656  -0.001218136494786
#         0.214429240579235   0.020907800279936  -0.640407310078860   0.438171965206373  -0.001984312130525   0.000672744414457   0.000815968852531   0.000020116755318
#         0.039388771853078   0.158515140964055   0.021611283755305  -0.627238094500448   0.444963982609944  -0.001482793266190  -0.000658657133152   0.000289435629589
#        -0.028159300455806   0.079366422909548   0.158022482575611   0.012192601874105  -0.632146169468256   0.444503863315704  -0.000536386191742   0.000149049199835
#        -0.023962585339180   0.005858919704412   0.078943901425177   0.150008766716844   0.008032778760745  -0.632505199692507   0.445310958035830   0.000012891795086
#        -0.006277091297826  -0.015051750417134   0.005747354831077   0.076845015215617   0.148923596940809   0.007947473300587  -0.632293311502353   0.445417719102865];
# FilterOptimizer.EvaluateWallFilter(K88');
