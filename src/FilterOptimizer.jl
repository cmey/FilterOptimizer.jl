module FilterOptimizer

using DSP
using Optim
using Plots
using Statistics


function main_simple_plot()
    ensemble_size = 8
    cutoff_frac = 0.1  # [fraction of Nyquist, i.e. of fs/2]

    input_vec, freq_vec = make_input(ensemble_size)

    filter_mat = make_filter(ensemble_size, cutoff_frac)

    output_vec = apply_wallfilter(input_vec, filter_mat)

    plot_response(freq_vec, make_response(input_vec), make_response(output_vec), ensemble_size)
end


function main_objective()
    ensemble_size = 8
    cutoff_frac = 0.1  # [fraction of fs]
    num_freqs = 100

    @assert 0 <= cutoff_frac <= 0.5
    cutoff_frac = cutoff_frac * 2  # [fraction of Nyquist, i.e. of fs/2]
    @assert 0 <= cutoff_frac <= 1

    input_vec, freq_vec = make_input(ensemble_size, num_freqs)

    desired_response = make_desired_response(cutoff_frac, num_freqs)
    @show desired_response

    filter_mat = make_filter(ensemble_size, cutoff_frac)

    function optim_iter(x)
        output_vec = apply_wallfilter(input_vec, x)
        output_response = make_response(output_vec)
        loss = loss_function(desired_response, output_response)
    end

    # Continuous, multivariate, optimization
    # Simulated Annealing, Evolutionary algo
    # CG, etc.: Compute / Auto Grad of f?
    # SPSA: black box, doesnt need gradient of f. Perturbs all params at the same time.
    # FD: black box, doesnt need gradient of f. Perturbs one param at a time.

    x0 = filter_mat
    result = optimize(optim_iter, x0, BFGS(); autodiff = :forward)
    @show result

    optimized_filter = Optim.minimizer(result)
    output_vec = apply_wallfilter(input_vec, optimized_filter)
    output_response = make_response(output_vec)

    plot_response(freq_vec, make_response(input_vec), output_response, ensemble_size)
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


function make_filter(ensemble_size, cutoff_frac)
    responsetype = Highpass(cutoff_frac)
    designmethod = Butterworth(4)
    digitalfilter(responsetype, designmethod)
    # need randn and not rand b/c need some negative coefficients in order to get a highpass filter
    randn(ensemble_size, ensemble_size)
end


function apply_wallfilter(input_vec, filter_mat)
    input_vec * complex(filter_mat)
end


function make_response(vec)
    mag = abs.(autocorrelation(vec, lag=0))
    pha = angle.(autocorrelation(vec, lag=1))
    (mag, pha)
end


function make_desired_response(cutoff_frac, num_freqs=100)
    @assert 0 <= cutoff_frac <= 1
    n_stops = Int(round(num_freqs * cutoff_frac))
    n_pass = num_freqs - n_stops
    mag = [repeat([0], n_stops) ; repeat([1], n_pass)]
    phase = collect(linspace_with_endpoint(0, π, num_freqs))
    (mag, phase)
end


"""Autocorrelation over 1st dimension for the given non-negative lag of 0, 1, 2, ...
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


function plot_response(freq_vec, input_response, output_response, ensemble_size)
    input_mag, input_pha = input_response
    output_mag, output_pha = output_response

    # plot frequency response (should attenuate lower freqs)
    # fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    # ax1.set_ylim(bottom=-30)
    # ax1.set_title("Ensemble = {}".format(ensemble_size))
    p1 = plot(freq_vec, [20log10.(input_mag) 20log10.(output_mag)],
        label=["Input Ensemble" "WF'ed Ensemble"],
        title="Autocorrelation (frequency) response of wall filter, Ens=$ensemble_size",
        ylabel="Autocor0 abs [log a.u.]",
    )

    # plot phase response
    p2 = plot(freq_vec, [input_pha output_pha],
        label=["Input Ensemble" "WF'ed Ensemble"],
        xlabel="Frequency (0 to Nyquist) [fraction of fs]",
        ylabel="Autocor1 phase [radians]",
    )

    plot(p1, p2, layout = (2, 1), legend = true)
end


end # module


# FilterOptimizer.main_simple_plot()
FilterOptimizer.main_objective()

