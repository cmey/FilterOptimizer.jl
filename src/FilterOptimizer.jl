module FilterOptimizer

using DSP
using Statistics
using Plots


function main()
    ensemble_size = 8
    pri = 1000e-6  # [s]
    cutoff_freq = 100  # [Hz]

    input_vec, freq_vec = make_input(ensemble_size)

    filter_mat = make_filter(ensemble_size, 1/pri, cutoff_freq)

    output_vec = apply_wallfilter(input_vec, filter_mat)

    response = make_response(input_vec, output_vec)

    plot_response(freq_vec * 1 / pri, response)
end


function make_input(ensemble_size, num_freqs=100)
    # generate input freqs
    time_vec = linspace_with_endpoint(0, ensemble_size-1, ensemble_size)
    freq_vec = linspace_with_endpoint(0, 1 / 2, num_freqs)
    outer = freq_vec .* time_vec'
    input_vec = exp.(1im .* outer .* 2π)
    input_vec, freq_vec
end


linspace_with_endpoint(start, last, length) = range(start, last, length=length)

linspace_without_endpoint(start, stop, length) = range(start, stop, length=length+2)[2:end-1]


function make_filter(ensemble_size,fs, cutoff_freq)
    responsetype = Highpass(cutoff_freq; fs=fs)
    designmethod = Butterworth(4)
    digitalfilter(responsetype, designmethod)
    return rand(ensemble_size, ensemble_size)
end


function apply_wallfilter(input_vec, filter_mat)
    input_vec * complex(filter_mat)
end


function make_response(input_vec, output_vec)
    input_ac0 = autocorrelation(input_vec, lag=0)
    output_ac0 = autocorrelation(output_vec, lag=0)
    input_ac1 = autocorrelation(input_vec, lag=1)
    output_ac1 = autocorrelation(output_vec, lag=1)
    (input_ac0, output_ac0, input_ac1, output_ac1)
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


function plot_response(freq_vec, response)
    input_ac0, output_ac0, input_ac1, output_ac1 = response

    # plot frequency response (should attenuate lower freqs)
    # fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    # fig.suptitle("Autocorrelation (frequency) response of wall filter")
    # ax1.plot(freq_vec, 20 * np.log10(abs(input_ac0)), label="Input Ensemble")
    # ax1.plot(freq_vec, 20 * np.log10(abs(output_ac0)), label="WF'ed Ensemble")
    # ax1.set_ylim(bottom=-30)
    # ax1.set_ylabel("Autocor0 abs [log a.u.]")
    # ax1.legend()
    # ax1.set_title("PRI = {} [µs], Ensemble = {}".format(pri * 1e6, ensemble_size))
    p1 = plot(freq_vec, [20log10.(abs.(input_ac0)) 20log10.(abs.(output_ac0))],
        label=["Input Ensemble" "WF'ed Ensemble"],
        title="Autocorrelation (frequency) response of wall filter",
        ylabel="Autocor0 abs [log a.u.]",
    )

    # # plot phase response
    # ax2.plot(freq_vec, np.angle(input_ac1), label="Input Ensemble")
    # ax2.plot(freq_vec, np.angle(output_ac1), label="WF'ed Ensemble")
    # ax2.set_xlabel("Frequency (0 to Nyquist) [Hz]")
    # ax2.set_ylabel("Autocor1 phase [radians]")
    # ax2.legend()
    p2 = plot(freq_vec, [angle.(input_ac1) angle.(output_ac1)],
        label=["Input Ensemble" "WF'ed Ensemble"],
        xlabel="Frequency (0 to Nyquist) [Hz]",
        ylabel="Autocor1 phase [radians]",
    )

    plot(p1, p2, layout = (2, 1), legend = true)
end


end # module


FilterOptimizer.main()
