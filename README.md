# Differentiable Phenomenological Model of CDMFT
Research project on phenomenological model of CDMFT results fitted with neural networks

# Three band emery model
The current version implements a fit of the Emery 3-bands model of cuprates. You can run it with

    python fit.py

Additional options are available, you can prompt

    python fit.py --help

to see them. For example

    python fit.py --anim

will produce a gif animation of the progression of the fit. From the three band model with

    mu = 5.8
    tpp = 1
    tpp' = 0.2
    tpd = 2.1
    ed = 0
    ep = 2.5

The resulting one-band fit yields:

    t = 0.92
    mu = -0.9 * t
    tp = -0.18 * t
    tpp = 0.12 * t

If one rather uses:

    mu = 5.8
    tpp = 1
    tpp' = 0.2
    tpd = 2.1
    ed = 0
    ep = 2.5

The resulting one-band fit yields:

    t = 0.92
    mu = -0.9 * t
    tp = -0.18 * t
    tpp = 0.12 * t

