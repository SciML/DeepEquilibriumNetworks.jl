using Documenter, DocumenterCitations, DeepEquilibriumNetworks

bib = CitationBibliography(joinpath(@__DIR__, "ref.bib"); style = :authoryear)

include("pages.jl")

DocMeta.setdocmeta!(
    DeepEquilibriumNetworks, :DocTestSetup, quote
        using DeepEquilibriumNetworks, Lux, NonlinearSolve, Random, SteadyStateDiffEq
    end; recursive = true
)

makedocs(;
    sitename = "Deep Equilibrium Networks",
    authors = "Avik Pal et al.",
    modules = [DeepEquilibriumNetworks],
    clean = true,
    doctest = true,
    linkcheck = true,
    format = Documenter.HTML(;
        assets = ["assets/favicon.ico"],
        canonical = "https://docs.sciml.ai/DeepEquilibriumNetworks/stable/"
    ),
    plugins = [bib],
    pages
)

deploydocs(; repo = "github.com/SciML/DeepEquilibriumNetworks.jl.git", push_preview = true)
