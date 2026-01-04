using Documenter, DocumenterCitations, DeepEquilibriumNetworks

cp("./docs/Manifest.toml", "./docs/src/assets/Manifest.toml"; force = true)
cp("./docs/Project.toml", "./docs/src/assets/Project.toml"; force = true)

bib = CitationBibliography(joinpath(@__DIR__, "ref.bib"); style = :authoryear)

include("pages.jl")

makedocs(;
    sitename = "Deep Equilibrium Networks",
    authors = "Avik Pal et al.",
    modules = [DeepEquilibriumNetworks],
    clean = true,
    doctest = false,  # Tested in CI
    linkcheck = true,
    format = Documenter.HTML(;
        assets = ["assets/favicon.ico"],
        canonical = "https://docs.sciml.ai/DeepEquilibriumNetworks/stable/"
    ),
    plugins = [bib],
    pages
)

deploydocs(; repo = "github.com/SciML/DeepEquilibriumNetworks.jl.git", push_preview = true)
