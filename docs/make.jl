using Documenter, DocumenterCitations, DeepEquilibriumNetworks

bib = CitationBibliography(joinpath(@__DIR__, "ref.bib"); sorting=:nyt)

include("pages.jl")

makedocs(bib; sitename="Fast Deep Equilibrium Networks", authors="Avik Pal et al.",
         clean=true, doctest=false, modules=[DeepEquilibriumNetworks],
         strict=[
           :doctest,
           :linkcheck,
           :parse_error,
           :example_block,
           # Other available options are
           # :autodocs_block, :cross_references, :docs_block, :eval_block, :example_block,
           # :footnote, :meta_block, :missing_docs, :setup_block
         ], checkdocs=:all,
         format=Documenter.HTML(; assets=["assets/favicon.ico"],
                                canonical="https://docs.sciml.ai/DeepEquilibriumNetworks/stable/"),
         pages=pages)

deploydocs(; repo="github.com/SciML/DeepEquilibriumNetworks.jl.git", push_preview=true)
