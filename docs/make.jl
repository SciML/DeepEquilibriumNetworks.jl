using Documenter, DocumenterCitations, FastDEQ

bib = CitationBibliography(joinpath(@__DIR__, "ref.bib"), sorting = :nyt)
include("pages.jl")

makedocs(
    bib,
    sitename = "Fast Deep Equilibrium Networks",
    authors = "Avik Pal et al.",
    clean = true,
    doctest = false,
    modules = [FastDEQ],
    format = Documenter.HTML(analytics = "UA-90474609-3",
                             #  assets = ["assets/favicon.ico"],
                             canonical="https://fastdeq.sciml.ai/stable/"),
    pages = pages
)

deploydocs(
   repo = "github.com/SciML/FastDEQ.jl.git";
   push_preview = true
)
