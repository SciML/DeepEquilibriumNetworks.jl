using Documenter, DocumenterCitations, FastDEQ

bib = CitationBibliography(joinpath(@__DIR__, "ref.bib"), sorting = :nyt)

makedocs(
    bib,
    sitename = "Fast Deep Equilibrium Networks",
    authors = "Avik Pal et al.",
    clean = true,
    doctest = false,
    modules = [FastDEQ],
    format = Documenter.HTML(#  analytics = "",
                             #  assets = ["assets/favicon.ico"],
                             canonical="https://fastdeq.sciml.ai/stable/"),
    pages = [
        "FastDEQ: Fast Deep Equilibrium Networks" => "index.md",
        "Manual" => [
            "Dynamical Systems" => "manual/solvers.md",
            "Non Linear Solvers" => "manual/nlsolve.md",
            "General Purpose Layers" => "manual/layers.md",
            "DEQ Layers" => "manual/deqs.md",
            "Miscellaneous" => "manual/misc.md",
        ],
        "References" => "references.md",
    ]
)

deploydocs(
   repo = "github.com/SciML/FastDEQ.jl.git";
   push_preview = true
)
