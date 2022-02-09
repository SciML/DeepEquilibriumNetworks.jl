using Documenter, DocumenterCitations, FastDEQ

bib = CitationBibliography("ref.bib", sorting = :nyt)

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
        "API" => [
            "Dynamical Systems" => "api/solvers.md",
            "Non Linear Solvers" => "api/nlsolve.md",
            "General Purpose Layers" => "api/layers.md",
            "DEQ Layers" => "api/deqs.md",
            "Miscellaneous" => "api/misc.md",
        ],
        "References" => "references.md",
    ]
)