using FastMultipole
using Documenter
using DocumenterCitations

DocMeta.setdocmeta!(FastMultipole, :DocTestSetup, :(using FastMultipole); recursive=true)

bib = CitationBibliography(joinpath(@__DIR__, "src", "refs.bib"))

makedocs(;
    plugins=[bib],
    modules=[FastMultipole],
    authors="Ryan Anderson <rymanderson@gmail.com> and contributors",
    sitename="FastMultipole.jl",
    format=Documenter.HTML(;
        canonical="https://flow.byu.edu/FastMultipole.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
		"Introduction" => "index.md",
        "Quick Start" => "quickstart.md",
        "Gravitational Example" => "guided_examples.md",
        "Vortex Filament Example" => "vortex_filament.md",
        "Tuning Parameters" => "tuning.md",
        "Multiple Systems" => "advanced_usage.md",
        "Automated Tuning" => "advanced_usage_2.md",
        "Reference" => "reference.md",
        # "Theory" => "theory.md"
    ],
    checkdocs=:none
)

deploydocs(;
    repo="github.com/byuflowlab/FastMultipole.jl",
    devbranch="main",
)
