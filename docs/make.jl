using Documenter
using MCMCTempering

DocMeta.setdocmeta!(MCMCTempering, :DocTestSetup, :(using MCMCTempering); recursive=true)

makedocs(
    sitename = "MCMCTempering",
    format = Documenter.HTML(),
    modules = [MCMCTempering],
    pages=["Home" => "index.md", "getting-started.md", "api.md"],
)

# Deply!
deploydocs(; repo="github.com/TuringLang/MCMCTempering.jl.git", push_preview=true)
