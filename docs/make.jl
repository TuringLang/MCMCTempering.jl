using Documenter
using MCMCTempering

DocMeta.setdocmeta!(MCMCTempering, :DocTestSetup, :(using MCMCTempering); recursive=true)

makedocs(
    sitename = "MCMCTempering",
    format = Documenter.HTML(),
    modules = [MCMCTempering],
    pages=["Home" => "index.md", "getting-started.md", "api.md", "design.md"],
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
