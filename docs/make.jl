using Documenter
using MCMCTempering

makedocs(
    sitename = "MCMCTempering",
    format = Documenter.HTML(),
    modules = [MCMCTempering]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
