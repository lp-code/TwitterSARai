# Run the function locally OR deploy it to the Azure Function whose name is passed.
#
# The Azure function core tools (or the Linux consumption plan) does not seem to
# support publishing to slots, even though they can be created in the portal.
# It is not described anywhere how to do it, and copying from one slot to another
# is not implemented either. So the only choice is to overwrite straight into
# the only (production) slot.

param (
    [string]$target = "localhost"
)

# The files that are required by the function are placed in the function's directory,
# in this way, they are available when running locally and they are included in the
# zip file when deployed to Azure.
Copy-Item "..\..\models\model.joblib" -Destination "SarInference" -Force
Copy-Item "..\data\data_utils.py" -Destination "SarInference" -Force
Copy-Item "..\models\model_utils.py" -Destination "SarInference" -Force

if ($target -eq "localhost") {
    func host start
}
else {
    func azure functionapp publish $target
}

# Remove the copied items again.
Remove-Item "SarInference\model.joblib"
Remove-Item "SarInference\data_utils.py"
Remove-Item "SarInference\model_utils.py"
