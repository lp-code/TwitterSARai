# Run the function locally OR deploy it to the Azure Function whose name is passed.
param (
    [string]$target = "localhost"
)

# The files that are required by the function are placed in the function's directory,
# in this way, they are available when running locally and they are included in the
# zip file when deployed to Azure.
Copy-Item "..\..\models\model.joblib" -Destination "SarInference" -Force
Copy-Item "..\data\data_utils.py" -Destination "SarInference" -Force

if ($target -eq "localhost") {
    func host start
}
else {
    func azure functionapp publish $target
}

# Remove the copied items again.
Remove-Item "SarInference\model.joblib"
Remove-Item "SarInference\data_utils.py"
