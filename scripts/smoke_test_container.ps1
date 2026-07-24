[CmdletBinding()]
param(
    [string]$ImageTag = "jmm-energy-api:smoke-test",
    [string]$ContainerName = "jmm-energy-api-smoke-test",
    [ValidateRange(1, 65535)]
    [int]$HostPort = 8000,
    [ValidateRange(10, 300)]
    [int]$StartupTimeoutSeconds = 90,
    [switch]$NoCache
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$BaseUri = "http://127.0.0.1:$HostPort"
$ExpectedPredictionKwh = 18424.13407399847
$PredictionToleranceKwh = 0.000001

function Invoke-DockerCommand {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    & docker @Arguments

    if ($LASTEXITCODE -ne 0) {
        throw (
            "Docker command failed with exit code {0}: docker {1}" -f
            $LASTEXITCODE,
            ($Arguments -join " ")
        )
    }
}

function Assert-Equal {
    param(
        [Parameter(Mandatory = $true)]
        $Actual,
        [Parameter(Mandatory = $true)]
        $Expected,
        [Parameter(Mandatory = $true)]
        [string]$Message
    )

    if ($Actual -ne $Expected) {
        throw (
            "{0}. Expected: {1}. Actual: {2}." -f
            $Message,
            $Expected,
            $Actual
        )
    }
}

function Assert-Near {
    param(
        [Parameter(Mandatory = $true)]
        [double]$Actual,
        [Parameter(Mandatory = $true)]
        [double]$Expected,
        [Parameter(Mandatory = $true)]
        [double]$Tolerance,
        [Parameter(Mandatory = $true)]
        [string]$Message
    )

    $Difference = [math]::Abs($Actual - $Expected)

    if ($Difference -gt $Tolerance) {
        throw (
            "{0}. Expected: {1}. Actual: {2}. Difference: {3}. " +
            "Tolerance: {4}." -f
            $Message,
            $Expected,
            $Actual,
            $Difference,
            $Tolerance
        )
    }
}

if (-not (Test-Path ".\Dockerfile")) {
    throw (
        "Dockerfile was not found. Run this script from the " +
        "repository root."
    )
}

if (-not (Test-Path ".\models\ensemble_2_0_model.joblib")) {
    throw "The official serving model artifact was not found."
}

if (-not (Test-Path ".\config\serving\model_metadata.json")) {
    throw "The serving model metadata was not found."
}

$ContainerStarted = $false

try {
    Write-Host "Checking Docker engine..."
    Invoke-DockerCommand -Arguments @(
        "info",
        "--format",
        "Docker Server Version: {{.ServerVersion}}"
    )

    $ExistingContainerNames = @(
        & docker ps -a --format "{{.Names}}"
    )

    if ($LASTEXITCODE -ne 0) {
        throw "Unable to list Docker containers."
    }

    if ($ExistingContainerNames -contains $ContainerName) {
        Write-Host "Removing previous smoke-test container..."
        Invoke-DockerCommand -Arguments @(
            "rm",
            "--force",
            $ContainerName
        )
    }

    $BuildArguments = @(
        "build",
        "--tag",
        $ImageTag
    )

    if ($NoCache) {
        $BuildArguments += "--no-cache"
    }

    $BuildArguments += "."

    Write-Host "Building Docker image..."
    Invoke-DockerCommand -Arguments $BuildArguments

    Write-Host "Starting smoke-test container..."
    Invoke-DockerCommand -Arguments @(
        "run",
        "--detach",
        "--name",
        $ContainerName,
        "--publish",
        "${HostPort}:8000",
        $ImageTag
    )

    $ContainerStarted = $true

    Write-Host "Waiting for API readiness..."
    $Deadline = (Get-Date).AddSeconds(
        $StartupTimeoutSeconds
    )
    $HealthResponse = $null
    $ReadyResponse = $null

    while ((Get-Date) -lt $Deadline) {
        try {
            $HealthResponse = Invoke-RestMethod `
                -Uri "$BaseUri/health" `
                -TimeoutSec 5

            if ($HealthResponse.status -eq "ok") {
                $ReadyResponse = Invoke-RestMethod `
                    -Uri "$BaseUri/ready" `
                    -TimeoutSec 5

                if ($ReadyResponse.status -eq "ready") {
                    break
                }
            }
        }
        catch {
            $HealthResponse = $null
            $ReadyResponse = $null
        }

        Start-Sleep -Seconds 2
    }

    if ($null -eq $ReadyResponse) {
        Write-Host "Container logs:"
        & docker logs $ContainerName
        throw (
            "The API did not become ready within {0} seconds." -f
            $StartupTimeoutSeconds
        )
    }

    Assert-Equal `
        -Actual $HealthResponse.status `
        -Expected "ok" `
        -Message "Health endpoint returned an unexpected status"

    Assert-Equal `
        -Actual $ReadyResponse.status `
        -Expected "ready" `
        -Message "Readiness endpoint returned an unexpected status"

    Assert-Equal `
        -Actual $ReadyResponse.modeling_version `
        -Expected "2.0" `
        -Message "Unexpected modeling version"

    Assert-Equal `
        -Actual $ReadyResponse.target `
        -Expected "active_energy_kWh" `
        -Message "Unexpected prediction target"

    Write-Host "Testing reference single prediction..."
    $PredictionBody = @{
        date = "2025-10-31"
        total_kg = 190741
        total_nominal_kg = 190631.01
        total_brix_units = 1931393
        total_hours = 58.93
        total_pallets = 0
        orders = 4
        avg_brix = 10.1411869
    } | ConvertTo-Json

    $PredictionResponse = Invoke-RestMethod `
        -Method Post `
        -Uri "$BaseUri/v1/predict" `
        -ContentType "application/json" `
        -Headers @{
            "X-Request-ID" = "automated-smoke-single"
        } `
        -Body $PredictionBody `
        -TimeoutSec 30

    Assert-Equal `
        -Actual $PredictionResponse.modeling_version `
        -Expected "2.0" `
        -Message "Single prediction used an unexpected model version"

    Assert-Equal `
        -Actual $PredictionResponse.target `
        -Expected "active_energy_kWh" `
        -Message "Single prediction returned an unexpected target"

    Assert-Near `
        -Actual ([double]$PredictionResponse.prediction_kwh) `
        -Expected $ExpectedPredictionKwh `
        -Tolerance $PredictionToleranceKwh `
        -Message "Reference prediction changed"

    Assert-Equal `
        -Actual $PredictionResponse.branch_disagreement_status `
        -Expected "moderate" `
        -Message "Unexpected branch disagreement status"

    Assert-Equal `
        -Actual $PredictionResponse.operational_range_status `
        -Expected "inside_typical_development_range" `
        -Message "Unexpected operational range status"

    if (
        $PredictionResponse.warning_codes -notcontains
        "UNSEEN_CALENDAR_VALUE"
    ) {
        throw (
            "The reference prediction did not report the expected " +
            "UNSEEN_CALENDAR_VALUE warning."
        )
    }

    if (
        $PredictionResponse.warning_codes -notcontains
        "MODERATE_BRANCH_DISAGREEMENT"
    ) {
        throw (
            "The reference prediction did not report the expected " +
            "MODERATE_BRANCH_DISAGREEMENT warning."
        )
    }

    Write-Host "Testing batch prediction consistency..."
    $BatchBody = @{
        records = @(
            @{
                date = "2025-10-31"
                total_kg = 190741
                total_nominal_kg = 190631.01
                total_brix_units = 1931393
                total_hours = 58.93
                total_pallets = 0
                orders = 4
                avg_brix = 10.1411869
            },
            @{
                date = "2025-11-01"
                total_kg = 180000
                total_nominal_kg = 179500
                total_brix_units = 1818000
                total_hours = 55
                total_pallets = 0
                orders = 4
                avg_brix = 10.1
            }
        )
    } | ConvertTo-Json -Depth 10

    $BatchResponse = Invoke-RestMethod `
        -Method Post `
        -Uri "$BaseUri/v1/predict/batch" `
        -ContentType "application/json" `
        -Headers @{
            "X-Request-ID" = "automated-smoke-batch"
        } `
        -Body $BatchBody `
        -TimeoutSec 30

    Assert-Equal `
        -Actual ([int]$BatchResponse.count) `
        -Expected 2 `
        -Message "Batch endpoint returned an unexpected record count"

    Assert-Equal `
        -Actual $BatchResponse.predictions[0].date `
        -Expected "2025-10-31" `
        -Message "Batch endpoint did not preserve record order"

    Assert-Equal `
        -Actual $BatchResponse.predictions[1].date `
        -Expected "2025-11-01" `
        -Message "Batch endpoint did not preserve record order"

    Assert-Near `
        -Actual (
            [double]$BatchResponse.predictions[0].prediction_kwh
        ) `
        -Expected (
            [double]$PredictionResponse.prediction_kwh
        ) `
        -Tolerance $PredictionToleranceKwh `
        -Message (
            "Single and one-row-equivalent batch predictions differ"
        )

    Write-Host "Testing operational metrics..."
    $MetricsResponse = Invoke-WebRequest `
        -Uri "$BaseUri/metrics" `
        -UseBasicParsing `
        -TimeoutSec 30

    Assert-Equal `
        -Actual ([int]$MetricsResponse.StatusCode) `
        -Expected 200 `
        -Message "Metrics endpoint returned an unexpected status"

    $MetricsText = [string]$MetricsResponse.Content

    $RequiredMetricNames = @(
        "jmm_http_requests_total",
        "jmm_http_request_duration_seconds",
        "jmm_prediction_records_total",
        "jmm_operational_range_total",
        "jmm_calendar_coverage_total",
        "jmm_branch_disagreement_total",
        "jmm_warning_codes_total"
    )

    foreach ($MetricName in $RequiredMetricNames) {
        if ($MetricsText -notmatch [regex]::Escape($MetricName)) {
            throw (
                "Metrics endpoint did not expose required metric: {0}" -f
                $MetricName
            )
        }
    }

    if (
        $MetricsText -notmatch
        'jmm_prediction_records_total\{endpoint="single"\}\s+1(?:\.0)?'
    ) {
        throw (
            "Metrics did not record exactly one successful single " +
            "prediction."
        )
    }

    if (
        $MetricsText -notmatch
        'jmm_prediction_records_total\{endpoint="batch"\}\s+2(?:\.0)?'
    ) {
        throw (
            "Metrics did not record exactly two successful batch " +
            "prediction records."
        )
    }

    Write-Host ""
    Write-Host "Docker smoke test PASSED."
    Write-Host (
        "Reference prediction: {0} kWh" -f
        $PredictionResponse.prediction_kwh
    )
    Write-Host "Batch records validated: 2"
}
finally {
    if ($ContainerStarted) {
        Write-Host "Removing smoke-test container..."
        & docker rm --force $ContainerName | Out-Null
    }
}