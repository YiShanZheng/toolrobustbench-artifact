param(
    [string]$Config = "configs/paper_all_family.yaml",
    [string]$ModelsConfig = "configs/models.yaml",
    [string]$ResultsRoot = "data/results/all-family",
    [string]$TempConfigDir = "data/generated/run_configs/all-family",
    [string[]]$Models = @(
        "gpt-5.4-mini", #
        "claude-sonnet-4-6", #
        "gemini-2.5-flash", #
        "gemini-2.5-pro", #
        "deepseek-v4-flash", #
        "deepseek-v4-pro", #
        "qwen-plus" #
    ),
    [string[]]$Families = @(
        "interface_exposure",
        "user_intent",
        "observation",
        "execution_environment"
    ),
    [switch]$BuildDataOnly,
    [switch]$Resume,
    [switch]$NoProgress
)

$ErrorActionPreference = "Stop"
$env:PYTHONPATH = "src"

function Convert-ToSlug([string]$Value) {
    return ($Value -replace '[\\/:*?"<>|\s]+', '_')
}

New-Item -ItemType Directory -Force -Path $TempConfigDir | Out-Null
New-Item -ItemType Directory -Force -Path $ResultsRoot | Out-Null

$runPlans = @()
foreach ($model in $Models) {
    $modelSlug = Convert-ToSlug $model
    foreach ($family in $Families) {
        $runName = "paper_all_family_${modelSlug}_${family}"
        $tempConfig = Join-Path $TempConfigDir "$runName.yaml"
        $resultsDir = Join-Path (Join-Path $ResultsRoot $modelSlug) $family

        python scripts/write_paper_run_config.py `
            --base $Config `
            --output $tempConfig `
            --model $model `
            --experiment-name $runName `
            --families $family

        $countArgs = @("scripts/count_run_items.py", "--config", $tempConfig, "--models-config", $ModelsConfig, "--results-dir", $resultsDir)
        if ($Resume) {
            $countArgs += "--resume"
        }
        $countPayload = python @countArgs | ConvertFrom-Json
        $runPlans += [pscustomobject]@{
            Model = $model
            Family = $family
            Label = "${model}-${family}"
            TempConfig = $tempConfig
            ResultsDir = $resultsDir
            Total = [int]$countPayload.remaining
        }
    }
}

$progressTotal = ($runPlans | Measure-Object -Property Total -Sum).Sum
if ($null -eq $progressTotal) {
    $progressTotal = 0
}
$progressTotal = [int]$progressTotal
$progressOffset = 0

foreach ($plan in $runPlans) {
        if ($BuildDataOnly) {
            $buildArgs = @(
                "-m", "toolrobust.cli.main",
                "--config", $plan.TempConfig,
                "--models-config", $ModelsConfig,
                "--progress-label", $plan.Label,
                "--progress-offset", $progressOffset,
                "--progress-total", $progressTotal,
                "build-data"
            )
            if ($NoProgress) {
                $buildArgs += "--no-progress"
            }
            python @buildArgs
        }
        else {
            $runArgs = @(
                "-m", "toolrobust.cli.main",
                "--config", $plan.TempConfig,
                "--models-config", $ModelsConfig,
                "--progress-label", $plan.Label,
                "--progress-offset", $progressOffset,
                "--progress-total", $progressTotal,
                "run",
                "--results-dir", $plan.ResultsDir
            )
            if ($Resume) {
                $runArgs += "--resume"
            }
            if ($NoProgress) {
                $runArgs += "--no-progress"
            }
            python @runArgs
        }
    $progressOffset += $plan.Total
}
