param(
    [string]$Config = "configs/paper_cross_family.yaml",
    [string]$ModelsConfig = "configs/models.yaml",
    [string]$ResultsRoot = "data/results/cross-family",
    [string]$TempConfigDir = "data/generated/run_configs/cross-family",
    [string[]]$Models = @(
        "gpt-5.4-mini", #
        "claude-sonnet-4-6", #
        "gemini-2.5-flash", #
        "gemini-2.5-pro", 
        "deepseek-v4-flash", #
        "deepseek-v4-pro", 
        "qwen-plus" #
    ),
    [string[]]$FamilyPairs = @(
        "interface_exposure,user_intent",
        "interface_exposure,observation",
        "interface_exposure,execution_environment",
        "user_intent,observation",
        "user_intent,execution_environment"
    ),
    [string[]]$CompositionIds = @(),
    [switch]$BuildDataOnly,
    [switch]$Resume,
    [switch]$NoProgress,
    [switch]$DisableStage2CompatGuard,
    [int]$Stage2CompatGuardMinIssues = 3,
    [int]$Stage2CompatGuardMinChecked = 8,
    [double]$Stage2CompatGuardFailureRate = 0.5
)

$ErrorActionPreference = "Stop"
$env:PYTHONPATH = "src"

function Convert-ToSlug([string]$Value) {
    return ($Value -replace '[\\/:*?"<>|\s,]+', '_')
}

New-Item -ItemType Directory -Force -Path $TempConfigDir | Out-Null
New-Item -ItemType Directory -Force -Path $ResultsRoot | Out-Null

$runPlans = @()
$compositionFilter = @($CompositionIds | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
foreach ($model in $Models) {
    $modelSlug = Convert-ToSlug $model
    if ($compositionFilter.Count -gt 0) {
        $runName = "paper_cross_family_${modelSlug}_selected_compositions"
        $tempConfig = Join-Path $TempConfigDir "$runName.yaml"
        $resultsDir = Join-Path (Join-Path $ResultsRoot $modelSlug) "selected_compositions"

        python scripts/write_paper_run_config.py `
            --base $Config `
            --output $tempConfig `
            --model $model `
            --experiment-name $runName `
            --cross-family `
            --composition-ids ($compositionFilter -join ",")

        $countArgs = @("scripts/count_run_items.py", "--config", $tempConfig, "--models-config", $ModelsConfig, "--results-dir", $resultsDir)
        if ($Resume) {
            $countArgs += "--resume"
        }
        $countPayload = python @countArgs | ConvertFrom-Json
        $runPlans += [pscustomobject]@{
            Model = $model
            FamilyPair = "selected_compositions"
            Label = "${model}-selected-compositions"
            TempConfig = $tempConfig
            ResultsDir = $resultsDir
            Total = [int]$countPayload.remaining
        }
        continue
    }

    foreach ($pair in $FamilyPairs) {
        $pairSlug = Convert-ToSlug $pair
        $runName = "paper_cross_family_${modelSlug}_${pairSlug}"
        $tempConfig = Join-Path $TempConfigDir "$runName.yaml"
        $resultsDir = Join-Path (Join-Path $ResultsRoot $modelSlug) $pairSlug

        python scripts/write_paper_run_config.py `
            --base $Config `
            --output $tempConfig `
            --model $model `
            --experiment-name $runName `
            --family-set $pair `
            --cross-family

        $countArgs = @("scripts/count_run_items.py", "--config", $tempConfig, "--models-config", $ModelsConfig, "--results-dir", $resultsDir)
        if ($Resume) {
            $countArgs += "--resume"
        }
        $countPayload = python @countArgs | ConvertFrom-Json
        $runPlans += [pscustomobject]@{
            Model = $model
            FamilyPair = $pair
            Label = "${model}-${pairSlug}"
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
            if ($DisableStage2CompatGuard) {
                $runArgs += "--disable-stage2-compat-guard"
            }
            $runArgs += @(
                "--stage2-compat-guard-min-issues", $Stage2CompatGuardMinIssues,
                "--stage2-compat-guard-min-checked", $Stage2CompatGuardMinChecked,
                "--stage2-compat-guard-failure-rate", $Stage2CompatGuardFailureRate
            )
            python @runArgs
        }
    $progressOffset += $plan.Total
}
