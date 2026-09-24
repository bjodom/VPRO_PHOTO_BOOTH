param([switch]$Hardware)
$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$repoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $repoRoot
try {
    $arguments = & .\run.ps1 -DryRun | ConvertFrom-Json
    if ('--kiosk' -notin $arguments -or '--generation-local-only' -notin $arguments) {
        throw 'Launcher must default to offline kiosk mode'
    }
    if ($arguments[[array]::IndexOf($arguments, '--rmbg-device') + 1] -ne 'CPU') {
        throw 'Launcher must default RMBG to CPU'
    }
    if ($arguments[[array]::IndexOf($arguments, '--generation-guidance-scale') + 1] -ne 7.0) {
        throw 'DreamShaper must default to CFG 7'
    }
    $juggernaut = & .\run.ps1 -GenerationEngine Juggernaut -DryRun | ConvertFrom-Json
    if ('--generation-guidance-scale' -in $juggernaut) {
        throw 'Juggernaut must keep its CLI default CFG unless explicitly overridden'
    }
    $gpu = & .\run.ps1 -RmbgDevice GPU -DryRun | ConvertFrom-Json
    if ($gpu[[array]::IndexOf($gpu, '--rmbg-device') + 1] -ne 'GPU') {
        throw 'Launcher must preserve the explicit RMBG GPU override'
    }
    $override = & .\run.ps1 -GenerationEngine DreamShaper -GenerationGuidanceScale 6 -DryRun | ConvertFrom-Json
    if ($override[[array]::IndexOf($override, '--generation-guidance-scale') + 1] -ne 6.0) {
        throw 'Explicit generation guidance override must win'
    }
    $legacyOverride = & .\run.ps1 -GenerationEngine DreamShaper -JuggernautGuidanceScale 6 -DryRun | ConvertFrom-Json
    if ($legacyOverride[[array]::IndexOf($legacyOverride, '--generation-guidance-scale') + 1] -ne 6.0) {
        throw 'Legacy Juggernaut guidance override must remain compatible'
    }
    $stepOverride = & .\run.ps1 -GenerationSteps 28 -DryRun | ConvertFrom-Json
    if ($stepOverride[[array]::IndexOf($stepOverride, '--generation-steps') + 1] -ne 28) {
        throw 'Neutral generation step override must be forwarded'
    }
    $social = & .\run.ps1 -Mode social -DryRun -JuggernautGuidedStrength 0 | ConvertFrom-Json
    if ('--run-social-pipeline' -notin $social -or '--generation-guided-strength' -notin $social) {
        throw 'Diagnostic mode or explicit overrides were lost'
    }
    foreach ($test in @('performance_recommendations_test', 'kiosk_app_test', 'kiosk_session_test',
        'kiosk_framing_test', 'juggernaut_runner_test', 'delivery_test', 'pose_device_test')) {
        & .\.venv\Scripts\python.exe "tests/$test.py"
        if ($LASTEXITCODE -ne 0) { throw "$test failed" }
    }
    if (-not $Hardware) { return }
    & .\.venv\Scripts\python.exe -m vpro.cli `
        --run-social-pipeline `
        --backend openvino `
        --model-path models/yolo26/yolo26x-pose_openvino_model `
        --device intel:gpu `
        --camera-index 0 `
        --capture-auto-start `
        --capture-delay-seconds 3 `
        --capture-wrist-stable-seconds 0.7 `
        --warmup-frames 6 `
        --compose-scene-image assets/scenes/portrait_scene_1080x1350.jpg `
        --compose-output-image outputs/pipeline_deterministic_test.jpg `
        --final-output-image outputs/pipeline_final_test.jpg `
        --generation-openvino-cache-dir outputs/openvino_cache/juggernaut `
        --generation-skip-guided `
        --capture-output outputs/pipeline_capture_test.jpg `
        --capture-rmbg-mask-output outputs/pipeline_mask_test.png `
        --capture-rmbg-foreground-output outputs/pipeline_fg_test.png
    if ($LASTEXITCODE -ne 0) { throw 'Hardware pipeline failed' }
}
finally {
    Pop-Location
}
