param([switch]$Hardware)
$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$repoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $repoRoot
try {
    $arguments = & .\run.ps1 -DryRun | ConvertFrom-Json
    if ('--kiosk' -notin $arguments -or '--juggernaut-local-only' -notin $arguments) {
        throw 'Launcher must default to offline kiosk mode'
    }
    $social = & .\run.ps1 -Mode social -DryRun -JuggernautGuidedStrength 0 | ConvertFrom-Json
    if ('--run-social-pipeline' -notin $social -or '--juggernaut-guided-strength' -notin $social) {
        throw 'Diagnostic mode or explicit overrides were lost'
    }
    foreach ($test in @('performance_recommendations_test', 'kiosk_app_test', 'kiosk_session_test',
        'kiosk_framing_test', 'juggernaut_runner_test', 'delivery_test')) {
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
        --compose-prop-image assets/props/lenovo.laptop.png `
        --compose-output-image outputs/pipeline_deterministic_test.jpg `
        --final-output-image outputs/pipeline_final_test.jpg `
        --juggernaut-openvino-cache-dir outputs/openvino_cache/juggernaut `
        --juggernaut-skip-guided `
        --capture-output outputs/pipeline_capture_test.jpg `
        --capture-rmbg-mask-output outputs/pipeline_mask_test.png `
        --capture-rmbg-foreground-output outputs/pipeline_fg_test.png
    if ($LASTEXITCODE -ne 0) { throw 'Hardware pipeline failed' }
}
finally {
    Pop-Location
}
