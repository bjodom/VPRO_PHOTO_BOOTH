$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$repoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $repoRoot
try {
    uv run vpro `
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
}
finally {
    Pop-Location
}
