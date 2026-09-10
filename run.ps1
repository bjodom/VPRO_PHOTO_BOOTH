<#!
.SYNOPSIS
    Run the complete vPRO capture -> compose -> Juggernaut pipeline.

.DESCRIPTION
    This is the primary Windows launcher for the project. It keeps paths rooted at the repository,
    optionally enables the parent-directory Intel proxy, and forwards production controls to the
    vpro CLI. Normal execution uses the project's .venv directly; uv is only needed for -Sync or
    -DownloadJuggernaut.

.EXAMPLE
    .\run.ps1

.EXAMPLE
    .\run.ps1 -UseIntelProxy -Sync -DownloadJuggernaut -CaptureAutoStart

.EXAMPLE
    .\run.ps1 -Device intel:npu -RmbgDevice NPU -JuggernautPreset identity-lock
#>

[CmdletBinding()]
param(
    [ValidateSet("openvino", "torch")]
    [string]$Backend = "openvino",
    [string]$ModelPath = "models/yolo26/yolo26x-pose_openvino_model",
    [string]$Device = "intel:gpu",
    [string]$RmbgDevice = "AUTO",
    [string]$JuggernautDevice = "GPU",
    [string]$JuggernautModelId = "OpenVINO/Juggernaut-XL-v9-fp16-ov",
    [ValidateSet("fast", "balanced", "high")]
    [string]$MaskQuality = "high",
    [ValidateSet("identity-lock", "balanced", "stylized")]
    [string]$JuggernautPreset = "balanced",
    [int]$CameraIndex = 0,
    [int]$CaptureWidth = 2560,
    [int]$CaptureHeight = 1440,
    [ValidateSet(0, 90, 180, 270)]
    [int]$CaptureRotate = 0,
    [int]$WarmupFrames = 12,
    [int]$CaptureDelaySeconds = 3,
    [double]$CaptureWristStableSeconds = 0.7,
    [string]$SceneImage = "assets/scenes/portrait_scene_1080x1350.jpg",
    [string]$PropImage = "assets/props/lenovo.laptop.png",
    [string]$RmbgModelDir = "models/rmbg/rmbg-1.4",
    [string]$ComposeOutputImage = "outputs/final_portrait_1080x1350.jpg",
    [string]$FinalOutputImage = "outputs/final_portrait_1080x1350_final.jpg",
    [string]$JuggernautGuidedOutput = "outputs/juggernaut_guided.jpg",
    [int]$JuggernautSteps = 0,
    [double]$JuggernautGuidanceScale = 0,
    [double]$JuggernautGuidedStrength = 0,
    [Nullable[int]]$JuggernautSeed,
    [switch]$Npu,
    [switch]$CaptureAutoStart,
    [switch]$JuggernautLocalOnly,
    [switch]$JuggernautNoPreload,
    [switch]$SkipGuided,
    [switch]$UseIntelProxy,
    [switch]$DownloadJuggernaut,
    [switch]$Sync
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

function Assert-Command([string]$Name) {
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Required command '$Name' was not found. Install uv and ensure it is on PATH."
    }
}

function Assert-Path([string]$Path, [string]$Description) {
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "$Description was not found: $Path"
    }
}

Assert-Path $ModelPath "YOLO model path"
Assert-Path $RmbgModelDir "RMBG model directory"
Assert-Path $SceneImage "scene image"
Assert-Path $PropImage "prop image"

if ($UseIntelProxy) {
    $ProxyScript = Join-Path (Split-Path -Parent $PSScriptRoot) "proxy.ps1"
    Assert-Path $ProxyScript "Intel proxy script"
    & $ProxyScript -Enable
}

if ($Sync) {
    Assert-Command "uv"
    Write-Host "Syncing project dependencies..." -ForegroundColor Cyan
    & uv sync
    if ($LASTEXITCODE -ne 0) {
        throw "uv sync failed with exit code $LASTEXITCODE"
    }
}

Assert-Path ".venv\Scripts\python.exe" "project Python environment"

if ($DownloadJuggernaut) {
    Assert-Command "uvx"
    Write-Host "Ensuring Juggernaut model is available: $JuggernautModelId" -ForegroundColor Cyan
    & uvx --from huggingface_hub hf download $JuggernautModelId
    if ($LASTEXITCODE -ne 0) {
        throw "Juggernaut model download failed with exit code $LASTEXITCODE"
    }
}

$Arguments = @(
    "--run-social-pipeline",
    "--backend", $Backend,
    "--model-path", $ModelPath,
    "--device", $Device,
    "--rmbg-device", $RmbgDevice,
    "--rmbg-model-dir", $RmbgModelDir,
    "--camera-index", $CameraIndex,
    "--capture-width", $CaptureWidth,
    "--capture-height", $CaptureHeight,
    "--capture-rotate", $CaptureRotate,
    "--warmup-frames", $WarmupFrames,
    "--capture-delay-seconds", $CaptureDelaySeconds,
    "--capture-wrist-stable-seconds", $CaptureWristStableSeconds,
    "--compose-scene-image", $SceneImage,
    "--compose-prop-image", $PropImage,
    "--compose-output-image", $ComposeOutputImage,
    "--final-output-image", $FinalOutputImage,
    "--juggernaut-guided-output", $JuggernautGuidedOutput,
    "--mask-quality", $MaskQuality,
    "--juggernaut-model-id", $JuggernautModelId,
    "--juggernaut-device", $JuggernautDevice,
    "--juggernaut-preset", $JuggernautPreset
)

if ($Npu) { $Arguments += "--npu" }
if ($CaptureAutoStart) { $Arguments += "--capture-auto-start" }
if ($JuggernautLocalOnly) { $Arguments += "--juggernaut-local-only" }
if ($JuggernautNoPreload) { $Arguments += "--juggernaut-no-preload" }
if ($SkipGuided) { $Arguments += "--juggernaut-skip-guided" }
if ($JuggernautSteps -gt 0) { $Arguments += @("--juggernaut-steps", $JuggernautSteps) }
if ($JuggernautGuidanceScale -gt 0) {
    $Arguments += @("--juggernaut-guidance-scale", $JuggernautGuidanceScale)
}
if ($JuggernautGuidedStrength -gt 0) {
    $Arguments += @("--juggernaut-guided-strength", $JuggernautGuidedStrength)
}
if ($null -ne $JuggernautSeed) { $Arguments += @("--juggernaut-seed", $JuggernautSeed) }

Write-Host "Starting vPRO social pipeline..." -ForegroundColor Green
$Python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
& $Python -m vpro.cli @Arguments
exit $LASTEXITCODE