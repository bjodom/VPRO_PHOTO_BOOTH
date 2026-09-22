<#
.SYNOPSIS
    Start the persistent vPRO event kiosk.

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
    .\run.ps1 -Device intel:gpu -RmbgDevice GPU -JuggernautPreset identity-lock
#>

[CmdletBinding()]
param(
    [ValidateSet("kiosk", "social")]
    [string]$Mode = "kiosk",
    [ValidateSet("openvino", "torch")]
    [string]$Backend = "openvino",
    [string]$ModelPath = "models/yolo26/yolo26x-pose_openvino_model",
    [string]$Device = "intel:npu",
    [string]$RmbgDevice = "CPU",
    [string]$JuggernautDevice = "GPU",
    [string]$JuggernautModelId = "OpenVINO/Juggernaut-XL-v9-fp16-ov",
    [ValidateSet("fast", "balanced", "high")]
    [string]$MaskQuality = "high",
    [ValidateSet("identity-lock", "balanced", "stylized")]
    [string]$JuggernautPreset = "balanced",
    [int]$CameraIndex = 0,
    [int]$CaptureWidth = 1920,
    [int]$CaptureHeight = 1080,
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
    [switch]$Sync,
    [switch]$AllowModelDownloads,
    [switch]$DryRun,
    [ValidateRange(1, 60)]
    [double]$PoseFps = 15,
    [ValidateRange(1, 65535)]
    [int]$KioskPort = 8000,
    [string]$OutputDir = "outputs/kiosk",
    [string]$DeliveryAdvertiseHost,
    [ValidateRange(0.25, 168)]
    [double]$RetentionHours = 24
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
Push-Location -LiteralPath $PSScriptRoot
try {

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
if ($Mode -eq "kiosk" -and $Backend -ne "openvino") {
    throw "Kiosk mode requires OpenVINO. Use -Mode social for other backends."
}
if ($Mode -eq "kiosk" -and $SkipGuided) {
    throw "Event kiosk requires generation. Use -Mode social -SkipGuided for diagnostic composition."
}

if ($UseIntelProxy) {
    $ProxyScript = Join-Path (Split-Path -Parent $PSScriptRoot) "proxy.ps1"
    Assert-Path $ProxyScript "Intel proxy script"
    & $ProxyScript -Enable
}

if ($Sync) {
    Assert-Command "uv"
    Write-Host "Syncing project dependencies..." -ForegroundColor Cyan
    & uv sync --extra gen --extra kiosk
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

if ($Mode -eq "kiosk") {
    $Arguments += @("--kiosk", "--kiosk-pose-fps", $PoseFps,
        "--kiosk-port", $KioskPort, "--kiosk-output-retention-hours", $RetentionHours,
        "--kiosk-output-dir", $OutputDir, "--open-browser")
    if ($DeliveryAdvertiseHost) {
        $Arguments += @("--delivery-advertise-host", $DeliveryAdvertiseHost)
    }
} else {
    $Arguments += "--run-social-pipeline"
}

if ($Npu) { $Arguments += "--npu" }
if ($CaptureAutoStart) { $Arguments += "--capture-auto-start" }
if ($JuggernautLocalOnly -or -not $AllowModelDownloads) { $Arguments += "--juggernaut-local-only" }
if ($JuggernautNoPreload) { $Arguments += "--juggernaut-no-preload" }
if ($SkipGuided) { $Arguments += "--juggernaut-skip-guided" }
if ($PSBoundParameters.ContainsKey("JuggernautSteps")) { $Arguments += @("--juggernaut-steps", $JuggernautSteps) }
if ($PSBoundParameters.ContainsKey("JuggernautGuidanceScale")) {
    $Arguments += @("--juggernaut-guidance-scale", $JuggernautGuidanceScale)
}
if ($PSBoundParameters.ContainsKey("JuggernautGuidedStrength")) {
    $Arguments += @("--juggernaut-guided-strength", $JuggernautGuidedStrength)
}
if ($null -ne $JuggernautSeed) { $Arguments += @("--juggernaut-seed", $JuggernautSeed) }

if ($DryRun) {
    $Arguments | ConvertTo-Json
    return
}
Write-Host "Starting vPRO $Mode..." -ForegroundColor Green
$Python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
& $Python -c "import cv2, openvino, optimum.intel, diffusers, fastapi, uvicorn, segno"
if ($LASTEXITCODE -ne 0) { throw "Dependencies are incomplete. Run .\run.ps1 -Sync before the event." }
& $Python -m vpro.cli @Arguments
if ($LASTEXITCODE -ne 0) { throw "vPRO exited with code $LASTEXITCODE" }
} finally {
    Pop-Location
}