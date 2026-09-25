<#
.SYNOPSIS
    Start the persistent vPRO event kiosk.

.DESCRIPTION
    This is the primary Windows launcher for the project. It keeps paths rooted at the repository,
    optionally enables the parent-directory Intel proxy, and forwards production controls to the
    vpro CLI. Normal execution uses the project's .venv directly; uv is only needed for -Sync or
    -DownloadModel.

.EXAMPLE
    .\run.ps1

.EXAMPLE
    .\run.ps1 -UseIntelProxy -Sync -DownloadModel -CaptureAutoStart

.EXAMPLE
    .\run.ps1 -Device intel:gpu -RmbgDevice GPU -JuggernautPreset identity-lock
#>

[CmdletBinding()]
param(
    [ValidateSet("kiosk", "social")]
    [string]$Mode = "kiosk",
    [ValidateSet("openvino")]
    [string]$Backend = "openvino",
    [string]$ModelPath = "models/yolo26/yolo26x-pose_openvino_model",
    [string]$Device = "intel:npu",
    [string]$RmbgDevice = "CPU",
    [string]$JuggernautDevice = "GPU",
    [ValidateSet("DreamShaper", "Juggernaut")]
    [string]$GenerationEngine = "DreamShaper",
    [string]$JuggernautModelId,
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
    [string]$RmbgModelDir = "models/rmbg/rmbg-1.4",
    [string]$ComposeOutputImage = "outputs/final_portrait_512x512.jpg",
    [string]$FinalOutputImage = "outputs/final_portrait_512x512_final.jpg",
    [string]$JuggernautGuidedOutput = "outputs/generation_guided.jpg",
    [int]$GenerationSteps = 0,
    [double]$GenerationGuidanceScale = 0,
    [double]$GenerationGuidedStrength = 0,
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
    [Alias("DownloadJuggernaut")]
    [switch]$DownloadModel,
    [switch]$Sync,
    [switch]$AllowModelDownloads,
    [switch]$DryRun,
    [ValidateRange(1, 60)]
    [double]$PoseFps = 15,
    [ValidateRange(1, 65535)]
    [int]$KioskPort = 8000,
    [string]$OutputDir = "outputs/kiosk",
    [ValidateSet("local-qr", "s3-qr", "twilio", "none")]
    [string]$DeliveryChannel = "s3-qr",
    [string]$DeliveryAdvertiseHost,
    [ValidateRange(0.25, 168)]
    [double]$RetentionHours = 24
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
Push-Location -LiteralPath $PSScriptRoot
try {

$DotEnvPath = Join-Path $PSScriptRoot ".env"
if (Test-Path -LiteralPath $DotEnvPath) {
    foreach ($Line in Get-Content -LiteralPath $DotEnvPath) {
        $TrimmedLine = $Line.Trim()
        if (-not $TrimmedLine -or $TrimmedLine.StartsWith("#")) {
            continue
        }
        if ($TrimmedLine -notmatch '^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$') {
            throw "Invalid .env entry: $TrimmedLine"
        }
        $Name = $Matches[1]
        $Value = $Matches[2].Trim()
        if (($Value.StartsWith('"') -and $Value.EndsWith('"')) -or
            ($Value.StartsWith("'") -and $Value.EndsWith("'"))) {
            $Value = $Value.Substring(1, $Value.Length - 2)
        }
        Set-Item -Path "Env:$Name" -Value $Value
    }
}

if (-not $JuggernautModelId) {
    $JuggernautModelId = if ($GenerationEngine -eq "Juggernaut") {
        "models/juggernaut-int8"
    } else {
        "OpenVINO/dreamshaper-8-inpainting-int8-ov"
    }
}
$GenerationWidth = if ($GenerationEngine -eq "Juggernaut") { 896 } else { 512 }
$GenerationHeight = if ($GenerationEngine -eq "Juggernaut") { 1120 } else { 512 }

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
if ($Mode -eq "kiosk" -and $Backend -ne "openvino") {
    throw "Kiosk mode requires OpenVINO. Use -Mode social for other backends."
}
if ($Mode -eq "kiosk" -and $SkipGuided) {
    throw "Event kiosk requires generation. Use -Mode social -SkipGuided for diagnostic composition."
}
if ($PSBoundParameters.ContainsKey("GenerationSteps") -and $PSBoundParameters.ContainsKey("JuggernautSteps") -and $GenerationSteps -ne $JuggernautSteps) {
    throw "Use -GenerationSteps or -JuggernautSteps, not both with different values. Prefer -GenerationSteps."
}
if ($PSBoundParameters.ContainsKey("GenerationGuidanceScale") -and $PSBoundParameters.ContainsKey("JuggernautGuidanceScale") -and $GenerationGuidanceScale -ne $JuggernautGuidanceScale) {
    throw "Use -GenerationGuidanceScale or -JuggernautGuidanceScale, not both with different values. Prefer -GenerationGuidanceScale."
}
if ($PSBoundParameters.ContainsKey("GenerationGuidedStrength") -and $PSBoundParameters.ContainsKey("JuggernautGuidedStrength") -and $GenerationGuidedStrength -ne $JuggernautGuidedStrength) {
    throw "Use -GenerationGuidedStrength or -JuggernautGuidedStrength, not both with different values. Prefer -GenerationGuidedStrength."
}

if ($UseIntelProxy) {
    $ProxyScript = Join-Path (Split-Path -Parent $PSScriptRoot) "proxy.ps1"
    Assert-Path $ProxyScript "Intel proxy script"
    & $ProxyScript -Enable
}

if ($Sync) {
    Assert-Command "uv"
    Write-Host "Syncing project dependencies..." -ForegroundColor Cyan
    & uv sync --extra gen --extra kiosk --extra delivery
    if ($LASTEXITCODE -ne 0) {
        throw "uv sync failed with exit code $LASTEXITCODE"
    }
}

Assert-Path ".venv\Scripts\python.exe" "project Python environment"

if ($DownloadModel) {
    if (Test-Path -LiteralPath $JuggernautModelId) {
        Write-Host "Using local generation model: $JuggernautModelId" -ForegroundColor Cyan
    } else {
        Assert-Command "uvx"
        Write-Host "Ensuring generation model is available: $JuggernautModelId" -ForegroundColor Cyan
        & uvx --from huggingface_hub hf download $JuggernautModelId
        if ($LASTEXITCODE -ne 0) {
            throw "Generation model download failed with exit code $LASTEXITCODE"
        }
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
    "--compose-output-image", $ComposeOutputImage,
    "--final-output-image", $FinalOutputImage,
    "--generation-guided-output", $JuggernautGuidedOutput,
    "--mask-quality", $MaskQuality,
    "--generation-model-id", $JuggernautModelId,
    "--generation-device", $JuggernautDevice,
    "--generation-preset", $JuggernautPreset,
    "--generation-width", $GenerationWidth,
    "--generation-height", $GenerationHeight
)

if ($Mode -eq "kiosk") {
    $Arguments += @("--kiosk", "--kiosk-pose-fps", $PoseFps,
        "--kiosk-port", $KioskPort, "--kiosk-output-retention-hours", $RetentionHours,
        "--kiosk-output-dir", $OutputDir, "--delivery-channel", $DeliveryChannel, "--open-browser")
    if ($DeliveryAdvertiseHost) {
        $Arguments += @("--delivery-advertise-host", $DeliveryAdvertiseHost)
    }
} else {
    $Arguments += "--run-social-pipeline"
}

if ($Npu) { $Arguments += "--npu" }
if ($CaptureAutoStart) { $Arguments += "--capture-auto-start" }
if ($JuggernautLocalOnly -or -not $AllowModelDownloads) { $Arguments += "--generation-local-only" }
if ($JuggernautNoPreload) { $Arguments += "--generation-no-preload" }
if ($SkipGuided) { $Arguments += "--generation-skip-guided" }
if ($PSBoundParameters.ContainsKey("GenerationSteps")) {
    $Arguments += @("--generation-steps", $GenerationSteps)
} elseif ($PSBoundParameters.ContainsKey("JuggernautSteps")) {
    $Arguments += @("--generation-steps", $JuggernautSteps)
}
if ($PSBoundParameters.ContainsKey("GenerationGuidanceScale")) {
    $Arguments += @("--generation-guidance-scale", $GenerationGuidanceScale)
} elseif ($PSBoundParameters.ContainsKey("JuggernautGuidanceScale")) {
    $Arguments += @("--generation-guidance-scale", $JuggernautGuidanceScale)
} elseif ($GenerationEngine -eq "DreamShaper") {
    $Arguments += @("--generation-guidance-scale", 7.0)
}
if ($PSBoundParameters.ContainsKey("GenerationGuidedStrength")) {
    $Arguments += @("--generation-guided-strength", $GenerationGuidedStrength)
} elseif ($PSBoundParameters.ContainsKey("JuggernautGuidedStrength")) {
    $Arguments += @("--generation-guided-strength", $JuggernautGuidedStrength)
}
if ($null -ne $JuggernautSeed) { $Arguments += @("--generation-seed", $JuggernautSeed) }

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