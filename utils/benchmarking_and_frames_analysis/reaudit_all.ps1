# =====================================================================
#  reaudit_all.ps1 - re-audit every session with the Stage 1 diagnostics
#
#  Adds vs previous audits:
#    - wide_cluster_rej counter          (frames eaten by the 0.65 m gate)
#    - rx kept / rx REJECTED split       (the censored distribution)
#    - INPUT folder provenance           (prevents the A_S01/B_S01 mix-up)
#    - correct denominators              (accum vs cluster, like-for-like)
#    - P0-5 ghost-gate fix               (accumulator no longer deadlocks)
#
#  PATH LAYOUT
#    Most sessions:  <ProtocolRoot>\<SESSION>\*.bin
#    outdoor_long :  06_outdoor_multiple_subjects\frames\*.bin  (different tree)
#
#    Sessions whose frames live somewhere non-standard are listed in
#    $PathOverrides below. Add new exceptions there rather than editing
#    the loop.
#
#  Usage:
#    .\reaudit_all.ps1                          # all sessions
#    .\reaudit_all.ps1 -Sessions outdoor_long   # one session
#    .\reaudit_all.ps1 -Sessions C_S03,D_S03    # subset
#    .\reaudit_all.ps1 -WhatIf                  # resolve paths only, no run
# =====================================================================

param(
    [string[]] $Sessions = @(
        'A_S01','A_S02','A_S03',
        'B_S01','B_S02','B_S03',
        'C_S01','C_S02','C_S03',
        'D_S01','D_S02','D_S03',
        'outdoor_long'
    ),
    [string] $ProtocolRoot =
        'testing\live_capture_validation\05_outdoor_data_collection\frames',
    [string] $Cfg    = 'firmware\r017m\cascade_r017m.cfg',
    [string] $Report = 'reaudit_summary.txt',
    [switch] $WhatIf
)

# ---------------------------------------------------------------------
#  Per-session path overrides.
#    In  = folder containing the .bin frames
#    Out = folder to receive frames.csv / points.csv / summary.txt
# ---------------------------------------------------------------------
$PathOverrides = @{
    'outdoor_long' = @{
        In  = 'testing\live_capture_validation\06_outdoor_multiple_subjects\frames'
        Out = 'testing\live_capture_validation\06_outdoor_multiple_subjects'
    }
}

function Resolve-SessionPaths {
    param([string] $Session)
    if ($PathOverrides.ContainsKey($Session)) {
        return @{
            In       = $PathOverrides[$Session].In
            Out      = $PathOverrides[$Session].Out
            Override = $true
        }
    }
    return @{
        In       = Join-Path $ProtocolRoot $Session
        Out      = Join-Path $ProtocolRoot $Session
        Override = $false
    }
}

$ErrorActionPreference = 'Continue'
$lines  = @()
$lines += '=' * 78
$lines += '  RE-AUDIT SUMMARY - Stage 1 diagnostics + P0-5 ghost-gate fix'
$lines += "  generated: $(Get-Date -Format 'yyyy-MM-dd HH:mm')"
$lines += '=' * 78

$ok = 0; $skipped = 0; $failed = 0

foreach ($s in $Sessions) {

    $p      = Resolve-SessionPaths -Session $s
    $inDir  = $p.In
    $outDir = $p.Out

    Write-Host ''
    Write-Host ('=' * 70) -ForegroundColor Cyan
    Write-Host "  $s" -ForegroundColor Cyan
    if ($p.Override) {
        Write-Host '  [path override - non-standard location]' -ForegroundColor Magenta
    }
    Write-Host ('=' * 70) -ForegroundColor Cyan
    Write-Host "  in  : $inDir"
    Write-Host "  out : $outDir"

    if (-not (Test-Path $inDir)) {
        Write-Host '  SKIP - folder not found' -ForegroundColor Yellow
        $lines += ''
        $lines += "$s : SKIPPED (folder not found: $inDir)"
        $skipped++
        continue
    }

    $nBin = (Get-ChildItem -Path $inDir -Filter *.bin -ErrorAction SilentlyContinue).Count
    if ($nBin -eq 0) {
        Write-Host '  SKIP - no .bin files' -ForegroundColor Yellow
        $lines += ''
        $lines += "$s : SKIPPED (no .bin files in $inDir)"
        $skipped++
        continue
    }
    Write-Host "  $nBin .bin files" -ForegroundColor Green

    if ($WhatIf) {
        Write-Host '  (WhatIf - not executed)' -ForegroundColor DarkGray
        $lines += ''
        $lines += "$s : WOULD RUN  in=$inDir  out=$outDir  ($nBin frames)"
        continue
    }

    if (-not (Test-Path $outDir)) {
        New-Item -ItemType Directory -Path $outDir -Force | Out-Null
    }

    python diag_pipeline_audit.py `
        --folder  $inDir `
        --cfg     $Cfg `
        --temporal-accum --accum-window 16 `
        --subject-frame --gait-features `
        --out-dir $outDir

    if ($LASTEXITCODE -ne 0) {
        Write-Host "  FAILED (exit $LASTEXITCODE)" -ForegroundColor Red
        $lines += ''
        $lines += "$s : FAILED (exit code $LASTEXITCODE)"
        $failed++
        continue
    }
    $ok++

    $sumPath = Join-Path $outDir 'summary.txt'
    $lines += ''
    $lines += '-' * 78
    if ($p.Override) {
        $lines += "$s   ($nBin input frames)   [path override]"
    } else {
        $lines += "$s   ($nBin input frames)"
    }
    $lines += "  source: $inDir"
    $lines += '-' * 78
    if (Test-Path $sumPath) {
        Get-Content $sumPath | Where-Object {
            $_ -match 'Stage 4.2 limb mode|P0-6 subject gate|Frames total|Frames with cloud|avg pts/frame|avg accum_pts|avg cluster_pts|wide_cluster_rej|sg_rej|sg_bypass|n_subjects|separation_m|compensated|multi_used|n_raw_clusters|comp_step_x|comp_residual_x|rx kept|rx REJECTED|x mean|y mean'
        } | ForEach-Object { $lines += "  $_" }
    } else {
        $lines += '  (summary.txt not produced)'
    }
}

$lines += ''
$lines += '=' * 78
$lines += "  RUN RESULT: $ok audited, $skipped skipped, $failed failed"
$lines += '=' * 78
$lines += '  WHAT TO LOOK FOR (post P0-5 fix)'
$lines += '=' * 78
$lines += '  1. accum_pts should RISE vs the previous run.'
$lines += '     Pre-fix the buffer froze during steady walking (>1.03 m/s),'
$lines += '     so accum_pts was pinned low. A/D were ~22-27; expect higher.'
$lines += '  2. rx median must NO LONGER equal rx max.'
$lines += '     B_S03/C_S03/D_S03 previously reported identical values across'
$lines += '     300+ frames - the frozen-buffer signature.'
$lines += '  3. wide_cluster_rej counts WILL change.'
$lines += '     Appendix B numbers were measured on frozen output and are void.'
$lines += '  4. cluster_noise should be non-zero.'
$lines += '     C_S03/D_S03 previously had accum == cluster exactly (zero noise).'
$lines += '  5. outdoor_long is the multi-subject session - expect larger rx'
$lines += '     and higher rx variance than any single-subject session.'

$lines | Set-Content -Path $Report -Encoding UTF8

Write-Host ''
Write-Host ('=' * 70) -ForegroundColor Green
Write-Host "  $ok audited, $skipped skipped, $failed failed" -ForegroundColor Green
Write-Host "  Wrote $Report" -ForegroundColor Green
Write-Host '  Paste that file into the chat.' -ForegroundColor Green
Write-Host ('=' * 70) -ForegroundColor Green
