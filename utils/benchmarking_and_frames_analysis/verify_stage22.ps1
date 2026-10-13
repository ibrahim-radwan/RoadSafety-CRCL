# =====================================================================
#  verify_stage22.ps1 — verify the focused azimuth FFT changed NOTHING
#
#  Runs every session through BOTH code paths from the same binary
#  (--full-az-cube selects the original), then diffs the outputs.
#
#  The acceptance test is EQUIVALENCE, not speed. A faster pipeline
#  producing different numbers is worse than a slow correct one.
#
#  ENCODING NOTE
#    The pipeline prints non-ASCII characters (arrows, multiplication
#    signs) in ~30 places. When Python's stdout is a CONSOLE it usually
#    copes; when it is redirected to a pipe or file it falls back to the
#    Windows locale codec (cp1252) and raises UnicodeEncodeError.
#    PYTHONIOENCODING forces UTF-8 regardless of where stdout goes, which
#    fixes it without touching 30 print statements in the pipeline.
#
#  Usage:
#    .\verify_stage22.ps1              # all 13 sessions
#    .\verify_stage22.ps1 -Quick       # edge cases + controls only
#    .\verify_stage22.ps1 -Verbose     # stream audit output to console
# =====================================================================

param(
    [switch] $Quick,
    [switch] $VerboseAudit,
    [string] $ProtocolRoot = 'testing\live_capture_validation\05_outdoor_data_collection\frames',
    [string] $Cfg          = 'firmware\r017m\cascade_r017m.cfg',
    [string] $VerifyRoot   = 'verify'
)

# Force UTF-8 for every child python process (see ENCODING NOTE above).
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8       = '1'

# Ordered so the branches most likely to differ run FIRST — fail fast.
$All = @(
    'A_S03',        # 0% detection -> exercises the empty-mask guard
    'D_S02',        # 0% detection over 309 frames
    'C_S02',        # 61.5% detection -> sparse RD columns, fallback branch
    'C_S01',        # 67.6% detection
    'C_S03',        # 78.4%, large
    'B_S01',        # most fragmented cloud (highest step_p90)
    'outdoor_long', # largest sample, 2 subjects
    'A_S01','A_S02','B_S02','B_S03','D_S01','D_S03'
)
$QuickSet = @('A_S03','D_S02','C_S02','B_S01','A_S01')
$Sessions = if ($Quick) { $QuickSet } else { $All }

$Overrides = @{
    'outdoor_long' = 'testing\live_capture_validation\06_outdoor_multiple_subjects\frames'
}

New-Item -ItemType Directory -Path (Join-Path $VerifyRoot 'logs') -Force | Out-Null

function Invoke-Audit {
    param([string]$InDir, [string]$OutDir, [switch]$FullCube, [string]$LogPath)

    $args = @(
        'diag_pipeline_audit.py',
        '--folder', $InDir, '--cfg', $Cfg,
        '--temporal-accum', '--accum-window', '16',
        '--subject-frame', '--gait-features',
        '--out-dir', $OutDir
    )
    if ($FullCube) { $args += '--full-az-cube' }

    if ($VerboseAudit) {
        & python @args
    } else {
        # Capture rather than discard, so a crash leaves evidence.
        & python @args *> $LogPath
    }
    return $LASTEXITCODE
}

$failed = @()
$n = 0
foreach ($s in $Sessions) {
    $n++
    $inDir = if ($Overrides.ContainsKey($s)) { $Overrides[$s] } else { Join-Path $ProtocolRoot $s }

    Write-Host ''
    Write-Host ("[$n/$($Sessions.Count)]  $s") -ForegroundColor Cyan

    if (-not (Test-Path $inDir)) {
        Write-Host '  SKIP - folder not found' -ForegroundColor Yellow
        continue
    }

    $outFull = Join-Path $VerifyRoot "fullcube\$s"
    $outFoc  = Join-Path $VerifyRoot "focused\$s"
    New-Item -ItemType Directory -Path $outFull -Force | Out-Null
    New-Item -ItemType Directory -Path $outFoc  -Force | Out-Null

    Write-Host '  original full-cube path ...' -ForegroundColor DarkGray
    $rc1 = Invoke-Audit -InDir $inDir -OutDir $outFull -FullCube `
                        -LogPath (Join-Path $VerifyRoot "logs\$s.fullcube.log")
    if ($rc1 -ne 0) {
        Write-Host "  FAILED (exit $rc1) - see verify\logs\$s.fullcube.log" -ForegroundColor Red
        $failed += "$s (fullcube)"
        continue
    }

    Write-Host '  focused path ...' -ForegroundColor DarkGray
    $rc2 = Invoke-Audit -InDir $inDir -OutDir $outFoc `
                        -LogPath (Join-Path $VerifyRoot "logs\$s.focused.log")
    if ($rc2 -ne 0) {
        Write-Host "  FAILED (exit $rc2) - see verify\logs\$s.focused.log" -ForegroundColor Red
        $failed += "$s (focused)"
        continue
    }

    Write-Host '  done' -ForegroundColor Green
}

if ($failed.Count -gt 0) {
    Write-Host ''
    Write-Host ('=' * 70) -ForegroundColor Red
    Write-Host "  $($failed.Count) audit run(s) FAILED - comparison would be invalid" -ForegroundColor Red
    $failed | ForEach-Object { Write-Host "    $_" -ForegroundColor Red }
    Write-Host '  Inspect verify\logs\ before proceeding.' -ForegroundColor Red
    Write-Host ('=' * 70) -ForegroundColor Red
    exit 2
}

Write-Host ''
Write-Host ('=' * 70) -ForegroundColor Cyan
Write-Host '  COMPARING' -ForegroundColor Cyan
Write-Host ('=' * 70) -ForegroundColor Cyan

python verify_stage22.py --batch `
    --before (Join-Path $VerifyRoot 'fullcube') `
    --after  (Join-Path $VerifyRoot 'focused')
$verdict = $LASTEXITCODE

Write-Host ''
if ($verdict -eq 0) {
    Write-Host '  EQUIVALENCE CONFIRMED - speedup below is a genuine win' -ForegroundColor Green
} else {
    Write-Host '  DIFFERENCES FOUND - roll back: az_fft_focused_en = False' -ForegroundColor Red
}

# Per-session proc_ms A/B on identical input — more reliable than the
# standalone benchmark, which reports 0 confirmed range bins per frame.
#
# Calls a FILE, not an inline here-string. `python - $arg @'...'@` passes the
# here-string as an ARGUMENT, not stdin, so python gets `-` with nothing piped
# and silently drops into the interactive REPL. Piping would need
# `@'...'@ | python -`. A file avoids the whole class of quoting bug.
Write-Host ''
if (Test-Path 'timing_report.py') {
    python timing_report.py --root $VerifyRoot
} else {
    Write-Host '  timing_report.py not found - skipping speedup table' -ForegroundColor Yellow
}

exit $verdict
