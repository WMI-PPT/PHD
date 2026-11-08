<#
.SYNOPSIS
    Auto-commit and push a Git repository.

.DESCRIPTION
    Stages everything, commits with a timestamp message, pulls with rebase,
    then pushes. Designed to be run by Windows Task Scheduler.
    Exits silently when there is nothing to commit.

.PARAMETER RepoPath
    Repository working tree. Defaults to the script's own folder.

.PARAMETER Branch
    Branch to push. Default: main

.PARAMETER MaxFileMB
    Warn (not block) if the staged diff is larger than this. Default: 200

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File autosync.ps1 -RepoPath "C:\Users\yifan\PycharmProjects"
#>
param(
    [string]$RepoPath = $PSScriptRoot,
    [string]$Branch = "main",
    [int]$MaxFileMB = 200
)

# IMPORTANT: git writes normal progress to stderr. With ErrorActionPreference=Stop
# PowerShell would turn that into a terminating error, so keep it Continue and
# rely on $LASTEXITCODE instead.
$ErrorActionPreference = "Continue"

# Keep the log OUTSIDE the repository, otherwise it would itself be a change
# and every run would produce an empty commit.
$logDir = Join-Path $env:LOCALAPPDATA "autosync"
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir -Force | Out-Null }
$LogFile = Join-Path $logDir ((Split-Path $RepoPath -Leaf) + ".log")

function Write-Log([string]$msg) {
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $msg
    Write-Host $line
    try { Add-Content -Path $LogFile -Value $line -Encoding UTF8 } catch { }
}

function Invoke-Git {
    param([string[]]$GitArgs)
    $out = & git @GitArgs 2>&1
    return @{
        Code   = $LASTEXITCODE
        Output = (($out | ForEach-Object { "$_" }) -join "`n").Trim()
    }
}

try {
    if (-not (Test-Path $RepoPath)) { Write-Log "ERROR: repo path not found: $RepoPath"; exit 1 }
    Set-Location $RepoPath

    $r = Invoke-Git @("rev-parse", "--is-inside-work-tree")
    if ($r.Code -ne 0 -or $r.Output -ne "true") {
        Write-Log "ERROR: not a git repository: $RepoPath"
        exit 1
    }

    $st = Invoke-Git @("status", "--porcelain")
    if ($st.Code -ne 0) { Write-Log "ERROR: git status failed: $($st.Output)"; exit 1 }
    if (-not $st.Output) { exit 0 }          # nothing changed, stay quiet

    $count = ($st.Output -split "`n" | Where-Object { $_ }).Count

    $add = Invoke-Git @("add", "-A")
    if ($add.Code -ne 0) { Write-Log "ERROR: git add failed: $($add.Output)"; exit 1 }

    # size guard
    $num = Invoke-Git @("diff", "--cached", "--numstat")
    $bytes = [int64]0
    foreach ($line in ($num.Output -split "`n")) {
        $parts = $line -split "`t"
        if ($parts.Length -ge 2 -and $parts[0] -match '^\d+$') { $bytes += [int64]$parts[0] }
    }
    $mb = [math]::Round($bytes / 1MB, 1)
    if ($mb -gt $MaxFileMB) { Write-Log "WARNING: staged change is $mb MB (limit $MaxFileMB MB) - check .gitignore" }

    $msg = "auto: {0} ({1} file(s) changed)" -f (Get-Date -Format "yyyy-MM-dd HH:mm"), $count
    $c = Invoke-Git @("commit", "-m", $msg)
    if ($c.Code -ne 0) { Write-Log "ERROR: commit failed: $($c.Output)"; exit 1 }
    $hash = (Invoke-Git @("rev-parse", "--short", "HEAD")).Output

    $remotes = (Invoke-Git @("remote")).Output
    if ($remotes -notmatch "origin") {
        Write-Log "OK  committed $hash ($count files), no remote configured yet"
        exit 0
    }

    Invoke-Git @("fetch", "origin") | Out-Null

    $lsr = Invoke-Git @("ls-remote", "--heads", "origin", $Branch)
    if ($lsr.Code -eq 0 -and $lsr.Output) {
        $pull = Invoke-Git @("pull", "--rebase", "--autostash", "origin", $Branch)
        if ($pull.Code -ne 0) {
            Write-Log "PULL FAILED (retry next run): $($pull.Output)"
            exit 2
        }
    }

    $push = Invoke-Git @("push", "-u", "origin", $Branch)
    if ($push.Code -ne 0) {
        Write-Log "PUSH FAILED (retry next run): $($push.Output)"
        exit 3
    }

    Write-Log "OK  committed $hash ($count files), pushed to origin/$Branch"
    exit 0
}
catch {
    Write-Log ("ERROR: " + $_.Exception.Message)
    exit 1
}
