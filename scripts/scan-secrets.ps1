# Scan repository for secrets and sensitive data
# Run: .\scripts\scan-secrets.ps1

param(
    [string]$RepoRoot = (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))
)

Write-Host "🔍 Scanning for secrets and sensitive patterns..."
Write-Host "=================================================="
Write-Host ""

$findings = 0
$patterns = @(
    "sk-",  # OpenAI API keys
    "ghp_",  # GitHub PAT
    "AKIA",  # AWS access key
    "password\s*=",
    "api_key",
    "secret_key"
)

# Function to scan files
function Test-Pattern {
    param([string]$Pattern, [string]$Path)

    $files = Get-ChildItem -Path $Path -Recurse -Include "*.py", "*.json", "*.md", "*.txt" -ErrorAction SilentlyContinue | `
        Where-Object { $_.FullName -notmatch "venv|__pycache__|\.git" }

    foreach ($file in $files) {
        try {
            $content = Get-Content $file -ErrorAction SilentlyContinue
            if ($content -match $Pattern) {
                Write-Host "⚠️  Found in: $($file.FullName)"
                Write-Host "   Pattern: $Pattern"
                Write-Host ""
                return $true
            }
        }
        catch {
            # Skip binary files
        }
    }
    return $false
}

# Scan patterns
Write-Host "Checking for secret patterns..."
foreach ($pattern in $patterns) {
    if (Test-Pattern -Pattern $pattern -Path $RepoRoot) {
        $findings++
    }
}

# Check for absolute paths (Windows)
Write-Host "Checking for absolute Windows paths..."
$absPathPattern = "C:\\Users|/Users/[A-Za-z]|/home/[a-z]"
if (Test-Pattern -Pattern $absPathPattern -Path $RepoRoot) {
    $findings++
}

# Check for credential files
Write-Host "Checking for credential files..."
$credFiles = @(".env", ".env*", "secrets.json", "credentials.json", "config.json", ".aws", ".ssh")
foreach ($file in $credFiles) {
    $found = Get-ChildItem -Path $RepoRoot -Recurse -Filter $file -ErrorAction SilentlyContinue | `
        Where-Object { $_.FullName -notmatch "venv|\.git" }

    if ($found) {
        Write-Host "❌ Found credential file: $($found.FullName)"
        $findings++
    }
}

# Check for large files (>5MB)
Write-Host ""
Write-Host "Checking for large files (>5MB)..."
$largeFiles = Get-ChildItem -Path $RepoRoot -Recurse -ErrorAction SilentlyContinue | `
    Where-Object { $_.Length -gt 5MB -and $_.FullName -notmatch "venv|\.git" }

if ($largeFiles) {
    foreach ($file in $largeFiles) {
        Write-Host "⚠️  Large file: $($file.FullName) ($([math]::Round($file.Length/1MB, 2)) MB)"
    }
}

Write-Host ""
Write-Host "=================================================="
if ($findings -eq 0) {
    Write-Host "✅ No secrets detected!" -ForegroundColor Green
    exit 0
}
else {
    Write-Host "⚠️  Found $findings potential issues. Please review." -ForegroundColor Yellow
    exit 1
}
