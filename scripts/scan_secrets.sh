#!/bin/bash
# Scan repository for secrets and sensitive data

echo "🔍 Scanning for secrets and sensitive patterns..."
echo "=================================================="

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)/.."
FINDINGS=0

# Patterns to search
declare -a PATTERNS=(
    "sk-"  # OpenAI API keys
    "ghp_"  # GitHub PAT
    "AKIA"  # AWS access key
    "password.*="
    "api_key"
    "secret_key"
    "/Users/"  # Absolute macOS paths
    "C:\\Users"  # Windows paths
    "ssh-rsa"  # SSH keys
    "BEGIN RSA PRIVATE"  # Private key markers
    "-----BEGIN"  # PEM markers
)

# Unicode obfuscation check
echo ""
echo "Checking for Unicode obfuscation (zero-width, bidirectional)..."
if grep -r -P '[\x{200B}\x{200C}\x{200D}\x{2060}\x{FEFF}\x{202A}-\x{202E}]' "$REPO_ROOT/src" "$REPO_ROOT/scripts" 2>/dev/null; then
    echo "❌ FOUND: Unicode obfuscation patterns"
    ((FINDINGS++))
else
    echo "✓ No Unicode obfuscation detected"
fi

# Scan for common secret patterns
echo ""
echo "Checking for secret patterns..."
for pattern in "${PATTERNS[@]}"; do
    if grep -ri "$pattern" "$REPO_ROOT/src" "$REPO_ROOT/scripts" "$REPO_ROOT/tests" 2>/dev/null | grep -v "\.pyc" | grep -v "__pycache__"; then
        echo "⚠️  FOUND: Pattern '$pattern'"
        ((FINDINGS++))
    fi
done

# Check for large files (potential data leaks)
echo ""
echo "Checking for large files (>5MB)..."
if find "$REPO_ROOT" -type f -size +5M ! -path "*/.git/*" ! -path "*/venv/*" ! -path "*/.venv/*"; then
    echo "⚠️  Large files found (check if intentional)"
fi

# Check for common credential files
echo ""
echo "Checking for credential files..."
CRED_FILES=(".env" "secrets.json" "credentials.json" "config.json" "settings.local.json" ".aws" ".ssh")
for file in "${CRED_FILES[@]}"; do
    if find "$REPO_ROOT" -name "$file" -not -path "*/.git/*" -not -path "*/.venv/*"; then
        echo "❌ FOUND: Credential file '$file'"
        ((FINDINGS++))
    fi
done

echo ""
echo "=================================================="
if [ $FINDINGS -eq 0 ]; then
    echo "✅ No secrets detected!"
    exit 0
else
    echo "⚠️  Found $FINDINGS potential issues. Please review."
    exit 1
fi
