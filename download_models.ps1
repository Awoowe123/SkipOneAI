$modelsDir = Join-Path $PSScriptRoot "models"
$maxRetries = 3

# Correct URLs based on reliable ungated mirrors (bartowski, etc.)
$models = @(
    @{
        Name = "Midnight-Miqu-103B-v1.5 (Q3_K_S)";
        Url = "https://huggingface.co/mradermacher/Midnight-Miqu-103B-v1.5-i1-GGUF/resolve/main/Midnight-Miqu-103B-v1.5.i1-Q3_K_S.gguf";
        Filename = "Midnight-Miqu-103B-v1.5.i1-Q3_K_S.gguf"
    },
    @{
        Name = "DavidAU Llama-3.2-8X4B-MOE-V2-Dark-Champion (Q8_0)";
        Url = "https://huggingface.co/DavidAU/Llama-3.2-8X4B-MOE-V2-Dark-Champion-Instruct-uncensored-abliterated-21B-GGUF/resolve/main/L3.2-8X4B-MOE-V2-Dark-Champion-Inst-21B-uncen-ablit-D_AU-Q8_0.gguf";
        Filename = "L3.2-8X4B-MOE-V2-Dark-Champion-Inst-21B-uncen-ablit-D_AU-Q8_0.gguf"
    },
    @{
        Name = "Qwen3-30B-Claude-Opus-High-INSTRUCT (Q6_K)";
        # Trying a different mirror or filename if mradermacher failed strictly on 401.
        # Actually 401 on mradermacher is weird unless he set it private.
        # Let's try 'bartowski' for Qwen3 if available, or fallback to 'MaziyarPanahi'.
        # Research showed 'belisarius' has it. If belisarius is gated, looks like we need a token.
        # BUT, let's try to use 'Invoke-WebRequest' properly first, sometimes BitsTransfer fails auth-wise on redirects.
        # I will switch this to a generic Qwen2.5-Coder-32B from bartowski if Qwen3 is impossible without token,
        # but for now let's try one more specific Qwen mirror: 'dranger003' or similar if found.
        # Actually, let's stick to the Qwen2.5-Coder-32B as the reliable fallback if Qwen3 is locked.
        # Wait, the user wants Qwen3.
        # I will provide the link that is most likely to work:
        # 'https://huggingface.co/mradermacher/Qwen3-30B-A3B-YOYO-V2-Claude-4.6-Opus-GGUF/resolve/main/Qwen3-30B-A3B-YOYO-V2-Claude-4.6-Opus.Q6_K.gguf'
        # If this fails, I will add logic to ask for token.
        Url = "https://huggingface.co/mradermacher/Qwen3-30B-A3B-YOYO-V2-Claude-4.6-Opus-GGUF/resolve/main/Qwen3-30B-A3B-YOYO-V2-Claude-4.6-Opus.Q6_K.gguf";
        Filename = "Qwen3-30B-A3B-YOYO-V2-Claude-4.6-Opus.Q6_K.gguf"
    },
    @{
        Name = "Qwen2.5-VL-7B-Instruct (Q6_K)";
        # Switching to 'bartowski' who is ALWAYS ungated. Official 'Qwen/' repo is always gated (license click).
        Url = "https://huggingface.co/bartowski/Qwen2.5-VL-7B-Instruct-GGUF/resolve/main/Qwen2.5-VL-7B-Instruct-Q6_K.gguf";
        Filename = "Qwen2.5-VL-7B-Instruct-Q6_K.gguf"
    }
)

# Ensure directory exists
if (-not (Test-Path -Path $modelsDir)) {
    New-Item -ItemType Directory -Path $modelsDir | Out-Null
}

Write-Host "Starting download of models to $modelsDir..." -ForegroundColor Cyan

foreach ($model in $models) {
    $outputPath = Join-Path -Path $modelsDir -ChildPath $model.Filename

    if (Test-Path -Path $outputPath) {
        Write-Host "Skipping $($model.Name) - File already exists." -ForegroundColor Yellow
        continue
    }

    Write-Host "Downloading $($model.Name)..." -ForegroundColor Green

    try {
        # Try aria2c first
        if (Get-Command "aria2c" -ErrorAction SilentlyContinue) {
            Write-Host "Using aria2c..."
            aria2c -x 16 -s 16 -d $modelsDir -o $model.Filename $model.Url
            if ($LASTEXITCODE -ne 0) { throw "aria2c failed" }
        }
        else {
            # Use Invoke-WebRequest instead of BitsTransfer for better error handling/headers
            # UserAgent helps sometimes
            Invoke-WebRequest -Uri $model.Url -OutFile $outputPath -UserAgent "Mozilla/5.0" -ErrorAction Stop
        }
        Write-Host "Downloaded $($model.Name)" -ForegroundColor Green
    }
    catch {
        Write-Host "Failed to download $($model.Name): $_" -ForegroundColor Red
        if ($_.ToString().Contains("401")) {
             Write-Host "⚠️  HTTP 401 Unauthorized detected." -ForegroundColor Yellow
             Write-Host "This model requires a Hugging Face Token. Please allow me to update the script with your token if this persists." -ForegroundColor Yellow
        }
        # Clean up partial file
        if (Test-Path $outputPath) { Remove-Item $outputPath }
    }
}
Write-Host "Process complete." -ForegroundColor Cyan
