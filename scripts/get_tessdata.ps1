# get_tessdata.ps1 - download tessdata_best into <project>\tessdata
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"   # much faster downloads in Windows PowerShell 5.1

$Root = Split-Path -Parent $PSScriptRoot
$Dir  = Join-Path $Root "tessdata"
New-Item -ItemType Directory -Force -Path $Dir | Out-Null

foreach ($l in "kor","kor_vert","chi_tra","chi_tra_vert","eng","osd") {
    $f = Join-Path $Dir "$l.traineddata"
    if (-not (Test-Path $f)) {
        Write-Host "downloading $l ..."
        Invoke-WebRequest -Uri "https://raw.githubusercontent.com/tesseract-ocr/tessdata_best/main/$l.traineddata" -OutFile $f
    }
}

# configs for hOCR output
$Cfg = Join-Path $Dir "configs"
if (-not (Test-Path (Join-Path $Cfg "hocr"))) {
    New-Item -ItemType Directory -Force -Path $Cfg | Out-Null
    $copied = $false

    # 1) copy from the installed Tesseract
    $cands = @()
    $cmd = Get-Command tesseract -ErrorAction SilentlyContinue
    if ($cmd) { $cands += (Join-Path (Split-Path $cmd.Source) "tessdata\configs") }
    $cands += "C:\Program Files\Tesseract-OCR\tessdata\configs"
    foreach ($c in $cands) {
        if (Test-Path $c) {
            Copy-Item "$c\*" $Cfg -Recurse -Force
            $copied = $true
            break
        }
    }

    # 2) fallback: download from GitHub
    if (-not $copied) {
        foreach ($n in "hocr","tsv","txt","pdf","quiet") {
            try {
                Invoke-WebRequest -Uri "https://raw.githubusercontent.com/tesseract-ocr/tessconfigs/main/configs/$n" -OutFile (Join-Path $Cfg $n)
            } catch { Write-Warning "could not fetch config: $n" }
        }
    }
}

Write-Host "tessdata ready: $Dir"