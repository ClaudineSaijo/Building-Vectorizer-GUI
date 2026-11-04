$ErrorActionPreference = 'Stop'
$py = 'C:\Users\Nakan\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe'
$work = 'D:\All4DS\BISHE\开题报告'
$target = 'D:\大三下学期作业\毕业课题参考文献\开题准备'
$docx = Join-Path $work '2027届开题报告-基于多任务深度学习的遥感影像建筑物矢量化提取.docx'
$docfile = Join-Path $work '2027届开题报告-基于多任务深度学习的遥感影像建筑物矢量化提取.doc'

$env:PYTHONIOENCODING = 'utf-8'
& $py -X utf8 (Join-Path $work 'edit_online.py')
if ($LASTEXITCODE -ne 0) { throw 'python edit failed' }

if (Test-Path -LiteralPath $docfile) { Remove-Item -LiteralPath $docfile -Force }
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
$doc = $word.Documents.Open($docx, $false, $false)
$doc.SaveAs([ref]$docfile, [ref]0)
$doc.Close(0)

$check = $word.Documents.Open($docfile, $false, $true)
'pages=' + $check.ComputeStatistics(2) + ' tables=' + $check.Tables.Count
$tableText = $check.Tables.Item(1).Range.Text
if ($tableText -match '指导形式：线上指导') { 'doc has online form: yes' } else { 'doc has online form: NO' }
if ($tableText -match '指导地点：线上（腾讯会议）') { 'doc has online place: yes' } else { 'doc has online place: NO' }
$check.Close(0)
$word.Quit()

Copy-Item -LiteralPath $docx -Destination $target -Force
Copy-Item -LiteralPath $docfile -Destination $target -Force

foreach ($p in $docx, $docfile) {
    $name = Split-Path -Leaf $p
    $a = (Get-FileHash -LiteralPath $p -Algorithm SHA256).Hash
    $b = (Get-FileHash -LiteralPath (Join-Path $target $name) -Algorithm SHA256).Hash
    "$name synced=$($a -eq $b)"
}
