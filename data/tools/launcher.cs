// CoCo Image Studio.exe 的原始碼（tools/make_launcher.py 用 Windows 內建的 C# 編譯器編，放在 data 資料夾旁邊）。
// .bat 沒辦法有自己的圖示，也不能釘在工作列，所以最外層放這個有 CoCo 圖示的小程式：它只做一件事——
// 在同一個主控台視窗執行 data\CoCo.bat（安裝、啟動都在那裡），結束代碼照傳回去。
// 主控台是這個 exe 開的，工作列上顯示的就是 CoCo 的圖示。
using System;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Reflection;
using System.Text;

[assembly: AssemblyTitle("CoCo Image Studio")]
[assembly: AssemblyProduct("CoCo Image Studio")]
[assembly: AssemblyCopyright("Copyright 2026 Chocochocococo")]
// 原始碼的雜湊值（make_launcher.py 編的時候填）：測試拿它確認 exe 是照現在的原始碼編的
[assembly: AssemblyInformationalVersion("SOURCE_HASH")]

static class Launcher
{
    static int Main()
    {
        string dir = AppDomain.CurrentDomain.BaseDirectory;
        string bat = Path.Combine(Path.Combine(dir, "data"), "CoCo.bat");
        if (!File.Exists(bat))
        {
            // 跟伺服器一樣看 Windows 的顯示語言：中文用繁中，其他英文
            bool zh = CultureInfo.CurrentUICulture.TwoLetterISOLanguageName == "zh";
            Console.OutputEncoding = Encoding.UTF8;
            Console.WriteLine(zh ? "[X] 找不到 data\\CoCo.bat，請先把整個壓縮檔完整解壓縮。"
                                 : "[X] data\\CoCo.bat is missing. Extract the whole archive first.");
            Console.WriteLine(zh ? "按任意鍵關閉。" : "Press any key to close.");
            if (!Console.IsInputRedirected) Console.ReadKey(true);
            return 1;
        }
        // Ctrl+C 交給 CoCo.bat 與伺服器處理，這個程式自己不跟著結束（不然主控台會被提早關掉）
        Console.CancelKeyPress += (s, e) => { e.Cancel = true; };
        var psi = new ProcessStartInfo(Environment.GetEnvironmentVariable("ComSpec") ?? "cmd.exe", "/c \"\"" + bat + "\"\"");
        psi.UseShellExecute = false;
        psi.WorkingDirectory = Path.GetDirectoryName(bat);
        using (var p = Process.Start(psi))
        {
            p.WaitForExit();
            return p.ExitCode;
        }
    }
}
