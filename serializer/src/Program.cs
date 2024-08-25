using System;
using System.IO;
using System.Text.Json;
using System.Collections.Generic;

namespace MajdataEdit
{
    class Program
    {
        static void Main(string[] args)
        {
            // 检查是否提供了文件绝对路径和 level_index 参数
            if (args.Length < 2)
            {
                Console.WriteLine("请提供要处理的文件绝对路径和 level_index 作为参数。");
                Environment.Exit(1); // 返回码 1 表示未提供足够的参数
            }

            // 获取文件绝对路径和 level_index
            string filePath = args[0];
            if (!int.TryParse(args[1], out int selectedLevel) || selectedLevel < 1 || selectedLevel > 6)
            {
                Console.WriteLine("无效的 level_index 参数。");
                Environment.Exit(2); // 返回码 2 表示无效的 level_index 参数
            }

            string baseDirectory = AppDomain.CurrentDomain.BaseDirectory;
            string tempsDirectory = Path.Combine(baseDirectory, "temps");

            string outputFileName = $"pre-rawchart-{Guid.NewGuid()}.json";
            string outputFilePath = Path.Combine(tempsDirectory, outputFileName);

            // 检查临时目录是否存在
            if (!Directory.Exists(tempsDirectory))
            {
                Directory.CreateDirectory(tempsDirectory);
            }

            // 检查文件是否存在
            if (!File.Exists(filePath))
            {
                Console.WriteLine("未找到指定的文件。");
                Environment.Exit(3); // 返回码 3 表示文件未找到
            }

            // 清空之前的 fumen 数据
            ClearFumens();

            // 读取并处理文件
            bool success = SimaiProcess.ReadData(filePath);

            if (!success)
            {
                Console.WriteLine("读取文件失败。");
                Environment.Exit(4); // 返回码 4 表示读取文件失败
            }

            // 检查选定的 level_index 是否有效
            if (SimaiProcess.fumens.Length <= selectedLevel || string.IsNullOrEmpty(SimaiProcess.fumens[selectedLevel]))
            {
                Console.WriteLine("无效的 level_index 选择。");
                Environment.Exit(5); // 返回码 5 表示无效的 level_index 选择
            }

            // 处理选定的 level_index
            string SetRawFumenText = SimaiProcess.fumens[selectedLevel];
            SimaiProcess.Serialize(SetRawFumenText);

            foreach (var note in SimaiProcess.notelist)
            {
                note.noteList = note.getNotes();
            }

            var jsonOutput = new List<object>();

            for (int i = 0; i < SimaiProcess.notelist.Count; i++)
            {
                var noteData = new
                {
                    Time = SimaiProcess.notelist[i].time,
                    Notes = new List<Dictionary<string, object>>()
                };

                for (int j = 0; j < SimaiProcess.notelist[i].noteList.Count; j++)
                {
                    var note = SimaiProcess.notelist[i].noteList[j];
                    var noteProperties = new Dictionary<string, object>
                    {
                        { "holdTime", note.holdTime },
                        { "isBreak", note.isBreak },
                        { "isEx", note.isEx },
                        { "isFakeRotate", note.isFakeRotate },
                        { "isForceStar", note.isForceStar },
                        { "isHanabi", note.isHanabi },
                        { "isSlideBreak", note.isSlideBreak },
                        { "isSlideNoHead", note.isSlideNoHead },
                        { "noteContent", note.noteContent ?? string.Empty }, // 处理可能为 null 的情况
                        { "noteType", note.noteType.ToString() },
                        { "slideStartTime", note.slideStartTime },
                        { "slideTime", note.slideTime },
                        { "startPosition", note.startPosition },
                        { "touchArea", note.touchArea }
                    };

                    noteData.Notes.Add(noteProperties);
                }

                jsonOutput.Add(noteData);
            }

            // 将 JSON 保存到指定目录
            string jsonString = JsonSerializer.Serialize(jsonOutput, new JsonSerializerOptions { WriteIndented = true });
            File.WriteAllText(outputFilePath, jsonString);

            // 删除处理成功的 .txt 文件
            try
            {
                File.Delete(filePath);
                Console.WriteLine($"成功删除文件: {filePath}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"删除文件失败: {ex.Message}");
            }

            Console.WriteLine($"成功加载 level_index {selectedLevel} 并保存到 {outputFilePath}");
            Console.WriteLine(outputFileName); // 输出文件名
            Environment.Exit(0); // 返回码 0 表示成功
        }

        static void ClearFumens()
        {
            // 清空 SimaiProcess.fumens 数组
            for (int i = 0; i < SimaiProcess.fumens.Length; i++)
            {
                SimaiProcess.fumens[i] = string.Empty;
            }
        }
    }
}
