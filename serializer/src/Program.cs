using System.Text.Json;

namespace MajdataEdit
{
    class Program
    {
        static void Main(string[] args)
        {
            // 设置文件路径
            string filePath = Path.Combine(Directory.GetCurrentDirectory(), "maidata.txt");
            string outputFilePath = Path.Combine(Directory.GetCurrentDirectory(), "pre-rawchart.json");

            // 检查文件是否存在
            if (!File.Exists(filePath))
            {
                Console.WriteLine("未找到 maidata.txt 文件。");
                Pause();
                return;
            }

            // 清空之前的 fumen 数据
            ClearFumens();

            // 读取并处理文件
            bool success = SimaiProcess.ReadData(filePath);

            if (!success)
            {
                Console.WriteLine("读取 maidata.txt 文件失败。");
                Pause();
                return;
            }

            // 列出所有可用的 level_index
            List<int> availableLevels = new List<int>();
            for (int level_index = 1; level_index <= 5; level_index++)
            {
                if (SimaiProcess.fumens.Length > level_index && !string.IsNullOrEmpty(SimaiProcess.fumens[level_index]))
                {
                    availableLevels.Add(level_index);
                }
            }

            if (availableLevels.Count == 0)
            {
                Console.WriteLine("未找到任何可用的 level_index。");
                Pause();
                return;
            }

            Console.WriteLine("可用的 level_index:");
            foreach (var level in availableLevels)
            {
                Console.WriteLine(level);
            }

            Console.Write("请选择要加载的 level_index: ");
            if (!int.TryParse(Console.ReadLine(), out int selectedLevel) || !availableLevels.Contains(selectedLevel))
            {
                Console.WriteLine("无效的 level_index 选择。");
                Pause();
                return;
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

            Console.WriteLine($"成功加载 level_index {selectedLevel} 并保存到 {outputFilePath}");
            Pause();
        }

        static void ClearFumens()
        {
            // 清空 SimaiProcess.fumens 数组
            for (int i = 0; i < SimaiProcess.fumens.Length; i++)
            {
                SimaiProcess.fumens[i] = null;
            }
        }

        static void Pause()
        {
            Console.WriteLine("按任意键继续...");
            Console.ReadLine();
        }
    }
}
