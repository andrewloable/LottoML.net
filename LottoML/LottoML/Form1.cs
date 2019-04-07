using MaterialSkin;
using MaterialSkin.Controls;
using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace LottoML
{
    public partial class Form1 : MaterialForm
    {
        private List<List<float>> data = new List<List<float>>();
        List<string> predictedResult;

        public Form1()
        {
            InitializeComponent();

            // Create a material theme manager and add the form to manage (this)
            MaterialSkinManager materialSkinManager = MaterialSkinManager.Instance;
            materialSkinManager.AddFormToManage(this);
            materialSkinManager.Theme = MaterialSkinManager.Themes.LIGHT;

            // Configure color schema
            materialSkinManager.ColorScheme = new ColorScheme(
                Primary.Blue400, Primary.Blue500,
                Primary.Blue500, Accent.LightBlue200,
                TextShade.WHITE
            );
        }

        private void MaterialCheckBox1_CheckedChanged(object sender, EventArgs e)
        {

        }

        private void log(string message)
        {
            txtLogs.AppendText(DateTime.Now.ToString() + " : " + message + Environment.NewLine);
        }

        private string[] GenerateCSVForTraining(List<List<float>> input)
        {
            List<string> retval = new List<string>();
            int slots = input[0].Count;
            for(int j=0; j<slots; j++)
            {
                string fn = Path.GetTempFileName();

                StringBuilder sb = new StringBuilder();
                sb.Append("prev1,prev2,prev3,prev4,prev5,prev6,prev7,prev8,prev9,result").Append(Environment.NewLine);

                for (int i = 9; i < input.Count; i++)
                {
                    sb.Append(input[i - 9][j]).Append(",");
                    sb.Append(input[i - 8][j]).Append(",");
                    sb.Append(input[i - 7][j]).Append(",");
                    sb.Append(input[i - 6][j]).Append(",");
                    sb.Append(input[i - 5][j]).Append(",");
                    sb.Append(input[i - 4][j]).Append(",");
                    sb.Append(input[i - 3][j]).Append(",");
                    sb.Append(input[i - 2][j]).Append(",");
                    sb.Append(input[i - 1][j]).Append(",");
                    sb.Append(input[i - 9][j]).Append(Environment.NewLine);
                }
                File.WriteAllText(fn, sb.ToString());
                retval.Add(fn);
            }            

            return retval.ToArray();
        }

        private void BtnBrowseCSV_Click(object sender, EventArgs e)
        {
            OpenFileDialog opf = new OpenFileDialog();
            opf.Filter = "CSV Files|*.csv";
            opf.Multiselect = false;
            if (opf.ShowDialog() == DialogResult.OK)
            {
                txtCSV.Text = opf.FileName;
                string[] lines = File.ReadAllLines(txtCSV.Text);
                if (lines.Length < 9)
                {
                    log("Insufficient Input Data");
                    MessageBox.Show("Insufficient Data, Need At Least 9 Records", "Cannot Process", MessageBoxButtons.OK, MessageBoxIcon.Stop);
                } else
                {
                    log("Processing Input");
                    int start = chkHeader.Checked ? 1 : 0;
                    for (int i = start; i < lines.Length; i++)
                    {
                        List<float> lineData = new List<float>();
                        var line = lines[i];
                        var columns = line.Split(new char[] { ',' });
                        foreach(var c in columns)
                        {
                            lineData.Add(float.Parse(c));
                        }
                        if (chkSort.Checked)
                        {
                            var sorted = lineData.OrderBy(r => r).ToList();
                            data.Add(sorted);
                        }
                        else
                        {
                            data.Add(lineData);
                        }
                    }
                    log("Data Loaded");
                    MessageBox.Show("Data Loaded. Press Generate Model To Start the Training", "Data Loaded", MessageBoxButtons.OK, MessageBoxIcon.Information);
                }
            }
        }

        private void BtnGenerateModel_Click(object sender, EventArgs e)
        {
            if (data != null && data.Count > 0)
            {
                groupBox1.Enabled = false;
                predictedResult = new List<string>();
                List<LottoWorker> workers = new List<LottoWorker>();
                var fn = GenerateCSVForTraining(data);

                for (int i=0; i<fn.Length; i++)
                {
                    string f = fn[i];                    

                    BackgroundWorker bw = new BackgroundWorker();
                    bw.WorkerReportsProgress = true;
                    bw.DoWork += Bw_DoWork;
                    bw.ProgressChanged += Bw_ProgressChanged;
                    bw.RunWorkerCompleted += Bw_RunWorkerCompleted;

                    LottoWorker lw = new LottoWorker();
                    lw.Worker = bw;
                    lw.Filename = f;
                    lw.Slot = i + 1;
                    workers.Add(lw);

                    lw.Worker.RunWorkerAsync(lw);
                }

                while (true)
                {
                RESTART:
                    Thread.Sleep(2000);
                    Application.DoEvents();
                    foreach (var lw in workers)
                    {
                        if (lw.Worker.IsBusy)
                        {
                            goto RESTART;
                        }
                    }
                    break;
                }

                string res = "PREDICTED RESULTS : " + String.Join(" - ", predictedResult.ToArray());
                log(res);
                MessageBox.Show(res, "Result", MessageBoxButtons.OK, MessageBoxIcon.Information);
                groupBox1.Enabled = true;
            } else
            {
                MessageBox.Show("Please load a dataset via CSV.", "No Data Available", MessageBoxButtons.OK, MessageBoxIcon.Stop);
            }
        }

        private void Bw_RunWorkerCompleted(object sender, RunWorkerCompletedEventArgs e)
        {
            LottoWorker lw = (LottoWorker)e.Result;
            var loadedModel = lw.Model;
            var mlContext = new MLContext();
            int lastRecord = data.Count - 1;
            var predictionEngine = mlContext.Model.CreatePredictionEngine<LottoData, LottoDataPrediction>(loadedModel);
            var lt = new LottoData()
            {
                prev1 = data[lastRecord - 8][lw.Slot-1],
                prev2 = data[lastRecord - 7][lw.Slot - 1],
                prev3 = data[lastRecord - 6][lw.Slot - 1],
                prev4 = data[lastRecord - 5][lw.Slot - 1],
                prev5 = data[lastRecord - 4][lw.Slot - 1],
                prev6 = data[lastRecord - 3][lw.Slot - 1],
                prev7 = data[lastRecord - 2][lw.Slot - 1],
                prev8 = data[lastRecord - 1][lw.Slot - 1],
                prev9 = data[lastRecord][lw.Slot - 1]
            };
            var predict = predictionEngine.Predict(lt);
            log("PREDICTED SLOT [" + lw.Slot + "] - " + predict.predictedresult.ToString());
            predictedResult.Add(predict.predictedresult.ToString());
            if (File.Exists(lw.Filename))
            {
                File.Delete(lw.Filename);
            }
        }

        private void Bw_DoWork(object sender, DoWorkEventArgs e)
        {
            LottoWorker lw = (LottoWorker)e.Argument;

            lw.Worker.ReportProgress(0, "SLOT [" + lw.Slot + "] - Reading Training Data");
            var mlContext = new MLContext();
            var loader = mlContext.Data.CreateTextLoader(new[]
            {
                new TextLoader.Column("prev1", DataKind.Single, 0),
                new TextLoader.Column("prev2", DataKind.Single, 1),
                new TextLoader.Column("prev3", DataKind.Single, 2),
                new TextLoader.Column("prev4", DataKind.Single, 3),
                new TextLoader.Column("prev5", DataKind.Single, 4),
                new TextLoader.Column("prev6", DataKind.Single, 5),
                new TextLoader.Column("prev7", DataKind.Single, 6),
                new TextLoader.Column("prev8", DataKind.Single, 7),
                new TextLoader.Column("prev9", DataKind.Single, 8),
                new TextLoader.Column("result", DataKind.Single, 9),
            }, hasHeader: true, separatorChar: ',');

            var trainingData = loader.Load(lw.Filename);
            var learningPipeline = mlContext.Transforms.Conversion.MapValueToKey("result")
                .Append(mlContext.Transforms.Concatenate("Features", "prev1", "prev2", "prev3", "prev4", "prev5", "prev6", "prev7", "prev8", "prev9"))
                .Append(mlContext.MulticlassClassification.Trainers.SdcaNonCalibrated(labelColumnName: "result", featureColumnName: "Features"))
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            lw.Worker.ReportProgress(0, "SLOT [" + lw.Slot + "] - Training Data. This may take a lot of time...");
            lw.Model = learningPipeline.Fit(trainingData);
            e.Result = lw;
        }

        private void Bw_ProgressChanged(object sender, ProgressChangedEventArgs e)
        {
            log(e.UserState.ToString());
        }
    }
}
