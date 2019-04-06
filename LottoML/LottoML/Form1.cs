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
using System.Threading.Tasks;
using System.Windows.Forms;

namespace LottoML
{
    public partial class Form1 : MaterialForm
    {
        private List<List<float>> data = new List<List<float>>();

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

        private string GenerateCSVForTraining(List<List<float>> input)
        {
            string fn = Path.GetTempFileName();

            StringBuilder sb = new StringBuilder();
            sb.Append("prev1,prev2,prev3,prev4,prev5,prev6,prev7,prev8,prev9,result").Append(Environment.NewLine);

            int slots = input[0].Count;
            for(int j=0; j<slots; j++)
            {
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
            }

            File.WriteAllText(fn, sb.ToString());

            return fn;
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
                BackgroundWorker bw = new BackgroundWorker();
                bw.WorkerReportsProgress = true;
                bw.DoWork += Bw_DoWork;                        
                bw.ProgressChanged += Bw_ProgressChanged;
                bw.RunWorkerCompleted += Bw_RunWorkerCompleted;

                groupBox1.Enabled = groupBox2.Enabled = false;
                bw.RunWorkerAsync();

            } else
            {
                MessageBox.Show("Please load a dataset via CSV.", "No Data Available", MessageBoxButtons.OK, MessageBoxIcon.Stop);
            }
        }

        private void Bw_RunWorkerCompleted(object sender, RunWorkerCompletedEventArgs e)
        {
            TransformerChain<Microsoft.ML.Transforms.KeyToValueMappingTransformer> model = (TransformerChain<Microsoft.ML.Transforms.KeyToValueMappingTransformer>)e.Result;
            BackgroundWorker bw = (BackgroundWorker)sender;

            SaveFileDialog sfd = new SaveFileDialog();
            sfd.Filter = "ML Model|*.ai";
            if (sfd.ShowDialog() == DialogResult.OK)
            {
                if (File.Exists(sfd.FileName))
                {
                    File.Delete(sfd.FileName);
                }

                using (var fileStream = new FileStream(sfd.FileName, FileMode.Create, FileAccess.Write, FileShare.Write))
                {
                    MLContext mlContext = new MLContext();
                    log("Saving Training Data");
                    mlContext.Model.Save(model, null, fileStream);
                }
                txtModel.Text = sfd.FileName;
                MessageBox.Show("Training Data Generated", "Training Complete", MessageBoxButtons.OK, MessageBoxIcon.Information);
                groupBox1.Enabled = groupBox2.Enabled = true;
            }
        }

        private void Bw_DoWork(object sender, DoWorkEventArgs e)
        {
            BackgroundWorker bw = (BackgroundWorker)sender;
            bw.ReportProgress(0, "Generate CSV for Training");
            var fn = GenerateCSVForTraining(data);

            bw.ReportProgress(0, "Reading Training Data");
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

            var trainingData = loader.Load(fn);
            var learningPipeline = mlContext.Transforms.Conversion.MapValueToKey("result")
                .Append(mlContext.Transforms.Concatenate("Features", "prev1", "prev2", "prev3", "prev4", "prev5", "prev6", "prev7", "prev8", "prev9"))
                .Append(mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy(labelColumnName: "result", featureColumnName: "Features"))
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            bw.ReportProgress(0, "Training Data. This may take a lot of time...");
            TransformerChain<Microsoft.ML.Transforms.KeyToValueMappingTransformer> model = learningPipeline.Fit(trainingData);

            e.Result = model;
        }

        private void Bw_ProgressChanged(object sender, ProgressChangedEventArgs e)
        {
            log(e.UserState.ToString());
        }

        private void BtnPredict_Click(object sender, EventArgs e)
        {
            groupBox1.Enabled = groupBox2.Enabled = false;
            var mlContext = new MLContext();
            ITransformer loadedModel;
            using (var fileStream = new FileStream(txtModel.Text, FileMode.Open, FileAccess.Read, FileShare.Read))
            {                
                DataViewSchema dvs;
                loadedModel = mlContext.Model.Load(fileStream, out dvs);
            }

            List<string> predictedResult = new List<string>();
            if (data != null && data.Count > 0)
            {
                int slots = data[0].Count;
                for (int j = 0; j < slots; j++)
                {
                    int lastRecord = data.Count - 1;
                    var predictionEngine = mlContext.Model.CreatePredictionEngine<LottoData, LottoDataPrediction>(loadedModel);
                    var predict = predictionEngine.Predict(
                        new LottoData()
                        {
                            prev1 = data[lastRecord - 8][j],
                            prev2 = data[lastRecord - 7][j],
                            prev3 = data[lastRecord - 6][j],
                            prev4 = data[lastRecord - 5][j],
                            prev5 = data[lastRecord - 4][j],
                            prev6 = data[lastRecord - 3][j],
                            prev7 = data[lastRecord - 2][j],
                            prev8 = data[lastRecord - 1][j],
                            prev9 = data[lastRecord][j]
                        }
                        );
                    predictedResult.Add(predict.predictedresult.ToString());
                }

                string res = "PREDICTED RESULTS : " + String.Join(" - ", predictedResult.ToArray());
                log(res);
                MessageBox.Show(res, "Result", MessageBoxButtons.OK, MessageBoxIcon.Information);
            }
            groupBox1.Enabled = groupBox2.Enabled = true;
        }

        private void BtnBrowseModel_Click(object sender, EventArgs e)
        {
            OpenFileDialog opf = new OpenFileDialog();
            opf.Filter = "ML Model|*.ai";
            opf.Multiselect = false;
            if (opf.ShowDialog() == DialogResult.OK)
            {
                txtModel.Text = opf.FileName;
                MessageBox.Show("Data Loaded. Press Generate Model To Start the Training", "Data Loaded", MessageBoxButtons.OK, MessageBoxIcon.Information);
            }
        }
    }
}
