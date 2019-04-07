using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LottoML
{
    public class Lotto
    {
        public float Slot1 { get; set; }
        public float Slot2 { get; set; }
        public float Slot3 { get; set; }
        public float Slot4 { get; set; }
        public float Slot5 { get; set; }
        public float Slot6 { get; set; }
        public int Slots { get; set; }
    }

    public class LottoData
    {

        [LoadColumn(0)]
        public float prev1;

        [LoadColumn(1)]
        public float prev2;

        [LoadColumn(2)]
        public float prev3;

        [LoadColumn(3)]
        public float prev4;

        [LoadColumn(4)]
        public float prev5;

        [LoadColumn(5)]
        public float prev6;

        [LoadColumn(6)]
        public float prev7;

        [LoadColumn(7)]
        public float prev8;

        [LoadColumn(8)]
        public float prev9;

        [LoadColumn(9)]
        public float result;
    }

    public class LottoDataPrediction
    {
        [ColumnName("PredictedLabel")]
        public float predictedresult;
    }

    public class LottoWorker
    {
        public BackgroundWorker Worker { get; set; }
        public int Slot { get; set; }
        public string Filename { get; set; }
        public TransformerChain<Microsoft.ML.Transforms.KeyToValueMappingTransformer> Model { get; set; }
    }
}
