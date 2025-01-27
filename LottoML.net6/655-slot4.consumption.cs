﻿// This file was auto-generated by ML.NET Model Builder. 

using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Linq;
using System.IO;
using System.Collections.Generic;
using Microsoft.ML.Transforms.TimeSeries;

namespace LottoML_net6
{
    public partial class _655_slot4
    {
        /// <summary>
        /// model input class for _655_slot4.
        /// </summary>
        #region model input class
        public class ModelInput
        {
            [LoadColumn(0)]
            [ColumnName(@"result")]
            public float Result { get; set; }

        }

        #endregion

        /// <summary>
        /// model output class for _655_slot4.
        /// </summary>
        #region model output class
        public class ModelOutput
        {
            [ColumnName(@"result")]
            public float[] Result { get; set; }

            [ColumnName(@"result_LB")]
            public float[] Result_LB { get; set; }

            [ColumnName(@"result_UB")]
            public float[] Result_UB { get; set; }

        }

        #endregion

        private static string MLNetModelPath = Path.GetFullPath(@"G:\projects\loabletech\LottoML.net\LottoML.net6\655-slot4.zip");

        public static readonly Lazy<TimeSeriesPredictionEngine<ModelInput, ModelOutput>> PredictEngine = new Lazy<TimeSeriesPredictionEngine<ModelInput, ModelOutput>>(() => CreatePredictEngine(), true);

        /// <summary>
        /// Use this method to predict on <see cref="ModelInput"/>.
        /// </summary>
        /// <param name="input">model input.</param>
        /// <returns><seealso cref=" ModelOutput"/></returns>
        public static ModelOutput Predict(ModelInput? input = null, int? horizon = null)
        {
            var predEngine = PredictEngine.Value;
            return predEngine.Predict(input, horizon);
        }

        private static TimeSeriesPredictionEngine<ModelInput, ModelOutput> CreatePredictEngine()
        {
            var mlContext = new MLContext();
            ITransformer mlModel = mlContext.Model.Load(MLNetModelPath, out var schema);
            return mlModel.CreateTimeSeriesEngine<ModelInput, ModelOutput>(mlContext);
        }
    }
}

