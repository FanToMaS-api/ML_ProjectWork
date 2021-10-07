using Microsoft.ML;
using Microsoft.ML.TimeSeries;
using ML_ProjectWork.Models;
using System;
using System.Collections.Generic;
using System.Linq;

namespace ML_ProjectWork
{
    /// <summary>
    ///     Ищет аномалии в данных
    /// </summary>
    internal static class AnomalyDetector
    {
        #region Public methods

        /// <summary>
        ///     Ищет аномалии в данных
        /// </summary>
        public static void FindAnomalies(MLContext mlContext, IDataView dataView)
        {
            var result = mlContext.AnomalyDetection.DetectEntireAnomalyBySrCnn(
                dataView,
                nameof(PredictionAnomalyModel.Preds),
                "Label",
                new SrCnnEntireAnomalyDetectorOptions
                {
                    Threshold = 0.05,
                    BatchSize = -1,
                    Period = 7,
                    Sensitivity = 80,
                    DetectMode = SrCnnDetectMode.AnomalyAndMargin,
                    DeseasonalityMode = SrCnnDeseasonalityMode.Median
                });

            var predictions = mlContext.Data.CreateEnumerable<PredictionAnomalyModel>(result, reuseRowObject: false);

            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine($"Count all anomalies: {predictions.ToList().Count}");
            Console.ForegroundColor = ConsoleColor.White;
        }

        /// <summary>
        ///     Выводит информацию об аномалиях
        /// </summary>
        public static void PrintInfo(IEnumerable<PredictionAnomalyModel> predictions)
        {
            var i = 2;
            var count = 0;
            foreach (var prediction in predictions)
            {
                if (prediction.Preds[0] == 1)
                {
                    Console.Write($"Row in file: {i} || ");
                    foreach (var pred in prediction.Preds)
                    {
                        Console.Write($"{pred}\t");
                    }

                    count++;
                    Console.WriteLine();
                }

                i++;
            }
        }
    }

    #endregion
}
