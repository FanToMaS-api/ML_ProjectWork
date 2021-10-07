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
            // Поиск сезонности у Label (цикличности)
            var seasonality = mlContext.AnomalyDetection.DetectSeasonality(dataView, "Label");

            // Определение аномалий
            // Threshold = Пороговое значение для определения аномалии,
            // BatchSize = Разделение входных данных на части (-1 = использование всех данных)
            // Sensitivity = Чувствительность
            // DetectMode = Задает тип выходных данных (размерность вектора)
            // DeseasonalityMode = Настройка поиска
            var result = mlContext.AnomalyDetection.DetectEntireAnomalyBySrCnn(
                dataView,
                nameof(PredictionAnomalyModel.Preds),
                "Label",
                new SrCnnEntireAnomalyDetectorOptions
                {
                    Threshold = 0.5,
                    BatchSize = -1,
                    Period = seasonality,
                    Sensitivity = 90,
                    DetectMode = SrCnnDetectMode.AnomalyAndMargin,
                    DeseasonalityMode = SrCnnDeseasonalityMode.Median
                });

            var predictions = mlContext.Data.CreateEnumerable<PredictionAnomalyModel>(result, false);

            var count = 0;
            foreach (var prediction in predictions)
            {
                // Preds[0] принимает значения 0 или 1, 1 только в том случае, если для данные превысили Threshold
                if (prediction.Preds[0] == 1)
                {
                    count++;
                }
            }

            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine($"Count all anomalies: {count}");
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
