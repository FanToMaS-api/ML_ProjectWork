using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.TimeSeries;
using ML_ProjectWork.Models;

namespace ML_ProjectWork.Helpers
{
    /// <summary>
    ///     Ищет аномалии в данных
    /// </summary>
    internal class AnomalyDetector
    {
        #region Fields

        private readonly MLContext _mlContext;

        /// <summary>
        ///     Разделитель данных в файле
        /// </summary>
        private readonly char _separatorChar;

        private readonly IDataView _anomalyDataView;

        /// <summary>
        ///     Пороговое значение для определения аномалии
        /// </summary>
        private readonly double _threshold;

        /// <summary>
        ///     Пороговое значение для определения аномалии
        /// </summary>
        private readonly double _sensitivity;

        #endregion

        #region .ctor

        public AnomalyDetector(string dataPath, double sensitivity = 85, double threshold = 0.5, char separatorChar = ',')
        {
            _threshold = threshold;
            _separatorChar = separatorChar;
            _mlContext = new MLContext(20212121);
            _sensitivity = sensitivity;

            // Получаю и обрабатываю данные для поиска аномалий различаются double и float у Label
            AnomalyHouses = File.ReadAllLines(dataPath)
                .Skip(1)
                .Select(ProcessingAnomalyData)
                .ToArray();

            //  Данные для поиска аномалий, различаются типом AnomalyHouseModel и HouseModel
            _anomalyDataView = _mlContext.Data.LoadFromEnumerable(AnomalyHouses);
        }

        #endregion

        #region Properties

        /// <summary>
        ///     Массив объектного представления данных
        /// </summary>
        public AnomalyHouseModel[] AnomalyHouses { get; init; }

        #endregion

        #region Public methods

        /// <summary>
        ///     Ищет аномалии в данных
        /// </summary>
        /// <returns>
        ///     Возвращает список с номерами строк с аномальными данными
        /// </returns>
        public List<int> FindAnomalies()
        {
            // TODO Добавить удаление аномалий
            // Поиск сезонности у Label (цикличности)
            var seasonality = _mlContext.AnomalyDetection.DetectSeasonality(_anomalyDataView, "Label");

            // Определение аномалий
            // Threshold = Пороговое значение для определения аномалии,
            // BatchSize = Разделение входных данных на части (-1 = использование всех данных)
            // Sensitivity = Чувствительность
            // DetectMode = Задает тип выходных данных (размерность вектора)
            // DeseasonalityMode = Настройка поиска
            var result = _mlContext.AnomalyDetection.DetectEntireAnomalyBySrCnn(
                _anomalyDataView,
                nameof(PredictionAnomalyModel.Preds),
                "Label",
                new SrCnnEntireAnomalyDetectorOptions
                {
                    Threshold = _threshold,
                    BatchSize = -1,
                    Period = seasonality,
                    Sensitivity = _sensitivity,
                    DetectMode = SrCnnDetectMode.AnomalyAndMargin,
                    DeseasonalityMode = SrCnnDeseasonalityMode.Median
                });

            var predictions = _mlContext.Data.CreateEnumerable<PredictionAnomalyModel>(result, false);

            var count = 0;
            var listToSkip = new List<int>();
            var allCount = 0;
            foreach (var prediction in predictions)
            {
                // Preds[0] принимает значения 0 или 1, 1 только в том случае, если для данные превысили Threshold
                if (prediction.Preds[0] == 1)
                {
                    count++;
                    listToSkip.Add(allCount);
                }

                allCount++;
            }

            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine($"Count all anomalies: {count}");
            Console.ForegroundColor = ConsoleColor.White;

            return listToSkip;
        }

        #endregion

        #region Private methods

        /// <summary>
        ///     Приводит выбивающиеся фичи к верному типу для поиска аномалий
        /// </summary>
        private AnomalyHouseModel ProcessingAnomalyData(string houseInfo)
        {
            var information = houseInfo.Split(_separatorChar);

            return new AnomalyHouseModel
            {
                Id = float.Parse(information[0].Replace("\"", "")),
                Price = double.Parse(information[2]),
                Bedrooms = float.Parse(information[3]),
                Bathrooms = float.Parse(information[4]),
                LivingArea = float.Parse(information[5]),
                Area = float.Parse(information[6]),
                Floors = float.Parse(information[7].Replace("\"", "")),
                IsWaterFront = float.Parse(information[8]),
                View = float.Parse(information[9]),
                Condition = float.Parse(information[10]),
                Grade = float.Parse(information[11]),
                SqftAbove = float.Parse(information[12]),
                SqftBasement = float.Parse(information[13]),
                YearBuilt = float.Parse(information[14]),
                YearRenovation = float.Parse(information[15].Replace("\"", "")),
                ZipCode = float.Parse(information[16].Replace("\"", "")),
                Lat = float.Parse(information[17]),
                Long = float.Parse(information[18]),
                SqftLiving15 = float.Parse(information[19]),
                SqftLot15 = float.Parse(information[20]),
            };
        }

        #endregion
    }
}
