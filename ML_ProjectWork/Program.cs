using ML_ProjectWork.AutoML;
using ML_ProjectWork.Helpers;
using ML_ProjectWork.ML;
using ML_ProjectWork.ML.Enum;
using ML_ProjectWork.Models;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;

namespace ML_ProjectWork
{
    internal class Program
    {
        private static void Main()
        {
            System.Threading.Thread.CurrentThread.CurrentCulture = new System.Globalization.CultureInfo("en-US");
            var dataPath = @"..\..\..\..\kc_house_data.csv";

            // всевозможные тренера (часть закомменчена для быстроты получения результата)
            var trainers = new List<TrainerModel>
            {
                TrainerModel.LbfgsPoissonRegression,
                //TrainerModel.FastForest,
                //TrainerModel.FastTree,
                TrainerModel.FastTreeTweedie,
                // TrainerModel.Gam,
                // TrainerModel.LightGbm,
                // TrainerModel.Ols,
                // TrainerModel.OnlineGradientDescent,
                // TrainerModel.Sdca
            };

            var anomalyDetector = new AnomalyDetector(dataPath);
            anomalyDetector.FindAnomalies();

            var meanAbsoluteError = 0.0;
            var bestModel = "";
            long time = 0;
            var timer = new Stopwatch();
            var rSquared = 0.0;
            var rootMeanSquaredError = 0.0;

            var house = new HouseModel
            {
                Id = 12,
                Bedrooms = 1,
                Bathrooms = 2,
                LivingArea = 200,
                Area = 250,
                Floors = 2,
                IsWaterFront = 1,
                View = 3,
                Condition = 4,
                Grade = 3,
                SqftAbove = 2,
                SqftBasement = 3,
                YearBuilt = 2012,
                YearRenovation = 2015,
                ZipCode = 01,
                Lat = (float)0.5112,
                Long = 40,
                SqftLiving15 = 500,
                SqftLot15 = 750,
            };

            var houseModels = File.ReadAllLines(dataPath)
                .Skip(1)
                .Select(ProcessingData)
                .ToArray();

            var normalizer = new Normalizer<HouseModel>(houseModels);
            normalizer.Normalize(house);

            foreach (var trainer in trainers)
            {
                timer.Start();

                IModel model = new RegressionModel(dataPath, 12, trainer);
                model.Fit();

                // Неточность предсказаний связана с аномальными данными в исходных
                PredictionModel.Predict(model, house, "100_000");

                // чтобы не засорять вывод в консоли
                // DataPresentor.PeekDataViewInConsole(3, model.PredictedData);

                timer.Stop();

                if (model.RSquared >= rSquared)
                {
                    meanAbsoluteError = model.MeanAbsoluteError;
                    rSquared = model.RSquared;
                    rootMeanSquaredError = model.RootMeanSquaredError;
                    bestModel = trainer.ToString();
                    time = timer.ElapsedMilliseconds;
                }

                timer.Reset();
            }

            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.WriteLine($"Best trainer: {bestModel}\n" +
                              $"Mean Absolute Error: {meanAbsoluteError}\n" +
                              $"Root Mean Squared Error: {rootMeanSquaredError}\n" +
                              $"RSquared: {rSquared:P2}\n" +
                              $"Time: {(double)time / 1000} s\n");

            var autoMl = new AutoMl<HouseModel>(dataPath, (a, _) => ProcessingData(a));
            autoMl.AutoRun(30);

            Console.ForegroundColor = ConsoleColor.White;
        }

        /// <summary>
        ///     Приводит выбивающиеся фичи к верному типу
        /// </summary>
        private static HouseModel ProcessingData(string houseInfo)
        {
            var information = houseInfo.Split(',');

            return new HouseModel
            {
                Id = float.Parse(information[0].Replace("\"", "")),
                Price = float.Parse(information[2]),
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
    }
}
