using ML_ProjectWork.AutoML;
using ML_ProjectWork.Helpers;
using ML_ProjectWork.ML;
using ML_ProjectWork.ML.Enum;
using ML_ProjectWork.Models;
using System;
using System.Collections.Generic;
using System.Diagnostics;

namespace ML_ProjectWork
{
    internal class Program
    {
        private static void Main()
        {
            System.Threading.Thread.CurrentThread.CurrentCulture = new System.Globalization.CultureInfo("en-US");
            var dataPath = @"..\..\..\..\kc_house_data.csv";

            // всевозможные тренера
            var trainers = new List<TrainerModel>
            {
                TrainerModel.LbfgsPoissonRegression,
                TrainerModel.FastForest,
                TrainerModel.FastTree,
                TrainerModel.FastTreeTweedie,
                TrainerModel.Gam,
                TrainerModel.LightGbm,
                TrainerModel.Ols,
                TrainerModel.OnlineGradientDescent,
                TrainerModel.Sdca
            };

            var anomalyDetector = new AnomalyDetector(dataPath, threshold: 0.5);
            var anomalyIndexes = anomalyDetector.FindAnomalies();

            var meanAbsoluteError = 0.0;
            var bestModel = "";
            long time = 0;
            var timer = new Stopwatch();
            var rSquared = 0.0;
            var rootMeanSquaredError = 0.0;

            var house = new HouseModel
            {
                Id = 7129300520,
                Bedrooms = 3,
                Bathrooms = 1,
                LivingArea = 1180,
                Area = 5650,
                Floors = 1,
                IsWaterFront = 0,
                View = 0,
                Condition = 3,
                Grade = 7,
                SqftAbove = 1180,
                SqftBasement = 0,
                YearBuilt = 1955,
                YearRenovation = 0,
                ZipCode = 98178,
                Lat = (float)47.5112,
                Long = (float)-122.257,
                SqftLiving15 = 1340,
                SqftLot15 = 5650,
            };

            // TODO: подумать над многопоточным обучением тренеров
            foreach (var trainer in trainers)
            {
                timer.Start();

                // не всегда уменьшение кол-ва фич является здравой идеей
                IModel model = new RegressionModel(dataPath, 0, trainer, isLoadSavedModel: true, anomalyIndexes);
                model.Fit();

                // model.Save();

                // Неточность предсказаний связана с аномальными данными в исходных
                PredictionModel.Predict(model, house, "221900");

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
