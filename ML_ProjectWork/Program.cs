using ML_ProjectWork.ML;
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

            // всевозможные тренера (часть закомменчена для быстроты получения результата)
            var trainers = new List<TrainerModel>
            {
                TrainerModel.LbfgsPoissonRegression,
                TrainerModel.FastForest,
                TrainerModel.FastTree,
                TrainerModel.FastTreeTweedie,
                // TrainerModel.Gam,
                // TrainerModel.LightGbm,
                // TrainerModel.Ols,
                // TrainerModel.OnlineGradientDescent,
                // TrainerModel.Sdca
            };

            var anomalyDetector = new AnomalyDetector(dataPath);
            anomalyDetector.FindAnomalies();

            var meanAbsoluteError = 150000.0;
            var bestModel = "";
            long time = 0;
            var timer = new Stopwatch();
            var rSquared = 0.0;
            var rootMeanSquaredError = 0.0;

            foreach (var trainer in trainers)
            {
                timer.Start();

                var model = new RegressionModel(dataPath, 10, trainer, isPeek: false);
                model.Fit();

                timer.Stop();

                if (model.MeanAbsoluteError <= meanAbsoluteError)
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
            Console.ForegroundColor = ConsoleColor.White;
        }
    }
}
