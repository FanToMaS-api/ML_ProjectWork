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

            var bestAccuracy = 0.0;
            var bestModel = "";
            var time = 100000000000;
            var timer = new Stopwatch();
            var isNeedAnomalyCalc = true;
            var testHouseModel = new HouseModel
            {
                Id = 12,
                Bedrooms = 2,
                Bathrooms = 3,
                LivingArea = 154,
                Area = 177,
                Floors = 2,
                IsWaterFront = 0,
                View = 2,
                Condition = 3,
                Grade = 3,
                SqftAbove = 3,
                SqftBasement = 4,
                YearBuilt = 1980,
                YearRenovation = 2010,
                ZipCode = 1353415,
                Lat = (float)47.5112,
                Long = (float)-122.257,
                SqftLiving15 = 250,
                SqftLot15 = 900,
            };

            foreach (var trainer in trainers)
            {
                timer.Start();

                var model = new Model(dataPath, 15, trainer, isNeedAnomalyCalc);
                model.Fit();

                model.Predict(testHouseModel);
                timer.Stop();

                if (model.Accuracy >= bestAccuracy)
                {
                    bestAccuracy = model.Accuracy;
                    bestModel = trainer.ToString();
                    time = timer.ElapsedMilliseconds;
                }

                isNeedAnomalyCalc = false;
                timer.Reset();
            }

            Console.WriteLine($"Best trainer: {bestModel}\nAccuracy: {bestAccuracy}\nTime: {(double)time / 1000} s\n");
        }
    }
}
