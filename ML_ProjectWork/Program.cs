using System;

namespace ML_ProjectWork
{
    internal class Program
    {
        private static void Main()
        {
            var dataPath = @"..\..\..\..\kc_house_data.csv";

            var model = new Model(dataPath, 3, TrainerModel.LbfgsPoissonRegression, nameof(HouseModel.Price));
            model.Fit();

            Console.WriteLine($"Model accuracy: {model.Accuracy}");
        }
    }
}
