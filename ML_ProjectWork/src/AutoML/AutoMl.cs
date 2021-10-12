using System;
using Microsoft.ML;
using Microsoft.ML.AutoML;

namespace ML_ProjectWork.AutoML
{
    /// <summary>
    ///     Автопоиск лучшего тренера встроенными средствами Microsoft
    /// </summary>
    internal class AutoMl
    {
        /// <summary>
        ///     Запускает автоматический поиск лучшего тренера
        /// </summary>
        public static void AutoRun(uint secondsToTrain, IDataView trainData, IDataView validationData)
        {
            var mlContext = new MLContext(20212121);
            var experiment = mlContext.Auto()
                .CreateRegressionExperiment(secondsToTrain)
                .Execute(trainData, validationData);

            var bestRun = experiment.BestRun;

            Console.WriteLine($"AutoMl Best Trainer: {bestRun.TrainerName}");
            Console.WriteLine($"AutoMl Mean Absolute Error: {bestRun.ValidationMetrics.MeanAbsoluteError}");
            Console.WriteLine($"AutoMl Root Mean Squared Error: {bestRun.ValidationMetrics.RootMeanSquaredError}");
            Console.WriteLine($"AutoMl RSquared: {bestRun.ValidationMetrics.RSquared:P2}");
        }
    }
}
