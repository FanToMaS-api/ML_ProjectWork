using System;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.AutoML;

namespace ML_ProjectWork.AutoML
{
    /// <summary>
    ///     Автопоиск лучшего тренера встроенными средствами Microsoft
    /// </summary>
    internal class AutoMl<T>
    where T : class
    {
        #region Private methods

        private readonly IDataView _dataView;

        #endregion

        #region .ctor

        /// <summary>
        ///     Конструктор с инициализацией через путь к файлу с данными
        /// </summary>
        public AutoMl(string dataPath, Func<string, int, T> func)
        {
            var objects = File.ReadAllLines(dataPath)
                .Skip(1)
                .Select(func)
                .ToArray();

            _dataView = new MLContext(20212121).Data.LoadFromEnumerable(objects);
        }

        /// <summary>
        ///     Конструктор с инициализацией через уже собранные данные
        /// </summary>
        public AutoMl(IDataView dataView)
        {
            _dataView = dataView;
        }

        #endregion

        #region Properties

        /// <summary>
        ///     Дучшая модель
        /// </summary>
        public ITransformer BestModel { get; private set; }

        #endregion

        #region Public methods

        /// <summary>
        ///     Запускает автоматический поиск лучшего тренера
        /// </summary>
        public void AutoRun(uint secondsToTrain)
        {
            var mlContext = new MLContext(20212121);
            var experiment = mlContext.Auto()
                .CreateRegressionExperiment(secondsToTrain)
                .Execute(_dataView);

            var bestRun = experiment.BestRun;
            BestModel = experiment.BestRun.Model;

            Console.WriteLine($"AutoMl Best Trainer: {bestRun.TrainerName}");
            Console.WriteLine($"AutoMl Mean Absolute Error: {bestRun.ValidationMetrics.MeanAbsoluteError:#.##}");
            Console.WriteLine($"AutoMl Root Mean Squared Error: {bestRun.ValidationMetrics.RootMeanSquaredError:#.##}");
            Console.WriteLine($"AutoMl RSquared: {bestRun.ValidationMetrics.RSquared:P2}");
        }

        #endregion
    }
}
