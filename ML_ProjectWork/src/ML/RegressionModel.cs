using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Trainers.LightGbm;
using ML_ProjectWork.Helpers;
using ML_ProjectWork.Models;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace ML_ProjectWork.ML
{
    /// <summary>
    ///     Класс занимается созданием и обучением модели, предсказанием результатов
    /// </summary>
    public class RegressionModel
    {
        #region Fields

        /// <summary>
        ///     Тип тренера
        /// </summary>
        private readonly TrainerModel _trainer;

        /// <summary>
        ///     Число для разделения данных на тестовые и тренировочные
        /// </summary>
        private readonly double _testFraction;

        /// <summary>
        ///     Кол-во фичей, которые мы не будет учитывать при обучении
        /// </summary>
        private readonly int _countExcludedFeatures;

        /// <summary>
        ///     Список фичей
        /// </summary>
        private readonly List<string> _featureNames;

        /// <summary>
        ///     Список исключенных фичей
        /// </summary>
        private readonly List<string> _dropColumns;

        private readonly MLContext _mlContext;

        /// <summary>
        ///     Данные
        /// </summary>
        private readonly IDataView _dataView;

        /// <summary>
        ///     Разделитель данных в файле
        /// </summary>
        private readonly char _separatorChar;

        /// <summary>
        ///     Модель
        /// </summary>
        private TransformerChain<ITransformer> _model;

        /// <summary>
        ///     Показывать ли данные в консоли с предсказаниями
        /// </summary>
        private readonly bool _isPeek;

        /// <summary>
        ///     Словарь возвращающий тренеров по типу
        /// </summary>
        private static readonly Dictionary<TrainerModel, IEstimator<ITransformer>> _trainers =
            new()
            {
                { TrainerModel.LbfgsPoissonRegression, new MLContext(212103).Regression.Trainers.LbfgsPoissonRegression() },
                { TrainerModel.FastForest, new MLContext(212103).Regression.Trainers.FastForest() },
                { TrainerModel.FastTree, new MLContext(212103).Regression.Trainers.FastTree() },
                { TrainerModel.FastTreeTweedie, new MLContext(212103).Regression.Trainers.FastTreeTweedie() },
                { TrainerModel.Gam, new MLContext(212103).Regression.Trainers.Gam() },
                { TrainerModel.LightGbm, new MLContext(212103).Regression.Trainers.LightGbm() },
                { TrainerModel.Ols, new MLContext(212103).Regression.Trainers.Ols() },
                { TrainerModel.OnlineGradientDescent, new MLContext(212103).Regression.Trainers.OnlineGradientDescent() },
                { TrainerModel.Sdca, new MLContext(212103).Regression.Trainers.Sdca() }
            };

        #endregion

        #region .ctor

        /// <inheritdoc cref="Model"/>
        public RegressionModel(
            string dataPath,
            int countExcludedFeatures,
            TrainerModel trainer,
            char separatorChar = ',',
            double testFraction = 0.2,
            bool isPeek = false)
        {
            _mlContext = new MLContext(212103); // 212103 - seed, чтобы при новом запуске результаты оставались теми же
            _countExcludedFeatures = countExcludedFeatures;
            _trainer = trainer;
            _isPeek = isPeek;

            _testFraction = testFraction;
            _dropColumns = new();
            _separatorChar = separatorChar;
            DataPath = dataPath;

            // Получаю и обрабатываю данные
            var housesData = File.ReadAllLines(dataPath)
                .Skip(1)
                .Select(ProcessingData)
                .ToArray();

            _dataView = _mlContext.Data.LoadFromEnumerable(housesData);

            // Изначально отбираю все фичи
            _featureNames = _dataView.Schema.Select(_ => _.Name).ToList();
        }

        #endregion

        #region Properties

        /// <summary>
        ///     Абсолютная ошибки модели
        /// </summary>
        public double MeanAbsoluteError { get; private set; }

        /// <summary>
        ///     Коэффициент детерминации (показывает насколько близки наши данные к прямой)
        /// </summary>
        public double RSquared { get; private set; }

        /// <summary>
        ///     КОрень из среднеквадратичной ошибки модели
        /// </summary>
        public double RootMeanSquaredError { get; private set; }

        /// <summary>
        ///     Путь к файлу с данными
        /// </summary>
        public string DataPath { get; init; }

        /// <summary>
        ///     Обученная модель
        /// </summary>
        public TransformerChain<ITransformer> Model => _model;

        /// <summary>
        ///     MlContext
        /// </summary>
        public MLContext MlContext => _mlContext;

        /// <summary>
        ///     Тренер
        /// </summary>
        public TrainerModel Trainer => _trainer;

        #endregion

        #region Public methods

        /// <summary>
        ///     Проводит обучение модели
        /// </summary>
        public void Fit()
        {
            DataPreparing();

            var trainedPipeline = CreatePipeline();

            // Разделение данных на тестовые и тренировочные в соответствии с фракцией
            var trainTestData = _mlContext.Data.TrainTestSplit(_dataView, _testFraction);
            var trainData = trainTestData.TrainSet;
            var testsData = trainTestData.TestSet;

            // Обучение модели
            _model = trainedPipeline.Fit(trainData);

            // Тестирование точности на тестовых данных
            var predsDataView = _model.Transform(testsData);
            PeekDataViewInConsole(_isPeek, predsDataView);

            var metrics = _mlContext.Regression.Evaluate(predsDataView);

            MeanAbsoluteError = metrics.MeanAbsoluteError;
            RSquared = metrics.RSquared;
            RootMeanSquaredError = metrics.RootMeanSquaredError;
        }

        /// <summary>
        ///     Сохраняет модель TODO продумать механизм выгрузки
        /// </summary>
        public void SaveModel()
        {
            _mlContext.Model.Save(_model, _dataView.Schema, $"{_trainer}.zip");
        }

        #endregion

        #region Private methods

        /// <summary>
        ///     Показывает какие данные парсит наша прога + приближенные предсказания
        /// </summary>
        private void PeekDataViewInConsole(bool isPeek, IDataView predsDataView)
        {
            if (!isPeek)
            {
                return;
            }

            // показать только 2
            var preViewTransformedData = predsDataView.Preview(2);
            foreach (var row in preViewTransformedData.RowView)
            {
                var columnCollection = row.Values;
                var lineToPrint = "Row--> ";
                foreach (var column in columnCollection)
                {
                    lineToPrint += $" | {column.Key}:{column.Value}";
                }

                Console.WriteLine(lineToPrint + "\n");
            }
        }

        /// <summary>
        ///     Приводит выбивающиеся фичи к верному типу
        /// </summary>
        private HouseModel ProcessingData(string houseInfo)
        {
            var information = houseInfo.Split(_separatorChar);

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

        /// <summary>
        ///     Готовит данные к обучению
        /// </summary>
        private void DataPreparing()
        {
            //  Получение названий фич в порядке уменьшения влияния на Label
            var sortedFeatures = FindBestFeatures();
            if (_countExcludedFeatures >= sortedFeatures.Count)
            {
                return;
            }

            //  Добавление фич для сброса, которые учитываться не будут
            for (var i = 0; i < _countExcludedFeatures; i++)
            {
                _dropColumns.Add(sortedFeatures[sortedFeatures.Count - 1 - i]);
            }

            _featureNames.Clear();
            for (var i = 0; i < sortedFeatures.Count - _countExcludedFeatures; i++)
            {
                _featureNames.Add(sortedFeatures[i]);
            }

            // Колонка Label может быть исключена, но ее исключать нельзя
            if (!_featureNames.Contains("Label"))
            {
                _featureNames.Add("Label");
            }
        }

        /// <summary>
        ///     Тренирует модель, находит индексы самых значимых фичей
        /// </summary>
        private List<string> FindBestFeatures()
        {
            var result = new List<string>();

            // подготовка pipeline
            var pipeline = _mlContext.Transforms.Concatenate("Features", _featureNames.ToArray())
                .Append(_mlContext.Transforms.NormalizeLogMeanVariance("Features"));

            var trainer = SetTrainer();
            var trainedPipeline = pipeline.Append(trainer);

            // Предварительная обработка данных
            var model = trainedPipeline.Fit(_dataView);
            var preprocessedData = model.Transform(_dataView);

            var indexes = GetIndexes(trainer, preprocessedData);

            foreach (var index in indexes)
            {
                // Добавляю имена фич в порядке значимости
                result.Add(_featureNames[index]);
            }

            return result;
        }

        /// <summary>
        ///     Возвращает индексы фич в порядке убывания значимости
        /// </summary>
        private List<int> GetIndexes(IEstimator<ITransformer> trainer, IDataView preprocessedData)
        {
            // Не получилось сделать универсально, так как тренеры вроду бы независимы
            if (_trainer == TrainerModel.FastForest)
            {
                return FeatureHelper.FeaturePermutation(
                    _mlContext,
                    (RegressionPredictionTransformer<FastForestRegressionModelParameters>)trainer.Fit(preprocessedData),
                    preprocessedData,
                    _featureNames.Count);
            }

            if (_trainer == TrainerModel.LbfgsPoissonRegression)
            {
                return FeatureHelper.FeaturePermutation(
                    _mlContext,
                    (RegressionPredictionTransformer<PoissonRegressionModelParameters>)trainer.Fit(preprocessedData),
                    preprocessedData,
                    _featureNames.Count);
            }

            if (_trainer == TrainerModel.Sdca)
            {
                return FeatureHelper.FeaturePermutation(
                    _mlContext,
                    (RegressionPredictionTransformer<LinearRegressionModelParameters>)trainer.Fit(preprocessedData),
                    preprocessedData,
                    _featureNames.Count);
            }

            if (_trainer == TrainerModel.FastTree)
            {
                return FeatureHelper.FeaturePermutation(
                    _mlContext,
                    (RegressionPredictionTransformer<FastTreeRegressionModelParameters>)trainer.Fit(preprocessedData),
                    preprocessedData,
                    _featureNames.Count);
            }

            if (_trainer == TrainerModel.FastTreeTweedie)
            {
                return FeatureHelper.FeaturePermutation(
                    _mlContext,
                    (RegressionPredictionTransformer<FastTreeTweedieModelParameters>)trainer.Fit(preprocessedData),
                    preprocessedData,
                    _featureNames.Count);
            }

            if (_trainer == TrainerModel.Gam)
            {
                return FeatureHelper.FeaturePermutation(
                    _mlContext,
                    (RegressionPredictionTransformer<GamRegressionModelParameters>)trainer.Fit(preprocessedData),
                    preprocessedData,
                    _featureNames.Count);
            }

            if (_trainer == TrainerModel.Ols)
            {
                return FeatureHelper.FeaturePermutation(
                    _mlContext,
                    (RegressionPredictionTransformer<OlsModelParameters>)trainer.Fit(preprocessedData),
                    preprocessedData,
                    _featureNames.Count);
            }

            if (_trainer == TrainerModel.OnlineGradientDescent)
            {
                return FeatureHelper.FeaturePermutation(
                    _mlContext,
                    (RegressionPredictionTransformer<LinearRegressionModelParameters>)trainer.Fit(preprocessedData),
                    preprocessedData,
                    _featureNames.Count);
            }

            if (_trainer == TrainerModel.LightGbm)
            {
                return FeatureHelper.FeaturePermutation(
                    _mlContext,
                    (RegressionPredictionTransformer<LightGbmRegressionModelParameters>)trainer.Fit(preprocessedData),
                    preprocessedData,
                    _featureNames.Count);
            }

            return new List<int>();
        }

        /// <summary>
        ///     Устанавливает тренера в соответствии с выбранным _trainer
        /// </summary>
        private IEstimator<ITransformer> SetTrainer()
        {
            if (!_trainers.TryGetValue(_trainer, out var trainer))
            {
                throw new Exception("Unknown trainer");
            }
            else
            {
                return trainer;
            }
        }

        /// <summary>
        ///     Создает pipeline
        /// </summary>
        private EstimatorChain<ITransformer> CreatePipeline()
        {
            // Подготовка основного pipline
            var pipeline = _mlContext.Transforms.Concatenate("Features", _featureNames.ToArray())
                .Append(_mlContext.Transforms.DropColumns(_dropColumns.ToArray()))
                .Append(_mlContext.Transforms.NormalizeLogMeanVariance("Features"));

            var trainer = SetTrainer();
            return pipeline.Append(trainer);
        }

        #endregion
    }
}
